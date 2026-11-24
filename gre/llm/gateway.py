"""Single LLM / embedding gateway.

All model calls go through here, so switching provider is a config string
(`openai/...`, `anthropic/...`, `gemini/...`, `ollama/...`, any LiteLLM model).
Records tokens, cost, latency; caches responses; enforces the budget.

`mock/<name>` models call a Python function registered with `register_mock`, which
makes offline tests and dry runs possible.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from gre.core.cache import Cache, content_hash
from gre.core.contracts import Usage

_MOCKS: dict[str, Callable[[list[dict]], str]] = {}
_MOCK_EMBEDS: dict[str, Callable[[list[str]], list[list[float]]]] = {}


def register_mock(name: str, fn: Callable[[list[dict]], str] | None = None,
                  embed: Callable[[list[str]], list[list[float]]] | None = None):
    """Register `mock/<name>`: fn(messages) -> response text."""
    if fn is not None:
        _MOCKS[name] = fn
    if embed is not None:
        _MOCK_EMBEDS[name] = embed


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, max_cost_usd: float | None):
        self.max = max_cost_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent += usd

    def check(self) -> None:
        if self.max is not None and self.spent >= self.max:
            raise BudgetExceeded(f"budget of ${self.max:.2f} exhausted (spent ${self.spent:.4f})")


_SEMAPHORES: dict[str, threading.Semaphore] = {}
_SEM_LOCK = threading.Lock()


def _semaphore(key: str, n: int) -> threading.Semaphore:
    with _SEM_LOCK:
        if key not in _SEMAPHORES:
            _SEMAPHORES[key] = threading.Semaphore(n)
        return _SEMAPHORES[key]


class LLMGateway:
    """Bound to one provider config; usage is accumulated into `usage`."""

    def __init__(self, provider: dict | None, *, cache: Cache | None = None, usage: Usage | None = None,
                 budget: Budget | None = None, seed: int | None = None, alias: str = ""):
        self.cfg = dict(provider or {})
        self.alias = alias
        self.model: str = self.cfg.get("model", "")
        self.cache = cache
        self.usage = usage if usage is not None else Usage()
        self.budget = budget or Budget(None)
        self.seed = seed
        self._sem = _semaphore(self.model, int(self.cfg.get("max_concurrency", 8)))

    # ---- chat -------------------------------------------------------------------------
    def _params(self, overrides: dict) -> dict:
        skip = {"model", "max_concurrency", "price_in", "price_out", "embedding"}
        p = {k: v for k, v in self.cfg.items() if k not in skip}
        p.setdefault("temperature", 0)
        p.update(overrides)
        if "api_key_env" in p:
            p["api_key"] = os.environ.get(p.pop("api_key_env"), None)
        return p

    def complete(self, prompt: str | list[dict], *, system: str | None = None, **overrides) -> str:
        if not self.model:
            raise RuntimeError("no model configured for this stage (set `provider:` in the stage spec)")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        params = self._params(overrides)
        key = content_hash(self.model, messages, {k: v for k, v in params.items() if k != "api_key"}, self.seed)
        if self.cache is not None:
            hit = self.cache.get("llm", key)
            if hit is not None:
                self.usage.cache_hits += 1
                return hit["text"]
        self.budget.check()
        t0 = time.perf_counter()
        with self._sem:
            text, tin, tout, cost = self._call(messages, params)
        dt = time.perf_counter() - t0
        self.usage.n_calls += 1
        self.usage.tokens_in += tin
        self.usage.tokens_out += tout
        self.usage.cost_usd += cost
        self.usage.latency_s += dt
        self.budget.charge(cost)
        if self.cache is not None:
            self.cache.set("llm", key, {"text": text})
        return text

    def _call(self, messages: list[dict], params: dict) -> tuple[str, int, int, float]:
        if self.model.startswith("mock/"):
            fn = _MOCKS.get(self.model.split("/", 1)[1])
            if fn is None:
                raise KeyError(f"no mock registered for {self.model}")
            text = fn(messages)
            tin = sum(len(str(m["content"]).split()) for m in messages)
            tout = len(text.split())
            return text, tin, tout, (tin * self.cfg.get("price_in", 0) + tout * self.cfg.get("price_out", 0)) / 1e6
        try:
            import litellm
        except ImportError as e:  # pragma: no cover
            raise ImportError("install the LLM extra: pip install 'gre[llm]'") from e
        litellm.drop_params = True
        if self.seed is not None:
            params.setdefault("seed", self.seed)
        retries = int(params.pop("retries", 3))
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = litellm.completion(model=self.model, messages=messages, **params)
                break
            except Exception as e:  # noqa: BLE001
                last = e
                self.usage.errors += 1
                time.sleep(min(2 ** attempt, 20))
        else:
            raise RuntimeError(f"LLM call failed after {retries} attempts: {last}") from last
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        tin = getattr(u, "prompt_tokens", 0) or 0
        tout = getattr(u, "completion_tokens", 0) or 0
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:  # noqa: BLE001  unknown pricing (local models)
            cost = (tin * self.cfg.get("price_in", 0) + tout * self.cfg.get("price_out", 0)) / 1e6
        return text, tin, tout, cost

    # ---- embeddings ---------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float] | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            hit = self.cache.get("emb", content_hash(self.model, t)) if self.cache else None
            if hit is not None:
                out[i] = hit
                self.usage.cache_hits += 1
            else:
                todo.append(i)
        if todo:
            vecs = self._embed([texts[i] for i in todo])
            for i, v in zip(todo, vecs):
                out[i] = v
                if self.cache:
                    self.cache.set("emb", content_hash(self.model, texts[i]), v)
        return out  # type: ignore[return-value]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.model.startswith("mock/"):
            return _MOCK_EMBEDS[self.model.split("/", 1)[1]](texts)
        import litellm
        self.budget.check()
        t0 = time.perf_counter()
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            with self._sem:
                resp = litellm.embedding(model=self.model, input=texts[i:i + 256],
                                         **{k: v for k, v in self._params({}).items() if k != "temperature"})
            vecs.extend(d["embedding"] for d in resp.data)
            try:
                cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
            except Exception:  # noqa: BLE001
                cost = 0.0
            self.usage.cost_usd += cost
            self.budget.charge(cost)
            self.usage.n_calls += 1
        self.usage.latency_s += time.perf_counter() - t0
        return vecs


class NullGateway(LLMGateway):
    def __init__(self):
        super().__init__(None)

    def complete(self, *a, **k) -> str:  # noqa: D401
        raise RuntimeError("this stage has no `provider:` configured, so ctx.llm is unavailable")
