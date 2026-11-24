"""Experiment runner.

Expands datasets x pipeline variants x modes x seeds, executes each stage per sample
(threaded, cached, budget-guarded), scores every applicable metric and persists results.

Evaluation modes
  isolated : each stage gets *gold* upstream input where the dataset has it
             (extract <- gold chunks, resolve <- gold mentions)  => measures the stage alone
  cascaded : each stage gets the pipeline's *own* upstream output
             (resolve <- mentions from predicted triples)       => measures error propagation
The graph stage is always end-to-end (pipeline's own triples + clusters).
"""
from __future__ import annotations

import inspect
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from gre.core import coerce
from gre.core.cache import Cache, content_hash, source_hash
from gre.core.config import ExperimentConfig, PipelineVariant, expand_pipelines, split_stage_spec
from gre.core.context import Context
from gre.core.contracts import Chunk, EntityCluster, KnowledgeGraph, Mention, Sample, Triple, Usage
from gre.core.registry import REGISTRY, MetricSpec, get_stage, import_object, load_builtins_and_plugins
from gre.llm.gateway import Budget, BudgetExceeded, LLMGateway, NullGateway

log = logging.getLogger("gre")

PIPELINE_ORDER = ("chunk", "extract", "resolve", "graph")
OUTPUT_COERCERS: dict[str, Callable] = {}


@dataclass
class RunSpec:
    exp: str
    variant: PipelineVariant
    dataset: str
    mode: str
    seed: int
    stages: list[str]                      # active stages, in order
    auto_graph: bool = False

    @property
    def label(self) -> str:
        return f"{self.variant.label} · {self.mode}"

    @property
    def run_id(self) -> str:
        raw = f"{self.variant.label}__{self.dataset}__{self.mode}__s{self.seed}"
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:150] + "_" + content_hash(raw, n=6)


@dataclass
class SampleResult:
    sample_id: str
    preds: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, dict] = field(default_factory=dict)
    error: str | None = None


# ----------------------------------------------------------------------------------------------
# stage resolution & argument binding
# ----------------------------------------------------------------------------------------------
@dataclass
class BoundStage:
    stage: str
    name: str
    fn: Callable
    version: str
    reserved: dict
    params: dict


_INSTANCES: dict[str, Any] = {}


def resolve_stage(stage: str, spec: dict) -> BoundStage:
    reserved, params = split_stage_spec(spec)
    if "callable" in reserved:
        obj = import_object(reserved["callable"])
        name = reserved["callable"]
        version = str(reserved.get("version", source_hash(obj)))
    else:
        s = get_stage(stage, reserved["adapter"])
        obj, name = s.fn, s.name
        version = f"{s.version}:{source_hash(obj)}"
    if inspect.isclass(obj):  # class adapters: instantiated once with matching params, then .run/__call__
        key = content_hash(name, params)
        if key not in _INSTANCES:
            init_params = {k: v for k, v in params.items() if k in inspect.signature(obj).parameters}
            _INSTANCES[key] = obj(**init_params)
        inst = _INSTANCES[key]
        fn = getattr(inst, "run", None) or getattr(inst, stage, None) or inst
    else:
        fn = obj
    return BoundStage(stage, name, fn, version, reserved, params)


def call_bound(fn: Callable, available: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return _maybe_await(fn(**available))
    kwargs, missing = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
            continue
        if name in available:
            kwargs[name] = available[name]
        elif p.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(f"{getattr(fn, '__qualname__', fn)} needs {missing}; available inputs: "
                        f"{sorted(k for k in available if not k.startswith('_'))}")
    return _maybe_await(fn(**kwargs))


def _maybe_await(result: Any) -> Any:
    """Support `async def` pipeline functions (each worker thread runs its own event loop)."""
    if inspect.isawaitable(result):
        import asyncio
        return asyncio.run(_await(result))
    return result


async def _await(aw):
    return await aw


def _coerce_output(stage: str, out: Any, field_map: dict | None, mentions: list[Mention]) -> Any:
    if stage in OUTPUT_COERCERS:
        return OUTPUT_COERCERS[stage](out)
    if stage == "extract":
        return coerce.to_triples(out, field_map)
    if stage == "resolve":
        return coerce.to_clusters(out, mentions, field_map)
    if stage == "graph":
        return coerce.to_kg(out, field_map)
    if stage == "chunk":
        return [c if isinstance(c, Chunk) else Chunk(**c) if isinstance(c, dict) else
                Chunk(chunk_id=f"c{i}", doc_id="", text=str(c)) for i, c in enumerate(out)]
    return out


def _dump(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    return obj


def _load(stage: str, data: Any) -> Any:
    if stage == "extract":
        return [Triple(**d) for d in data]
    if stage == "resolve":
        return [EntityCluster(**d) for d in data]
    if stage == "graph":
        return KnowledgeGraph(**data)
    if stage == "chunk":
        return [Chunk(**d) for d in data]
    return data


# ----------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: ExperimentConfig, *, out_dir: str | Path | None = None, quick: bool = False,
                 limit: int | None = None, stages: list[str] | None = None, force: bool = False,
                 progress: Callable[[str], None] | None = None):
        load_builtins_and_plugins(cfg.plugins)
        if cfg.local_datasets:
            from gre.datasets.local import register_local_datasets
            for opt in cfg.local_datasets.values():   # absolute paths so `gre report/score` work from anywhere
                opt["path"] = str(Path(opt["path"]).resolve())
                if opt.get("schema"):
                    opt["schema"] = str(Path(opt["schema"]).resolve())
            register_local_datasets(cfg.local_datasets)
        self.cfg = cfg
        self.quick = quick
        self.limit = limit
        self.stage_filter = stages or cfg.stages
        self.force = force
        self.out = Path(out_dir or cfg.output_dir) / cfg.name
        self.cache = Cache(cfg.runtime.cache_path, enabled=cfg.runtime.cache)
        self.budget = Budget(cfg.budget.get("max_cost_usd"))
        self.say = progress or (lambda s: log.info(s))
        self._datasets: dict[str, tuple[Any, list[Sample]]] = {}

    # ---------------- planning ---------------------------------------------------------------
    def dataset_specs(self) -> list[str]:
        from gre.datasets import SUITES
        specs = list(self.cfg.datasets)
        if self.cfg.suite:
            if self.cfg.suite not in SUITES:
                raise KeyError(f"unknown suite {self.cfg.suite!r}; available {sorted(SUITES)}")
            specs += [s for s in SUITES[self.cfg.suite] if s not in specs]
        return specs

    def load_dataset(self, spec: str):
        if spec not in self._datasets:
            from gre.datasets.base import load_samples
            ds, samples = load_samples(spec, quick_limit=self.cfg.runtime.quick_limit if self.quick else None)
            if self.limit:
                samples = samples[: self.limit]
            self._datasets[spec] = (ds, samples)
        return self._datasets[spec]

    def plan(self) -> list[RunSpec]:
        from gre.core.registry import get_dataset
        from gre.datasets.base import parse_dataset_spec
        runs = []
        self.skipped: list[str] = []
        seen_effective: dict[str, str] = {}
        for spec in self.dataset_specs():
            ds_cls = get_dataset(parse_dataset_spec(spec)[0])
            supported = set(ds_cls.supported_stages)
            for v in expand_pipelines(self.cfg):
                active = [s for s in PIPELINE_ORDER if s in v.stages and (s in supported or s == "chunk")]
                if self.stage_filter:
                    active = [s for s in active if s in self.stage_filter or s == "chunk"]
                # resolve without gold mentions needs extract upstream
                if "resolve" in active and "extract" not in active and not _has_gold_mentions(ds_cls):
                    active.remove("resolve")
                auto_graph = ("graph" in supported and "extract" in active and "graph" not in active
                              and (not self.stage_filter or "graph" in self.stage_filter))
                if auto_graph:
                    active.append("graph")
                if not [s for s in active if s != "chunk"]:
                    continue
                cascade_differs = "chunk" in active or ("extract" in active and "resolve" in active
                                                        and _has_gold_mentions(ds_cls))
                # two variants that are identical on this dataset's stages => run once
                eff = content_hash(spec, {s: v.stages.get(s) for s in active})
                if eff in seen_effective:
                    self.skipped.append(f"{spec}: {v.label} == {seen_effective[eff]} (identical on stages {active})")
                    continue
                seen_effective[eff] = v.label
                for mode in self.cfg.mode:
                    if mode == "cascaded" and not cascade_differs and "isolated" in self.cfg.mode:
                        continue
                    for seed in self.cfg.seeds:
                        runs.append(RunSpec(self.cfg.name, v, spec, mode, seed, active, auto_graph))
        return runs

    # ---------------- gateways ---------------------------------------------------------------
    def _gateway(self, alias: str | None, usage: Usage, seed: int, kind: str = "providers") -> LLMGateway | None:
        if not alias:
            return None
        pool = self.cfg.providers if kind == "providers" else self.cfg.embeddings
        if alias in pool:
            pcfg = pool[alias]
        elif "/" in str(alias):          # raw LiteLLM model string
            pcfg = {"model": alias}
        else:
            raise KeyError(f"unknown {kind[:-1]} alias {alias!r}; define it under `{kind}:`")
        return LLMGateway(pcfg, cache=self.cache, usage=usage, budget=self.budget, seed=seed, alias=alias)

    # ---------------- execution --------------------------------------------------------------
    def run_all(self, runs: list[RunSpec] | None = None) -> list[dict]:
        runs = runs if runs is not None else self.plan()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(self.cfg.model_dump_json(indent=2))
        summaries = []
        for i, r in enumerate(runs, 1):
            self.say(f"[{i}/{len(runs)}] {r.dataset} :: {r.label} (seed {r.seed}) stages={r.stages}")
            try:
                summaries.append(self.run_one(r))
            except BudgetExceeded as e:
                self.say(f"[budget] {e} — stopping; partial results kept")
                break
        from gre.results import write_results_db
        write_results_db(self.out)
        return summaries

    def run_one(self, r: RunSpec) -> dict:
        rdir = self.out / r.run_id
        spec_hash = content_hash(r.variant.model_dump(), r.dataset, r.mode, r.seed, r.stages,
                                 self.cfg.scoring.model_dump(), self.quick, self.limit)
        done = rdir / "scores.json"
        if done.exists() and not self.force:
            prev = json.loads(done.read_text())
            if prev.get("spec_hash") == spec_hash:
                self.say("  ↳ up to date (use --force to recompute)")
                return prev
        ds, samples = self.load_dataset(r.dataset)
        bound = {s: resolve_stage(s, r.variant.stages[s]) for s in r.stages if s in r.variant.stages}
        t0 = time.perf_counter()
        results: list[SampleResult] = []
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.runtime.workers)) as ex:
            futs = {ex.submit(self._run_sample, r, bound, s): s for s in samples}
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    results.append(f.result())
                except BudgetExceeded:
                    for other in futs:
                        other.cancel()
                    self._persist(r, rdir, ds, samples, results, spec_hash, time.perf_counter() - t0, partial=True)
                    raise
                if k % max(1, len(samples) // 10) == 0 or k == len(samples):
                    self.say(f"  {k}/{len(samples)} samples")
        order = {s.id: i for i, s in enumerate(samples)}
        results.sort(key=lambda x: order.get(x.sample_id, 0))
        return self._persist(r, rdir, ds, samples, results, spec_hash, time.perf_counter() - t0)

    def _run_sample(self, r: RunSpec, bound: dict[str, BoundStage], sample: Sample) -> SampleResult:
        res = SampleResult(sample.id)
        chunks = sample.chunks or ([Chunk(chunk_id=f"{sample.id}_c0", doc_id=sample.id, text=sample.text,
                                          char_end=len(sample.text))] if sample.text else [])
        state: dict[str, Any] = {"text": sample.text, "chunks": chunks, "schema": sample.graph_schema,
                                 "mentions": sample.mentions, "candidate_pairs": sample.candidate_pairs,
                                 "triples": [], "clusters": None, "sample": sample}
        outputs: dict[str, Any] = {}
        try:
            for st in r.stages:
                if st == "graph" and st not in bound:   # auto graph builder
                    outputs["graph"] = coerce.build_kg(state["triples"], state["clusters"])
                    continue
                b = bound[st]
                inputs = dict(state)
                if st == "extract" and r.mode == "isolated":
                    inputs["chunks"] = chunks if sample.chunks else state["chunks"]
                if st == "resolve":
                    use_gold = sample.mentions and (r.mode == "isolated" or "extract" not in r.stages)
                    if not use_gold:
                        inputs["mentions"] = _mentions_from_triples(state["triples"], sample.id)
                        inputs["candidate_pairs"] = None
                    state["resolve_on_gold"] = bool(use_gold)
                out, usage = self._exec_stage(r, b, inputs, sample)
                res.usage[st] = usage.model_dump()
                outputs[st] = out
                if st == "chunk":
                    state["chunks"] = out
                elif st == "extract":
                    _assign_mention_ids(out, sample.id)
                    state["triples"] = out
                elif st == "resolve":
                    # clusters over gold mentions don't align with predicted triples -> graph uses names
                    state["clusters"] = None if state["resolve_on_gold"] else out
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
            log.debug(traceback.format_exc())
            if self.cfg.runtime.fail_fast:
                raise
        # missing outputs score as empty predictions (failures must hurt)
        for st in r.stages:
            if st not in outputs:
                outputs[st] = {"extract": [], "resolve": coerce.to_clusters(None, sample.mentions),
                               "graph": KnowledgeGraph(), "chunk": []}[st]
        res.preds = {k: _dump(v) for k, v in outputs.items()}
        res.metrics = self.score_sample(r, sample, outputs, resolve_on_gold=bool(state.get("resolve_on_gold")))
        return res

    def _exec_stage(self, r: RunSpec, b: BoundStage, inputs: dict, sample: Sample) -> tuple[Any, Usage]:
        usage = Usage()
        llm = self._gateway(b.reserved.get("provider"), usage, r.seed)
        emb = self._gateway(b.reserved.get("embedding"), usage, r.seed, kind="embeddings")
        ctx = Context(stage=b.stage, params=b.params, llm=llm or NullGateway(), embedder=emb, usage=usage,
                      seed=r.seed, run_id=r.run_id, sample_id=sample.id)
        avail = {**inputs, **b.params, "ctx": ctx, "params": b.params}
        use_cache = self.cfg.runtime.cache and b.reserved.get("cache", True)
        key = None
        if use_cache:
            prov = self.cfg.providers.get(b.reserved.get("provider"), b.reserved.get("provider"))
            embc = self.cfg.embeddings.get(b.reserved.get("embedding"), b.reserved.get("embedding"))
            in_keys = {"extract": ("chunks", "schema"), "resolve": ("mentions", "candidate_pairs"),
                       "graph": ("triples", "clusters"), "chunk": ("text",)}[b.stage]
            key = content_hash(b.stage, b.name, b.version, b.params, prov, embc, r.seed,
                               {k: inputs.get(k) for k in in_keys})
            hit = self.cache.get("stage", key)
            if hit is not None:
                cached_usage = Usage(**hit["usage"])
                cached_usage.cache_hits += 1
                return _load(b.stage, hit["out"]), cached_usage
        raw = call_bound(b.fn, avail)
        out = _coerce_output(b.stage, raw, b.reserved.get("field_map"), inputs.get("mentions") or [])
        if key:
            self.cache.set("stage", key, {"out": _dump(out), "usage": usage.model_dump()})
        return out, usage

    # ---------------- scoring ----------------------------------------------------------------
    def metric_selected(self, full: str) -> bool:
        inc, exc = self.cfg.scoring.include, self.cfg.scoring.exclude
        if inc and not any(fnmatch(full, p) for p in inc):
            return False
        return not any(fnmatch(full, p) for p in exc)

    def score_sample(self, r: RunSpec, sample: Sample, outputs: dict, resolve_on_gold: bool = True) -> dict:
        out: dict[str, Any] = {}
        judge_usage = Usage()
        judge = self._gateway(self.cfg.judge.get("provider"), judge_usage, 0) if self.cfg.judge else None
        emb = None
        if self.cfg.scoring.soft_method == "embedding":
            emb = self._gateway(self.cfg.scoring.embedding, judge_usage, 0, kind="embeddings")
        for full, m in REGISTRY.metrics.items():
            if m.stage not in outputs or not self.metric_selected(full):
                continue
            if m.stage == "resolve" and not resolve_on_gold:
                continue  # predicted mentions have no gold alignment; graph metrics cover cascaded ER
            if not _requires_ok(m, sample):
                continue
            ctx = Context(stage=m.stage, scoring=self.cfg.scoring, judge=judge, embedder=emb, usage=judge_usage,
                          sample_id=sample.id, run_id=r.run_id)
            try:
                out[full] = m.fn(outputs[m.stage], sample, ctx)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("metric %s failed on %s: %s", full, sample.id, e)
                out[full] = None
        if judge_usage.n_calls or judge_usage.cache_hits:
            out["_judge_usage"] = judge_usage.model_dump()
        return out

    # ---------------- persistence ------------------------------------------------------------
    def _persist(self, r: RunSpec, rdir: Path, ds, samples: list[Sample], results: list[SampleResult],
                 spec_hash: str, wall: float, partial: bool = False) -> dict:
        from gre.results import aggregate_run
        rdir.mkdir(parents=True, exist_ok=True)
        with open(rdir / "samples.jsonl", "w") as f:
            for x in results:
                f.write(json.dumps(x.__dict__, default=str) + "\n")
        summary = aggregate_run(results, REGISTRY.metrics, self.cfg.scoring.n_bootstrap)
        summary.update({
            "run_id": r.run_id, "exp": r.exp, "label": r.label, "pipeline": r.variant.pipeline,
            "matrix": r.variant.matrix, "dataset": r.dataset, "dataset_name": ds.name, "mode": r.mode,
            "seed": r.seed, "stages": r.stages, "auto_graph": r.auto_graph,
            "stage_specs": r.variant.stages, "n_samples": len(samples), "n_scored": len(results),
            "wall_time_s": wall, "partial": partial, "spec_hash": spec_hash, "quick": self.quick,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        (rdir / "scores.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    def close(self) -> None:
        self.cache.close()


# ----------------------------------------------------------------------------------------------
def _has_gold_mentions(ds_cls) -> bool:
    g = getattr(ds_cls, "gold_mentions", None)
    return ("resolve" in ds_cls.supported_stages) if g is None else bool(g)


def _requires_ok(m: MetricSpec, sample: Sample) -> bool:
    for f in m.requires:
        v = getattr(sample, f, None)
        if v is None or (hasattr(v, "__len__") and len(v) == 0):
            return False
    return True


def _assign_mention_ids(triples: list[Triple], sid: str) -> None:
    for i, t in enumerate(triples):
        if not t.subject.mention_id:
            t.subject.mention_id = f"{sid}_t{i}_s"
        if not t.object.mention_id:
            t.object.mention_id = f"{sid}_t{i}_o"


def _mentions_from_triples(triples: list[Triple], sid: str) -> list[Mention]:
    _assign_mention_ids(triples, sid)
    out, seen = [], set()
    for t in triples:
        for m in (t.subject, t.object):
            if m.mention_id not in seen:
                seen.add(m.mention_id)
                out.append(m)
    return out
