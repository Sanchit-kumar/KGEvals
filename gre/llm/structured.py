"""JSON structured output with repair + retry; tracks JSON validity per model."""
from __future__ import annotations

import json
import re
from typing import Any

from gre.llm.gateway import LLMGateway

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """Parse model output as JSON, tolerating code fences, prose around it, trailing commas."""
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object found")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    frag = t[start:end + 1]
    frag = re.sub(r",\s*([}\]])", r"\1", frag)
    return json.loads(frag)


def complete_json(llm: LLMGateway, prompt: str, *, system: str | None = None, retries: int = 1,
                  json_mode: bool = True, **kw) -> Any | None:
    """Return parsed JSON or None. Counts valid / invalid into llm.usage."""
    extra = dict(kw)
    if json_mode and not llm.model.startswith("mock/"):
        extra.setdefault("response_format", {"type": "json_object"})
    msg: list[dict] = [{"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        text = llm.complete(msg, system=system, **extra)
        try:
            out = parse_json(text)
            llm.usage.json_valid += 1
            return out
        except (ValueError, json.JSONDecodeError) as e:
            llm.usage.json_invalid += 1
            msg = msg + [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That was not valid JSON ({e}). Reply with only the JSON."}]
    return None
