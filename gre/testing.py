"""Deterministic mock models for offline tests / demos: `mock/good`, `mock/lazy`, `mock/judge`, `mock/hash-embed`.

Enable with `plugins: [gre.testing]` in an experiment YAML.
"""
from __future__ import annotations

import hashlib
import json
import re

from gre.llm.gateway import register_mock

_PATTERNS = [(r"([A-Z][\w ]*?) works for ([A-Z][\w ]*?)[.,]", "worksFor"),
             (r"([A-Z][\w ]*?) is located in ([A-Z][\w ]*?)[.,]", "locatedIn"),
             (r"([A-Z][\w ]*?) was born in ([A-Z][\w ]*?)[.,]", "bornIn")]


def _text_of(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    m = re.search(r'Text:\s*"""(.*?)"""', content, re.S)
    return m.group(1) if m else content


def _extract(text: str) -> list[dict]:
    out = []
    for pat, rel in _PATTERNS:
        for s, o in re.findall(pat, text):
            out.append({"subject": s.strip(), "predicate": rel, "object": o.strip()})
    return out


def _good(messages):
    if "same real-world entity" in messages[-1]["content"]:
        c = messages[-1]["content"]
        a = re.search(r"Record A: (.*)", c).group(1).split(" (type")[0].split(";")[0].lower()
        b = re.search(r"Record B: (.*)", c).group(1).split(" (type")[0].split(";")[0].lower()
        return json.dumps({"match": a.startswith(b) or b.startswith(a)})
    return json.dumps({"triples": _extract(_text_of(messages))})


def _lazy(messages):
    if "same real-world entity" in messages[-1]["content"]:
        return "not sure"   # invalid JSON on purpose -> exercises json_valid_rate
    t = _extract(_text_of(messages))[:1]
    return "Here you go:\n```json\n" + json.dumps({"triples": t}) + "\n```"


def _judge(messages):
    c = messages[-1]["content"]
    ctx = c.split("Fact:")[0].lower()
    fact = re.search(r"Fact: (.*)", c).group(1).lower()
    words = [w for w in re.findall(r"[a-z]+", fact) if len(w) > 3]
    return "1" if words and all(w in ctx.replace(" ", "") or w in ctx for w in words) else "0"


def _hash_embed(texts):
    vecs = []
    for t in texts:
        v = [0.0] * 64
        for tok in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        vecs.append(v)
    return vecs


register_mock("good", _good)
register_mock("lazy", _lazy)
register_mock("judge", _judge)
register_mock("hash-embed", embed=_hash_embed)
