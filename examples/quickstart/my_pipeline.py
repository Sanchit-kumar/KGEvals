"""Stand-in for YOUR pipeline. Plain functions, no gre imports needed.

gre calls each function once per sample and passes arguments by NAME:
  text, chunks, schema, mentions, candidate_pairs, triples, clusters, sample, ctx, params
  + any extra key you put in the YAML stage block (e.g. `strip_suffixes: true`).
"""
from __future__ import annotations

import re

_NAME = r"[A-Z]\w*(?: [A-Z]\w*)*"
_PATTERNS = {"worksFor": rf"({_NAME}) works for ({_NAME})",
             "locatedIn": rf"({_NAME}) is located in ({_NAME})",
             "bornIn": rf"({_NAME}) was born in ({_NAME})"}


async def extract(chunks, schema):
    """async is fine. Return any triple shape: tuples, dicts, Triple objects or a networkx graph."""
    allowed = {r.name for r in schema.relation_types} if schema else set(_PATTERNS)
    out = []
    for ch in chunks:
        for rel, pat in _PATTERNS.items():
            if rel in allowed:
                out += [(s, rel, o.rstrip(".")) for s, o in re.findall(pat, ch.text)]
    return out


def resolve(mentions, candidate_pairs=None, strip_suffixes: bool = True):
    """Return {mention_id: cluster_key} (or a list of id-groups, or match pairs)."""
    def key(text: str) -> str:
        t = text.lower().replace("mr. ", "").strip()
        if strip_suffixes:
            t = re.sub(r"\s+(inc|corp|ltd|labs)\.?$", "", t)
        return t.split()[0] if t else t      # first token: "maria lopez" ~ "maria"
    return {m.mention_id: key(m.text) for m in mentions}
