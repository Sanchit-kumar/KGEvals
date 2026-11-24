"""Reference resolvers: exact, fuzzy, embedding-threshold, LLM pairwise (with fuzzy blocking).

Every resolver gets `mentions` (+ optional `candidate_pairs` for labelled-pair benchmarks)
and returns clusters. If candidate_pairs is given only those pairs are compared.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from rapidfuzz import fuzz

from gre.core.coerce import _union_find, clusters_from_groups
from gre.core.contracts import EntityCluster, Mention
from gre.core.registry import stage
from gre.core.textnorm import norm
from gre.llm.structured import parse_json


def _key(m: Mention, field: str | None) -> str:
    if field and field in m.attributes:
        return norm(m.attributes[field])
    return norm(m.text)


def _pairs(mentions: list[Mention], candidate_pairs, same_type: bool):
    if candidate_pairs is not None:
        return list(candidate_pairs)
    out = []
    for a, b in combinations(mentions, 2):
        if same_type and a.type and b.type and a.type != b.type:
            continue
        out.append((a.mention_id, b.mention_id))
    return out


def _cluster(matches, mentions) -> list[EntityCluster]:
    return clusters_from_groups(_union_find(matches, [m.mention_id for m in mentions]), mentions)


@stage("resolve", "exact", version="1", description="Merge mentions with identical normalised names.")
def exact(mentions: list[Mention], candidate_pairs=None, field: str | None = None,
          same_type: bool = True) -> list[EntityCluster]:
    by = {m.mention_id: m for m in mentions}
    matches = [(a, b) for a, b in _pairs(mentions, candidate_pairs, same_type)
               if _key(by[a], field) == _key(by[b], field)]
    return _cluster(matches, mentions)


@stage("resolve", "fuzzy", version="1", description="RapidFuzz token_set_ratio >= threshold (0-1).")
def fuzzy(mentions: list[Mention], candidate_pairs=None, threshold: float = 0.85, field: str | None = None,
          same_type: bool = True, scorer: str = "token_set_ratio") -> list[EntityCluster]:
    by = {m.mention_id: m for m in mentions}
    fn = getattr(fuzz, scorer)
    matches = [(a, b) for a, b in _pairs(mentions, candidate_pairs, same_type)
               if fn(_key(by[a], field), _key(by[b], field)) / 100 >= threshold]
    return _cluster(matches, mentions)


@stage("resolve", "embedding", version="1", description="Cosine similarity of embeddings >= threshold.")
def embedding(mentions: list[Mention], ctx, candidate_pairs=None, threshold: float = 0.85,
              field: str | None = None, same_type: bool = True) -> list[EntityCluster]:
    texts = [m.attributes.get(field, m.text) if field else m.text for m in mentions]
    vecs = np.asarray(ctx.embed([str(t) for t in texts]), dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    idx = {m.mention_id: i for i, m in enumerate(mentions)}
    matches = [(a, b) for a, b in _pairs(mentions, candidate_pairs, same_type)
               if float(vecs[idx[a]] @ vecs[idx[b]]) >= threshold]
    return _cluster(matches, mentions)


PAIR_PROMPT_V1 = """Do these two records refer to the same real-world entity?

Record A: {a}
Record B: {b}

Answer with JSON: {{"match": true}} or {{"match": false}}"""


@stage("resolve", "llm_pairwise", version="1",
       description="LLM yes/no per pair; candidate pairs or fuzzy blocking (block_threshold) to limit calls.")
def llm_pairwise(mentions: list[Mention], ctx, candidate_pairs=None, block_threshold: float = 0.5,
                 same_type: bool = True, auto_match_exact: bool = True, prompt: str | None = None) -> list[EntityCluster]:
    by = {m.mention_id: m for m in mentions}

    def show(m: Mention) -> str:
        return m.text + (f" (type: {m.type})" if m.type and m.type != "product" else "") + \
            (f"\n  context: {m.context[:300]}" if m.context else "")

    pairs = _pairs(mentions, candidate_pairs, same_type)
    if candidate_pairs is None:  # blocking
        pairs = [(a, b) for a, b in pairs
                 if fuzz.token_set_ratio(norm(by[a].text), norm(by[b].text)) / 100 >= block_threshold]
    matches = []
    for a, b in pairs:
        if auto_match_exact and candidate_pairs is None and norm(by[a].text) == norm(by[b].text):
            matches.append((a, b))
            continue
        txt = ctx.llm.complete((prompt or PAIR_PROMPT_V1).format(a=show(by[a]), b=show(by[b])))
        try:
            ok = bool(parse_json(txt).get("match"))
            ctx.llm.usage.json_valid += 1
        except Exception:  # noqa: BLE001
            ctx.llm.usage.json_invalid += 1
            ok = "true" in txt.lower() or txt.strip().lower().startswith("yes")
        if ok:
            matches.append((a, b))
    return _cluster(matches, mentions)
