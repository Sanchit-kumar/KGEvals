"""Extraction metrics: strict / soft triple P/R/F1, entity & relation F1, ontology conformance,
hallucination (Text2KGBench definitions), domain/range violations."""
from __future__ import annotations

from typing import Callable

from rapidfuzz import fuzz

from gre.core.contracts import Mention, Sample, Triple
from gre.core.registry import metric
from gre.core.textnorm import norm, norm_rel


def _names(m: Mention) -> list[str]:
    return [norm(n) for n in m.names() if n]


def _strict_score(p: Triple, g: Triple) -> float:
    ok = (norm(p.subject.text) in _names(g.subject) and norm_rel(p.predicate) == norm_rel(g.predicate)
          and norm(p.object.text) in _names(g.object))
    return 1.0 if ok else 0.0


def _fuzzy_sim(a: str, bs: list[str]) -> float:
    return max((fuzz.token_sort_ratio(a, b) / 100.0 for b in bs), default=0.0)


def _soft_scorer(ctx) -> Callable[[Triple, Triple], float]:
    method = getattr(ctx.scoring, "soft_method", "fuzzy") if ctx and ctx.scoring else "fuzzy"
    if method == "embedding" and ctx.embedder is not None:
        import numpy as np
        cache: dict[str, np.ndarray] = {}

        def vec(t: str):
            if t not in cache:
                v = np.asarray(ctx.embed([t])[0], dtype=float)
                cache[t] = v / (np.linalg.norm(v) or 1.0)
            return cache[t]

        def sim(a: str, bs: list[str]) -> float:
            return max((float(vec(a) @ vec(b)) for b in bs), default=0.0)
    else:
        sim = _fuzzy_sim

    def score(p: Triple, g: Triple) -> float:
        return min(sim(norm(p.subject.text), _names(g.subject)),
                   sim(norm_rel(p.predicate), [norm_rel(g.predicate)]),
                   sim(norm(p.object.text), _names(g.object)))
    return score


def match_triples(pred: list[Triple], gold: list[Triple], score: Callable[[Triple, Triple], float],
                  threshold: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending similarity (each gold matched at most once)."""
    cands = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s = score(p, g)
            if s >= threshold:
                cands.append((s, i, j))
    cands.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j, s))
    return out


def _dedup(triples: list[Triple]) -> list[Triple]:
    seen, out = set(), []
    for t in triples:
        k = (norm(t.subject.text), norm_rel(t.predicate), norm(t.object.text))
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _prf_counts(n_match: int, n_pred: int, n_gold: int) -> dict:
    return {"tp": n_match, "fp": n_pred - n_match, "fn": n_gold - n_match}


@metric("extract", "triple_strict", aggregate="micro_prf", headline=True, requires=("gold_triples",),
        description="Exact (normalised) triple match; subject/object may match any gold alias.")
def triple_strict(pred: list[Triple], sample: Sample, ctx):
    pred, gold = _dedup(pred), sample.gold_triples or []
    m = match_triples(pred, gold, _strict_score, 1.0)
    return _prf_counts(len(m), len(pred), len(gold))


@metric("extract", "triple_soft", aggregate="micro_prf", headline=True, requires=("gold_triples",),
        description="Similarity-tolerant triple match (min component similarity >= soft_match_threshold).")
def triple_soft(pred: list[Triple], sample: Sample, ctx):
    pred, gold = _dedup(pred), sample.gold_triples or []
    thr = ctx.scoring.soft_match_threshold if ctx and ctx.scoring else 0.8
    m = match_triples(pred, gold, _soft_scorer(ctx), thr)
    return _prf_counts(len(m), len(pred), len(gold))


@metric("extract", "entity", aggregate="micro_prf", requires=("gold_triples",),
        description="Set of entity names appearing in triples (alias-aware).")
def entity_prf(pred: list[Triple], sample: Sample, ctx):
    gold_ents: list[set[str]] = []
    seen: set[frozenset] = set()
    for t in sample.gold_triples or []:
        for m in (t.subject, t.object):
            key = frozenset(_names(m))
            if key not in seen:
                seen.add(key)
                gold_ents.append(set(key))
    pred_ents = {norm(m.text) for t in pred for m in (t.subject, t.object)}
    matched_gold = sum(1 for g in gold_ents if g & pred_ents)
    matched_pred = sum(1 for p in pred_ents if any(p in g for g in gold_ents))
    return {"tp": matched_pred, "fp": len(pred_ents) - matched_pred, "fn": len(gold_ents) - matched_gold}


@metric("extract", "relation", aggregate="micro_prf", requires=("gold_triples",),
        description="Set of relation types used per sample.")
def relation_prf(pred: list[Triple], sample: Sample, ctx):
    g = {norm_rel(t.predicate) for t in sample.gold_triples or []}
    p = {norm_rel(t.predicate) for t in pred}
    return _prf_counts(len(g & p), len(p), len(g))


@metric("extract", "ontology_conformance", requires=("graph_schema",),
        description="Fraction of predicted triples whose relation exists in the schema.")
def ontology_conformance(pred: list[Triple], sample: Sample, ctx):
    if not pred or not sample.graph_schema or not sample.graph_schema.relation_types:
        return None
    rels = {norm_rel(r.name) for r in sample.graph_schema.relation_types}
    return sum(norm_rel(t.predicate) in rels for t in pred) / len(pred)


def _in_text(s: str, text: str) -> bool:
    s, text = norm(s), norm(text)
    return bool(s) and (s in text or fuzz.partial_ratio(s, text) >= 90)


@metric("extract", "hallucination", higher_is_better=False, requires=("text",),
        description="Text2KGBench: subject/object not in text (nor ontology concepts); relation not in ontology.")
def hallucination(pred: list[Triple], sample: Sample, ctx):
    if not pred:
        return None
    text = sample.text or ""
    concepts = {norm(e.name) for e in sample.graph_schema.entity_types} if sample.graph_schema else set()
    rels = {norm_rel(r.name) for r in sample.graph_schema.relation_types} if sample.graph_schema else set()

    def ent_h(m: Mention) -> bool:
        return not _in_text(m.text, text) and norm(m.text) not in concepts
    out = {"subject": sum(ent_h(t.subject) for t in pred) / len(pred),
           "object": sum(ent_h(t.object) for t in pred) / len(pred)}
    if rels:
        out["relation"] = sum(norm_rel(t.predicate) not in rels for t in pred) / len(pred)
    return out


@metric("extract", "domain_range_violation", higher_is_better=False, requires=("graph_schema",),
        description="Among typed predictions with a schema relation: subject/object type != relation domain/range.")
def domain_range_violation(pred: list[Triple], sample: Sample, ctx):
    schema = sample.graph_schema
    if not schema:
        return None
    checked = bad = 0
    for t in pred:
        r = schema.relation(t.predicate)
        if r is None:
            continue
        for typ, want in ((t.subject.type, r.domain), (t.object.type, r.range)):
            if typ and want and want != "Any":
                checked += 1
                bad += norm(typ) != norm(want)
    return bad / checked if checked else None


@metric("extract", "n_triples", description="Average predicted triples per sample (after dedup).")
def n_triples(pred: list[Triple], sample: Sample, ctx):
    return float(len(_dedup(pred)))
