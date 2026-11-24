"""Entity-resolution metrics: pairwise P/R/F1, B³, CEAF-e, MUC, ARI, over/under-merge rates."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import linear_sum_assignment

from gre.core.contracts import EntityCluster, Sample
from gre.core.registry import metric


def _assign(clusters: list[EntityCluster], universe: list[str]) -> dict[str, int]:
    """mention_id -> cluster index restricted to `universe`; unassigned mentions become singletons."""
    out: dict[str, int] = {}
    uni = set(universe)
    for i, c in enumerate(clusters):
        for m in c.mention_ids:
            if m in uni and m not in out:
                out[m] = i
    nxt = len(clusters)
    for m in universe:
        if m not in out:
            out[m] = nxt
            nxt += 1
    return out


def _groups(assign: dict[str, int]) -> list[set[str]]:
    g: dict[int, set[str]] = {}
    for m, c in assign.items():
        g.setdefault(c, set()).add(m)
    return list(g.values())


def _universe(sample: Sample) -> list[str]:
    return [m.mention_id for m in sample.mentions]


# ---------- pairwise (works for labelled-pair benchmarks and full clusterings) -----------------
@metric("resolve", "pairwise", aggregate="micro_prf", headline=True,
        description="Pair decisions (same predicted cluster) vs gold pairs / gold co-clustering.")
def pairwise(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    pa = _assign(pred, uni)
    if sample.gold_pairs:
        tp = fp = fn = 0
        for a, b, label in sample.gold_pairs:
            same = pa.get(a) == pa.get(b)
            tp += same and label == 1
            fp += same and label == 0
            fn += (not same) and label == 1
        return {"tp": tp, "fp": fp, "fn": fn}
    if not sample.gold_clusters:
        return None
    ga = _assign(sample.gold_clusters, uni)
    tp = fp = fn = 0
    for a, b in combinations(uni, 2):
        p, g = pa[a] == pa[b], ga[a] == ga[b]
        tp += p and g
        fp += p and not g
        fn += g and not p
    return {"tp": tp, "fp": fp, "fn": fn}


@metric("resolve", "bcubed", headline=True, requires=("gold_clusters", "mentions"),
        description="B-cubed precision / recall / F1 (mention-level).")
def bcubed(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    pa, ga = _assign(pred, uni), _assign(sample.gold_clusters or [], uni)
    pg, gg = {}, {}
    for m in uni:
        pg.setdefault(pa[m], set()).add(m)
        gg.setdefault(ga[m], set()).add(m)
    P = np.mean([len(pg[pa[m]] & gg[ga[m]]) / len(pg[pa[m]]) for m in uni]) if uni else 0.0
    R = np.mean([len(pg[pa[m]] & gg[ga[m]]) / len(gg[ga[m]]) for m in uni]) if uni else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"p": float(P), "r": float(R), "f1": float(F)}


@metric("resolve", "ceaf_e", requires=("gold_clusters", "mentions"),
        description="Entity-based CEAF (phi4 similarity, optimal alignment).")
def ceaf_e(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    P = _groups(_assign(pred, uni))
    G = _groups(_assign(sample.gold_clusters or [], uni))
    if not P or not G:
        return None
    sim = np.array([[2 * len(g & p) / (len(g) + len(p)) for p in P] for g in G])
    r, c = linear_sum_assignment(-sim)
    total = sim[r, c].sum()
    prec, rec = total / len(P), total / len(G)
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"p": float(prec), "r": float(rec), "f1": float(f)}


def _muc_side(key: list[set[str]], resp_assign: dict[str, int]) -> tuple[int, int]:
    num = den = 0
    for k in key:
        parts = {resp_assign[m] for m in k}
        num += len(k) - len(parts)
        den += len(k) - 1
    return num, den


@metric("resolve", "muc", requires=("gold_clusters", "mentions"), description="MUC link-based P/R/F1.")
def muc(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    pa, ga = _assign(pred, uni), _assign(sample.gold_clusters or [], uni)
    rn, rd = _muc_side(_groups(ga), pa)
    pn, pd = _muc_side(_groups(pa), ga)
    R = rn / rd if rd else 1.0
    P = pn / pd if pd else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"p": P, "r": R, "f1": F}


@metric("resolve", "ari", requires=("gold_clusters", "mentions"), description="Adjusted Rand Index.")
def ari(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    n = len(uni)
    if n < 2:
        return None
    pa, ga = _assign(pred, uni), _assign(sample.gold_clusters or [], uni)
    cont: dict[tuple[int, int], int] = {}
    for m in uni:
        cont[(ga[m], pa[m])] = cont.get((ga[m], pa[m]), 0) + 1
    a: dict[int, int] = {}
    b: dict[int, int] = {}
    for (g, p), v in cont.items():
        a[g] = a.get(g, 0) + v
        b[p] = b.get(p, 0) + v
    sum_ij = sum(comb(v, 2) for v in cont.values())
    sa, sb = sum(comb(v, 2) for v in a.values()), sum(comb(v, 2) for v in b.values())
    exp = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else float((sum_ij - exp) / (mx - exp))


@metric("resolve", "merge_errors", higher_is_better=False, requires=("gold_clusters", "mentions"),
        description="over_merge: predicted clusters mixing >1 gold entity; under_merge: gold entities split.")
def merge_errors(pred: list[EntityCluster], sample: Sample, ctx):
    uni = _universe(sample)
    pa, ga = _assign(pred, uni), _assign(sample.gold_clusters or [], uni)
    P, G = _groups(pa), _groups(ga)
    over = sum(len({ga[m] for m in p}) > 1 for p in P) / len(P) if P else 0.0
    under = sum(len({pa[m] for m in g}) > 1 for g in G) / len(G) if G else 0.0
    return {"over_merge": over, "under_merge": under,
            "dedup_ratio": len(P) / len(uni) if uni else 0.0}
