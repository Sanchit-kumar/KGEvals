"""Graph-level metrics on the final KnowledgeGraph: fact retention (MINE), coverage, duplicates, structure."""
from __future__ import annotations

import networkx as nx
from rapidfuzz import fuzz, process

from gre.core.contracts import KnowledgeGraph, Sample
from gre.core.registry import metric
from gre.core.textnorm import norm

JUDGE_PROMPT_V1 = """You are checking whether a knowledge graph contains a fact.

Knowledge-graph context (subject - relation - object lines):
{context}

Fact: {fact}

Does the context contain the information stated in the fact (it may be paraphrased or need combining lines)?
Answer with a single character: 1 if yes, 0 if no."""


def _to_nx(kg: KnowledgeGraph) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in kg.nodes:
        g.add_node(n["id"], **n)
    for e in kg.edges:
        g.add_edge(e["source"], e["target"], predicate=e.get("predicate", ""))
    return g


def retrieve_context(kg: KnowledgeGraph, query: str, top_k: int = 8, hops: int = 2, max_lines: int = 60) -> str:
    """MINE-style retrieval: top-k nodes most similar to the query, expanded `hops` hops."""
    if not kg.nodes:
        return ""
    g = _to_nx(kg)
    names = {n["id"]: " ".join([str(n.get("name", n["id"]))] + list(n.get("aliases", []))) for n in kg.nodes}
    hits = process.extract(norm(query), {k: norm(v) for k, v in names.items()}, scorer=fuzz.token_set_ratio,
                           limit=top_k)
    seeds = [h[2] for h in hits]
    und = g.to_undirected(as_view=True)
    keep: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(und.neighbors(n))
        frontier = nxt - keep
        keep |= nxt
    lines = []
    for u, v, d in g.edges(data=True):
        if u in keep and v in keep:
            lines.append(f"{g.nodes[u].get('name', u)} - {d.get('predicate', '')} - {g.nodes[v].get('name', v)}")
    # rank lines by similarity to the query so the most relevant survive truncation
    lines.sort(key=lambda l: -fuzz.token_set_ratio(norm(query), norm(l)))
    return "\n".join(lines[:max_lines])


@metric("graph", "fact_retention", headline=True, requires=("gold_facts",), needs_llm=True,
        description="MINE: fraction of gold facts a judge LLM finds in the retrieved KG subgraph.")
def fact_retention(kg: KnowledgeGraph, sample: Sample, ctx):
    if ctx.judge is None:
        return None
    hits = 0
    for fact in sample.gold_facts or []:
        context = retrieve_context(kg, fact)
        if not context:
            continue
        ans = ctx.judge.complete(JUDGE_PROMPT_V1.format(context=context, fact=fact))
        hits += ans.strip()[:1] == "1"
    return hits / len(sample.gold_facts) if sample.gold_facts else None


@metric("graph", "fact_coverage_lexical", requires=("gold_facts",),
        description="LLM-free proxy: fraction of facts whose best-matching KG edge has token-set similarity >= 0.7.")
def fact_coverage_lexical(kg: KnowledgeGraph, sample: Sample, ctx):
    edges = [norm(t) for t in kg.edge_texts()]
    if not sample.gold_facts:
        return None
    if not edges:
        return 0.0
    ok = sum(process.extractOne(norm(f), edges, scorer=fuzz.token_set_ratio)[1] >= 70 for f in sample.gold_facts)
    return ok / len(sample.gold_facts)


def _gold_entities(sample: Sample) -> list[set[str]]:
    ents, seen = [], set()
    for t in sample.gold_triples or []:
        for m in (t.subject, t.object):
            key = frozenset(norm(n) for n in m.names())
            if key not in seen:
                seen.add(key)
                ents.append(set(key))
    return ents


@metric("graph", "entity_coverage", requires=("gold_triples",),
        description="Fraction of gold entities present as a node (name or alias).")
def entity_coverage(kg: KnowledgeGraph, sample: Sample, ctx):
    gold = _gold_entities(sample)
    if not gold:
        return None
    node_names = {norm(n) for nd in kg.nodes for n in [nd.get("name", ""), *nd.get("aliases", [])]}
    return sum(bool(g & node_names) for g in gold) / len(gold)


@metric("graph", "duplicate_node_rate", higher_is_better=False, requires=("gold_triples",),
        description="Fraction of matched nodes that duplicate another node of the same gold entity.")
def duplicate_node_rate(kg: KnowledgeGraph, sample: Sample, ctx):
    gold = _gold_entities(sample)
    matched = dup = 0
    for g in gold:
        k = sum(1 for nd in kg.nodes if norm(nd.get("name", "")) in g)
        if k:
            matched += k
            dup += k - 1
    return dup / matched if matched else None


@metric("graph", "structure", description="Nodes, edges, avg degree, orphan rate, giant-component fraction.")
def structure(kg: KnowledgeGraph, sample: Sample, ctx):
    g = _to_nx(kg)
    n = g.number_of_nodes()
    if n == 0:
        return {"nodes": 0.0, "edges": 0.0}
    comps = list(nx.weakly_connected_components(g))
    return {"nodes": float(n), "edges": float(g.number_of_edges()),
            "avg_degree": 2 * g.number_of_edges() / n,
            "orphan_rate": sum(1 for x in g.nodes if g.degree(x) == 0) / n,
            "giant_component": max(len(c) for c in comps) / n,
            "relation_types": float(len({d.get("predicate") for _, _, d in g.edges(data=True)}))}
