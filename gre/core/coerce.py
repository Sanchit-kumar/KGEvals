"""Coerce loosely-typed pipeline outputs into contracts so user code needs no rewrite.

Supported shapes
  triples : list[Triple] | list[dict] | list[tuple(s,p,o[,conf])] | {"triples": [...]} |
            networkx graph (edge attr predicate/relation/label) | KnowledgeGraph
  clusters: list[EntityCluster] | {mention_id: cluster_id} | list[list[mention_id]] |
            list[(a, b)] match pairs | list[dict]
  kg      : KnowledgeGraph | dict | networkx graph | list of triples
A YAML ``field_map`` renames keys first, e.g. {subject: head, predicate: rel, object: tail}.
"""
from __future__ import annotations

from typing import Any, Iterable

from gre.core.contracts import EntityCluster, KnowledgeGraph, Mention, Triple

_S_KEYS = ("subject", "sub", "s", "head", "h", "source", "subj")
_P_KEYS = ("predicate", "rel", "relation", "p", "r", "label", "type_", "property")
_O_KEYS = ("object", "obj", "o", "tail", "t", "target")


def _remap(d: dict, field_map: dict | None) -> dict:
    if not field_map:
        return d
    out = dict(d)
    for ours, theirs in field_map.items():
        if theirs in d:
            out[ours] = d[theirs]
    return out


def _first(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _mention(v: Any, typ: str | None = None) -> Mention:
    if isinstance(v, Mention):
        return v
    if isinstance(v, dict):
        d = dict(v)
        d.setdefault("text", d.get("name") or d.get("id") or "")
        if typ and not d.get("type"):
            d["type"] = typ
        return Mention(**{k: d[k] for k in d if k != "name"})
    return Mention(text=str(v), type=typ or "")


def _is_nx(obj: Any) -> bool:
    return hasattr(obj, "edges") and hasattr(obj, "nodes") and hasattr(obj, "is_directed")


def to_triples(obj: Any, field_map: dict | None = None) -> list[Triple]:
    if obj is None:
        return []
    if isinstance(obj, KnowledgeGraph):
        return kg_to_triples(obj)
    if _is_nx(obj):
        return to_triples(nx_to_kg(obj))
    if isinstance(obj, dict):
        for k in ("triples", "relations", "edges", "facts"):
            if k in obj:
                return to_triples(obj[k], field_map)
        raise TypeError(f"cannot find triples in dict with keys {list(obj)[:10]}")
    out: list[Triple] = []
    for item in obj:
        if isinstance(item, Triple):
            out.append(item)
        elif isinstance(item, dict):
            d = _remap(item, field_map)
            s, p, o = _first(d, _S_KEYS), _first(d, _P_KEYS), _first(d, _O_KEYS)
            if s is None or p is None or o is None:
                continue
            out.append(Triple(subject=_mention(s, d.get("subject_type") or d.get("head_type")),
                              predicate=str(p),
                              object=_mention(o, d.get("object_type") or d.get("tail_type")),
                              confidence=d.get("confidence"), evidence=d.get("evidence")))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            conf = item[3] if len(item) > 3 and isinstance(item[3], (int, float)) else None
            out.append(Triple(subject=_mention(item[0]), predicate=str(item[1]),
                              object=_mention(item[2]), confidence=conf))
        elif hasattr(item, "__dict__"):  # dataclass / object with attributes
            out.extend(to_triples([vars(item)], field_map))
    return out


def _union_find(pairs: Iterable[tuple[str, str]], ids: Iterable[str]) -> list[list[str]]:
    parent: dict[str, str] = {i: i for i in ids}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return list(groups.values())


def clusters_from_groups(groups: Iterable[Iterable[str]], mentions: list[Mention] | None = None) -> list[EntityCluster]:
    by_id = {m.mention_id: m for m in mentions or []}
    out = []
    for i, g in enumerate(groups):
        g = list(g)
        first = by_id.get(g[0]) if g else None
        out.append(EntityCluster(canonical_id=f"e{i}", canonical_name=first.text if first else (g[0] if g else ""),
                                 type=first.type if first else "", mention_ids=g))
    return out


def to_clusters(obj: Any, mentions: list[Mention] | None = None, field_map: dict | None = None) -> list[EntityCluster]:
    ids = [m.mention_id for m in mentions or []]
    if obj is None:
        return clusters_from_groups([[i] for i in ids], mentions)
    if isinstance(obj, dict) and "clusters" in obj:
        return to_clusters(obj["clusters"], mentions, field_map)
    if isinstance(obj, dict):  # mention_id -> cluster id
        groups: dict[Any, list[str]] = {}
        for mid, cid in obj.items():
            groups.setdefault(cid, []).append(str(mid))
        return _fill_singletons(clusters_from_groups(groups.values(), mentions), ids, mentions)
    items = list(obj)
    if not items:
        return clusters_from_groups([[i] for i in ids], mentions)
    first = items[0]
    if isinstance(first, EntityCluster):
        return _fill_singletons(items, ids, mentions)
    if isinstance(first, dict):
        cl = []
        for i, d in enumerate(items):
            d = _remap(d, field_map)
            mids = d.get("mention_ids") or d.get("members") or d.get("mentions") or []
            cl.append(EntityCluster(canonical_id=str(d.get("canonical_id", d.get("id", f"e{i}"))),
                                    canonical_name=str(d.get("canonical_name", d.get("name", ""))),
                                    type=str(d.get("type", "")), mention_ids=[str(m) for m in mids]))
        return _fill_singletons(cl, ids, mentions)
    # list of groups or list of match pairs: union-find handles both (overlaps merge transitively)
    pairs = [(str(g[i]), str(g[i + 1])) for g in map(list, items) for i in range(len(g) - 1)]
    singles = [str(g[0]) for g in map(list, items) if len(g) == 1]
    return clusters_from_groups(_union_find(pairs, list(dict.fromkeys(ids + singles))), mentions)


def _fill_singletons(cl: list[EntityCluster], ids: list[str], mentions) -> list[EntityCluster]:
    seen = {m for c in cl for m in c.mention_ids}
    extra = [[i] for i in ids if i not in seen]
    if extra:
        more = clusters_from_groups(extra, mentions)
        for j, c in enumerate(more):
            c.canonical_id = f"singleton{j}"
        cl = list(cl) + more
    return cl


def nx_to_kg(g: Any) -> KnowledgeGraph:
    nodes = [{"id": str(n), "name": str(d.get("name", n)), **{k: v for k, v in d.items() if k != "name"}}
             for n, d in g.nodes(data=True)]
    edges = []
    for u, v, d in g.edges(data=True):
        p = d.get("predicate") or d.get("relation") or d.get("label") or d.get("type") or ""
        edges.append({"source": str(u), "target": str(v), "predicate": str(p)})
    return KnowledgeGraph(nodes=nodes, edges=edges)


def kg_to_triples(kg: KnowledgeGraph) -> list[Triple]:
    nodes = {n["id"]: n for n in kg.nodes}
    out = []
    for e in kg.edges:
        s, o = nodes.get(e["source"], {}), nodes.get(e["target"], {})
        out.append(Triple(subject=Mention(text=str(s.get("name", e["source"])), type=str(s.get("type", ""))),
                          predicate=str(e.get("predicate", "")),
                          object=Mention(text=str(o.get("name", e["target"])), type=str(o.get("type", "")))))
    return out


def build_kg(triples: list[Triple], clusters: list[EntityCluster] | None = None) -> KnowledgeGraph:
    """Default graph builder: one node per canonical entity (or per normalised name)."""
    from gre.core.textnorm import norm
    mention_to_cluster: dict[str, EntityCluster] = {}
    for c in clusters or []:
        for m in c.mention_ids:
            mention_to_cluster[m] = c
    nodes: dict[str, dict] = {}

    def node_for(m: Mention) -> str:
        c = mention_to_cluster.get(m.mention_id) if m.mention_id else None
        nid = f"c:{c.canonical_id}" if c else f"n:{norm(m.text)}"
        if nid not in nodes:
            nodes[nid] = {"id": nid, "name": (c.canonical_name if c and c.canonical_name else m.text),
                          "type": m.type or (c.type if c else ""), "aliases": []}
        if m.text != nodes[nid]["name"] and m.text not in nodes[nid]["aliases"]:
            nodes[nid]["aliases"].append(m.text)
        return nid

    edges, seen = [], set()
    for t in triples:
        s, o = node_for(t.subject), node_for(t.object)
        k = (s, t.predicate, o)
        if k not in seen:
            seen.add(k)
            edges.append({"source": s, "target": o, "predicate": t.predicate})
    return KnowledgeGraph(nodes=list(nodes.values()), edges=edges, clusters=clusters or [])


def to_kg(obj: Any, field_map: dict | None = None) -> KnowledgeGraph:
    if isinstance(obj, KnowledgeGraph):
        return obj
    if _is_nx(obj):
        return nx_to_kg(obj)
    if isinstance(obj, dict) and "nodes" in obj and "edges" in obj:
        return KnowledgeGraph(**obj)
    return build_kg(to_triples(obj, field_map))
