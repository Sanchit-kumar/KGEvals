"""Default graph builder: canonical nodes from clusters (or normalised names), deduplicated edges."""
from __future__ import annotations

from gre.core.coerce import build_kg
from gre.core.contracts import EntityCluster, KnowledgeGraph, Triple
from gre.core.registry import stage


@stage("graph", "networkx", version="1", description="In-memory graph from triples + clusters (default).")
def networkx_graph(triples: list[Triple], clusters: list[EntityCluster] | None = None) -> KnowledgeGraph:
    return build_kg(triples, clusters)
