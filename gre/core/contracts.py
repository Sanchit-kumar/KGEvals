"""Stage contracts: the typed artifacts every stage reads and writes.

Every adapter / user callable is scored through these models, which is what
makes stages swappable and individually measurable.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

STAGES = ("parse", "chunk", "schema", "extract", "resolve", "graph")


class _M(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class BBox(_M):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class Block(_M):
    id: str
    type: Literal["title", "heading", "paragraph", "table", "formula", "figure",
                  "list", "caption", "header", "footer"]
    text: str
    bbox: BBox | None = None
    order: int = 0


class ParsedDoc(_M):
    doc_id: str
    source_path: str = ""
    blocks: list[Block] = []
    markdown: str = ""
    meta: dict = {}


class Chunk(_M):
    chunk_id: str
    doc_id: str
    text: str
    block_ids: list[str] = []
    char_start: int = 0
    char_end: int = 0


class EntityType(_M):
    name: str
    description: str = ""
    properties: list[str] = []


class RelationType(_M):
    name: str
    domain: str = "Any"
    range: str = "Any"
    description: str = ""


class GraphSchema(_M):
    entity_types: list[EntityType] = []
    relation_types: list[RelationType] = []
    owl_path: str | None = None
    shacl_path: str | None = None

    def relation(self, name: str) -> RelationType | None:
        from gre.core.textnorm import norm_rel
        key = norm_rel(name)
        for r in self.relation_types:
            if norm_rel(r.name) == key:
                return r
        return None


class Mention(_M):
    mention_id: str = ""
    text: str
    type: str = ""
    chunk_id: str = ""
    char_start: int | None = None
    char_end: int | None = None
    aliases: list[str] = []          # gold only: other surface forms of the same entity
    context: str | None = None       # surrounding text (helps ER)
    attributes: dict[str, Any] = {}  # structured records (entity matching)

    def names(self) -> list[str]:
        return [self.text, *self.aliases]


class Triple(_M):
    subject: Mention
    predicate: str
    object: Mention
    confidence: float | None = None
    evidence: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.subject.text, self.predicate, self.object.text)


class EntityCluster(_M):
    canonical_id: str
    canonical_name: str = ""
    type: str = ""
    mention_ids: list[str]


class KnowledgeGraph(_M):
    nodes: list[dict] = []
    edges: list[dict] = []   # {"source","target","predicate", ...}
    clusters: list[EntityCluster] = []
    graph_schema: GraphSchema | None = Field(default=None, alias="schema")

    def edge_texts(self) -> list[str]:
        names = {n["id"]: n.get("name", n["id"]) for n in self.nodes}
        return [f"{names.get(e['source'], e['source'])} {e.get('predicate', '')} "
                f"{names.get(e['target'], e['target'])}" for e in self.edges]


class Sample(_M):
    """One evaluation unit: inputs for the pipeline + gold for each supported stage."""
    id: str
    dataset: str
    text: str | None = None
    chunks: list[Chunk] = []
    graph_schema: GraphSchema | None = Field(default=None, alias="schema")
    mentions: list[Mention] = []                     # resolve input (gold mentions)
    candidate_pairs: list[tuple[str, str]] | None = None
    gold_triples: list[Triple] | None = None
    gold_clusters: list[EntityCluster] | None = None
    gold_pairs: list[tuple[str, str, int]] | None = None  # labelled (a, b, match?)
    gold_facts: list[str] | None = None
    meta: dict = {}


class Usage(_M):
    n_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hits: int = 0
    json_valid: int = 0
    json_invalid: int = 0
    errors: int = 0

    def add(self, other: "Usage") -> None:
        for f in type(self).model_fields:
            setattr(self, f, getattr(self, f) + getattr(other, f))
