"""Tiny offline dataset covering extract / resolve / graph — used by tests and `gre init`."""
from __future__ import annotations

from gre.core.contracts import (Chunk, EntityCluster, EntityType, GraphSchema, Mention, RelationType,
                                Sample, Triple)
from gre.core.registry import dataset
from gre.datasets.base import Dataset

_SCHEMA = GraphSchema(
    entity_types=[EntityType(name=n) for n in ("Person", "Organization", "City")],
    relation_types=[RelationType(name="worksFor", domain="Person", range="Organization"),
                    RelationType(name="locatedIn", domain="Organization", range="City"),
                    RelationType(name="bornIn", domain="Person", range="City")],
)

_DOCS = [
    ("d1", "Alice Smith works for Acme Corp. Acme Corp is located in Berlin. Alice was born in Paris.",
     [("Alice Smith", "Person", "worksFor", "Acme Corp", "Organization"),
      ("Acme Corp", "Organization", "locatedIn", "Berlin", "City"),
      ("Alice Smith", "Person", "bornIn", "Paris", "City")]),
    ("d2", "Bob Jones works for Globex. Globex is located in Tokyo.",
     [("Bob Jones", "Person", "worksFor", "Globex", "Organization"),
      ("Globex", "Organization", "locatedIn", "Tokyo", "City")]),
    ("d3", "Carol White works for Initech. Initech is located in Austin. Carol was born in Austin.",
     [("Carol White", "Person", "worksFor", "Initech", "Organization"),
      ("Initech", "Organization", "locatedIn", "Austin", "City"),
      ("Carol White", "Person", "bornIn", "Austin", "City")]),
    ("d4", "Dan Brown works for Umbrella. Umbrella is located in Raccoon City.",
     [("Dan Brown", "Person", "worksFor", "Umbrella", "Organization"),
      ("Umbrella", "Organization", "locatedIn", "Raccoon City", "City")]),
]

# mentions for resolve: surface forms with gold clusters
_MENTIONS = {
    "d1": [("Alice Smith", "Person", "alice"), ("Alice", "Person", "alice"), ("Acme Corp", "Organization", "acme"),
           ("Acme", "Organization", "acme"), ("Berlin", "City", "berlin"), ("Paris", "City", "paris")],
    "d2": [("Bob Jones", "Person", "bob"), ("Bob", "Person", "bob"), ("Globex", "Organization", "globex"),
           ("Globex Inc.", "Organization", "globex"), ("Tokyo", "City", "tokyo")],
    "d3": [("Carol White", "Person", "carol"), ("Carol", "Person", "carol"), ("Initech", "Organization", "initech"),
           ("Austin", "City", "austin"), ("Austin, TX", "City", "austin")],
    "d4": [("Dan Brown", "Person", "dan"), ("Umbrella", "Organization", "umb"), ("Umbrella Corp", "Organization", "umb"),
           ("Raccoon City", "City", "rc")],
}


@dataset("toy")
class Toy(Dataset):
    description = "4 hand-written docs with gold triples, mention clusters and facts (offline smoke test)."
    supported_stages = ("extract", "resolve", "graph")
    license = "CC0"

    def load(self, split: str) -> list[Sample]:
        out = []
        for doc_id, text, triples in _DOCS:
            gold = [Triple(subject=Mention(text=s, type=st), predicate=p, object=Mention(text=o, type=ot))
                    for s, st, p, o, ot in triples]
            mentions, groups = [], {}
            for i, (t, typ, ent) in enumerate(_MENTIONS[doc_id]):
                mid = f"{doc_id}_m{i}"
                mentions.append(Mention(mention_id=mid, text=t, type=typ, context=text))
                groups.setdefault(ent, []).append(mid)
            clusters = [EntityCluster(canonical_id=k, mention_ids=v) for k, v in groups.items()]
            facts = [f"{s} {p} {o}" for s, _, p, o, _ in triples]
            out.append(Sample(id=doc_id, dataset="toy", text=text,
                              chunks=[Chunk(chunk_id=f"{doc_id}_c0", doc_id=doc_id, text=text, char_end=len(text))],
                              schema=_SCHEMA, mentions=mentions, gold_triples=gold, gold_clusters=clusters,
                              gold_facts=facts))
        return out
