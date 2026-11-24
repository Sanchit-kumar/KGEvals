"""Your own gold data as a benchmark, declared in YAML (no Python needed).

    local_datasets:
      my_gold:
        path: data/my_gold.jsonl          # one sample per line (see docs/INTEGRATION.md)
        stages: [extract, resolve, graph] # what the gold supports
        schema: schemas/domain.yaml       # optional: applied to samples without their own schema

Line format (only `id` + what your stages need):
    {"id": "doc1", "text": "...",
     "gold_triples": [["Alice", "worksFor", "Acme"], {"subject": "Acme", "predicate": "locatedIn", "object": "Berlin"}],
     "mentions": [{"mention_id": "m1", "text": "Alice"}, {"mention_id": "m2", "text": "Alice Smith"}],
     "gold_clusters": [["m1", "m2"]],
     "gold_facts": ["Alice works for Acme"]}
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from gre.core import coerce
from gre.core.contracts import EntityCluster, EntityType, GraphSchema, Mention, RelationType, Sample
from gre.core.registry import REGISTRY
from gre.datasets.base import Dataset


def load_schema(path: str | Path) -> GraphSchema:
    """YAML/JSON schema. Accepts the GraphSchema shape or the short form
    {entity_types: [Person, ...], relations: [{name, domain, range}] | ["worksFor", ...]}."""
    raw = yaml.safe_load(Path(path).read_text())
    if "relation_types" in raw:
        return GraphSchema(**raw)
    ents = [EntityType(name=e) if isinstance(e, str) else EntityType(**e) for e in raw.get("entity_types", [])]
    rels = [RelationType(name=r) if isinstance(r, str) else RelationType(**r) for r in raw.get("relations", [])]
    return GraphSchema(entity_types=ents, relation_types=rels)


class LocalJsonl(Dataset):
    path: str = ""
    schema_path: str | None = None
    description = "Local gold dataset (JSONL)."
    license = "yours"

    @property
    def root(self) -> Path:
        return Path(self.path).parent

    def load(self, split: str) -> list[Sample]:
        schema = load_schema(self.schema_path) if self.schema_path else None
        out = []
        with open(self.path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    out.append(self._sample(json.loads(line), schema))
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"{self.path}:{n}: {e}") from e
        return out

    def _sample(self, d: dict, schema: GraphSchema | None) -> Sample:
        d = dict(d)
        d.setdefault("dataset", self.name)
        if "id" not in d:
            raise ValueError("each line needs an 'id'")
        d["id"] = str(d["id"])
        if d.get("gold_triples") is not None:
            d["gold_triples"] = coerce.to_triples(d["gold_triples"])
        if d.get("mentions"):
            d["mentions"] = [Mention(**m) if isinstance(m, dict) else Mention(mention_id=f"{d['id']}_m{i}", text=str(m))
                             for i, m in enumerate(d["mentions"])]
        if d.get("gold_clusters") is not None:
            cl = d["gold_clusters"]
            d["gold_clusters"] = ([EntityCluster(**c) for c in cl] if cl and isinstance(cl[0], dict)
                                  else coerce.clusters_from_groups(cl, d.get("mentions")))
        if isinstance(d.get("schema"), str):
            d["schema"] = load_schema(Path(self.path).parent / d["schema"])
        elif d.get("schema") is None and schema is not None:
            d["schema"] = schema
        return Sample(**d)


def register_local_datasets(defs: dict[str, dict] | None, base_dir: str | Path = ".") -> None:
    """Register each `local_datasets:` entry as a dataset class (idempotent)."""
    for name, opt in (defs or {}).items():
        path = Path(opt["path"])
        if not path.is_absolute():
            path = Path(base_dir) / path
        schema = opt.get("schema")
        if schema and not Path(schema).is_absolute():
            schema = str(Path(base_dir) / schema)
        cls = type(f"Local_{name}", (LocalJsonl,), {
            "name": name, "path": str(path.resolve()), "schema_path": schema,
            "supported_stages": tuple(opt.get("stages", ("extract", "resolve", "graph"))),
            "gold_mentions": opt.get("gold_mentions"),
            "description": opt.get("description", f"Local gold dataset {path.name}"),
            "license": opt.get("license", "yours"),
        })
        REGISTRY.datasets[name] = cls
