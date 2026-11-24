"""Re-DocRED (Tan et al., EMNLP 2022): document-level relation extraction, 96 Wikidata relations.

Each sample = one Wikipedia document. Supports:
  extract : gold entity-level triples (subject/object carry every alias in the doc)
  resolve : gold mentions (vertexSet) -> gold entity clusters (mention clustering / coref)
  graph   : entity coverage + duplicate-node rate on the built graph
Relation P-ids are mapped to Wikidata English labels (fetched once and cached).
Source: https://github.com/tonytan48/Re-DocRED   (DocRED-derived; MIT code, data terms of DocRED)
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from gre.core.contracts import (Chunk, EntityCluster, EntityType, GraphSchema, Mention, RelationType,
                                Sample, Triple)
from gre.core.registry import dataset
from gre.datasets.base import Dataset, fetch

_RAW = "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data"
_TYPES = {"PER": "person", "ORG": "organization", "LOC": "location", "TIME": "time", "NUM": "number",
          "MISC": "miscellaneous"}


def _detok(tokens: list[str]) -> str:
    s = " ".join(tokens)
    for a, b in ((" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), (" 's", "'s"), ("( ", "("), (" )", ")"),
                 (" n't", "n't"), (" %", "%"), ("`` ", '"'), (" ''", '"')):
        s = s.replace(a, b)
    return s


@dataset("redocred")
class ReDocRED(Dataset):
    description = "Re-DocRED: 500 test Wikipedia docs, ~35 gold relation facts/doc over 96 relations; entity clusters."
    supported_stages = ("extract", "resolve", "graph")
    splits = ("train", "dev", "test")
    license = "MIT (repo) / DocRED data terms — verify"
    homepage = "https://github.com/tonytan48/Re-DocRED"
    citation = "Tan et al. Revisiting DocRED. EMNLP 2022. arXiv:2205.12696"

    def download(self) -> None:
        for sp in self.splits:
            fetch(f"{_RAW}/{sp}_revised.json", self.root / f"{sp}_revised.json")
        rel_path = self.root / "rel_info.json"
        if not rel_path.exists():
            pids = sorted({l["r"] for sp in self.splits
                           for d in json.loads((self.root / f"{sp}_revised.json").read_text()) for l in d["labels"]})
            rel_path.write_text(json.dumps(_wikidata_labels(pids), indent=1))

    def load(self, split: str) -> list[Sample]:
        docs = json.loads((self.root / f"{split}_revised.json").read_text())
        rel = json.loads((self.root / "rel_info.json").read_text())
        schema = GraphSchema(entity_types=[EntityType(name=v) for v in _TYPES.values()],
                             relation_types=[RelationType(name=v) for v in sorted(set(rel.values()))])
        out = []
        for i, d in enumerate(docs):
            sid = f"redocred_{split}_{i}"
            sents = [_detok(s) for s in d["sents"]]
            text = " ".join(sents)
            mentions, clusters, ents = [], [], []
            for e, vs in enumerate(d["vertexSet"]):
                typ = _TYPES.get(vs[0]["type"], vs[0]["type"])
                names = list(dict.fromkeys(m["name"] for m in vs))
                ents.append(Mention(mention_id=f"{sid}_e{e}", text=names[0], type=typ, aliases=names[1:]))
                mids = []
                for k, m in enumerate(vs):
                    mid = f"{sid}_e{e}_m{k}"
                    mentions.append(Mention(mention_id=mid, text=m["name"], type=typ, context=sents[m["sent_id"]]))
                    mids.append(mid)
                clusters.append(EntityCluster(canonical_id=f"e{e}", canonical_name=names[0], type=typ, mention_ids=mids))
            gold = [Triple(subject=ents[l["h"]], predicate=rel.get(l["r"], l["r"]), object=ents[l["t"]])
                    for l in d["labels"]]
            out.append(Sample(id=sid, dataset=self.name, text=text,
                              chunks=[Chunk(chunk_id=sid, doc_id=sid, text=text, char_end=len(text))],
                              schema=schema, mentions=mentions, gold_triples=gold, gold_clusters=clusters,
                              meta={"title": d.get("title", "")}))
        return out


def _wikidata_labels(pids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(pids), 50):
        batch = pids[i:i + 50]
        url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=labels&languages=en&ids="
               + urllib.parse.quote("|".join(batch)))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gre-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            for pid, ent in data.get("entities", {}).items():
                out[pid] = ent.get("labels", {}).get("en", {}).get("value", pid)
        except Exception:  # noqa: BLE001  offline: fall back to P-ids
            out.update({p: p for p in batch})
    return out
