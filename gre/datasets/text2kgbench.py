"""Text2KGBench (Mihindukulasooriya et al., ISWC 2023): ontology-conformant triple extraction.

Two parts:  DBpedia-WebNLG (19 ontologies, 2,014 test sentences)
            Wikidata-TekGen (10 ontologies, 4,062 test sentences)
Each sample is one sentence + its ontology (GraphSchema). Gold: triples.
Source: https://github.com/cenguix/Text2KGBench   (data: CC BY-SA-derived; check before commercial use)
"""
from __future__ import annotations

import json

from gre.core.contracts import Chunk, EntityType, GraphSchema, Mention, RelationType, Sample, Triple
from gre.core.registry import dataset
from gre.datasets.base import Dataset, fetch

_RAW = "https://raw.githubusercontent.com/cenguix/Text2KGBench/main/data"

_WEBNLG = ["1_university", "2_musicalwork", "3_airport", "4_building", "5_athlete", "6_politician", "7_company",
           "8_celestialbody", "9_astronaut", "10_comicscharacter", "11_meanoftransportation", "12_monument",
           "13_food", "14_writtenwork", "15_sportsteam", "16_city", "17_artist", "18_scientist", "19_film"]
_TEKGEN = ["1_movie", "2_music", "3_sport", "4_book", "5_military", "6_computer", "7_space", "8_politics",
           "9_nature", "10_culture"]


class _Text2KG(Dataset):
    part: str = ""
    ontologies: list[str] = []
    supported_stages = ("extract", "graph")
    license = "CC BY-SA 4.0 (WebNLG) / CC BY-SA (TekGen) — verify"
    homepage = "https://github.com/cenguix/Text2KGBench"
    citation = "Mihindukulasooriya et al. Text2KGBench. ISWC 2023. arXiv:2308.02357"
    stratify_key = "ontology"

    def download(self) -> None:
        for o in self.ontologies:
            fetch(f"{_RAW}/{self.part}/ontologies/{o}_ontology.json", self.root / "ontologies" / f"{o}_ontology.json")
            fetch(f"{_RAW}/{self.part}/ground_truth/ont_{o}_ground_truth.jsonl",
                  self.root / "ground_truth" / f"ont_{o}_ground_truth.jsonl")

    def _schema(self, o: str) -> GraphSchema:
        d = json.loads((self.root / "ontologies" / f"{o}_ontology.json").read_text())
        label = {c["qid"]: c["label"] for c in d.get("concepts", [])}
        return GraphSchema(
            entity_types=[EntityType(name=c["label"]) for c in d.get("concepts", [])],
            relation_types=[RelationType(name=r["label"], domain=label.get(r.get("domain"), r.get("domain") or "Any"),
                                         range=label.get(r.get("range"), r.get("range") or "Any"))
                            for r in d.get("relations", [])])

    def load(self, split: str) -> list[Sample]:
        out = []
        for o in self.ontologies:
            schema = self._schema(o)
            path = self.root / "ground_truth" / f"ont_{o}_ground_truth.jsonl"
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                gold = [Triple(subject=Mention(text=t["sub"]), predicate=t["rel"], object=Mention(text=t["obj"]))
                        for t in r.get("triples", [])]
                out.append(Sample(id=r["id"], dataset=self.name, text=r["sent"],
                                  chunks=[Chunk(chunk_id=r["id"], doc_id=r["id"], text=r["sent"],
                                                char_end=len(r["sent"]))],
                                  schema=schema, gold_triples=gold, meta={"ontology": o}))
        return out


@dataset("text2kgbench_webnlg")
class Text2KGWebNLG(_Text2KG):
    description = "Text2KGBench DBpedia-WebNLG: 19 DBpedia ontologies, ontology-guided sentence→triples."
    part = "dbpedia_webnlg"
    ontologies = _WEBNLG


@dataset("text2kgbench_tekgen")
class Text2KGTekGen(_Text2KG):
    description = "Text2KGBench Wikidata-TekGen: 10 Wikidata ontologies, ontology-guided sentence→triples."
    part = "wikidata_tekgen"
    ontologies = _TEKGEN
