"""MINE-1 (KGGen, Mo et al. 2025): does the built KG retain the facts of a document?

101 short essays x 15 gold facts. A pipeline builds a KG per essay; for each fact we retrieve a
subgraph and ask a judge LLM whether it supports the fact (metric `graph.fact_retention`).
The HF dataset also ships per-essay scores of KGGen / GraphRAG / OpenIE -> used as SOTA references.
Source: https://huggingface.co/datasets/josancamon/kg-gen-MINE-evaluation-dataset
"""
from __future__ import annotations

import json
import re
import urllib.request

from gre.core.contracts import Chunk, Sample
from gre.core.registry import dataset
from gre.datasets.base import Dataset

_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=josancamon/kg-gen-MINE-evaluation-dataset"
         "&config=default&split=train&offset={off}&length=100")


@dataset("mine")
class Mine(Dataset):
    description = "MINE-1 (KGGen): 101 essays x 15 facts; measures fact retention of the built KG (LLM judge)."
    supported_stages = ("extract", "resolve", "graph")
    license = "see HF dataset card (kg-gen, MIT code) — verify"
    homepage = "https://github.com/stair-lab/kg-gen"
    citation = "Mo et al. KGGen: Extracting Knowledge Graphs from Plain Text with Language Models. arXiv:2502.09956"
    quick_limit = 10
    gold_mentions = False   # resolve runs on mentions from the pipeline's own triples (affects the graph)

    def download(self) -> None:
        path = self.root / "mine.jsonl"
        if path.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        rows, off = [], 0
        while True:
            req = urllib.request.Request(_ROWS.format(off=off), headers={"User-Agent": "gre-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            batch = [x["row"] for x in data["rows"]]
            rows += batch
            off += len(batch)
            if not batch or off >= data.get("num_rows_total", 0):
                break
        keep = ("id", "essay_topic", "essay_content", "generated_queries",
                "kggen_accuracy", "graphrag_accuracy", "openie_accuracy")
        path.write_text("\n".join(json.dumps({k: r.get(k) for k in keep}) for r in rows))

    def load(self, split: str) -> list[Sample]:
        out = []
        for line in (self.root / "mine.jsonl").read_text().splitlines():
            r = json.loads(line)
            text = re.sub(r"^```|```$", "", (r["essay_content"] or "").strip()).strip()
            sid = f"mine_{r['id']}"
            out.append(Sample(id=sid, dataset=self.name, text=text,
                              chunks=[Chunk(chunk_id=f"{sid}_p{i}", doc_id=sid, text=p)
                                      for i, p in enumerate(x for x in text.split("\n\n") if x.strip())],
                              gold_facts=list(r["generated_queries"] or []),
                              meta={"topic": r.get("essay_topic"),
                                    "ref": {"KGGen": r.get("kggen_accuracy"), "GraphRAG": r.get("graphrag_accuracy"),
                                            "OpenIE": r.get("openie_accuracy")}}))
        return out
