"""Entity-matching benchmarks for the resolve stage (pairwise P/R/F1 on labelled pairs).

  wdc_products   WDC Products (Peeters, Der & Bizer, EDBT 2024) — 80% corner cases, seen entities,
                 4,500 labelled test pairs. Hardest current EM benchmark. Variants via options.
  abt_buy        Abt-Buy (Magellan / DeepMatcher), 1,916 test pairs.
  amazon_google  Amazon-Google (Magellan / DeepMatcher), 2,293 test pairs.

Each Sample is a block of `pairs_per_sample` labelled pairs; the records involved become
`mentions` (text = serialised record, attributes = raw fields) and the pairs `candidate_pairs`.
A resolver outputs clusters; a pair is predicted "match" iff both records share a cluster.
"""
from __future__ import annotations

import csv
import gzip
import json
import zipfile

from gre.core.contracts import Mention, Sample
from gre.core.registry import dataset
from gre.datasets.base import Dataset, chunked, fetch


def _serialise(attrs: dict) -> str:
    return "; ".join(f"{k}: {v}" for k, v in attrs.items() if v not in (None, "", "None", "nan"))


def _block_samples(name: str, pairs: list[tuple[dict, dict, int]], per: int) -> list[Sample]:
    """pairs: (left_record, right_record, label); records carry an 'id' key."""
    out = []
    for b, block in enumerate(chunked(pairs, per)):
        mentions: dict[str, Mention] = {}
        cps, gold = [], []
        for left, right, label in block:
            for rec in (left, right):
                if rec["id"] not in mentions:
                    attrs = {k: v for k, v in rec.items() if k != "id"}
                    mentions[rec["id"]] = Mention(mention_id=rec["id"], text=_serialise(attrs), type="product",
                                                  attributes=attrs)
            cps.append((left["id"], right["id"]))
            gold.append((left["id"], right["id"], int(label)))
        out.append(Sample(id=f"{name}_b{b}", dataset=name, mentions=list(mentions.values()),
                          candidate_pairs=cps, gold_pairs=gold))
    return out


class _Magellan(Dataset):
    hf_repo: str = ""
    supported_stages = ("resolve",)
    splits = ("train", "valid", "test")
    license = "research use (Magellan data repository) — verify"
    citation = "Mudgal et al. Deep Learning for Entity Matching. SIGMOD 2018"
    quick_limit = 6

    def download(self) -> None:
        base = f"https://huggingface.co/datasets/{self.hf_repo}/resolve/main"
        for f in ("tableA.csv", "tableB.csv", *(f"{s}.csv" for s in self.splits)):
            fetch(f"{base}/{f}", self.root / f)

    def load(self, split: str) -> list[Sample]:
        def table(fn, prefix):
            with open(self.root / fn, newline="", encoding="utf-8") as f:
                return {r["id"]: {"id": f"{prefix}:{r['id']}", **{k: v for k, v in r.items() if k != "id"}}
                        for r in csv.DictReader(f)}
        A, B = table("tableA.csv", "A"), table("tableB.csv", "B")
        with open(self.root / f"{split}.csv", newline="", encoding="utf-8") as f:
            pairs = [(A[r["ltable_id"]], B[r["rtable_id"]], int(r["label"])) for r in csv.DictReader(f)]
        return _block_samples(self.name, pairs, int(self.options.get("pairs_per_sample", 50)))


@dataset("abt_buy")
class AbtBuy(_Magellan):
    description = "Abt-Buy product matching (Magellan/DeepMatcher benchmark), 1,916 test pairs."
    hf_repo = "matchbench/Abt-Buy"
    homepage = "https://huggingface.co/datasets/matchbench/Abt-Buy"


@dataset("amazon_google")
class AmazonGoogle(_Magellan):
    description = "Amazon-Google software product matching (Magellan/DeepMatcher), 2,293 test pairs."
    hf_repo = "matchbench/Amazon-Google"
    homepage = "https://huggingface.co/datasets/matchbench/Amazon-Google"


@dataset("wdc_products")
class WDCProducts(Dataset):
    description = ("WDC Products: product matching with 80% corner cases (hard negatives / hard positives); "
                   "4,500 labelled test pairs, seen-entity variant by default.")
    supported_stages = ("resolve",)
    license = "research use (Web Data Commons) — verify"
    homepage = "https://webdatacommons.org/largescaleproductcorpus/wdc-products/"
    citation = "Peeters, Der, Bizer. WDC Products: A Multi-Dimensional Entity Matching Benchmark. EDBT 2024"
    quick_limit = 6
    _URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/{cc}pair.zip"

    @property
    def _cc(self) -> str:
        return str(self.options.get("corner_cases", 80))           # 80 | 50 | 20

    @property
    def _unseen(self) -> str:
        return {"0": "000", "50": "050", "100": "100"}[str(self.options.get("unseen", 0))]

    def download(self) -> None:
        fetch(self._URL.format(cc=self._cc), self.root / f"{self._cc}pair.zip")

    def load(self, split: str) -> list[Sample]:
        z = zipfile.ZipFile(self.root / f"{self._cc}pair.zip")
        fn = f"wdcproducts{self._cc}cc20rnd{self._unseen}un_gs.json.gz"
        rows = [json.loads(l) for l in gzip.decompress(z.read(fn)).decode().splitlines() if l.strip()]
        fields = ("brand", "title", "description", "price", "priceCurrency")

        def rec(r, side):
            return {"id": f"W:{r['id_' + side]}", **{f: r.get(f"{f}_{side}") for f in fields}}
        pairs = [(rec(r, "left"), rec(r, "right"), int(r["label"])) for r in rows]
        return _block_samples(self.name, pairs, int(self.options.get("pairs_per_sample", 50)))
