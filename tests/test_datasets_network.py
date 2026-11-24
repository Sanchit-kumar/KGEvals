"""Loader checks against the real public datasets. Run with: pytest -m network"""
import pytest

from gre.core.registry import get_dataset, load_builtins_and_plugins

pytestmark = pytest.mark.network
load_builtins_and_plugins()

EXPECT = {  # name: (min samples, field that must be populated)
    "text2kgbench_webnlg": (2000, "gold_triples"),
    "text2kgbench_tekgen": (4000, "gold_triples"),
    "redocred": (500, "gold_clusters"),
    "mine": (100, "gold_facts"),
    "wdc_products": (80, "gold_pairs"),
    "abt_buy": (35, "gold_pairs"),
    "amazon_google": (40, "gold_pairs"),
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_loader(name):
    ds = get_dataset(name)()
    ds.download()
    samples = ds.load(ds.default_split)
    n, field = EXPECT[name]
    assert len(samples) >= n
    assert all(getattr(s, field) for s in samples[:20])
    if "resolve" in ds.supported_stages:
        ids = {m.mention_id for m in samples[0].mentions}
        for a, b, *_ in samples[0].gold_pairs or []:
            assert a in ids and b in ids


def test_redocred_relation_labels_resolved():
    s = get_dataset("redocred")().load("test")[0]
    assert not any(t.predicate.startswith("P") and t.predicate[1:].isdigit() for t in s.gold_triples)
