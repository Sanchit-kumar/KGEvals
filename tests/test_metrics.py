"""Metric correctness against hand-computed values."""
import pytest

from gre.core.config import ScoringCfg
from gre.core.context import Context
from gre.core.contracts import EntityCluster, GraphSchema, Mention, RelationType, Sample, Triple
from gre.metrics import er, extract
from gre.metrics.stats import aggregate_prf, paired_bootstrap

CTX = Context(stage="x", scoring=ScoringCfg())


def _er_sample():
    ms = [Mention(mention_id=i, text=i) for i in "abcde"]
    gold = [EntityCluster(canonical_id="g1", mention_ids=list("abc")), EntityCluster(canonical_id="g2", mention_ids=list("de"))]
    return Sample(id="s", dataset="t", mentions=ms, gold_clusters=gold)


PRED = [EntityCluster(canonical_id="p1", mention_ids=list("ab")), EntityCluster(canonical_id="p2", mention_ids=list("cde"))]


def test_bcubed():
    r = er.bcubed(PRED, _er_sample(), CTX)
    assert r["p"] == pytest.approx(11 / 15) and r["r"] == pytest.approx(11 / 15)


def test_muc():
    r = er.muc(PRED, _er_sample(), CTX)
    assert r["p"] == pytest.approx(2 / 3) and r["r"] == pytest.approx(2 / 3)


def test_ceaf_e():
    r = er.ceaf_e(PRED, _er_sample(), CTX)
    assert r["p"] == pytest.approx(0.8) and r["r"] == pytest.approx(0.8)


def test_ari():
    assert er.ari(PRED, _er_sample(), CTX) == pytest.approx(0.4 / 2.4)
    s = _er_sample()
    assert er.ari(s.gold_clusters, s, CTX) == pytest.approx(1.0)


def test_pairwise_from_clusters_and_pairs():
    assert er.pairwise(PRED, _er_sample(), CTX) == {"tp": 2, "fp": 2, "fn": 2}
    s = Sample(id="s", dataset="t", mentions=[Mention(mention_id=i, text=i) for i in "abcd"],
               gold_pairs=[("a", "b", 1), ("c", "d", 0), ("a", "c", 1)])
    pred = [EntityCluster(canonical_id="x", mention_ids=["a", "b"]), EntityCluster(canonical_id="y", mention_ids=["c", "d"])]
    assert er.pairwise(pred, s, CTX) == {"tp": 1, "fp": 1, "fn": 1}


def test_merge_errors():
    r = er.merge_errors(PRED, _er_sample(), CTX)
    assert r["over_merge"] == pytest.approx(0.5) and r["under_merge"] == pytest.approx(0.5)


def _t(s, p, o, **kw):
    return Triple(subject=Mention(text=s, **kw.get("s", {})), predicate=p, object=Mention(text=o))


def test_triple_strict_aliases_and_camelcase():
    gold = [Triple(subject=Mention(text="Alice Smith", aliases=["Alice"]), predicate="bornIn", object=Mention(text="Paris")),
            _t("Acme", "locatedIn", "Berlin")]
    s = Sample(id="s", dataset="t", gold_triples=gold)
    pred = [_t("alice", "born in", "Paris"), _t("Acme", "locatedIn", "Munich"), _t("alice", "born in", "paris")]
    assert extract.triple_strict(pred, s, CTX) == {"tp": 1, "fp": 1, "fn": 1}   # duplicate deduped


def test_triple_soft_more_lenient_than_strict():
    s = Sample(id="s", dataset="t", gold_triples=[_t("Insular_Government_of_the_Philippine_Islands", "foundationPlace", "Manila")])
    pred = [_t("Insular Government of Philippine Islands", "foundationPlace", "Manila")]
    assert extract.triple_strict(pred, s, CTX)["tp"] == 0
    assert extract.triple_soft(pred, s, CTX)["tp"] == 1


def test_conformance_hallucination_domain_range():
    schema = GraphSchema(relation_types=[RelationType(name="worksFor", domain="Person", range="Organization")])
    s = Sample(id="s", dataset="t", text="Bob works for Globex.", schema=schema, gold_triples=[])
    pred = [Triple(subject=Mention(text="Bob", type="Person"), predicate="worksFor", object=Mention(text="Globex", type="City")),
            _t("Bob", "likes", "Pizza")]
    assert extract.ontology_conformance(pred, s, CTX) == 0.5
    h = extract.hallucination(pred, s, CTX)
    assert h["relation"] == 0.5 and h["object"] == 0.5 and h["subject"] == 0.0
    assert extract.domain_range_violation(pred, s, CTX) == 0.5


def test_aggregate_prf_micro():
    out = aggregate_prf([{"tp": 1, "fp": 1, "fn": 0}, {"tp": 3, "fp": 0, "fn": 1}], n_boot=50)
    assert out["p"]["value"] == pytest.approx(0.8) and out["r"]["value"] == pytest.approx(0.8)
    assert out["f1"]["ci_low"] <= out["f1"]["value"] <= out["f1"]["ci_high"]


def test_paired_bootstrap_detects_difference():
    a = [0.1] * 30
    b = [0.9] * 30
    r = paired_bootstrap(a, b)
    assert r["delta"] == pytest.approx(0.8) and r["p_value"] < 0.01
