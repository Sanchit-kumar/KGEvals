"""Config expansion, overrides, dataset specs, coercion, binding, cache keys."""
import networkx as nx
import pytest

from gre.core import coerce
from gre.core.cache import content_hash
from gre.core.config import ExperimentConfig, apply_overrides, expand_pipelines
from gre.core.contracts import Mention
from gre.core.runner import call_bound
from gre.datasets.base import parse_dataset_spec, stratified_subsample


def _cfg(**kw):
    base = {"name": "x", "datasets": ["toy"],
            "pipelines": {"a": {"extract": {"adapter": "llm_structured", "provider": "${matrix.llm}"}},
                          "b": {"resolve": {"adapter": "fuzzy", "threshold": "${matrix.thr}"}},
                          "c": {"resolve": {"adapter": "exact"}}},
            "matrix": {"llm": ["m1", "m2"], "thr": [0.8, 0.9]}}
    base.update(kw)
    return ExperimentConfig(**base)


def test_matrix_expands_only_referenced_vars():
    vs = expand_pipelines(_cfg())
    labels = sorted(v.label for v in vs)
    assert labels == ["a[llm=m1]", "a[llm=m2]", "b[thr=0.8]", "b[thr=0.9]", "c"]
    b = next(v for v in vs if v.label == "b[thr=0.9]")
    assert b.stages["resolve"]["threshold"] == 0.9   # type preserved


def test_overrides():
    raw = apply_overrides({"matrix": {"llm": ["a"]}}, ["matrix.llm=[x,y]", "scoring.soft_match_threshold=0.7"])
    assert raw["matrix"]["llm"] == ["x", "y"] and raw["scoring"]["soft_match_threshold"] == 0.7


def test_bad_stage_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", pipelines={"a": {"extrakt": {"adapter": "x"}}})


def test_dataset_spec():
    assert parse_dataset_spec("redocred") == ("redocred", None, None, None)
    assert parse_dataset_spec("redocred:test[:200]") == ("redocred", "test", None, 200)
    assert parse_dataset_spec("abt_buy[10:20]") == ("abt_buy", None, 10, 20)


def test_stratified_subsample_round_robin():
    from gre.core.contracts import Sample
    ss = [Sample(id=f"{k}{i}", dataset="d", meta={"o": k}) for k in "ab" for i in range(10)]
    out = stratified_subsample(ss, 4, lambda s: s.meta["o"])
    assert sorted(s.meta["o"] for s in out) == ["a", "a", "b", "b"]


def test_coerce_triples_many_shapes():
    assert len(coerce.to_triples([{"head": "A", "rel": "r", "tail": "B"}])) == 1
    t = coerce.to_triples([{"src": "A", "edge": "r", "dst": "B"}], field_map={"subject": "src", "predicate": "edge", "object": "dst"})
    assert t[0].subject.text == "A" and t[0].object.text == "B"
    assert coerce.to_triples([("A", "r", "B", 0.9)])[0].confidence == 0.9
    g = nx.DiGraph()
    g.add_edge("A", "B", relation="r")
    assert coerce.to_triples(g)[0].predicate == "r"
    assert len(coerce.to_triples({"triples": [("A", "r", "B")]})) == 1


def test_coerce_clusters_many_shapes():
    ms = [Mention(mention_id=i, text=i) for i in "abcd"]

    def groups(cl):
        return sorted(sorted(c.mention_ids) for c in cl)
    assert groups(coerce.to_clusters({"a": 1, "b": 1, "c": 2}, ms)) == [["a", "b"], ["c"], ["d"]]
    assert groups(coerce.to_clusters([("a", "b"), ("b", "c")], ms)) == [["a", "b", "c"], ["d"]]
    assert groups(coerce.to_clusters([["a", "d"], ["b"]], ms)) == [["a", "d"], ["b"], ["c"]]
    assert groups(coerce.to_clusters(None, ms)) == [["a"], ["b"], ["c"], ["d"]]


def test_call_bound_by_name_and_missing():
    def f(chunks, ctx, threshold=0.5):
        return (chunks, threshold)
    assert call_bound(f, {"chunks": 1, "ctx": None, "threshold": 0.9, "other": 3}) == (1, 0.9)
    with pytest.raises(TypeError, match="needs"):
        call_bound(lambda mentions: 1, {"chunks": 1})


def test_content_hash_stable():
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
