"""Integration features: own gold JSONL next to built-in datasets, async callables, cache invalidation."""
import json

import gre
from gre.datasets.local import load_schema

GOLD = [
    {"id": "g1", "text": "Maria Lopez works for Nimbus. Maria was born in Porto.",
     "gold_triples": [["Maria Lopez", "worksFor", "Nimbus"],
                      {"subject": {"text": "Maria Lopez", "aliases": ["Maria"]}, "predicate": "bornIn", "object": "Porto"}],
     "mentions": [{"mention_id": "m1", "text": "Maria Lopez"}, {"mention_id": "m2", "text": "Maria"},
                  {"mention_id": "m3", "text": "Nimbus"}],
     "gold_clusters": [["m1", "m2"], ["m3"]],
     "gold_facts": ["Maria Lopez works for Nimbus"]},
]

PIPE = '''
import re
PAT = r"([A-Z]\\w*(?: [A-Z]\\w*)*) works for ([A-Z]\\w*)"
async def extract(chunks):
    return [(s, "worksFor", o) for c in chunks for s, o in re.findall(PAT, c.text)]
def resolve(mentions):
    return {m.mention_id: m.text.split()[0].lower() for m in mentions}
'''


def _setup(tmp_path, monkeypatch, pipe=PIPE):
    (tmp_path / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in GOLD))
    (tmp_path / "schema.yaml").write_text("entity_types: [Person]\nrelations:\n  - {name: worksFor, domain: Person, range: Org}\n"
                                          "  - bornIn\n")
    (tmp_path / "userpipe.py").write_text(pipe)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return {
        "name": "integ", "mode": ["isolated"], "plugins": ["gre.testing"],
        "local_datasets": {"my_gold": {"path": "gold.jsonl", "stages": ["extract", "resolve", "graph"],
                                       "schema": "schema.yaml"}},
        "datasets": ["my_gold", "toy"],                       # own gold + a built-in benchmark together
        "providers": {"judge": {"model": "mock/judge"}}, "judge": {"provider": "judge"},
        "pipelines": {"mine": {"extract": {"callable": "userpipe:extract"},
                               "resolve": {"callable": "userpipe:resolve"}}},
        "scoring": {"n_bootstrap": 20},
        "runtime": {"workers": 1, "cache_path": str(tmp_path / "c.sqlite")},
        "output_dir": str(tmp_path / "runs"),
    }


def test_local_gold_runs_with_builtin_dataset(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    summaries, paths = gre.evaluate(cfg)
    by_ds = {s["dataset"]: s for s in summaries}
    assert set(by_ds) == {"my_gold", "toy"}
    m = by_ds["my_gold"]["metrics"]
    assert m["extract.triple_strict.p"]["value"] == 1.0          # async extract worked
    assert m["extract.triple_strict.r"]["value"] == 0.5          # bornIn not extracted
    assert m["resolve.bcubed.f1"]["value"] == 1.0
    html = (tmp_path / "runs" / "integ" / "report" / "report.html").read_text()
    assert "my_gold" in html and "toy" in html


def test_cache_invalidates_on_module_constant_change(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    s1, _ = gre.evaluate(cfg, report=False)
    # change only a module-level constant: outputs must be recomputed, not served from cache
    (tmp_path / "userpipe.py").write_text(PIPE.replace("works for", "employed by"))
    import importlib
    import sys
    sys.modules.pop("userpipe", None)
    importlib.invalidate_caches()
    from gre.core.config import ExperimentConfig
    from gre.core.runner import Runner
    r = Runner(ExperimentConfig(**cfg), force=True)
    s2 = r.run_all()
    r.close()
    gold = {s["dataset"]: s for s in s2}["my_gold"]["metrics"]
    assert gold["extract.triple_strict.r"]["value"] == 0.0


def test_schema_short_form(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("entity_types: [A]\nrelations: [r1, {name: r2, domain: A, range: B}]\n")
    s = load_schema(p)
    assert [r.name for r in s.relation_types] == ["r1", "r2"] and s.relation("r2").range == "B"
