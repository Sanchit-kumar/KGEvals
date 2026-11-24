"""End-to-end: toy experiment with mock models, BYO pipeline, caching, custom metric, re-score, CI gate."""
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import gre
from gre.cli import app

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, **extra):
    cfg = {
        "name": "e2e", "datasets": ["toy"], "mode": ["isolated", "cascaded"], "plugins": ["gre.testing"],
        "providers": {"good": {"model": "mock/good", "price_in": 1.0, "price_out": 1.0},
                      "lazy": {"model": "mock/lazy"}, "judge": {"model": "mock/judge"}},
        "judge": {"provider": "judge"},
        "pipelines": {"base": {"extract": {"adapter": "llm_structured", "provider": "${matrix.llm}"},
                               "resolve": {"adapter": "fuzzy", "threshold": 0.6}}},
        "matrix": {"llm": ["good", "lazy"]},
        "scoring": {"n_bootstrap": 50},
        "runtime": {"workers": 2, "cache_path": str(tmp_path / "cache.sqlite")},
        "output_dir": str(tmp_path / "runs"),
    }
    cfg.update(extra)
    return cfg


def test_toy_end_to_end(tmp_path):
    summaries, paths = gre.evaluate(_cfg(tmp_path))
    assert len(summaries) == 4
    by = {s["label"]: s for s in summaries}
    good, lazy = by["base[llm=good] · isolated"], by["base[llm=lazy] · isolated"]
    assert good["metrics"]["extract.triple_strict.f1"]["value"] > lazy["metrics"]["extract.triple_strict.f1"]["value"]
    assert good["metrics"]["resolve.bcubed.f1"]["value"] == pytest.approx(1.0)
    assert good["metrics"]["graph.fact_retention"]["value"] > lazy["metrics"]["graph.fact_retention"]["value"]
    assert good["usage"]["total"]["cost_usd"] > 0
    assert "resolve.bcubed.f1" not in by["base[llm=good] · cascaded"]["metrics"]   # no gold alignment in cascade
    out = tmp_path / "runs" / "e2e"
    assert (out / "results.duckdb").exists() and (out / "results.csv").exists()
    assert {p.name for p in paths} == {"report.html", "report.md", "report.json"}
    assert "Leaderboard" in (out / "report" / "report.html").read_text()


def test_cache_makes_rerun_free(tmp_path):
    gre.evaluate(_cfg(tmp_path), report=False)
    # force recompute: stage outputs come from cache, no new LLM calls
    from gre.core.config import ExperimentConfig
    from gre.core.runner import Runner
    r = Runner(ExperimentConfig(**_cfg(tmp_path)), force=True)
    s = r.run_all()
    r.close()
    assert all(x["usage"]["total"]["cache_hits"] > 0 for x in s)


def test_bring_your_own_pipeline_and_metric(tmp_path, monkeypatch):
    pkg = tmp_path / "mypipe.py"
    pkg.write_text(
        "import gre\n"
        "def extract(text):\n"
        "    import re\n"
        "    return [(a, 'worksFor', b) for a, b in re.findall(r'(\\w+ \\w+) works for (\\w+(?: Corp)?)', text)]\n"
        "class Resolver:\n"
        "    def __init__(self, lower=True): self.lower = lower\n"
        "    def run(self, mentions):\n"
        "        return {m.mention_id: m.text.split()[0].lower() for m in mentions}\n"
        "@gre.metric('extract', 'starts_upper')\n"
        "def starts_upper(pred, sample, ctx):\n"
        "    return sum(t.subject.text[:1].isupper() for t in pred) / len(pred) if pred else None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = _cfg(tmp_path, plugins=["gre.testing", "mypipe"], mode=["isolated"],
               pipelines={"mine": {"extract": {"callable": "mypipe:extract"},
                                   "resolve": {"callable": "mypipe:Resolver", "lower": True}}}, matrix={})
    summaries, _ = gre.evaluate(cfg, report=False)
    m = summaries[0]["metrics"]
    assert m["extract.starts_upper"]["value"] == 1.0
    assert 0 < m["extract.triple_strict.r"]["value"] < 1
    assert "resolve.bcubed.f1" in m


def test_cli_score_and_check(tmp_path):
    gre.evaluate(_cfg(tmp_path), report=False)
    exp = tmp_path / "runs" / "e2e"
    cli = CliRunner()
    r = cli.invoke(app, ["score", str(exp)])
    assert r.exit_code == 0, r.output
    ok = cli.invoke(app, ["check", str(exp), "--fail-if", "extract.triple_strict.f1 < 0.01"])
    assert ok.exit_code == 0, ok.output
    bad = cli.invoke(app, ["check", str(exp), "--fail-if", "extract.triple_strict.f1 < 0.99"])
    assert bad.exit_code == 1
    rep = cli.invoke(app, ["report", str(exp), "-f", "md"])
    assert rep.exit_code == 0 and (exp / "report" / "report.md").exists()


def test_failures_score_as_empty(tmp_path, monkeypatch):
    (tmp_path / "boom.py").write_text("def extract(chunks):\n    raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = _cfg(tmp_path, mode=["isolated"], matrix={},
               pipelines={"broken": {"extract": {"callable": "boom:extract"}}})
    s, _ = gre.evaluate(cfg, report=False)
    assert s[0]["usage"]["error_rate"] == 1.0
    assert s[0]["metrics"]["extract.triple_strict.r"]["value"] == 0.0


def test_budget_guard_stops(tmp_path):
    cfg = _cfg(tmp_path, budget={"max_cost_usd": 1e-9}, mode=["isolated"])
    cfg["runtime"]["workers"] = 1
    s, _ = gre.evaluate(cfg, report=False)
    assert len(s) < 2
