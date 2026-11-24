"""`gre` command line: init · run · score · report · compare · check · datasets · metrics · adapters."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="GraphEvals — benchmark knowledge-graph construction pipelines against SOTA datasets.")
datasets_app = typer.Typer(no_args_is_help=True, help="List / download / inspect benchmark datasets.")
app.add_typer(datasets_app, name="datasets")
con = Console(width=max(120, int(os.environ.get("COLUMNS", "0") or 0)) if not sys.stdout.isatty() else None)


def _env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    sys.path.insert(0, str(Path.cwd()))   # so `callable: my_pkg.mod:fn` resolves from the project root


# ------------------------------------------------------------------------------------------------
@app.command()
def run(config: Path = typer.Argument(..., exists=True, help="experiment YAML"),
        set_: list[str] = typer.Option([], "--set", help="override, e.g. --set matrix.llm=[gpt,claude]"),
        suite: Optional[str] = typer.Option(None, help="add a dataset suite, e.g. kg_sota"),
        dataset: list[str] = typer.Option([], "--dataset", "-d", help="replace datasets (repeatable)"),
        stages: Optional[str] = typer.Option(None, help="comma list: evaluate only these stages"),
        quick: bool = typer.Option(False, "--quick", help="stratified subsample (runtime.quick_limit per dataset)"),
        limit: Optional[int] = typer.Option(None, help="first N samples per dataset"),
        dry_run: bool = typer.Option(False, "--dry-run", help="show expanded runs + cost estimate only"),
        force: bool = typer.Option(False, help="recompute runs even if up to date"),
        no_cache: bool = typer.Option(False, "--no-cache", help="ignore cached LLM calls / stage outputs"),
        report: bool = typer.Option(True, help="build the report afterwards"),
        verbose: bool = typer.Option(False, "-v")):
    """Run an experiment matrix (resumable + cached)."""
    _env()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    from gre.core.config import load_config
    from gre.core.runner import Runner
    overrides = list(set_)
    if suite:
        overrides.append(f"suite={suite}")
    if dataset:
        overrides.append("datasets=[" + ",".join(dataset) + "]")
    if no_cache:
        overrides.append("runtime.cache=false")
    cfg = load_config(config, overrides)
    runner = Runner(cfg, quick=quick, limit=limit, stages=stages.split(",") if stages else None, force=force,
                    progress=lambda s: con.print(s, highlight=False, markup=False))
    plan = runner.plan()
    if dry_run:
        _print_plan(runner, plan)
        return
    if not plan:
        con.print("[yellow]Nothing to run: no pipeline stage matches the datasets' supported stages.[/]")
        raise typer.Exit(1)
    summaries = runner.run_all(plan)
    runner.close()
    _print_summary(summaries)
    if report and summaries:
        from gre.report.builder import build_report
        paths = build_report(runner.out)
        con.print("\n[bold]Report:[/] " + "  ".join(str(p) for p in paths))


def _print_plan(runner, plan):
    from gre.core.runner import resolve_stage  # noqa: F401
    t = Table(title=f"{len(plan)} runs", show_lines=False)
    for c in ("dataset", "pipeline / matrix", "mode", "seed", "stages", "samples", "≈LLM calls", "≈$"):
        t.add_column(c)
    total = 0.0
    for r in plan:
        ds, samples = runner.load_dataset(r.dataset)
        calls, cost = _estimate(runner, r, samples)
        total += cost
        t.add_row(r.dataset, escape(r.variant.label), r.mode, str(r.seed), ",".join(r.stages), str(len(samples)),
                  str(calls), f"{cost:.3f}")
    con.print(t)
    for msg in getattr(runner, "skipped", []):
        con.print(f"[dim]skip duplicate: {escape(msg)}[/]")
    con.print(f"Estimated total ≈ ${total:.2f} (rough: ~4 chars/token, 400 output tokens per call; "
              f"cache hits not counted). Budget: {runner.cfg.budget.get('max_cost_usd')}")


def _price(model: str) -> tuple[float, float]:
    try:
        import litellm
        info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
        return info.get("input_cost_per_token", 0.0), info.get("output_cost_per_token", 0.0)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _estimate(runner, r, samples) -> tuple[int, float]:
    calls, cost = 0, 0.0
    for st in r.stages:
        spec = r.variant.stages.get(st, {})
        prov = spec.get("provider")
        if not prov:
            continue
        model = runner.cfg.providers.get(prov, {}).get("model", prov)
        pin, pout = _price(model)
        for s in samples:
            if st == "extract":
                n = max(1, len(s.chunks))
                tin = sum(len(c.text) for c in s.chunks) / 4 + 700 * n
            elif st == "resolve":
                n = len(s.candidate_pairs) if s.candidate_pairs else max(1, len(s.mentions)) * 2
                tin = 120 * n
            else:
                n, tin = 1, 500
            calls += n
            cost += tin * pin + 400 * n * pout
    if runner.cfg.judge and "graph" in r.stages:
        model = runner.cfg.providers.get(runner.cfg.judge.get("provider"), {}).get("model", "")
        pin, pout = _price(model)
        n = sum(len(s.gold_facts or []) for s in samples)
        calls += n
        cost += n * (900 * pin + 2 * pout)
    return calls, cost


def _print_summary(summaries):
    t = Table(title="Summary (primary metrics)")
    for c in ("dataset", "pipeline · mode", "metric", "value", "95% CI", "$ total", "errors"):
        t.add_column(c)
    for s in summaries:
        heads = [(k, v) for k, v in s["metrics"].items() if v.get("headline")]
        for k, v in heads[:3] or list(s["metrics"].items())[:1]:
            ci = f"{100 * v['ci_low']:.1f}–{100 * v['ci_high']:.1f}" if v.get("ci_low") is not None else ""
            t.add_row(s["dataset"], escape(s["label"]), k, f"{100 * v['value']:.1f}" if v["value"] is not None else "–",
                      ci, f"{s['usage']['total']['cost_usd']:.4f}", f"{100 * s['usage']['error_rate']:.0f}%")
    con.print(t)


# ------------------------------------------------------------------------------------------------
@app.command()
def report(exp_dir: Path = typer.Argument(..., exists=True),
           format: list[str] = typer.Option(["html", "md", "json"], "--format", "-f"),
           baseline: Optional[str] = typer.Option(None, help="label substring of the baseline row")):
    """Build HTML / Markdown / JSON report for an experiment directory."""
    _env()
    from gre.report.builder import build_report
    for p in build_report(exp_dir, format, baseline):
        con.print(f"wrote {p}")


@app.command()
def score(exp_dir: Path = typer.Argument(..., exists=True),
          config: Optional[Path] = typer.Option(None, help="config with new scoring/plugins (default: stored)")):
    """Re-score stored predictions with the current metrics (e.g. after adding a custom metric)."""
    _env()
    from gre.core.config import ExperimentConfig, PipelineVariant, load_config
    from gre.core.contracts import Usage  # noqa: F401
    from gre.core.runner import Runner, RunSpec, SampleResult, _load
    from gre.results import write_results_db
    cfg = load_config(config) if config else ExperimentConfig(**json.loads((exp_dir / "config.json").read_text()))
    runner = Runner(cfg, out_dir=exp_dir.parent)
    runner.out = exp_dir
    n = 0
    for sp in sorted(exp_dir.glob("*/scores.json")):
        old = json.loads(sp.read_text())
        variant = PipelineVariant(pipeline=old["pipeline"], matrix=old["matrix"], stages=old["stage_specs"])
        r = RunSpec(old["exp"], variant, old["dataset"], old["mode"], old["seed"], old["stages"], old["auto_graph"])
        runner.quick = old.get("quick", False)
        ds, samples = runner.load_dataset(old["dataset"])
        by_id = {s.id: s for s in samples}
        results = []
        for line in (sp.parent / "samples.jsonl").read_text().splitlines():
            rec = json.loads(line)
            s = by_id.get(rec["sample_id"])
            if s is None:
                continue
            outputs = {st: _load(st, v) for st, v in rec["preds"].items()}
            res = SampleResult(rec["sample_id"], rec["preds"], usage=rec["usage"], error=rec.get("error"))
            on_gold = bool(s.mentions) and (r.mode == "isolated" or "extract" not in r.stages)
            res.metrics = runner.score_sample(r, s, outputs, resolve_on_gold=on_gold)
            results.append(res)
        runner._persist(r, sp.parent, ds, samples, results, old["spec_hash"], old["wall_time_s"], old.get("partial", False))
        n += 1
    write_results_db(exp_dir)
    runner.close()
    con.print(f"re-scored {n} runs")


# ------------------------------------------------------------------------------------------------
def _index(exp_dir: Path) -> dict:
    """(label, dataset) -> run, with usage numbers exposed as `usage.*` metrics for check/compare."""
    from gre.results import load_runs
    out = {}
    for r in load_runs(exp_dir):
        u = r["usage"]
        for k in ("error_rate", "json_valid_rate", "cost_per_sample_usd", "latency_per_sample_s"):
            r["metrics"][f"usage.{k}"] = {"value": u.get(k), "higher_is_better": k == "json_valid_rate"}
        out[(r["label"], r["dataset"])] = r
    return out


@app.command()
def compare(base: Path = typer.Argument(..., exists=True), new: Path = typer.Argument(..., exists=True),
            metric: Optional[str] = typer.Option(None, help="metric to compare (default: headline metrics)")):
    """Diff two experiments run-by-run with paired bootstrap significance."""
    from gre.metrics.stats import paired_bootstrap
    from gre.results import per_sample_series
    a, b = _index(base), _index(new)
    t = Table(title=f"{new.name} vs {base.name}")
    for c in ("dataset", "pipeline · mode", "metric", "base", "new", "Δ", "p"):
        t.add_column(c)
    for key in sorted(set(a) & set(b)):
        ra, rb = a[key], b[key]
        ms = [metric] if metric else [k for k, v in rb["metrics"].items() if v.get("headline")]
        for m in ms:
            va, vb = ra["metrics"].get(m, {}).get("value"), rb["metrics"].get(m, {}).get("value")
            if va is None or vb is None:
                continue
            sa = per_sample_series(Path(ra["_dir"]) / "samples.jsonl", m)
            sb = per_sample_series(Path(rb["_dir"]) / "samples.jsonl", m)
            common = [k for k in sa if k in sb and sa[k] is not None and sb[k] is not None]
            pb = paired_bootstrap([sa[k] for k in common], [sb[k] for k in common]) if len(common) > 1 else {}
            d = vb - va
            t.add_row(key[1], escape(key[0]), m, f"{100 * va:.1f}", f"{100 * vb:.1f}",
                      f"[{'green' if d >= 0 else 'red'}]{100 * d:+.1f}[/]",
                      f"{pb['p_value']:.3f}" if pb.get("p_value") is not None else "–")
    con.print(t)


_RULE = re.compile(r"^\s*(?P<metric>[\w.\-*]+)\s*(?P<op><=|>=|<|>)\s*(?P<thr>[-+]?\d*\.?\d+)\s*$")


@app.command()
def check(exp_dir: Path = typer.Argument(..., exists=True),
          baseline: Optional[Path] = typer.Option(None, help="baseline experiment dir (deltas) — else absolute"),
          fail_if: list[str] = typer.Option(..., "--fail-if",
                                            help='e.g. "extract.triple_soft.f1 < -0.02" (delta vs baseline) or '
                                                 '"resolve.pairwise.f1 < 0.7" (absolute, no baseline)')):
    """CI gate: exit 1 if any rule fires. Use in GitHub Actions to block regressions."""
    from fnmatch import fnmatch
    new = _index(exp_dir)
    base = _index(baseline) if baseline else None
    failures = []
    for rule in fail_if:
        m = _RULE.match(rule)
        if not m:
            raise typer.BadParameter(f"bad rule {rule!r}")
        pat, op, thr = m["metric"], m["op"], float(m["thr"])
        for key, r in new.items():
            for name, v in r["metrics"].items():
                if not fnmatch(name, pat) or v["value"] is None:
                    continue
                if base is not None:
                    if key not in base or name not in base[key]["metrics"] or base[key]["metrics"][name]["value"] is None:
                        continue
                    x = v["value"] - base[key]["metrics"][name]["value"]
                else:
                    x = v["value"]
                hit = {"<": x < thr, "<=": x <= thr, ">": x > thr, ">=": x >= thr}[op]
                if hit:
                    failures.append(f"{key[1]} :: {key[0]} :: {name} = {x:+.4f} ({rule})")
    for f in failures:
        con.print(f"[red]FAIL[/] {escape(f)}")
    if failures:
        raise typer.Exit(1)
    con.print("[green]all checks passed[/]")


# ------------------------------------------------------------------------------------------------
@datasets_app.command("list")
def datasets_list():
    """Registered datasets, their stages and licenses."""
    from gre.core.registry import REGISTRY, load_builtins_and_plugins
    from gre.datasets import SUITES
    load_builtins_and_plugins()
    t = Table()
    for c in ("name", "stages", "license", "description"):
        t.add_column(c, overflow="fold")
    for n, cls in sorted(REGISTRY.datasets.items()):
        t.add_row(n, ",".join(cls.supported_stages), cls.license, cls.description)
    con.print(t)
    con.print("Suites: " + "; ".join(f"[bold]{k}[/]={','.join(v)}" for k, v in SUITES.items()))


@datasets_app.command("download")
def datasets_download(names: list[str] = typer.Argument(None, help="dataset names or a suite (default: kg_sota)")):
    """Download datasets into $GRE_DATA_DIR (default ~/.cache/gre/data)."""
    from gre.core.registry import get_dataset, load_builtins_and_plugins
    from gre.datasets import SUITES
    load_builtins_and_plugins()
    todo = []
    for n in names or ["kg_sota"]:
        todo += SUITES.get(n, [n])
    for n in dict.fromkeys(todo):
        ds = get_dataset(n)()
        con.print(f"→ {n} …", end=" ")
        ds.download()
        samples = ds.load(ds.default_split)
        con.print(f"[green]ok[/] {len(samples)} {ds.default_split} samples  ({ds.root})")


@datasets_app.command("info")
def datasets_info(name: str):
    """Show stats and one example sample."""
    from gre.core.registry import get_dataset
    ds = get_dataset(name)()
    ds.download()
    s = ds.load(ds.default_split)
    con.print(ds.info())
    con.print({"samples": len(s), "gold_triples": sum(len(x.gold_triples or []) for x in s),
               "mentions": sum(len(x.mentions) for x in s), "gold_pairs": sum(len(x.gold_pairs or []) for x in s),
               "gold_facts": sum(len(x.gold_facts or []) for x in s)})
    con.print(s[0].model_dump(exclude_defaults=True, by_alias=True), soft_wrap=True)


@app.command()
def metrics():
    """List registered metrics (built-in + plugins)."""
    from gre.core.registry import REGISTRY, load_builtins_and_plugins
    load_builtins_and_plugins()
    t = Table()
    for c in ("metric", "aggregate", "better", "headline", "description"):
        t.add_column(c, overflow="fold")
    for n, m in sorted(REGISTRY.metrics.items()):
        t.add_row(n, m.aggregate, "↑" if m.higher_is_better else "↓", "★" if m.headline else "", m.description)
    con.print(t)


@app.command()
def adapters():
    """List registered stage adapters (built-in + plugins)."""
    from gre.core.registry import REGISTRY, load_builtins_and_plugins
    load_builtins_and_plugins()
    t = Table()
    for c in ("stage", "adapter", "version", "description"):
        t.add_column(c, overflow="fold")
    for (st, n), s in sorted(REGISTRY.stages.items()):
        t.add_row(st, n, s.version, s.description)
    con.print(t)


# ------------------------------------------------------------------------------------------------
_INIT_CFG = """name: my_first_eval
description: Evaluate my KG pipeline against SOTA benchmarks
suite: kg_sota            # text2kgbench (webnlg+tekgen), redocred, mine, wdc_products, abt_buy, amazon_google
# your own gold set runs next to the SOTA ones (format: docs/INTEGRATION.md section 5)
# local_datasets:
#   my_gold: { path: data/my_gold.jsonl, stages: [extract, resolve, graph], schema: schemas/domain.yaml }
# datasets: [my_gold]
seeds: [0]
mode: [isolated, cascaded]

providers:                # any LiteLLM model string; keys come from .env
  gpt:    { model: "openai/gpt-4.1-mini" }
  claude: { model: "anthropic/claude-sonnet-4-5" }
  local:  { model: "ollama/qwen3:8b", api_base: "http://localhost:11434" }
embeddings:
  default: { model: "openai/text-embedding-3-small" }
judge: { provider: gpt, prompt_version: v1 }   # keep fixed across experiments

pipelines:
  baseline:               # built-in reference adapters
    extract: { adapter: llm_structured, provider: "${matrix.llm}" }
    resolve: { adapter: fuzzy, threshold: 0.85 }
  mine:                   # YOUR pipeline: point at your functions, no rewrite needed
    extract: { callable: "my_pipeline:extract", provider: "${matrix.llm}" }
    resolve: { callable: "my_pipeline:resolve" }

matrix:
  llm: [gpt, claude]

scoring: { soft_match_threshold: 0.8, soft_method: fuzzy }
budget:  { max_cost_usd: 20 }
runtime: { workers: 8, quick_limit: 20 }
report:  { formats: [html, md, json], baseline: baseline }
"""

_INIT_PIPE = '''"""Wrap your existing pipeline here. Parameters are injected *by name*:
text, chunks, schema, mentions, candidate_pairs, triples, clusters, sample, ctx, params + YAML params.
Return plain dicts/tuples/networkx — gre coerces them to its contracts."""


def extract(chunks, schema, ctx):
    triples = []
    for ch in chunks:
        # e.g. my_kg_builder.extract(ch.text, llm=ctx.llm) — ctx.llm.complete(prompt) uses the matrix model
        prompt = f"List (subject, relation, object) facts in JSON {{\\"triples\\":[...]}} from: {ch.text}"
        from gre.llm.structured import complete_json
        data = complete_json(ctx.llm, prompt) or {}
        triples += data.get("triples", [])          # dicts with subject/predicate/object (any common key names)
    return triples


def resolve(mentions, candidate_pairs=None):
    # return clusters as list of lists of mention ids, a {mention_id: cluster_id} dict, or match pairs
    by_name = {}
    for m in mentions:
        by_name.setdefault(m.text.lower().strip(), []).append(m.mention_id)
    return list(by_name.values())
'''


@app.command()
def init(path: Path = typer.Argument(Path("."), help="project directory")):
    """Scaffold an experiment config, a pipeline wrapper and .env.example."""
    path.mkdir(parents=True, exist_ok=True)
    files = {path / "experiments" / "my_first_eval.yaml": _INIT_CFG, path / "my_pipeline.py": _INIT_PIPE,
             path / ".env.example": "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n"}
    for p, content in files.items():
        if p.exists():
            con.print(f"skip {p} (exists)")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        con.print(f"created {p}")
    con.print("\nNext:\n  cp .env.example .env   # add keys\n"
              "  gre run experiments/my_first_eval.yaml --quick --dry-run\n"
              "  gre run experiments/my_first_eval.yaml --quick")


if __name__ == "__main__":  # pragma: no cover
    app()
