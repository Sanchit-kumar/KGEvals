"""Build HTML / Markdown / JSON reports from an experiment directory."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from gre.core.textnorm import norm, norm_rel
from gre.metrics.stats import paired_bootstrap
from gre.results import load_runs, per_sample_series

TEMPLATES = Path(__file__).parent / "templates"
STAGE_ORDER = {"extract": 0, "resolve": 1, "graph": 2}
# secondary columns always shown in the leaderboard when present
SECONDARY = ["extract.triple_strict.f1", "extract.triple_soft.f1", "extract.triple_soft.p", "extract.triple_soft.r",
             "extract.entity.f1", "extract.ontology_conformance", "extract.hallucination.relation",
             "resolve.pairwise.f1", "resolve.pairwise.p", "resolve.pairwise.r", "resolve.bcubed.f1",
             "resolve.ceaf_e.f1", "graph.fact_retention", "graph.fact_coverage_lexical", "graph.entity_coverage",
             "graph.duplicate_node_rate"]


COUNT_METRICS = ("extract.n_triples", "graph.structure.nodes", "graph.structure.edges",
                 "graph.structure.avg_degree", "graph.structure.relation_types")


def _fmt(v, metric: str | None = None):
    """Percent for 0-1 metrics, plain number for counts."""
    if v is None:
        return "–"
    if metric and metric.startswith(COUNT_METRICS):
        return f"{v:.3g}"
    return f"{100 * v:.1f}"


def _sota_refs(dataset_name: str, runs: list[dict]) -> list[dict]:
    refs = yaml.safe_load((Path(__file__).parent.parent / "datasets" / "sota_refs.yaml").read_text()) or {}
    out = []
    for ref in refs.get(dataset_name, []):
        v = ref.get("value")
        if isinstance(v, str) and v.startswith("from_dataset:"):
            v = _ref_from_dataset(dataset_name, runs, v.split(":", 1)[1])
        out.append({**ref, "value": v})
    return out


def _ref_from_dataset(dataset_name: str, runs: list[dict], system: str) -> float | None:
    """Average a reference score shipped inside the dataset, over the *same* samples as the runs."""
    if not runs:
        return None
    try:
        from gre.core.registry import get_dataset
        ds = get_dataset(dataset_name)()
        ds.download()
        ids = {json.loads(l)["sample_id"] for l in (Path(runs[0]["_dir"]) / "samples.jsonl").read_text().splitlines()}
        vals = [s.meta["ref"][system] for s in ds.load(ds.default_split)
                if s.id in ids and s.meta.get("ref", {}).get(system) is not None]
        return sum(vals) / len(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def _error_samples(run: dict, metric: str, k: int) -> list[dict]:
    path = Path(run["_dir"]) / "samples.jsonl"
    if not path.exists() or k <= 0:
        return []
    from gre.core.registry import load_builtins_and_plugins
    load_builtins_and_plugins()
    series = per_sample_series(path, metric)
    if not series:
        return []
    worst = sorted(series.items(), key=lambda kv: kv[1])[:k]
    want = {sid for sid, _ in worst}
    recs = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["sample_id"] in want:
            recs[r["sample_id"]] = r
    gold = _gold_lookup(run, want)
    out = []
    for sid, val in worst:
        r = recs.get(sid, {})
        item = {"sample_id": sid, "value": val, "error": r.get("error")}
        g = gold.get(sid)
        if g is not None and "extract" in r.get("preds", {}):
            pred = r["preds"]["extract"]
            pk = {(norm(t["subject"]["text"]), norm_rel(t["predicate"]), norm(t["object"]["text"])) for t in pred}
            gk = []
            for t in g.gold_triples or []:
                names_s = {norm(n) for n in t.subject.names()}
                names_o = {norm(n) for n in t.object.names()}
                hit = any(s in names_s and p == norm_rel(t.predicate) and o in names_o for s, p, o in pk)
                gk.append((t.subject.text, t.predicate, t.object.text, hit))
            item["text"] = (g.text or "")[:700]
            item["missed"] = [x[:3] for x in gk if not x[3]][:12]
            gold_keys = {(norm(a), norm_rel(b), norm(c)) for a, b, c, _ in gk}
            item["spurious"] = [(t["subject"]["text"], t["predicate"], t["object"]["text"]) for t in pred
                                if (norm(t["subject"]["text"]), norm_rel(t["predicate"]), norm(t["object"]["text"]))
                                not in gold_keys][:12]
        elif g is not None and "resolve" in r.get("preds", {}) and g.gold_pairs:
            assign = {}
            for i, c in enumerate(r["preds"]["resolve"]):
                for m in c["mention_ids"]:
                    assign[m] = i
            by = {m.mention_id: m.text for m in g.mentions}
            fp = [(by[a], by[b]) for a, b, lab in g.gold_pairs if lab == 0 and assign.get(a) == assign.get(b)]
            fn = [(by[a], by[b]) for a, b, lab in g.gold_pairs if lab == 1 and assign.get(a) != assign.get(b)]
            item["false_merges"] = [(a[:160], b[:160]) for a, b in fp[:5]]
            item["missed_merges"] = [(a[:160], b[:160]) for a, b in fn[:5]]
        out.append(item)
    return out


_GOLD_CACHE: dict[str, dict] = {}


def _gold_lookup(run: dict, ids: set[str]) -> dict:
    spec = run["dataset"]
    if spec not in _GOLD_CACHE:
        try:
            from gre.core.registry import get_dataset
            from gre.datasets.base import parse_dataset_spec
            name, split, _, _ = parse_dataset_spec(spec)
            ds = get_dataset(name)()
            ds.download()
            _GOLD_CACHE[spec] = {s.id: s for s in ds.load(split or ds.default_split)}
        except Exception:  # noqa: BLE001
            _GOLD_CACHE[spec] = {}
    return {i: _GOLD_CACHE[spec][i] for i in ids if i in _GOLD_CACHE[spec]}


def _group_seeds(runs: list[dict]) -> list[dict]:
    """Merge seeds of the same (label, dataset): mean of values, widest CI."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["label"], r["dataset"])].append(r)
    out = []
    for (label, ds), rs in groups.items():
        base = dict(rs[0])
        if len(rs) > 1:
            metrics = {}
            for name in set().union(*(r["metrics"] for r in rs)):
                ms = [r["metrics"][name] for r in rs if name in r["metrics"] and r["metrics"][name]["value"] is not None]
                if not ms:
                    continue
                metrics[name] = {**ms[0], "value": sum(m["value"] for m in ms) / len(ms),
                                 "ci_low": min(m.get("ci_low") or m["value"] for m in ms),
                                 "ci_high": max(m.get("ci_high") or m["value"] for m in ms)}
            base["metrics"] = metrics
            base["seeds"] = [r["seed"] for r in rs]
        else:
            base["seeds"] = [rs[0]["seed"]]
        base["_runs"] = rs
        out.append(base)
    return out


def _pareto(points: list[tuple[float, float, str]]) -> set[str]:
    """Points (cost, quality, label): keep those not dominated (lower cost & higher quality)."""
    front, best = set(), -1.0
    for c, q, lab in sorted(points, key=lambda p: (p[0], -p[1])):
        if q > best:
            front.add(lab)
            best = q
    return front


def _scatter_svg(rows: list[dict], metric: str) -> str | None:
    pts = [(r["cost_per_sample"], r["metrics"][metric]["value"], r["label"]) for r in rows
           if metric in r["metrics"] and r["metrics"][metric]["value"] is not None]
    if len(pts) < 2:
        return None
    front = _pareto(pts)
    W, H, pad = 560, 260, 44
    cmax = max(p[0] for p in pts) or 1e-9
    qs = [p[1] for p in pts]
    qmin, qmax = min(qs), max(qs)
    if qmax - qmin < 1e-9:
        qmin, qmax = qmin - 0.05, qmax + 0.05
    sx = lambda c: pad + (W - 2 * pad) * (c / cmax)  # noqa: E731
    sy = lambda q: H - pad - (H - 2 * pad) * ((q - qmin) / (qmax - qmin))  # noqa: E731
    fr = sorted([p for p in pts if p[2] in front])
    els = [f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" class="axis"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" class="axis"/>',
           f'<text x="{W / 2}" y="{H - 8}" class="lbl" text-anchor="middle">cost per sample (USD)  0 – {cmax:.2g}</text>',
           f'<text x="12" y="{H / 2}" class="lbl" transform="rotate(-90 12 {H / 2})" text-anchor="middle">'
           f'{metric} ({100 * qmin:.0f}–{100 * qmax:.0f})</text>']
    if len(fr) > 1:
        els.append('<polyline class="front" points="' + " ".join(f"{sx(c):.1f},{sy(q):.1f}" for c, q, _ in fr) + '"/>')
    for c, q, lab in pts:
        cls = "pt on" if lab in front else "pt"
        els.append(f'<circle cx="{sx(c):.1f}" cy="{sy(q):.1f}" r="5" class="{cls}"><title>{lab}: '
                   f'{100 * q:.1f} @ ${c:.4f}</title></circle>')
        els.append(f'<text x="{sx(c) + 7:.1f}" y="{sy(q) - 6:.1f}" class="ptl">{lab[:38]}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="quality vs cost">{"".join(els)}</svg>'


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in ("gre", "litellm", "pydantic", "rapidfuzz"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        out["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return out


def build_context(exp_dir: str | Path, baseline: str | None = None, error_k: int = 5) -> dict:
    exp_dir = Path(exp_dir)
    runs = load_runs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"no runs found under {exp_dir}")
    cfg = json.loads((exp_dir / "config.json").read_text()) if (exp_dir / "config.json").exists() else {}
    if cfg.get("local_datasets"):
        from gre.datasets.local import register_local_datasets
        register_local_datasets(cfg["local_datasets"])
    baseline = baseline or (cfg.get("report") or {}).get("baseline")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in _group_seeds(runs):
        by_ds[r["dataset"]].append(r)
    datasets = []
    for ds, rows in by_ds.items():
        for r in rows:
            u = r["usage"]
            r["cost_per_sample"] = u.get("cost_per_sample_usd") or 0.0
            r["latency_per_sample"] = u.get("latency_per_sample_s") or 0.0
        all_metrics = sorted(set().union(*(r["metrics"] for r in rows)),
                             key=lambda m: (STAGE_ORDER.get(m.split(".")[0], 9), m))
        headline = [m for m in all_metrics if any(r["metrics"].get(m, {}).get("headline") for r in rows)]
        cols = list(dict.fromkeys(headline + [m for m in SECONDARY if m in all_metrics]))
        primary = headline[0] if headline else (cols[0] if cols else None)
        rows.sort(key=lambda r: -(r["metrics"].get(primary, {}).get("value") or -1) if primary else 0)
        # best per column
        best = {}
        for m in cols:
            vals = [(r["metrics"][m]["value"], r["label"]) for r in rows if m in r["metrics"]
                    and r["metrics"][m]["value"] is not None]
            if vals and len({round(v, 9) for v, _ in vals}) > 1:   # no "best" when all tie
                hib = next(r["metrics"][m] for r in rows if m in r["metrics"]).get("higher_is_better", True)
                best[m] = (max if hib else min)(vals)[1]
        # significance vs baseline on the primary metric (paired bootstrap over shared samples)
        base_row = next((r for r in rows if baseline and baseline in r["label"]), rows[-1] if len(rows) > 1 else None)
        sig = {}
        if primary and base_row is not None:
            bser = per_sample_series(Path(base_row["_dir"]) / "samples.jsonl", primary)
            for r in rows:
                if r is base_row:
                    continue
                ser = per_sample_series(Path(r["_dir"]) / "samples.jsonl", primary)
                common = [k for k in ser if k in bser and ser[k] is not None and bser[k] is not None]
                if len(common) >= 2:
                    sig[r["label"]] = paired_bootstrap([bser[k] for k in common], [ser[k] for k in common])
        # isolated vs cascaded
        modes = defaultdict(dict)
        for r in rows:
            modes[r["label"].rsplit(" · ", 1)[0]][r["mode"]] = r
        cascade = []
        for lab, mm in modes.items():
            if "isolated" in mm and "cascaded" in mm:
                for m in all_metrics:
                    a = mm["isolated"]["metrics"].get(m, {}).get("value")
                    b = mm["cascaded"]["metrics"].get(m, {}).get("value")
                    if a is not None and b is not None and abs(b - a) > 1e-9 and not m.startswith(COUNT_METRICS):
                        cascade.append({"pipeline": lab, "metric": m, "isolated": a, "cascaded": b, "delta": b - a})
        dsname = rows[0].get("dataset_name", ds.split(":")[0])
        datasets.append({
            "name": ds, "dataset_name": dsname, "rows": rows, "cols": cols, "all_metrics": all_metrics,
            "primary": primary, "best": best, "sig": sig, "baseline": base_row["label"] if base_row else None,
            "sota": _sota_refs(dsname, rows), "cascade": cascade,
            "scatter": _scatter_svg(rows, primary) if primary else None,
            "errors": {r["label"]: _error_samples(r, primary, error_k) for r in rows} if primary else {},
        })
    return {"exp": exp_dir.name, "cfg": cfg, "datasets": datasets, "n_runs": len(runs),
            "generated": dt.datetime.now().isoformat(timespec="seconds"), "versions": _versions(),
            "fmt": _fmt, "total_cost": sum(r["usage"]["total"]["cost_usd"] for r in runs),
            "quick": any(r.get("quick") for r in runs)}


def build_report(exp_dir: str | Path, formats: list[str] | None = None, baseline: str | None = None,
                 out_dir: str | Path | None = None) -> list[Path]:
    exp_dir = Path(exp_dir)
    ctx = build_context(exp_dir, baseline)
    out_dir = Path(out_dir or exp_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    written = []
    formats = formats or (ctx["cfg"].get("report") or {}).get("formats") or ["html", "md", "json"]
    if "html" in formats:
        p = out_dir / "report.html"
        p.write_text(env.get_template("report.html.j2").render(**ctx))
        written.append(p)
    if "md" in formats:
        p = out_dir / "report.md"
        p.write_text(_markdown(ctx))
        written.append(p)
    if "json" in formats:
        p = out_dir / "report.json"
        slim = {k: v for k, v in ctx.items() if k not in ("fmt",)}
        for d in slim["datasets"]:
            d["rows"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in d["rows"]]
        p.write_text(json.dumps(slim, indent=2, default=str))
        written.append(p)
    return written


def _markdown(ctx: dict) -> str:
    f = ctx["fmt"]
    lines = [f"# GraphEvals report — {ctx['exp']}", "",
             f"Generated {ctx['generated']} · {ctx['n_runs']} runs · total spend ${ctx['total_cost']:.4f}"
             + (" · **quick mode (subsampled)**" if ctx["quick"] else ""), ""]
    for d in ctx["datasets"]:
        lines += [f"## {d['name']}", "", "| pipeline | " + " | ".join(d["cols"]) + " | $/sample | n |",
                  "|---|" + "---|" * (len(d["cols"]) + 2)]
        for r in d["rows"]:
            cells = []
            for m in d["cols"]:
                v = r["metrics"].get(m)
                s = f(v["value"], m) if v else "–"
                if v and v.get("ci_low") is not None and v["value"] is not None and not m.startswith(COUNT_METRICS):
                    s += f" ±{100 * (v['ci_high'] - v['ci_low']) / 2:.1f}"
                if d["best"].get(m) == r["label"]:
                    s = f"**{s}**"
                cells.append(s)
            lines.append(f"| {r['label']} | " + " | ".join(cells) + f" | {r['cost_per_sample']:.5f} | {r['n_scored']} |")
        for ref in d["sota"]:
            if ref.get("value") is not None:
                lines.append(f"| _SOTA ref: {ref['system']}_ ({ref['metric']}) | {f(ref['value'])} |" +
                             " |" * (len(d["cols"]) + 1))
        lines.append("")
    lines += ["Values are percentages; ± is half the 95% bootstrap CI. **Bold** = best in column.", ""]
    return "\n".join(lines)
