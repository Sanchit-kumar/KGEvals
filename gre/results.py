"""Aggregation of per-sample results and the queryable results store (DuckDB + CSV)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gre.core.contracts import Usage
from gre.metrics.stats import aggregate_mean, aggregate_prf, per_sample_f1


def _flatten(prefix: str, v: Any) -> dict[str, float | None]:
    if isinstance(v, dict):
        return {f"{prefix}.{k}": (float(x) if isinstance(x, (int, float)) else None) for k, x in v.items()}
    return {prefix: float(v) if isinstance(v, (int, float)) else None}


def aggregate_run(results, metric_specs: dict, n_boot: int = 1000) -> dict:
    """-> {"metrics": {name: {value, ci_low, ci_high, n, higher_is_better, headline}}, "usage": {...}}"""
    metrics: dict[str, dict] = {}
    for full, spec in metric_specs.items():
        vals = [r.metrics.get(full) for r in results if full in r.metrics]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        base = {"higher_is_better": spec.higher_is_better, "headline": False, "stage": spec.stage}
        if spec.aggregate == "micro_prf":
            for k, agg in aggregate_prf(vals, n_boot).items():
                metrics[f"{full}.{k}"] = {**agg, **base, "headline": spec.headline and k == "f1"}
        else:
            per_key: dict[str, list[float]] = {}
            for v in vals:
                for k, x in _flatten(full, v).items():
                    if x is not None:
                        per_key.setdefault(k, []).append(x)
            for k, xs in per_key.items():
                is_head = spec.headline and (k == full or k == f"{full}.f1")
                metrics[k] = {**aggregate_mean(xs, n_boot), **base, "headline": is_head}
    # usage
    per_stage: dict[str, Usage] = {}
    total, judge = Usage(), Usage()
    for r in results:
        for st, u in r.usage.items():
            per_stage.setdefault(st, Usage()).add(Usage(**u))
            total.add(Usage(**u))
        if "_judge_usage" in r.metrics:
            judge.add(Usage(**r.metrics["_judge_usage"]))
    n = max(1, len(results))
    jv = total.json_valid + total.json_invalid
    usage = {"total": total.model_dump(), "by_stage": {k: v.model_dump() for k, v in per_stage.items()},
             "judge": judge.model_dump(),
             "cost_per_sample_usd": total.cost_usd / n, "latency_per_sample_s": total.latency_s / n,
             "json_valid_rate": (total.json_valid / jv) if jv else None,
             "error_rate": sum(1 for r in results if r.error) / n,
             "errors_sample": [r.error for r in results if r.error][:5]}
    return {"metrics": metrics, "usage": usage}


def per_sample_series(samples_path: Path, metric: str) -> dict[str, float]:
    """Per-sample values of a metric (micro_prf metrics -> per-sample F1) for paired tests."""
    base, _, sub = metric.rpartition(".")
    out = {}
    for line in samples_path.read_text().splitlines():
        r = json.loads(line)
        m = r["metrics"]
        if metric in m and isinstance(m[metric], (int, float)):
            out[r["sample_id"]] = float(m[metric])
        elif base in m and isinstance(m[base], dict):
            v = m[base]
            if {"tp", "fp", "fn"} <= set(v) and sub == "f1":
                out[r["sample_id"]] = per_sample_f1(v)
            elif sub in v and isinstance(v[sub], (int, float)):
                out[r["sample_id"]] = float(v[sub])
    return out


def load_runs(exp_dir: str | Path) -> list[dict]:
    exp_dir = Path(exp_dir)
    runs = []
    for p in sorted(exp_dir.glob("*/scores.json")):
        d = json.loads(p.read_text())
        d["_dir"] = str(p.parent)
        runs.append(d)
    return runs


def rows(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        common = {"exp": r["exp"], "run_id": r["run_id"], "label": r["label"], "pipeline": r["pipeline"],
                  "matrix": json.dumps(r["matrix"]), "dataset": r["dataset"], "mode": r["mode"], "seed": r["seed"]}
        for name, m in r["metrics"].items():
            out.append({**common, "stage": m.get("stage", name.split(".")[0]), "metric": name, "value": m["value"],
                        "ci_low": m.get("ci_low"), "ci_high": m.get("ci_high"), "n": m.get("n")})
        u = r["usage"]
        for k in ("cost_usd", "tokens_in", "tokens_out", "latency_s", "n_calls", "cache_hits"):
            out.append({**common, "stage": "usage", "metric": f"usage.{k}", "value": u["total"][k],
                        "ci_low": None, "ci_high": None, "n": r["n_scored"]})
        for k in ("json_valid_rate", "error_rate", "cost_per_sample_usd"):
            out.append({**common, "stage": "usage", "metric": f"usage.{k}", "value": u.get(k),
                        "ci_low": None, "ci_high": None, "n": r["n_scored"]})
    return out


def write_results_db(exp_dir: str | Path) -> Path:
    """(Re)build runs/<exp>/results.duckdb + results.csv from all scores.json."""
    import csv

    exp_dir = Path(exp_dir)
    data = rows(load_runs(exp_dir))
    csv_path = exp_dir / "results.csv"
    cols = ["exp", "run_id", "label", "pipeline", "matrix", "dataset", "mode", "seed", "stage", "metric",
            "value", "ci_low", "ci_high", "n"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    try:
        import duckdb
        db = exp_dir / "results.duckdb"
        con = duckdb.connect(str(db))
        con.execute("DROP TABLE IF EXISTS results")
        con.execute(f"CREATE TABLE results AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header=true)")
        con.close()
    except Exception:  # noqa: BLE001  duckdb optional at runtime
        pass
    return csv_path
