"""Aggregation with bootstrap confidence intervals + paired significance tests."""
from __future__ import annotations

import numpy as np


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate_mean(values: list[float], n_boot: int = 1000, seed: int = 0) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    if v.size > 1 and n_boot:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        means = v[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
    else:
        lo = hi = v.mean()
    return {"value": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(v.size)}


def aggregate_prf(counts: list[dict], n_boot: int = 1000, seed: int = 0) -> dict[str, dict]:
    """Micro P/R/F1 over samples; CI by resampling samples (keeps within-sample correlation)."""
    c = np.asarray([[d["tp"], d["fp"], d["fn"]] for d in counts if d], dtype=float)
    if c.size == 0:
        return {}
    tp, fp, fn = c.sum(axis=0)
    p, r, f = _prf(tp, fp, fn)
    out = {"p": {"value": p}, "r": {"value": r}, "f1": {"value": f}}
    rng = np.random.default_rng(seed)
    if len(c) > 1 and n_boot:
        idx = rng.integers(0, len(c), size=(n_boot, len(c)))
        s = c[idx].sum(axis=1)  # (n_boot, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            bp = np.where(s[:, 0] + s[:, 1] > 0, s[:, 0] / (s[:, 0] + s[:, 1]), 0.0)
            br = np.where(s[:, 0] + s[:, 2] > 0, s[:, 0] / (s[:, 0] + s[:, 2]), 0.0)
            bf = np.where(bp + br > 0, 2 * bp * br / (bp + br), 0.0)
        for k, arr in (("p", bp), ("r", br), ("f1", bf)):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[k].update(ci_low=float(lo), ci_high=float(hi))
    else:
        for k in out:
            out[k].update(ci_low=out[k]["value"], ci_high=out[k]["value"])
    for k in out:
        out[k]["n"] = int(len(c))
    return out


def per_sample_f1(d: dict | None) -> float | None:
    if not d:
        return None
    return _prf(d["tp"], d["fp"], d["fn"])[2]


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """H0: mean(a) == mean(b) on paired samples. Returns delta (b - a), CI and two-sided p-value."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {"delta": float(np.mean(y - x)) if x.size else None, "p_value": None, "n": int(x.size)}
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(p, 1.0)),
            "n": int(d.size)}
