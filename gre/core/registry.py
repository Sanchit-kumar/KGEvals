"""Plugin registry: stages (adapters / user callables), metrics and datasets.

Three ways to plug something in:
  1. decorator      @stage("extract", name="mine")   @metric("extract", name="x")   @dataset("my_ds")
  2. YAML string    extract: { callable: "my_pkg.module:function" }
  3. entry point    [project.entry-points."gre.plugins"] my = "my_pkg.gre_plugin"
"""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from gre.core.contracts import STAGES


@dataclass
class StageSpec:
    stage: str
    name: str
    fn: Callable
    version: str = "0"
    description: str = ""


@dataclass
class MetricSpec:
    stage: str
    name: str
    fn: Callable
    aggregate: Literal["mean", "micro_prf"] = "mean"
    higher_is_better: bool = True
    headline: bool = False
    requires: tuple[str, ...] = ()       # Sample fields that must be non-empty
    description: str = ""
    needs_llm: bool = False


@dataclass
class Registry:
    stages: dict[tuple[str, str], StageSpec] = field(default_factory=dict)
    metrics: dict[str, MetricSpec] = field(default_factory=dict)
    datasets: dict[str, type] = field(default_factory=dict)


REGISTRY = Registry()


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def stage(stage_name: str, name: str | None = None, version: str = "0", description: str = ""):
    """Register a function or class as an adapter for a pipeline stage.

    The function's parameters are bound *by name* from what the runner has available:
    ``text, chunks, schema, mentions, candidate_pairs, triples, clusters, sample, ctx, params``
    plus any key from the YAML stage params. Return anything `gre.coerce` understands.
    """
    _check_stage(stage_name)

    def deco(obj):
        n = name or getattr(obj, "__name__", str(obj))
        REGISTRY.stages[(stage_name, n)] = StageSpec(stage_name, n, obj, version,
                                                     description or (inspect.getdoc(obj) or "").split("\n")[0])
        return obj
    return deco


def metric(stage_name: str, name: str | None = None, *, aggregate: str = "mean",
           higher_is_better: bool = True, headline: bool = False,
           requires: tuple[str, ...] | list[str] = (), description: str = "", needs_llm: bool = False):
    """Register a metric.

    Signature: ``fn(pred, sample, ctx) -> float | dict[str, float] | None``.
    * aggregate="mean": per-sample float (or dict of floats) averaged with bootstrap CI.
    * aggregate="micro_prf": return {"tp","fp","fn"}; precision/recall/F1 are micro-averaged.
    Return None to skip a sample (e.g. no gold available).
    """
    def deco(fn):
        n = name or fn.__name__
        full = f"{stage_name}.{n}"
        REGISTRY.metrics[full] = MetricSpec(stage_name, n, fn, aggregate, higher_is_better, headline,
                                            tuple(requires), description or (inspect.getdoc(fn) or "").split("\n")[0],
                                            needs_llm)
        return fn
    return deco


def dataset(name: str):
    """Register a Dataset subclass under ``name``."""
    def deco(cls):
        cls.name = name
        REGISTRY.datasets[name] = cls
        return cls
    return deco


def import_object(path: str) -> Any:
    """Resolve ``"pkg.module:attr"`` (or ``"pkg.module.attr"``)."""
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        mod, _, attr = path.rpartition(".")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


_LOADED = False


def load_builtins_and_plugins(extra_modules: list[str] | None = None) -> None:
    """Import built-in adapters/metrics/datasets, entry-point plugins, and config `plugins:`."""
    global _LOADED
    if not _LOADED:
        import gre.adapters  # noqa: F401
        import gre.datasets  # noqa: F401
        import gre.metrics  # noqa: F401
        try:
            eps = importlib.metadata.entry_points(group="gre.plugins")
        except TypeError:  # pragma: no cover
            eps = importlib.metadata.entry_points().get("gre.plugins", [])
        for ep in eps:
            ep.load()
        _LOADED = True
    for m in extra_modules or []:
        importlib.import_module(m)


def get_stage(stage_name: str, name: str) -> StageSpec:
    load_builtins_and_plugins()
    try:
        return REGISTRY.stages[(stage_name, name)]
    except KeyError:
        avail = sorted(n for s, n in REGISTRY.stages if s == stage_name)
        raise KeyError(f"no adapter {name!r} for stage {stage_name!r}; available: {avail}") from None


def get_dataset(name: str):
    load_builtins_and_plugins()
    try:
        return REGISTRY.datasets[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(REGISTRY.datasets)}") from None
