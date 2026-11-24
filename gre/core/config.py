"""Experiment config: YAML -> pydantic, `${matrix.x}` interpolation, matrix expansion, --set overrides."""
from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from gre.core.contracts import STAGES

_VAR = re.compile(r"\$\{matrix\.([A-Za-z0-9_]+)\}")
RESERVED_STAGE_KEYS = {"adapter", "callable", "provider", "embedding", "field_map", "version", "cache"}


class ScoringCfg(BaseModel):
    soft_match_threshold: float = 0.8
    soft_method: str = "fuzzy"          # fuzzy | embedding
    embedding: str = "default"           # alias in `embeddings:` when soft_method=embedding
    include: list[str] = []              # metric names / globs; empty = all applicable
    exclude: list[str] = []
    n_bootstrap: int = 1000
    error_samples: int = 5


class RuntimeCfg(BaseModel):
    workers: int = 8
    cache: bool = True
    cache_path: str = ".gre_cache/cache.sqlite"
    quick_limit: int = 20
    fail_fast: bool = False


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    datasets: list[str] = []
    suite: str | None = None
    seeds: list[int] = [0]
    mode: list[str] = ["isolated"]
    stages: list[str] | None = None      # restrict evaluated stages
    providers: dict[str, dict] = {}
    embeddings: dict[str, dict] = {}
    judge: dict = {}
    pipelines: dict[str, dict[str, Any]]
    matrix: dict[str, list[Any]] = {}
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    report: dict = {"formats": ["html", "md", "json"]}
    budget: dict = {"max_cost_usd": None}
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    plugins: list[str] = []              # python modules to import (register adapters/metrics)
    local_datasets: dict[str, dict] = {}  # your own gold JSONL files: {name: {path, stages, schema}}
    output_dir: str = "runs"

    @field_validator("seeds", "mode", mode="before")
    @classmethod
    def _listify(cls, v):
        return v if isinstance(v, list) else [v]

    @field_validator("mode")
    @classmethod
    def _modes(cls, v):
        bad = set(v) - {"isolated", "cascaded"}
        if bad:
            raise ValueError(f"mode must be isolated/cascaded, got {bad}")
        return v

    @field_validator("pipelines")
    @classmethod
    def _pipes(cls, v):
        for pname, spec in v.items():
            for st, s in spec.items():
                if st in ("framework", "description"):
                    continue
                if st not in STAGES:
                    raise ValueError(f"pipeline {pname!r}: unknown stage {st!r} (expected {STAGES})")
                if not isinstance(s, dict) or not ({"adapter", "callable"} & set(s)):
                    raise ValueError(f"pipeline {pname!r} stage {st!r} needs `adapter:` or `callable:`")
        return v


def _set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    """`--set matrix.llm=[claude,local] --set scoring.soft_match_threshold=0.85`"""
    raw = copy.deepcopy(raw)
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"override must be key=value, got {o!r}")
        k, v = o.split("=", 1)
        _set_path(raw, k.strip(), yaml.safe_load(v))
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw = apply_overrides(raw, overrides)
    raw.setdefault("name", Path(path).stem)
    return ExperimentConfig(**raw)


def _vars_in(obj: Any) -> set[str]:
    if isinstance(obj, str):
        return set(_VAR.findall(obj))
    if isinstance(obj, dict):
        return set().union(*(_vars_in(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_vars_in(v) for v in obj)) if obj else set()
    return set()


def _interp(obj: Any, values: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        m = _VAR.fullmatch(obj)
        if m:  # keep type for whole-value substitutions
            return values[m.group(1)]
        return _VAR.sub(lambda mm: str(values[mm.group(1)]), obj)
    if isinstance(obj, dict):
        return {k: _interp(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, values) for v in obj]
    return obj


class PipelineVariant(BaseModel):
    pipeline: str
    matrix: dict[str, Any]
    stages: dict[str, dict]

    @property
    def label(self) -> str:
        if not self.matrix:
            return self.pipeline
        return self.pipeline + "[" + ",".join(f"{k}={v}" for k, v in self.matrix.items()) + "]"


def expand_pipelines(cfg: ExperimentConfig) -> list[PipelineVariant]:
    """Expand each pipeline only over the matrix variables it actually references."""
    out = []
    for pname, spec in cfg.pipelines.items():
        used = sorted(_vars_in(spec))
        missing = [u for u in used if u not in cfg.matrix]
        if missing:
            raise ValueError(f"pipeline {pname!r} references undefined matrix vars {missing}")
        combos = itertools.product(*(cfg.matrix[u] for u in used)) if used else [()]
        for combo in combos:
            vals = dict(zip(used, combo))
            stages = {k: v for k, v in _interp(spec, vals).items() if k in STAGES}
            out.append(PipelineVariant(pipeline=pname, matrix=vals, stages=stages))
    return out


def split_stage_spec(spec: dict) -> tuple[dict, dict]:
    """(reserved keys, adapter params)."""
    reserved = {k: v for k, v in spec.items() if k in RESERVED_STAGE_KEYS}
    params = {k: v for k, v in spec.items() if k not in RESERVED_STAGE_KEYS}
    return reserved, params
