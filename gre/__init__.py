"""GraphEvals (`gre`): benchmark knowledge-graph construction pipelines stage-by-stage against SOTA datasets.

Public API:
    from gre import stage, metric, dataset            # plug-in decorators
    from gre import Chunk, Triple, Mention, EntityCluster, KnowledgeGraph, GraphSchema, Sample
    from gre import evaluate                          # programmatic run
"""
from gre.core.contracts import (Chunk, EntityCluster, EntityType, GraphSchema, KnowledgeGraph, Mention,  # noqa: F401
                                RelationType, Sample, Triple)
from gre.core.registry import dataset, metric, stage  # noqa: F401
from gre.llm.gateway import register_mock  # noqa: F401

__version__ = "0.1.0"


def evaluate(config, *, overrides: list[str] | None = None, quick: bool = False, limit: int | None = None,
             stages: list[str] | None = None, report: bool = True, out_dir: str | None = None):
    """Run an experiment from a YAML path or dict. Returns (summaries, report_paths)."""
    from pathlib import Path

    from gre.core.config import ExperimentConfig, apply_overrides, load_config
    from gre.core.runner import Runner
    if isinstance(config, dict):
        cfg = ExperimentConfig(**apply_overrides(config, overrides))
    else:
        cfg = load_config(Path(config), overrides)
    runner = Runner(cfg, out_dir=out_dir, quick=quick, limit=limit, stages=stages)
    try:
        summaries = runner.run_all()
    finally:
        runner.close()
    paths = []
    if report and summaries:
        from gre.report.builder import build_report
        paths = build_report(runner.out)
    return summaries, paths
