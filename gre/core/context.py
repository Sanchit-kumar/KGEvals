"""`ctx` object handed to every adapter and metric."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from gre.core.contracts import Usage
from gre.llm.gateway import LLMGateway, NullGateway


@dataclass
class Context:
    stage: str
    params: dict[str, Any] = field(default_factory=dict)
    llm: LLMGateway = field(default_factory=NullGateway)
    embedder: LLMGateway | None = None
    judge: LLMGateway | None = None
    usage: Usage = field(default_factory=Usage)
    seed: int = 0
    run_id: str = ""
    sample_id: str = ""
    scoring: Any = None                       # ScoringCfg (metrics only)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("gre"))
    extras: dict[str, Any] = field(default_factory=dict)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is None:
            raise RuntimeError("no embedding model configured (stage `embedding:` or scoring.embedding)")
        return self.embedder.embed(texts)
