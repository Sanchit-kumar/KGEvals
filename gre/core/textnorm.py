"""Text normalisation shared by adapters and metrics."""
from __future__ import annotations

import re
import unicodedata

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WS = re.compile(r"\s+")
_EDGE_PUNCT = " \t\n\"'`.,;:()[]{}"


def norm(s: object) -> str:
    """Lowercase, unify underscores/whitespace, strip quotes and edge punctuation."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = t.replace("_", " ")
    t = _WS.sub(" ", t).strip(_EDGE_PUNCT).lower()
    return t


def norm_rel(s: object) -> str:
    """Relation names: split camelCase, then norm. `foundationPlace` == `foundation place`."""
    if s is None:
        return ""
    return norm(_CAMEL.sub(" ", str(s)))
