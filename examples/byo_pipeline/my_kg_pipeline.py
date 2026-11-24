"""Example: evaluating an EXISTING pipeline without rewriting it.

Pretend `MyKGBuilder` is your production code. Two thin wrappers expose it to gre:
  * decorator style  -> @gre.stage("extract", name="my_extractor")      (use `adapter: my_extractor`)
  * string style     -> `callable: "examples.byo_pipeline.my_kg_pipeline:resolve"` (no gre import needed)
Plus a custom metric registered with @gre.metric — it appears in every report automatically.
"""
from __future__ import annotations

import re

import gre
from gre.llm.structured import complete_json


# ---------------------------------------------------------------- your existing code (unchanged)
class MyKGBuilder:
    def __init__(self, llm_call):
        self.llm_call = llm_call            # any callable prompt -> dict

    def extract(self, text: str, relations: list[str]) -> list[dict]:
        rel_hint = f"Use only these relations: {', '.join(relations)}." if relations else ""
        out = self.llm_call(
            f"Extract facts as JSON {{\"facts\": [{{\"head\":..., \"rel\":..., \"tail\":...}}]}}. {rel_hint}\n\n{text}")
        return (out or {}).get("facts", [])


def normalise_name(s: str) -> str:
    return re.sub(r"\b(inc|corp|ltd|the)\b\.?", "", s.lower()).strip(" ,.")


# ---------------------------------------------------------------- thin gre wrappers
@gre.stage("extract", name="my_extractor", version="1")
def extract(chunks, schema, ctx):
    """ctx.llm is the model chosen by the experiment matrix — your code never hardcodes a provider."""
    builder = MyKGBuilder(lambda prompt: complete_json(ctx.llm, prompt))
    relations = [r.name for r in schema.relation_types] if schema else []
    facts = []
    for ch in chunks:
        facts += builder.extract(ch.text, relations)
    return facts                    # dicts with head/rel/tail -> coerced to Triple automatically


def resolve(mentions, candidate_pairs=None, strip_suffixes: bool = True):
    """Return {mention_id: cluster_key}; gre turns it into clusters."""
    key = normalise_name if strip_suffixes else str.lower
    return {m.mention_id: key(m.text) for m in mentions}


# ---------------------------------------------------------------- a custom metric
@gre.metric("extract", "predicate_diversity", higher_is_better=False,
            description="Distinct predicates / triples (lower = more consistent vocabulary).")
def predicate_diversity(pred, sample, ctx):
    return len({t.predicate.lower() for t in pred}) / len(pred) if pred else None
