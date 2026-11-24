"""Reference extractor: schema-guided LLM triple extraction with JSON output (any LiteLLM provider)."""
from __future__ import annotations

from gre.core.contracts import Chunk, GraphSchema, Mention, Triple
from gre.core.registry import stage
from gre.llm.structured import complete_json

PROMPT_V1 = """Extract a knowledge graph from the text as (subject, predicate, object) triples.
{schema_block}
Rules:
- Use entity names exactly as written in the text (resolve pronouns to the named entity).
- {pred_rule}
- Only extract facts stated or directly implied by the text. Do not add outside knowledge.
- Return JSON: {{"triples": [{{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}}]}}
{examples}
Text:
\"\"\"{text}\"\"\"
"""


def _schema_block(schema: GraphSchema | None, max_rel: int = 200) -> str:
    if not schema or not schema.relation_types:
        return "No fixed ontology: use short, lowercase, verb-phrase predicates."
    concepts = ", ".join(e.name for e in schema.entity_types) or "any"
    rels = "\n".join(f"- {r.name}({r.domain}, {r.range})" for r in schema.relation_types[:max_rel])
    return f"Ontology concepts: {concepts}\nOntology relations (name(domain, range)):\n{rels}"


@stage("extract", "llm_structured", version="1",
       description="Schema-guided LLM extraction with JSON output, repair and retry.")
def llm_structured(chunks: list[Chunk], schema: GraphSchema | None, ctx, examples: list | None = None,
                   prompt: str | None = None, json_mode: bool = True) -> list[Triple]:
    closed = bool(schema and schema.relation_types)
    ex = ""
    if examples:
        ex = "Examples:\n" + "\n".join(f"Text: {e['text']}\nOutput: {e['output']}" for e in examples) + "\n"
    out: list[Triple] = []
    for ch in chunks:
        p = (prompt or PROMPT_V1).format(
            schema_block=_schema_block(schema),
            pred_rule="Predicates MUST be one of the ontology relation names." if closed
            else "Keep predicates concise and consistent.",
            examples=ex, text=ch.text)
        data = complete_json(ctx.llm, p, json_mode=json_mode)
        items = data.get("triples", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for t in items:
            if not isinstance(t, dict) or not all(t.get(k) for k in ("subject", "predicate", "object")):
                continue
            out.append(Triple(subject=Mention(text=str(t["subject"]), type=str(t.get("subject_type") or ""),
                                              chunk_id=ch.chunk_id),
                              predicate=str(t["predicate"]),
                              object=Mention(text=str(t["object"]), type=str(t.get("object_type") or ""),
                                             chunk_id=ch.chunk_id)))
    return out
