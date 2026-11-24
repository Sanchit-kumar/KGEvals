# GraphRAG Ingestion Evaluation Platform — Build Spec

> **Purpose:** a configurable platform to benchmark GraphRAG **ingestion** pipelines (parsing → schema/ontology → extraction → entity resolution → graph), where you can swap **frameworks, models and LLM providers** per stage, run against public + in-house gold datasets, and get a comparable **report**.
> Phase 2 (retrieval / end-to-end QA) plugs into the same design later.
>
> Research snapshot: **Nov 2026**. Items marked ⚠️ need verification (license / API) before relying on them.

---

## 0. How to use this file

Paste into your repo root (e.g. `docs/SPEC.md` or `tools.md`) and ask your coding assistant in VS Code:

> "Read `tools.md`. Scaffold the repo per §7, implement Milestone 1 from §10, with tests."

Work milestone by milestone. Each milestone ends in something runnable.

---

## 1. Key design decisions

1. **Stage contracts, not frameworks.** Every stage reads/writes a serialized, versioned artifact (§3). Frameworks are *adapters* behind those contracts. This is what makes stages swappable and individually scoreable.
2. **One LLM gateway.** All LLM/embedding calls go through a single client (recommended: **LiteLLM**) so switching provider = changing a config string (`openai/gpt-…`, `anthropic/claude-…`, `gemini/…`, `ollama/…`, `vllm` / OpenAI-compatible, Azure, Bedrock). Gateway also records tokens, cost, latency, and caches responses.
3. **Config-driven experiments.** YAML defines a *matrix* (datasets × pipelines × models × seeds). The runner expands it into runs.
4. **Two evaluation modes per stage:**
   - **Isolated** — stage fed *gold* input from the previous stage (measures the stage alone).
   - **Cascaded** — stage fed the *predicted* output of the previous stage.
   - `cascaded − isolated` = error propagated from upstream (**oracle-substitution ablation**). This is the platform's most valuable diagnostic.
5. **Strict + soft scoring always reported together.** Strict exact-match and similarity-tolerant (embedding / fuzzy, threshold calibrated, default 0.8) scores can flip conclusions about LLM extraction.
6. **LLM-as-judge is a metric with its own reliability.** Any judge metric must log judge model + prompt version and support a human-agreement check (Cohen's κ / Gwet's AC1) on a sample.
7. **Everything cached and reproducible.** Content-hash every artifact; rerunning a config hits cache unless inputs/config changed.

---

## 2. Pipeline stages and pluggable tools

| Stage | Contract out | Adapters to implement (priority order) |
|---|---|---|
| **S1 Parse** | `ParsedDoc` | Docling (MIT), MinerU (⚠️ Apache + extra conditions), Marker (⚠️ weights licensing for >$5M revenue), PaddleOCR-VL, olmOCR, Unstructured, VLM-as-parser (any vision LLM via gateway), hosted: LlamaParse / Azure DI / Textract / Mistral OCR |
| **S2 Chunk** | `Chunk[]` | fixed-token, paragraph-group, layout/section-aware, semantic |
| **S3 Schema / Ontology** | `GraphSchema` (+ optional OWL) | user-provided (YAML/OWL), LLM schema induction (neo4j-graphrag `SchemaFromTextExtractor`, AutoSchemaKG-style), ontology matching for alignment (LogMap / DeepOnto / OntoAligner) |
| **S4 Extract** | `Triple[]` with source spans | LLM structured-output extractor (any provider), neo4j-graphrag `LLMEntityRelationExtractor`, KGGen, GLiNER2 (+GLiREL) for cheap NER/RE, LangChain `LLMGraphTransformer`, LlamaIndex `PropertyGraphIndex` extractors |
| **S5 Resolve (ER)** | `EntityCluster[]` + canonical graph | exact/normalized, fuzzy (RapidFuzz), embedding + threshold, LLM pairwise / batch matcher, Graphiti-style cascade (exact → MinHash-LSH → LLM), Splink/Zingg for attribute-rich records |
| **S6 Graph build/store** | `KnowledgeGraph` (JSON-LD/NetworkX export) + optional DB write | NetworkX (default, in-memory), Neo4j, FalkorDB, Memgraph; RDF via rdflib |
| **Whole-framework adapters** | export to the contracts above | Microsoft GraphRAG (read its per-phase parquet tables), neo4j-graphrag `SimpleKGPipeline`, LightRAG, Cognee, Graphiti |

Whole-framework adapters let you benchmark "framework as a black box" **and** score its intermediate outputs where it exposes them.

---

## 3. Stage contracts (Pydantic models)

```python
# core/contracts.py
from pydantic import BaseModel
from typing import Literal

class BBox(BaseModel):
    page: int; x0: float; y0: float; x1: float; y1: float

class Block(BaseModel):
    id: str
    type: Literal["title","heading","paragraph","table","formula","figure","list","caption","header","footer"]
    text: str                      # markdown for text, HTML for tables, LaTeX for formulas
    bbox: BBox | None = None
    order: int                     # reading order

class ParsedDoc(BaseModel):
    doc_id: str
    source_path: str
    blocks: list[Block]
    markdown: str                  # full linearized output
    meta: dict = {}

class Chunk(BaseModel):
    chunk_id: str; doc_id: str; text: str
    block_ids: list[str]; char_start: int; char_end: int

class EntityType(BaseModel):
    name: str; description: str = ""; properties: list[str] = []

class RelationType(BaseModel):
    name: str; domain: str; range: str; description: str = ""

class GraphSchema(BaseModel):
    entity_types: list[EntityType]
    relation_types: list[RelationType]
    owl_path: str | None = None    # optional formal ontology
    shacl_path: str | None = None  # optional shapes for validation

class Mention(BaseModel):
    mention_id: str; text: str; type: str
    chunk_id: str; char_start: int | None = None; char_end: int | None = None

class Triple(BaseModel):
    subject: Mention; predicate: str; object: Mention | str   # str = literal
    confidence: float | None = None
    evidence: str | None = None    # supporting text span

class EntityCluster(BaseModel):
    canonical_id: str; canonical_name: str; type: str
    mention_ids: list[str]

class KnowledgeGraph(BaseModel):
    nodes: list[dict]; edges: list[dict]
    clusters: list[EntityCluster]
    schema_: GraphSchema | None = None

class StageResult(BaseModel):
    stage: str; adapter: str; config_hash: str
    artifact_path: str
    usage: dict    # tokens_in, tokens_out, cost_usd, latency_s, n_calls
```

Every adapter: `run(input_artifact, config) -> output_artifact` and must emit `usage`.

---

## 4. Datasets

### 4.1 Public benchmarks (load via `datasets/` loaders → convert to contracts)

| Stage | Dataset | Size / content | Metrics | Link | License |
|---|---|---|---|---|---|
| Parse | **OmniDocBench** (v1.6/1.7) | 1,651 PDF pages, 10 doc types, 5 languages; layout, text, table, formula, reading-order gold | text NED, table TEDS, formula CDM, reading-order ED, Overall | github.com/opendatalab/OmniDocBench · HF `opendatalab/OmniDocBench` | ⚠️ research only, non-commercial |
| Parse | **olmOCR-Bench** | 1,403 PDFs, 7,010 pass/fail unit tests (tables, math, multi-column, headers/footers, old scans) | unit-test pass rate | HF `allenai/olmOCR-bench` | ODC-BY |
| Parse | **PureDocBench** (2026) | clean / degraded / real-world, harder than OmniDocBench (best ≈74/100) | as above | arXiv 2605.07492 | CC BY 4.0 (paper) ⚠️ check data |
| Parse | DP-Bench (Upstage) | 200 pages | NID, TEDS, TEDS-S | HF `upstage/dp-bench` | ⚠️ |
| Parse (layout/table) | DocLayNet, PubTabNet, FinTabNet | 80k pages / 568k tables / financial tables | layout mAP, TEDS | arXiv 2206.01062, 1911.10683, 2005.00589 | CDLA-Permissive (DocLayNet) ⚠️ others |
| Ontology learning | **LLMs4OL 2025** | Text2Onto, term typing, taxonomy discovery (OBI, MatOnto, SWEET, DOID, Schema.org, FoodOn …) | P/R/F1 | sites.google.com/view/llms4ol2025 · CodaLab 23065 | ⚠️ |
| Ontology matching | **OAEI 2025** tracks (Anatomy, Conference, Bio-ML, KG …) | reference alignments | P/R/F, coherence | oaei.ontologymatching.org | per track ⚠️ |
| Ontology-conformant extraction | **Text2KGBench** | Wikidata-TekGen 13,474 sent./10 ontologies; DBpedia-WebNLG 4,860 sent./19 ontologies | triple P/R/F1, ontology conformance, hallucination | Zenodo 7916716 · arXiv 2308.02357 | ⚠️ |
| Ontology-conformant extraction | **Text2KGBench-LettrIA** (2025) | re-annotated WebNLG part, 19 ontologies, >14k triples | macro F1 entity/attr/property/relation, domain/range adherence | CEUR Vol-4041 paper3 | ⚠️ |
| Doc-level RE | **Re-DocRED** | 4,053 Wikipedia docs, 96 relations | F1, Ign-F1 | arXiv 2205.12696 | ⚠️ (DocRED-derived) |
| RE / NER | SciERC, WebNLG, REBEL, CrossNER, CoNLL-2003 | classic sets | entity / triple F1 | HF `Babelscape/rebel-dataset` etc. | various ⚠️ |
| Open KG construction | **MINE-1 / MINE-2** (KGGen) | 100 articles × 15 gold facts; 20.4k WikiQA Qs | fact retention (LLM judge on 2-hop subgraph) | github.com/stair-lab/kg-gen · arXiv 2502.09956 | ⚠️ |
| Structured → KG | **BLINKG** (2026) | CSV/JSON/XML + ontology + gold mappings | P/R/F per mapping task | github.com/citiususc/blinkg | Apache-2.0 |
| Entity matching | **WDC Products**, Abt-Buy, Amazon-Google, Walmart-Amazon, DBLP-ACM, DBLP-Scholar | labeled record pairs | pairwise P/R/F1 | webdatacommons.org · Magellan/DeepMatcher repo | ⚠️ |
| Entity linking | ZESHEL, ZELDA, AIDA-CoNLL, KILT | mention → KB entity | accuracy, micro-F1 | arXiv 1906.07348, 2009.02252 | various ⚠️ |
| Coref / clustering | (use ER clusters from your gold, or SciERC coref) | | B³, CEAF-e, MUC, ARI | | |
| Graph quality (downstream) | **TDG-Bench** | MedMentions vs gold UMLS-NCI subgraph | GNN node-classification acc | github.com/OthmaneKabal/text_driven_kg_bench · HF space `othmanekabal/tdg-bench` | ⚠️ |
| Downstream probe | **GraphRAG-Bench** (Xiang et al., ICLR 2026) | NCCN medical + novels, 4 task levels; graph stats + QA | avg degree, clustering, tokens, Evidence Recall, ACC | github.com/GraphRAG-Bench/GraphRAG-Benchmark · graphrag-bench.github.io | ⚠️ |
| Downstream probe | HotpotQA, MuSiQue, 2WikiMultiHopQA, MultiHop-RAG | multi-hop QA | answer-entity coverage in KG, Recall@k, EM/F1 | HF hub | CC BY-SA etc. |

**Rule:** always run on public sets *and* on your in-house gold set. Public parsing leaderboards are saturated and often run by tool vendors (OmniDocBench ← MinerU team; olmOCR-Bench ← Ai2; ParseBench ← LlamaIndex).

### 4.2 In-house gold set ("GoldKG") — build this; it's the real asset

Pick 30–100 representative documents from your domain and annotate in layers (store as JSONL in contract format):

| Layer | What to annotate | Effort tip |
|---|---|---|
| L1 Parse | 100–300 pages: reading-order markdown, tables (HTML), plus olmOCR-style unit tests ("table cell X under header Y", "footer not in body") | unit tests are cheapest and most discriminative |
| L2 Schema | reference schema / OWL + SHACL shapes | one-time, domain expert |
| L3 Triples | gold triples with source spans, per chunk | pre-label with strongest LLM, humans correct (record correction rate) |
| L4 ER | gold entity clusters (mention → canonical id) incl. hard negatives (similar names, different entities) | |
| L5 Facts / CQs | 10–20 key facts per doc + competency questions with gold answers | bridges to Phase 2 retrieval eval |

Use **Label Studio** or **Argilla** for annotation; export to contract JSONL.

---

## 5. Metrics (per stage)

| Stage | Metrics | Implementation |
|---|---|---|
| Parse | normalized edit distance, TEDS (tables), CDM/formula match, reading-order ED, unit-test pass rate, pages/sec, $/1k pages | port OmniDocBench & olmOCR-Bench scorers; `apted`/`zss` for TEDS |
| Chunk | boundary F1 vs gold sections, % chunks splitting a table/sentence, size distribution | custom |
| Schema | type & relation P/R/F1 vs gold schema (soft label match), alignment F1, OWL consistency (HermiT/owlready2), SHACL conformance rate | pySHACL, owlready2, DeepOnto |
| Extract | **strict** triple P/R/F1; **soft** triple P/R/F1 (embedding sim ≥ τ, τ calibrated on a labeled sample); entity F1, relation F1; hallucinated type/relation rate; domain/range violations; evidence-span faithfulness; JSON validity rate; MINE-style fact retention | custom + LLM judge |
| ER | pairwise P/R/F1, **B³ P/R/F1**, CEAF-e, MUC, ARI; over-merge & under-merge rate; dedup ratio (entities after/before) | custom (small), `scikit-learn` ARI |
| Graph (intrinsic) | gold entity coverage, gold fact coverage, duplicate-node rate, #nodes/#edges, avg degree, clustering coeff., giant-component fraction, orphan rate, relation-type reuse, schema conformance % | NetworkX + pySHACL |
| Cost / perf (all) | tokens in/out, $ cost, wall-clock, #LLM calls, cache hit rate, failures/retries | gateway instrumentation |
| Judge reliability | judge vs human agreement (κ / AC1) on ≥100 samples; judge model + prompt version logged | custom |
| Stats | mean ± 95% CI over seeds (bootstrap), paired significance test between configs (paired bootstrap / McNemar) | `scipy` |

---

## 6. Experiment config (YAML)

```yaml
# experiments/exp_001.yaml
name: extraction-model-sweep
seed: [1, 2, 3]
datasets:
  - text2kgbench_lettria
  - redocred:test[:200]
  - goldkg_inhouse
mode: [isolated, cascaded]          # oracle-substitution ablation

providers:                           # resolved through LiteLLM
  gpt:      { model: "openai/gpt-4.1",                 temperature: 0 }
  claude:   { model: "anthropic/claude-sonnet-4-5",    temperature: 0 }
  gemini:   { model: "gemini/gemini-2.5-pro",          temperature: 0 }
  local:    { model: "ollama/qwen3:8b", api_base: "http://localhost:11434" }
  vllm:     { model: "openai/Qwen3-32B", api_base: "http://gpu-box:8000/v1" }
embeddings:
  default:  { model: "openai/text-embedding-3-large" }
judge:      { provider: claude, prompt_version: v1 }  # fixed across runs!

pipelines:
  modular_a:
    parse:   { adapter: docling }
    chunk:   { adapter: paragraph_group, max_tokens: 800 }
    schema:  { adapter: provided, path: schemas/domain.yaml }
    extract: { adapter: llm_structured, provider: "${matrix.llm}" }
    resolve: { adapter: embedding_threshold, threshold: 0.85, embedding: default }
    graph:   { adapter: networkx }
  neo4j_simplekg:
    framework: neo4j_graphrag          # whole-framework adapter
    llm: "${matrix.llm}"
  msgraphrag:
    framework: microsoft_graphrag
    llm: "${matrix.llm}"

matrix:
  llm: [gpt, claude, gemini, local]

scoring:
  soft_match_threshold: 0.8
  report: [html, markdown, json]
budget:
  max_cost_usd: 50                   # runner aborts beyond this
```

CLI:

```bash
gre run experiments/exp_001.yaml         # run matrix (resumable, cached)
gre run experiments/exp_001.yaml --dry-run   # show expanded runs + cost estimate
gre score runs/exp_001                   # (re)score stored artifacts
gre report runs/exp_001 --format html    # build report
gre compare runs/exp_001 runs/exp_002    # diff two experiments
gre ui                                   # Streamlit dashboard
```

---

## 7. Repo layout

```
graphrag-eval/
├── pyproject.toml                 # uv / poetry; python 3.11+
├── gre/
│   ├── core/
│   │   ├── contracts.py           # §3
│   │   ├── registry.py            # adapter registry (entry points / decorators)
│   │   ├── config.py              # YAML → pydantic, matrix expansion, ${} interpolation
│   │   ├── cache.py               # content-hash artifact cache (diskcache / sqlite)
│   │   └── runner.py              # executes runs, isolated vs cascaded, budget guard
│   ├── llm/
│   │   ├── gateway.py             # LiteLLM wrapper: retries, caching, usage/cost logging
│   │   └── structured.py          # JSON-schema / instructor-style structured outputs
│   ├── adapters/
│   │   ├── parse/   (docling.py, mineru.py, marker.py, vlm.py, llamaparse.py …)
│   │   ├── chunk/   (fixed.py, paragraph.py, layout.py, semantic.py)
│   │   ├── schema/  (provided.py, llm_induce.py, ontology_align.py)
│   │   ├── extract/ (llm_structured.py, gliner.py, kggen.py, neo4j_extractor.py)
│   │   ├── resolve/ (exact.py, fuzzy.py, embedding.py, llm_pairwise.py, cascade.py, splink.py)
│   │   ├── graph/   (networkx_store.py, neo4j_store.py, falkordb_store.py)
│   │   └── frameworks/ (ms_graphrag.py, neo4j_simplekg.py, lightrag.py, cognee.py, graphiti.py)
│   ├── datasets/
│   │   ├── base.py                # Dataset interface: inputs + gold per stage
│   │   ├── omnidocbench.py, olmocr_bench.py, text2kgbench.py, redocred.py,
│   │   ├── mine.py, wdc_products.py, magellan.py, graphrag_bench.py, goldkg.py
│   ├── metrics/
│   │   ├── parse.py (ned, teds, reading_order, unit_tests)
│   │   ├── schema.py, extract.py (strict/soft triple F1, hallucination, domain/range)
│   │   ├── er.py (pairwise, bcubed, ceaf, muc, ari, over/under-merge)
│   │   ├── graph.py (coverage, structural, shacl)
│   │   ├── judge.py (LLM judge + agreement)
│   │   └── stats.py (bootstrap CI, paired tests)
│   ├── report/
│   │   ├── builder.py             # aggregate → tables, charts
│   │   └── templates/report.html.j2
│   ├── ui/app.py                  # Streamlit: pick runs, compare, drill into errors
│   └── cli.py                     # Typer
├── experiments/                   # YAML configs
├── schemas/                       # domain schemas, OWL, SHACL
├── data/                          # downloaded datasets (gitignored) + goldkg/
├── runs/                          # artifacts, scores, reports (gitignored)
└── tests/
```

**Storage of results:** each run → `runs/<exp>/<run_id>/{config.yaml, stage_artifacts/*.jsonl, usage.json, scores.json}`; plus a single **DuckDB** (or SQLite) table `results(exp, run_id, dataset, pipeline, stage, adapter, model, provider, seed, mode, metric, value)` for fast slicing. Optional: log to **MLflow** for UI/history.

---

## 8. Report contents

Generated per experiment (HTML + Markdown + JSON):

1. **Summary leaderboard** per stage: rows = pipeline × model/provider; columns = key metrics, mean ± 95% CI, cost, latency. Best per column highlighted; significance markers vs baseline.
2. **Quality vs cost** scatter (e.g. triple soft-F1 vs $/1M chars), Pareto frontier marked.
3. **Error attribution** waterfall: isolated vs cascaded per stage → where quality is lost.
4. **Strict vs soft** gap chart for extraction.
5. **ER diagnostics:** over/under-merge examples, confusion of cluster sizes.
6. **Graph profile:** coverage, duplicates, degree distribution, schema violations (top SHACL violations).
7. **Error samples:** random + worst-case examples per stage with source text (drill-down links).
8. **Judge reliability** box: judge model, prompt version, human agreement.
9. **Reproducibility footer:** config hash, git commit, package versions, dataset versions, date.

---

## 9. Implementation notes & gotchas

- **Provider switching:** LiteLLM model strings handle OpenAI, Anthropic, Gemini, Azure, Bedrock, Vertex, Mistral, Groq, Together, Ollama, vLLM/OpenAI-compatible. Keep API keys in `.env`, never in YAML. Normalize structured output: use JSON-schema mode where supported, otherwise validate + repair + retry; record **JSON-validity rate** as a metric (it differs a lot between models).
- **Fairness across providers:** same prompts (versioned in `prompts/`), temperature 0, same max tokens, same chunking; log provider-side model version string returned in responses.
- **Caching key** = hash(adapter name + adapter version + config + input artifact hash + prompt version + model). Lets you change only the ER stage and reuse parsing/extraction.
- **Isolated mode** needs gold of previous stage: Text2KGBench/Re-DocRED give gold text (skip parse); GoldKG gives every layer. Not every dataset supports every stage — dataset declares `supported_stages`.
- **Soft matching threshold:** calibrate τ on ~200 human-labeled pairs; store τ with results.
- **Framework adapters:** Microsoft GraphRAG accepts only `.txt`/CSV input — feed it parsed markdown from S1 and read its output parquet tables; neo4j-graphrag components can be called individually; LightRAG/Graphiti may need small exporters to dump entities/relations.
- **Graph stores:** default to NetworkX for scoring (no infra). Kuzu was archived in Oct 2025 — avoid as a default embedded store.
- **Budget guard:** `--dry-run` estimates tokens from chunk counts × per-call estimates; runner stops at `max_cost_usd`.
- **Parallelism:** async gateway with per-provider rate limits (`asyncio.Semaphore`), resumable runs (skip completed artifacts).
- **Licenses:** OmniDocBench is non-commercial; verify every ⚠️ dataset before using results commercially or redistributing.

---

## 10. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M1** | Core: contracts, config + matrix expansion, LiteLLM gateway with usage/cost logging + cache, runner, CLI `run/score/report` skeleton | `gre run` executes a 2-model × 1-dataset toy experiment end-to-end and writes `scores.json` |
| **M2** | Extraction + ER track: Text2KGBench(-LettrIA) & Re-DocRED loaders; `llm_structured` extractor; ER adapters (exact, fuzzy, embedding, LLM pairwise); strict/soft triple F1, hallucination, domain/range, B³/CEAF/pairwise | model sweep across 3+ providers produces a leaderboard table |
| **M3** | Parsing track: Docling, MinerU, Marker, VLM-parser adapters; OmniDocBench + olmOCR-Bench loaders & scorers | parser comparison report incl. cost/page |
| **M4** | Schema/ontology track: provided + LLM-induced schema; schema metrics; pySHACL validation; graph intrinsic metrics | schema conformance & graph profile in report |
| **M5** | Framework adapters: neo4j-graphrag SimpleKGPipeline, Microsoft GraphRAG, LightRAG (+ Cognee/Graphiti optional) | frameworks appear as rows alongside modular pipelines |
| **M6** | Isolated vs cascaded mode + error-attribution waterfall; bootstrap CIs & paired tests | report shows where quality is lost per stage |
| **M7** | GoldKG in-house dataset loader + annotation export (Label Studio/Argilla); LLM-judge module + agreement check; MINE-style fact retention | full report on your own documents |
| **M8** | Streamlit UI (select runs, compare, drill into errors); HTML report polish | non-developers can run and compare |
| **M9 (Phase 2 hook)** | Retrieval interface: competency questions + GraphRAG-Bench / HotpotQA coverage probe; RAGAS/DeepEval for retrieval & answer metrics | ingestion configs ranked by downstream QA too |

---

## 11. Key references

- neo4j-graphrag-python KG builder — https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html
- Microsoft GraphRAG outputs — https://microsoft.github.io/graphrag/index/outputs
- LightRAG — https://github.com/HKUDS/LightRAG
- Cognee ontologies — https://docs.cognee.ai/core-concepts/further-concepts/ontologies
- Graphiti — https://github.com/getzep/graphiti
- KGGen / MINE — https://arxiv.org/abs/2502.09956
- D2A-HKG (LLM judge validation) — https://arxiv.org/abs/2510.11297
- Text2KGBench — https://arxiv.org/abs/2308.02357 · LettrIA revision — https://ceur-ws.org/Vol-4041/paper3.pdf
- BLINKG — https://github.com/citiususc/blinkg
- OmniDocBench — https://github.com/opendatalab/OmniDocBench
- olmOCR-Bench — https://huggingface.co/datasets/allenai/olmOCR-bench
- PureDocBench — https://arxiv.org/abs/2605.07492
- OAEI 2025 results — https://ceur-ws.org/Vol-4144/om2025-oaei-paper0.pdf
- LLMs4OL 2025 — https://arxiv.org/abs/2508.19428
- Entity matching with LLMs (WDC/Magellan numbers) — https://arxiv.org/abs/2409.08185
- GraphRAG-Bench — https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- TDG-Bench — https://arxiv.org/abs/2605.05476
- RAG vs GraphRAG (KG coverage loss) — https://arxiv.org/abs/2502.11371
- Unbiased GraphRAG evaluation — https://arxiv.org/abs/2506.06331
- pySHACL — https://github.com/RDFLib/pySHACL · DeepOnto — https://github.com/KRR-Oxford/DeepOnto
- LiteLLM — https://github.com/BerriAI/litellm