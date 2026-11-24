# Integration guide

How to evaluate **your** KG pipeline (and **your** gold data) with `gre`, step by step.
Every snippet here is runnable. The complete working version is in [examples/quickstart/](../examples/quickstart/).

- [1. Install](#1-install)
- [2. The 5-minute path](#2-the-5-minute-path)
- [3. Wrap your pipeline](#3-wrap-your-pipeline)
- [4. Choose models](#4-choose-models)
- [5. Add your golden dataset (next to the SOTA ones)](#5-add-your-golden-dataset-next-to-the-sota-ones)
- [6. Pick datasets, stages and modes](#6-pick-datasets-stages-and-modes)
- [7. Read the results](#7-read-the-results)
- [8. Add your own metric](#8-add-your-own-metric)
- [9. Recipes](#9-recipes)
- [10. CI regression gate](#10-ci-regression-gate)
- [11. YAML reference](#11-yaml-reference)
- [12. CLI cheat-sheet](#12-cli-cheat-sheet)
- [13. Troubleshooting](#13-troubleshooting)

---

## 1. Install

```bash
git clone <this repo> graphevals && cd graphevals
uv venv && uv pip install -e ".[llm]"      # or: pip install -e ".[llm]"
gre --help
```

If your pipeline lives in another repo, install `gre` into **that** repo's environment (`pip install -e /path/to/graphevals[llm]`), then run `gre` from your repo root. That way `callable: "your_pkg.module:fn"` imports your code.

## 2. The 5-minute path

```bash
gre run examples/quickstart/experiment.yaml     # offline: your-style pipeline + your-style gold + toy set
open runs/quickstart/report/report.html
```

Then copy the pattern into your project:

```bash
gre init .                                      # creates experiments/my_first_eval.yaml, my_pipeline.py, .env.example
cp .env.example .env                            # add API keys
gre run experiments/my_first_eval.yaml --quick --dry-run   # what will run + estimated cost
gre run experiments/my_first_eval.yaml --quick             # ~20 samples per dataset
```

## 3. Wrap your pipeline

A KG pipeline is split into **stages**. Implement only the stages you have:

| stage | your function gets (by name) | it returns |
|---|---|---|
| `extract` | `chunks`, `schema`, `text`, `ctx`, … | triples |
| `resolve` | `mentions`, `candidate_pairs`, `ctx`, … | clusters of mention ids |
| `graph` *(optional)* | `triples`, `clusters`, `ctx`, … | a graph |

### Arguments are injected by name

Declare only the parameters you need. Everything below is available:

| name | type | notes |
|---|---|---|
| `text` | `str` | full sample text |
| `chunks` | `list[Chunk]` | `.chunk_id .doc_id .text`. One chunk per sample unless the dataset provides more |
| `schema` | `GraphSchema \| None` | `.entity_types[].name`, `.relation_types[].name/.domain/.range` |
| `mentions` | `list[Mention]` | `.mention_id .text .type .context .attributes` (resolve input) |
| `candidate_pairs` | `list[(id, id)] \| None` | set for entity-matching benchmarks: only these pairs are scored |
| `triples` | `list[Triple]` | upstream extraction output (graph stage) |
| `clusters` | `list[EntityCluster] \| None` | upstream resolve output (graph stage) |
| `sample` | `Sample` | the whole sample (avoid reading `gold_*` fields; that's cheating) |
| `ctx` | `Context` | `ctx.llm`, `ctx.embed()`, `ctx.params`, `ctx.seed`, `ctx.log` |
| `params` / any YAML key | | extra keys in the YAML stage block arrive as kwargs |

Your function is called **once per sample**. Samples run in parallel threads (`runtime.workers`). `async def` functions are supported.

### Return whatever is natural

gre coerces it to its contracts:

```python
# triples: any of these
[("Alice", "worksFor", "Acme")]                                   # tuples (+ optional confidence)
[{"subject": "Alice", "predicate": "worksFor", "object": "Acme"}]  # dicts; also head/rel/tail, s/p/o, source/target
[{"subject": "Alice", "subject_type": "Person", ...}]               # types enable domain/range checks
{"triples": [...]}                                                  # wrapped
networkx.DiGraph  (edge attr predicate | relation | label)          # graphs
gre.Triple(...)                                                     # native

# clusters: any of these
{"m1": "alice", "m2": "alice", "m3": "acme"}     # mention_id -> cluster key
[["m1", "m2"], ["m3"]]                           # groups (missing ids become singletons)
[("m1", "m2")]                                   # match pairs (merged transitively)
```

Unusual key names? Map them in YAML: `field_map: {subject: src, predicate: edge, object: dst}`.

### Reference it from YAML (no gre import needed)

```python
# my_pipeline.py: your existing code stays where it is
from my_company.kg import Extractor, canonicalize

def extract(chunks, schema, ctx, max_triples: int = 50):
    ex = Extractor(llm=ctx.llm.complete)              # use ctx.llm so the matrix controls the model
    return [t for c in chunks for t in ex.run(c.text, ontology=schema)][:max_triples]

def resolve(mentions):
    return {m.mention_id: canonicalize(m.text) for m in mentions}
```
```yaml
pipelines:
  mine:
    extract: { callable: "my_pipeline:extract", provider: "${matrix.llm}", max_triples: 80 }
    resolve: { callable: "my_pipeline:resolve" }
```

### Classes

`callable: "my_pipeline:MyResolver"` is instantiated **once** with the YAML keys that match `__init__`. gre then calls `.run(...)` if it exists, otherwise the method named after the stage, otherwise `__call__`. Put expensive setup (loading a model, an index) in `__init__`.

### Decorator style (optional)

```python
import gre

@gre.stage("extract", name="my_extractor", version="2")
def extract(chunks, ctx): ...
```
Add `plugins: [my_pipeline]` to the YAML and reference it with `adapter: my_extractor`.

### Using the LLM gateway (`ctx.llm`)

```python
text = ctx.llm.complete("prompt", system="optional system prompt", max_tokens=800)
from gre.llm.structured import complete_json
data = complete_json(ctx.llm, "Return JSON {...}")      # parsed dict/list or None; repairs + retries
vecs = ctx.embed(["a", "b"])                             # needs `embedding:` on the stage
```

Calls through `ctx.llm` are cached and budget-guarded, and they count toward the report's cost, latency and JSON-validity columns. If your code has its own LLM client, it still works, but those calls won't show up in cost or caching.

## 4. Choose models

Define aliases once, then sweep them:

```yaml
providers:
  gpt:    { model: "openai/gpt-4.1-mini" }
  claude: { model: "anthropic/claude-sonnet-4-5" }
  gemini: { model: "gemini/gemini-2.5-flash" }
  local:  { model: "ollama_chat/qwen2.5-coder:7b", api_base: "http://localhost:11434", max_tokens: 2048 }
  vllm:   { model: "openai/Qwen3-32B", api_base: "http://gpu-box:8000/v1" }
embeddings:
  default: { model: "openai/text-embedding-3-small" }

pipelines:
  mine:
    extract: { callable: "my_pipeline:extract", provider: "${matrix.llm}" }
matrix:
  llm: [gpt, claude, local]      # one run per value. Only pipelines that use ${matrix.llm} are expanded
```

* API keys go in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …), never in YAML.
* To switch models without editing the file: `gre run exp.yaml --set matrix.llm=[claude]`.
* Extra provider keys (`temperature`, `max_tokens`, `api_base`, `timeout`, `max_concurrency`) are passed through. Temperature defaults to 0.
* Unknown pricing (local models)? Add `price_in` / `price_out` in $ per 1M tokens to get cost columns.

## 5. Add your golden dataset (next to the SOTA ones)

Your gold set is a JSONL file declared in the YAML. It runs **in the same experiment and report** as the public benchmarks:

```yaml
local_datasets:
  my_gold:
    path: data/my_gold.jsonl             # relative to where you run gre
    stages: [extract, resolve, graph]    # what your gold supports
    schema: schemas/domain.yaml          # optional, applied to every line without its own schema

suite: kg_sota                           # the 7 public benchmarks …
datasets: [my_gold]                      # … plus your golden set = 8 datasets in one run
```

### Line format

Include only `id` plus the fields your stages need:

```json
{"id": "doc1",
 "text": "Maria Lopez works for Nimbus Labs. Maria was born in Porto.",
 "gold_triples": [["Maria Lopez", "worksFor", "Nimbus Labs"],
                  {"subject": {"text": "Maria Lopez", "aliases": ["Maria"]}, "predicate": "bornIn", "object": "Porto"}],
 "mentions": [{"mention_id": "m1", "text": "Maria Lopez", "type": "Person", "context": "..."},
              {"mention_id": "m2", "text": "Maria", "type": "Person"}],
 "gold_clusters": [["m1", "m2"]],
 "gold_facts": ["Maria Lopez works for Nimbus Labs"],
 "meta": {"source": "contract_17.pdf"}}
```

| field | enables | notes |
|---|---|---|
| `text` (or `chunks`) | extract | `chunks: [{"chunk_id", "doc_id", "text"}]` if you pre-chunk |
| `gold_triples` | triple P/R/F1, entity F1, hallucination, graph coverage | tuples or dicts. Use `aliases` so any surface form counts |
| `mentions` + `gold_clusters` | B³, CEAF-e, MUC, ARI, pairwise | clusters as id-groups or `{"canonical_id", "mention_ids"}` |
| `mentions` + `candidate_pairs` + `gold_pairs` | pairwise P/R/F1 (entity-matching style) | `gold_pairs: [["a","b",1], ["a","c",0]]` |
| `gold_facts` | MINE-style fact retention (needs `judge:`) | short natural-language facts |
| `schema` | ontology conformance, domain/range | per line (path) or via `schema:` option |

### Schema file (short form)

```yaml
entity_types: [Person, Organization, City]
relations:
  - { name: worksFor, domain: Person, range: Organization }
  - bornIn                                   # name only is fine
```

### Tips for building the gold set

* **Size:** 30–100 representative documents beat thousands of easy ones.
* **Labelling:** pre-label with your strongest model, have humans correct it, and record the correction rate.
* **Hard negatives:** for ER, include similar names that are *different* entities.
* **Facts:** add 10–20 key facts per document. They also feed future retrieval evaluation.
* **Python loader:** if you need one (e.g. reading from a DB), subclass `gre.datasets.base.Dataset` with `@gre.dataset("name")` and implement `load(split) -> list[gre.Sample]`.

## 6. Pick datasets, stages and modes

```yaml
suite: kg_sota            # or: extract | resolve | graph | smoke
datasets: [my_gold, redocred:test[:100], abt_buy]   # name[:split][[a:b]] slices
mode: [isolated, cascaded]
stages: [extract, resolve]                          # optional filter
```

| dataset | stages | use it to measure |
|---|---|---|
| `text2kgbench_webnlg`, `text2kgbench_tekgen` | extract | schema-guided extraction, hallucination |
| `redocred` | extract, resolve, graph | document-level relations, mention clustering |
| `mine` | graph (+ extract, resolve) | does the built KG retain the facts? |
| `wdc_products`, `abt_buy`, `amazon_google` | resolve | entity matching on labelled pairs |

* A pipeline only runs on datasets whose stages it implements. A resolver-only pipeline skips Text2KGBench automatically, and identical runs are de-duplicated.
* **isolated** feeds each stage gold upstream input, which tests the stage alone.
* **cascaded** feeds it your own upstream output (resolve gets mentions from your extracted triples), which tests error propagation.
* The report shows `cascaded − isolated` per metric.
* If you have no `graph` stage, a default builder turns your triples and clusters into the graph that the graph metrics score.

## 7. Read the results

```
runs/<exp>/
  report/report.html   leaderboard (CIs, best, ▲▼ significance vs baseline), SOTA refs, quality-vs-cost,
                       cascade deltas, all metrics, worst samples with missed / spurious triples
  report/report.md     same tables for PRs / chat
  report/report.json   machine-readable
  results.duckdb       SELECT * FROM results WHERE metric LIKE 'extract.%'
  results.csv
  <run_id>/samples.jsonl   per-sample predictions + metrics (debug a single document)
  <run_id>/scores.json     aggregated metrics + usage
```

* Pick the baseline for ▲/▼ markers with `report: {baseline: "baseline[llm=gpt]"}` (a label substring), or `gre report runs/x --baseline ...`.
* Set `seeds: [0, 1, 2]` to average over seeds. Use it for any stochastic pipeline.

## 8. Add your own metric

```python
# my_metrics.py
import gre

@gre.metric("extract", "typed_precision", description="share of triples with both entity types set")
def typed_precision(pred, sample, ctx):              # pred: list[Triple] | list[EntityCluster] | KnowledgeGraph
    return sum(bool(t.subject.type and t.object.type) for t in pred) / len(pred) if pred else None

@gre.metric("extract", "critical_relations", aggregate="micro_prf", requires=("gold_triples",))
def critical_relations(pred, sample, ctx):           # micro P/R/F1 from counts
    crit = {"owns", "acquired"}
    g = {(t.subject.text, t.predicate) for t in sample.gold_triples if t.predicate in crit}
    p = {(t.subject.text, t.predicate) for t in pred if t.predicate in crit}
    return {"tp": len(g & p), "fp": len(p - g), "fn": len(g - p)}
```
```yaml
plugins: [my_metrics]
scoring: { include: [], exclude: ["graph.structure*"] }   # optional glob filters
```
```bash
gre score runs/<exp> --config exp.yaml    # re-score stored predictions: no pipeline re-run, no LLM cost
```

* Return `None` to skip a sample.
* `ctx.judge` gives you the judge LLM, and `ctx.scoring.soft_match_threshold` the soft-match threshold.
* `gre metrics` lists everything that is registered.

## 9. Recipes

**Evaluate only my entity resolver** (gold mentions in, clusters out):
```yaml
suite: resolve
pipelines: { mine: { resolve: { callable: "my_pipeline:resolve" } } }
```

**Whole framework as a black box** (text in, graph out):
```python
def extract(text, ctx):
    g = MyFramework(llm=ctx.llm.complete).build(text)   # networkx graph or {"nodes","edges"}
    return g                                            # edges -> triples; default graph builder runs after
```

**Compare against the built-in baselines:**
```yaml
pipelines:
  mine:     { extract: { callable: "my_pipeline:extract", provider: "${matrix.llm}" } }
  baseline: { extract: { adapter: llm_structured, provider: "${matrix.llm}" },
              resolve: { adapter: fuzzy, threshold: 0.85 } }      # also: exact, embedding, llm_pairwise
```

**Hyper-parameter sweep:**
```yaml
pipelines:
  mine: { resolve: { callable: "my_pipeline:resolve", threshold: "${matrix.thr}" } }
matrix: { thr: [0.75, 0.85, 0.95] }
```

**From Python / a notebook:**
```python
import gre
summaries, report_paths = gre.evaluate("exp.yaml", overrides=["matrix.llm=[claude]"], quick=True)
```

## 10. CI regression gate

```bash
gre run exp.yaml --quick -d my_gold
gre check runs/exp --baseline runs_main/exp --fail-if "extract.triple_soft.f1 < -0.02" \
                                            --fail-if "resolve.bcubed.f1 < -0.02"
gre check runs/exp --fail-if "usage.error_rate > 0.05"          # absolute rule (no baseline)
```
`check` exits with code 1 if any rule fires. Metric names accept globs, e.g. `"extract.*.f1 < -0.03"`.

## 11. YAML reference

```yaml
name: my_eval                     # output folder runs/<name>
description: ...
suite: kg_sota                    # optional dataset suite
datasets: [my_gold, redocred]     # name[:split][[a:b]]
local_datasets: { my_gold: { path: ..., stages: [...], schema: ... } }
seeds: [0]
mode: [isolated, cascaded]
stages: [extract, resolve, graph] # optional filter
plugins: [my_pipeline, my_metrics]# modules to import (decorators register on import)

providers:  { alias: { model: "<litellm model>", ...passthrough params } }
embeddings: { alias: { model: "<litellm embedding model>" } }
judge:      { provider: alias, prompt_version: v1 }   # keep fixed across compared experiments

pipelines:
  <name>:
    <stage>:                      # chunk | extract | resolve | graph
      adapter: <registered name>  #   or
      callable: "module:function" #   function or class
      provider: alias | "${matrix.llm}"   # -> ctx.llm
      embedding: alias            # -> ctx.embed
      field_map: {subject: ..., predicate: ..., object: ...}
      cache: true                 # set false for non-deterministic stages you always want re-run
      version: "2"                # optional manual cache-buster
      <any other key>: value      # passed to your function as a keyword argument
matrix: { llm: [a, b], thr: [0.8, 0.9] }

scoring: { soft_match_threshold: 0.8, soft_method: fuzzy|embedding, embedding: default,
           include: [], exclude: [], n_bootstrap: 1000, error_samples: 5 }
budget:  { max_cost_usd: 50 }     # hard stop. Partial results are kept
runtime: { workers: 8, cache: true, cache_path: .gre_cache/cache.sqlite, quick_limit: 20, fail_fast: false }
report:  { formats: [html, md, json], baseline: "label substring" }
output_dir: runs
```

## 12. CLI cheat-sheet

| command | what it does |
|---|---|
| `gre init [dir]` | scaffold config, pipeline wrapper, `.env.example` |
| `gre run exp.yaml` | run the matrix (cached, resumable) and build the report |
| `  --dry-run` | list runs + estimated LLM calls and $ |
| `  --quick` / `--limit N` | stratified subsample / first N samples per dataset |
| `  --set k=v` | override any YAML key (`matrix.llm=[claude]`, `scoring.soft_match_threshold=0.85`) |
| `  -d name` / `--suite s` / `--stages a,b` | choose datasets / suite / stages |
| `  --force` / `--no-cache` | recompute finished runs / ignore cached LLM calls and stage outputs |
| `gre report runs/x [--baseline lbl]` | rebuild the HTML/MD/JSON report |
| `gre score runs/x [--config exp.yaml]` | re-score stored predictions with current metrics |
| `gre compare runs/a runs/b` | per-run deltas with paired-bootstrap p-values |
| `gre check runs/x [--baseline runs/y] --fail-if "..."` | CI gate |
| `gre datasets list / download [names] / info name` | dataset registry |
| `gre metrics` / `gre adapters` | registered metrics / stage adapters |

## 13. Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError` for your callable | run `gre` from your project root (it is added to `sys.path`), or `pip install -e` your package |
| `... needs ['x']; available inputs: [...]` | rename the parameter to one of the listed names, or give it a default |
| `this stage has no provider: configured` | add `provider:` to the stage that uses `ctx.llm` |
| results didn't change after editing code | the cache is keyed by your module's source, so a change in a *different* module (a helper you import) is not seen. Use `--no-cache`, or bump `version:` |
| metrics show `–` | the dataset lacks that gold (see `gre datasets info`), or the stage isn't in your pipeline |
| run stopped early | `budget.max_cost_usd` was reached. Partial results are kept. Raise it and re-run (finished runs are skipped) |
| local model very slow / endless output | set `max_tokens` on the provider. JSON mode on small models can loop |
| `error_rate` > 0 | `runs/<exp>/<run>/samples.jsonl` has the exception per sample. Use `runtime.fail_fast: true` to stop at the first one |
| fact retention looks too high | the judge is too weak or lenient. Use a strong model and keep `judge` fixed across experiments |
