# GraphEvals (`gre`)

Benchmark **knowledge-graph construction pipelines** stage by stage (extraction → entity resolution → graph) against public SOTA datasets and your own gold data. Plug in your existing code without rewriting it, switch LLMs with a config string, add metrics, and get one comparable report.

```
your pipeline ──► gre runner ──► 7 public benchmarks + your gold set ──► leaderboard · CIs · cost · error samples
   (functions)     (matrix of models × pipelines × modes × seeds, cached, budget-guarded)
```

> **Start here → [docs/INTEGRATION.md](docs/INTEGRATION.md)**: a step-by-step guide covering how to wrap your pipeline, choose models, add your golden dataset next to the SOTA ones, add metrics, read the report, and gate CI. It also has the YAML and CLI references and troubleshooting.

## Quickstart

```bash
uv venv && uv pip install -e ".[llm,dev]"
gre run examples/quickstart/experiment.yaml       # offline: example pipeline + example gold JSONL + toy set
open runs/quickstart/report/report.html
```

Then for your own project:

```bash
gre datasets download kg_sota                     # ~40 MB into ~/.cache/gre/data
gre init my_eval && cd my_eval                    # scaffold config + pipeline wrapper + .env.example
gre run experiments/my_first_eval.yaml --quick --dry-run   # matrix + cost estimate
gre run experiments/my_first_eval.yaml --quick             # ~20 stratified samples per dataset
open runs/my_first_eval/report/report.html
```

[examples/quickstart/](examples/quickstart/) is the smallest complete integration:
- `my_pipeline.py`: your code (an `async` extractor and a resolver)
- `gold.jsonl` + `schema.yaml`: your gold data
- `experiment.yaml`: runs them next to a baseline

## Plug in your pipeline (no rewrite)

Parameters are injected **by name**. Ask only for what you need: `text, chunks, schema, mentions, candidate_pairs, triples, clusters, sample, ctx, params`, plus any key from the YAML stage block.

```python
# my_pipeline.py
def extract(chunks, schema, ctx):                  # ctx.llm = the model picked by the matrix
    return my_builder.run([c.text for c in chunks], llm=ctx.llm.complete)

def resolve(mentions, threshold=0.9):              # YAML params arrive as kwargs
    return {m.mention_id: my_canon(m.text) for m in mentions}
```
```yaml
pipelines:
  mine:
    extract: { callable: "my_pipeline:extract", provider: "${matrix.llm}" }
    resolve: { callable: "my_pipeline:resolve", threshold: 0.9 }
```

* **Outputs can be loose.** Triples can be `Triple`s, dicts (`subject/predicate/object`, `head/rel/tail`, `s/p/o`…), tuples, `{"triples": [...]}`, or a networkx graph. Clusters can be lists of id-lists, `{mention_id: cluster}`, or match pairs. Use `field_map:` for unusual key names.
* **Classes work too.** `callable: "pkg:MyResolver"` is instantiated once with the YAML params, and then `.run(...)` is called.
* **Decorator style** works as well: `@gre.stage("extract", name="my_extractor")`, then `adapter: my_extractor` in the YAML. See [examples/byo_pipeline/](examples/byo_pipeline/).
* **`async def` functions** are supported.
* **Programmatic use:** `gre.evaluate("exp.yaml", overrides=["matrix.llm=[claude]"], quick=True)`.

## Switch models

Every model call goes through one LiteLLM gateway (`ctx.llm`), so any provider is just a string: `openai/…`, `anthropic/…`, `gemini/…`, `ollama_chat/…`, `vllm`/OpenAI-compatible, Bedrock, Azure, and more. The gateway records tokens, $ cost, latency, cache hits and JSON validity.

```bash
gre run exp.yaml --set matrix.llm=[claude,local]      # override from the CLI
gre run exp.yaml --stages resolve                     # evaluate one stage only
gre run exp.yaml -d redocred:test[:50]                # any dataset slice
```

## Benchmarks (`suite: kg_sota`)

| dataset | stage | test size | what it stresses | headline metric |
|---|---|---|---|---|
| `text2kgbench_webnlg` | extract | 2,014 sent · 19 ontologies | ontology-conformant extraction (DBpedia) | triple F1, conformance, hallucination |
| `text2kgbench_tekgen` | extract | 4,062 sent · 10 ontologies | ontology-conformant extraction (Wikidata) | triple F1, hallucination |
| `redocred` | extract · resolve · graph | 500 docs · ~35 facts/doc · 96 relations | document-level, cross-sentence relations; mention clustering | triple F1, B³, duplicate-node rate |
| `mine` | graph (+extract/resolve) | 101 essays × 15 facts | does the built KG *retain* facts? (LLM judge) | fact retention (+ KGGen/GraphRAG/OpenIE refs) |
| `wdc_products` | resolve | 4,500 pairs, 80% corner cases | hardest product entity matching | pairwise F1 |
| `abt_buy` | resolve | 1,916 pairs | classic EM benchmark | pairwise F1 (Ditto ref) |
| `amazon_google` | resolve | 2,293 pairs | classic EM benchmark | pairwise F1 (Ditto ref) |

Other suites are `extract`, `resolve`, `graph` and `smoke`. Run `gre datasets list` and `gre datasets info redocred` for details. **Licenses vary, and several are research-only. Check them before commercial use.**

## Evaluation modes

* **isolated**: each stage gets *gold* upstream input (resolve ← gold mentions). This measures the stage on its own.
* **cascaded**: each stage gets the pipeline's *own* upstream output (resolve ← mentions from predicted triples). The report's "cascaded − isolated" table shows where quality is lost.
* The graph stage is always end-to-end. If a pipeline has no graph stage, a default NetworkX builder is used.

## Metrics

Run `gre metrics` to see the full list. Built-in metrics include:
* **extract**: strict and soft triple P/R/F1 (alias-aware, camelCase-insensitive, greedy 1:1 matching), entity F1, relation F1, ontology conformance, Text2KGBench hallucination (subject/relation/object), domain/range violations.
* **resolve**: pairwise P/R/F1, B³, CEAF-e, MUC, ARI, over/under-merge, dedup ratio.
* **graph**: MINE fact retention (LLM judge, fixed prompt version), an LLM-free lexical fact coverage, entity coverage, duplicate-node rate, and structure (degree, orphans, giant component).
* **usage**: $ per sample, latency, JSON-valid rate, error rate.

Every metric gets a **95% bootstrap CI** over samples. Comparisons use a **paired bootstrap** test.

### Add a metric

```python
@gre.metric("extract", "my_precision", aggregate="micro_prf")   # or aggregate="mean" -> float | dict
def my_precision(pred, sample, ctx):
    return {"tp": ..., "fp": ..., "fn": ...}
```
Add the module to `plugins:` and run `gre score runs/<exp>`. Stored predictions are **re-scored without re-running the pipeline**. Use `scoring.include` / `scoring.exclude` (globs) to pick metrics.

## Your golden dataset, next to the SOTA ones

Declare a JSONL file in the YAML. It is benchmarked in the **same run and report** as the public datasets, with the same metrics, CIs, significance markers and error samples:

```yaml
local_datasets:
  my_gold: { path: data/my_gold.jsonl, stages: [extract, resolve, graph], schema: schemas/domain.yaml }
suite: kg_sota          # 7 public benchmarks
datasets: [my_gold]     # + your gold set
```
```json
{"id": "doc1", "text": "Maria Lopez works for Nimbus Labs.", "gold_triples": [["Maria Lopez", "worksFor", "Nimbus Labs"]],
 "mentions": [{"mention_id": "m1", "text": "Maria Lopez"}, {"mention_id": "m2", "text": "Maria"}], "gold_clusters": [["m1", "m2"]],
 "gold_facts": ["Maria Lopez works for Nimbus Labs"]}
```
Every field is optional except `id`; include what your stages need. See [INTEGRATION.md §5](docs/INTEGRATION.md#5-add-your-golden-dataset-next-to-the-sota-ones) for the full format. If you need a Python loader instead (DB, API), subclass `gre.datasets.base.Dataset` with `@gre.dataset("name")`.

## Reports & results

`runs/<exp>/report/report.{html,md,json}` contains:
* a leaderboard per dataset, with CIs, best-per-column highlighting and ▲/▼ significance markers against your baseline
* SOTA reference rows
* a quality-vs-cost Pareto chart
* the cascade error-propagation table
* every metric for every run
* the worst samples, with missed and spurious triples (or false and missed merges)
* a reproducibility footer

All results also land in `runs/<exp>/results.duckdb` and `results.csv`:
```sql
SELECT label, value FROM results WHERE dataset='redocred' AND metric='extract.triple_soft.f1' ORDER BY value DESC;
```

## Daily workflow features

| need | how |
|---|---|
| iterate fast | `--quick` (stratified subsample), `--limit N`, `--stages extract` |
| don't pay twice | content-hash cache of LLM calls **and** stage outputs; editing only `resolve` reuses extraction; editing your module invalidates its cached outputs (`--no-cache` to bypass) |
| control spend | `--dry-run` cost estimate, `budget.max_cost_usd` hard stop (partial results kept) |
| resume | finished runs are skipped unless `--force` |
| compare experiments | `gre compare runs/a runs/b` (paired bootstrap p-values) |
| block regressions in CI | `gre check runs/new --baseline runs/main --fail-if "extract.triple_soft.f1 < -0.02"` |
| failures count | a crashing stage scores as an empty prediction and shows in `error_rate` |

## Layout

```
gre/core/       contracts · registry (@stage/@metric/@dataset) · config+matrix · cache · runner · coerce
gre/llm/        LiteLLM gateway (cost/latency/cache/budget, mock models) · structured JSON w/ repair
gre/adapters/   reference baselines: llm_structured extractor · exact/fuzzy/embedding/llm_pairwise resolvers · networkx graph
gre/datasets/   text2kgbench · redocred · mine · entity_matching (wdc/abt-buy/amazon-google) · local (your JSONL) · toy · sota_refs.yaml
gre/metrics/    extract · er · graph · stats (bootstrap, paired tests)
gre/report/     builder + HTML template
docs/           INTEGRATION.md: step-by-step integration guide
examples/       quickstart (offline, own gold data) · byo_pipeline (decorator style + custom metric)
experiments/    toy · kg_sota (cloud models) · kg_sota_local (Ollama) · real_smoke
```

## Roadmap

Following [tools.md](tools.md):
* **M3**: parsing track (Docling/MinerU/Marker, OmniDocBench, olmOCR-Bench)
* **M4**: schema induction + SHACL
* **M5**: whole-framework adapters (MS GraphRAG, LightRAG, neo4j-graphrag)
* **M7**: annotation export (Label Studio/Argilla) + judge-agreement (κ) check
* **M8**: Streamlit UI

Text2KGBench-LettrIA will be added once its data is publicly released.

## Tests

```bash
pytest            # offline unit + e2e (mock models)
pytest -m network # real dataset loaders
```
