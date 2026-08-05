# Query-Adaptive RAG Pipeline

This repository now supports two research corpora:

- QASPER for evidence-aware open-corpus retrieval and answer evaluation;
- Natural Questions Open with a frozen one-million-passage DPR Wikipedia subset.

## Install

For tests:

```powershell
python -m pip install -r requirements/dev.txt
```

For QASPER/NQ experiments:

```powershell
python -m pip install -r requirements/experiment.txt -c requirements/constraints/verified.txt
```

API credentials belong in the process environment:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

Inline credentials in YAML are rejected. Persisted JSON/JSONL removes credential fields and
redacts key-shaped values.

## QASPER

Prepare the local Hugging Face dataset once:

```powershell
python scripts/prepare_qasper.py
```

Run the single-paper offline smoke workflow:

```powershell
python scripts/run_qasper_smoke.py --config configs/qasper_smoke.yaml --max-questions 3
```

Build and run the formal open-corpus evaluation:

```powershell
python scripts/check_environment.py --config configs/qasper_baseline.yaml --strict-credentials
python scripts/build_index.py --config configs/qasper_baseline.yaml
$runId = "qasper_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts/run_qasper_eval.py --config configs/qasper_baseline.yaml --run-id $runId
```

The reported protocol is `qasper_open_corpus_text_extractive_single_evidence_v2`. It evaluates the
fixed `answerable + text-only + extractive + single-evidence` validation slice against the global
train/validation/test paper corpus. It is not a full-QASPER score.

## Natural Questions + DPR Wikipedia

Prepare the frozen database and question splits from local DPR source archives:

```powershell
python scripts/prepare_nq_dpr_wiki.py `
  --wikipedia data/raw/dpr/psgs_w100.tsv.gz `
  --questions data/raw/dpr/biencoder-nq-dev.json.gz `
  --output-dir data/nq_open_dpr_wiki_1m
```

Build the exact FlatIP reference and run the fixed evaluation split:

```powershell
python scripts/check_environment.py --config configs/nq_dpr_wiki.yaml --strict-credentials
python scripts/build_index.py --config configs/nq_dpr_wiki.yaml
$runId = "nq_flat_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts/run_nq_eval.py --config configs/nq_dpr_wiki.yaml --run-id $runId
```

Available controlled variants:

| Config | Retrieval condition |
|---|---|
| `configs/nq_dpr_wiki.yaml` | exact FAISS FlatIP reference |
| `configs/nq_dpr_wiki_hnsw.yaml` | HNSW |
| `configs/nq_dpr_wiki_ivf_flat.yaml` | IVF-Flat |
| `configs/nq_dpr_wiki_ivf_pq.yaml` | IVF-PQ |
| `configs/nq_dpr_wiki_rerank.yaml` | FlatIP candidates plus cross-encoder reranking |
| `configs/nq_dpr_wiki_bm25.yaml` | memory-mapped BM25S lexical retrieval |

Build and evaluate the BM25 baseline:

```powershell
python scripts/check_environment.py --config configs/nq_dpr_wiki_bm25.yaml --strict-credentials
python scripts/build_bm25_index.py --config configs/nq_dpr_wiki_bm25.yaml
$runId = "nq_bm25_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts/run_nq_eval.py --config configs/nq_dpr_wiki_bm25.yaml --run-id $runId
```

The first BM25 stage intentionally reuses the active immutable build's `chunks.jsonl` and
zero-based vector IDs. Its sparse arrays live under `artifacts/_sparse_indexes/`, have their own
manifest, and are bound to the chunk SHA-256, row count, BM25 parameters, BM25S version, and source
fingerprint. Query startup validates that manifest and loads the sparse arrays with mmap; it does
not load the query embedding model or FAISS index.

The frozen corpus is gold-conditioned and is intended for controlled comparisons inside this
pipeline. Do not report it as a standard full-Wikipedia NQ benchmark.

## Query an existing build

```powershell
python scripts/run_query.py `
  --config configs/nq_dpr_wiki.yaml `
  --query "Who wrote Hamlet?" `
  --no-log
```

`--top-k 0` explicitly gates retrieval. Candidate depth, final depth, ANN search parameters, and
reranking remain separate query-time controls.

## Project layout

```text
configs/                     QASPER and NQ experiment configurations
data/                        dataset documentation and local ignored datasets
scripts/                     entry points plus shared CLI boundary support
src/*.py                     contracts, stage factories, and pipeline orchestration only
src/chunkers/                chunking component implementations
src/context_builders/        retrieved-hit to prompt-context implementations
src/embedders/               document and query embedding implementations
src/generators/              answer-generation implementations
src/indexes/                 NumPy and FAISS vector-index implementations
src/loaders/                 QASPER and DPR Wikipedia corpus adapters
src/model_backends/          shared pinned-model resource resolution
src/persistence/             artifact validation, storage, and run-output writers
src/preparers/               deterministic dataset preparation components
src/prompts/                 prompt implementations
src/rerankers/               reranking implementations and contract
src/retrievers/              dense/BM25 retrieval, sparse-index build, and lazy chunk access
src/text/                    sentence splitting and token counting
src/evaluators/              QASPER/NQ protocol validation and metrics
artifacts/                   local immutable encoded corpora and builds (ignored)
outputs/                     local query/evaluation runs (ignored)
tests/                       unit, contract, and integration coverage
```

Root-level `src` modules define shared contracts or connect components; implementation directories
do not contain pipeline orchestration. Persisted-artifact I/O lives in
`src/persistence/artifact_io.py`, while every disk trust-boundary check remains together in
`src/persistence/artifact_validation.py`. An in-memory verified handle is passed onward so the same
artifact is not checked again within one process.

Builds, runs, and evaluations remain separate reproducibility boundaries. Completed manifests,
artifact size/SHA-256 checks, embedding-space validation, vector-ID consistency, staging, atomic
commit, and reload verification protect persisted artifacts. Evaluation rows are checkpointed
individually and merged once at completion. Historical build directories are not migrated in place.

After source refactoring, existing QASPER/NQ outputs remain valid historical records, but a new
build is required before running the changed source tree.

## Tests

```powershell
C:\Users\12442\anaconda3\python.exe -m pytest -q
```

Focused backend tests use the markers declared in `pytest.ini`. QASPER and NQ keep independent
selection and metric contracts even though they share the same build/query/evaluation runner.

Further details:

- [Dataset contracts](data/README.md)
- [NQ/DPR architecture and experiment plan](docs/nq_dpr_pipeline_plan.md)
