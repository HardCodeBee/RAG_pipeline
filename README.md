# Reproducible Naive RAG Pipeline

This project provides a compact RAG pipeline for demonstration and reproducible experiments:

```text
PDF/QASPER corpus → chunking → embeddings → exact dense retrieval → prompt → generation → evaluation
```

Use `configs/smoke.yaml` for an offline demonstration, `configs/baseline.yaml` for the formal
five-paper baseline, and `configs/qasper_baseline.yaml` for formal QASPER evaluation. Python 3.11
is the verified environment.

## Configuration profiles

| Configuration | Corpus | Embedding | Index | Generator | Intended use |
| --- | --- | --- | --- | --- | --- |
| `configs/smoke.yaml` | Five local PDFs | Deterministic hashing | NumPy exact search | Extractive | Offline end-to-end demonstration |
| `configs/baseline.yaml` | Five local PDFs | Pinned BGE small | FAISS FlatIP | OpenAI | Formal five-paper experiment |
| `configs/qasper_smoke.yaml` | One QASPER validation paper | Deterministic hashing | NumPy exact search | Extractive | Offline QASPER adapter check |
| `configs/qasper_baseline.yaml` | All QASPER papers | Pinned BGE small | FAISS FlatIP | OpenAI | Formal QASPER open-corpus evaluation |

Backends are explicit. A requested SentenceTransformer, FAISS, or OpenAI backend must be available;
the pipeline does not silently replace it with a lightweight fallback. FAISS FlatIP performs exact
inner-product search in this project, not approximate nearest-neighbor search.

## Install

Create an environment and install the offline demo dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/dev.txt -c requirements/constraints/verified.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`.

For the formal baseline or QASPER, also install the experiment dependencies:

```powershell
python -m pip install -r requirements/experiment.txt -r requirements/dev.txt -c requirements/constraints/verified.txt
```

## Repository and data layout

```text
configs/                         four explicit experiment configurations
data/corpus/                     five versioned PDF inputs
data/corpus_manifest.json        frozen PDF file sizes and SHA-256 values
data/questions_v1.jsonl          24 development questions
data/questions_heldout_v1.jsonl  5 held-out questions
data/processed/qasper/           local QASPER cache, generated and Git-ignored
scripts/                         build, query, evaluation, and maintenance entry points
src/                             pipeline implementation
tests/                           offline regression and optional backend tests
artifacts/<build_id>/            generated chunks, embeddings, index, and manifest
outputs/<run_id>/                generated answers, metadata, and metrics
```

`data/` contains experiment inputs, not generated vectors. The PDF corpus and question files are
versioned; `data/processed/`, `artifacts/`, and `outputs/` are local generated data excluded from
Git. Questions and gold labels are used only during evaluation and are never inserted into the
retrieval index.

## Offline demo

The smoke configuration uses hashing embeddings, NumPy exact search, and an extractive generator.
It requires no API key or model download.

```powershell
# Build the index
python scripts/build_index.py --config configs/smoke.yaml

# Ask one question
python scripts/run_query.py --config configs/smoke.yaml --query "What are the stages of Naive RAG?" --no-log

# Evaluate the development questions
$runId = "smoke_$(Get-Date -Format yyyyMMdd_HHmmss)"
python scripts/run_eval.py --config configs/smoke.yaml --questions data/questions_v1.jsonl --run-id $runId
```

Build artifacts are written to `artifacts/`; query and evaluation results are written to `outputs/`.
`run_eval.py` prints a concise demonstration summary, while the complete metrics remain in
`outputs/<run_id>/summary.csv` and `metadata.json`. Use a new `--run-id` for each new evaluation.

## Formal five-paper baseline

The formal configuration uses the pinned `BAAI/bge-small-en-v1.5` model, FAISS FlatIP exact
retrieval, top-5 context, and `gpt-4o-mini`.

```powershell
$env:OPENAI_API_KEY = "your-key"

# Check dependencies and credentials
python scripts/check_environment.py --config configs/baseline.yaml --strict-credentials

# Build the formal index
python scripts/build_index.py --config configs/baseline.yaml

# Ask one question
python scripts/run_query.py --config configs/baseline.yaml --query "What is retrieval-augmented generation?" --no-log

# Evaluate the development set
$runId = "baseline_$(Get-Date -Format yyyyMMdd_HHmmss)"
python scripts/run_eval.py --config configs/baseline.yaml --questions data/questions_v1.jsonl --run-id $runId
```

The API key is read only from `OPENAI_API_KEY`; inline credentials in YAML are rejected. Persisted
JSON/JSONL metadata removes credential fields and redacts key-shaped values. The environment check
verifies credential presence but does not make an OpenAI request.

## QASPER

Prepare the official Hugging Face QASPER DatasetDict once; the first run requires network access:

```powershell
python scripts/prepare_qasper.py
```

The saved copy contains `train`, `validation`, and `test` under
`data/processed/qasper/hf_dataset/`. Later runs use this local copy. Run the single-paper offline
smoke with:

```powershell
python scripts/run_qasper_smoke.py --max-questions 3
```

The formal QASPER configuration is offline for Hugging Face model loading, so the pinned BGE
snapshot must already exist in the local Hugging Face cache. If necessary, cache it once:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')"
```

Run the formal open-corpus evaluation:

```powershell
$env:OPENAI_API_KEY = "your-key"
python scripts/check_environment.py --config configs/qasper_baseline.yaml --strict-credentials
$runId = "qasper_$(Get-Date -Format yyyyMMdd_HHmmss)"
python scripts/run_qasper_eval.py --config configs/qasper_baseline.yaml --run-id $runId
```

The standalone `configs/qasper_baseline.yaml` explicitly selects the all-split QASPER corpus,
FAISS, OpenAI, and the pinned BGE tokenizer/model revision already in the local Hugging Face cache.
`run_qasper_eval.py` builds or validates and reuses the matching index automatically.

All train, validation, and test papers enter the retrieval corpus; papers other than the target
paper act as distractors. Evaluation questions come only from validation and use protocol
`qasper_open_corpus_text_extractive_single_evidence_v2` and question slice
`answerable_text_only_extractive_single_evidence_v1`. An eligible reference must be answerable,
extractive, text-only, and backed by exactly one non-empty evidence unit. This slice excludes
multi-evidence, table/figure, abstractive, yes/no, and unanswerable cases, so its scores must not be
reported as full-QASPER performance. Here, "extractive" describes the selected reference type; the
formal generator is still OpenAI.

For a small paid backend check, add `--max-questions 5`. To ask a free-form question against the
full QASPER index, build it once and use the same explicit configuration:

```powershell
python scripts/build_index.py --config configs/qasper_baseline.yaml
python scripts/run_query.py --config configs/qasper_baseline.yaml --query "Your question" --no-log
```

## Common CLI operations

Override retrieval depth without rebuilding the index:

```powershell
python scripts/run_query.py --config configs/baseline.yaml --query "Your question" --top-k 10 --no-log
python scripts/run_eval.py --config configs/baseline.yaml --questions data/questions_v1.jsonl --run-id baseline_top10 --top-k 10
python scripts/run_qasper_eval.py --config configs/qasper_baseline.yaml --run-id qasper_top10 --top-k 10
```

Resume an interrupted evaluation with the original run ID and exactly the same question, build,
run, evaluation, and effective top-k identities:

```powershell
python scripts/run_eval.py --config configs/baseline.yaml --questions data/questions_v1.jsonl --run-id baseline_run --resume
python scripts/run_qasper_eval.py --config configs/qasper_baseline.yaml --run-id qasper_run --resume
```

If the original run used `--top-k` or `--max-questions`, repeat the same overrides when resuming.
`run_query.py` creates an immutable output directory unless `--no-log` is supplied. Use
`python scripts/<name>.py --help` to inspect all supported arguments.

## Recompute generic metrics

Recompute metrics from saved generic evaluation results without running retrieval or generation:

```powershell
python scripts/recompute_metrics.py --source-run-dir outputs/<source_run_id> --questions data/questions_v1.jsonl --run-id <reanalyzed_run_id>
```

The question file must match the original run. This command does not recompute QASPER open-corpus
metrics.

## Artifacts and reproducibility

Each `artifacts/<build_id>/` directory is immutable and contains:

```text
manifest.json
chunks.jsonl
embeddings.npy
index.faiss       # FAISS builds only
```

Each completed evaluation under `outputs/<run_id>/` contains:

```text
metadata.json     experiment identity and effective configuration
results.jsonl     one retrieval/generation record per question
summary.csv       complete aggregate metrics
```

| Identity | Changes when |
| --- | --- |
| Build identity | Corpus, loader, chunking, document embedding, index, or build-stage source changes |
| Run identity | Build, query settings, top-k, context, prompt, generator, or run-stage source changes |
| Evaluation identity | Question set, metric implementation, protocol, or evaluation-stage source changes |

Existing build directories are validated before use, including completion status, expected build
ID, artifact size and SHA-256, vector ordering, embedding dimensions, and index consistency.
Changing only `top_k`, prompt, or generation settings does not rebuild document embeddings, but it
does create a different run identity. Source or protocol changes can make an old partial run
incompatible with `--resume`; use a new run ID instead of modifying historical outputs.

## Tests

```powershell
# Offline test suite
python -m pytest -q

# Optional pinned BGE and FAISS backend test
$env:RUN_FULL_BACKEND_TESTS = "1"
python -m pytest -q tests/test_full_backend.py
Remove-Item Env:RUN_FULL_BACKEND_TESTS
```

The regular suite is offline. The optional backend test loads the pinned BGE model and FAISS but
does not call OpenAI.

## Scope and limitations

- `smoke.yaml` and `qasper_smoke.yaml` verify behavior; they are not quality baselines.
- `questions_v1.jsonl` is a development set used during diagnostics, not unseen test performance.
- `questions_heldout_v1.jsonl` contains five questions and should be run only after freezing the
  configuration; do not tune on its result.
- Automatic exact match and token F1 are limited for long-form answers. Interpret them together
  with source/evidence retrieval, refusal behavior, and saved provider metadata.
- The active baseline has no reranker, BM25/hybrid retrieval, query rewriting, HyDE, agentic loop,
  HNSW, or dynamic top-k.

Additional data documentation:

- [Corpus and data layout](data/README.md)
- [Question-set semantics](data/QUESTIONS_V1.md)
