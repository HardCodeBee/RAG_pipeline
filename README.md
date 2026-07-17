# Reproducible Naive RAG Baseline

This repository provides a compact, research-oriented Retrieval-Augmented Generation (RAG)
baseline over five papers. It is intentionally narrow: the goal is to make retrieval evidence,
generation quality, and experiment identity easy to inspect and reproduce without a general
plugin framework.

The fixed baseline is:

```text
PDF corpus
→ minimal text cleaning
→ sentence-preserving fixed-size chunks
→ BAAI/bge-small-en-v1.5 embeddings
→ FAISS FlatIP dense retrieval
→ fixed top-k context and prompt
→ gpt-4o-mini generation
→ evidence-aware evaluation
```

The offline smoke configuration follows the same chain but uses deterministic hashing embeddings,
a NumPy index, and an extractive generator. It requires no API key or model download.

## Baseline specification

| Stage | Formal baseline |
| --- | --- |
| Corpus | 5 included research papers, 90 extractable PDF pages |
| Chunking | Sentence-preserving, 300-token limit, 50-token overlap budget |
| Embedding | `BAAI/bge-small-en-v1.5`, pinned revision, normalized 384-d vectors |
| Index | FAISS FlatIP, exact search |
| Retrieval | Dense top-5 |
| Prompt | `fixed_qa_v1` |
| Generator | `gpt-4o-mini`, temperature 0, maximum 512 output tokens |
| Evaluation | Source/evidence retrieval metrics, answer overlap, answerability/refusal, latency, and token usage |

No reranker, query rewriting, HyDE, agentic loop, BM25, HNSW, or dynamic top-k is part of this
baseline.

## Repository layout

```text
configs/                 two active runtime configurations
data/corpus/             the five versioned PDF inputs
data/questions_v1.jsonl  development/evaluation questions
data/questions_heldout_v1.jsonl
requirements/            base, experiment, development, and verified constraints
scripts/                 build, query, evaluation, environment, and metric entry points
src/                     pipeline implementation
tests/                   behavior, identity, artifact, metric, and backend tests
```

Generated `artifacts/`, raw `outputs/`, caches, and local credentials are intentionally excluded
from Git. Every generated build and run records hashes that bind it to the corpus, effective
configuration, relevant source files, question set, and evaluation definition.

## Quick start: offline smoke configuration

Python 3.11 is the verified environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/dev.txt -c requirements/constraints/verified.txt

python scripts/build_index.py --config configs/smoke.yaml
python scripts/run_query.py --config configs/smoke.yaml --query "What are the stages of Naive RAG?"
python scripts/run_eval.py --config configs/smoke.yaml --questions data/questions_v1.jsonl --run-id smoke_demo
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`; the Python commands
are unchanged.

## Reproduce the formal baseline

Install the full experiment dependencies and check the local environment:

```powershell
python -m pip install -r requirements/experiment.txt -r requirements/dev.txt -c requirements/constraints/verified.txt
$env:OPENAI_API_KEY = "your-key"
python scripts/check_environment.py --config configs/baseline.yaml --strict-credentials
```

Build and evaluate:

```powershell
python scripts/build_index.py --config configs/baseline.yaml
python scripts/run_eval.py --config configs/baseline.yaml --questions data/questions_v1.jsonl --run-id baseline_v1_reproduction
```

The API key may be supplied through `generation.api_key` or `OPENAI_API_KEY`. By project policy,
the configured/effective key is recorded in plaintext in manifests and run metadata. Credentials do
not enter scientific identity hashes, so changing only a key does not force an index rebuild.

To recompute metrics from a completed run without repeating retrieval or generation:

```powershell
python scripts/recompute_metrics.py --source-run-dir outputs/<source_run_id> --run-id <reanalyzed_run_id>
```

An external LLM API is not guaranteed to return byte-identical text on a future call. The saved
answer, actual provider model, response ID, prompt hash, and token usage are treated as the
authoritative generation observation.

## Reproducibility identities

| Identity | Changes when |
| --- | --- |
| Build identity | Corpus, loader, chunking, document embedding, index, or active Python source changes |
| Run identity | Build, query embedding, top-k, context, prompt, generator, or active Python source changes |
| Evaluation identity | Questions, metrics version, or active Python source changes |

Schema v6 uses one canonical embedding-space record in each build manifest; query-only settings such
as `query_prefix` remain part of run identity instead of the built document space. The schema keeps
the single source digest over `src/**/*.py` and `scripts/*.py`, zero-based vector IDs, and one sequence
digest binding the chunk-to-index mapping. Build directories remain immutable and validate artifact
size and SHA-256, vector order, dimensions, and embedding space before query execution.
Resume is accepted only when schema, question, build, run, evaluation, source, and effective top-k
identities match. Schema-v5 and older artifacts remain historical records and require their matching code
revision; rebuild them before use with schema-v6 code.

## Tests

```powershell
python -m pytest -q

$env:RUN_FULL_BACKEND_TESTS = "1"
python -m pytest -q tests/test_full_backend.py
```

The regular suite is fully offline. The full-backend gate loads the pinned BGE tokenizer/model and
FAISS but does not call OpenAI. This repository does not use GitHub Actions; validation is run
locally before a release.

## Scope and limitations

- The corpus contains only hundreds of chunks. It cannot support conclusions about HNSW, disk
  ANNS, TTFT overlap, or AquaPipe-scale systems behavior.
- `questions_v1.jsonl` is a development set, not an unseen benchmark.
- `questions_heldout_v1.jsonl` has not been run and should only be used after the baseline is
  frozen.
- Automatic exact-match and token-overlap metrics are weak for long-form RAG answers. Interpret
  them together with page-level evidence retrieval, refusal behavior, and saved provider metadata.

Data documentation:

- [Corpus and data layout](data/README.md)
- [Question-set semantics](data/QUESTIONS_V1.md)
