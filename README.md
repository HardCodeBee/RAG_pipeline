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
Backends are always explicit: a failed OpenAI, SentenceTransformer, or FAISS request is reported as
an error and is never silently replaced by another backend.

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
configs/                 baseline, offline smoke, and QASPER smoke configurations
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

## QASPER single-paper smoke

Prepare the reusable local dataset once, then run the QASPER smoke directly from that saved copy:

```powershell
python -m pip install -r requirements/experiment.txt
python scripts/prepare_qasper.py
python scripts/run_qasper_smoke.py --max-questions 3
```

`qasper_smoke.yaml` indexes one validation paper with the same lightweight backends as
`smoke.yaml`. It checks local loading, indexing, generation, and Answer/Evidence F1 on the fixed
`answerable_text_only_extractive_single_evidence_v1` question slice without creating a second
transformed dataset. It is not a full QASPER benchmark: the one-paper limit keeps retrieval within
the paper associated with each question.

For the formal open-corpus evaluation, run:

```powershell
python scripts/run_qasper_eval.py --run-id qasper_text_extractive_v1
```

The formal evaluator uses validation questions only. A question enters the evaluation when at
least one annotator supplied an answerable extractive answer with exactly one non-empty text
evidence unit; only references satisfying all four conditions are scored. All train, validation,
and test papers remain in the retrieval corpus as distractors. The summary records the slice name,
selection counts, evidence hit rate, and Answer F1 conditional on evidence hit or miss.

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

The OpenAI API key is read only from `OPENAI_API_KEY`. Inline credential fields are rejected by
configuration validation and credentials are never written to manifests or run metadata.

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
| Build identity | Corpus, loader, chunking, document embedding, index, or build-stage source changes |
| Run identity | Build, query embedding, top-k, context, prompt, generator, or query-stage source changes |
| Evaluation identity | Questions, metrics version, or evaluator-stage source changes |

Build manifests use `build_spec` as the single source of build inputs and corpus inventory.
Top-level `corpus` and `chunking` sections contain only realized build statistics, while query-only
settings such as `query_prefix` remain part of run identity instead of the built document space.
Build, run, and evaluation use separate explicit source groups, while a full source snapshot digest
is retained for audit. Zero-based vector IDs and one sequence digest bind chunks to vector rows.
`embeddings.npy` is the canonical vector artifact: NumPy searches it directly, while FAISS builds
add only `index.faiss`. Build directories remain immutable and validate artifact size and SHA-256,
vector order, dimensions, and embedding space before query execution. Resume is accepted only when
question, build, run, evaluation, and effective top-k identities match.

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
