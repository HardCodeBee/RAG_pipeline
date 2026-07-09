# Naive RAG v1

This is the first runnable baseline for the RAG pipeline project.

It intentionally implements a simple, transparent pipeline:

```text
PDFs -> text extraction -> fixed sentence-preserving chunking -> embeddings
-> flat inner-product vector search -> fixed top-k retrieval -> fixed prompt
-> fixed generator -> JSONL/CSV logging
```

## What Is Included

- Page-level PDF loading with source/page metadata.
- Basic text cleaning.
- Sentence-preserving fixed-size chunking.
- Embedding interface:
  - uses `sentence-transformers` when available
  - falls back to a deterministic local hashing embedder when unavailable
- Vector index interface:
  - uses FAISS `IndexFlatIP` when available
  - falls back to exact NumPy inner-product search when unavailable
- Fixed top-k dense retrieval.
- Fixed QA prompt.
- Generator interface:
  - uses OpenAI when configured and available
  - falls back to a local extractive answer generator for offline smoke tests
- Full per-query logging:
  - retrieved chunks
  - similarity scores
  - prompt
  - token counts
  - retrieval/generation/total latency
  - simple retrieval expected-source hit metric

## What Is Intentionally Excluded

These are research extensions for later controlled ablations:

- semantic segmentation
- reranking
- dynamic top-k
- LLM self-feedback
- query rewriting
- HyDE
- agentic RAG
- AquaPipe-style ANNS/prefill pipeline
- training

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The code can run without the optional full-stack dependencies because it has local fallbacks. For the intended full baseline, install `sentence-transformers`, `faiss-cpu`, `openai`, and `tiktoken`.

To use OpenAI generation, set:

```bash
set OPENAI_API_KEY=your_key_here
```

Without this key, `run_query.py` and `run_eval.py` use the local extractive fallback generator.

## Build Index

From this directory:

```bash
python scripts/build_index.py --config config.yaml
```

This reads PDFs from `../paper` and writes:

- `data/processed/chunks.jsonl`
- `data/processed/embeddings.npy`
- `data/processed/faiss.index`
- `data/processed/manifest.json`

## Run One Query

```bash
python scripts/run_query.py --config config.yaml --query "What problem does SAGE solve in RAG?"
```

Optional:

```bash
python scripts/run_query.py --config config.yaml --query "What are the three stages of Naive RAG?" --top-k 3
```

## Run Evaluation

```bash
python scripts/run_eval.py --config config.yaml --questions data/questions.jsonl
```

This writes:

- `outputs/runs/<run_id>_results.jsonl`
- `outputs/runs/<run_id>_summary.csv`

## How This Prepares SAGE-Style Work

The chunker, retriever, and generator are separated so later work can add:

- `semantic_chunker.py`
- reranker modules
- gradient-based dynamic chunk selection
- LLM self-feedback

The current logs preserve retrieved chunk IDs, sources, scores, prompts, and latency, which are needed for noisy retrieval and missing retrieval analysis.

## How This Prepares AquaPipe-Style Work

The index layer is isolated behind `FlatIPIndex`, and every query logs retrieval latency, generation latency, total latency, and token usage. Later work can add FAISS HNSW/IVF or disk-based ANNS implementations without rewriting the full pipeline.

