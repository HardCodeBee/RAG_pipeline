# RAG Pipeline Preparation

## 1. Project Snapshot

- Root task files:
  - `任务及要求.txt`: requires a runnable, logged, extensible Naive RAG baseline.
  - `研究方向.txt`: positions the work as RAG systems / RAG pipeline research.
- Papers in `paper/`:
  - `01_AquaPipe A Quality-Aware Pipeline for Knowledge Retrieval.pdf`
  - `02_SAGE_A_Framework_of_Precise_Retrieval_for_RAG.pdf`
  - `03_Gao_RAG_Survey.pdf`
- `RAG_pipeline/` was empty before this preparation note.

## 2. Confirmed Task

The first deliverable should be:

```text
Naive RAG v1: fixed chunking + dense retrieval + fixed top-K + fixed prompt + fixed LLM
```

The goal is not to reproduce SAGE or AquaPipe immediately. The goal is to build a transparent baseline that can later support controlled ablations.

Must include:

1. PDF/document loading
2. Text extraction and cleaning
3. Fixed sentence-preserving chunking
4. Chunk embedding
5. FAISS vector index
6. Query embedding
7. Fixed top-k retrieval
8. Fixed prompt construction
9. Fixed LLM generation
10. JSONL/CSV logging for retrieval, generation, tokens, and latency

Must exclude in v1:

- semantic segmentation
- reranking
- dynamic top-k
- LLM self-feedback
- query rewriting
- HyDE
- agentic RAG
- AquaPipe-style ANNS/prefill pipeline
- training

## 3. Research Direction

This is a RAG systems project. The research question is how pipeline components should be designed, combined, evaluated, and optimized under different queries, document structures, task requirements, and evaluation targets.

The first system should therefore be modular and config-driven, not a one-off script.

## 4. Paper Reading Notes

### 4.1 RAG Survey

Key role for this project: provides the overall taxonomy and baseline template.

- Naive RAG consists of three stages: indexing, retrieval, and generation.
- Indexing includes cleaning/extracting raw files such as PDFs, segmenting text into chunks, embedding chunks, and storing vectors in a vector database.
- Retrieval embeds the user query with the same embedding model, computes similarity, and returns top-k chunks.
- Generation combines the original query and retrieved chunks into a prompt for the LLM.
- Advanced RAG adds pre-retrieval and post-retrieval optimization, such as query optimization, reranking, and context compression.
- Modular RAG generalizes the pipeline with replaceable modules, routing, memory, search, and adaptive/iterative retrieval.
- Evaluation should separate retrieval quality and generation quality:
  - retrieval: hit rate, MRR, NDCG, expected-source hit@k
  - generation: answer accuracy, faithfulness, relevance
  - robustness abilities: noise robustness, negative rejection, information integration, counterfactual robustness

Implication for v1:

- Implement the simple indexing -> retrieval -> generation chain first.
- Log enough metadata to evaluate both retrieval and generation later.
- Keep advanced modules as interfaces, not enabled behavior.

### 4.2 SAGE

Key role for this project: explains why retrieval precision fails and what future ablations should target.

Main diagnosis:

- RAG failures often come from inaccurate retrieval, not only weak LLMs.
- Limitation 1: fixed or naive segmentation can break semantic units, weakening the match between question and chunk.
- Limitation 2: fixed top-k creates a tradeoff:
  - small k can miss target chunks
  - large k can introduce noisy chunks

SAGE components:

- semantic segmentation model for semantically complete chunks
- gradient-based chunk selection after reranking, using score drops to select a dynamic number of chunks
- LLM self-feedback to decide whether retrieved context is excessive or insufficient

Important experimental pattern:

- SAGE compares Naive RAG, Naive RAG + Segmentation, Naive RAG + Selection, Naive RAG + Feedback, and full SAGE.
- This confirms that our v1 must stay simple so later modules can be evaluated by controlled ablation.

Implication for v1:

- The chunker interface must be replaceable.
- The retriever must expose top-k, scores, chunk IDs, and source metadata.
- Logs must preserve retrieved chunks so noisy retrieval and missing retrieval can be diagnosed.
- Do not implement dynamic top-k yet, but make it easy to add later.

### 4.3 AquaPipe

Key role for this project: explains why system latency and index abstraction matter.

Main diagnosis:

- Large RAG knowledge bases push ANNS indexes toward disk-based storage.
- Disk-based ANNS retrieval can dominate RAG response time and TTFT.
- Larger corpora and larger top-k both increase retrieval latency.

AquaPipe components:

- recall-aware ANNS prefetching
- adaptive error correction for wrong early returns
- dynamic pipeline granularity for balancing prefill overlap and GPU efficiency

Metrics emphasized:

- RAG response time / TTFT
- independent ANNS latency
- recall loss
- prefill overhead

Implication for v1:

- Use FAISS `IndexFlatIP` first for transparency.
- Still define an index abstraction with `build`, `search`, `save`, and `load`.
- Log retrieval latency, generation latency, total latency, input tokens, and output tokens from the beginning.
- Do not implement ANNS/prefill overlap in v1.

## 5. Baseline Architecture To Build Next

Recommended structure:

```text
RAG_pipeline/
  README.md
  requirements.txt
  config.yaml
  data/
    raw/
      papers/
    processed/
  src/
    loaders/
      pdf_loader.py
    chunkers/
      fixed_chunker.py
    embedders/
      sbert_embedder.py
    indexes/
      faiss_index.py
    retrievers/
      dense_retriever.py
    generators/
      llm_generator.py
    evaluators/
      logger.py
      metrics.py
    pipeline.py
  scripts/
    build_index.py
    run_query.py
    run_eval.py
  outputs/
    runs/
```

Core data records:

```json
{
  "doc_id": "SAGE_2025",
  "source": "02_SAGE_A_Framework_of_Precise_Retrieval_for_RAG.pdf",
  "page": 3,
  "text": "..."
}
```

```json
{
  "chunk_id": "SAGE_2025_p3_c12",
  "doc_id": "SAGE_2025",
  "source": "02_SAGE_A_Framework_of_Precise_Retrieval_for_RAG.pdf",
  "page_start": 3,
  "page_end": 3,
  "text": "...",
  "token_count": 287
}
```

```json
{
  "rank": 1,
  "chunk_id": "SAGE_2025_p2_c4",
  "score": 0.7821,
  "source": "02_SAGE_A_Framework_of_Precise_Retrieval_for_RAG.pdf",
  "page_start": 2,
  "text": "..."
}
```

## 6. Initial Corpus

Available now:

- AquaPipe
- SAGE
- RAG Survey

Task file suggests eventually adding:

- Lewis 2020 RAG
- Karpukhin 2020 DPR

For the first run, the current three PDFs are enough for a sanity-check corpus.

## 7. Environment Check

Already available:

- Python 3.11.5
- `numpy`
- `pypdf`
- `yaml`
- `pandas`

Missing for full v1:

- `sentence_transformers`
- `faiss`
- `openai`
- `tiktoken`

## 8. Immediate Next Steps

1. Scaffold the directory structure above.
2. Add `requirements.txt` and `config.yaml`.
3. Copy or reference PDFs from `paper/` into `data/raw/papers/`.
4. Implement `pdf_loader.py` with page-level metadata.
5. Implement fixed sentence-preserving chunking.
6. Implement embeddings and FAISS index.
7. Implement retrieval, prompt construction, generation, and full JSONL logging.
8. Add a small `questions.jsonl` with 20-30 hand-written QA examples.
9. Run initial sanity checks:
   - top_k = 1, 3, 5, 10
   - chunk_size = 150, 300, 500
   - log retrieval/generation/total latency

