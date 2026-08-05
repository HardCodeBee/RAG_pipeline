# NQ Open + DPR Wikipedia pipeline plan

## Frozen experimental contract

- Protocol: `nq_open_dpr_wiki_1m_gold_preserving_v1`.
- Source corpus: DPR 2018 `psgs_w100.tsv.gz`.
- Source questions: structured DPR NQ dev retriever file.
- Corpus size: exactly 1,000,000 passages.
- Question split: 500 calibration and 1,500 evaluation questions.
- Selection: all selected-question positives, all same-title passages, at most 50 official hard
  negatives per question, then a seeded deterministic background reservoir.
- Corpus schema: exactly `id`, `title`, and `text`.
- One selected DPR passage maps to one existing `PageRecord` and one existing `ChunkRecord`.
- A DPR passage is never split merely because its BGE subword count exceeds 512; its full text and
  measured token count remain in the chunk, while the pinned embedding model applies its declared
  512-token encoding limit.
- No selection role, question membership, split, positive flag, or new record identity field is
  added.
- Questions/answers remain outside the corpus and index.

## Target architecture

```text
official compressed files
  │
  ├─ deterministic preparation ── corpus + calibration/evaluation + SHA manifests
  │
  └─ corpus loader (streaming)
       └─ pre-segmented chunk adapter
            └─ reusable encoded corpus
                 ├─ chunks.jsonl
                 ├─ chunk_offsets.npy (uint64 byte offsets)
                 └─ embeddings.npy (float32 memmap)
                      ├─ FAISS FlatIP
                      ├─ FAISS HNSW
                      ├─ FAISS IVF-Flat
                      └─ FAISS IVF-PQ
                           └─ QueryPlan
                                ├─ retrieval gate
                                ├─ candidate_k / final_k
                                ├─ ef_search / nprobe
                                └─ optional cross-encoder reranker
                                     └─ existing context / prompt / generator / logging
                                          └─ NQ retrieval + short-answer metrics
```

The public `build_id`, run identity, evaluation identity, `PageRecord`, `ChunkRecord`, and
`SearchHit` contracts remain in place. The encoded-corpus cache is an internal build optimization,
not a new passage identity.

## RAM and index-build controls

The machine has 15.8 GiB RAM and an 8 GiB RTX 4060 Laptop GPU. The design therefore avoids:

- a list of one million `ChunkRecord` objects at build or query time;
- a list of one million passage strings before embedding;
- loading the full embedding matrix into ordinary RAM;
- recomputing BGE embeddings for every ANN index;
- GPU FAISS, which would compete with the embedder for limited VRAM.

The formal BGE document batch is 64. A 128-row batch was faster on ordinary short passages but
reached roughly 7.76/8.0 GiB on the rare 512-token tail and triggered WDDM paging; 64 keeps
long-sequence batches below that cliff and is the stable hardware-specific choice.

Instead:

- corpus and pre-segmented chunks are iterators;
- chunks are written once and indexed with UTF-8 byte offsets;
- document embeddings are encoded in bounded GPU batches into an `open_memmap` array;
- Windows file handles and mappings are flushed and closed before atomic rename;
- FAISS adds vectors in bounded batches;
- IVF/PQ training uses a fixed-seed sample;
- query-time text mapping seeks only the returned vector IDs;
- hard links reuse encoded-corpus files across build directories when possible.

Approximate steady artifact sizes for one million 384-dimensional vectors:

| Artifact | Approximate size |
| --- | ---: |
| float32 embeddings / FlatIP vectors | 1.43 GiB |
| chunk offsets | 7.6 MiB plus NPY header |
| HNSW graph overhead (`M=32`) | hundreds of MiB in addition to vectors |
| IVF-Flat | vectors plus inverted-list overhead |
| IVF-PQ (`m=48`, 8 bit) | roughly 46 MiB of codes plus coarse/index metadata |

Peak RAM still depends on FAISS index type. FlatIP/IVF-Flat/HNSW keep float vectors in the FAISS
process, while IVF-PQ is the memory-oriented alternative. Only one large FAISS build should run at
a time on this machine.

## Identity boundaries

| Change | Encoded-corpus rebuild | Index rebuild | New run identity |
| --- | --- | --- | --- |
| corpus/subset/loader/chunking/document embedding | yes | yes | yes |
| Flat vs HNSW vs IVF/PQ | no | yes | yes |
| HNSW `M` / `ef_construction` | no | yes | yes |
| IVF `nlist`, PQ layout, training seed/sample | no | yes | yes |
| `ef_search`, `nprobe`, `candidate_k`, `final_k` | no | no | yes |
| reranker model/revision | no | no | yes |
| prompt/generator | no | no | yes |
| metric implementation/question split | no | no | new evaluation identity |

Static index parameters stay in the build spec. Search parameters and reranker settings stay in
the run spec. The encoded-corpus spec also fingerprints only the relevant producer dependencies
(for this configuration: NumPy, Transformers, SentenceTransformers, PyTorch/CUDA, and PyYAML).
Index-only factory and config changes are outside the encoded-corpus source boundary, so adding an
ANN implementation does not invalidate cached document embeddings.

## Evaluation matrix

Use FlatIP as the exact reference for every fixed query set. For each ANN configuration report:

- positive passage Hit@k and MRR;
- answer-containing-passage Hit@k;
- answer exact match and token F1;
- ANN overlap/recall relative to exact FlatIP;
- embedding, index-search, mapping, reranking, generation, and total latency;
- index build time and artifact size;
- provider input/output tokens and OpenAI cost outside this repository if required.

Recommended controlled comparisons:

1. FlatIP with fixed `k`.
2. HNSW: fixed build, sweep `ef_search`.
3. IVF-Flat: fixed build, sweep `nprobe`.
4. IVF-PQ: fixed build, sweep `nprobe`, compare memory and quality.
5. Candidate retrieval followed by the pinned cross-encoder, keeping the underlying index fixed.
6. `top_k=0` no-retrieval ablation.
7. Later QueryPlan policies that select retrieval, index, search effort, and k per query.

Do not tune on the 1,500-question evaluation split. Use calibration for component and policy
selection, then freeze config and source before evaluation.

The 1M corpus is transductive by design because the selected questions' positives, same-title
passages, and official hard negatives are retained before background sampling. It is intended for
controlled component and ANN comparisons on this frozen corpus; its absolute scores must not be
compared with standard full-Wikipedia NQ/DPR results.

## Completion and validation gates

- Dataset preparation is deterministic, idempotent, atomic, SHA-verified, and gold-preserving.
- QASPER configs and historical QASPER/NQ manifests remain independently auditable.
- Encoded-corpus artifacts validate row count, dtype, shape, size, SHA, and vector sequence.
- FAISS FlatIP, HNSW, IVF-Flat, and IVF-PQ use real `faiss-cpu`.
- Exact FlatIP is checked against NumPy; ANN indexes are persistence/ID checked and tested against
  FlatIP recall on deterministic fixtures.
- GPU availability is verified with an actual CUDA tensor operation and pinned BGE encoding.
- OpenAI is called through the existing Responses API path when supported by the installed SDK.
- A local key file is ignored by Git, read only at the CLI boundary, never logged, captured in
  generator memory, and removed from the process environment after generator construction.
- Offline regression, optional full-backend tests, a real FAISS build, and a real OpenAI NQ
  integration run must all pass before reporting completion.
