# Data and corpus

This directory contains the versioned PDF corpus and the question sets used by the baseline.
Generated chunks, embeddings, and indexes do not belong here; the current pipeline writes them to
the ignored `artifacts/<build_id>/` directory.

## Corpus

`corpus/` contains the exact five PDFs used by the reported experiment:

1. AquaPipe: *A Quality-Aware Pipeline for Knowledge Retrieval*
2. SAGE: *A Framework of Precise Retrieval for RAG*
3. Gao et al.: RAG survey
4. Lewis et al. (2020): Retrieval-Augmented Generation
5. Karpukhin et al. (2020): Dense Passage Retrieval

The machine-readable file names, byte sizes, and SHA-256 values are frozen in
[`corpus_manifest.json`](corpus_manifest.json). Build manifests independently recompute the same
per-file hashes and an aggregate corpus identity before indexing.

Evidence annotations use 1-based physical PDF pages produced by `pypdf`, not the page numbers
printed inside a paper.

## Development questions

`questions_v1.jsonl` contains 24 questions:

- 22 answerable questions;
- 2 deliberately unanswerable questions;
- single-paper, multi-evidence, and cross-paper cases;
- page-level evidence spanning all five PDFs.

Each line is one UTF-8 JSON object. Important fields are:

- `question_id`: stable unique identifier;
- `question`: query text;
- `gold_answer`: manually written reference answer;
- `answerable`: whether the corpus contains enough evidence;
- `question_type`: evidence structure or unanswerable type;
- `expected_sources`: expected document-level coverage;
- `evidence`: required page-level claims;
- `scope_sources` and `unanswerable_reason`: audit information for unanswerable cases.

Evidence claims are combined with AND semantics. Within one claim, `alternatives` represents
multiple equally valid page ranges and is evaluated with OR semantics. This prevents a valid
supporting passage from being scored as a miss merely because another page states the same claim.

This set was used for both baseline evaluation and initial top-k/chunk-size diagnostics. Its
reported numbers are therefore development results, not held-out performance.

## Held-out questions

`questions_heldout_v1.jsonl` contains five additional questions, one per paper. It has not been
executed. Run it only after freezing the baseline configuration, and do not use its result for
another tuning cycle.

Further annotation details are documented in [`QUESTIONS_V1.md`](QUESTIONS_V1.md).

## QASPER local cache

Install `requirements/experiment.txt`, then cache the official Hugging Face DatasetDict once:

```powershell
python scripts/prepare_qasper.py
```

The first run loads Hugging Face's Parquet conversion of QASPER and saves all three logical splits
under `data/processed/qasper/hf_dataset/`. Later runs use `load_from_disk` and do not contact the
network. Use `train` for training or examples, `validation` for development, and leave `test`
untouched until the pipeline configuration is frozen. This stage does not transform QASPER into
pipeline records or separate its labels; that belongs to the loader.
