# Dataset contracts

`data/` contains dataset inputs and prepared corpora. Embeddings, vector indexes, and evaluation
results do not belong here; they are written to `artifacts/` and `outputs/`.

Large downloaded and prepared datasets are Git-ignored. Their manifests, file hashes, row counts,
and protocol checks are the reproducibility boundary.

## QASPER

Prepare once:

```powershell
python scripts/prepare_qasper.py
```

The local dataset is stored under:

```text
data/processed/qasper/hf_dataset/
```

Later runs use `load_from_disk` and do not contact the network. The QASPER loader maps article
units to `PageRecord` values; answer references and evidence labels remain outside retrieval
chunks.

The formal evaluation uses all train/validation/test papers as the retrieval corpus and the fixed
validation slice:

```text
answerable + text-only + extractive + single-evidence
```

Protocol: `qasper_open_corpus_text_extractive_single_evidence_v2`.

## Natural Questions Open + DPR Wikipedia

Expected local DPR source archives:

```text
data/raw/dpr/
  psgs_w100.tsv.gz
  biencoder-nq-dev.json.gz
```

Prepare the canonical local dataset:

```powershell
python scripts/prepare_nq_dpr_wiki.py `
  --wikipedia data/raw/dpr/psgs_w100.tsv.gz `
  --questions data/raw/dpr/biencoder-nq-dev.json.gz `
  --output-dir data/nq_open_dpr_wiki_1m
```

Generated layout:

```text
data/nq_open_dpr_wiki_1m/
  manifest.json
  corpus/
    manifest.json
    passages.jsonl
    passage_ids.txt
  questions/
    manifest.json
    calibration.jsonl
    evaluation.jsonl
```

Canonical counts:

- 1,000,000 passages;
- 500 calibration questions;
- 1,500 evaluation questions;
- up to 50 official hard negatives per selected question;
- complete positive-passage coverage for the selected questions.

Questions, answer aliases, positive passage IDs, and hard negatives stay in the question files.
The corpus loader sees only `corpus/`; answer labels are never copied into passage rows.

The protocol is `nq_open_dpr_wiki_1m_gold_preserving_v1`. The corpus is deliberately
gold-conditioned and is suitable for controlled component and ANN comparisons inside this
pipeline, not absolute comparison with standard NQ full-Wikipedia results.
