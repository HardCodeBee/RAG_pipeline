# BEIR dataset contracts

`data/` contains only prepared BEIR inputs. Embeddings, vector indexes, sparse
indexes, and evaluation results belong under `artifacts/` and `outputs/`.

## Supported families

- ArguAna
- CQADupStack (twelve independently evaluated forum units)
- FiQA-2018
- HotpotQA
- NFCorpus
- Natural Questions
- Touché-2020

Prepare every family with:

```powershell
python scripts/prepare_beir.py --dataset all --download
```

Official archives are cached under `data/beir/_archives/`. Each prepared unit
uses the following verified layout:

```text
data/beir/<unit>/
  manifest.json
  source_manifest.json
  corpus/corpus.jsonl
  queries/queries.jsonl
  qrels/<split>.tsv
```

The preparation protocol is `beir_corpus_queries_v2`. It preserves official
corpus, query, and qrels identifiers, keeps one BEIR corpus row as one retrieval
unit, validates archive identity and row descriptors, and writes content hashes
to the unit manifest.

The loader verifies the manifest chain and artifact hashes before streaming a
corpus. Replacing a corpus, query file, qrels file, or source manifest therefore
changes or invalidates the dataset identity instead of silently reusing an
incompatible offline artifact.

`data/beir/nq` is the official BEIR Natural Questions unit and is part of the
retained BEIR benchmark. It is not a separate dataset integration.
