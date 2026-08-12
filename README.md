# Query-Adaptive RAG Pipeline

This repository uses BEIR as its only experiment dataset family. The supported
selection is ArguAna, CQADupStack, FiQA-2018, HotpotQA, NFCorpus, Natural
Questions, and Touché-2020. CQADupStack is prepared as twelve independent
forum units.

## Install

For tests:

```powershell
python -m pip install -r requirements/dev.txt
```

For experiments:

```powershell
python -m pip install -r requirements/experiment.txt -c requirements/constraints/verified.txt
```

API credentials belong in the process environment. Inline credentials in YAML
are rejected, and persisted metadata redacts credential-shaped values.

## Prepare BEIR

Prepare every selected official archive, downloading a missing archive into
`data/beir/_archives`:

```powershell
python scripts/prepare_beir.py --dataset all --download
```

To prepare one already-downloaded archive without network access:

```powershell
python scripts/prepare_beir.py --dataset nfcorpus --archive data/beir/_archives/nfcorpus.zip
```

Preparation preserves raw BEIR corpus, query, and qrels IDs and writes verified
manifests under `data/beir/<unit>/`.

## Build reusable offline artifacts

Check one configuration and build all missing dense and SQLite BM25 artifacts:

```powershell
python scripts/check_environment.py --config configs/beir/nfcorpus_dense_top5.yaml
python scripts/build_beir_artifacts.py
```

Use repeated `--dataset` arguments to restrict the artifact build. Completed
encoded corpora, vector builds, and sparse indexes are immutable and reusable
for later questions when the BEIR unit and physical build configuration match.
Query-time settings such as the question text and retrieval depth do not require
re-encoding the corpus.

## Evaluate or query

Run one retrieval-only condition:

```powershell
python scripts/run_beir_eval.py `
  --config configs/beir/nfcorpus_dense_top5.yaml `
  --run-id beir_nfcorpus_dense_example
```

Run the four-condition BEIR suite on selected prepared units:

```powershell
python scripts/run_beir_suite.py --dataset nfcorpus --run-id beir_nfcorpus_suite_example
```

Ask a new question using an existing offline build:

```powershell
python scripts/run_query.py `
  --config configs/beir/nfcorpus_dense_top5.yaml `
  --query "What evidence is available for this question?" `
  --no-log
```

## Project layout

```text
configs/beir/               BEIR experiment and offline-analysis configurations
data/beir/                  prepared BEIR units and local official archives
scripts/                    preparation, build, query, evaluation, and analysis entry points
src/loaders/                verified BEIR corpus/query loader
src/preparers/              deterministic BEIR preparation
src/evaluators/             BEIR retrieval metrics and suite logic
src/indexes/                exact dense index implementations
src/retrievers/             dense and BM25 retrieval implementations
src/persistence/            immutable artifact and run-output validation
artifacts/                   local encoded corpora, builds, and sparse indexes
outputs/                     local evaluation and query outputs
tests/                       contract, unit, and integration coverage
```

Build, run, and evaluation identities remain separate. Artifact descriptors pin
file sizes and SHA-256 values; loaders validate prepared manifests before use;
and completed artifacts are loaded read-only rather than modified in place.

## Tests

```powershell
C:\Users\12442\anaconda3\python.exe -m pytest -q
```

Further details:

- [BEIR dataset contracts](data/README.md)
- [BEIR corpus integration](docs/beir_corpus_integration.md)
- [BEIR experiment results](docs/beir_experiment_results.md)
