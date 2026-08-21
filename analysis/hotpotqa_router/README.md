# HotpotQA No-retrieval/BM25/Dense Router Analysis

This package contains the lightweight, shareable boundary of the HotpotQA
mechanism-aware router study. It is intended for reviewing the protocol,
aggregate outcomes, feature design, and Phase 3 learning curve without
publishing raw model calls or large generated artifacts.

## Research question

For a fixed HotpotQA corpus, SQLite BM25 retriever, BGE Dense retriever,
generator, prompt, and context protocol, can deployment-available pre-retrieval
signals choose among:

- `no_context`: run the same generator with an empty context;
- `bm25`: retrieve with SQLite BM25;
- `dense`: retrieve with BGE-small-en-v1.5.

The final research utility is blinded three-level Answer Correctness (AC). The
revised Phase 3 used normalized token F1 only as a user-approved, low-cost
screen before any additional AC judging.

## Staged protocol and current result

| Phase | Question | Result |
|---|---|---|
| 0 | Are the data, artifacts, action contracts, and group split frozen without test leakage? | `GO` |
| 1 | Does correct evidence improve answers, and is the AC judge sufficiently calibrated? | `GO` |
| 2 | Is there stable N/B/D answer-utility oracle headroom over the best fixed action? | `GO` |
| 3 | Can pre-retrieval query/corpus signals recover that headroom? | `EARLY STOP` after the completed 4,800-query point |

Phase 1 observed AC `0.4700` with no context and `0.9400` with gold-page
context. Phase 2 selected Dense as the best fixed dev action at AC `0.7583`;
the N/B/D AC oracle was `0.8333`, giving headroom `+0.0750` with 95% CI
`[0.0489, 0.1050]`.

The revised Phase 3 completed nested 1,200, 2,400, and 4,800-query group-OOF
experiments. Full XGBoost median F1 gain over fixed Dense was respectively
`-0.00696`, `-0.00312`, and `-0.00281`. At 4,800 queries only one of five
training seeds was positive, with a best gain of `+0.00183`, below the
preregistered practical threshold `+0.01`.

The 4,800-query experiment itself is complete. The planned 9,600-query point,
model freeze, fresh-dev evaluation, AC confirmation, and final holdout were not
completed. Partial 9,600-generation state is excluded from every published
metric.

## Phase 3 supervision and model

For each query, three repeated F1 outcomes define utilities for `N`, `B`, and
`D`. Training uses the fixed pairs `(N,B)`, `(N,D)`, and `(B,D)`:

```text
label  = 1[mean_f1(left) > mean_f1(right)]
weight = abs(mean_f1(left) - mean_f1(right))
```

Exact ties are skipped. All pair rows from one query and all queries in the
same Phase 0 information-need group remain in the same cross-validation fold.

The input design combines shared query/corpus features, a pair indicator, and
pair-specific feature interactions. The primary model is a shallow XGBoost
binary preference classifier; regularized logistic regression is the capacity
control. Three pair probabilities are aggregated with a frozen Borda-style
probability sum. This is LTRR-style pairwise supervision, not XGBoost's
`rank:pairwise` objective.

## Online information boundary

Allowed inputs are query text transformations and corpus-static statistics:

- 17 lexical compatibility values based on vocabulary coverage, OOV, IDF/DF,
  rare terms, and numeric/year/capitalized/quoted anchors;
- a 384-dimensional frozen BGE query embedding;
- 13 similarities and density summaries against 64 frozen corpus prototypes.

The generic surface block, including query length and question-word counts,
was excluded. Current-query retrieved documents, retrieval scores/ranks,
qrels, gold evidence, generated answers, F1, and AC are forbidden online
features.

## Included artifacts

- [`config.yaml`](config.yaml): compact frozen protocol and execution boundary;
- [`feature_schema.json`](feature_schema.json): deployed feature blocks without
  raw feature matrices;
- [`phase1_metrics.json`](phase1_metrics.json): evaluator/generator sanity and
  human calibration aggregates;
- [`phase2_metrics.json`](phase2_metrics.json): action means, AC oracle
  headroom, repeat diagnostics, and winner counts;
- [`phase3_learning_curve.csv`](phase3_learning_curve.csv): fixed actions,
  oracle, cross-seed gain range, and tie counts.

Implementation entry points are:

- `scripts/run_router_phase1.py`
- `scripts/run_router_phase2.py`
- `scripts/run_router_phase3.py`
- `scripts/run_router_phase3_train.py`
- `src/evaluators/hotpot_answer.py`

The full research rationale and evidence boundaries are in
`topic.md` and `RAG_pipeline_mechanism_aware_router_codex_plan.md`.

## Interpretation boundary

The completed evidence supports stable answer-level action headroom but does
not establish that the tested pre-retrieval features can exploit it. It does
not prove that query-only routing is impossible, that an AC-trained router
would necessarily fail, or that any lexical, embedding, or prototype block is
individually responsible. Fresh-dev, complete remove-one-block ablations,
counterfactual tests, OOD tests, and final-holdout confirmation were not run.

This package intentionally excludes raw questions, query/group identifiers,
reference answers, predictions, contexts, provider request/response payloads,
state databases, checkpoints, and artifact hash inventories.

