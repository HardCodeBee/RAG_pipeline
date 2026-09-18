# Autonomous Router research review package

## Purpose and status

This directory is the Git review record for the autonomous HotpotQA
BM25-versus-Dense Router research. It contains the research questions,
predeclared or frozen protocols where available, source code, compact result
tables, numerical checks, and the decision record. Experimental execution is
currently `paused_by_user`; this package does not claim a deployable Router or
a confirmed answer-quality improvement.

Start with:

1. [README.md](README.md) for the current status and historical phases.
2. [registry.yaml](registry.yaml) for phase status, consumed-source boundaries
   and artifact references.
3. [autonomous_research_experiments_20260917.md](autonomous_research_experiments_20260917.md)
   for E01--E19 and later branches.
4. [research_contribution_review_20260917.md](research_contribution_review_20260917.md)
   for the contribution assessment and later closed recipes.
5. [research_resumption_20260915.md](research_resumption_20260915.md) and
   [research_progress_report_20260915.md](research_progress_report_20260915.md)
   for the independent-validation record.

## Included in Git

- `phases/`: frozen phase configurations, conclusions and compact summaries;
- `m6_*.md` and `m6_*.json`: M6 protocols, results, checks and source audits;
- `m6_objective_readout_v1/`: compact fold outputs and frozen predictions;
- `figures/`: figures used by the review records;
- `scripts/` at repository root: experiment, recomputation and audit source;
- `researcher.md` at repository root: the autonomous-research operating method.

The Phase 2.11 old30-versus-new35 comparison is fully represented by its
configuration, conclusion and compact result files under `phases/phase31/`.

## Evidence not present in Git

The following local artifact roots are excluded from normal Git because they
contain reusable arrays, model checkpoints, caches and execution databases:

| Local root | Approximate size at packaging | Review role |
|---|---:|---|
| `../work/router_research/` | 4.91 GiB | Per-experiment data, answer outcomes, frozen actions, independent-validation ledgers and selected predictions |
| `outputs/router/hotpotqa_bd_router_v1/runs/` | 4.94 GiB | Generated feature arrays, run configurations, checkpoints, caches and intermediate outputs |

Consequently, links in historical reports that begin with `../../../work/` are
local evidence pointers, not remote links. They are not accessible from a
GitHub checkout.

## Required external audit release

A full remote audit release must include, with SHA-256 manifests, at least:

1. `validation_fresh4000_e03b_e1_v1/`: protocol, questions, frozen actions,
   features, final answer outcomes, evaluation and integrity checks.
2. `validation_pool12566_NI_v1/`: the corresponding protocol, actions,
   features, outcomes, evaluation and checks.
3. `m6_beir_validation_v1/`: the corresponding protocol, actions, answer
   outcomes, evaluation, freeze record and collector-ledger checks.
4. `phase31_feature_comparison_v1/`: effective configuration, feature
   extraction record, old-model reproduction record, OOF predictions and
   per-query comparison table.
5. `m6_output_adaptation_v1/`: protocol, result summary, bootstrap arrays,
   frozen predictions, fold test arrays, fit/calibration arrays and numerical
   checks.
6. `static_pairing_bank_v1/`: protocol, input checks, summary and the
   statistics needed by the published calculation.

Do not include repeated epoch checkpoints, corpus caches, SQLite WAL/SHM
files, locks, credentials or local environments in the Git repository. A
release manifest must distinguish these reconstructable execution artifacts
from the frozen inputs and outputs needed to recompute each reported result.

## Reproduction boundary

The repository provides the implementation and compact evidence needed to
inspect the claims and recomputation logic. Exact end-to-end reproduction also
requires the public dataset versions, the configured retrievers and generators,
and the external audit release above. Any future archive should preserve source
eligibility and consumption status; it must not portray consumed development
cohorts as fresh confirmation data.
