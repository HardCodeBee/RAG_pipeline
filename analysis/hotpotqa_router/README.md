# HotpotQA BM25/Dense Router experiments

This directory is the local, versioned index for the HotpotQA Router study.
Large reusable arrays remain under `outputs/router/hotpotqa_bd_router_v1/runs/`;
execution databases and checkpoints are not research records.

This Git review package contains the protocols, compact results and source code.
Large local arrays, execution databases and checkpoints are intentionally excluded;
see [the review-package guide](REVIEW_PACKAGE.md) for the exact evidence boundary
and the external-artifact inventory required for full result recomputation.

## Current research status — 2026-09-18

Experimental execution is currently paused by the user. The entries below are
the completed record; they do not authorize further training, retrieval,
generation, feature extraction or candidate search.

The latest [pre-retrieval feature reassessment](pre_retrieval_feature_reassessment_20260918.md)
revisits the user's question about query-feature sufficiency. The historical stop
applies to the tested recipes, not to every possible pre-retrieval representation.
The 207D audit was univariate; the 35D replacement was actually trained. Positive
internal representation evidence has not become a stable independent answer gain.
A single label-free calculation documents rare-term loss in the 500,000-document
sample and a missing-versus-zero ambiguity in two selected cooccurrence features.
That diagnosis does not establish an effect on answer quality. Sections 8.31–8.32
of the continuing review record the completed follow-up: a full 2,322,515-term
bank, common features for 12,889 fixed-role queries, and exactly three model fits.
IDF-mixed corpus alignment gained -0.001057/+0.009954 F1 over Dense on the old/BEIR
consumed development cohorts; increments over the matched base were
-0.000528/+0.009121 and over the coordinate control were -0.001804/+0.008104.
The fixed recipe failed its source-stability and gain gates and is closed.
No bootstrap, paid call, retrieval, or access to the consumed 304 or reserved
895 questions followed. The full bank remains a reusable input resource;
the positive BEIR point estimate does not establish independent answer gains.

The [detailed autonomous-research experiment review](autonomous_research_experiments_20260917.md)
covers E01–E19 and the subsequent measurement, supervision, acquisition,
representation, M6 validation and adaptation studies, with linked evidence for
each question, design, result and conclusion through its stated cutoff.
Subsequent work under the unchanged RAG configuration is recorded in the
[continuing contribution review](research_contribution_review_20260917.md)
records the completed capability-fusion, state-utility and complete-cohort
training-extension screens in Sections 8.11, 8.15 and 8.17. All three failed
their development gates. The latest single fit added 4,165 completely paired
training queries; it trailed the original head by 0.001106 and 0.000819 F1
on the two consumed development cohorts. Sections 8.18–8.19 record the
subsequent learned term-aggregation hypothesis, its overlap with prior work,
and the completed five-arm experiment: 5 convex fits and 9,000 learned updates.
Correct-IDF attention gained -0.005513/+0.002728 F1 over Dense on the two
development cohorts and failed the fixed-pooling and shuffled-IDF comparisons.
No quality candidate passed. Training remained unsettled at the declared cap;
the recipe is closed without rejecting all possible learned pooling methods.
The stable-gain goal remains unmet. Sections 8.20–8.22 record the completed
external development measurement under the same environment: 304 questions in
256 groups, 1,824 successful answers, zero generation failures or retries.
User-approved collection stayed within US$2: reported tokens at frozen prices
cost US$0.4017544, with per-attempt conservative rounding recorded as US$0.402515.
Fixed-IDF F gained -0.001645 F1 versus Dense (97.5% CI [-0.011397, +0.004381])
and failed its gate; historical M6 matched Dense on this sample. The answer-informed
leave-one-repeat-out diagnostic gained +0.030887 (97.5% CI [+0.013985, +0.051009])
and passed its separate diagnostic gate. This supports investigating one query-only
mechanism, not a deployable-method gain. Another 895 questions in 750 groups remain
unused for inference, retrieval and answer collection. The evidence concerns the
conditional 2Wiki source; original-HotpotQA confirmation is still missing.

Sections 8.23–8.24 record the subsequent fixed local-residual fusion experiment
on these consumed 304 questions: four group-OOF folds, three inner calibration
folds, 16 neighbors, and matched group-block shuffling. Local correction gained
-0.006203 F1 versus Dense/global calibration and -0.004049 versus the shuffled
control. It failed its point gates; no bootstrap, parameter expansion, new
encoder inference, retrieval or paid generation followed. The recipe is closed.
Sections 8.25–8.26 record the subsequent entity-proxy masked dual-view experiment:
608 frozen-encoder inputs and 20 group-OOF linear fits, with original-only,
masked-only, matched-position-mask and duplicate-original controls. The primary
dual view gained -0.004266 F1 versus Dense, +0.000117 versus original-only,
-0.003859 versus masked-only, -0.002921 versus matched masking, and -0.001253
versus duplicated original features. It failed its point gates; bootstrap draws,
new retrievals and new answer calls were all zero. The fixed recipe is closed.
Section 8.27 ends this candidate-search round on the consumed 304 questions.
The next step is a bounded method and independent-confirmation design, with no
new fit on these 304 questions and no access to the reserved 895 questions.
No effective method or publishable stable-gain claim has been established.

Sections 8.28–8.29 select static term-conditioned BM25-weighted Dense moments
for further input design. A corpus-only pilot used 32 terms, 4,951 postings and
4,950 existing document vectors, taking 4.3243 seconds and saving 97,295 bytes.
Three synthetic checks established lookup composition, information beyond
separate lexical/Dense marginals, and a same-moments/different-top1 limitation.
No query, label, encoder, Router fit or answer call was used. This establishes
bounded construction feasibility, not answer-quality gains or vocabulary coverage.
Sections 8.31–8.32 subsequently completed full-vocabulary coverage, the matching
Dense query interface, and one fixed development experiment; that recipe failed.
The consumed-304 search remains closed and the reserved 895 remain untouched.

The [complete progress report and next steps](research_progress_report_20260915.md)
summarizes the completed study, the current evidence limits, and the conditional
plan following the completed Student, compensation, trained-encoder readout and
direct M6-output adaptation experiments.

The [resumption review](research_resumption_20260915.md) reconciles the later
independent study with the historical phases below. The complete 5,209-query
M6 validation found +0.003704 F1 versus Dense (97.5% interval
[-0.001379, +0.008972]); the predefined gain standard was not met.
See the [validation conclusion](m6_beir_validation_conclusion_20260915.md).

The subsequent [fixed-M6 objective and threshold experiment](m6_objective_readout_results_20260915.md)
and [fixed-M6 nonlinear Fourier extension](m6_fourier_readout_results_20260915.md)
have completed and passed separate numerical checks. Both failed their
prespecified improvement gates. They used the consumed old pool; they are not
additional independent-source validations. The stable-gain research goal remains
unmet. The linked external
[research progress record](../../../work/router_research/validation_fresh4000_e03b_e1_v1/progress.json)
has a stale running probe node: its final synchronization was rejected by
automatic approval review because the account usage limit was reached.
The completed results and separate checks linked here establish the actual state.

The next [matched M6/probe experiment](m6_probe_readout_results_20260915.md)
has also completed and passed separate numerical checks: real post-retrieval
probe inputs improved F1 by +0.009581 over M6 and exceeded both fixed shuffled
controls. Its transfer-investigation gate passed. This is consumed-pool,
post-retrieval diagnostic evidence. The subsequent
[strict OOF M6 Student experiment](m6_student_transfer_results_20260915.md)
completed 30 Teacher and 15 Student fits and passed independent numerical checks.
Probe Student minus Pre Student was -0.000514 (99% interval
[-0.001734, +0.000697]); Probe Student minus Direct was -0.000828
([-0.002730, +0.001083]). Neither transfer nor candidate-preparation gate passed;
the fixed alpha=.5/lambda=.001 recipe is closed.
The [distillation rationale](m6_distillation_rationale_20260915.md) records the
pre-experiment theory. The [completed test-group audit](m6_test_group_identity_results_20260915.md)
found 236 of 7,405 queries sharing historical groups. The completed
[selection-history audit](m6_test_selection_history_audit_20260915.md) found that
the test source informed retrieval-policy and parameter exploration. Its 7,169
group-disjoint queries cannot be described as a fresh confirmation cohort.

The [fixed-artifact localization](m6_student_localization_results_20260915.md)
confirmed that probe target changes reached Student heads and changed 199 actions,
and identified a common regularization shrinkage component. A subsequent
[single analytic compensation experiment](m6_student_compensation_results_20260915.md)
completed 15 new fits at Student lambda=.0005, with matched Direct controls.
Compensation brought Pre Student close to the original Direct head, but ProbeC
minus PreC was -0.000364 and ProbeC minus original Direct was -0.000387; both
improvement gates failed. Independent checks passed. This mechanism branch is
closed. The subsequent [representation and supervision coverage audit](representation_supervision_gap_audit_20260915.md)
confirmed that full fine-tuning, LP-to-FT, support supervision and query-relation
training were already completed. It identified an untested crossing of the
completed H checkpoint with the mean-layer6 readout. The [fixed four-condition
experiment](m6_trained_readout_results_20260915.md) completed 15 readout fits and
passed independent checks with two documented storage-dtype adaptations.
H6 minus original M6 was -0.004059 (99 1/6% interval [-0.008474, +0.000388]);
both gates failed. The fixed H-checkpoint readout branch is closed, with no
new encoder training or API calls. The subsequent [direct M6-output adaptation](m6_output_adaptation_results_20260917.md)
has now completed all five folds, ten trajectories and 30,720 logical optimizer
steps. Its interrupted prefix was recovered under the frozen
[recovery protocol](m6_output_adaptation_resume_plan_20260917.md): all 38 original
files were preserved and 2,304 saved steps replayed exactly before continuation.
Formal session81144, numerical checker59652 and the recovery audit all exited 0.

Adaptation A minus matched head-only C was -0.001026 (98.75% interval
[-0.003258, +0.001183]); A minus original M6 was -0.000379
([-0.002892, +0.002057]). A minus Dense was +0.004417
([+0.000018, +0.008672]). Both predefined gates failed. Training loss decreased
substantially, but the fixed recipe did not establish the required incremental
answer utility. This consumed-pool branch is closed; no new candidate or runtime
integration followed. No retrieval, generation or API calls were added.

The [fixed-model repeat-weight diagnostic](m6_repeat_weight_diagnostic_results_20260917.md)
has completed and passed independent numerical checks. It identifies an extra
local gradient distinct from normalization-induced regularization, but adds no
trained candidate or independent utility evidence. This diagnostic is archived.
Further old-pool candidate searches are paused; cross9 training does not follow
automatically. The [contribution and literature review](research_contribution_review_20260917.md)
sets the next priority: a falsifiable central claim and a feasible independent
validation design. Current evidence is insufficient for a new-method claim.
The [progress report](research_progress_report_20260915.md#73-官方隐藏test待核实资源尚未认证)
also records an official hidden-test metadata precheck. Per-query paired scores,
group mapping, evaluation access and source eligibility remain unconfirmed.

## Historical phase conclusions

| Phase | Result status | Decision | Deployable result |
|---|---|---|---|
| 0–2.6 | complete | `STOP_BEFORE_AC_LABELS` | no |
| 2.7 | complete | `STOP_QUERY_ONLY_V1` | no |
| 2.8 | complete | `DO_NOT_OPEN_CONFIRMATION_POOL` | no |
| 2.8b | complete; natural-2000 consumed for candidate selection | `NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY` | no |
| 2.11 | complete; old30/new35 internal diagnostic | `NO_CLEAR_INTERNAL_DIFFERENCE` | no |

The current runtime still selects one configured BM25 or Dense retriever. These
offline experiments do not add an adaptive Router to runtime.

Phase 2.8b is not a final test. Its 2,000-query natural-distribution sample was
consumed once for candidate selection and must not be reused for threshold,
feature, or model tuning. Its final holdout was unopened at that phase;
later source use and consumption are recorded in the resumption review.

## Local layout

```text
analysis/hotpotqa_router/
├── README.md
├── registry.yaml
├── phases/
│   ├── phase00_26/
│   ├── phase27/
│   ├── phase28/
│   └── phase28b/
└── auxiliary/
    ├── contriever/
    ├── nq_dpr_probe/
    └── retrieval_context/
```

Each phase directory contains:

- `config.yaml`: effective frozen protocol;
- `conclusion.md`: human-readable research record;
- `results/`: compact metrics, decisions, comparisons, and validation data.

`registry.yaml` is the global phase and artifact index. It distinguishes
`spec_status`, `result_status`, consumed data, deployability, dependencies, and
the next logical gate, and links to the later research progress record.

Historical frozen manifests keep their original path strings for evidentiary
fidelity; use `registry.yaml` for the current local location.

## Canonical local data

The paths are intentionally left in their original frozen run directories so
that stored hashes and Phase 2.9/2.10 dependencies remain valid.

- Phase 2.7 base: `phase27_model_audit_9600_v1/snapshot/`;
- Phase 2.8 sources: exported labels/outcomes, new features, and training views;
- Phase 2.8b natural-2000: frozen features, predictions, query metrics, and
  prediction/open manifests.

SQLite state, WAL/SHM files, Python caches, and candidate-by-seed task shards
are execution material rather than canonical experiment records.

## Local commands

```powershell
C:\Users\12442\anaconda3\python.exe -X utf8 scripts\router_workspace.py status
C:\Users\12442\anaconda3\python.exe -X utf8 scripts\router_workspace.py verify
C:\Users\12442\anaconda3\python.exe -X utf8 scripts\router_workspace.py verify phase28b
```

To deterministically rebuild the Phase 2.8 T0/T1/T2 views from the retained
Phase 2.7 snapshot and Phase 2.8 expansion sources:

```powershell
C:\Users\12442\anaconda3\python.exe -X utf8 scripts\router_workspace.py rebuild phase28_views
```

The rebuild command makes no provider calls. It is not run implicitly by
`status` or `verify`.

## Main execution entry points

- `scripts/run_router_phase27_model_audit.py`
- `scripts/run_router_phase27_privileged_teacher.py`
- `scripts/run_router_phase28_query_expansion.py`
- `scripts/build_router_phase28_training_views.py`
- `scripts/summarize_router_phase28_training.py`
- `scripts/run_router_phase28b_holdout.py`
- `scripts/run_router_phase31_feature_comparison.py`

Shared cross-stage imports are exposed through `src/router_experiments/` so
new stages do not need to import private helpers from older stage scripts.

## Evidence boundaries

- Oracle headroom proves action heterogeneity, not a deployable Router.
- Phase 2.8 T1/T2 gains are winner-balanced OOF diagnostics, not natural-
  distribution gains.
- Phase 2.8b natural-2000 is consumed candidate-selection evidence.
- Post-retrieval probe and gold/qrels features are diagnostics and are not
  pre-retrieval deployment features.
- Answer F1, Answer Correctness, retrieval metrics, and evidence diagnostics
  remain separate estimands.

The predefined old30/new35 comparison is recorded in [Phase 2.11](phases/phase31/conclusion.md).
It reuses the original 9,600-query cohort and is post-selection internal evidence; it does not reopen natural-2000.
