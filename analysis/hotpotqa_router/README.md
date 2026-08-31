# HotpotQA BM25/Dense Router experiments

This directory is the local, versioned index for the HotpotQA Router study.
Large reusable arrays remain under `outputs/router/hotpotqa_bd_router_v1/runs/`;
execution databases and checkpoints are not research records.

No remote artifact store, DVC, Git LFS, or cloud synchronization is required.

## Current conclusion

| Phase | Result status | Decision | Deployable result |
|---|---|---|---|
| 0–2.6 | complete | `STOP_BEFORE_AC_LABELS` | no |
| 2.7 | complete | `STOP_QUERY_ONLY_V1` | no |
| 2.8 | complete | `DO_NOT_OPEN_CONFIRMATION_POOL` | no |
| 2.8b | complete; natural-2000 consumed for candidate selection | `NO_QUALIFIED_CANDIDATE_DIAGNOSTIC_TOP2_ONLY` | no |

The current runtime still selects one configured BM25 or Dense retriever. These
offline experiments do not add an adaptive Router to runtime.

Phase 2.8b is not a final test. Its 2,000-query natural-distribution sample was
consumed once for candidate selection and must not be reused for threshold,
feature, or model tuning. The official final holdout remains unopened.

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

`registry.yaml` is the only global phase and artifact index. It distinguishes
`spec_status`, `result_status`, consumed data, deployability, dependencies, and
the next logical gate.

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
