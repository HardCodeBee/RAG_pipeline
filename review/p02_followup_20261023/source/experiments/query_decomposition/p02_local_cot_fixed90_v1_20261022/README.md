# Ordinary short-reasoning full90 experiment

The scientific design is in `PROTOCOL.md`; the exact frozen sources and
execution decision are in `PROTOCOL.frozen.json` and `RUN_AUTHORIZATION.json`.
This is a development comparison on an already exposed fixed cohort.

Run from the repository root using Python's UTF-8 mode. Before freezing:

```powershell
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.offline_checks
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.validate_controller_offline
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.score --self-test
```

Root records the scorer receipt and independent review, freezes the exact
sources, and records authorization under the user's autonomous small-research
instruction. Freeze alone does not authorize execution. Do not regenerate
test receipts after freezing; they are part of the frozen provenance.

Queue the approved experiment behind its direct predecessors:

```powershell
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.run run --wait
```

Waiting writes metadata only. It does not create an LLM checkpoint, obtain
an API key or connect to SSH. Both predecessor condition sets must be fully
attempted, have zero requests in flight, and the reference experiment's
original and continuation workers must be closed. The operator supplies the
remote password only when the hidden `Remote password:` prompt appears.
Never put it in a file, command argument or environment variable.

After closure the controller uses one independent remote A6000 allocation.
It waits for the correct worker's READY, checks the first24 structural
contracts and continues all four arms. Returned algorithm failures are kept;
technical held tasks are skipped without blind replay. If the allocation
ends, unstarted conditions remain unstarted and no second job is submitted.

Read metadata without invoking a service:

```powershell
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.run report
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.run predecessors
```

After all conditions are attempted and request activity has stopped:

```powershell
python -X utf8 -B -m experiments.query_decomposition.p02_local_cot_fixed90_v1_20261022.score
```

Scoring is local and read-only against the checkpoint. It writes reports in
this experiment's output folder. Review full90 and ordinary-present84,
dataset results and the fresh N128 paired controls, technical missingness,
algorithm failures, tokens and cost together. This experiment never resumes
the formal500 evaluation or selects a replacement P02 automatically.
