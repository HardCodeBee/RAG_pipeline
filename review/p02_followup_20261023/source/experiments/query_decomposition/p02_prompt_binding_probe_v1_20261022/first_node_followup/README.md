# No-history first-ordinary followup

This is an isolated negative control for the fixed-input binding probe.
Three of the four native-F1 harmed minimal4 cases first changed at an ordinary
node with no previous answers. The fourth changed at an ordinary node with
history. The reviewed failures include two locally reasonable abstentions with
subsequent evidence-coverage changes, one final exclusion failure, and one
same-entity name extension penalized by the metric; they are not four newly
established factual errors. See the preserved sibling
`minimal4_loss_review/REVIEW.md` and `CASE_EVIDENCE.json` in the output tree.

The followup selects five first ordinary nodes from each dataset in the same
already exposed fixed90, ordered by
`SHA256('20261022|case_id|first_ordinary_no_history')`. Every selected node has
`step_id=1` and an empty actual history. Selection does not read final scores,
gold answers, or error labels. All 15 source hashes match the already public
manifest with Git blob SHA `06c67be3c2091ef4066366b8c0c87c2b4d1ece17`.

The control uses that exact original case/node request ID and full payload,
including complete stop-ended content and usage. Two candidates respectively
delete the reference-resolution sentence or replace just that sentence with
the descriptive binding rule. All other examples, identifiers, empty-history
wording, reader text/order, and native128 output allowance are fixed. The
few-shot examples still contain their historical references. Any effect here
could involve global instruction/example priming or generation variation; it
does not automatically show binding to actual previous answers.

Commands from the repository root:

```powershell
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.first_node_followup dry-run
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.first_node_followup run --limit 2
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.first_node_followup run
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.first_node_followup report
```

Only `run` can submit. It reuses the parent runner in its own CLI process with
an independent output directory, checkpoint, ledger namespace, report and hold
files under `first_node_followup/`. Parent source files, protocol and running
processes are not changed. The shared provider gate, complete-output contract,
bounded five-attempt recovery and unknown-charge preservation remain in force.
There is no formatter or downstream retrieval/goal/final execution, and no
native final F1/EM result. The main evaluation remains paused.
