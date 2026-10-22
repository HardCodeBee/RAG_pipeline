# Ordinary few-shot representation probe E0 / E1 / E2

This is an isolated fixed-reader component development experiment. Main formal evaluation remains paused. Preparation and validation make no provider requests, perform no retrieval or GPU work, and do not read references, native scores, diagnostic correctness labels or new candidate outputs.

The30 nodes are exactly the prior frozen15 with-history ordinary nodes and15 no-history first ordinary nodes, with5 per group/dataset. Preserve original within-group order and interleave the two groups. No resampling, replacement or selection by performance is permitted.

All three arms preserve the same actual original/current question, history, reader packet/order, four demonstration original questions, answers, labels and order, main instructions, model settings and128-token output limit. Only these two demonstration CURRENT subquestions may change:

| Arm | Example1 current subquestion | Example3 current subquestion |
|---|---|---|
| E0 | What is the length of #1? | How many species does #1 contain other than Callicoma serratifolia? |
| E1 | What is the length of Colorado River? | How many species does Callicoma contain other than Callicoma serratifolia? |
| E2 | What is the length of the river into which Pack Creek runs after it goes through the Spanish Valley? | How many species does the genus that contains Callicoma serratifolia contain other than Callicoma serratifolia? |

E1 entities are already in each demonstration's prior answer. E2 expresses those same demonstration relations; the river description is literally present in its demonstration original question. The genus description verbalizes the existing prior question/answer. Neither introduces gold information about any actual tested query. Unlike minimal4, this experiment never edits demonstration original questions or their task/scope. The original Resolve rule remains intact in every arm.

E0 is a **fresh synchronized original-prompt condition**, run in this experiment's separate checkpoint with `StrictAsk(seed=None)`. Complete historical receipts are separate references, never a substitute for E0 new inference. All90 payloads must be unique within this checkpoint. Restarts reuse only this experiment's saved complete/terminal receipts; never regenerate successful outputs or choose the best result.

Within each node, arm order rotates by SHA256(`20261022|case_id|ordinary_fewshot_order`) modulo3. Nodes from the same original case share the rotation. Freeze the resulting90-task order. These rotations reduce order imbalance; they do not establish absence of service drift or estimate multiple same-prompt draws. E0 versus the independent historical response provides one temporal/noise comparison, not an unrestricted reroll policy.

Use the previously frozen shared runtime SHA256 `5c6c5c4914e995bce4a995ebb60ff9acb29517c5e5d77118bd2ac2413091b98e`: shared serial provider gate, minimum3-second submission-start interval, SDK/HTTP retries0, maximum5 actual technical attempts including the first across restarts, persistent reservations and retired receipts. Unknown responses with any responseID/content/raw response/usage remain held. Only previously delegated bounded no-receipt recovery is allowed. There is no fixed legacy budget cap.

Length, empty, refusal and content-filter receipts are terminal algorithm outcomes and remain in the90-condition accounting; do not resample or rerun them. Missing/invalid usage, identity mismatch and unresolved requests remain technical holds, separately enumerated. A failed technical condition stays missing, not silently zero. Stage expense monitoring compares known cost against the paired historical component: single3x and recent20 mean1.5x alerts; unknown usage is not zero. This is not full-query cost benchmarking.

Report coverage, terminal status, text differences versus fresh E0 and independent historical reference, tokens and estimated expense. Later semantic review must separate correct facts, necessary information, roles and abstention. Do not report final F1/EM, infer final quality from text changes, or claim this isolates the literal `#` character or proves a binding mechanism.

The local public-source proof reuses the previously live-verified public manifest identity at commit `ccc12b6afe158a00790f94826d8c5c58e6f33275`, then rechecks all30 source hashes and same-node original/current questions, history, reader, prompt and original requestID. Only public source-derived task fields and these nonsensitive demonstration substitutions enter requests. The proof does not claim a new remote fetch.

Offline commands:

```powershell
python -B -m experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run prepare
python -B -m experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.validate_offline
python -B -m experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run dry-run
```

Root must first record this concrete decision in the shared plan and create this experiment's `RUN_AUTHORIZATION.json` with `approved=true`, matching experiment and decision ID. The runner checks that record and the main-evaluation paused state. Preparation itself does not authorize execution. Credentials are read only inside the existing temporary key/DNS context; no credential is copied into this directory.

Prepared execution command for root after authorization review:

```powershell
python -B -m experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run run --limit 3
python -B -m experiments.query_decomposition.p02_ordinary_fewshot_probe_v1_20261022.run run
```

Both commands continue the same checkpoint and90-task order. `--limit` limits newly processed conditions, including technical held ones; it does not reset attempt counts. A `STOP_REQUESTED` file in the isolated output directory stops between conditions or before provider submission. The read-only report path never constructs Cache.
