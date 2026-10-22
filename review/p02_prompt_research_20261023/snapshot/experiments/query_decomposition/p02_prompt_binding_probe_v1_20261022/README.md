# Fixed-input prompt-binding probes

This is a development component probe, not a complete-RAG or formal evaluation.
The 30 frozen source nodes contain 15 ordinary nodes with history and 15 goal
nodes, balanced at five of each role per dataset. Selection uses seed20261022
and SHA256 ordering, without scores or reference answers.

Every control comes from that original fixed90 case/node's request ID and exact
complete payload. The two candidates change only the reference-resolution
sentence: deletion or descriptive entity/role binding. History, evidence,
examples, model, and the native128 output allowance stay fixed. No formatter,
retrieval, final-answer generation, or external judge is run.

Commands from the repository root:

```powershell
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.run dry-run
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.validate_offline
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.run run --limit 4
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.run run
python -B -m experiments.query_decomposition.p02_prompt_binding_probe_v1_20261022.run report
```

Only `run` can submit requests. It loads the existing credential through
`temporary_api_key` and uses the existing process-only provider DNS context.
No credential is copied into this package or its outputs. A whole-run lock
prevents duplicate writers. The shared provider gate permits one cooperating
request at a time with at least three seconds between starts. Legacy runners
do not participate and must be scheduled separately.

The same persistent checkpoint is resumed by every invocation. SDK and HTTP
automatic retries are zero. Technical recovery is bounded to five actual
attempts including saved prior attempts. Unknown requests with any response ID,
usage, raw response, or returned content are held for receipt review. Unknown
requests with no returned evidence may be reissued within that bound under the
user's delegated small-experiment authorization; each old receipt and charge
reservation remains intact. Length-ended/empty/refused responses are terminal
algorithm outcomes and are never repeated for a better answer.

`REPORT.json` records component coverage, output changes, tokens and costs.
Held technical work and terminal algorithm failures remain explicit. The
global ledger includes unknown liabilities; there is no historical fixed300
budget cap. Component cost alerts compare the corresponding historical reader
stage and are not presented as full-query RAG cost measurements. Semantic
correctness requires separate review of the saved fixed inputs and outputs;
there is no final F1/EM result from this local probe.

Main evaluation holds remain in place. This directory does not control any
other experiment's processes or checkpoints.
