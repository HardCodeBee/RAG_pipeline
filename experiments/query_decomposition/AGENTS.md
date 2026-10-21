# Current RAG evaluation agreement

Read `unified_rag_eval_v1_20261021/PLAN.txt`, `PLAN.json`, and `DECISIONS.jsonl`
before changing an evaluation in this research family. These are the current
agreement; older frozen plans describe historical execution only.

- Primary method: `p02_multigrain_timing_cachefix_v1`, using a separate execution
  variant over the existing baseline-v2 request and complete-input layers.
- `s02_child4`: historical complete-RAG route reference only. No new S02 run.
- Baselines: integrity-repaired v2; original campaign scores are references only.
- Revision006: main evaluation does not enforce equal per-query retrieval or
  generation budgets across methods. Use paper best reported corresponding
  dataset Answer-F1 configurations where available, with explicit source and
  common-resource adaptation records. Do not silently relabel old local-dev
  choices as official best, or start new local F1 tuning by default.
- Iter-RetGen primary parameter selection is now paper k5 with T4/T6/T7 for
  HotpotQA/2Wiki/MuSiQue, pending local compatibility validation. The former
  k10/T5 matched50 is historical budget-matched reference, not main default.
- Equal-budget complete-RAG evaluation is a separate extension whose scope is
  approved but sample, method pool, budget values and stopping design are still
  pending. It is not a prerequisite for main evaluation. Never infer identical
  generation cost from identical per-request output allowance.
- Known training-question overlap evidence remains intact. No extra overlap
  experiment, exclusions, replacement questions or retraining is queued. Do not
  claim no overlap or entirely unseen evaluation questions, or hide material
  provenance when making independence/generalization claims.
- Only complete-RAG final-answer evaluation belongs in the current primary table.
  Same-final-prompt control and four mechanism ablations are deferred.
- Aim for all 500 fixed questions per dataset/method. Do not publish partial
  official scores or a scorable-count metric while recovery is still possible.
  If 500 is confirmed unattainable, disclose actual denominator and reasons.
- Metrics: native F1/EM, input/output tokens, estimated LLM expense, local
  component processing time and client TTFT separately. Do not add an LLM call
  count metric. TTFT starts when the client submits and ends at first nonempty
  content, not the first role/heartbeat frame; it includes network/provider wait.
- Keep all evaluations paused. Do not clear hold files, start supervisors,
  submit API requests, schedule GPU jobs, or interpret a plan edit as approval.
- Approved timing cohort: 50 fixed stratified queries per dataset, shared by
  P02 and nine v2 baselines (1500 measurement flows). Warm model/index, disable
  response and retrieval-result caches, concurrency one original query. Time
  P02 decomposition separately and label hardware. Do not replace quality
  results or label this timing subset as a full-500 mean. The user delegated
  sample/hardware details: seed20261021 and 50 IDs/dataset are now frozen in
  execution_details/TIMING_COHORT.frozen.json. Prefer remote A6000 GPUs; the
  proposed fixed local RTX4060 assignment is withdrawn. Verify actual remote
  devices/resources at allocation time; this is not resume authorization.
- The user removed fixed cumulative and per-stage budget caps for this main
  evaluation, subject to reasonable spending with no anomalies. The old 300
  cap and proposed 320 cap are superseded. Preserve usage, provenance, all
  historical charges and unknown liabilities; do not silently replay unknown
  requests. This policy does not resume evaluation or modify frozen runners.
- Failure recovery: at most five actual attempts including the initial attempt
  at a failed logical stage per original query/method; count across restarts and
  SDK/controller layers. Record and temporarily skip after exhaustion. Do not
  replace questions, infer irrecoverability, or rerun algorithm failures for F1.
  Unknown/in-flight requests need receipt verification before any paid replay.
- Expense increases notify the user against historical same-dataset/method
  per-query means (label same-method cross-dataset proxies when needed).
  Operational defaults: a single query above3x or recent20 complete fresh-flow
  mean above1.5x. No hidden budget cap; cache-only/partial runs do not dilute
  trend means, and unknown usage is not zero. See the frozen execution details.
- The user requires explicit approval of **every new evaluation decision**:
  methods, samples, metrics, measurement conditions, cache/retry rules, ranking,
  archive changes, budget and execution. Record their approval in the same plan
  and append-only decision log before acting. Silence is not approval.
- Physically archived directories have an index at
  `../../archives/rag_evaluation_history_20261021/INDEX.json` and approved
  original-path junctions. Do not delete compatibility paths or frozen evidence.
- Preserve historical bytes and provenance. Do not relabel old P02 results as
  shared-layer results or count reference runs as newly completed main results.
