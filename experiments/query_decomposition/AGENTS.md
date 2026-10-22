# Current RAG evaluation agreement

Read `unified_rag_eval_v1_20261021/PLAN.txt`, `PLAN.json`, and `DECISIONS.jsonl`
before changing an evaluation in this research family. Revision008 is the
current editorial review; revision007 experimental design remains unchanged.
Revision006 and earlier protocols remain historical
evidence; their ten-method matrix is no longer the required scope.

- The user delegated concrete design and public review publication of six main
  methods, three existing analysis groups, and a P02 QA/raw final-interface
  comparison. This supersedes repeated confirmation for those plan details.
  It does not authorize experiment execution. Keep every STOP_REQUESTED file.
  Do not start a supervisor, submit generation requests/GPU jobs, resume a
  historical small probe, or treat prior prompt-research authority as overriding
  the latest instruction to keep evaluation paused.
- Main method: frozen `p02_multigrain_timing_cachefix_v1`, independent execution
  variant `p02_shared_baseline_v2`. Keep prompts, examples, ordinary/goal QA,
  history and double-route retrieval behavior; no prompt candidate promotion.
- Required references: Top10, Top10+CoT3, QD+Rerank, Self-Ask, DecomP.
  Fixed source-grounded representative configurations are specified in
  `reviews/20261023_narrative_revision007/METHODS.json`. No new F1 tuning or
  paper-peak selection. Explicitly label common Qwen/BGE/reader adaptations.
  ReAct, IRCoT, Iter-RetGen and Hybrid are historical references only. Preserve
  Iter's approved k5/T4,T6,T7 source; do not schedule a run for an appendix.
- Main scope: the unchanged formal500 per dataset, six methods, 9000 outcomes.
  S02 remains history only. No resampling, result-based filtering, extra overlap
  experiment, exclusion or retraining. Keep provenance; do not claim entirely
  unseen questions, no known overlap, full official test or SOTA reproduction.
- Three existing analysis groups reuse historical P02-family evidence with its
  original denominators and frozen protocols, including goal512 and intervention
  arms. They are not unified-layer main scores.
  A partial/enriched diagnostic sample is not a population error-rate estimate.
  Cases do not establish component necessity; incomplete probes stay incomplete.
- New QA/raw design: fixed stratified100 per dataset, 300 queries and600 final
  conditions. Reuse the same valid shared-layer upstream; change only final
  auxiliary representation under a common pre-frozen zero-shot prompt. Include
  all actual ordinary and goal reader material in Raw, preserving association
  and deduplication. No new retrieval/reranking/summarization, no silent truncation.
  Native final output is not the common-prompt QA arm. Historical S02 or native P02
  traces cannot stand in for unvalidated shared-layer results. The complete
  protocol, prompt and qid-only list are in the revision007 qa_raw directory.
- Quality: native F1/EM with frozen alias rules; aim for all500 per method/dataset.
  Algorithm failures retain protocol-defined zero; infrastructure/unknown cases
  remain pending, not zero. Do not publish partial official scores while recovery
  remains possible. If500 becomes unattainable, disclose actual n and reasons.
- Report full-flow input/output tokens and logical LLM cost separately from new
  expenditure. Do not add an LLM-call-count metric. Cached logical usage is not
  newly billed usage. QA/raw incremental cost counts only its new finals, while
  both arm-level logical costs include the shared upstream.
- Strict C1 reuse remains: same method/question/stage, full messages/settings,
  stop+usage and intact retrieval. Only nonbinding historical caps may differ.
  Changed upstream invalidates downstream. Candidate counts are not validated
  completions; migration has not run. Preserve old bytes and liabilities.
- Timing is optional and not a main-completion gate: use the existing frozen50
  queries per dataset for six methods,900 flows; warm models/index, no response
  or retrieval-result caches, query concurrency1. Existing frozen files retain
  their old method list as history; revision007 supersedes that list only.
  Measure decomposition separately; verify actual remote hardware before any
  approved run. Separate local component time from first-content client TTFT;
  neither cached elapsed time nor first role/heartbeat is model TTFT.
- No fixed cumulative/per-stage budget cap has been reinstated. The revision007
  budget gives conditional reuse, all-fresh and retry/input-growth scenarios;
  estimates are not guarantees. GPU price is unknown. Preserve unknown request
  liabilities and never silently replay them. Single-query>3x or recent20 full
  fresh-query mean>1.5x historical expense triggers notification; changed method
  parameters must accompany interpretation of historical references.
- Recovery: at most five actual attempts including initial, per failed logical
  stage/query/method, shared across SDK/controllers/restarts. Exhaustion means
  record and temporarily skip, not replace a query or prove irrecoverability.
  Unknown requests require receipt checks before any paid replay.
- No equal-budget, multi-model/retriever, new judge, further baseline sweep or
  four component-necessity ablations are queued. Earlier extension intentions
  are preserved in history. Changes beyond this delegated plan must be recorded;
  evaluation resume still requires explicit user authorization.
- Archive compatibility junctions and frozen evidence stay intact. The entire
  `docs/` tree is local-only. Public review publication is limited to the agreed
  plan/evidence summary package, excluding credentials, datasets and raw billing.

- Revision008 only repairs contribution wording, conditional prompt interpretation,
  evidence links, R02 resource attribution and historical conclusion errata.
  All experiment scope, samples and prompt bytes remain unchanged. The common
  prompt has a specified evidence policy; do not claim neutrality or attribute
  whole-system differences solely to sequential solving/history. Historical
  archival authorization does not authorize any new move in this revision.

- Revision009 is explicitly authorized engineering preparation: new six-method entry, durable request/accounting, QA/raw binding and offline tests, followed by public engineering review. Keep revision007 scientific scope and all frozen sources/inputs/prompts; new code lives beside old adapters. Do not execute formal evaluation, remove STOP files, or write historical ledgers. Runtime entry is `unified_rag_eval_v1_20261021/runtime.py`; old run.main is not the new entry.
