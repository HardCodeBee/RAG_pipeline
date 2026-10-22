# Reference notation, history bindings and local task priority

This is an autonomous, isolated development experiment approved by the user's
request to advance the three prompt-mechanism comparisons. Main formal500,
baseline changes, timing and publication remain paused. No old frozen artifact
or unknown reservation is modified. Execution needs a root decision and an
authorization bound to the final protocol hash; preparing this file alone does
not submit a request.

## Questions and arms

All comparisons use the actual frozen R02/v12 ordinary subquestion, its original
question, identical recorded history and identical reader passages. No manual
decomposition, gold answers or diagnostic correctness labels enter inference.
Each intervention is compared separately with a fresh original-prompt control.

* C0: unchanged native ordinary-QA prompt.
* S1: consistently rename structural reference markers from `#N` to `@N` in
  the resolution instruction, demonstration reference labels and placeholders,
  and actual history/current labels. Preserve actual question values, answers
  and reader passages exactly. These real R02 subquestion bodies have no numeric
  placeholders, so this tests presentation and demonstration notation, not
  actual-question placeholder resolution or a new prompting paradigm.
* B1: retain raw Current and add candidate description-to-prior-answer span
  bindings. Locate changed spans by token-offset SequenceMatcher alignment of
  raw Current against the same node's previously accepted retrieval rewrite,
  with casefold matching and autojunk disabled. Enclose all non-equal edits in
  one contiguous substitution window, including equal tokens between those
  edits, so a shared name inside a referring description is not cut away.
  Only unchanged question-frame text remains outside the window. If a shared
  capitalized non-function-word name prefix immediately borders the window,
  conservatively include that prefix before checking literal answer support.
  The left value must be copied from raw Current; the right value must be a
  contiguous literal span of an existing prior answer, copied with its original
  characters and accompanied by supporting step IDs. Exclude pure punctuation,
  interrogative/function-word replacements, targets beginning with function
  words other than articles, and targets without a name-like or numeric token.
  Do not use an answer key, new model output, hand-repaired
  mapping or additional retrieval. The block is explicitly candidate information,
  not a guarantee that a referent or semantic role was identified correctly.
  If no eligible mapping exists, send exactly C0; do not insert an empty-binding
  message. Original, Current, history and reader remain unedited. This differs
  from the previous full rewritten-question candidate block and from replacing
  the actual Current question.
* P1: move the existing sentence `The original question supplies scope; answer
  only the current subquestion.` from the initial instruction block to directly
  before the Actual Current field. Preserve its exact words once, and preserve
  the original-question block and every question/answer/passage. This identifies
  a local-position/priority-cue effect, not every possible task-priority mechanism.

Do not combine the three factors, use shared conversation history, substitute
the actual Current, or add M4/ROLE/goal synthesis. B includes structure, added
tokens and local salience as well as denotation alignment; do not claim those
influences have been eliminated. Symbol changes can also change tokenization.

## Frozen stage samples

Begin with the existing eight O8 ordinary nodes (six known local QA failures and
two reasonable-abstention boundaries), in their frozen order. Then append the
existing thirty ordinary-fewshot nodes, preserving their order and removing only
exact duplicate node inputs. Preserve membership in both source cohorts when
deduplicating. Different reader/history snapshots of a query remain different
node inputs and remain linked to the same original query. Report the natural
thirty-node and exposed stress-eight results separately; the union is not a
population accuracy sample. The input audit found 38 distinct snapshots from
35 original queries, with no exact cross-cohort duplicate. Final counts and
source hashes are frozen before candidate generation. No new answer, F1 or
semantic label selects a node.

The first eight nodes have 32 logical arm conditions. Rotate arm order per
original query using a fixed 20261022 hash and retain every result. All controls
are fresh in one new checkpoint with seed=None. Within this experiment, exactly
identical full payloads may share one receipt and charge; record every logical
use. B no-op cases are applicability checks, not binding successes. Record B
coverage, proposed mappings and literal support before inspecting outputs.

## Execution and cost

Use the already checked NativeAsk/SubmissionAwareStrictAsk and complete-message
transport from context90, passing ordinary string prompts only. Preserve the
native QA128 and optional original less-than-four-whitespace-words sentence
formatter128. The formatter always receives raw Current and the same reader.
Save initial and accepted answers separately. No decomposition, retrieval, goal,
final generation, official answer scorer or external judge is used in this stage.

Keep the shared serial provider gate and minimum three-second start interval,
SDK/HTTP retries zero, at most five actual technical attempts across restarts,
exact requested/returned model, three nonnegative integer token fields and their
sum, returned-evidence/ledger protections and all unknown liabilities. Complete
algorithm-terminal length/refusal/filter/empty receipts are retained without
quality rerolls; incomplete receipts and provider rejection are technical held.
No query is replaced. There is no fixed spending cap. Record all input/output
tokens, distinct-attempt estimated charges and unresolved reservations, without
an LLM-call-count or timing benchmark metric. Component expense references are
component references, not complete-query expense means.

The initial32 is a bounded fixed-reader probe sharing the existing gate with
context90; it does not create another provider scheduler or GPU job. The rest
of the frozen node union is deferred until the context90 run has attempted all
planned conditions, ended its local requests and closed its isolated GPU worker.
Its one known technical hold does not require full270 quality successes before
the deferred work can start. Queue status, predecessor identity and any missing
completion evidence must be explicit. Never change or resume context90/main
holds in order to release this queue.

The default `run` processes only the initial32 stress conditions, even if the
predecessor later finishes. The deferred120 additionally require explicit
`run --defer afterguard` and `approved_deferred_conditions: true` in the exact
protocol-bound authorization. One authorization may expressly approve both
phases, but the deferred flag and predecessor guard still apply. Optional
`--wait-for-guard` persists `DEFERRED_WAIT_STATUS.json` and checks predecessor
metadata every thirty seconds before opening a provider client, Cache, key or
ledger. Local STOP exits this waiting mode. The guard binds the predecessor's
frozen protocol, inputs and exact270-task order to its checkpoint identity and
matching remote job/READY receipts; its native FINISHED leaf must identify the
same GPU and stopped no-LLM worker. All planned tasks must have a terminal or
technical-held record and no request may remain in flight; it does not require
270 quality successes. Existing technical-held conditions count as attempted
and are skipped by normal resume/defer, retaining their unknown liabilities;
any technical recovery requires a separate explicit action.

## Interpretation and continuation

Review all candidates with method labels hidden and first labels locked. Judge
against raw Current, history and reader, separating complete factual correctness,
responsiveness, required information, entity/relation/scope errors and reasonable
abstention. True incomplete relevant facts may be correct; a correct name inside
a false complete relation is wrong. Preserve concrete unresolved fact work items.
Do not infer semantic improvement from a different string or from final gold.
Cluster repeated nodes by original query; preserve regressions and B applicability.

This stage produces component evidence and no final F1/EM claim. The next planned
complete-RAG comparison uses fresh C0/S1/B1/P1 on the same exposed fixed90 actual
R02 inputs, with B computed from each arm's live accepted rewrite and its own
history. All ordinary nodes use the rule, including no-op bindings; goal,
formatter and final stay native. Upstream changes require actual downstream
rewrites and retrieval. Report native F1/EM, paired improvements/regressions,
whole-flow input/output tokens and cost separately from the historical P02.
That executor and its own exact authorization are not provided by this stage's
authorization. Do not promote a method or claim general F1 gains from these
stage probes. Do not use the eight-node score to present a selected subset as
the fixed90 result.
