# Complete-RAG ordinary CoT comparison

This is an independent development experiment, not a selected method or a
claim of improved F1. Exact execution authority is in RUN_AUTHORIZATION.json
and its matching main-plan decision after source freezing. The 38-node
semantic review is complete; its findings are summarized below.
Main formal evaluation, baseline changes and timing remain paused.

Use the exact exposed fixed90 R02/v12 questions from the reference-factor
experiment: 30 each from HotpotQA, 2Wiki and MuSiQue. Keep all six goal-only
cases and all other no-op conditions. Preserve original question arrays; do
not substitute manual repairs or select cases by the new stage results.

The four arms are N128, N256, F256 and R256 from the frozen ordinary-stage
experiment. N128 uses the exact native ordinary prompt and 128 output tokens;
N256 changes only its allowance to 256. F256 emits factual Support plus a
complete factual Answer; R256 emits concise facts/essential operation plus
Answer at the same 256 allowance. F→R compares these particular instructions
and demonstrations, including their realized lengths, not pure internal thought.

Each arm starts fresh and uses its own newly accepted answers in all later
rewrite, supplementary retrieval and QA calls. Native original/rewritten
interleaved Top10 retrieval is unchanged. Raw model receipts retain Support
and Answer, but only parsed Answer enters the original fewer-than-four-word
formatter at128 tokens, the next-node history, goal prompt and final prompt.
Support is audit material only. Goal128, its native consistency instruction,
native rewrite256, formatter128 and final64 are unchanged. Final length is
accepted under the existing native rule; ordinary/goal/formatter/rewrite
length is an algorithm terminal. Parsing a complete stop-ended F/R response
can also terminate the algorithm; never repair it with another model, reroll,
silently fall back to a historical answer, or classify it as a technical retry.

Persist raw generation, support, parsed answer and the accepted post-formatter
answer separately. Preserve candidate output even if parsing or formatting
fails before a full node is committed. Never overwrite native stage receipts
with the parsed answer. No previous flow's response, rewrite or reader packet
is seeded into this experiment; exact full-payload reuse within its own
namespace remains possible, including unchanged goal-only conditions.

Precommit all90×4=360 conditions with deterministic rotated order. First use
the existing six engineering cases×4=24 to check request identity, raw/parsed
boundary, dynamic-history propagation, complete receipts and retrieval
contracts. This is not quality selection: returned format/algorithm failures
stay in the denominator. Technical held tasks are retained separately and
not automatically revisited. Continue the full fixed cohort once engineering
contracts pass; do not choose an arm from the first24 F1.

Before any future execution, root must independently review, freeze exact
sources and dependencies, approve an exact-hash authorization and decision,
and verify the reference-factor full90 experiment is closed (all360 attempted,
zero in flight, matching final worker/scheduler closure). An incomplete
reference-factor run cannot be bypassed on the basis of quality. A separate
root-authorized technical scheduling change would require its own recorded
decision. New experiment uses a separate remote retrieval worker and directory
`cot90_<protocol hash prefix>`; no active worker is shared. A new allocation
may wait for scheduler startup and verified READY for up to ten minutes;
normal PENDING state does not immediately stop it. Local stop, terminal
worker state or timeout closes the owned worker and preserves the allocation
record; no automatic additional submission follows. Remote GPU receives
query-only retrieval work and public resource identities, not model keys,
messages, gold or metrics. Prefer the already verified remote A6000 settings,
but recheck resource availability before a future freeze/submission.

Future execution inherits the audited shared provider gate, fresh seedNone,
five durable returnless technical attempts at most, full returned-evidence and
ledger-before-cache guards, current-namespace recovery only, and preservation
of all unknown reservations. No automatic additional GPU allocation. Compare
per-query confirmed expense to historical full-query expense and flag a
threefold increase; separate confirmed cost from unknown reservations.

Report native F1/EM by dataset and equal-weight dataset mean, paired same-query
gains/harms against fresh N128, paired N128→N256 and F256→R256, input/output
tokens and estimated cost. Keep historical native P02 only as a separate
reference. Algorithm failures remain zero-score outcomes; technical missing
cases are excluded from matched quality denominators and explicitly counted.
Report all90 first and ordinary-present84 as a sensitivity. This repeatedly
exposed development comparison cannot replace later independent confirmation.

The completed fixed-input stage38 review has 37 adjudicated answers per arm
and one pending fact per arm: accepted-answer correctness N128/N256/F256/R256
is 30/29/31/32. R256 versus N128 has three gains and one harm. In the separate
29 adjudicated supplemental nodes both N128 and R256 have 28 correct; the
apparent improvement is concentrated in the initial eight diagnostic nodes.
These are component judgments, not complete-RAG F1. F/R also impose an explicit
evidence constraint in their frozen prompt package: their effects cannot be
attributed purely to Support or reasoning. The Araripe-father and Nanny-creator
cases show previously correct entities replaced by abstention under this
constraint. Retain all four arms; do not select on these component results.
