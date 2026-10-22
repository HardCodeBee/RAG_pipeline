# Local evidence combination before an ordinary subanswer

This is a bounded development experiment under the user's autonomous small-
experiment authorization and explicit 2026-10-22 request to try CoT prompts
with progressively connected P02 subquestions. Main evaluation, timing,
baseline changes, publication and candidate promotion remain paused.

## Evidence and question

The completed result-binding/operator study found G0/GF/GO final F1 of
76.62/73.76/70.21 on the same90 questions. Adding explicit operation sentences
to goal demonstrations did not improve that version. Its ordinary example-
binding comparison had30 versus29 correct of38, with one pending in each arm.
The completed shared-context study found reset/flat/chat F1 of
76.43/69.90/71.59 on common89, while flat/chat total tokens increased about82%.
The reference-factor stress8 probe showed only one repair for both a symbol
change and a literal binding annotation, which does not identify a dominant
mechanism. Information-flow diagnosis also distinguishes absent evidence,
reader selection loss, missing intermediate information and misuse of available
facts. A longer explanation cannot restore facts never supplied to the reader.

This experiment asks whether a short explicit connection from available facts
to the CURRENT subquestion improves ordinary answers, beyond a larger output
allowance or a facts-only summary. It does not repeat goal GF/GO, combine all
past candidates, alter frozen decompositions or accumulate previous passages.

## Fixed arms and inputs

Use exactly the existing reference probe's38 ordinary nodes:8 declared stress
nodes plus30 supplemental nodes,35 original queries, actual frozen R02/v12
questions. Copy its input bytes; no selection by new scores or completion order.
Each arm gets the identical original/current question, accepted prior QA and
reader passages. These are repeatedly exposed development inputs, not a fresh
holdout or a complete-RAG evaluation.

| Arm | Ordinary output allowance | Output protocol |
|---|---:|---|
| N128 | 128 tokens | Exact native one-factual-sentence prompt |
| N256 | 256 tokens | Exact same native prompt as N128 |
| F256 | 256 tokens | Support: concise necessary facts; Answer: full factual sentence |
| R256 | 256 tokens | Support: concise necessary fact connection/operation; Answer: full factual sentence |

N128→N256 isolates the allowance change. N256→F256 changes factual summary and
output packaging. F256→R256 compares these specific facts-only and explanatory
protocols at the same allowance and field structure. Realized text lengths and
chosen facts can still differ; this is not a proof of a pure internal-reasoning
effect or a new prompting paradigm.

F/R preserve native task scope, prior-answer candidate status, correction
conditions, all four original teaching questions, histories, passages and
example final answers. Only the contradictory native output paragraph is
replaced. Their examples have the same supplied fact sets, with brief operation
language added in R. No case-specific repair or outside fact enters a template.
Support normally stays within60 words. It is an unverified model explanation,
not independently established evidence. The answer must preserve subject,
relation direction, requested attribute, units and necessary conditions.
Missing facts do not imply zero, a complete list, no, or absence.

## Output boundary

The structured response must contain exactly one Support field followed by one
Answer field, both nonempty, with no additional output blocks. Whitespace may
vary. No second model call repairs missing headers or contradictory content.
Format failure is an algorithm outcome retained in the planned denominator;
no reroll or fallback to an old answer is allowed. Preserve the original raw
receipt, parsed Support, parsed Answer and accepted final fact separately.

The existing native short-answer formatter is applied only when the parsed
Answer has fewer than4 words, to ALL arms under the same rule. Its prompt,
raw-current question, reader and128-token allowance remain native. Support
does not enter this formatter or any history. This first fixed-reader phase
has no live subsequent step, rewrite, goal, final or retrieval operation.

## Review and reporting

Method-masked local review records two distinct quality axes:

1. Complete emitted response correctness: every stated fact and asserted
   connection in Support and Answer must be correct, and the Answer must
   address the current subquestion. A correct final entity with a false full
   statement or a false Support does not count as a fully correct response.
2. Accepted fact-answer correctness and useful information retained: assess
   the fact that would actually be passed downstream, separately from Support.

True incomplete relevant answers may be correct; true off-target background
is an answer error. Reasonable evidence-insufficient abstention can be correct
but lacks the needed intermediate value. Unsupported facts are not automatically
false. First use supplied/local evidence; specific unresolved facts remain
pending work items, explicitly excluded from affected semantic denominators.
Full response, fact-only and information retention results must not be merged.

Report paired improvement/tie/harm against each appropriate parent, stress8
and supplemental30 separately, parser/algorithm failures, pending counts,
input/output tokens and logical usage cost. Keep actual new returned-usage
expense and unknown reservations separate. A stage-only count is not a complete
query cost window. Existing component3x reference alerts remain; no LLM-call-
count metric or complete-RAG F1 is inferred from these nodes.

## Execution and ordering

All152 node-arm conditions are precommitted in deterministic rotated order.
First32 stress conditions provide engineering inspection, not a quality-based
selection gate; all four arms and all30 supplemental nodes remain planned.
Receipt/wire/identity violations require technical inspection. Genuine model
format/content failures remain algorithm outcomes rather than technical retry
opportunities. Keep frozen controls and every failed outcome.

Before provider/key/cache work, require the exact preceding reference-stage152
and result-binding/operator346 completion records with no in-flight requests,
and either a passing exact-protocol first24 engineering gate of factor90 or
the explicitly recorded technical-pause alternative below. This CPU/API-only
probe shares the existing serialized provider gate; it allocates no GPU and
does not modify another experiment's requests or outcomes.

Operational update before freezing: factor90 job160987 ended normally after
its local runner hit a technical event-logging error, at2 successful conditions
plus1 technical hold, zero in-flight. The independent watcher preserved the
exact scheduler closure and wrote that experiment's STOP. An exact-hash root
decision may release this independent fixed-reader probe on that verified
technical pause (matching job/protocol/directory, terminal scheduler receipt,
local STOP and zero in-flight), without requiring an unachievable first24 gate
while the other controller is repaired. This release cannot depend on quality
scores and does not authorize replay of its held task. Keep factor90 repair
offline until this bounded probe is done; its continuation is separately
recorded. Preserve the old scheduler receipts, STOP and reservations.

Fresh isolated namespace, no historical response seed, full-payload exact
reuse only within this run. At most5 durable actual technical attempts including
initial submission/restarts. Only current-namespace returnless failures may be
recovered under the inherited audited policy; any returned content/model/ID/
finish/usage or returned-usage ledger evidence blocks blind replay. Historical
unknowns and existing held tasks are never reissued. Preserve all reservations.
Root must freeze source/prompt/input/runtime dependencies, write exact-hash
authorization and append the main decision before any provider submission.

## Subsequent exploration

Once these labels are locked, inspect whether improvements come from correct
prior-result selection, directed relations, attribute selection or justified
uncertainty, and whether explanations introduce new false statements. A
separate complete-RAG comparison must use each arm's own newly generated
answers and actual downstream retrieval before claiming F1 improvement.
An answer-only handoff and a compact Support-plus-answer handoff are separate
future treatments; do not silently feed Support into the current pipeline.
Do not combine a goal synthesis change with an ordinary CoT change in the
same first comparison. Further confirmation should use a preselected new
development subset and disclose prior diagnostic exposure.
