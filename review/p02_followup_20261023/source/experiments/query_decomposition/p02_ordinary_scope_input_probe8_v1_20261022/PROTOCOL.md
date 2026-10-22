# Ordinary original-question input probe (development only)

This package is prepared after the dynamic balanced20 development comparison
found no improvement on the 19 available pairs. It is a stage mechanism test,
not a complete RAG evaluation or a new estimate of population accuracy. The
formal evaluation remains paused. Preparing and freezing does not authorize
provider requests; root must separately approve this exact protocol hash.

Use all eight already exposed formal R02/v12 ordinary-QA examples from the
existing I/C challenge, in their original order: six local-QA failure examples
and two reasonable-abstention boundaries (5 HotpotQA, 3 2Wiki). No new outcome,
historical final score or answer selects or drops an example. Historical source
hashes, actual messages, decomposition arrays and prior public-source proof are
rechecked offline. No new public-network provenance assertion is made.

O0 sends the original complete ordinary-QA payload, freshly generated. O1
deletes only the `Original question:` header, its exact value and following
blank line at the beginning of the actual-task segment. All instructions,
four demonstrations, history, raw Current subquestion, retrieved passages,
model settings and 128 output-token cap stay byte-for-byte identical. The
instruction `The original question supplies scope` remains present in both
arms; its dangling reference in O1 is intentional. Shorter input and shifted
positions remain part of this contrast, so a gain would not prove a unique
root-question contamination mechanism.

Both arms are fresh in one new checkpoint with seed=None; no historical I/C
response database is provided as a seed. Each node has two conditions, with
O0/O1 first order alternating by case index (four cases in each order). All
16 conditions are retained. No formatter, retriever, goal, final answer,
decomposer or official scorer is invoked. The raw Current subquestion remains
the evaluation target, with original entity/relation/scope constraints. A true
answer about another entity is not a repair.

Use the unchanged, pinned balanced20 NativeAsk/SubmissionAwareStrictAsk adapter
and shared gate/NoCapBudget. SDK/HTTP retries are zero; technical attempts have
a durable total maximum of five including initial submission and restarts.
Unknown requests without returned evidence use that bounded existing recovery;
returned response/content/refusal/finish/model/usage or returned ledger evidence
holds without blind resubmission. Unknown reservations are retained. The
shared gate serializes submissions and enforces a three-second minimum start
interval. There is no fixed cost ceiling. This stage-only probe records tokens,
logical stage costs, distinct new ledger charges and unresolved reservations.
A stage above three times its historical stage-cost reference is flagged as a
stage diagnostic. Sixteen stages never become a "recent20 complete-query"
expense trend, and no query-time or LLM-call-count metric is claimed.

A normal result requires the exact requested and returned model; three
nonnegative integer token fields (excluding booleans), with matching sum;
finish_reason=stop; nonempty content and no refusal. Length, content_filter,
refusal or empty text with full identity/usage integrity is an algorithm
terminal, preserved without reroll. Missing usage, mismatched model, incomplete
receipt, provider block or exhausted technical attempts stays technical-held,
separate from truth/quality. Resume uses exact complete payload binding in
this checkpoint only. The adapter's inherited internal `full8/` stage tag is
diagnostic; the new output directory, task IDs and ledger namespace isolate O8.

After methods are masked, review each candidate against raw Current, frozen
reader and history. Label full factual statements and the needed information
separately. Existing rules remain: true but incomplete relevant facts can be
truth-correct; reasonable evidence-insufficient abstention is truth-correct;
incorrect full relational statements or answers to a different target are
wrong. Do not count a truthful abstention or a wording/format change as a
substantive repair. Preserve every regression and scope/temporal/identity harm.

Predeclared continuation gate: **stop this branch if there is no substantive
factual or necessary-information repair, or if any clear scope harm occurs**.
Unresolved technical missingness prevents declaring the gate passed. Any
successful stage candidate would still require root approval for a fresh
complete dynamic RAG comparison before an overall F1 claim or promotion. No
automatic follow-up, combination with ROLE/M4, or main-evaluation restart is
authorized here.

Offline commands (0 API, 0 SSH, 0 GPU):

```
python -m experiments.query_decomposition.p02_ordinary_scope_input_probe8_v1_20261022.run prepare
python -m experiments.query_decomposition.p02_ordinary_scope_input_probe8_v1_20261022.validate_offline
python -m experiments.query_decomposition.p02_ordinary_scope_input_probe8_v1_20261022.run freeze
python -m experiments.query_decomposition.p02_ordinary_scope_input_probe8_v1_20261022.run dry-run
```

The optional later `run` action requires a root-written RUN_AUTHORIZATION with
the frozen protocol hash and a recorded decision ID. Report and completed
reentry read SQLite with mode=ro; they never construct Cache or a provider.
