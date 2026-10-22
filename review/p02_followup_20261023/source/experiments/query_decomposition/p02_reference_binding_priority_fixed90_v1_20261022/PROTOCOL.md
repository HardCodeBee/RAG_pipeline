# Ordinary prompt factors on fixed90: execution draft

This protocol prepares 360 complete native P02 RAG flows: the same exposed
fixed90 actual frozen R02/v12 outputs, 30 per dataset, under C0/S1/B1/P1.
All cases and arms remain planned, including 6 goal-only cases and binding
no-ops. Existing stage results do not select candidates, questions or budgets.

C0 is the original ordinary QA prompt. S1 consistently renames allowed #N
references/labels to @N; real self-contained question bodies contain no such
references, so this isolates presentation rather than genuine placeholder
resolution. B1 adds the frozen stage probe's candidate span binding when a
literal prior-answer-supported substitution exists. P1 relocates the exact
existing current-subquestion priority sentence. These are separate factors.
No M4, ROLE, actual-current replacement, shared conversation or goal synthesis
is combined. The frozen stage templates are imported without modification.

Every arm uses its own newly accepted prior answers and live accepted rewrite.
Its new answers feed all downstream native rewrites and real retrieval.
Historical answers, rewrites, reader packets and maps are not inserted.
Goal QA, optional short-answer sentence formatter, final native root evidence,
caps (rewrite256, QA128, formatter128, final64), examples and retrieval remain
native. Every LLM request is a fresh single-user string; prior QA appears only
where the native prompt supplies it. Full payload identity permits exact reuse
within this isolated run only, with first-owner actual fees paid once.

The same preselected engineering6 are first, yielding 24 flows. Their gate
checks complete receipt/model/usage, single-user wire, ordinary-only template
application and unchanged goal/formatter/final construction. It does not read
gold answers or native scores. Valid algorithm terminals are retained as zero;
technical uncertainty or contract failure pauses for engineering review. After
the gate passes, all remaining84 run regardless of observed quality.

Before any model client, API-key context, SSH connection or GPU submission,
both pinned predecessors must have attempted every condition: context90 all270
with no in-flight requests and matching job/READY/stopped FINISHED receipts;
stageprobe all152 with no in-flight requests and exact checkpoint identity.
Successful outcomes plus durable held tasks count as attempted. Missing quality
does not delay release; unknown requests are neither resubmitted nor imputed.
`--wait-for-predecessors` performs only read-only metadata checks every30seconds.
The new output STOP cancels this wait without touching predecessor holds.

Execution uses the pinned NativeAsk/SubmissionAwareStrictAsk/shared provider
gate and NoCapBudget. At most five actual technical attempts include initial
submission and restarts. Only this new namespace's returnless failures may use
the inherited recovery path; any content/model/response ID/finish/usage or
returned-usage ledger evidence blocks blind replay. Every reservation remains
visible. Durable held conditions are skipped on restart, and answers are never
regenerated for quality. SDK and HTTP retries remain zero.

A separate `factor90_<protocolsha16>` A6000 retrieval-only worker uses the same
three pinned resources and 90-qid allowlist, 4CPU/32G and an explicit job time
limit in EXECUTION_SETTINGS.json (draft6hours, root verifies partition maximum
before freezing). When less than15minutes remain by the conservative submission
clock, no new full condition starts; its task remains unstarted rather than
held. Before each dispatch, a stopped allocation, worker FINISHED/error/STOP
marker or unconfirmed worker connection also prevents dispatch and preserves
remaining tasks as unstarted; available worker receipts are captured. The
current condition finishes normally before worker closure. This is
not a guarantee against an unexpectedly slow flow; in-flight evidence remains
protected if scheduler termination occurs. No automatic new allocation is
created for the remaining work. No key, gold
answer or LLM prompt crosses SSH. The inherited bridge is loaded as a private
module instance; parent sources/globals/outputs are untouched. A durable Slurm
submission sentinel prevents silent resubmission. Remote credentials are read
only by ephemeral getpass; environment passwords are rejected. The optional
pinned address from the root's read-only partition/DNS preflight supplies the
TCP socket, while Paramiko still checks cluster1.comp.anu.edu.au under
RejectPolicy. Only the new client instance's connect method is wrapped;
neither global DNS nor Paramiko classes nor parent modules are patched. Worker stop
and READY/FINISHED receipt capture belong only to this isolated job.

Status and engineering never compute F1. The explicit offline `score` module
reports per-dataset native F1/EM, paired improvement/tie/harm versus fresh C0,
and same-ID historical P02 references separately. Missing technical cases keep
visible denominators and bounds; no partial score is presented as complete90.
Input/output tokens and logical estimated expense are separate from actual
first-owner charges and unresolved reserves. Existing query3x and recent20
complete-paid-flow1.5x historical expense alerts remain. No LLM call metric or
timing benchmark is added. Native metrics alone do not prove factual repairs;
method-masked semantic review precedes a root interpretation of differing final
texts. There is no automatic method promotion or formal500/timing restart.

Preparation and tests authorize no services. Root must independently review,
freeze, then create exact-hash RUN_AUTHORIZATION and append the main decision.
The authorization must explicitly retain current-namespace returnless-only
recovery and maximum5 attempts. No authorization or decision is written by
this preparation module.
