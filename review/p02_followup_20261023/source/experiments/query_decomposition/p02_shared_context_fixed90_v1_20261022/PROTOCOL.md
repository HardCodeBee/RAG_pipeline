# Fixed90 reset / flat / chat complete-RAG development comparison

Offline preparation only. This package has no execution authorization until
root records a decision and exact protocol-hash RUN_AUTHORIZATION. Formal500,
baseline changes, timing and publication remain paused. Existing core,
frozen experiments, unknown receipts and main STOP files are unchanged.

Use exactly the exposed W/L fixed90:30 HotpotQA,30 2Wiki,30 MuSiQue, with their
actual historical R02/v12 arrays. There are176 ordinary nodes and90 goals per
arm, including6 goal-only queries. This is not the formal1500 generation batch,
manual60, or an unseen confirmation cohort. Source IDs/arrays, public material,
complete historical model/usage/payload/retrieval chains and references are
checked offline by source_inputs.py. Only id/qid/dataset/question/supporting_questions
enter inference; references, predictions, scores and provenance stay local.

Three arms are predeclared, each with90 fresh complete flows (270 total):

* reset: exact native single-user child/goal prompt at every stage.
* flat: previous turns' complete **raw native child prompts** and accepted
  answers, ordered as `User:\n...\n\nAssistant:\n...`, followed by the current
  native prompt labelled User. With no previous turn, send the original string
  unchanged.
* chat: the same previous raw prompts/accepted answers as alternating
  user/assistant messages, ending with current user. With no previous turn,
  use the same native single-user wire.

One Conversation is created for each original query and arm. ordinary and goal
share it. Only raw prompts are archived, never an earlier expanded flat/chat
wire. Pipeline calls record_node after optional formatting succeeds; the
external recorder must succeed before context commit. The committed assistant
answer is the same accepted answer used by native history/rewrite/final.
Raw initial and optional formatter requests/results remain separately saved.
This is a canonical accepted-answer transcript, not an assertion that a
formatted answer was the literal initial assistant response.

All current native prompt bytes remain unchanged: original question, prior QA,
four ordinary examples, instructions and current reader. Repeated examples and
historical QA inside those raw prompts are intentionally retained in both flat
and chat. Cumulative context can therefore grow quadratically in stage count;
it is not recursively/exponentially nested. Source growth estimates are
explicitly byte and old-usage proxies, not new measured input token counts.

The original core Pipeline.execute is wrapped, not copied or monkeypatched.
No M4, ROLE, O1, actual-current substitution, retraining, new decomposition,
retriever parameter or answer-style change is combined. Rewriter256,
ordinary/goal128, short-answer formatter128 and final64 remain native. Rewrite,
formatter and final are independent single-user calls. The formatter uses raw
Current and that arm's reader. Final receives the original native root packet
and helper, not conversation messages or an expanded final retrieval packet.
Every changed answer drives its arm's real subsequent rewrite and retrieval.

flat versus chat holds archived textual information and order fixed while
changing role serialization. Expanded context versus reset also changes old
reader availability, repetition, position and length; do not attribute its
effect exclusively to model memory or role. Shared client objects do not give
server-side persistent state: every API call resends its whole messages list.

The complete90 sample is fixed before inference. Order begins with6 queries,
two per dataset with the minimum SHA256 of
`p02-shared-context-fixed90-v1|20261022|` + canonical ID, followed by the other84
in original source order. Arm start order rotates with seed20261022. Those
first18 conditions form an **engineering gate only**: check exact payloads,
model/all3 usage, per-query isolation, native current fields and context rebuilt
from raw prompts/accepted answers. It never reads gold, prediction correctness
or F1. Legal QA/goal/formatter/rewrite algorithm terminal receipts retain zero
and do not fail the gate. Technical missingness or a wire contract defect
pauses before the remaining84 for engineering review. After contracts pass,
automatically continue the same frozen remaining252 conditions, irrespective
of pilot quality. No quality-based6/8/20 stop, resampling or prompt adjustment.

Messages are built before request identity/cache/reservation. Full ordered
messages, model and generation settings form the cache key; no last-question
shortcut. All arms are seed=None in a new namespace with no historical LLM or
retrieval seeding. Exact same-run full requests and exact resource/query
retrieval may share receipts; shared charges are paid and attributed once.

Before any provider call, use the conservative envelope
sum(UTF8 content bytes +64 framing units per message)+max_tokens <=128000.
Over-limit is technical held with the measured estimate; never truncate,
silently drop history or substitute a smaller protocol. This bounds this
research's input/price regime, not the model's documented total context window
and not a monetary cap. Official provider references checked by root:
https://help.aliyun.com/en/model-studio/multi-round-conversation and
https://help.aliyun.com/en/model-studio/qwen-plus .

Reuse the isolated complete-message transport with persistent full-payload
cache, one global cooperating provider request and start gap3s, NoCapBudget,
SDK/HTTP retries0, at most5 actual technical attempts including restarts.
Any response/usage/model/content/refusal/finish sign or returned ledger charge
prevents blind replay. Unknown receipts/liabilities stay intact. Propagation
requires exact requested/returned model and nonnegative integer prompt,
completion,total usage with total=sum. QA/rewrite/formatter require stop;
complete valid length/empty/refusal/filter are algorithm zero without reroll.
Native final alone may accept nonempty length as success_truncated. Incomplete
identity or usage is held, never silently zero. Cache resumes rebuild contexts
using the same accepted receipts and commit sequence.

Use one new isolated `context90_<protocolhash16>` retrieval-only A6000 job,
4CPU/32G/max4h and idle30min. Only dataset/qid/query/k/resource identity crosses
SSH, no LLM key/messages/references. Three frozen portable resources, BGE model
bytes, dependency hashes, VALIDATED stamps, actual READY device and whitelisted
qids are verified. New job submission has a deployment lock and durable unknown
submission guard. After a worker finishes, missing uncached retrieval is held;
never silently allocate another GPU. Capture FINISHED and scheduler closure.

Primary results: complete fixed90 native official F1/EM per dataset and overall;
all three paired contrasts flat-reset,chat-reset,chat-flat. Initialize each
dataset scorer once. Algorithm failures stay zero; technical missingness gets
fixed-denominator bounds. Historical same90 is separate from fresh reset; no
nonconcurrent drift is credited to context. Final text differences are method
masked and locked before interpretation; no automatic method promotion.

Record full-prefix logical input/output tokens and expense, plus actual new
distinct-attempt expense and unknown reservations. All repeated prefix tokens
count. Same-run sharing is not an independent cold deployment saving. Keep
original expense alerts: one query above3x its complete-query reference or
recent20 complete paid queries above1.5x matched references. Expected context
growth does not disable alerts. No fixed spending cap, LLM-count metric or
new timing claim. All270 outcomes and limitations remain reportable.

Offline checks and freezing:

```
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.run prepare
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.validate_messages_offline
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.validate_remote_offline
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.validate_offline
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.run freeze
python -B -m experiments.query_decomposition.p02_shared_context_fixed90_v1_20261022.run dry-run
```

Only run may create the actual checkpoint, load ephemeral credentials, connect
SSH or submit API/GPU work; it verifies approved exact experiment/hash/decision
before those actions. Re-entry after all270 terminal returns through read-only
report before Cache or bridge creation. Report/monitor SQLite is mode=ro.
