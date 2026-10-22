# Fixed90 goal synthesis: G2 / I / F

The user explicitly approved external Qwen API execution and reasonable costs
in the current side conversation, replaced the proposed42 with the existing
fixed90, and requested no repeated small-scope permission questions. The exact
authorization and interpretation are in `NETWORK_AUTHORIZATION.json`.

This freezes the same original90 questions,30 each from HotpotQA,2Wiki and
MuSiQue. No question is replaced. All six questions without ordinary child
questions are retained; the ordinary-present84 sensitivity is also reported.
This is repeatedly exposed development evidence, not an independent test.

The only interventions are goal and final regeneration over frozen actual
R02/v12 decomposition, ordinaryQA, goal rewrite and reader, and final root.
G2 uses the old synthesis prompt; I the revised instructions; F the same I
plus exactly the four fictional examples shown to the user. I-vs-F isolates
example insertion. G2-vs-I changes an instruction package, including output
labels and the length instruction. No sentence formatter, goal512/final64,
and the same common synthesis final wrapper/model/decoding in all arms.
No fresh retrieval, GPU, decomposer or external evaluator is executed.

Exact full-payload complete stop/usage/model matches from the old pilot can
be reused. Cached results are not chosen by quality. New requests use the
existing shared provider gate and cumulative ledger. SDK/HTTP retries0 and
at most5 durable known technical attempts. Returned incomplete algorithm
outcomes are retained without reroll. Unknown requests are held.

Six goal payloads from the sandbox-blocked42 attempt remain unknown. The
frozen `EXTERNAL_UNKNOWN.frozen.jsonl` guard prevents a new output directory
from silently resubmitting them. They are not assigned zero, replaced or
hidden from the planned90 denominator. Their old reservations remain intact.
If those conditions remain held, report exact paired subsets and missingness;
never label available-only results as complete90 scores.

Report native F1/EM, paired improvements/ties/regressions, input/output tokens,
logical goal/final and full-flow estimates, and new actual receipt charges.
Frozen upstream cost is logical reused work, not newly billed inference.
No fresh full-flow time or TTFT claim is made. The main formal campaign,
timing campaign and publication remain unapproved and paused.

```powershell
python -B -m experiments.query_decomposition.p02_goal_fewshot_fixed90_v1_20261022.run --dry-run
python -B -u -m experiments.query_decomposition.p02_goal_fewshot_fixed90_v1_20261022.run --run
python -B -m experiments.query_decomposition.p02_goal_fewshot_fixed90_v1_20261022.run --report
```
