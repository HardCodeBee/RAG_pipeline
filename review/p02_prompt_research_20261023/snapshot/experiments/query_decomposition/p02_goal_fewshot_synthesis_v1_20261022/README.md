# Goal synthesis with four demonstrations

The user explicitly requested this small experiment in the side conversation
after reviewing the complete instruction and four synthetic demonstrations.
This experiment does not change or resume the main formal RAG evaluation.

## Comparison

- **G2**: the existing synthesis paragraph prompt. Both goal and final are
  reused only after exact full-payload, complete-content, stop and usage checks.
- **I**: the revised instructions shown to the user, without demonstrations.
- **F**: exactly the same I instructions plus the four displayed fictional
  demonstrations. Example insertion is the only difference between I and F.

The cases are the existing stratified pilot15 (five per dataset) and historical
goal-error challenge27 (17 2Wiki and ten HotpotQA), without overlap. Report the
two samples separately: challenge27 is targeted diagnostic material, and all
42 cases have already been exposed during development. Neither sample provides
an independent test of generalization. The examples teach procedures implicated
by previously inspected errors, which makes independent confirmation especially
important before claiming a general improvement.

Freeze actual R02/v12 decomposition, ordinary questions and full answers, goal
rewrite, goal reader passages, final root passages, model/decoding and common
final presentation. Only regenerate goal and final for I/F. The output limits
remain goal512/final64 with no sentence formatter. No decomposer, retrieval,
GPU, external evaluator or new main-evaluation conditions are run.

## Interpretation

G2-to-I changes an instruction package, including output labels and the length
instruction. I-to-F isolates the displayed demonstrations. Neither comparison
isolates the presence of reasoning text from all other generated content.
Final-answer native F1/EM, paired gains/losses, goal/final input/output tokens,
logical stage cost and new paid usage are retained. These measurements do not
claim full-flow latency, TTFT or fresh complete-RAG cost.

The runtime uses the existing shared provider gate and cumulative expense
ledger. Complete exact matches may be reused. SDK/HTTP retries are disabled;
known technical failures receive at most five durable actual attempts.
Unknown requests are held, not automatically reissued. Successful but
incomplete/length-ended responses are retained as algorithm outcomes and are
never regenerated to improve scores. Main campaign holds remain unchanged.

## Research motivating the candidate

- Wei et al.: https://arxiv.org/abs/2201.11903
- Zhou et al.: https://arxiv.org/abs/2205.10625
- Khot et al.: https://arxiv.org/abs/2210.02406
- Wang et al.: https://aclanthology.org/2023.acl-long.153/
- Zhang et al.: https://arxiv.org/abs/2210.03493

These papers motivate a local hypothesis; they do not establish P02 gains.

## Commands

```powershell
python -B -m experiments.query_decomposition.p02_goal_fewshot_synthesis_v1_20261022.run --dry-run
python -B -u -m experiments.query_decomposition.p02_goal_fewshot_synthesis_v1_20261022.run --run
python -B -m experiments.query_decomposition.p02_goal_fewshot_synthesis_v1_20261022.run --report
```

`--prepare` is a one-time freeze operation and refuses an existing protocol.
`--limit` counts new conditions, allowing an engineering-only first batch.
The immutable checkpoint prevents overwriting previously recorded outcomes.
