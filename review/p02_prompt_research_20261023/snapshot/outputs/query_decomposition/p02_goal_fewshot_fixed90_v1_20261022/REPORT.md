# Fixed90 goal few-shot experiment

Frozen-prefix exposed development comparison; main evaluation remains paused.

Planned:90 cases/270 conditions. All270 conditions have terminal records. Three-arm scorable paired cases:86. Unknown held conditions:7 across4 questions.
Returned algorithm failures:1; retained as zero-score outcomes in the paired denominator. Unknown requests remain unscored.

| Dataset | Matched n | Historical native P02 F1/EM | G2 F1/EM | I F1/EM | F F1/EM |
|---|---:|---:|---:|---:|---:|
| all | 86 | 75.99/66.28 | 68.38/58.14 | 69.87/59.30 | 68.78/58.14 |
| hotpotqa | 26 | 80.50/69.23 | 76.61/65.38 | 76.01/61.54 | 71.45/57.69 |
| 2wiki | 30 | 94.33/90.00 | 83.01/73.33 | 86.13/76.67 | 90.02/80.00 |
| musique | 30 | 53.74/40.00 | 46.63/36.67 | 48.28/40.00 | 45.23/36.67 |
| Dataset equal mean | same matched subsets | 76.19/66.41 | 68.75/58.46 | 70.14/59.40 | 68.90/58.12 |

Historical native P02 is a reference with its original goal/final protocol; G2/I/F use the common512/64 synthesis interface.
The matched table never compares different denominators. Missing unknown requests are not zero or automatically reissued.

New estimated expense from returned usage:CNY1.140178; current preserved request reservations:CNY0.027652; preserved old42 unknown reserve:CNY0.054802. Reservations are not confirmed spend; current reservations conservatively include connect failures, detailed in OFFLINE_AUDIT.json.

Costs below use the same85 successful cases in all three arms. They are logical goal+final usage including exact cached work, rather than newly paid complete-RAG costs. The length-ended goal is excluded only from this completed-flow cost comparison; its returned usage is included in new actual returned-usage expense above.

| Arm | Input tokens/query | Output tokens/query | Goal+final CNY/query | Logical full-flow CNY/query |
|---|---:|---:|---:|---:|
| G2 | 4706.2 | 225.7 | 0.004216 | 0.008335 |
| I | 4908.7 | 209.2 | 0.004345 | 0.008464 |
| F | 5918.7 | 214.2 | 0.005163 | 0.009282 |

No fresh retrieval, decomposer, GPU, external judge, timing campaign, method promotion or publication was performed.
These are development results on exposed examples; no independent generalization or pure reasoning-effect claim is justified.

## Interpretation and verification

G2 is the old synthesis paragraph. I is the revised instruction package without demonstrations. F inserts the four frozen fictional demonstrations into I. The actual frozen templates are stored with the experiment.

Dataset-equal F1 changes: I minus G2 = 1.3920 points; F minus G2 = 0.1478; F minus I = -1.2442.

On the same86 cases, I improves7/hurts7/ties72 vs G2. F improves6/hurts8/ties72 vs G2; vs I it improves5/hurts7/ties74. The pairwise-only I/F set in SUMMARY.json additionally contains one case and is clearly labelled n87.

All three synthesis variants remain below matched historical native P02. That historical comparison also changes the goal/final interface and output allowance, so it is not a pure test of reasoning text.

The demonstrated gain is dataset-dependent: F helps 2Wiki but hurts HotpotQA and MuSiQue. Some losses are native answer-format mismatches; others are genuine added conditions or unnecessary abstention. See [actual cases](CASE_REVIEW.md) and full local case evidence.

All I87 and F87 returned goals use a single paragraph and include at least one requested marker. Four F outputs and one I output contain both marker forms; format compliance does not establish reasoning correctness. One F output reaches the512-token cap; it remains a scored failure.

Exploratory paired bootstrap, resampling within each dataset with equal dataset weights, seed20261022 and10000 replicates; all intervals below include zero. These exposed development cases do not establish a stable improvement.

| Contrast | Macro F1 delta (points) | Exploratory95% interval |
|---|---:|---:|
| G2_to_I | 1.39 | [-4.18, 7.09] |
| G2_to_F | 0.15 | [-5.40, 5.72] |
| I_to_F | -1.24 | [-6.03, 3.25] |

On the same85 completed-flow cases, I goal+final expense is3.06% above G2; F is22.46% above G2 and18.82% above I. Full-flow estimates add identical reused upstream usage, not new upstream spending.

Offline audit passed: full frozen payload identity, model identity, finish reasons, raw usage totals, predictions, same-case native scoring, per-receipt ledger expense, frozen inputs and main hold hashes. Two typed pre-transmission connection timeouts were recovered within the predeclared5-attempt rule; their reservations remain conservatively retained. Seven unresolved condition outcomes across4 questions were not reissued.

Next research direction, not executed here: prioritize preserving intermediate relation facts, avoiding added question conditions and excessive abstention, and isolating final answer brevity from goal synthesis. Do not infer that longer explanations or more examples alone improve F1. No variant is promoted by this side experiment.
