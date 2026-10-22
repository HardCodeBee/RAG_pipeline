# Result binding and goal operation experiments

Completed bounded side-conversation development research. Main formal evaluation remains paused. No decomposer, retrieval or production implementation was changed.

## Ordinary QA: result-use notes

D uses the descriptive E2 demonstrations. B adds two example-only notes showing how to extract the requested prior result and bind it into the current question. The actual task is identical. All76 raw ordinary outputs completed successfully.

| Cohort | Nodes/arm | D correct / wrong / pending | B correct / wrong / pending |
|---|---:|---:|---:|
| all | 38 | 30 / 7 / 1 | 29 / 8 / 1 |
| stress8 | 8 | 3 / 5 / 0 | 2 / 6 / 0 |
| fewshot30 | 30 | 27 / 2 / 1 | 27 / 2 / 1 |

Paired correctness on37 adjudicated nodes: {'harmed': 1, 'tied': 36}. Necessary-information retention: {'harmed': 2, 'tied': 35}. Only5 of38 answer texts differ.
The wrong-parent case changes Liu Yuan to Liu Yin. The Araripe case changes a correct named father into a reasonable abstention: its correctness label stays positive but its useful intermediate information disappears. No correctness repair appears in this comparison.
65 labels reuse exact prior answer text under identical actual-task context;11 outputs were newly reviewed locally. This is informed Codex review, not independent human gold. One malformed-entity case remains pending in both arms. Eight cases were selected stress cases. No end-to-end F1 is inferred from these ordinary-node labels.

## Goal: factual versus operation demonstrations

All270 goal conditions have terminal records. Statuses:{'success': 270}. The table uses the same90 scorable questions in all three arms. Returned algorithm failures are zero-score outcomes; unresolved technical cases remain excluded.
G0 preserves the original goal fact-sentence prompt at the common512-token cap. GF uses compact synthesis instructions and six fact/conclusion examples. GO adds explicit operation sentences to those same examples. All arms use the same neutral final wrapper and final64-token cap. Historical native P02 has a different native goal/final interface and is shown only as a reference.

| Dataset | Same n | Historical native F1 / EM | G0 F1 / EM | GF F1 / EM | GO F1 / EM |
|---|---:|---:|---:|---:|---:|
| all | 90 | 76.69 / 66.67 | 76.62 / 65.56 | 73.76 / 63.33 | 70.21 / 58.89 |
| hotpotqa | 30 | 81.99 / 70.00 | 81.99 / 70.00 | 75.75 / 60.00 | 73.20 / 60.00 |
| 2wiki | 30 | 94.33 / 90.00 | 92.80 / 86.67 | 94.13 / 90.00 | 91.69 / 83.33 |
| musique | 30 | 53.74 / 40.00 | 55.07 / 40.00 | 51.42 / 40.00 | 45.74 / 33.33 |

| Comparison | Improved / harmed / tied | F1 change (points) | Dataset-stratified95% bootstrap interval |
|---|---:|---:|---:|
| G0_to_GF | 8 / 8 / 74 | -2.86 | [-7.89, 1.71] |
| G0_to_GO | 6 / 13 / 71 | -6.41 | [-11.50, -1.92] |
| GF_to_GO | 4 / 9 / 77 | -3.55 | [-8.11, 0.46] |

Bootstrap intervals use10000 paired resamples, stratified by dataset with equal dataset weights. They describe this exposed development sample, not independent confirmation. The ordinary-present84 sensitivity and output-format counts are in OFFLINE_AUDIT.json.

## Connection with the preceding research

The following macro means use exactly the same86 questions that are scorable in both this experiment and the preceding G2/I/F experiment. The runs are nonconcurrent. All goal input records are byte-identical; native historical P02 retains its separate protocol. This prevents comparing different missing-case denominators.

| Arm | Dataset-equal F1 | Dataset-equal EM |
|---|---:|---:|
| historical_native_P02 | 76.19 | 66.41 |
| G0 | 76.13 | 65.30 |
| G2 | 68.75 | 58.46 |
| I | 70.14 | 59.40 |
| F | 68.90 | 58.12 |
| GF | 73.99 | 63.85 |
| GO | 70.30 | 59.40 |

## Expense and reproducibility

New returned-usage expense for the entire round:CNY1.299408. Preserved unresolved reservations:CNY0.000000. Reservations are not confirmed expense. The audit reconciles every returned usage record to this experiment's ledger namespace.

| Goal arm | Same successful n | Input tokens/query | Output tokens/query | Goal+final CNY/query |
|---|---:|---:|---:|---:|
| G0 | 90 | 4525.8 | 36.5 | 0.003694 |
| GF | 90 | 5403.8 | 90.0 | 0.004503 |
| GO | 90 | 5559.6 | 103.3 | 0.004654 |

These per-query figures are logical goal+final costs on the same successful cases, including any exact current-run duplicate reuse. They are not newly measured complete-RAG cost or timing. No LLM-call-count metric is added.

Frozen design and prompts are in the experiment directory. Per-condition outputs, source contexts, statement reviews, changed-final cases and the independent offline recount are retained in this output directory. No result has been promoted to the main method or published.
