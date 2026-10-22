# Result binding and goal operations

This is a bounded side-conversation development experiment authorized by the user on 2026-10-22. Main evaluation remains paused. The original R02/v12 decomposition, production source, old experiments and parent runners are preserved.

The previous fixed90 goal experiment did not establish an improvement: on its 86 matched scorable questions, revised instructions and four examples both had confidence intervals crossing zero against G2. That result motivates distinguishing demonstration content from surface compatibility.

## Ordinary answer experiment

Use exactly the existing 38 source nodes: eight stress cases and the previous thirty ordinary-example cases, with 35 distinct original queries. There are 21 nodes with history and 17 without. They are exposed development material, not a representative accuracy sample.

* D: existing descriptive-question demonstrations (E2).
* B: exactly D plus two example-only notes identifying the value requested by the preceding question, distinguishing it from the other entity in the answer, and showing how that value supplies the current question's argument.

Actual original/current questions, history, retrieved passages, instructions and decoding are unchanged. Both arms return the raw ordinary answer at 128 tokens. No sentence formatter or downstream rerun is included. Review factual correctness of the full statement, whether it answers the current question, and retention of necessary information separately. Reasonable abstention remains correct when the needed fact is unavailable; a correct but incomplete answer is not automatically wrong. Unresolved factual checks remain pending. This experiment cannot establish an end-to-end F1 improvement.

## Goal experiment

Use the same fixed90: 30 HotpotQA, 30 2Wiki and 30 MuSiQue. Six goal-only cases remain in the full cohort; report the 84 ordinary-present cases as sensitivity analysis. Ordinary QA, goal rewriting, retrieved packets and final root evidence are frozen.

* G0: original historical goal fact-sentence prompt, at the common 512-token cap.
* GF: concise synthesis instructions plus six fictional demonstrations containing facts and a conclusion.
* GO: identical instructions, fictional input facts and conclusions, with one essential-operation sentence added to each demonstration.

The six demonstrations cover directed relations, comparison and answer-type mapping, shared set membership, attribute selection and prior-answer correction, decisive partial information, and unresolved comparisons. Examples use fictional entities. Both GF and GO ask for the same concise synthesis output. Final generation has the same neutral candidate-information wrapper, unchanged root evidence and 64-token cap in all three arms.

GF to GO tests the effect of explicitly demonstrating the operation. It also adds input tokens, so this is not a length-controlled proof about reasoning itself. G0 to either synthesis arm changes a bundle of prompt/output properties. Historical native P02 is a separately marked reference because its goal cap and final interface differ.

## Execution and analysis

Fresh controls are interleaved in a deterministic per-case order. No historical-response seed is used; current-run exact duplicate payloads may reuse complete receipts. The existing provider gate and ledger account for reasonable spending. Known technical failures have at most five durable attempts; requests with uncertain submission status are held without reissue. Returned incomplete outputs are algorithm failures, not opportunities to reroll. The run ends after 346 condition outcomes or documented holds.

Use native dataset F1/EM, common-case paired gains/losses/ties and dataset-stratified paired bootstrap intervals. Returned algorithm failures remain zero-score outcomes. Unknown requests are excluded with explicit denominators. Compare token/cost measures on the same successful cases and record actual new expense separately. Inspect changed answers and operation failures locally without an external judging model.

No candidate is promoted from exposed development evidence alone. A positive result would justify a separate fresh complete-RAG confirmation on new questions. A negative result should narrow the hypothesis, rather than trigger unbounded prompt changes on these same cases.
