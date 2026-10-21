Read outputs/query_decomposition/multigrain_4b_post_sft_opt_v1_20261018/ERROR_ANALYSIS_AND_PLAN.txt first.
BASELINE_FROZEN.json preserves the immutable checkpoint165/v12 control.
prepare.py builds diagnostic evidence and two one-example prompt trials; it does not alter training.
report.py expands all22 literal failed cases, with source ambiguity and rubric limits retained.
run_prompts.py executes same checkpoint, same dev150, seed1729, greedy1024 on allocated A6000 only.
PROTOCOL_FROZEN.json records prior decisions; CHANGES.txt contains exact prompt diffs.
confirmation90.questions.frozen.jsonl was selected before confirmation generation; CONFIRMATION90_FROZEN.json records original desired and available quotas.
Original final300 outputs must not enter prompt optimization or development selection.
No API calls or new model training.
