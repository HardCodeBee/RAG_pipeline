Execution: v12 + reviewed3500, original Qwen3-4B-Instruct-2507

Read in this order:
1. outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017/EXECUTION_REPORT.txt
2. second_review/batch_*.json and replacement_labels.reviewed.json: each retained/revised/quarantined label decision.
3. prepare_data.py, validate_data.py: frozen v002, independent dev150 and train-only diagnostic150.
4. remote_v003/code/contract.py, train.py, launch.py: current submitted schedule, isolated smoke, fresh SFT.
5. select_checkpoint.py, predict_selected.py: dev-only checkpoint choice and same300 final evaluation.

Current run: Slurm153978; remote root /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003.
Current data: outputs/query_decomposition/multigrain_4b_sft3500_v1_20261017/data_v003_exact_v12/.
v002 labels remain unchanged; v003 retains the exact v12 final newline in every system message.
The original remote/ and stage.py describe preserved v002153977, cancelled while pending before any training.
Submission is not evidence that training started. Read Slurm and results/smoke/SMOKE_VERIFIED.json,
results/train/metrics.jsonl, results/train/STATE.json and actual checkpoint COMPLETE.json.

No training/API teacher was used to construct these labels. Current Codex authored/reviewed them;
this is not independent human gold. No API calls in this execution. Old training/data/adapters remain unchanged.

After training finishes:
  On remote login node (packaging only, no model load):
    REMOTE_PY export_return.py --package REMOTE_PACKAGE --archive /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/training_return.zip
  The exporter is outside the immutable training bundle; upload it separately first.
  Download and validate RETURN_MANIFEST.json and all members before local extraction.
  Audit all150 literal dev predictions for every available checkpoint110/165/220/275/330.
  Each review JSONL row must include id, raw_output, verdict(pass/fail/uncertain), reason and failed_dimensions.
  Run select_checkpoint.py RETURNED_DIR --reviews REVIEW_DIR.
  Stage SELECTED_CHECKPOINT.json with the frozen eval300 file SHA256 added as final_queries_sha256.
  Submit predict_selected.py to Slurm using the same verified base/v12/greedy settings.
  Review the full300 outputs, retain errors/uncertainty in denominator and compute paired changes.

Checkpoint choice cannot use eval300 or the old prompt-selection100. References are for dev loss;
semantic correctness is not JSON-string equality to one authored decomposition.
If training is interrupted, preserve all attempts. Explicit verified resume uses train.py --resume CHECKPOINT;
uncommitted metrics/predictions after that checkpoint must be archived before replay, not silently merged.
