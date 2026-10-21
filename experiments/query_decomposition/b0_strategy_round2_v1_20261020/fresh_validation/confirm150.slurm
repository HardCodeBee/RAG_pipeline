#!/usr/bin/env bash
#SBATCH --job-name=b0r2-confirm150
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=infer-%j.out
#SBATCH --error=infer-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec bash /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003/runtime/run_stage.slurm /home/projects/u7830488/demand_parser_route2_training_v1/20261009T191901Z/route2_question_decomposer/environment/venv/bin/python /home/projects/u7830488/b0_strategy_round2_v1_20261020/confirm150/remote_infer_id_query_v001.py --base-package /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003 --checkpoint /home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v001/training40/r02_expanded/checkpoint-final --checkpoint-sha 728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92 --queries /home/projects/u7830488/b0_strategy_round2_v1_20261020/confirm150/sealed150.id_query.frozen.jsonl --queries-sha 585b91d8f5094f9c472a83f44a9dee3660770b99f8b70d5af952450618ea8b94 --prompt /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003/v12.txt --output /home/projects/u7830488/b0_strategy_round2_v1_20261020/confirm150/results
