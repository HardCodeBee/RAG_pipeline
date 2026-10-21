#!/usr/bin/env bash
#SBATCH --job-name=p02-formal1500
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/projects/u7830488/p02_formal1500_v1_20261021
#SBATCH --output=/home/projects/u7830488/p02_formal1500_v1_20261021/infer-%j.out
#SBATCH --error=/home/projects/u7830488/p02_formal1500_v1_20261021/infer-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec bash /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003/runtime/run_stage.slurm /home/projects/u7830488/demand_parser_route2_training_v1/20261009T191901Z/route2_question_decomposer/environment/venv/bin/python /home/projects/u7830488/p02_formal1500_v1_20261021/remote_infer.py --base-package /home/projects/u7830488/multigrain_4b_sft3500_v1_20261017/package_v003 --checkpoint /home/projects/u7830488/multigrain_4b_coverage_retrain_v3_20261018/results/v001/training40/r02_expanded/checkpoint-final --checkpoint-sha 728fb4761116debf44ca4c7078a09283939beeb590b35e4a8647e6131a9fce92 --queries /home/projects/u7830488/p02_formal1500_v1_20261021/queries1500.id_query.frozen.jsonl --queries-sha 6dae32373db4fbb1a21cc4ce28bb2f25ad846777b29407f7d1555d1bc09b0b5a --prompt /home/projects/u7830488/p02_formal1500_v1_20261021/v12.txt --output /home/projects/u7830488/p02_formal1500_v1_20261021/results
