#!/usr/bin/env bash
#SBATCH --job-name=demand-parser
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Generic launcher only. Stage scripts must implement work, signal handling and
# resumable checkpoints; see REMOTE_IMPLEMENTATION.txt. Submit from the new run
# directory so Slurm logs stay with this experiment.
# Usage: sbatch /absolute/path/run_stage.slurm \
#          /absolute/path/venv/bin/python /absolute/path/train.py [stage args...]

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'ERROR: GPU stages require a valid Slurm allocation.' >&2
    exit 2
fi
if [[ "$#" -lt 2 ]]; then
    printf '%s\n' 'Usage: run_stage.slurm ABSOLUTE_PYTHON ABSOLUTE_STAGE_SCRIPT [args...]' >&2
    exit 2
fi

stage_python="$1"
stage_script="$2"
shift 2

if [[ "$stage_python" != /* || ! -f "$stage_python" || ! -x "$stage_python" ]]; then
    printf '%s\n' 'ERROR: Python must be an existing absolute executable path.' >&2
    exit 2
fi
if [[ "$stage_script" != /* || ! -f "$stage_script" || ! -r "$stage_script" ]]; then
    printf '%s\n' 'ERROR: stage script must be an existing absolute readable path.' >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf 'Slurm job: %s; node: %s; working directory: %s\n' \
    "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}" "$PWD"
printf 'Python: %s\nStage script: %s\n' "$stage_python" "$stage_script"

# This preflight runs on the allocated batch node, never on the login node.
# CUDA_VISIBLE_DEVICES is provided by Slurm; do not override its allocation.
"$stage_python" - <<'PY'
import json
import os
import platform
import torch

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("No Slurm allocation found")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable inside the allocation")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Expected one visible GPU, got {torch.cuda.device_count()}")
gpu = torch.cuda.get_device_name(0)
if "A6000" not in gpu.upper():
    raise SystemExit(f"Expected the requested A6000 GPU, got {gpu}")
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": gpu,
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}, ensure_ascii=False), flush=True)
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The planned BF16 run is unsupported on this allocation")
PY

# Slurm sends the advance USR1 signal to this srun job step. The Python stage
# must handle it; this launcher does not implement checkpointing itself.
exec srun --ntasks=1 --unbuffered "$stage_python" "$stage_script" "$@"
