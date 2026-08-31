#!/bin/bash
#SBATCH --job-name=aigc-p7-cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=aigc-cache-%j.out

set -euo pipefail

# Slurm executes a spooled copy under /var/lib/slurm/slurmd, so BASH_SOURCE
# does not point into the repository inside a submitted job.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export PATH="$HOME/.local/bin:$PATH"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR:-/tmp/aigc-${USER}-${SLURM_JOB_ID}}"
else
  export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
fi
mkdir -p "$TMPDIR"

command -v uv >/dev/null || { echo "uv is not available on the compute node" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"

echo "job          ${SLURM_JOB_ID:-manual}"
echo "host         $(hostname)"
echo "repository   $PROJECT_ROOT"
echo "batch        $CACHE_BATCH_SIZE"
nvidia-smi

uv run --frozen python scripts/preflight_train.py --require-cuda
uv run --frozen python scripts/cache_features.py \
  --split train val \
  --device cuda \
  --batch_size "$CACHE_BATCH_SIZE"
uv run --frozen python scripts/preflight_train.py --require-cuda --require-cache
