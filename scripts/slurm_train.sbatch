#!/bin/bash
#SBATCH --job-name=aigc-p7-train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=aigc-train-%j.out

set -euo pipefail

# Slurm executes a spooled copy under /var/lib/slurm/slurmd, so BASH_SOURCE
# does not point into the repository inside a submitted job.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
mkdir -p "$TMPDIR"

command -v uv >/dev/null || { echo "uv is not available on the compute node" >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"

BRANCH_MODE="${BRANCH_MODE:-both}"
case "$BRANCH_MODE" in
  both) BRANCHES="clip,srm" ;;
  clip|srm) BRANCHES="$BRANCH_MODE" ;;
  *) echo "BRANCH_MODE must be one of: both, clip, srm" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
LAMBDA_CONSISTENCY="${LAMBDA_CONSISTENCY:-1.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-auc_val_matched}"
LIMIT="${LIMIT:-}"
RUN_TAG="${RUN_TAG:-p7-${BRANCHES//,/-}-${SLURM_JOB_ID:-manual}}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/runs/$RUN_TAG}"

echo "job          ${SLURM_JOB_ID:-manual}"
echo "host         $(hostname)"
echo "repository   $PROJECT_ROOT"
echo "branches     $BRANCHES"
echo "batch        $BATCH_SIZE"
echo "epochs       $EPOCHS"
echo "lambda       $LAMBDA_CONSISTENCY"
echo "limit        ${LIMIT:-full}"
echo "output       $OUT_DIR"
nvidia-smi

PREFLIGHT=(--require-cuda)
if [[ "$BRANCHES" == *clip* ]]; then
  PREFLIGHT+=(--require-cache)
fi
uv run --frozen python scripts/preflight_train.py "${PREFLIGHT[@]}"
TRAIN_ARGS=(
  --branches "$BRANCHES"
  --device cuda
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --lambda_consistency "$LAMBDA_CONSISTENCY"
  --early_stop_metric "$EARLY_STOP_METRIC"
  --out "$OUT_DIR"
)
if [[ -n "$LIMIT" ]]; then
  TRAIN_ARGS+=(--limit "$LIMIT")
fi
uv run --frozen python -m provenance.train "${TRAIN_ARGS[@]}"
