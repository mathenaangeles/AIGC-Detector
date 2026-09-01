#!/bin/bash
#SBATCH --job-name=aigc-p10-calibrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=aigc-calibrate-%j.out

set -euo pipefail

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

CHECKPOINT="${CHECKPOINT:-runs/p8-gated-kl1/model.pt}"
PROTOCOL="${PROTOCOL:-bias_matched}"
# Four fits ViT-L/14 plus the two lightweight heads in FP32 on 12 GiB GPUs.
# BF16-capable GPUs can safely override this upward at submission time.
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CROPS="${MAX_CROPS:-8}"
OVERLAP="${OVERLAP:-0.5}"
LIMIT="${LIMIT:-}"
CALIBRATION_OUT="${CALIBRATION_OUT:-runs/p8-gated-kl1/calibration.json}"
REPORT_OUT="${REPORT_OUT:-reports/calibration.md}"

echo "job          ${SLURM_JOB_ID:-manual}"
echo "host         $(hostname)"
echo "repository   $PROJECT_ROOT"
echo "checkpoint   $CHECKPOINT"
echo "protocol     $PROTOCOL"
echo "batch        $BATCH_SIZE"
echo "max crops    $MAX_CROPS"
echo "overlap      $OVERLAP"
echo "limit        ${LIMIT:-full}"
echo "calibration  $CALIBRATION_OUT"
echo "report       $REPORT_OUT"
nvidia-smi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --protocol "$PROTOCOL"
  --device cuda
  --batch_size "$BATCH_SIZE"
  --max_crops "$MAX_CROPS"
  --overlap "$OVERLAP"
  --num_workers "${SLURM_CPUS_PER_TASK:-8}"
  --out "$CALIBRATION_OUT"
  --report "$REPORT_OUT"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

uv run --frozen python -m provenance.calibrate "${ARGS[@]}"
