#!/bin/bash
#SBATCH --job-name=aigc-p11-errors
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=aigc-errors-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="${SLURM_TMPDIR:-/tmp/aigc-${USER}-${SLURM_JOB_ID:-manual}}"
mkdir -p "$TMPDIR"

command -v uv >/dev/null || { echo "uv is not available on the compute node" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"

CHECKPOINT="${CHECKPOINT:-runs/p8-gated-kl1/model.pt}"
PROTOCOL="${PROTOCOL:-bias_matched}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-8}"
TOP_K="${TOP_K:-12}"
LIMIT="${LIMIT:-}"
REPORT_OUT="${REPORT_OUT:-reports/error_analysis.md}"
SHEET_OUT="${SHEET_OUT:-reports/error_contact_sheet.png}"
JSON_OUT="${JSON_OUT:-reports/error_analysis.json}"

echo "job          ${SLURM_JOB_ID:-manual}"
echo "host         $(hostname)"
echo "repository   $PROJECT_ROOT"
echo "checkpoint   $CHECKPOINT"
echo "protocol     $PROTOCOL"
echo "batch        $BATCH_SIZE"
echo "crops        $CROPS_PER_IMAGE"
echo "top k        $TOP_K"
echo "limit        ${LIMIT:-full}"
echo "report       $REPORT_OUT"
echo "contact      $SHEET_OUT"
nvidia-smi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --protocol "$PROTOCOL"
  --device cuda
  --batch_size "$BATCH_SIZE"
  --crops_per_image "$CROPS_PER_IMAGE"
  --num_workers "${SLURM_CPUS_PER_TASK:-8}"
  --top_k "$TOP_K"
  --out "$REPORT_OUT"
  --contact_sheet "$SHEET_OUT"
  --json_out "$JSON_OUT"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

uv run --frozen python scripts/error_analysis.py "${ARGS[@]}"
