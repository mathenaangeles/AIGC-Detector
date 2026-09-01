#!/bin/bash
#SBATCH --job-name=aigc-p9-eval
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=aigc-eval-%j.out

set -euo pipefail

# Slurm executes a spooled copy, so the submission directory is the repository.
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

PROTOCOL="${PROTOCOL:-bias_matched}"
case "$PROTOCOL" in
  raw|bias_matched) ;;
  *) echo "PROTOCOL must be raw or bias_matched" >&2; exit 2 ;;
esac

EVAL_RUNS="${EVAL_RUNS:-clip=runs/p7-clip-kl1 srm=runs/p7-srm-kl1 both_no_kl=runs/p7-both-kl0 both_kl=runs/p7-both-kl1 gated=runs/p8-gated-kl1}"
read -r -a RUN_ARGS <<< "$EVAL_RUNS"
BATCH_SIZE="${BATCH_SIZE:-32}"
CROPS_PER_IMAGE="${CROPS_PER_IMAGE:-8}"
LIMIT="${LIMIT:-}"
SPLIT="${SPLIT:-}"
OUT="${OUT:-$PROJECT_ROOT/reports/robustness_table.md}"

echo "job          ${SLURM_JOB_ID:-manual}"
echo "host         $(hostname)"
echo "repository   $PROJECT_ROOT"
echo "protocol     $PROTOCOL"
echo "split        ${SPLIT:-protocol default}"
echo "runs         $EVAL_RUNS"
echo "batch        $BATCH_SIZE"
echo "crops        $CROPS_PER_IMAGE"
echo "limit        ${LIMIT:-full}"
echo "output       $OUT"
nvidia-smi

EVAL_ARGS=(
  --runs "${RUN_ARGS[@]}"
  --protocol "$PROTOCOL"
  --device cuda
  --batch_size "$BATCH_SIZE"
  --crops_per_image "$CROPS_PER_IMAGE"
  --out "$OUT"
)
if [[ -n "$LIMIT" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$SPLIT" ]]; then
  EVAL_ARGS+=(--split "$SPLIT")
fi

uv run --frozen python -m provenance.evaluate "${EVAL_ARGS[@]}"
