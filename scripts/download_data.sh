#!/usr/bin/env bash
# Dataset acquisition for AIGC Detector.
#
# Targeted subset, not the full ~325 GB. Frozen CLIP + ~8.8M trainable params
# does not need 210k images; raise SID_TRAIN_SHARDS to scale up.
#
#   SID_Set   20/249 train + 2/34 val shards   ~10 GB   HuggingFace, public
#   WildFake  Real/coco, Real/imagenet, Other  ~17 GB   ModelScope, public
#   COCO      val2017                          815 MB   direct (eval, held out)
#   DALLE     Diffusion_based/DALLE.zip        25.6 GB  ModelScope (eval, held out)
#   CIFAKE                                     skipped  needs Kaggle credentials;
#                                                       32x32, sanity check only
#
# Usage:  scripts/download_data.sh [sid|wildfake|coco|dalle|all]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${DATA_RAW:-$ROOT/data/raw}"
SID_TRAIN_SHARDS="${SID_TRAIN_SHARDS:-20}"
SID_VAL_SHARDS="${SID_VAL_SHARDS:-2}"
mkdir -p "$RAW"

# The /repo/files? API endpoint returns the git-lfs pointer (136 bytes), not the
# blob. /resolve/ serves the real file, supports Range, and so resumes.
MS_RESOLVE="https://www.modelscope.cn/datasets/hy2628982280/WildFake/resolve/master"
MS_API="https://www.modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

fetch_sid() {
  log "SID_Set: $SID_TRAIN_SHARDS train + $SID_VAL_SHARDS val shards -> $RAW/SID_Set"
  SID_TRAIN_SHARDS="$SID_TRAIN_SHARDS" SID_VAL_SHARDS="$SID_VAL_SHARDS" DEST="$RAW/SID_Set" \
  uv run --project "$ROOT" python - <<'PY'
import os
from huggingface_hub import snapshot_download
n_tr, n_va = int(os.environ["SID_TRAIN_SHARDS"]), int(os.environ["SID_VAL_SHARDS"])
patterns = [f"data/train-{i:05d}-of-00249.parquet" for i in range(n_tr)]
patterns += [f"data/validation-{i:05d}-of-00034.parquet" for i in range(n_va)]
patterns += ["README.md", "config.json"]
p = snapshot_download("saberzl/SID_Set", repo_type="dataset", local_dir=os.environ["DEST"],
                      allow_patterns=patterns, max_workers=8)
print("SID_Set ->", p)
PY
}

# ModelScope serves blobs over plain HTTP with resume; curl -C - is enough.
ms_get() {
  local remote="$1" out="$RAW/WildFake/$(basename "$remote")"
  mkdir -p "$(dirname "$out")"
  log "WildFake: $remote"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$out" "$MS_RESOLVE/${remote}"
  if [ "$(wc -c <"$out")" -lt 1000 ]; then
    log "ERROR: $out is $(wc -c <"$out") bytes -- an lfs pointer, not the blob"; return 1
  fi
}

fetch_wildfake() {
  ms_get "Images/Real/coco.zip"
  ms_get "Images/Real/imagenet.zip"
  ms_get "Images/Other_based.zip"
  log "WildFake label CSVs"
  mkdir -p "$RAW/WildFake/label_csv_files"
  for c in real_coco real_imagenet dalle2 dalle3; do
    curl -sL --fail -o "$RAW/WildFake/label_csv_files/$c.csv" \
      "$MS_RESOLVE/label_csv_files/${c}.csv" || true
  done
}

fetch_coco() {
  log "COCO val2017 (eval, held out)"
  curl -L --fail --retry 3 -C - -o "$RAW/val2017.zip" \
    "http://images.cocodataset.org/zips/val2017.zip"
}

fetch_dalle() {
  log "WildFake DALLE.zip 25.6 GB (eval, held out) -- this is the long one"
  ms_get "Images/Diffusion_based/DALLE.zip"
}

case "${1:-all}" in
  sid)      fetch_sid ;;
  wildfake) fetch_wildfake ;;
  coco)     fetch_coco ;;
  dalle)    fetch_dalle ;;
  all)      fetch_coco; fetch_sid; fetch_wildfake; fetch_dalle ;;
  *) echo "usage: $0 [sid|wildfake|coco|dalle|all]" >&2; exit 2 ;;
esac
log "done"
