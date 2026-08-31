"""Validate the local data and cache state before starting an expensive P7 run.

Usage:
    uv run python scripts/preflight_train.py
    uv run python scripts/preflight_train.py --require-cuda --require-cache
"""

import argparse
import collections
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provenance.branches.clip_probe import (  # noqa: E402
    FeatureCache,
    cache_dir,
    cache_meta,
    crop_boxes,
)
from provenance.data import assert_eval_isolated, load_manifest, select  # noqa: E402


def human(n_bytes):
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def cache_keys(rows, n_crops, cfg):
    size, seed = int(cfg.data.crop_size), int(cfg.seed)
    keys = set()
    for row in rows:
        boxes = crop_boxes(int(row["width"]), int(row["height"]), size,
                           int(n_crops), seed, row["path"])
        keys.update(f"{row['path']}@{left},{top},{size}" for left, top in boxes)
    return keys


def label_counts(rows):
    return collections.Counter(int(row["label"] == 1) for row in rows)


def existing_parent(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    problems = []

    print("AIGC Detector P7 preflight")
    print(f"  config       {args.config}")
    print(f"  seed         {cfg.seed}")
    print(f"  torch        {torch.__version__}")
    print(f"  cuda         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu          {torch.cuda.get_device_name(0)}")
    elif args.require_cuda:
        problems.append("CUDA was required but torch.cuda.is_available() is false")

    manifest_path = os.path.abspath(str(cfg.data.manifest))
    matched_path = os.path.abspath(str(cfg.data.val_matched_manifest))
    if not os.path.exists(manifest_path):
        problems.append(f"missing manifest: {manifest_path}")
        rows = []
    else:
        rows = load_manifest(manifest_path)
        try:
            n_eval = assert_eval_isolated(rows, cfg)
        except AssertionError as exc:
            problems.append(str(exc))
            n_eval = 0
        print(f"  manifest     {manifest_path} ({len(rows):,} rows, {n_eval:,} eval isolated)")

    train_rows = select(rows, split="train", labels=(0, 1))
    val_rows = select(rows, split="val", labels=(0, 1))
    for name, split_rows in (("train", train_rows), ("val", val_rows)):
        counts = label_counts(split_rows)
        print(f"  {name:<12} {len(split_rows):,} images "
              f"(real {counts[0]:,}, synthetic {counts[1]:,})")
        if not counts[0] or not counts[1]:
            problems.append(f"{name} split does not contain both classes")

    missing_files = [row["path"] for row in train_rows + val_rows
                     if not os.path.isfile(os.path.join(str(cfg.paths.root), row["path"]))]
    if missing_files:
        problems.append(f"{len(missing_files)} train/val files are missing; first: {missing_files[0]}")
    else:
        print(f"  image files  all {len(train_rows) + len(val_rows):,} train/val paths exist")

    matched_rows = []
    if os.path.exists(matched_path):
        matched_rows = select(load_manifest(matched_path), split="val_matched", labels=(0, 1))
        counts = label_counts(matched_rows)
        print(f"  val_matched  {len(matched_rows):,} images "
              f"(real {counts[0]:,}, synthetic {counts[1]:,})")
        if not counts[0] or not counts[1]:
            problems.append("val_matched does not contain both classes")
    else:
        problems.append(f"missing matched-validation manifest: {matched_path}")

    needed = cache_keys(train_rows, int(cfg.data.crops_per_image), cfg)
    needed |= cache_keys(val_rows, 1, cfg)
    needed |= cache_keys(matched_rows, 1, cfg)
    token_bytes = (int(cfg.data.crop_size) // 14) ** 2 * 1024 * 2
    cache_path = os.path.abspath(cache_dir(cfg))
    index_path = os.path.join(cache_path, "index.json")
    cached = set()
    cache_ok = False

    if os.path.exists(index_path):
        try:
            cache = FeatureCache(cache_path)
            cache.check_compatible(cache_meta(cfg))
            cached = set(cache.index)
            cache_ok = needed.issubset(cached)
        except (OSError, ValueError) as exc:
            problems.append(f"incompatible or unreadable cache: {exc}")
    missing_cache = len(needed - cached)
    print(f"  clip cache   {len(needed) - missing_cache:,}/{len(needed):,} required crops covered")
    print(f"  cache path   {cache_path}")
    if missing_cache:
        print(f"  cache work   {missing_cache:,} crops, approximately "
              f"{human(missing_cache * token_bytes)} remaining")
    if args.require_cache and not cache_ok:
        problems.append(f"CLIP cache is incomplete: {missing_cache:,} required crops missing")

    disk = shutil.disk_usage(existing_parent(cache_path))
    print(f"  disk free    {human(disk.free)}")
    if missing_cache * token_bytes > disk.free * 0.9:
        problems.append("insufficient free disk for the estimated CLIP token cache")

    if problems:
        print("\nNOT READY")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nPREFLIGHT PASSED")
    print("  All requested checks passed. Use --require-cuda and --require-cache for a full run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
