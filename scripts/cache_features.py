"""Run the frozen CLIP tower once and cache its patch tokens.

The tower is frozen, so its output per crop is a constant. Paying for it once
turns every later head-training run into a memmap read. This script is the only
thing that ever loads ViT-L/14.

    STORAGE. 512 KiB per crop (256 tokens x 1024 dims x fp16), so
    crops_per_image x 512 KiB per image -- 2 MiB/image at the default 4.
    The estimate is printed before any work starts. Use --limit first.

Resumable: crops already in the index are skipped, so an interrupted run
continues where it stopped. A cache built under different crop, seed or
bias_match settings is refused rather than silently reused.

Usage:
    uv run python scripts/cache_features.py --limit 32          # smoke test
    uv run python scripts/cache_features.py --split train val
"""

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provenance.branches.clip_probe import (  # noqa: E402
    FeatureCache,
    FrozenCLIP,
    cache_dir,
    cache_meta,
    enumerate_crops,
    load_crop,
    resolve_device,
    to_pixels,
)
from provenance.data import load_manifest, select, stratified_subset  # noqa: E402


def human(n_bytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n_bytes < 1024 or unit == "TiB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on images, balanced across labels, for smoke tests")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--shard_size", type=int, default=None)
    parser.add_argument("--allow_eval", action="store_true",
                        help="cache data/eval/ features; for evaluation only, never for fitting")
    parser.add_argument("--dry_run", action="store_true", help="report the plan and stop")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if "eval" in args.split and not args.allow_eval:
        parser.error("data/eval/ is held out; pass --allow_eval if this is for evaluation")

    torch.manual_seed(int(cfg.seed))
    rows = select(load_manifest(str(cfg.data.manifest)), split=args.split,
                  labels=(0, 1, 2) if cfg.data.include_tampered else (0, 1))
    rows.sort(key=lambda r: r["path"])
    if args.limit:
        # Balanced, not the first N by path: the manifest is path-sorted, so a
        # plain head() of it is one class and a smoke test proves nothing.
        rows = stratified_subset(rows, max(1, int(args.limit) // len(set(r["label"] for r in rows))),
                                 key=lambda r: r["label"], seed=int(cfg.seed))
        rows.sort(key=lambda r: r["path"])
    if not rows:
        parser.error(f"no rows for split {args.split} in {cfg.data.manifest}")

    out = args.out or cache_dir(cfg)
    meta = cache_meta(cfg)
    cache = FeatureCache(out, shard_size=int(args.shard_size or cfg.model.clip.cache_shard_size))
    cache.open_for_write(meta)

    crops = enumerate_crops(rows, cfg)
    todo = [c for c in crops if c[2] not in cache]
    size = int(cfg.data.crop_size)
    per_crop = (size // 14) ** 2 * 1024 * 2

    print(f"config      {args.config}   seed {cfg.seed}")
    print(f"backbone    {meta['arch']} / {meta['pretrained']}   frozen")
    print(f"images      {len(rows)}   splits {args.split}"
          + (f"   (--limit {args.limit})" if args.limit else ""))
    print(f"crops       {len(crops)} total, {len(crops) - len(todo)} cached, {len(todo)} to do")
    print(f"bias_match  {meta['bias_match']} @ q{meta['match_quality']}")
    print(f"storage     {human(per_crop)}/crop -> {human(per_crop * len(todo))} to write")
    print(f"cache       {out}")
    if args.dry_run or not todo:
        print("nothing to do" if not todo else "dry run, stopping")
        return

    device = resolve_device(args.device)
    print(f"device      {device}\n")
    backbone = FrozenCLIP(meta["arch"], meta["pretrained"], device=device)
    assert not backbone.visual.training and not any(p.requires_grad for p in backbone.parameters())

    started = time.time()
    batch, keys = [], []
    done = 0

    def run_batch():
        nonlocal batch, keys, done
        if not batch:
            return
        cls, tokens = backbone(torch.stack(batch).to(device))
        cls, tokens = cls.cpu().numpy(), tokens.cpu().numpy()
        for i, key in enumerate(keys):
            cache.add(key, tokens[i], cls[i])
        done += len(keys)
        batch, keys = [], []
        rate = done / max(time.time() - started, 1e-6)
        print(f"\r  {done}/{len(todo)} crops  {rate:.1f}/s  "
              f"eta {(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min", end="", flush=True)

    try:
        for row, box, key in todo:
            crop = load_crop(os.path.join(str(cfg.paths.root), row["path"]), box, size,
                             do_match=meta["bias_match"], quality=meta["match_quality"])
            batch.append(to_pixels(crop))
            keys.append(key)
            if len(batch) >= int(args.batch_size):
                run_batch()
        run_batch()
    finally:
        cache.flush()

    elapsed = time.time() - started
    print(f"\n\ncached {done} crops in {elapsed / 60:.1f} min "
          f"({human(per_crop * done)}); index holds {len(cache)}")


if __name__ == "__main__":
    main()
