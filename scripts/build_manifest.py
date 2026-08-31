"""Rebuild data/manifest.csv, and the bpp-matched val subset beside it.

Eval isolation is asserted inside build_manifest, not here, so every caller
gets the check whether or not it came through this script.

Usage:
    uv run python scripts/build_manifest.py
    uv run python scripts/build_manifest.py --no-val-matched   # skip the decode pass
"""

import argparse
import collections
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from provenance.data import build_manifest, build_val_matched  # noqa: E402

LABELS = {0: "real", 1: "synthetic", 2: "tampered"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-val-matched", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    rows, dropped, report = build_manifest(cfg, out_path=args.out, with_report=True)

    counts = collections.Counter((r["source"], r["split"], r["label"]) for r in rows)
    width = max(len(s) for s, _, _ in counts)
    print(f"{'source':<{width}}  {'split':<12}{'label':<12}{'n':>8}")
    print("-" * (width + 34))
    for (source, split, label), n in sorted(counts.items()):
        print(f"{source:<{width}}  {split:<12}{LABELS.get(label, label):<12}{n:>8}")
    print(f"\n{len(rows)} rows, {dropped} dropped below min_side {cfg.data.min_side}")
    print(f"eval isolation asserted: {report['n_eval']} rows are split=eval")

    for source, info in report.items():
        if isinstance(info, dict) and info.get("strategy") == "generator":
            print(f"\n{source}: generator holdout")
            print(f"  generators ({len(info['generators'])}): {', '.join(info['generators'])}")
            print(f"  held out for val: {', '.join(info['held_out'])}")
            for name in info["held_out"]:
                print(f"    {name:<24}{info['counts'][name]:>8} images")

    if not args.no_val_matched:
        matched, stats = build_val_matched(
            cfg, rows, caliper_sd=float(cfg.data.val_matched_caliper_sd))
        print(f"\nval_matched: {stats['n_matched']} rows "
              f"({stats['pairs']} pairs, {stats['dropped']} unmatched) "
              f"from {stats['n_val']} val rows")
        if stats["pairs"]:
            before, after = stats["bpp_mean_before"], stats["bpp_mean_after"]
            print(f"  caliper   {stats['caliper']:.5f} bpp   "
                  f"mean gap {stats['mean_gap']:.5f}   max {stats['max_gap']:.5f}")
            print(f"  bpp mean  before  real {before['real']:.4f}  "
                  f"synthetic {before['synthetic']:.4f}  "
                  f"(delta {before['real'] - before['synthetic']:+.4f})")
            print(f"            after   real {after['real']:.4f}  "
                  f"synthetic {after['synthetic']:.4f}  "
                  f"(delta {after['real'] - after['synthetic']:+.4f})")
            print(f"  -> {cfg.data.val_matched_manifest}")

    print(f"\nseed {cfg.seed}  ->  {args.out or cfg.data.manifest}")


if __name__ == "__main__":
    main()
