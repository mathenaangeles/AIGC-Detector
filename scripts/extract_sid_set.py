"""Extract SID_Set parquet shards to image files, preserving original bytes.

The image column holds the original encoded file, so it is written verbatim.
Never round-trip through PIL here: re-encoding rewrites the JPEG quantisation
tables that shortcut.py reads and that the SRM branch depends on.

Labels: 0 real, 1 full synthetic, 2 tampered. Tampered is extracted but excluded
from binary training -- a tampered image did pass through a camera pipeline.

Usage: uv run python scripts/extract_sid_set.py [--limit-per-shard N]
"""

import argparse
import collections
import glob
import os
import sys

import pyarrow.parquet as pq

LABEL_DIR = {0: "real", 1: "synthetic", 2: "tampered"}
MAGIC = [(b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"RIFF", "webp"),
         (b"GIF8", "gif"), (b"BM", "bmp")]


def extension_of(raw):
    for magic, ext in MAGIC:
        if raw.startswith(magic):
            return ext
    return "bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/SID_Set/data")
    ap.add_argument("--out", default="data/train/sid_set")
    ap.add_argument("--limit-per-shard", type=int, default=None)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "*.parquet")))
    if not shards:
        print(f"no parquet shards under {args.src}", file=sys.stderr)
        return 2
    for name in LABEL_DIR.values():
        os.makedirs(os.path.join(args.out, name), exist_ok=True)

    stats = collections.Counter()
    fmt_by_label = collections.Counter()
    for n, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        seen = 0
        # Skip the mask column: large, and unused by the binary task.
        for batch in pf.iter_batches(batch_size=64, columns=["img_id", "image", "label"]):
            for row in batch.to_pylist():
                if args.limit_per_shard and seen >= args.limit_per_shard:
                    break
                seen += 1
                raw = row["image"]["bytes"]
                ext = extension_of(raw)
                dest = os.path.join(args.out, LABEL_DIR[row["label"]], f"{row['img_id']}.{ext}")
                fmt_by_label[(row["label"], ext)] += 1
                if os.path.exists(dest) and os.path.getsize(dest) == len(raw):
                    stats["skipped"] += 1
                    continue
                with open(dest, "wb") as f:
                    f.write(raw)
                stats[LABEL_DIR[row["label"]]] += 1
            if args.limit_per_shard and seen >= args.limit_per_shard:
                break
        print(f"[{n}/{len(shards)}] {os.path.basename(shard)} "
              f"real={stats['real']} synthetic={stats['synthetic']} "
              f"tampered={stats['tampered']} skipped={stats['skipped']}", flush=True)

    print("\nformat by label (the confound, measured):")
    for (label, ext), count in sorted(fmt_by_label.items()):
        print(f"  {label} {LABEL_DIR[label]:<10} {ext:<5} {count:>7}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
