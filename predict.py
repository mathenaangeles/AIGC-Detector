#!/usr/bin/env python3
"""AIGC Detector inference entry point.

Contract (graded deliverable):
    python predict.py --image_dir DIR [--out predictions.json]
    -> JSON array of {"image_path": str, "pred": float}
       where pred is P(AI-generated), in [0, 1].

Must run on CPU. This file deliberately imports only the standard library so it
keeps working before `uv sync`, on a judge's bare Python, and while the model
code underneath it is being rewritten. When real weights land, gate the model
import inside predict_paths() and flip STUB.
"""

import argparse
import json
import os
import sys

STUB = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(image_dir):
    """Recursively collect image paths in deterministic sorted order."""
    found = []
    for root, dirs, files in os.walk(image_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__MACOSX")
        for name in files:
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def format_path(path, image_dir, mode):
    if mode == "absolute":
        return os.path.abspath(path)
    if mode == "basename":
        return os.path.basename(path)
    return os.path.relpath(path, image_dir)


def predict_paths(paths):
    """Return P(AI-generated) per path. Stub: uninformative 0.5 for every image."""
    return [0.5 for _ in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict P(AI-generated) for a directory of images.")
    parser.add_argument("--image_dir", required=True, help="Directory of images, searched recursively.")
    parser.add_argument("--out", default="predictions.json", help="Output JSON path.")
    parser.add_argument(
        "--path_mode",
        default="relative",
        choices=("relative", "absolute", "basename"),
        help="How image_path is written in the output.",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.image_dir):
        parser.error(f"--image_dir is not a directory: {args.image_dir}")

    paths = find_images(args.image_dir)
    if not paths:
        parser.error(f"no images found under {args.image_dir} (looked for {sorted(IMAGE_EXTENSIONS)})")

    preds = predict_paths(paths)
    records = [
        {"image_path": format_path(p, args.image_dir, args.path_mode), "pred": float(s)}
        for p, s in zip(paths, preds)
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    banner = " [STUB: constant 0.5, not a trained model]" if STUB else ""
    print(f"wrote {len(records)} predictions to {args.out}{banner}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
