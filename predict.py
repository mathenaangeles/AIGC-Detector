#!/usr/bin/env python3
"""AIGC Detector real-checkpoint inference entry point.

Contract (graded deliverable):
    python predict.py --image_dir DIR [--out predictions.json]
    -> JSON array of {"image_path": str, "pred": float}
       where pred is P(AI-generated), in [0, 1].

The model path uses overlapping crops and four transformation-time-augmentation
views per crop. The required JSON remains deliberately minimal; crop/TTA
variance and the calibrated operating-point decision go to a detailed sidecar.
"""

import argparse
import json
import os
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_CHECKPOINT = "runs/p8-gated-kl1/model.pt"


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


def predict_paths(paths, checkpoint=DEFAULT_CHECKPOINT, calibration=None, device="cpu",
                  protocol="bias_matched", batch_size=4, max_crops=8,
                  overlap=0.5, num_workers=0):
    """Return detailed real-model predictions in the input path order."""
    from provenance.inference import CheckpointPredictor

    predictor = CheckpointPredictor(
        checkpoint, device=device, calibration=calibration, protocol=protocol)
    return predictor.predict_paths(
        list(paths), batch_size=batch_size, max_crops=max_crops,
        overlap=overlap, num_workers=num_workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict P(AI-generated) for a directory of images.")
    parser.add_argument("--image_dir", required=True, help="Directory of images, searched recursively.")
    parser.add_argument("--out", default="predictions.json", help="Output JSON path.")
    parser.add_argument("--detailed_out", default=None,
                        help="Detailed sidecar; defaults to predictions_detailed.json beside --out.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--calibration", default="auto",
        help="Calibration JSON, 'auto' for one beside the checkpoint, or 'none'.",
    )
    parser.add_argument("--device", default="cpu",
                        help="Inference device. The required portable path is --device cpu.")
    parser.add_argument("--protocol", choices=("raw", "bias_matched"),
                        default="bias_matched")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_crops", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--path_mode",
        default="relative",
        choices=("relative", "absolute", "basename"),
        help="How image_path is written in the output.",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.image_dir):
        parser.error(f"--image_dir is not a directory: {args.image_dir}")
    if args.batch_size < 1 or args.max_crops < 1 or args.num_workers < 0:
        parser.error("batch_size and max_crops must be positive; num_workers cannot be negative")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("overlap must be in [0, 1)")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")

    paths = find_images(args.image_dir)
    if not paths:
        parser.error(f"no images found under {args.image_dir} (looked for {sorted(IMAGE_EXTENSIONS)})")

    if args.calibration == "auto":
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                 "calibration.json")
        calibration = candidate if os.path.isfile(candidate) else None
    elif args.calibration.lower() == "none":
        calibration = None
    else:
        calibration = args.calibration
        if not os.path.isfile(calibration):
            parser.error(f"calibration not found: {calibration}")

    details = predict_paths(
        paths, checkpoint=args.checkpoint, calibration=calibration,
        device=args.device, protocol=args.protocol, batch_size=args.batch_size,
        max_crops=args.max_crops, overlap=args.overlap,
        num_workers=args.num_workers)
    records = [
        {"image_path": format_path(path, args.image_dir, args.path_mode),
         "pred": float(detail["pred"])}
        for path, detail in zip(paths, details)
    ]
    detailed_records = [
        {"image_path": record["image_path"], **detail}
        for record, detail in zip(records, details)
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    detailed_out = args.detailed_out or os.path.join(
        out_dir, "predictions_detailed.json")
    os.makedirs(os.path.dirname(os.path.abspath(detailed_out)), exist_ok=True)
    with open(detailed_out, "w") as f:
        json.dump(detailed_records, f, indent=2)

    calibration_note = f"calibrated with {calibration}" if calibration else "uncalibrated"
    print(f"wrote {len(records)} predictions to {args.out}", file=sys.stderr)
    print(f"wrote stability details to {detailed_out} ({calibration_note})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
