"""Temperature-scale a checkpoint and select its fixed 1% FPR threshold."""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_manifest, select
from .evaluate import auc, fixed_fpr_threshold
from .inference import (
    TTA_NAMES,
    CheckpointPredictor,
    checkpoint_sha256,
    prediction_from_margins,
)


def _torch_predictions(margin_groups, log_temperature):
    temperature = log_temperature.exp().clamp(0.05, 100.0)
    predictions = []
    for values in margin_groups:
        probabilities = torch.sigmoid(
            torch.as_tensor(values, dtype=torch.float64) / temperature)
        probabilities = probabilities.sort().values
        if len(probabilities) >= 3:
            probabilities = probabilities[1:-1]
        predictions.append(probabilities.mean())
    return torch.stack(predictions)


def binary_nll(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-(labels * np.log(probabilities)
                   + (1.0 - labels) * np.log(1.0 - probabilities)).mean())


def fit_temperature(margin_groups, labels, max_iter=100):
    """Minimise image-level NLL after per-view scalar temperature scaling."""
    if not margin_groups or len(margin_groups) != len(labels):
        raise ValueError("margin groups and labels must be non-empty and equally sized")
    target = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=int(max_iter),
        tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        probabilities = _torch_predictions(margin_groups, log_temperature)
        loss = F.binary_cross_entropy(
            probabilities.clamp(1e-12, 1.0 - 1e-12), target)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 100.0))
    if not math.isfinite(temperature):
        raise FloatingPointError("temperature optimisation produced a non-finite result")
    return temperature


def calibrated_predictions(margin_groups, temperature):
    return np.asarray([
        prediction_from_margins(values, temperature)[0]
        for values in margin_groups
    ])


def write_report(path, result):
    lines = [
        "# Temperature Calibration",
        "",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Protocol: `{result['protocol']}`",
        f"- Split: `{result['split']}` ({result['n_images']:,} images)",
        f"- Temperature: `{result['temperature']:.6f}`",
        f"- NLL before/after: `{result['nll_before']:.6f}` / `{result['nll_after']:.6f}`",
        f"- ROC AUC: `{result['auc']:.6f}`",
        f"- Target FPR: `{100 * result['target_fpr']:.2f}%`",
        f"- Threshold: `{result['threshold']:.6f}`",
        f"- Achieved FPR: `{100 * result['achieved_fpr']:.2f}%`",
        f"- Accuracy at threshold: `{result['accuracy']:.6f}`",
        "",
        "The threshold is fitted on calibrated clean/TTA scores and must be used only with "
        "the checkpoint, protocol, crop grid, and TTA settings recorded in the JSON artifact.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="runs/p8-gated-kl1/model.pt")
    parser.add_argument("--protocol", choices=("raw", "bias_matched"),
                        default="bias_matched")
    parser.add_argument("--split", default=None,
                        help="override raw=val or bias_matched=val_matched")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_crops", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_fpr", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="calibration JSON; defaults beside the checkpoint")
    parser.add_argument("--report", default="reports/calibration.md")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_crops < 1 or args.num_workers < 0:
        parser.error("batch_size and max_crops must be positive; num_workers cannot be negative")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("overlap must be in [0, 1)")

    predictor = CheckpointPredictor(
        args.checkpoint, device=args.device, protocol=args.protocol)
    cfg = predictor.cfg
    split = args.split or ("val_matched" if args.protocol == "bias_matched" else "val")
    manifest = (str(cfg.data.val_matched_manifest) if split == "val_matched"
                else str(cfg.data.manifest))
    rows = select(load_manifest(manifest), split=split, labels=(0, 1))
    if args.limit:
        from .train import balanced_cap
        rows = balanced_cap(rows, int(args.limit))
    if not rows:
        parser.error(f"no rows for split {split!r} in {manifest}")
    labels = np.asarray([int(row["label"] == 1) for row in rows])
    if len(np.unique(labels)) < 2:
        parser.error("calibration requires both real and synthetic classes")
    target_fpr = float(
        args.target_fpr if args.target_fpr is not None else cfg.calibrate.target_fpr)
    if not 0.0 <= target_fpr <= 1.0:
        parser.error("target_fpr must be between 0 and 1")

    paths = [os.path.join(str(cfg.paths.root), row["path"]) for row in rows]
    print(f"checkpoint  {args.checkpoint}")
    print(f"protocol    {args.protocol}")
    print(f"split       {split} ({len(rows):,} images)")
    print(f"views       <= {args.max_crops} crops x {len(TTA_NAMES)} TTA")
    print(f"device      {predictor.device}\n")
    margin_groups, _, crop_counts = predictor.score_margins(
        paths, batch_size=args.batch_size, max_crops=args.max_crops,
        overlap=args.overlap, num_workers=args.num_workers)

    uncalibrated = calibrated_predictions(margin_groups, 1.0)
    temperature = fit_temperature(margin_groups, labels)
    calibrated = calibrated_predictions(margin_groups, temperature)
    threshold = fixed_fpr_threshold(calibrated, labels, target_fpr)
    predictions = calibrated >= threshold
    achieved_fpr = float(predictions[labels == 0].mean())

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "calibration.json")
    result = {
        "schema_version": 1,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "protocol": args.protocol,
        "split": split,
        "manifest": manifest,
        "n_images": len(rows),
        "n_real": int(np.sum(labels == 0)),
        "n_synthetic": int(np.sum(labels == 1)),
        "crop_size": int(cfg.data.crop_size),
        "max_crops": int(args.max_crops),
        "mean_crops": float(np.mean(crop_counts)),
        "overlap": float(args.overlap),
        "tta": list(TTA_NAMES),
        "aggregation": "trimmed_mean",
        "stability": "population_variance_over_crop_tta_probabilities",
        "temperature": temperature,
        "nll_before": binary_nll(uncalibrated, labels),
        "nll_after": binary_nll(calibrated, labels),
        "auc": auc(calibrated, labels),
        "target_fpr": target_fpr,
        "threshold": threshold,
        "achieved_fpr": achieved_fpr,
        "accuracy": float(np.mean(predictions == labels)),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    write_report(args.report, result)

    print(f"temperature {temperature:.6f}")
    print(f"NLL         {result['nll_before']:.6f} -> {result['nll_after']:.6f}")
    print(f"threshold   {threshold:.6f} @ {100 * achieved_fpr:.2f}% FPR")
    print(f"\n-> {out}\n-> {args.report}")


if __name__ == "__main__":
    main()
