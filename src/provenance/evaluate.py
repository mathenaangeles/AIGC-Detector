"""Checkpoint-driven clean-vs-transformed robustness grid.

Overall AUC on this data is not interpretable on its own. The shortcut probe
puts a content-blind logistic regression on the SID_Set train split at AUC
1.0000 raw and 0.7249 after bias matching, and the surviving 0.7249 comes from
exactly one feature: bytes per pixel after a common JPEG encode. Synthetic
content is simply more compressible than photographic content, and any detector
trained here can reach the mid-0.7s by learning that and nothing else.

So two controls are reported next to every headline number.

    bpp_only    A logistic regression on the single bpp scalar, fitted
                out-of-fold. This is the floor. A model that does not clear it
                by a wide margin has not learned anything the file size does not
                already say.

    stratified  AUC computed inside each of 5 bpp quantile bins and averaged.
                Within a bin the classes are close to bpp-matched, so the
                shortcut is largely unavailable and what is left is signal the
                model found in the pixels. The gap between overall and
                stratified AUC is the part of the score the confound is paying
                for.

Neither control makes the shortcut go away; they make it visible. `bpp_only`
fits on whatever split it is handed, so it is a measurement in the same sense
shortcut.py is, and nothing downstream reads its coefficients.

The CLI accepts one or more trained run directories/checkpoints, scores clean
and every configured transform severity, and writes AUC plus fixed-operating-
point accuracy tables. CLIP tokens are computed once per batch and shared across
all methods so an ablation compares identical pixels without repeating the
expensive frozen tower.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .branches.clip_probe import FrozenCLIP, crop_boxes, resolve_device
from .data import bias_match, load_manifest, reflect_pad_to, select
from .transforms import apply_named, eval_grid

N_BPP_BINS = 5


def auc(scores, labels):
    """ROC AUC, or None where it is undefined because one class is absent."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, np.asarray(scores)))


def quantile_bins(values, n_bins=N_BPP_BINS):
    """Assign each value to a quantile bin. Ties collapse bins rather than empty them."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right"), edges


def stratified_auc(scores, labels, values, n_bins=N_BPP_BINS):
    """Per-bin AUC over quantile bins of `values`, plus the mean over usable bins.

    A bin holding one class only has no AUC. It is reported with auc=None and
    left out of the mean rather than counted as 0.5, which would understate a
    good model and flatter a bad one.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    assignment, edges = quantile_bins(values, n_bins)

    bins = []
    for b in range(int(assignment.max()) + 1):
        mask = assignment == b
        bins.append({
            "bin": b,
            "n": int(mask.sum()),
            "n_positive": int(labels[mask].sum()),
            "value_lo": float(np.asarray(values)[mask].min()) if mask.any() else None,
            "value_hi": float(np.asarray(values)[mask].max()) if mask.any() else None,
            "auc": auc(scores[mask], labels[mask]),
        })

    usable = [b["auc"] for b in bins if b["auc"] is not None]
    return {
        "bins": bins,
        "edges": [float(e) for e in edges],
        "mean_auc": float(np.mean(usable)) if usable else None,
        "n_bins_used": len(usable),
        "n_bins_empty_of_a_class": len(bins) - len(usable),
    }


def bpp_baseline(bpp, labels, seed=1337, folds=5):
    """Out-of-fold AUC of a logistic regression on the bpp scalar alone.

    Out-of-fold because a one-feature model fitted and scored on the same rows
    still reports an optimistic AUC, and this number is used as a floor that
    other numbers are compared against.
    """
    X = np.asarray(bpp, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels)
    folds = int(min(folds, np.bincount(y).min()))
    if folds < 2:
        raise ValueError("need at least 2 examples of each class to cross-validate")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=int(seed)))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {"row": "bpp_only", "n": int(len(y)), "n_positive": int(y.sum()),
            "auc": float(roc_auc_score(y, oof)), "folds": folds, "scores": oof}


def evaluate_scores(scores, labels, bpp, name="model", n_bins=N_BPP_BINS):
    """One results row: overall AUC, bpp-stratified AUC, and the bins behind it."""
    return {
        "row": name,
        "n": int(len(labels)),
        "n_positive": int(np.asarray(labels).sum()),
        "auc": auc(scores, labels),
        "bpp_stratified": stratified_auc(scores, labels, bpp, n_bins),
    }


def compare_rows(labels, bpp, scored=None, seed=1337, folds=5, n_bins=N_BPP_BINS):
    """The results table: the bpp-only baseline row, then a row per scored model."""
    baseline = bpp_baseline(bpp, labels, seed=seed, folds=folds)
    rows = [{**evaluate_scores(baseline.pop("scores"), labels, bpp, "bpp_only", n_bins),
             "folds": baseline["folds"], "baseline": True}]
    for name, scores in (scored or {}).items():
        rows.append(evaluate_scores(scores, labels, bpp, name, n_bins))
    return rows


def format_table(rows):
    """Results table: overall AUC, bpp-stratified mean, the gap, and each bin."""
    n_bins = max(len(r["bpp_stratified"]["bins"]) for r in rows)
    header = (f"{'row':<16}{'n':>7}{'AUC':>9}{'bpp-strat':>11}{'gap':>8}  "
              + "".join(f"{'bin' + str(i):>8}" for i in range(n_bins)))
    lines = [header, "-" * len(header)]
    for row in rows:
        strat = row["bpp_stratified"]
        mean = strat["mean_auc"]
        gap = None if (mean is None or row["auc"] is None) else row["auc"] - mean
        per_bin = "".join(f"{b['auc']:>8.3f}" if b["auc"] is not None else f"{'--':>8}"
                          for b in strat["bins"])
        lines.append(
            f"{row['row']:<16}{row['n']:>7}"
            f"{row['auc']:>9.4f}" + (f"{mean:>11.4f}" if mean is not None else f"{'--':>11}")
            + (f"{gap:>8.4f}" if gap is not None else f"{'--':>8}") + "  " + per_bin
        )
    lines.append("-" * len(header))
    lines.append("gap = overall minus bpp-stratified mean: the score the confound pays for.")
    return "\n".join(lines)


def to_tensor(img):
    array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def conform(img, size):
    """Restore a transformed crop to model dimensions without resampling."""
    img = reflect_pad_to(img.convert("RGB"), int(size))
    width, height = img.size
    left, top = (width - size) // 2, (height - size) // 2
    return img.crop((left, top, left + size, top + size))


class RobustnessDataset(Dataset):
    """Deterministic native-resolution crops for one robustness condition."""

    def __init__(self, rows, cfg, protocol, condition=None, n_crops=None):
        self.rows = list(rows)
        self.root = str(cfg.paths.root)
        self.size = int(cfg.data.crop_size)
        self.seed = int(cfg.seed)
        self.protocol = str(protocol)
        self.quality = int(cfg.data.match_quality)
        self.condition = condition
        self.n_crops = int(n_crops or cfg.eval.crops_per_image)
        if self.protocol not in ("raw", "bias_matched"):
            raise ValueError(f"unknown protocol {protocol!r}")

    def __len__(self):
        return len(self.rows) * self.n_crops

    def __getitem__(self, index):
        row_index, crop_index = divmod(index, self.n_crops)
        row = self.rows[row_index]
        boxes = crop_boxes(int(row["width"]), int(row["height"]), self.size,
                           self.n_crops, self.seed, row["path"])
        left, top = boxes[crop_index]
        with Image.open(os.path.join(self.root, row["path"])) as image:
            image.load()
            image = reflect_pad_to(image.convert("RGB"), self.size)
            crop = image.crop((left, top, left + self.size, top + self.size))
        if self.protocol == "bias_matched":
            crop = bias_match(crop, self.quality)
        if self.condition is not None:
            _, name, value, variant = self.condition
            crop = apply_named(crop, name, value, variant)
        crop = conform(crop, self.size)
        return (to_tensor(crop), int(row["label"] == 1), row_index,
                torch.tensor([float(row["width"]), float(row["height"])]))


def parse_run_spec(spec):
    """Parse NAME=PATH or infer NAME from a run directory/checkpoint path."""
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = os.path.basename(os.path.dirname(path)) if path.endswith(".pt") else os.path.basename(path.rstrip("/"))
    checkpoint = path if path.endswith(".pt") else os.path.join(path, "model.pt")
    if not name:
        raise ValueError(f"could not infer a method name from {spec!r}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    return name, checkpoint


def load_methods(specs, device):
    """Load all lightweight heads; return one shared frozen CLIP if required."""
    from .train import BranchEnsemble

    methods, clip_signature = {}, None
    for spec in specs:
        name, path = parse_run_spec(spec)
        if name in methods:
            raise ValueError(f"duplicate method name {name!r}; use NAME=PATH")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        cfg = OmegaConf.create(checkpoint["config"])
        branches = list(checkpoint["branches"])
        model = BranchEnsemble(cfg, branches, backbone=None).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        signature = None
        if "clip" in branches:
            signature = (str(cfg.model.clip.arch), str(cfg.model.clip.pretrained))
            if clip_signature is not None and signature != clip_signature:
                raise ValueError(f"CLIP mismatch: {signature} != {clip_signature}")
            clip_signature = signature
        methods[name] = {"path": path, "model": model, "branches": branches,
                         "gating": bool(cfg.model.gating.get("enabled", False))}
    backbone = (FrozenCLIP(*clip_signature, device=device) if clip_signature is not None else None)
    return methods, backbone


def aggregate_crops(scores, row_indices, n_rows, mode="trimmed_mean"):
    grouped = [[] for _ in range(int(n_rows))]
    for score, index in zip(scores, row_indices):
        grouped[int(index)].append(float(score))
    if any(not values for values in grouped):
        raise ValueError("at least one image received no crop scores")
    output = []
    for values in grouped:
        ordered = sorted(values)
        if mode == "trimmed_mean" and len(ordered) >= 3:
            ordered = ordered[1:-1]
        elif mode not in ("trimmed_mean", "mean"):
            raise ValueError(f"unknown crop aggregation {mode!r}")
        output.append(float(np.mean(ordered)))
    return np.asarray(output)


@torch.no_grad()
def score_condition(methods, backbone, loader, device, n_rows, aggregate="trimmed_mean", amp=True):
    crop_scores = {name: [] for name in methods}
    row_indices = []
    # The unclamped SRM branch can overflow FP16. Older CUDA devices without
    # BF16 support therefore evaluate in FP32 rather than using an unsafe
    # automatic fallback.
    use_amp = bool(amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16
    for pixels, _, indices, image_sizes in loader:
        pixels = pixels.to(device, non_blocking=True)
        image_sizes = image_sizes.to(device, non_blocking=True)
        tokens = None
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
            if backbone is not None:
                tokens = backbone(pixels)[1]
            for name, method in methods.items():
                model_tokens = tokens if "clip" in method["branches"] else None
                logits = method["model"](pixels, model_tokens, image_sizes=image_sizes)
                probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError(f"{name} produced a non-finite score")
                crop_scores[name].extend(float(value) for value in probabilities.cpu())
        row_indices.extend(int(index) for index in indices)
    return {name: aggregate_crops(values, row_indices, n_rows, aggregate)
            for name, values in crop_scores.items()}


def fixed_fpr_threshold(scores, labels, target_fpr):
    """Conservative threshold fitted on clean real scores and then held fixed."""
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    negatives = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if not len(negatives):
        raise ValueError("fixed-FPR threshold needs negative examples")
    allowed = int(math.floor(float(target_fpr) * len(negatives)))
    if allowed == 0:
        return float(np.nextafter(negatives[0], np.inf))
    if allowed == len(negatives):
        return float(np.nextafter(negatives[-1], -np.inf))
    boundary = negatives[allowed - 1]
    # Include the boundary when it does not introduce tied scores beyond the
    # requested false-positive budget; otherwise move just above the tie.
    if int(np.sum(negatives >= boundary)) <= allowed:
        return float(boundary)
    return float(np.nextafter(boundary, np.inf))


def accuracy_at(scores, labels, threshold):
    predictions = np.asarray(scores) >= float(threshold)
    return float(np.mean(predictions == np.asarray(labels)))


def markdown_table(title, methods, conditions, metric, mean_transformed=False):
    columns = ["Method", *conditions]
    if mean_transformed:
        columns.append("Mean transformed")
    lines = [f"### {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"]
    for name in methods:
        values = [metric[name][condition] for condition in conditions]
        cells = [name, *("--" if value is None else f"{value:.4f}" for value in values)]
        if mean_transformed:
            transformed = [value for value in values[1:] if value is not None]
            cells.append("--" if not transformed else f"{np.mean(transformed):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path, result):
    conditions = list(result["conditions"])
    methods = list(result["methods"])
    metadata = [
        "# Robustness Grid",
        "",
        f"- Protocol: `{result['protocol']}`",
        f"- Split: `{result['split']}` ({result['n_images']:,} images)",
        f"- Crops per image: {result['crops_per_image']}",
        f"- Fixed operating point: {100 * result['target_fpr']:.2f}% target FPR, fitted on clean real scores per method",
    ]
    auc_table = markdown_table("ROC AUC", methods, conditions, result["auc"], True)
    accuracy_table = markdown_table(
        "Accuracy at the clean fixed-FPR threshold", methods, conditions, result["accuracy"])
    thresholds = ["### Operating points", "", "| Method | Threshold | Clean FPR |", "| --- | ---: | ---: |"]
    for name in methods:
        op = result["operating_points"][name]
        thresholds.append(f"| {name} | {op['threshold']:.6f} | {op['clean_fpr']:.4f} |")
    text = ("\n".join(metadata) + "\n\n" + auc_table + "\n\n" + accuracy_table
            + "\n\n" + "\n".join(thresholds) + "\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run dirs/checkpoints, optionally NAME=PATH")
    parser.add_argument("--protocol", choices=("raw", "bias_matched"), required=True)
    parser.add_argument(
        "--split", default=None,
        help="override the protocol's default split (raw=val, bias_matched=val_matched)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--crops_per_image", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test image cap")
    parser.add_argument("--target_fpr", type=float, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--out", default="reports/robustness_table.md")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
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
        parser.error("evaluation requires both real and synthetic classes")

    device = resolve_device(args.device)
    methods, backbone = load_methods(args.runs, device)
    conditions = [("clean", None), *[(item[0], item) for item in eval_grid(cfg)]]
    n_crops = int(args.crops_per_image or cfg.eval.crops_per_image)
    batch_size = int(args.batch_size or cfg.eval.batch_size)
    num_workers = int(cfg.data.num_workers if args.num_workers is None else args.num_workers)
    target_fpr = float(args.target_fpr if args.target_fpr is not None else cfg.calibrate.target_fpr)
    if n_crops < 1 or batch_size < 1 or num_workers < 0:
        parser.error("crops_per_image and batch_size must be positive; num_workers cannot be negative")
    if not 0.0 <= target_fpr <= 1.0:
        parser.error("target_fpr must be between 0 and 1")
    common = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                  pin_memory=device.type == "cuda")
    scores = {name: {} for name in methods}

    print(f"protocol    {args.protocol}")
    print(f"split       {split} ({len(rows):,} images x {n_crops} crops)")
    print(f"methods     {', '.join(methods)}")
    print(f"conditions  {len(conditions)}")
    print(f"device      {device}\n")
    started = time.time()
    for index, (label, condition) in enumerate(conditions, 1):
        dataset = RobustnessDataset(rows, cfg, args.protocol, condition, n_crops)
        loader = DataLoader(dataset, **common)
        condition_scores = score_condition(
            methods, backbone, loader, device, len(rows), str(cfg.eval.aggregate), not args.no_amp)
        for name, values in condition_scores.items():
            scores[name][label] = values
        elapsed = time.time() - started
        print(f"[{index:>2}/{len(conditions)}] {label:<28} {elapsed / 60:.1f} min", flush=True)

    auc_grid, accuracy_grid, operating_points = {}, {}, {}
    for name in methods:
        threshold = fixed_fpr_threshold(scores[name]["clean"], labels, target_fpr)
        clean_predictions = scores[name]["clean"] >= threshold
        clean_fpr = float(clean_predictions[labels == 0].mean())
        operating_points[name] = {"threshold": threshold, "clean_fpr": clean_fpr}
        auc_grid[name] = {label: auc(scores[name][label], labels) for label, _ in conditions}
        accuracy_grid[name] = {
            label: accuracy_at(scores[name][label], labels, threshold) for label, _ in conditions}

    result = {
        "protocol": args.protocol, "split": split, "manifest": manifest,
        "n_images": len(rows), "crops_per_image": n_crops, "target_fpr": target_fpr,
        "elapsed_sec": time.time() - started, "conditions": [label for label, _ in conditions],
        "methods": {name: {key: value for key, value in method.items() if key != "model"}
                    for name, method in methods.items()},
        "auc": auc_grid, "accuracy": accuracy_grid, "operating_points": operating_points,
    }
    write_report(args.out, result)
    json_path = args.json_out or os.path.splitext(args.out)[0] + ".json"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\n-> {args.out}\n-> {json_path}")


if __name__ == "__main__":
    main()
