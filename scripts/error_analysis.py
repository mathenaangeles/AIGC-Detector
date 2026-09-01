#!/usr/bin/env python3
"""Rank false positives/negatives across the robustness grid and render them."""

import argparse
import json
import os
import textwrap
import time
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from provenance.branches.clip_probe import resolve_device
from provenance.data import bias_match, generator_of, load_manifest, reflect_pad_to, select
from provenance.evaluate import (
    RobustnessDataset,
    fixed_fpr_threshold,
    load_methods,
    score_condition,
)
from provenance.transforms import apply_named, eval_grid


def generator_name(row, cfg):
    """Return only generator identities supported by source metadata."""
    if int(row["label"]) != 1:
        return "not_applicable"
    source = str(row["source"])
    spec = cfg.sources.get(source)
    if spec is not None and spec.get("generator_from"):
        return generator_of(row, spec)
    # These eval sources name one known generator directly.
    if source in {"dalle_advanced"}:
        return source
    return "unknown"


def transform_type(condition):
    if condition == "clean":
        return "clean"
    for name in ("gaussian_blur", "gaussian_noise", "color_jitter",
                 "center_crop", "jpeg", "resize"):
        if condition.startswith(name + "_"):
            return name
    return condition


def table_for_groups(title, groups):
    lines = [f"### {title}", "", "| Group | Evaluated | FP | FN | FPR | FNR |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in groups.items():
        # Evaluated includes both classes, so derive denominators from the
        # error-entry metadata accumulated alongside each group.
        real = values.get("real", 0)
        synthetic = values.get("synthetic", 0)
        fpr = values["fp"] / real if real else 0.0
        fnr = values["fn"] / synthetic if synthetic else 0.0
        lines.append(
            f"| {name} | {values['evaluated']:,} | {values['fp']:,} | "
            f"{values['fn']:,} | {fpr:.4f} | {fnr:.4f} |")
    return "\n".join(lines)


def grouped_counts(entries, key):
    groups = defaultdict(lambda: {
        "evaluated": 0, "real": 0, "synthetic": 0, "fp": 0, "fn": 0})
    for entry in entries:
        bucket = groups[str(entry[key])]
        bucket["evaluated"] += 1
        bucket["real" if entry["label"] == 0 else "synthetic"] += 1
        if entry["error"]:
            bucket[entry["error"]] += 1
    return dict(sorted(groups.items()))


def representative_crop(row, cfg, protocol, condition):
    size = int(cfg.data.crop_size)
    path = os.path.join(str(cfg.paths.root), row["path"])
    with Image.open(path) as image:
        image.load()
        image = reflect_pad_to(image.convert("RGB"), size)
        width, height = image.size
        left, top = (width - size) // 2, (height - size) // 2
        crop = image.crop((left, top, left + size, top + size))
    if protocol == "bias_matched":
        crop = bias_match(crop, int(cfg.data.match_quality))
    if condition is not None:
        _, name, value, variant = condition
        crop = apply_named(crop, name, value, variant)
        crop = reflect_pad_to(crop, size)
        width, height = crop.size
        left, top = (width - size) // 2, (height - size) // 2
        crop = crop.crop((left, top, left + size, top + size))
    return crop


def load_font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(path, ranked, rows, cfg, protocol, conditions, columns=6):
    cell_width, image_height, caption_height = 260, 224, 100
    cell_height = image_height + caption_height
    n_rows = max(1, (len(ranked) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, n_rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font, bold = load_font(13), load_font(14, bold=True)
    condition_map = dict(conditions)
    for index, example in enumerate(ranked):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        crop = representative_crop(rows[example["row_index"]], cfg, protocol,
                                   condition_map[example["condition"]])
        crop.thumbnail((cell_width - 8, image_height - 8), Image.Resampling.LANCZOS)
        px = x + (cell_width - crop.width) // 2
        py = y + (image_height - crop.height) // 2
        sheet.paste(crop, (px, py))
        colour = (190, 30, 45) if example["error"] == "fp" else (30, 80, 190)
        draw.rectangle((x + 1, y + 1, x + cell_width - 2, y + cell_height - 2),
                       outline=colour, width=4)
        draw.text((x + 7, y + image_height + 4),
                  f"{example['error'].upper()}  p={example['score']:.4f}",
                  fill=colour, font=bold)
        caption = (f"{example['condition']}\n{example['source']} / "
                   f"{example['generator']}\n{example['path']}")
        wrapped_lines = []
        for line in caption.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=35) or [""])
        wrapped = "\n".join(wrapped_lines)
        draw.multiline_text((x + 7, y + image_height + 24), wrapped,
                            fill="black", font=font, spacing=2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path, format="PNG", optimize=True)


def write_report(path, result):
    transform_groups = result["groups"]["transform"]
    source_groups = result["groups"]["source"]
    generator_groups = result["groups"]["generator"]

    def worst(groups, metric):
        usable = []
        denominator = "real" if metric == "fp" else "synthetic"
        for name, values in groups.items():
            if values[denominator]:
                usable.append((values[metric] / values[denominator], name, values[metric]))
        return max(usable, default=(0.0, "none", 0))

    fp_rate, fp_group, fp_n = worst(transform_groups, "fp")
    fn_rate, fn_group, fn_n = worst(transform_groups, "fn")
    lines = [
        "# Error Analysis",
        "",
        f"- Protocol: `{result['protocol']}`",
        f"- Split: `{result['split']}` ({result['n_images']:,} images)",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Fixed clean threshold: `{result['threshold']:.6f}` at "
        f"`{100 * result['achieved_clean_fpr']:.2f}%` FPR",
        f"- Conditions: `{result['n_conditions']}`; crops per image: "
        f"`{result['crops_per_image']}`",
        f"- Contact sheet: [`{os.path.basename(result['contact_sheet'])}`]"
        f"({os.path.basename(result['contact_sheet'])})",
        "",
        "## Measured failure modes",
        "",
        f"- The highest false-positive rate occurs under `{fp_group}` "
        f"({fp_n} errors, {100 * fp_rate:.2f}% of real condition-cases).",
        f"- The highest false-negative rate occurs under `{fn_group}` "
        f"({fn_n} errors, {100 * fn_rate:.2f}% of synthetic condition-cases).",
    ]
    if set(generator_groups) <= {"unknown", "not_applicable"}:
        lines.append(
            "- Generator-level attribution is unavailable for this split. SID_Set does not "
            "carry generator metadata, so the report marks synthetic generator as `unknown` "
            "instead of inventing a grouping.")
    else:
        _, generator, count = worst(generator_groups, "fn")
        lines.append(
            f"- `{generator}` contributes the highest generator-specific false-negative "
            f"rate ({count} errors).")
    lines.extend([
        "- These are associations under controlled degradations, not causal claims about "
        "image semantics. Inspect the ranked contact sheet before assigning a visual cause.",
        "",
        table_for_groups("By transform type", transform_groups),
        "",
        table_for_groups("By source", source_groups),
        "",
        table_for_groups("By generator where known", generator_groups),
        "",
        "## Highest-confidence errors",
        "",
        "| Rank | Error | Condition | Score | Confidence | Source | Generator | Path |",
        "| ---: | --- | --- | ---: | ---: | --- | --- | --- |",
    ])
    for rank, example in enumerate(result["ranked"], 1):
        lines.append(
            f"| {rank} | {example['error'].upper()} | {example['condition']} | "
            f"{example['score']:.4f} | {example['confidence']:.4f} | "
            f"{example['source']} | {example['generator']} | `{example['path']}` |")
    lines.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/p8-gated-kl1/model.pt")
    parser.add_argument("--protocol", choices=("raw", "bias_matched"),
                        default="bias_matched")
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--crops_per_image", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_fpr", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="reports/error_analysis.md")
    parser.add_argument("--contact_sheet", default="reports/error_contact_sheet.png")
    parser.add_argument("--json_out", default="reports/error_analysis.json")
    args = parser.parse_args()
    if min(args.batch_size, args.crops_per_image, args.top_k) < 1 or args.num_workers < 0:
        parser.error("batch_size, crops_per_image, and top_k must be positive")

    cfg = OmegaConf.load(args.config)
    split = args.split or ("val_matched" if args.protocol == "bias_matched" else "val")
    manifest = (str(cfg.data.val_matched_manifest) if split == "val_matched"
                else str(cfg.data.manifest))
    rows = select(load_manifest(manifest), split=split, labels=(0, 1))
    if args.limit:
        from provenance.train import balanced_cap
        rows = balanced_cap(rows, int(args.limit))
    if not rows or len({row["label"] for row in rows}) < 2:
        parser.error("error analysis requires real and synthetic rows")
    labels = np.asarray([int(row["label"] == 1) for row in rows])
    device = resolve_device(args.device)
    methods, backbone = load_methods([f"gated={args.checkpoint}"], device)
    conditions = [("clean", None), *[(item[0], item) for item in eval_grid(cfg)]]
    loader_args = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda")

    scores, started = {}, time.time()
    for index, (name, condition) in enumerate(conditions, 1):
        dataset = RobustnessDataset(
            rows, cfg, args.protocol, condition, args.crops_per_image)
        values = score_condition(
            methods, backbone, DataLoader(dataset, **loader_args), device,
            len(rows), str(cfg.eval.aggregate), amp=True)["gated"]
        scores[name] = values
        print(f"[{index:>2}/{len(conditions)}] {name:<28} "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

    target_fpr = float(args.target_fpr if args.target_fpr is not None
                       else cfg.calibrate.target_fpr)
    threshold = fixed_fpr_threshold(scores["clean"], labels, target_fpr)
    clean_fpr = float(np.mean(scores["clean"][labels == 0] >= threshold))
    entries, false_positives, false_negatives = [], [], []
    for condition, values in scores.items():
        for row_index, (row, label, score) in enumerate(zip(rows, labels, values)):
            error = ""
            if label == 0 and score >= threshold:
                error = "fp"
            elif label == 1 and score < threshold:
                error = "fn"
            entry = {
                "row_index": row_index, "path": row["path"], "label": int(label),
                "source": row["source"], "generator": generator_name(row, cfg),
                "condition": condition, "transform": transform_type(condition),
                "score": float(score), "error": error,
                "confidence": float(abs(score - threshold)) if error else 0.0,
            }
            entries.append(entry)
            if error == "fp":
                false_positives.append(entry)
            elif error == "fn":
                false_negatives.append(entry)

    false_positives.sort(key=lambda item: (-item["confidence"], item["path"], item["condition"]))
    false_negatives.sort(key=lambda item: (-item["confidence"], item["path"], item["condition"]))
    ranked = false_positives[:args.top_k] + false_negatives[:args.top_k]
    contact_sheet(args.contact_sheet, ranked, rows, cfg, args.protocol, conditions)
    result = {
        "protocol": args.protocol, "split": split, "manifest": manifest,
        "checkpoint": args.checkpoint, "n_images": len(rows),
        "n_conditions": len(conditions), "crops_per_image": args.crops_per_image,
        "target_fpr": target_fpr, "threshold": threshold,
        "achieved_clean_fpr": clean_fpr, "elapsed_sec": time.time() - started,
        "n_false_positives": len(false_positives),
        "n_false_negatives": len(false_negatives),
        "contact_sheet": args.contact_sheet,
        "groups": {
            "transform": grouped_counts(entries, "transform"),
            "source": grouped_counts(entries, "source"),
            "generator": grouped_counts(entries, "generator"),
        },
        "ranked": ranked,
    }
    write_report(args.out, result)
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nFP {len(false_positives)}  FN {len(false_negatives)}")
    print(f"-> {args.contact_sheet}\n-> {args.out}\n-> {args.json_out}")


if __name__ == "__main__":
    main()
