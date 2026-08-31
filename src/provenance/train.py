"""Training loop: cross-entropy plus transformation-consistency KL.

    loss = CE(f(I), y) + CE(f(T(I)), y) + lambda * KL(f(I) || f(T(I)))

One T per batch, drawn from transforms.sample_random and replayed across the
batch by apply_records, so every crop in a step sees the same degradation at the
same strength. Per-sample T would let the consistency term average over
transforms within a step, which is a weaker constraint: the point is that the
two passes differ by exactly one known degradation.

The KL is not detached. Detaching the clean pass would make it a teacher and the
loss a distillation, which pulls the transformed prediction toward the clean one
and leaves the clean one free; as written both are pulled together, which is
what "the prediction should not depend on the degradation" actually says. Set
`train.consistency_detach` to get the teacher form.

Branches train together and are selected with --branches. The CLIP tower is
frozen throughout -- it is held outside the module tree so it never enters the
optimiser, the checkpoint, or the parameter count.

Model selection defaults to AUC on the bpp-matched val split, not plain val. On
this data a model can reach val AUC in the mid-0.7s by reading compressibility
alone (see evaluate.py), so plain val AUC will happily select the run that
learned the shortcut best. val_matched has that cue largely removed, so it ranks
checkpoints by signal that survives it. Both are computed and written out every
epoch; --early_stop_metric switches which one decides.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .branches import srm as srm_branch
from .branches.clip_probe import (
    FeatureCache,
    FrozenCLIP,
    build_probe,
    cache_dir,
    cache_meta,
    crop_boxes,
    resolve_device,
)
from .data import bias_match, load_manifest, reflect_pad_to, row_bpp, select
from .evaluate import auc, stratified_auc
from .fuse import build as build_gate
from .transforms import apply_records, sample_random

BRANCHES = ("clip", "srm")
PARAM_BUDGET = 2_000_000_000


def conform(img, size):
    """Bring a transformed image back to `size` by padding and cropping. Never resizes.

    center_crop returns a smaller image and resize returns the original size, so
    after a composed T the batch is ragged. Reflect-padding and centre-cropping
    restores a common size without resampling, which resizing would do -- and
    resampling rewrites exactly the high-frequency structure the SRM branch reads.
    """
    img = reflect_pad_to(img.convert("RGB"), size)
    width, height = img.size
    left, top = (width - size) // 2, (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_image(tensor):
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def amp_dtype(device):
    """Use BF16 where CUDA supports it; SRM residuals can overflow FP16."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class CropDataset(Dataset):
    """Deterministic native-resolution crops, plus the cache key naming each one.

    The boxes come from crop_boxes, the same function the feature cache is keyed
    on, so the clean pass can read cached CLIP tokens instead of running the
    tower. That fixes the crops for the whole run rather than redrawing them per
    epoch; crops_per_image is the crop augmentation.
    """

    def __init__(self, rows, cfg, train=True):
        if any(r.get("split") == "eval" for r in rows):
            raise ValueError("eval rows reached CropDataset; data/eval/ is held out")
        self.rows = list(rows)
        self.root = str(cfg.paths.root)
        self.size = int(cfg.data.crop_size)
        self.seed = int(cfg.seed)
        self.do_match = bool(cfg.data.bias_match)
        self.quality = int(cfg.data.match_quality)
        self.n = int(cfg.data.crops_per_image) if train else 1

    def __len__(self):
        return len(self.rows) * self.n

    def __getitem__(self, index):
        row = self.rows[index // self.n]
        box = crop_boxes(int(row["width"]), int(row["height"]), self.size, self.n,
                         self.seed, row["path"])[index % self.n]
        with Image.open(os.path.join(self.root, row["path"])) as img:
            img.load()
            img = reflect_pad_to(img.convert("RGB"), self.size)
            crop = img.crop((box[0], box[1], box[0] + self.size, box[1] + self.size))
        if self.do_match:
            crop = bias_match(crop, self.quality)
        return (to_tensor(crop), int(row["label"] == 1),
                f"{row['path']}@{box[0]},{box[1]},{self.size}", index // self.n,
                torch.tensor([float(row["width"]), float(row["height"])]))


class BranchEnsemble(nn.Module):
    """Selected branches with global P7 or degradation-aware P8 fusion.

    The frozen tower is held in a list so nn.Module never sees it: out of
    parameters(), out of state_dict(), out of the optimiser, and out of the
    parameter count. A 300M-parameter backbone in every checkpoint is the
    accident this prevents.
    """

    def __init__(self, cfg, names, backbone=None):
        super().__init__()
        unknown = [n for n in names if n not in BRANCHES]
        if unknown:
            raise ValueError(f"unknown branch(es) {unknown}; expected from {BRANCHES}")
        if not names:
            raise ValueError("at least one branch is required")

        self.names = list(names)
        self._backbone = [backbone]
        self.probe = build_probe(cfg) if "clip" in self.names else None
        self.srm = srm_branch.build(cfg) if "srm" in self.names else None
        self.weights = nn.Parameter(torch.zeros(len(self.names)))
        self.gated = bool(cfg.model.gating.get("enabled", False))
        if self.gated and len(self.names) < 2:
            raise ValueError("gating requires at least two selected branches")
        self.gate = build_gate(cfg, len(self.names)) if self.gated else None

    @property
    def backbone(self):
        return self._backbone[0]

    def branch_logits(self, pixels, tokens=None):
        out = {}
        for name in self.names:
            if name == "clip":
                if tokens is None:
                    if self.backbone is None:
                        raise RuntimeError("clip branch needs a backbone or cached tokens")
                    tokens = self.backbone(pixels)[1]
                out["clip"] = self.probe(tokens)
            else:
                out["srm"] = self.srm(pixels)
        return out

    def fusion_weights(self, pixels, image_sizes=None):
        if self.gate is not None:
            return self.gate(pixels, image_sizes)
        return torch.softmax(self.weights, dim=0).expand(len(pixels), -1)

    def forward(self, pixels, tokens=None, image_sizes=None):
        logits = self.branch_logits(pixels, tokens)
        if len(logits) == 1:
            return next(iter(logits.values()))
        weights = self.fusion_weights(pixels, image_sizes)
        stacked = torch.stack([logits[name] for name in self.names], dim=1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)

    def freeze_branches(self):
        if self.gate is None:
            raise ValueError("--freeze_branches requires degradation-aware gating")
        for module in (self.probe, self.srm):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.weights.requires_grad_(False)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def consistency_kl(clean_logits, transformed_logits, detach=False):
    """KL(f(I) || f(T(I))). F.kl_div's `target` is the left argument of the KL."""
    target = F.softmax(clean_logits.detach() if detach else clean_logits, dim=-1)
    return F.kl_div(F.log_softmax(transformed_logits, dim=-1), target, reduction="batchmean")


def transform_batch(pixels, records, rng, size):
    """Apply one drawn T to every crop in the batch."""
    return torch.stack([to_tensor(conform(apply_records(to_image(p), records, rng), size))
                        for p in pixels])


def cached_tokens(cache, keys, device):
    if cache is None:
        return None
    return torch.from_numpy(np.stack([cache.tokens(k) for k in keys])).to(device)


@torch.no_grad()
def score_split(model, loader, device, cache=None, amp=False):
    """P(synthetic) per crop, averaged per image. Returns (scores, labels, row_index)."""
    model.eval()
    sums, counts, labels = {}, {}, {}
    for pixels, y, keys, row_index, image_sizes in loader:
        pixels = pixels.to(device, non_blocking=True)
        image_sizes = image_sizes.to(device, non_blocking=True)
        tokens = cached_tokens(cache, keys, device) if cache else None
        with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype(device)):
            probs = torch.softmax(
                model(pixels, tokens, image_sizes=image_sizes).float(), dim=-1)[:, 1]
        for p, label, index in zip(probs.cpu().numpy(), y.numpy(), row_index.numpy()):
            sums[int(index)] = sums.get(int(index), 0.0) + float(p)
            counts[int(index)] = counts.get(int(index), 0) + 1
            labels[int(index)] = int(label)
    order = sorted(sums)
    return (np.array([sums[i] / counts[i] for i in order]),
            np.array([labels[i] for i in order]), order)


def evaluate_split(model, loader, device, bpp, cache=None, amp=False, n_bins=5):
    scores, labels, order = score_split(model, loader, device, cache=cache, amp=amp)
    values = np.asarray(bpp)[order] if bpp is not None else None
    out = {"auc": auc(scores, labels), "n": int(len(labels))}
    if values is not None:
        strat = stratified_auc(scores, labels, values, n_bins)
        out["bpp_stratified_auc"] = strat["mean_auc"]
        out["bpp_bins"] = strat["bins"]
        if out["auc"] is not None and strat["mean_auc"] is not None:
            out["shortcut_gap"] = out["auc"] - strat["mean_auc"]
    return out


def balanced_cap(rows, limit):
    """Cap a binary split without turning a smoke run into a one-class split."""
    if not limit or len(rows) <= int(limit):
        return list(rows)
    buckets = {0: [], 1: []}
    for row in rows:
        buckets[int(row["label"] == 1)].append(row)
    if not all(buckets.values()):
        return list(rows)[:int(limit)]

    limit = max(2, int(limit))
    quotas = {0: limit // 2, 1: limit // 2}
    quotas[1] += limit % 2
    selected = buckets[0][:quotas[0]] + buckets[1][:quotas[1]]
    if len(selected) < limit:
        used = {0: min(quotas[0], len(buckets[0])), 1: min(quotas[1], len(buckets[1]))}
        for label in (0, 1):
            take = min(limit - len(selected), len(buckets[label]) - used[label])
            selected.extend(buckets[label][used[label]:used[label] + take])
    return sorted(selected, key=lambda row: row["path"])


def require_binary(rows, name):
    labels = {int(row["label"] == 1) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"{name} needs both real and synthetic rows; found labels {sorted(labels)}")


def format_metric(value):
    return "--" if value is None else f"{value:.4f}"


def build_loaders(cfg, args):
    rows = load_manifest(str(cfg.data.manifest))
    labels = (0, 1, 2) if cfg.data.include_tampered else (0, 1)
    train_rows = select(rows, split="train", labels=labels)
    val_rows = select(rows, split="val", labels=labels)

    matched_rows = []
    matched_path = str(cfg.data.val_matched_manifest)
    if os.path.exists(matched_path):
        matched_rows = select(load_manifest(matched_path), split="val_matched", labels=(0, 1))

    if args.limit:
        train_rows = balanced_cap(train_rows, args.limit)
        val_rows = balanced_cap(val_rows, max(4, args.limit // 4))
        matched_rows = balanced_cap(matched_rows, max(4, args.limit // 4))

    require_binary(train_rows, "train split")
    require_binary(val_rows, "val split")
    if matched_rows:
        require_binary(matched_rows, "val_matched split")

    common = dict(num_workers=int(cfg.data.num_workers), pin_memory=torch.cuda.is_available())
    train_dataset = CropDataset(train_rows, cfg, train=True)
    train = DataLoader(train_dataset, shuffle=True,
                       drop_last=len(train_dataset) >= int(cfg.train.batch_size),
                       batch_size=int(cfg.train.batch_size), **common)
    val = DataLoader(CropDataset(val_rows, cfg, train=False), shuffle=False,
                     batch_size=int(cfg.eval.batch_size), **common)
    matched = None
    if matched_rows:
        matched = DataLoader(CropDataset(matched_rows, cfg, train=False), shuffle=False,
                             batch_size=int(cfg.eval.batch_size), **common)
    return train, val, matched, train_rows, val_rows, matched_rows


def open_cache(cfg, names, train_rows, val_rows, matched_rows):
    """The clean-pass token cache, if it exists and covers every crop we will ask for.

    Only the clean pass can use it: T(I) is drawn fresh each step and was never
    cached. A partially populated cache is refused rather than half-used, because
    falling back per batch would silently mix cached and live tokens.
    """
    if "clip" not in names:
        return None
    root = cache_dir(cfg)
    if not os.path.exists(os.path.join(root, "index.json")):
        return None
    cache = FeatureCache(root).load()
    cache.check_compatible(cache_meta(cfg))

    size, seed = int(cfg.data.crop_size), int(cfg.seed)
    needed = []
    for rows, n in ((train_rows, int(cfg.data.crops_per_image)), (val_rows, 1), (matched_rows, 1)):
        for row in rows:
            for box in crop_boxes(int(row["width"]), int(row["height"]), size, n, seed, row["path"]):
                needed.append(f"{row['path']}@{box[0]},{box[1]},{size}")
    missing = [k for k in needed if k not in cache]
    if missing:
        print(f"cache      {len(missing)}/{len(needed)} crops missing; running the tower live")
        return None
    print(f"cache      {root} covers all {len(needed)} clean crops")
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--branches", default="clip,srm",
                        help="comma-separated subset of " + ",".join(BRANCHES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_consistency", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_metric", default="auc_val_matched",
                        choices=["auc_val_matched", "auc_val", "bpp_stratified_val"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="cap train rows, for smoke tests")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--gating", action="store_true",
                        help="enable degradation-aware per-image branch fusion")
    parser.add_argument("--init_checkpoint", default=None,
                        help="initialize branch weights from a compatible checkpoint")
    parser.add_argument("--freeze_branches", action="store_true",
                        help="train only the gate; requires --gating and --init_checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    for key, value in (("epochs", args.epochs), ("lr", args.lr), ("batch_size", args.batch_size),
                       ("lambda_consistency", args.lambda_consistency)):
        if value is not None:
            cfg.train[key] = value
    if args.gating:
        cfg.model.gating.enabled = True
    if args.freeze_branches and not args.init_checkpoint:
        parser.error("--freeze_branches requires --init_checkpoint")

    names = [n.strip() for n in args.branches.split(",") if n.strip()]
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(args.device)
    # Mixed precision is CUDA-only here. BF16 is preferred on supported GPUs:
    # the unclamped SRM residuals can overflow FP16 before gradient scaling can
    # protect the optimiser or BatchNorm running statistics.
    amp = bool(cfg.train.amp) and not args.no_amp and device.type == "cuda"
    mixed_dtype = amp_dtype(device)

    train_loader, val_loader, matched_loader, train_rows, val_rows, matched_rows = \
        build_loaders(cfg, args)
    cache = open_cache(cfg, names, train_rows, val_rows, matched_rows)

    backbone = None
    if "clip" in names:
        backbone = FrozenCLIP(str(cfg.model.clip.arch), str(cfg.model.clip.pretrained), device=device)
        assert not any(p.requires_grad for p in backbone.parameters())
    model = BranchEnsemble(cfg, names, backbone=backbone).to(device)

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        checkpoint_branches = checkpoint.get("branches")
        if checkpoint_branches != names:
            raise ValueError(
                f"checkpoint branches {checkpoint_branches} do not match requested {names}")
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("gate.")]
        if unexpected or missing:
            raise ValueError(f"incompatible checkpoint; missing={missing}, unexpected={unexpected}")
    if args.freeze_branches:
        model.freeze_branches()

    n_trainable = model.n_trainable()
    n_model = sum(p.numel() for p in model.parameters())
    n_total = n_model + (sum(p.numel() for p in backbone.parameters()) if backbone else 0)
    if n_total >= PARAM_BUDGET:
        raise ValueError(f"{n_total:,} parameters, budget is under {PARAM_BUDGET:,}")

    out_dir = args.out or os.path.join(str(cfg.paths.runs), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))

    print(f"branches   {','.join(names)}")
    print(f"fusion     {'degradation-aware gate' if model.gated else 'global softmax'}"
          + ("  [branches frozen]" if args.freeze_branches else ""))
    if args.init_checkpoint:
        print(f"initialise  {args.init_checkpoint}")
    print(f"params     {n_trainable:,} trainable, {n_total:,} total (budget {PARAM_BUDGET:,})")
    print(f"data       {len(train_rows)} train images x {cfg.data.crops_per_image} crops, "
          f"{len(val_rows)} val, {len(matched_rows)} val_matched")
    print(f"loss       CE(clean) + CE(T) + {float(cfg.train.lambda_consistency)} * KL"
          + ("  [detached]" if bool(cfg.train.get("consistency_detach", False)) else ""))
    print(f"device     {device}   amp {amp}"
          + (f" ({str(mixed_dtype).removeprefix('torch.')})" if amp else "")
          + f"   seed {seed}")
    print(f"select     {args.early_stop_metric}, patience {args.patience}\n")

    print("computing bpp for the held-out splits", flush=True)
    bpp_val = np.array([row_bpp(r, cfg) for r in val_rows])
    bpp_matched = (np.array([r["bpp"] if "bpp" in r else row_bpp(r, cfg) for r in matched_rows])
                   if matched_rows else None)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    epochs = int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs * len(train_loader)))
    # BF16 has FP32-like exponent range and does not need gradient scaling.
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp and mixed_dtype == torch.float16)
    lam = float(cfg.train.lambda_consistency)
    detach = bool(cfg.train.get("consistency_detach", False))
    size = int(cfg.data.crop_size)

    history, best, best_state, bad_epochs = [], None, None, 0
    started = time.time()

    for epoch in range(epochs):
        model.train()
        if args.freeze_branches:
            if model.probe is not None:
                model.probe.eval()
            if model.srm is not None:
                model.srm.eval()
        if backbone is not None:
            assert not backbone.visual.training, "the frozen tower left eval mode"
        totals, seen = {"loss": 0.0, "ce_clean": 0.0, "ce_trans": 0.0, "kl": 0.0}, 0

        for step, (pixels, y, keys, _, image_sizes) in enumerate(train_loader):
            rng = np.random.default_rng([seed, epoch, step])
            _, records = sample_random(to_image(pixels[0]), rng, cfg)
            transformed = transform_batch(pixels, records, rng, size).to(device, non_blocking=True)
            pixels, y = pixels.to(device, non_blocking=True), y.to(device, non_blocking=True)
            image_sizes = image_sizes.to(device, non_blocking=True)
            tokens = cached_tokens(cache, keys, device) if cache else None

            with torch.autocast(device_type=device.type, enabled=amp, dtype=mixed_dtype):
                clean_logits = model(pixels, tokens, image_sizes=image_sizes)
                trans_logits = model(transformed, image_sizes=image_sizes)
                ce_clean = F.cross_entropy(clean_logits, y)
                ce_trans = F.cross_entropy(trans_logits, y)
                kl = consistency_kl(clean_logits, trans_logits, detach=detach)
                loss = ce_clean + ce_trans + lam * kl

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            # Unclamped SRM residuals are heavy-tailed; this is the guard the
            # disabled TLU used to provide implicitly.
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            batch = y.shape[0]
            seen += batch
            for key, value in (("loss", loss), ("ce_clean", ce_clean),
                               ("ce_trans", ce_trans), ("kl", kl)):
                totals[key] += float(value.detach()) * batch
            if step % 50 == 0:
                print(f"\r  epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss {float(loss.detach()):.4f}  T={'+'.join(r['name'] for r in records)}",
                      end="", flush=True)

        metrics = {"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                   **{k: v / max(seen, 1) for k, v in totals.items()}}
        metrics["val"] = evaluate_split(model, val_loader, device, bpp_val, cache, amp,
                                        int(cfg.eval.bpp_bins))
        if matched_loader is not None:
            metrics["val_matched"] = evaluate_split(model, matched_loader, device, bpp_matched,
                                                    cache, amp, int(cfg.eval.bpp_bins))

        selected = {"auc_val": metrics["val"]["auc"],
                    "auc_val_matched": metrics.get("val_matched", {}).get("auc"),
                    "bpp_stratified_val": metrics["val"].get("bpp_stratified_auc")}
        score = selected[args.early_stop_metric]
        if score is None:
            score = metrics["val"]["auc"]
            metrics["selection_fallback"] = "auc_val"
        metrics["selection_score"] = score
        history.append(metrics)

        matched_str = (f"  matched {format_metric(metrics['val_matched']['auc'])}"
                       if "val_matched" in metrics else "")
        print(f"\r  epoch {epoch}  loss {metrics['loss']:.4f}  "
              f"ce {metrics['ce_clean']:.4f}/{metrics['ce_trans']:.4f}  kl {metrics['kl']:.4f}  "
              f"val {format_metric(metrics['val']['auc'])}{matched_str}  "
              f"strat {format_metric(metrics['val'].get('bpp_stratified_auc'))}")

        if best is None or score > best["selection_score"]:
            best, bad_epochs = metrics, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "branches": names, "epoch": epoch,
                        "gating": model.gated,
                        "config": OmegaConf.to_container(cfg, resolve=True)},
                       os.path.join(out_dir, "model.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"  early stop: {args.early_stop_metric} has not improved in "
                      f"{bad_epochs} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "seed": seed, "branches": names, "device": str(device), "amp": amp,
        "amp_dtype": str(mixed_dtype).removeprefix("torch.") if amp else None,
        "early_stop_metric": args.early_stop_metric, "patience": int(args.patience),
        "epochs_run": len(history), "epochs_configured": epochs,
        "n_trainable": n_trainable, "n_total_params": n_total,
        "gating": model.gated, "branches_frozen": bool(args.freeze_branches),
        "init_checkpoint": args.init_checkpoint,
        "lambda_consistency": lam, "consistency_detach": detach,
        "used_token_cache": cache is not None,
        "n_train_images": len(train_rows), "n_val_images": len(val_rows),
        "n_val_matched_images": len(matched_rows),
        "elapsed_sec": time.time() - started,
        "best": best, "history": history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nbest epoch {best['epoch']}  {args.early_stop_metric} "
          f"{format_metric(best['selection_score'])}")
    print(f"  val          auc {format_metric(best['val']['auc'])}  "
          f"bpp-stratified {format_metric(best['val'].get('bpp_stratified_auc'))}  "
          f"gap {format_metric(best['val'].get('shortcut_gap'))}")
    if "val_matched" in best:
        print(f"  val_matched  auc {format_metric(best['val_matched']['auc'])}  "
              f"bpp-stratified "
              f"{format_metric(best['val_matched'].get('bpp_stratified_auc'))}")
    print(f"\n-> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
