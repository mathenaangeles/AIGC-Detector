"""Manifests, native-resolution cropping, and paired clean/transformed loading.

The manifest records container format, dimensions and file size alongside the
label, because on this data those columns are themselves a near-perfect
classifier: SID_Set label 0 is 100% JPEG at mixed resolutions, label 1 is 100%
PNG at 1024x1024. shortcut.py fits on exactly these columns to quantify the
confound; bias_match destroys it before the model sees a pixel.

Crops are taken at native resolution and never resized. Images smaller than the
crop are reflect-padded (numpy, which permits pad width beyond the axis length;
torch's reflect pad does not).
"""

import bisect
import csv
import hashlib
import io
import os
import struct

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import sample_random

MANIFEST_FIELDS = ("path", "label", "source", "split", "format", "width", "height", "bytes")
VAL_MATCHED_FIELDS = MANIFEST_FIELDS + ("bpp", "match_gap")

_MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"RIFF", "webp"),
          (b"GIF8", "gif"), (b"BM", "bmp"), (b"II*\x00", "tiff"), (b"MM\x00*", "tiff"))


def probe_header(path):
    """Container format and dimensions from the file header alone -- no full decode."""
    with open(path, "rb") as f:
        head = f.read(32)
    fmt = next((name for magic, name in _MAGIC if head.startswith(magic)), "other")
    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception:
        width = height = 0
    return fmt, width, height, os.path.getsize(path)


def _stable_unit(key):
    """Uniform [0,1) from a stable hash, so adding files never reshuffles the split."""
    digest = hashlib.sha256(key.encode()).digest()
    return struct.unpack("<Q", digest[:8])[0] / 2 ** 64


def stratified_subset(rows, max_per_stratum, key, seed=0):
    """Cap rows per stratum, deterministically. Used to tame WildFake's generator skew."""
    if not max_per_stratum:
        return rows
    buckets = {}
    for row in rows:
        buckets.setdefault(key(row), []).append(row)
    kept = []
    for name, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: _stable_unit(f"{seed}:{name}:{r['path']}"))
        kept.extend(bucket[:max_per_stratum])
    return kept


def _iter_files(root, extensions):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in extensions:
                yield os.path.join(dirpath, name)


def generator_of(row, spec):
    """The generator a row came from, per the source's `generator_from` rule.

    `parent` reads the immediately enclosing directory, which is how WildFake
    lays its generators out. Sources that carry no generator information return
    None and must not be split by generator.
    """
    rule = str(spec.get("generator_from", "parent"))
    if rule == "parent":
        return os.path.basename(os.path.dirname(row["path"]))
    if rule == "grandparent":
        return os.path.basename(os.path.dirname(os.path.dirname(row["path"])))
    raise ValueError(f"unknown generator_from {rule!r}")


def assign_splits(source_rows, spec, cfg):
    """Mark each non-eval row train or val. Returns (rows, info).

    Two regimes. The default holds out a hash-keyed fraction of images, which
    measures generalisation to new images from generators the model has already
    seen. `split_by: generator` instead holds out whole generators, which
    measures the thing the task actually asks for -- generalisation to a
    generator that did not exist at training time. The two numbers are not
    comparable and the second is always the lower one.
    """
    declared = str(spec.get("split", "train"))
    if declared == "eval":
        return source_rows, {"strategy": "declared_eval"}

    if str(spec.get("split_by", "image")) != "generator":
        for row in source_rows:
            row["split"] = ("val" if _stable_unit(f"{cfg.seed}:{row['path']}")
                            < float(cfg.data.val_fraction) else "train")
        return source_rows, {"strategy": "image_hash"}

    # Only generated images have a generator. Real images are split by hash as
    # usual -- holding out their directory would empty the negative class from
    # train, and "camera" is not a generator to generalise away from.
    n_holdout = int(spec.get("holdout_generators", 2))
    fakes = [row for row in source_rows if row["label"] == 1]
    generators = sorted({generator_of(row, spec) for row in fakes})
    if len(generators) <= n_holdout:
        # Falling back to a random split here would silently produce a manifest
        # that claims a generator holdout it does not have.
        raise ValueError(
            f"split_by: generator needs more than {n_holdout} generators, found "
            f"{len(generators)}: {generators}. This source carries no usable "
            f"generator structure -- remove split_by or fix generator_from."
        )

    held = set(sorted(generators, key=lambda g: _stable_unit(f"{cfg.seed}:generator:{g}"))[:n_holdout])
    counts = {}
    for row in source_rows:
        if row["label"] == 1:
            generator = generator_of(row, spec)
            counts[generator] = counts.get(generator, 0) + 1
            row["split"] = "val" if generator in held else "train"
        else:
            row["split"] = ("val" if _stable_unit(f"{cfg.seed}:{row['path']}")
                            < float(cfg.data.val_fraction) else "train")

    return source_rows, {"strategy": "generator", "generators": generators,
                         "held_out": sorted(held), "counts": counts,
                         "n_real_split_by_hash": len(source_rows) - len(fakes)}


def assert_eval_isolated(rows, cfg):
    """Every row under data/eval/, and every row of an eval-declared source, is split=eval.

    Checked on the rows themselves rather than trusted from the config, because
    the failure this guards against -- a held-out image reaching a train loader
    -- is silent, survives every unit test that does not look for it, and
    invalidates every number the run produces.
    """
    # Manifest paths are relative to paths.root, so the guard has to be too.
    eval_prefix = os.path.normpath(
        os.path.relpath(str(cfg.paths.eval), str(cfg.paths.root))) + os.sep
    eval_sources = {name for name, spec in cfg.sources.items()
                    if str(spec.get("split", "train")) == "eval"}

    offenders = []
    for row in rows:
        under_eval = os.path.normpath(row["path"]).startswith(eval_prefix)
        if (under_eval or row["source"] in eval_sources) and row["split"] != "eval":
            offenders.append((row["path"], row["source"], row["split"]))
        if row["split"] == "eval" and not under_eval:
            offenders.append((row["path"], row["source"], "eval-but-not-under-data/eval"))

    if offenders:
        listed = "\n  ".join(f"{p} [{s}] -> {sp}" for p, s, sp in offenders[:10])
        raise AssertionError(
            f"{len(offenders)} row(s) break eval isolation:\n  {listed}"
            + ("\n  ..." if len(offenders) > 10 else "")
        )
    return len([r for r in rows if r["split"] == "eval"])


def build_manifest(cfg, out_path=None, write=True, with_report=False):
    """Scan every configured source into a manifest, assigning train/val/eval splits."""
    extensions = {e.lower() for e in cfg.data.extensions}
    min_side = int(cfg.data.min_side)
    rows, dropped, report = [], 0, {}

    for source, spec in cfg.sources.items():
        root = str(spec.root)
        if not os.path.isdir(root):
            continue
        declared_split = str(spec.get("split", "train"))
        source_rows = []
        for class_name, class_spec in spec.classes.items():
            prefix = str(class_spec.glob).split("**")[0].strip("/")
            class_root = os.path.join(root, prefix) if prefix else root
            if not os.path.isdir(class_root):
                continue
            for path in _iter_files(class_root, extensions):
                fmt, width, height, nbytes = probe_header(path)
                if declared_split != "eval" and min(width, height) < min_side:
                    dropped += 1
                    continue
                source_rows.append({
                    "path": os.path.relpath(path, cfg.paths.root),
                    "label": int(class_spec.label), "source": source,
                    "split": declared_split, "format": fmt,
                    "width": width, "height": height, "bytes": nbytes,
                })

        cap = spec.get("max_per_stratum")
        if cap:
            stratify_on = str(spec.get("stratify_on", "source"))
            def stratum(row, on=stratify_on):
                parent = os.path.basename(os.path.dirname(row["path"]))
                return f"{parent}:{row['label']}" if on == "parent" else f"{row['source']}:{row['label']}"
            source_rows = stratified_subset(source_rows, int(cap), stratum, cfg.seed)

        source_rows, info = assign_splits(source_rows, spec, cfg)
        report[source] = {**info, "n": len(source_rows)}
        rows.extend(source_rows)

    rows.sort(key=lambda r: r["path"])
    report["n_eval"] = assert_eval_isolated(rows, cfg)
    if write:
        out_path = out_path or str(cfg.data.manifest)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return (rows, dropped, report) if with_report else (rows, dropped)


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("label", "width", "height", "bytes"):
            row[key] = int(row[key])
        for key in ("bpp", "match_gap"):
            if key in row:
                row[key] = float(row[key])
    return rows


def select(rows, split=None, labels=None, sources=None):
    out = rows
    if split is not None:
        wanted = {split} if isinstance(split, str) else set(split)
        out = [r for r in out if r["split"] in wanted]
    if labels is not None:
        out = [r for r in out if r["label"] in set(labels)]
    if sources is not None:
        out = [r for r in out if r["source"] in set(sources)]
    return out


def reflect_pad_to(img, size):
    """Reflect-pad up to at least `size` on both axes. No-op if already large enough."""
    width, height = img.size
    if width >= size and height >= size:
        return img
    pad_x, pad_y = max(0, size - width), max(0, size - height)
    left, top = pad_x // 2, pad_y // 2
    arr = np.asarray(img.convert("RGB"))
    padded = np.pad(arr, ((top, pad_y - top), (left, pad_x - left), (0, 0)), mode="reflect")
    return Image.fromarray(padded, mode="RGB")


def random_crop(img, size, rng):
    img = reflect_pad_to(img, size)
    width, height = img.size
    left = int(rng.integers(0, width - size + 1))
    top = int(rng.integers(0, height - size + 1))
    return img.crop((left, top, left + size, top + size))


def bias_match(img, quality):
    """Re-encode to a common JPEG quality, collapsing the container-format cue."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def encoded_bpp(img, quality):
    """Bytes per pixel after one JPEG encode at `quality`. Nothing is written to disk.

    The whole-file bytes column in the manifest is not comparable across the
    SID_Set classes -- label 0 is JPEG and label 1 is PNG, so file size measures
    the container. Re-encoding through one encoder makes the number a property
    of the pixels: how compressible this content is, which is the part of the
    confound bias_match cannot remove.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.tell() / float(img.size[0] * img.size[1])


def row_bpp(row, cfg, root=None):
    """Encoded bpp of one deterministic native-resolution crop of a manifest row.

    Measured on a crop, not the whole image, because a crop is what the model
    sees. The crop box is keyed on the path so the value is stable run to run.
    """
    size = int(cfg.data.crop_size)
    stream = int(hashlib.sha1(row["path"].encode()).hexdigest()[:8], 16)
    path = os.path.join(root or str(cfg.paths.root), row["path"])
    with Image.open(path) as img:
        img.load()
        crop = random_crop(img, size, np.random.default_rng([int(cfg.seed), stream]))
    return encoded_bpp(crop, int(cfg.data.match_quality))


def bpp_matched_pairs(rows, bpp, caliper=None):
    """Greedy nearest-neighbour pairing across labels on bpp. Returns (pairs, gaps).

    Each label-1 row is matched, nearest first, to the not-yet-taken label-0 row
    closest to it in bpp. Greedy in ascending order of the best available gap,
    so the pairs that can be matched tightly are matched before the tails eat
    the pool. A pair whose gap exceeds `caliper` is dropped rather than kept as
    a bad match, which is what stops the subset from re-acquiring the very
    imbalance it exists to remove.
    """
    pos = sorted((r for r in rows if r["label"] == 1), key=lambda r: (bpp[r["path"]], r["path"]))
    neg = sorted((r for r in rows if r["label"] == 0), key=lambda r: (bpp[r["path"]], r["path"]))
    neg_bpp = [bpp[r["path"]] for r in neg]
    taken = [False] * len(neg)

    candidates = []
    for row in pos:
        i = bisect.bisect_left(neg_bpp, bpp[row["path"]])
        candidates.append((row, i))

    pairs, gaps = [], []
    order = sorted(candidates, key=lambda c: bpp[c[0]["path"]])
    for row, hint in order:
        target = bpp[row["path"]]
        best, best_gap = None, float("inf")
        # Walk outwards from the insertion point; the first untaken neighbour on
        # each side bounds everything further out on that side.
        for step, limit in ((-1, -1), (1, len(neg))):
            i = hint if step > 0 else hint - 1
            while i != limit:
                if not taken[i]:
                    gap = abs(neg_bpp[i] - target)
                    if gap < best_gap:
                        best, best_gap = i, gap
                    break
                i += step
        if best is None:
            break
        if caliper is not None and best_gap > caliper:
            continue
        taken[best] = True
        pairs.append((row, neg[best]))
        gaps.append(best_gap)
    return pairs, gaps


def build_val_matched(cfg, rows, out_path=None, write=True, caliper_sd=0.2):
    """A bpp-matched subset of the val split, as its own manifest.

    Written beside the main manifest rather than into it: `split` is one column,
    a val row cannot also be a val_matched row, and dropping matched rows out of
    val would quietly shrink the split every other metric is read from. Training
    splits are not touched.
    """
    val_rows = [r for r in rows if r["split"] == "val" and r["label"] in (0, 1)]
    if not val_rows:
        return [], {"n_val": 0, "n_matched": 0, "pairs": 0, "dropped": 0}

    bpp = {r["path"]: row_bpp(r, cfg) for r in val_rows}
    values = np.array([bpp[r["path"]] for r in val_rows])
    caliper = float(caliper_sd) * float(values.std()) if caliper_sd else None
    pairs, gaps = bpp_matched_pairs(val_rows, bpp, caliper=caliper)

    matched = []
    for (positive, negative), gap in zip(pairs, gaps):
        for row in (positive, negative):
            matched.append({**row, "split": "val_matched", "bpp": round(bpp[row["path"]], 6),
                            "match_gap": round(gap, 6)})
    matched.sort(key=lambda r: r["path"])

    pos_bpp = np.array([bpp[p["path"]] for p, _ in pairs])
    neg_bpp = np.array([bpp[n["path"]] for _, n in pairs])
    stats = {
        "n_val": len(val_rows),
        "n_matched": len(matched),
        "pairs": len(pairs),
        "dropped": len([r for r in val_rows if r["label"] == 1]) - len(pairs),
        "caliper": caliper,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "max_gap": float(np.max(gaps)) if gaps else None,
        "bpp_mean_before": {"real": float(values[[r["label"] == 0 for r in val_rows]].mean()),
                            "synthetic": float(values[[r["label"] == 1 for r in val_rows]].mean())},
        "bpp_mean_after": {"real": float(neg_bpp.mean()) if len(neg_bpp) else None,
                           "synthetic": float(pos_bpp.mean()) if len(pos_bpp) else None},
    }

    if write and matched:
        out_path = out_path or str(cfg.data.val_matched_manifest)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=VAL_MATCHED_FIELDS)
            writer.writeheader()
            writer.writerows(matched)
    return matched, stats


def to_tensor(img):
    """float32 CHW in [0,1]. Per-branch normalisation happens in the branch."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class CropDataset(Dataset):
    """Random native-resolution crops. `paired` yields (clean, transformed) of the SAME crop."""

    def __init__(self, rows, cfg, paired=False, train=True, root=None):
        if any(r["split"] == "eval" for r in rows):
            raise ValueError("eval rows reached CropDataset; data/eval/ is held out")
        self.rows = rows
        self.cfg = cfg
        self.paired = paired
        self.train = train
        self.root = root or str(cfg.paths.root)
        self.crop_size = int(cfg.data.crop_size)
        self.crops_per_image = int(cfg.data.crops_per_image) if train else 1
        self.do_match = bool(cfg.data.bias_match)
        self.match_quality = int(cfg.data.match_quality)
        self.seed = int(cfg.seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows) * self.crops_per_image

    def __getitem__(self, index):
        row = self.rows[index // self.crops_per_image]
        rng = np.random.default_rng([self.seed, self.epoch, index])

        with Image.open(os.path.join(self.root, row["path"])) as img:
            img.load()
            crop = random_crop(img, self.crop_size, rng)

        if self.do_match:
            crop = bias_match(crop, self.match_quality)

        label = torch.tensor(float(row["label"] == 1))
        if not self.paired:
            return to_tensor(crop), label
        transformed, _ = sample_random(crop, rng, self.cfg)
        return to_tensor(crop), to_tensor(transformed), label


def make_loaders(cfg, manifest_path=None, paired=True):
    """Train/val loaders over the binary task. Tampered rows are dropped unless enabled."""
    rows = load_manifest(manifest_path or str(cfg.data.manifest))
    labels = (0, 1, 2) if cfg.data.include_tampered else (0, 1)
    train_rows = select(rows, split="train", labels=labels)
    val_rows = select(rows, split="val", labels=labels)

    common = dict(batch_size=int(cfg.train.batch_size), num_workers=int(cfg.data.num_workers),
                  pin_memory=torch.cuda.is_available(), drop_last=False)
    train_ds = CropDataset(train_rows, cfg, paired=paired, train=True)
    val_ds = CropDataset(val_rows, cfg, paired=False, train=False)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(val_ds, shuffle=False, **common))
