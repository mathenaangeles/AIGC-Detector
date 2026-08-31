"""Content-blind confound probe: JPEG quantisation tables, resolution, bpp, chroma subsampling.

Reads nothing but the file header. No pixel is ever decoded on the raw path, so
any separation this probe achieves is separation a detector could achieve
without looking at the image -- an encoder fingerprint, not a generator
fingerprint. A high AUC here is a statement about the dataset, not the task.

This is the one place that fits on data/eval/. The fit is a measurement, not a
model: the coefficients are reported and discarded, nothing is written to
runs/ that train.py or predict.py reads back, and `fit_probe` refuses to hand
out a pickled estimator. Scores are out-of-fold, so the reported AUC is honest
about the confound rather than about the fit.

`bias_match` is the counterfactual: re-encode everything through one JPEG
encoder at one quality and one subsampling mode, crop (never resize) to one
size, and every feature this probe uses is constant by construction except the
quantisation tables, which are now the encoder's rather than the generator's.
The AUC it drops to is the part of the confound the pipeline actually removes.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, JpegImagePlugin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import _iter_files, _stable_unit, random_crop

QUANT_LEN = 64
SUBSAMPLING = {0: "444", 1: "422", 2: "420"}

FEATURE_NAMES = (
    [f"q_luma_{i}" for i in range(QUANT_LEN)]
    + [f"q_chroma_{i}" for i in range(QUANT_LEN)]
    + [f"subsampling_{name}" for name in SUBSAMPLING.values()]
    + ["subsampling_other", "width", "height", "aspect_ratio", "bytes_per_pixel",
       "is_jpeg", "has_chroma_table"]
)


def read_structure(path):
    """Container, dimensions, byte count, quantisation tables and subsampling mode.

    Image.open parses the JPEG headers lazily -- quantisation tables and the
    component sampling factors are populated without decoding scan data.
    """
    nbytes = os.path.getsize(path)
    with Image.open(path) as im:
        fmt = (im.format or "other").lower()
        width, height = im.size
        tables, sampling = {}, -1
        if fmt in ("jpeg", "mpo"):
            tables = {int(k): list(v) for k, v in getattr(im, "quantization", {}).items()}
            try:
                sampling = int(JpegImagePlugin.get_sampling(im))
            except Exception:
                sampling = -1
    return {"format": fmt, "width": width, "height": height, "bytes": nbytes,
            "quantization": tables, "subsampling": sampling}


def _table(tables, index):
    """A quantisation table as 64 floats. Absent (non-JPEG, or greyscale chroma) is zeros."""
    values = tables.get(index)
    if not values:
        return np.zeros(QUANT_LEN, dtype=np.float64)
    padded = np.zeros(QUANT_LEN, dtype=np.float64)
    values = np.asarray(values[:QUANT_LEN], dtype=np.float64)
    padded[:len(values)] = values
    return padded


def features(path):
    """One content-blind feature vector, ordered to match FEATURE_NAMES."""
    info = read_structure(path)
    width, height = max(info["width"], 1), max(info["height"], 1)
    is_jpeg = float(bool(info["quantization"]))
    onehot = np.zeros(len(SUBSAMPLING) + 1, dtype=np.float64)
    onehot[list(SUBSAMPLING).index(info["subsampling"]) if info["subsampling"] in SUBSAMPLING else -1] = 1.0

    return np.concatenate([
        _table(info["quantization"], 0),
        _table(info["quantization"], 1),
        onehot,
        np.array([width, height, width / height, info["bytes"] / (width * height),
                  is_jpeg, float(1 in info["quantization"])], dtype=np.float64),
    ])


def featurise(rows, root="."):
    """Feature matrix and label vector over manifest-shaped rows."""
    X = np.stack([features(os.path.join(root, row["path"])) for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    return X, y


def collect(cfg, sources=None, max_per_class=None, seed=None):
    """Enumerate the eval sources straight from disk.

    Deliberately not read from data/manifest.csv: the manifest is a training
    artifact and goes stale when eval data is re-extracted, and the probe's
    whole claim rests on the files as they sit on disk right now.
    """
    sources = list(sources or cfg.shortcut.sources)
    seed = int(cfg.seed if seed is None else seed)
    extensions = {e.lower() for e in cfg.data.extensions}
    root = str(cfg.paths.root)
    rows = []

    for source in sources:
        spec = cfg.sources[source]
        for class_spec in spec.classes.values():
            prefix = str(class_spec.glob).split("**")[0].strip("/")
            class_root = os.path.join(str(spec.root), prefix) if prefix else str(spec.root)
            if not os.path.isdir(class_root):
                continue
            for path in _iter_files(class_root, extensions):
                rows.append({"path": os.path.relpath(path, root), "source": source,
                             "label": int(class_spec.label)})

    if max_per_class:
        buckets = {}
        for row in rows:
            buckets.setdefault(row["label"], []).append(row)
        rows = []
        for label, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda r: _stable_unit(f"{seed}:shortcut:{r['path']}"))
            rows.extend(bucket[:int(max_per_class)])
        rows.sort(key=lambda r: r["path"])
    return rows


def fit_probe(X, y, seed=1337, folds=5, C=1.0, max_iter=2000, top_k=12):
    """Out-of-fold ROC AUC for logistic regression on content-blind features.

    Returns metrics only. The estimator is intentionally not returned: this
    probe measures a dataset, and nothing downstream may condition on a model
    that saw the held-out split.
    """
    y = np.asarray(y)
    folds = int(min(folds, np.bincount(y).min()))
    if folds < 2:
        raise ValueError("need at least 2 examples of each class to cross-validate")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(C), max_iter=int(max_iter), random_state=int(seed)),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    per_fold = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    coefs = model.fit(X, y).named_steps["logisticregression"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:int(top_k)]

    return {
        "auc": float(roc_auc_score(y, oof)),
        "auc_per_fold": [float(v) for v in per_fold],
        "auc_std": float(per_fold.std()),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_features": int(X.shape[1]),
        "folds": folds,
        "seed": int(seed),
        "top_features": [{"name": FEATURE_NAMES[i], "coef": float(coefs[i])} for i in order],
    }


def _matched_path(rel_path, out_dir, label):
    digest = hashlib.sha1(rel_path.encode()).hexdigest()[:16]
    return os.path.join(out_dir, str(label), f"{digest}.jpg")


def bias_match(rows, out_dir, root=".", quality=90, size=256, subsampling="4:2:0",
               seed=1337, reuse=True):
    """Re-encode every image through one encoder at one size, and return the new rows.

    Crop, never resize: resizing resamples, which rewrites the very high-frequency
    structure the real detector is meant to read. Images shorter than `size` on
    either axis are reflect-padded first; the count is returned so a set that is
    mostly padding can be recognised as such.

    Every non-quantisation feature is constant afterwards by construction --
    same dimensions, same subsampling, and bytes-per-pixel is all that is left
    of file size. Residual AUC is what survives a single shared encoder.
    """
    size = int(size)
    matched, padded = [], 0

    for row in rows:
        src = os.path.join(root, row["path"])
        dst = _matched_path(row["path"], out_dir, row["label"])
        stream = int(hashlib.sha1(row["path"].encode()).hexdigest()[:8], 16)

        with Image.open(src) as img:
            if min(img.size) < size:
                padded += 1
            if not (reuse and os.path.exists(dst)):
                img.load()
                crop = random_crop(img, size, np.random.default_rng([int(seed), stream]))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                crop.convert("RGB").save(dst, format="JPEG", quality=int(quality),
                                         subsampling=subsampling, optimize=False)
        matched.append({**row, "path": os.path.relpath(dst, root)})

    return matched, padded


def compare(cfg, rows=None, out_dir=None):
    """Fit the probe raw and bias-matched, and return both rows of the table."""
    root = str(cfg.paths.root)
    seed = int(cfg.seed)
    rows = rows if rows is not None else collect(cfg)
    out_dir = out_dir or os.path.join(str(cfg.paths.cache), "bias_matched")

    X, y = featurise(rows, root=root)
    raw = fit_probe(X, y, seed=seed, folds=int(cfg.shortcut.folds),
                    C=float(cfg.shortcut.C), max_iter=int(cfg.shortcut.max_iter))

    matched_rows, padded = bias_match(
        rows, out_dir, root=root, quality=int(cfg.shortcut.match_quality),
        size=int(cfg.shortcut.match_size), subsampling=str(cfg.shortcut.match_subsampling),
        seed=seed,
    )
    Xm, ym = featurise(matched_rows, root=root)
    matched = fit_probe(Xm, ym, seed=seed, folds=int(cfg.shortcut.folds),
                        C=float(cfg.shortcut.C), max_iter=int(cfg.shortcut.max_iter))

    return {
        "seed": seed,
        "sources": list(cfg.shortcut.sources),
        "match": {"quality": int(cfg.shortcut.match_quality), "size": int(cfg.shortcut.match_size),
                  "subsampling": str(cfg.shortcut.match_subsampling), "padded_images": padded},
        "rows": [{"set": "raw", **raw}, {"set": "bias-matched", **matched}],
    }


def format_table(result):
    """Two-row comparison, plain text."""
    header = f"{'set':<14}{'n':>7}{'ROC AUC':>10}{'+/- fold':>10}"
    lines = [header, "-" * len(header)]
    for row in result["rows"]:
        lines.append(f"{row['set']:<14}{row['n']:>7}{row['auc']:>10.4f}{row['auc_std']:>10.4f}")
    delta = result["rows"][0]["auc"] - result["rows"][-1]["auc"]
    lines.append("-" * len(header))
    lines.append(f"{'removed':<14}{'':>7}{delta:>10.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cap = args.max_per_class if args.max_per_class is not None else cfg.shortcut.max_per_class
    rows = collect(cfg, max_per_class=cap)
    result = compare(cfg, rows=rows)
    result["max_per_class"] = int(cap) if cap else None

    out_dir = args.out or os.path.join(str(cfg.paths.runs), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "shortcut.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(format_table(result))
    print(f"\nseed {result['seed']}  ->  {out_path}")


if __name__ == "__main__":
    main()
