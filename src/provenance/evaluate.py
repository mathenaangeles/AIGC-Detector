"""Clean-vs-transformed AUC grid over the six competition transforms, with bpp controls.

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

The transform grid itself is not wired up here yet -- there is no trained model
to run it on. `evaluate_scores` is the metrics layer it will call, and takes
score arrays from any source.
"""

import argparse
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_manifest, row_bpp, select

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val",
                        help="manifest split to measure; val_matched reads its own manifest")
    parser.add_argument("--n_bins", type=int, default=N_BPP_BINS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    manifest = (str(cfg.data.val_matched_manifest) if args.split == "val_matched"
                else str(cfg.data.manifest))
    rows = select(load_manifest(manifest), split=args.split, labels=(0, 1))
    if not rows:
        parser.error(f"no rows for split {args.split!r} in {manifest}")

    bpp = np.array([r["bpp"] if "bpp" in r else row_bpp(r, cfg) for r in rows])
    labels = np.array([int(r["label"] == 1) for r in rows])
    table = compare_rows(labels, bpp, seed=int(cfg.seed), n_bins=int(args.n_bins))

    out_dir = args.out or os.path.join(str(cfg.paths.runs), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"evaluate_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump({"seed": int(cfg.seed), "split": args.split, "manifest": manifest,
                   "n_bins": int(args.n_bins), "rows": table}, f, indent=2)

    print(format_table(table))
    print(f"\nsplit {args.split}  seed {cfg.seed}  ->  {out_path}")


if __name__ == "__main__":
    main()
