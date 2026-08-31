import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from provenance import train
from provenance.data import MANIFEST_FIELDS


def rows_by_label(n=6, split="train"):
    return [
        {"path": f"{label}/{i:03d}.png", "label": label, "split": split}
        for label in (0, 1)
        for i in range(n)
    ]


def test_balanced_cap_preserves_both_classes_on_path_sorted_rows():
    rows = rows_by_label()
    selected = train.balanced_cap(rows, 4)
    assert len(selected) == 4
    assert [sum(r["label"] == label for r in selected) for label in (0, 1)] == [2, 2]


def test_require_binary_rejects_an_auc_split_with_one_class():
    with pytest.raises(ValueError, match="both real and synthetic"):
        train.require_binary(rows_by_label()[:4], "val split")


def test_consistency_kl_has_the_requested_direction_and_gradient_policy():
    clean = torch.tensor([[3.0, -1.0]], requires_grad=True)
    transformed = torch.tensor([[-1.0, 3.0]], requires_grad=True)
    loss = train.consistency_kl(clean, transformed)
    assert loss > 0
    loss.backward()
    assert clean.grad is not None and transformed.grad is not None

    clean = torch.tensor([[3.0, -1.0]], requires_grad=True)
    transformed = torch.tensor([[-1.0, 3.0]], requires_grad=True)
    train.consistency_kl(clean, transformed, detach=True).backward()
    assert clean.grad is None and transformed.grad is not None


def write_fixture(root):
    rows = []
    rng = np.random.default_rng(7)
    for split in ("train", "val"):
        for label in (0, 1):
            for i in range(4):
                rel = f"data/{split}/{label}/{i}.png"
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                base = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                if label:
                    base = (base // 4) * 4
                Image.fromarray(base, "RGB").save(path)
                rows.append({
                    "path": rel, "label": label, "source": "fixture", "split": split,
                    "format": "png", "width": 32, "height": 32,
                    "bytes": os.path.getsize(path),
                })

    manifest = root / "data/manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def tiny_config(tmp_path):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.paths.data = str(tmp_path / "data")
    cfg.paths.cache = str(tmp_path / "cache")
    cfg.paths.runs = str(tmp_path / "runs")
    cfg.data.manifest = str(tmp_path / "data/manifest.csv")
    cfg.data.val_matched_manifest = str(tmp_path / "data/missing-matched.csv")
    cfg.data.crop_size = 32
    cfg.data.crops_per_image = 1
    cfg.data.num_workers = 0
    cfg.data.bias_match = False
    cfg.model.srm.channels = [4, 8]
    cfg.model.srm.pool_after = 2
    cfg.model.srm.dropout = 0.0
    cfg.train.epochs = 1
    cfg.train.batch_size = 64
    cfg.train.amp = False
    cfg.eval.batch_size = 4
    cfg.eval.bpp_bins = 2
    return cfg


def test_small_limit_keeps_a_batch_and_both_classes(tmp_path):
    write_fixture(tmp_path)
    cfg = tiny_config(tmp_path)
    loaders = train.build_loaders(cfg, SimpleNamespace(limit=4))
    train_loader, _, _, train_rows, val_rows, _ = loaders
    assert len(train_loader) == 1
    assert {r["label"] for r in train_rows} == {0, 1}
    assert {r["label"] for r in val_rows} == {0, 1}
    assert next(iter(train_loader))[0].shape == (4, 3, 32, 32)


def test_srm_only_cli_writes_checkpoint_metrics_and_config(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    cfg = tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    out = tmp_path / "run"
    OmegaConf.save(cfg, config_path)
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", str(config_path), "--branches", "srm", "--epochs", "1",
        "--limit", "4", "--device", "cpu", "--no_amp", "--out", str(out),
    ])

    train.main()

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["branches"] == ["srm"]
    assert metrics["epochs_run"] == 1
    assert metrics["best"]["val"]["auc"] is not None
    assert metrics["used_token_cache"] is False
    assert (out / "model.pt").exists()
    assert (out / "config.yaml").exists()
