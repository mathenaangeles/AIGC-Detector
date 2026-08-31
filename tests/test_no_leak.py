"""data/eval/ must never reach anything that fits.

The failure this guards against is silent. A held-out image that slips into a
train loader does not raise, does not change the loss curve visibly, and
produces an eval number that looks like success. Every assertion here is
therefore about paths actually yielded by a real loader, not about the config
that was supposed to configure it.
"""

import os

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from provenance.branches.clip_probe import CachedTokenDataset, enumerate_crops
from provenance.data import (
    CropDataset,
    assert_eval_isolated,
    build_manifest,
    build_val_matched,
    load_manifest,
    make_loaders,
    select,
)


def write_image(path, size, fmt, detail=1):
    """Blocky noise. `detail` sets the block size, and so how compressible it is.

    Flat noise would give every image of a class the same bpp and the two
    classes disjoint ranges, which the val_matched caliper would correctly
    reject wholesale -- leaving the matching assertions vacuous. Spanning the
    same detail levels in both classes makes their bpp distributions overlap,
    which is the case worth testing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(abs(hash(path)) % 2**32)
    block = max(1, int(detail))
    coarse = rng.integers(0, 256, (size[1] // block + 1, size[0] // block + 1, 3), dtype=np.uint8)
    arr = np.kron(coarse, np.ones((block, block, 1), dtype=np.uint8))[:size[1], :size[0]]
    Image.fromarray(arr, "RGB").save(path, format=fmt)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path
    for i in range(8):
        write_image(str(root / "data/train/sid_set/real" / f"r{i}.jpg"), (400, 300), "JPEG", i + 1)
        write_image(str(root / "data/train/sid_set/synthetic" / f"s{i}.png"), (512, 512), "PNG", i + 1)
    for i in range(4):
        write_image(str(root / "data/eval/coco_val2017" / f"c{i}.jpg"), (640, 480), "JPEG", i + 1)
        write_image(str(root / "data/eval/dalle_advanced" / f"d{i}.jpg"), (1024, 1024), "JPEG", i + 1)

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(root)
    cfg.paths.eval = str(root / "data/eval")
    cfg.sources.coco_val2017.root = str(root / "data/eval/coco_val2017")
    cfg.sources.dalle_advanced.root = str(root / "data/eval/dalle_advanced")
    cfg.sources.sid_set.root = str(root / "data/train/sid_set")
    cfg.data.min_side = 128
    cfg.data.crop_size = 128
    # Enough of both classes land in val for the matched subset to be non-empty,
    # so the val_matched assertions below actually run instead of skipping.
    cfg.data.val_fraction = 0.5
    cfg.data.num_workers = 0
    cfg.data.manifest = str(root / "manifest.csv")
    cfg.data.val_matched_manifest = str(root / "manifest_val_matched.csv")
    del cfg.sources.wildfake
    return cfg


def eval_paths(cfg):
    return {os.path.relpath(os.path.join(dirpath, name), str(cfg.paths.root))
            for dirpath, _, names in os.walk(str(cfg.paths.eval)) for name in names}


# -- the manifest ---------------------------------------------------------


def test_every_eval_source_row_is_split_eval(tree):
    rows, _ = build_manifest(tree)
    held = [r for r in rows if r["source"] in ("coco_val2017", "dalle_advanced")]
    assert held, "fixture produced no eval rows; the test would pass vacuously"
    for row in held:
        assert row["split"] == "eval", f"{row['path']} [{row['source']}] -> {row['split']}"


def test_every_row_under_data_eval_is_split_eval(tree):
    rows, _ = build_manifest(tree)
    on_disk = eval_paths(tree)
    assert on_disk
    seen = {r["path"] for r in rows if r["split"] == "eval"}
    assert seen == on_disk, f"missing {on_disk - seen}, extra {seen - on_disk}"


def test_no_train_or_val_row_lives_under_data_eval(tree):
    rows, _ = build_manifest(tree)
    prefix = os.path.normpath(str(tree.paths.eval)) + os.sep
    for row in rows:
        if row["split"] != "eval":
            assert not os.path.normpath(
                os.path.join(str(tree.paths.root), row["path"])).startswith(prefix), row["path"]


def test_assert_eval_isolated_fails_loudly(tree):
    rows, _ = build_manifest(tree)
    victim = next(r for r in rows if r["split"] == "eval")
    victim["split"] = "train"
    with pytest.raises(AssertionError, match="eval isolation"):
        assert_eval_isolated(rows, tree)


def test_assert_eval_isolated_catches_a_train_row_mislabelled_eval(tree):
    rows, _ = build_manifest(tree)
    victim = next(r for r in rows if r["split"] == "train")
    victim["split"] = "eval"
    with pytest.raises(AssertionError, match="eval isolation"):
        assert_eval_isolated(rows, tree)


# -- the loaders ----------------------------------------------------------


def test_train_loader_yields_no_path_under_data_eval(tree):
    build_manifest(tree)
    train_loader, val_loader = make_loaders(tree, paired=False)
    forbidden = eval_paths(tree)
    for loader in (train_loader, val_loader):
        paths = {row["path"] for row in loader.dataset.rows}
        assert paths, "loader is empty; the test would pass vacuously"
        assert not paths & forbidden, f"eval data in loader: {sorted(paths & forbidden)}"


def test_train_loader_actually_produces_batches(tree):
    """Guards the guard: an empty loader satisfies every leak assertion above."""
    build_manifest(tree)
    train_loader, _ = make_loaders(tree, paired=False)
    pixels, labels = next(iter(train_loader))
    assert pixels.shape[0] == labels.shape[0] > 0
    assert pixels.shape[1:] == (3, int(tree.data.crop_size), int(tree.data.crop_size))


def test_crop_dataset_refuses_eval_rows(tree):
    rows, _ = build_manifest(tree)
    with pytest.raises(ValueError, match="held out"):
        CropDataset(select(rows, split="eval"), tree)


def test_crop_dataset_refuses_a_single_smuggled_eval_row(tree):
    rows, _ = build_manifest(tree)
    smuggled = select(rows, split="train", labels=(0, 1)) + [select(rows, split="eval")[0]]
    with pytest.raises(ValueError, match="held out"):
        CropDataset(smuggled, tree)


def test_cached_token_dataset_refuses_eval_rows(tree):
    rows, _ = build_manifest(tree)
    with pytest.raises(ValueError, match="held out"):
        CachedTokenDataset(select(rows, split="eval"), tree, cache={})


def test_cache_keys_for_train_never_name_an_eval_file(tree):
    rows, _ = build_manifest(tree)
    forbidden = eval_paths(tree)
    keys = [key for _, _, key in enumerate_crops(select(rows, split=("train", "val")), tree)]
    assert keys
    for key in keys:
        assert key.rsplit("@", 1)[0] not in forbidden


# -- the bpp-matched val subset -------------------------------------------


def test_val_matched_holds_no_eval_row_and_no_train_row(tree):
    rows, _ = build_manifest(tree)
    matched, stats = build_val_matched(tree, rows)
    assert matched, "fixture produced no matched pairs; the assertions below would be vacuous"
    val = {r["path"] for r in select(rows, split="val")}
    for row in matched:
        assert row["split"] == "val_matched"
        assert row["path"] in val
    assert stats["n_matched"] == 2 * stats["pairs"]


def test_val_matched_is_class_balanced(tree):
    rows, _ = build_manifest(tree)
    matched, _ = build_val_matched(tree, rows)
    assert matched, "fixture produced no matched pairs; the assertion below would be vacuous"
    assert sum(r["label"] == 0 for r in matched) == sum(r["label"] == 1 for r in matched)


def test_building_val_matched_leaves_the_training_splits_untouched(tree):
    rows, _ = build_manifest(tree)
    before = {r["path"]: r["split"] for r in rows}
    build_val_matched(tree, rows)
    after = {r["path"]: r["split"] for r in load_manifest(str(tree.data.manifest))}
    assert before == after
