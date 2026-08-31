import io
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from provenance.data import (
    CropDataset,
    assign_splits,
    bpp_matched_pairs,
    build_manifest,
    load_manifest,
    probe_header,
    random_crop,
    reflect_pad_to,
    row_bpp,
    select,
    stratified_subset,
)


def write_image(path, size, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(abs(hash(path)) % 2**32)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")
    img.save(path, format=fmt)


@pytest.fixture
def tree(tmp_path):
    """Mirrors the real confound: reals are JPEG at mixed sizes, fakes PNG at 1024."""
    root = tmp_path
    for i in range(6):
        write_image(str(root / "data/train/sid_set/real" / f"r{i}.jpg"), (400, 300), "JPEG")
        write_image(str(root / "data/train/sid_set/synthetic" / f"s{i}.png"), (512, 512), "PNG")
    for i in range(3):
        write_image(str(root / "data/train/sid_set/tampered" / f"t{i}.jpg"), (512, 512), "JPEG")
    write_image(str(root / "data/eval/coco_val2017" / "c0.jpg"), (640, 480), "JPEG")
    write_image(str(root / "data/eval/dalle_advanced" / "d0.jpg"), (1024, 1024), "JPEG")

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(root)
    cfg.data.min_side = 128
    cfg.data.manifest = str(root / "manifest.csv")
    del cfg.sources.wildfake
    return cfg


def test_manifest_columns_and_split(tree):
    rows, _ = build_manifest(tree)
    assert {r["source"] for r in rows} == {"sid_set", "coco_val2017", "dalle_advanced"}
    for row in rows:
        assert set(row) >= {"path", "label", "source", "split", "format", "width", "height", "bytes"}
    assert all(r["split"] == "eval" for r in rows if r["source"].startswith(("coco", "dalle")))
    assert all(r["split"] in {"train", "val"} for r in rows if r["source"] == "sid_set")


def test_manifest_records_the_confound(tree):
    rows, _ = build_manifest(tree)
    fmts = {(r["label"], r["format"]) for r in rows if r["source"] == "sid_set"}
    assert (0, "jpeg") in fmts and (1, "png") in fmts


def test_split_is_stable_when_files_are_added(tree):
    before = {r["path"]: r["split"] for r in build_manifest(tree, write=False)[0]}
    write_image(os.path.join(tree.paths.root, "data/train/sid_set/real/r99.jpg"), (400, 300), "JPEG")
    after = {r["path"]: r["split"] for r in build_manifest(tree, write=False)[0]}
    for path, split in before.items():
        assert after[path] == split


def test_min_side_drops_small_images_but_not_eval(tree):
    write_image(os.path.join(tree.paths.root, "data/train/sid_set/real/tiny.jpg"), (32, 32), "JPEG")
    rows, dropped = build_manifest(tree, write=False)
    assert dropped == 1
    assert not any(r["path"].endswith("tiny.jpg") for r in rows)


def test_manifest_roundtrip(tree):
    rows, _ = build_manifest(tree)
    loaded = load_manifest(str(tree.data.manifest))
    assert len(loaded) == len(rows)
    assert all(isinstance(r["label"], int) and isinstance(r["width"], int) for r in loaded)


def test_probe_header_does_not_decode(tmp_path):
    p = str(tmp_path / "a.png")
    write_image(p, (64, 48), "PNG")
    fmt, w, h, n = probe_header(p)
    assert (fmt, w, h) == ("png", 64, 48) and n > 0


def test_reflect_pad_beyond_axis_length():
    small = Image.fromarray(np.zeros((32, 32, 3), np.uint8), "RGB")
    out = reflect_pad_to(small, 224)
    assert out.size == (224, 224)


def test_random_crop_native_resolution_no_resize():
    # Each pixel encodes its own (y, x), so the crop's corner reveals its offset
    # and the whole window can be checked verbatim -- any rescaling breaks this.
    h, w = 300, 400
    ys, xs = np.mgrid[0:h, 0:w]
    src = np.stack([ys % 256, xs % 256, np.zeros_like(ys)], axis=-1).astype(np.uint8)
    img = Image.fromarray(src, "RGB")

    for seed in range(8):
        crop = random_crop(img, 224, np.random.default_rng(seed))
        assert crop.size == (224, 224)
        arr = np.asarray(crop)
        top, left = int(arr[0, 0, 0]), int(arr[0, 0, 1])
        assert np.array_equal(arr, src[top:top + 224, left:left + 224])


def test_stratified_subset_caps_and_is_deterministic():
    rows = [{"path": f"g{g}/{i}.jpg", "source": "w", "label": g % 2} for g in range(3) for i in range(50)]
    key = lambda r: r["path"].split("/")[0]
    a = stratified_subset(rows, 10, key, seed=1)
    b = stratified_subset(rows, 10, key, seed=1)
    assert [r["path"] for r in a] == [r["path"] for r in b]
    counts = {}
    for r in a:
        counts[key(r)] = counts.get(key(r), 0) + 1
    assert set(counts.values()) == {10}


def test_paired_returns_same_crop_two_views(tree):
    rows, _ = build_manifest(tree, write=False)
    train = select(rows, split=("train", "val"), labels=(0, 1))
    ds = CropDataset(train, tree, paired=True, train=True)
    clean, transformed, label = ds[0]
    assert clean.shape == (3, 224, 224)
    assert clean.dtype == torch.float32 and 0.0 <= float(clean.min()) and float(clean.max()) <= 1.0
    assert transformed.shape[0] == 3
    assert not torch.equal(clean, transformed)
    assert label.item() in (0.0, 1.0)


def test_dataset_is_deterministic_per_epoch(tree):
    rows, _ = build_manifest(tree, write=False)
    train = select(rows, split=("train", "val"), labels=(0, 1))
    a = CropDataset(train, tree, paired=True, train=True)
    b = CropDataset(train, tree, paired=True, train=True)
    assert torch.equal(a[3][0], b[3][0])
    b.set_epoch(1)
    assert not torch.equal(a[3][0], b[3][0])


def test_eval_rows_are_refused(tree):
    rows, _ = build_manifest(tree, write=False)
    eval_rows = select(rows, split="eval")
    assert eval_rows
    with pytest.raises(ValueError, match="held out"):
        CropDataset(eval_rows, tree)


def test_bias_match_destroys_the_format_cue(tree):
    rows, _ = build_manifest(tree, write=False)
    train = select(rows, split=("train", "val"), labels=(0, 1))
    ds = CropDataset(train, tree, paired=False, train=True)
    assert ds.do_match and ds.match_quality == 90
    tree.data.bias_match = False
    raw = CropDataset(train, tree, paired=False, train=True)
    assert not raw.do_match


def make_rows(bpps_by_label):
    rows, bpp = [], {}
    for label, values in bpps_by_label.items():
        for i, value in enumerate(values):
            path = f"{label}/{i}.jpg"
            rows.append({"path": path, "label": label})
            bpp[path] = value
    return rows, bpp


def test_matching_pairs_each_positive_with_its_nearest_negative():
    rows, bpp = make_rows({1: [0.10, 0.20, 0.30], 0: [0.31, 0.11, 0.21]})
    pairs, gaps = bpp_matched_pairs(rows, bpp, caliper=None)
    assert len(pairs) == 3
    for positive, negative in pairs:
        assert abs(bpp[positive["path"]] - bpp[negative["path"]]) == pytest.approx(0.01, abs=1e-9)
    assert max(gaps) == pytest.approx(0.01, abs=1e-9)


def test_matching_never_reuses_a_negative():
    rows, bpp = make_rows({1: [0.10, 0.10, 0.10], 0: [0.10, 0.50, 0.90]})
    pairs, _ = bpp_matched_pairs(rows, bpp, caliper=None)
    used = [negative["path"] for _, negative in pairs]
    assert len(used) == len(set(used)) == 3


def test_caliper_drops_a_pair_rather_than_matching_it_badly():
    rows, bpp = make_rows({1: [0.10, 0.90], 0: [0.11, 0.95]})
    pairs, gaps = bpp_matched_pairs(rows, bpp, caliper=0.02)
    assert len(pairs) == 1
    assert bpp[pairs[0][0]["path"]] == pytest.approx(0.10)
    assert gaps[0] == pytest.approx(0.01, abs=1e-9)


def test_matching_stops_when_negatives_run_out():
    rows, bpp = make_rows({1: [0.1, 0.2, 0.3, 0.4], 0: [0.15, 0.25]})
    pairs, _ = bpp_matched_pairs(rows, bpp, caliper=None)
    assert len(pairs) == 2


def test_matching_is_deterministic():
    rows, bpp = make_rows({1: [0.1, 0.2, 0.3], 0: [0.15, 0.25, 0.35]})
    first = [(p["path"], n["path"]) for p, n in bpp_matched_pairs(rows, bpp, caliper=None)[0]]
    second = [(p["path"], n["path"]) for p, n in bpp_matched_pairs(rows, bpp, caliper=None)[0]]
    assert first == second


def test_matched_subset_shrinks_the_class_bpp_gap(tree):
    rows, _ = build_manifest(tree, write=False)
    val = [r for r in rows if r["split"] == "val" and r["label"] in (0, 1)]
    if not val:
        pytest.skip("no val rows in fixture")
    bpp = {r["path"]: row_bpp(r, tree) for r in val}
    before = abs(np.mean([bpp[r["path"]] for r in val if r["label"] == 0])
                 - np.mean([bpp[r["path"]] for r in val if r["label"] == 1]))
    pairs, _ = bpp_matched_pairs(val, bpp, caliper=None)
    after = abs(np.mean([bpp[n["path"]] for _, n in pairs])
                - np.mean([bpp[p["path"]] for p, _ in pairs]))
    assert after <= before + 1e-12


def generator_tree(tmp_path, n_generators=5):
    """A source laid out the way WildFake is: one directory per generator."""
    for g in range(n_generators):
        for i in range(4):
            write_image(str(tmp_path / f"data/train/wf/Other_based/gen{g}" / f"f{i}.jpg"),
                        (400, 300), "JPEG")
    for i in range(6):
        write_image(str(tmp_path / "data/train/wf/Real/camera" / f"r{i}.jpg"), (400, 300), "JPEG")

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.data.min_side = 128
    cfg.data.manifest = str(tmp_path / "manifest.csv")
    for name in list(cfg.sources):
        del cfg.sources[name]
    cfg.sources.wf = OmegaConf.create({
        "root": str(tmp_path / "data/train/wf"), "split": "train",
        "split_by": "generator", "generator_from": "parent", "holdout_generators": 2,
        "classes": {"real": {"glob": "Real/**", "label": 0},
                    "fake": {"glob": "Other_based/**", "label": 1}},
    })
    return cfg


def test_generator_split_holds_out_whole_generators(tmp_path):
    cfg = generator_tree(tmp_path)
    rows, _, report = build_manifest(cfg, write=False, with_report=True)
    info = report["wf"]
    assert info["strategy"] == "generator"
    assert len(info["held_out"]) == 2

    by_generator = {}
    for row in rows:
        if row["label"] != 1:
            continue
        by_generator.setdefault(os.path.basename(os.path.dirname(row["path"])), set()).add(
            row["split"])
    for generator, splits in by_generator.items():
        assert len(splits) == 1, f"{generator} is split across {splits}"
        assert (splits == {"val"}) == (generator in set(info["held_out"]))


def test_no_held_out_generator_appears_in_train(tmp_path):
    cfg = generator_tree(tmp_path)
    rows, _, report = build_manifest(cfg, write=False, with_report=True)
    held = set(report["wf"]["held_out"])
    train = {os.path.basename(os.path.dirname(r["path"]))
             for r in rows if r["split"] == "train" and r["label"] == 1}
    assert not train & held


def test_generator_holdout_leaves_real_images_in_train(tmp_path):
    """Reals have no generator; holding out their directory would empty the class."""
    cfg = generator_tree(tmp_path)
    rows, _, report = build_manifest(cfg, write=False, with_report=True)
    real_train = [r for r in rows if r["label"] == 0 and r["split"] == "train"]
    assert real_train, "generator holdout removed every real image from train"
    assert "camera" not in report["wf"]["generators"]
    assert report["wf"]["n_real_split_by_hash"] == sum(1 for r in rows if r["label"] == 0)


def test_generator_holdout_is_stable_across_runs(tmp_path):
    cfg = generator_tree(tmp_path)
    first = build_manifest(cfg, write=False, with_report=True)[2]["wf"]["held_out"]
    second = build_manifest(cfg, write=False, with_report=True)[2]["wf"]["held_out"]
    assert first == second


def test_generator_holdout_changes_with_the_seed(tmp_path):
    cfg = generator_tree(tmp_path, n_generators=8)
    held = set()
    for seed in (1, 2, 3, 4, 5):
        cfg.seed = seed
        held.add(tuple(build_manifest(cfg, write=False, with_report=True)[2]["wf"]["held_out"]))
    assert len(held) > 1, "holdout ignores the seed"


def test_generator_split_refuses_a_source_with_too_few_generators(tmp_path):
    cfg = generator_tree(tmp_path, n_generators=2)
    with pytest.raises(ValueError, match="no usable generator structure"):
        build_manifest(cfg, write=False)


def test_sid_set_does_not_claim_a_generator_split():
    """SID_Set has no generator metadata; a holdout there would be invented."""
    cfg = OmegaConf.load("configs/default.yaml")
    assert str(cfg.sources.sid_set.get("split_by", "image")) == "image"
