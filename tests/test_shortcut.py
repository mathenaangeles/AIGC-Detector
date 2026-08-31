import os

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from provenance.shortcut import (
    FEATURE_NAMES,
    bias_match,
    collect,
    featurise,
    features,
    fit_probe,
    format_table,
    read_structure,
)


def write_image(path, size, fmt, seed=0, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path, format=fmt, **kwargs)


@pytest.fixture
def tree(tmp_path):
    """The real confound in miniature: reals are 4:4:4 JPEG at camera sizes,
    fakes are 4:2:0 JPEG at 1024 square, with a slice of PNG."""
    root = tmp_path
    for i in range(12):
        write_image(str(root / "data/eval/coco_val2017" / f"c{i}.jpg"), (640, 480), "JPEG",
                    seed=i, quality=75, subsampling="4:4:4")
        if i < 9:
            write_image(str(root / "data/eval/dalle_advanced" / f"d{i}.jpg"), (1024, 1024), "JPEG",
                        seed=100 + i, quality=95, subsampling="4:2:0")
        else:
            write_image(str(root / "data/eval/dalle_advanced" / f"d{i}.jpg"), (1024, 1024), "PNG",
                        seed=100 + i)

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(root)
    cfg.shortcut.max_per_class = 12
    cfg.shortcut.folds = 3
    return cfg


def test_feature_vector_matches_names(tree):
    path = str(tree.paths.root) + "/data/eval/coco_val2017/c0.jpg"
    vector = features(path)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_structure_read_without_decoding(tree, monkeypatch):
    """Content-blind is the whole claim: a decode anywhere on this path is a bug."""
    path = str(tree.paths.root) + "/data/eval/dalle_advanced/d0.jpg"
    info = read_structure(path)
    assert info["format"] == "jpeg"
    assert info["subsampling"] == 2
    assert set(info["quantization"]) == {0, 1}

    def forbidden(self, *args, **kwargs):
        raise AssertionError("read_structure decoded pixel data")

    monkeypatch.setattr(Image.Image, "load", forbidden)
    assert read_structure(path) == info
    assert np.isfinite(features(path)).all()


def test_png_has_no_quant_tables(tree):
    info = read_structure(str(tree.paths.root) + "/data/eval/dalle_advanced/d11.jpg")
    assert info["format"] == "png"
    assert info["quantization"] == {}
    vector = features(str(tree.paths.root) + "/data/eval/dalle_advanced/d11.jpg")
    assert vector[FEATURE_NAMES.index("is_jpeg")] == 0.0
    assert vector[:128].sum() == 0.0


def test_probe_separates_the_confound(tree):
    rows = collect(tree)
    assert {r["label"] for r in rows} == {0, 1}
    X, y = featurise(rows, root=str(tree.paths.root))
    result = fit_probe(X, y, seed=int(tree.seed), folds=int(tree.shortcut.folds))
    assert result["auc"] > 0.99
    assert result["n"] == len(rows)


def test_bias_match_makes_the_encoder_constant(tree):
    rows = collect(tree)
    out_dir = os.path.join(str(tree.paths.cache), "matched")
    matched, padded = bias_match(rows, out_dir, root=str(tree.paths.root),
                                 quality=90, size=256, subsampling="4:2:0", seed=int(tree.seed))
    assert padded == 0
    assert len(matched) == len(rows)

    structures = [read_structure(os.path.join(str(tree.paths.root), r["path"])) for r in matched]
    assert {s["format"] for s in structures} == {"jpeg"}
    assert {(s["width"], s["height"]) for s in structures} == {(256, 256)}
    assert {s["subsampling"] for s in structures} == {2}
    assert len({tuple(s["quantization"][0]) for s in structures}) == 1

    X, _ = featurise(matched, root=str(tree.paths.root))
    varying = {FEATURE_NAMES[i] for i in np.flatnonzero(X.std(axis=0) > 0)}
    assert varying == {"bytes_per_pixel"}


def test_bias_match_crops_rather_than_resizes(tree):
    """A 256px window of a 640px gradient spans ~2/5 of its range; a downscale
    of the whole frame would still span all of it."""
    rel = "data/eval/coco_val2017/gradient.jpg"
    path = os.path.join(str(tree.paths.root), rel)
    ramp = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    Image.fromarray(np.stack([ramp] * 3, axis=-1), "RGB").save(path, quality=100)

    out_dir = os.path.join(str(tree.paths.cache), "crop")
    matched, _ = bias_match([{"path": rel, "label": 0}], out_dir, root=str(tree.paths.root),
                            quality=100, size=256, seed=int(tree.seed))

    with Image.open(os.path.join(str(tree.paths.root), matched[0]["path"])) as im:
        got = np.asarray(im.convert("L"), dtype=np.int16)

    assert got.shape == (256, 256)
    assert np.ptp(got) < 0.6 * 255


def test_bias_match_is_deterministic_and_reuses(tree):
    rows = collect(tree)
    out_dir = os.path.join(str(tree.paths.cache), "det")
    first, _ = bias_match(rows, out_dir, root=str(tree.paths.root), seed=int(tree.seed))
    digests = [os.path.getsize(os.path.join(str(tree.paths.root), r["path"])) for r in first]
    second, _ = bias_match(rows, out_dir, root=str(tree.paths.root), seed=int(tree.seed), reuse=False)
    assert [os.path.getsize(os.path.join(str(tree.paths.root), r["path"])) for r in second] == digests
    assert [r["path"] for r in second] == [r["path"] for r in first]


def test_table_has_two_rows(tree):
    rows = collect(tree)
    X, y = featurise(rows, root=str(tree.paths.root))
    fit = fit_probe(X, y, seed=int(tree.seed), folds=3)
    text = format_table({"rows": [{"set": "raw", **fit}, {"set": "bias-matched", **fit}]})
    assert "raw" in text and "bias-matched" in text and "removed" in text
