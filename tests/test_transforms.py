import math

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from provenance.transforms import (
    NAMES,
    apply_named,
    eval_grid,
    gaussian_kernel_size,
    sample_random,
)

CFG = OmegaConf.load("configs/default.yaml")
GRID = eval_grid(CFG)


@pytest.fixture
def img():
    # Textured, not flat: contrast, saturation and blur are near-no-ops on a
    # smooth image and "changes the image" would fail spuriously.
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8), mode="RGB")


def arr(image):
    return np.asarray(image)


@pytest.mark.parametrize("label,name,value,variant", GRID, ids=[g[0] for g in GRID])
def test_deterministic(img, label, name, value, variant):
    a = apply_named(img, name, value, variant)
    b = apply_named(img, name, value, variant)
    assert np.array_equal(arr(a), arr(b))


@pytest.mark.parametrize("label,name,value,variant", GRID, ids=[g[0] for g in GRID])
def test_changes_image(img, label, name, value, variant):
    out = apply_named(img, name, value, variant)
    if name == "center_crop":
        assert out.size != img.size
    else:
        assert not np.array_equal(arr(out), arr(img))


@pytest.mark.parametrize("label,name,value,variant", GRID, ids=[g[0] for g in GRID])
def test_dimensions(img, label, name, value, variant):
    out = apply_named(img, name, value, variant)
    if name == "center_crop":
        assert out.size == (int(128 * value), int(128 * value))
    else:
        assert out.size == img.size


def test_grid_covers_every_transform():
    assert {g[1] for g in GRID} == set(NAMES)
    assert len(GRID) == 15
    assert [g[0] for g in GRID if g[1] == "color_jitter"] == [
        "color_jitter_s0.2_lo",
        "color_jitter_s0.2_hi",
    ]


def test_color_jitter_requires_variant(img):
    with pytest.raises(ValueError):
        apply_named(img, "color_jitter", 0.2)
    with pytest.raises(ValueError):
        apply_named(img, "jpeg", 90, variant="lo")


def test_jitter_extremes_differ(img):
    lo = apply_named(img, "color_jitter", 0.2, "lo")
    hi = apply_named(img, "color_jitter", 0.2, "hi")
    assert not np.array_equal(arr(lo), arr(hi))
    assert arr(lo).mean() < arr(img).mean() < arr(hi).mean()


@pytest.mark.parametrize("sigma,expected", [(0.5, 5), (1.0, 7), (2.0, 13)])
def test_kernel_size(sigma, expected):
    assert gaussian_kernel_size(sigma) == expected
    assert expected == 2 * math.ceil(3 * sigma) + 1


def test_sample_random_reproducible(img):
    a, ra = sample_random(img, np.random.default_rng(7), CFG)
    b, rb = sample_random(img, np.random.default_rng(7), CFG)
    assert np.array_equal(arr(a), arr(b))
    assert ra == rb


def test_sample_random_composes_one_or_two(img):
    for seed in range(20):
        out, records = sample_random(img, np.random.default_rng(seed), CFG)
        names = [r["name"] for r in records]
        assert 1 <= len(records) <= 2
        assert len(set(names)) == len(names)
        assert set(names) <= set(NAMES)
        if "center_crop" in names:
            assert names[-1] == "center_crop"
        else:
            assert out.size == img.size


@pytest.mark.parametrize("seed", range(20))
def test_sample_random_records_match_applied(img, seed):
    out, records = sample_random(img, np.random.default_rng(seed), CFG)
    names = {r["name"] for r in records}
    # gaussian_noise and color_jitter draw from the passed rng, so apply_named
    # cannot reproduce them from the record alone; the rest replay exactly.
    if names & {"gaussian_noise", "color_jitter"}:
        pytest.skip(f"not replayable from records alone: {sorted(names)}")
    replay = img
    for record in records:
        replay = apply_named(replay, record["name"], record["value"])
    assert np.array_equal(arr(replay), arr(out))
