import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from provenance.branches.clip_probe import (
    AttentionProbe,
    CachedTokenDataset,
    ClipProbe,
    FeatureCache,
    FrozenCLIP,
    build_probe,
    cache_key,
    cache_meta,
    crop_boxes,
    enumerate_crops,
    load_crop,
    to_pixels,
)

CFG = OmegaConf.load("configs/default.yaml")
WIDTH, N_TOKENS = 1024, 256


@pytest.fixture
def cfg(tmp_path):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.paths.cache = str(tmp_path / "cache")
    cfg.data.crops_per_image = 2
    return cfg


@pytest.fixture
def image(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "img.jpg"
    arr = rng.integers(0, 256, (400, 500, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path, quality=95)
    return str(path)


class _StubVisual(nn.Module):
    """Shaped like open_clip's ViT-L/14 visual tower, without the 1.7 GB."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, WIDTH, kernel_size=14, stride=14, bias=False)
        self.ln_post = nn.LayerNorm(WIDTH)
        self.output_tokens = False

    def forward(self, x):
        patches = self.conv1(x).flatten(2).transpose(1, 2)
        patches = self.ln_post(patches)
        return patches.mean(1), patches


@pytest.fixture
def stub_clip(monkeypatch):
    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = lambda *a, **k: (
        types.SimpleNamespace(visual=_StubVisual()), None, None)
    monkeypatch.setitem(sys.modules, "open_clip", module)
    return module


# -- crop keys ----------------------------------------------------------


def test_crop_boxes_are_deterministic_and_epoch_free():
    a = crop_boxes(500, 400, 224, 4, 1337, "data/x.jpg")
    b = crop_boxes(500, 400, 224, 4, 1337, "data/x.jpg")
    assert a == b
    assert a != crop_boxes(500, 400, 224, 4, 1337, "data/y.jpg")
    assert a != crop_boxes(500, 400, 224, 4, 999, "data/x.jpg")


def test_crop_boxes_stay_in_frame():
    for width, height in [(500, 400), (224, 224), (300, 1000)]:
        for left, top in crop_boxes(width, height, 224, 8, 1337, "k"):
            assert 0 <= left <= max(width, 224) - 224
            assert 0 <= top <= max(height, 224) - 224


def test_undersized_images_do_not_produce_negative_boxes():
    assert crop_boxes(64, 64, 224, 3, 1337, "small") == [(0, 0)] * 3


def test_key_carries_path_and_coords():
    assert cache_key("a/b.jpg", (12, 34), 224) == "a/b.jpg@12,34,224"


def test_load_crop_returns_the_named_window(image):
    crop = load_crop(image, (60, 40), 224, do_match=False)
    assert crop.size == (224, 224)
    with Image.open(image) as full:
        expect = np.asarray(full.convert("RGB"))[40:264, 60:284]
    assert np.array_equal(np.asarray(crop), expect)


def test_load_crop_bias_matches(image):
    raw = load_crop(image, (60, 40), 224, do_match=False)
    matched = load_crop(image, (60, 40), 224, do_match=True, quality=50)
    assert matched.size == raw.size
    assert not np.array_equal(np.asarray(raw), np.asarray(matched))


# -- probe --------------------------------------------------------------


def test_probe_is_under_two_million_params():
    probe = build_probe(CFG)
    n = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    assert n < 2_000_000
    assert ClipProbe(probe=probe).n_trainable() == n


def test_probe_rejects_an_over_budget_width():
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.clip.probe_dim = 1024
    with pytest.raises(ValueError, match="budget"):
        build_probe(cfg)


def test_probe_shapes_and_attention():
    probe = AttentionProbe(width=WIDTH, dim=512, heads=8).eval()
    tokens = torch.randn(3, N_TOKENS, WIDTH)
    assert probe(tokens).shape == (3, 2)
    logits, attn = probe(tokens, return_attention=True)
    assert attn.shape == (3, 8, N_TOKENS)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 8), atol=1e-5)
    # The explicit-softmax path and the fused SDPA path must agree.
    assert torch.allclose(logits, probe(tokens), atol=1e-4)


def test_probe_head_divisibility():
    with pytest.raises(ValueError, match="divide"):
        AttentionProbe(width=WIDTH, dim=500, heads=8)


def test_probe_is_permutation_equivariant_over_tokens():
    """Attention pooling has no positional term, so token order must not matter."""
    probe = AttentionProbe(width=WIDTH, dim=512, heads=8, dropout=0.0).eval()
    tokens = torch.randn(2, N_TOKENS, WIDTH)
    shuffled = tokens[:, torch.randperm(N_TOKENS)]
    assert torch.allclose(probe(tokens), probe(shuffled), atol=1e-4)


# -- frozen backbone ----------------------------------------------------


def test_backbone_is_frozen_and_stays_eval(stub_clip):
    backbone = FrozenCLIP("ViT-L-14-quickgelu", "openai")
    assert not any(p.requires_grad for p in backbone.parameters())
    assert not backbone.visual.training

    model = ClipProbe(backbone=backbone, probe=AttentionProbe(width=WIDTH, dim=512))
    model.train()
    assert model.probe.training, "probe must train"
    assert not backbone.visual.training, "backbone must not leave eval mode"
    assert model.n_trainable() == sum(p.numel() for p in model.probe.parameters())


def test_openai_weights_refuse_plain_gelu(stub_clip):
    """The bare arch name silently runs the wrong activation on these weights."""
    with pytest.raises(ValueError, match="quickgelu"):
        FrozenCLIP("ViT-L-14", "openai")
    FrozenCLIP("ViT-L-14", "laion2b_s32b_b82k")


def test_config_arch_is_quickgelu():
    assert "quickgelu" in str(CFG.model.clip.arch).lower()


def test_backbone_emits_no_grad_tokens(stub_clip):
    backbone = FrozenCLIP("ViT-L-14-quickgelu", "openai")
    cls, tokens = backbone(torch.rand(2, 3, 224, 224))
    assert tokens.shape == (2, N_TOKENS, WIDTH)
    assert cls.shape == (2, WIDTH)
    assert not tokens.requires_grad
    assert backbone.n_tokens(224) == N_TOKENS


def test_backbone_normalises_with_clip_statistics(stub_clip):
    backbone = FrozenCLIP("ViT-L-14-quickgelu", "openai")
    seen = {}
    backbone.visual.register_forward_pre_hook(lambda m, args: seen.update(x=args[0]))
    backbone(torch.zeros(1, 3, 224, 224))
    expect = (0.0 - backbone.mean) / backbone.std
    assert torch.allclose(seen["x"], expect.expand_as(seen["x"]))


def test_gradient_reaches_the_probe_only(stub_clip):
    model = ClipProbe(backbone=FrozenCLIP("ViT-L-14-quickgelu", "openai"),
                      probe=AttentionProbe(width=WIDTH, dim=512))
    logits = model(torch.rand(2, 3, 224, 224))
    logits.sum().backward()
    assert all(p.grad is not None for p in model.probe.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


# -- cache --------------------------------------------------------------


def test_cache_roundtrip(tmp_path):
    root = str(tmp_path / "c")
    cache = FeatureCache(root, shard_size=2).open_for_write({"arch": "ViT-L-14"})
    tokens = {f"k{i}": np.random.rand(N_TOKENS, WIDTH).astype(np.float32) for i in range(5)}
    for key, value in tokens.items():
        cache.add(key, value, value[0])
    cache.flush()

    reopened = FeatureCache(root)
    assert len(reopened) == 5
    for key, value in tokens.items():
        assert key in reopened
        got = reopened.tokens(key)
        assert got.shape == (N_TOKENS, WIDTH)
        assert got.dtype == np.float32
        assert np.allclose(got, value, atol=1e-3)
        assert np.allclose(reopened.cls(key), value[0], atol=1e-3)


def test_cache_shards_rather_than_one_file_per_crop(tmp_path):
    root = str(tmp_path / "c")
    cache = FeatureCache(root, shard_size=2).open_for_write({})
    for i in range(5):
        cache.add(f"k{i}", np.zeros((4, 8), np.float32), np.zeros(8, np.float32))
    cache.flush()
    shards = [f for f in os.listdir(root) if f.startswith("tokens_")]
    assert len(shards) == 3 and len(cache) == 5


def test_cache_is_stored_as_float16(tmp_path):
    root = str(tmp_path / "c")
    cache = FeatureCache(root, shard_size=4).open_for_write({})
    cache.add("k", np.ones((4, 8), np.float32), np.ones(8, np.float32))
    cache.flush()
    assert np.load(os.path.join(root, "tokens_00000.npy")).dtype == np.float16


def test_cache_refuses_incompatible_settings(tmp_path):
    root = str(tmp_path / "c")
    cache = FeatureCache(root, shard_size=1).open_for_write({"arch": "ViT-L-14", "bias_match": True})
    cache.add("k", np.zeros((4, 8), np.float32), np.zeros(8, np.float32))
    cache.flush()

    reopened = FeatureCache(root)
    reopened.check_compatible({"arch": "ViT-L-14", "bias_match": True})
    with pytest.raises(ValueError, match="bias_match"):
        reopened.check_compatible({"arch": "ViT-L-14", "bias_match": False})


def test_cache_resumes(tmp_path):
    root = str(tmp_path / "c")
    first = FeatureCache(root, shard_size=2).open_for_write({})
    for i in range(3):
        first.add(f"k{i}", np.zeros((4, 8), np.float32), np.zeros(8, np.float32))
    first.flush()

    second = FeatureCache(root, shard_size=2).open_for_write({})
    assert "k0" in second and "k9" not in second
    second.add("k9", np.ones((4, 8), np.float32), np.ones(8, np.float32))
    second.flush()
    assert len(FeatureCache(root)) == 4
    assert np.allclose(FeatureCache(root).tokens("k9"), 1.0)


# -- cached dataset -----------------------------------------------------


def _populate(cfg, rows, cache_path):
    cache = FeatureCache(cache_path, shard_size=8).open_for_write(cache_meta(cfg))
    for _, _, key in enumerate_crops(rows, cfg):
        cache.add(key, np.random.rand(4, 8).astype(np.float32), np.zeros(8, np.float32))
    cache.flush()
    return cache


def test_cached_dataset_reads_tokens_and_labels(cfg, tmp_path):
    rows = [{"path": "a.jpg", "label": 0, "width": 500, "height": 400, "split": "train"},
            {"path": "b.jpg", "label": 1, "width": 500, "height": 400, "split": "train"}]
    cache = _populate(cfg, rows, str(tmp_path / "tok"))
    ds = CachedTokenDataset(rows, cfg, cache=FeatureCache(str(tmp_path / "tok")))
    assert len(ds) == 2 * int(cfg.data.crops_per_image)
    tokens, label = ds[0]
    assert tokens.shape == (4, 8) and label.dtype == torch.long
    assert sorted(int(ds[i][1]) for i in range(len(ds))) == [0, 0, 1, 1]
    assert len(cache) == len(ds)


def test_cached_dataset_refuses_eval_rows(cfg):
    rows = [{"path": "e.jpg", "label": 0, "width": 500, "height": 400, "split": "eval"}]
    with pytest.raises(ValueError, match="held out"):
        CachedTokenDataset(rows, cfg)


def test_cached_dataset_refuses_missing_crops(cfg, tmp_path):
    rows = [{"path": "a.jpg", "label": 0, "width": 500, "height": 400, "split": "train"}]
    _populate(cfg, rows, str(tmp_path / "tok"))
    rows.append({"path": "c.jpg", "label": 1, "width": 500, "height": 400, "split": "train"})
    with pytest.raises(ValueError, match="not cached"):
        CachedTokenDataset(rows, cfg, cache=FeatureCache(str(tmp_path / "tok")))


def test_pixels_are_unit_scaled(image):
    pixels = to_pixels(load_crop(image, (0, 0), 224))
    assert pixels.shape == (3, 224, 224)
    assert pixels.dtype == torch.float32
    assert 0.0 <= float(pixels.min()) and float(pixels.max()) <= 1.0
