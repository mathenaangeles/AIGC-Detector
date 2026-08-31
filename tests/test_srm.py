import collections

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from provenance.branches.srm import (
    AXES_4,
    KERNEL_CLASSES,
    NEIGHBOURS_8,
    SQUARE_3,
    SQUARE_5,
    SRMFilterBank,
    SRMNet,
    build,
    srm_kernels,
)

CFG = OmegaConf.load("configs/default.yaml")
BANK = srm_kernels()


# -- the filter bank ----------------------------------------------------


def test_bank_is_the_standard_thirty():
    assert BANK.shape == (30, 5, 5)
    assert collections.Counter(KERNEL_CLASSES) == {
        "D1": 8, "D2": 4, "D3": 8, "S3": 1, "E3": 4, "S5": 1, "E5": 4
    }
    assert len(KERNEL_CLASSES) == 30


def test_every_kernel_is_high_pass():
    """A zero-sum kernel is blind to the local mean. This is the whole point."""
    assert np.allclose(BANK.reshape(30, -1).sum(axis=1), 0.0, atol=1e-6)


def test_kernels_are_distinct():
    assert len({tuple(k.ravel()) for k in BANK}) == 30


def test_orientation_sets_are_complete():
    assert len(set(NEIGHBOURS_8)) == 8
    assert len(set(AXES_4)) == 4
    assert all(-1 <= d <= 1 for pair in NEIGHBOURS_8 for d in pair)


def test_square_kernels_match_the_published_tables():
    assert np.array_equal(SQUARE_3, [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    assert SQUARE_5[2, 2] == -12 and SQUARE_5[0, 0] == -1
    assert np.array_equal(SQUARE_5, SQUARE_5.T), "S5 is symmetric"
    assert np.array_equal(SQUARE_5, np.rot90(SQUARE_5)), "S5 is isotropic"
    # S3 and S5 land at indices 20 and 25 of the bank, divided by c = 4 and 12.
    assert np.allclose(BANK[20][1:4, 1:4], SQUARE_3 / 4.0)
    assert np.allclose(BANK[25], SQUARE_5 / 12.0)


def test_first_order_kernels_are_plain_differences():
    for k in BANK[:8]:
        assert k[2, 2] == -1.0
        assert np.count_nonzero(k) == 2
        assert k.sum() == 0.0


def test_normalisation_keeps_kernels_comparable():
    """Dividing by c is what lets one TLU threshold serve all 30 classes."""
    peaks = np.abs(BANK).max(axis=(1, 2))
    assert peaks.min() >= 0.5 and peaks.max() <= 1.5


def test_edge_kernels_are_four_rotations_of_a_half_plane():
    e3 = BANK[21:25]
    assert all(np.count_nonzero(k) == 6 for k in e3), "one row of S3 zeroed"
    e5 = BANK[26:30]
    assert all(np.count_nonzero(k) == 15 for k in e5), "two rows of S5 zeroed"
    for group in (e3, e5):
        for k in group[1:]:
            assert not np.allclose(k, group[0]), "rotations must differ"


# -- the bank as a layer ------------------------------------------------


def test_bank_is_not_trainable():
    bank = SRMFilterBank()
    assert list(bank.parameters()) == []
    assert "weight" in dict(bank.named_buffers())


def test_bank_stays_out_of_train_mode():
    bank = SRMFilterBank()
    bank.train()
    assert not bank.training


def test_bank_expands_channels_depthwise():
    bank = SRMFilterBank(in_channels=3)
    out = bank(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 90, 64, 64)
    assert bank.out_channels == 90


def test_bank_is_blind_to_flat_content():
    """Constant input is pure content and no residual: output must be zero."""
    bank = SRMFilterBank(threshold=0.0)
    for level in (0.0, 0.5, 1.0):
        out = bank(torch.full((1, 3, 64, 64), level))
        assert torch.allclose(out[..., 2:-2, 2:-2], torch.zeros(1), atol=1e-3)


def test_bank_ignores_a_dc_shift():
    """Adding a constant changes content, not noise. The response must not move."""
    bank = SRMFilterBank(threshold=0.0)
    x = torch.rand(1, 3, 64, 64) * 0.5
    a = bank(x)[..., 2:-2, 2:-2]
    b = bank(x + 0.25)[..., 2:-2, 2:-2]
    assert torch.allclose(a, b, atol=1e-2)


def test_bank_responds_to_noise_far_more_than_to_a_gradient():
    bank = SRMFilterBank(threshold=0.0)
    ramp = torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(1, 3, 64, 64).contiguous()
    noise = torch.rand(1, 3, 64, 64)
    smooth = bank(ramp)[..., 2:-2, 2:-2].abs().mean()
    rough = bank(noise)[..., 2:-2, 2:-2].abs().mean()
    assert rough > 20 * smooth


def test_tlu_clamps_residuals():
    bank = SRMFilterBank(threshold=3.0)
    out = bank(torch.rand(1, 3, 64, 64))
    assert out.abs().max() <= 3.0 + 1e-5

    unclamped = SRMFilterBank(threshold=0.0)(torch.rand(1, 3, 64, 64))
    assert unclamped.abs().max() > 3.0, "the fixture should exceed the threshold"


def test_bank_uses_the_zero_to_255_scale():
    """SRM's constants were derived for 0-255; the rescale is folded into the weight."""
    bank = SRMFilterBank(threshold=0.0)
    plain = torch.from_numpy(srm_kernels())
    assert torch.allclose(bank.weight[0, 0], plain[0] * 255.0)


# -- the network --------------------------------------------------------


def test_network_is_about_five_million_params():
    model = build(CFG)
    n = model.n_trainable()
    assert 4_000_000 <= n <= 6_000_000, f"{n:,} outside the ~5M budget"
    assert n == sum(p.numel() for p in model.features.parameters()) + \
        sum(p.numel() for p in model.classifier.parameters())


def test_fixed_bank_contributes_no_trainable_params():
    model = build(CFG)
    assert not any(p.requires_grad for p in model.srm.parameters())
    assert list(model.srm.parameters()) == []


def test_network_accepts_native_resolution_crops():
    model = build(CFG).eval()
    with torch.no_grad():
        for shape in [(224, 224), (256, 256), (96, 96), (32, 32), (100, 180), (301, 97)]:
            assert model(torch.rand(1, 3, *shape)).shape == (1, 2)


def test_network_rejects_crops_below_the_floor():
    model = build(CFG).eval()
    with pytest.raises(ValueError, match="minimum"):
        model(torch.rand(1, 3, 16, 16))


def test_network_never_resizes():
    """No interpolate/resize anywhere: crops arrive at native resolution."""
    import inspect

    from provenance.branches import srm as module
    source = inspect.getsource(module)
    for banned in ("interpolate", "F.upsample", "Resize", "adaptive_avg_pool"):
        assert banned not in source


def test_gradient_flows_to_the_cnn_but_not_the_bank():
    model = build(CFG)
    model(torch.rand(2, 3, 64, 64)).sum().backward()
    assert all(p.grad is not None for p in model.features.parameters())
    assert not list(model.srm.buffers("weight")) or model.srm.weight.grad is None


def test_bank_weight_survives_an_optimiser_step():
    """The fixed layer must be identical after training, not merely detached."""
    model = build(CFG)
    before = model.srm.weight.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(
            model(torch.rand(4, 3, 64, 64)), torch.tensor([0, 1, 0, 1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.equal(model.srm.weight, before)


def test_pooling_is_delayed(monkeypatch):
    """Blocks before pool_after see the residual at native scale."""
    model = SRMNet(channels=(8, 8, 8, 8), pool_after=2)
    kinds = [type(m).__name__ for m in model.features]
    assert kinds.index("AvgPool2d") > 2, "pooling must not precede two conv blocks"
    assert kinds.count("AvgPool2d") == 1


def test_embedding_is_mean_and_std():
    model = SRMNet(channels=(8, 16), dropout=0.0).eval()
    with torch.no_grad():
        logits, embedding = model(torch.rand(2, 3, 64, 64), return_embedding=True)
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 32), "16 channels of mean plus 16 of std"


def smooth_content(n, size=64, seed=0):
    """Low-frequency content, the regime the TLU threshold was chosen for.

    White noise is the wrong fixture here: its residuals run ~90 on the 0-255
    scale and clip at TLU=3 almost everywhere, so nothing downstream can see
    anything. A real photo crop leaves ~17% of residuals clipped.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(n, 3, size // 8, size // 8, generator=generator)
    return torch.nn.functional.interpolate(coarse, size=(size, size), mode="bilinear")


def test_tlu_threshold_of_three_suits_photographic_content():
    """The literal 3.0 clips a minority of residuals on smooth content, not all of it.

    Pinned at 3.0 rather than read from the config, which now disables the TLU.
    This is the property that made 3.0 the standard choice, and it still holds;
    it is just not the property that decides the matter here.
    """
    clipped = SRMFilterBank(threshold=0.0)(smooth_content(4)).abs() > 3.0
    assert 0.0 < float(clipped.float().mean()) < 0.5


def test_tlu_is_disabled_by_default():
    """Clamping leaks compressibility: bpp R^2 0.977 at t=3 against 0.492 unclamped.

    Every non-zero threshold measured was worse on both bpp leakage and AUC on
    the bpp-matched val split. See the module docstring and configs/default.yaml.
    """
    assert float(CFG.model.srm.tlu_threshold) == 0.0


def test_disabled_threshold_leaves_residuals_untouched():
    x = smooth_content(2)
    assert torch.equal(SRMFilterBank(threshold=0.0)(x), SRMFilterBank(threshold=0)(x))
    assert build(CFG).srm.threshold == 0.0


def test_network_can_learn_a_noise_difference():
    """Sanity: a residual-domain signal is learnable end to end."""
    torch.manual_seed(int(CFG.seed))
    model = SRMNet(channels=(16, 32), dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def batch(n=16, seed=0):
        base = smooth_content(n, seed=seed)
        labels = torch.arange(n) % 2
        noisy = base + torch.randn_like(base) * 0.01 * labels.view(-1, 1, 1, 1)
        return noisy.clamp(0, 1), labels

    for step in range(40):
        x, y = batch(seed=step)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x, y = batch(64, seed=999)
        accuracy = (model(x).argmax(1) == y).float().mean()
    assert accuracy > 0.9, f"only {accuracy:.2f} on separable noise"
