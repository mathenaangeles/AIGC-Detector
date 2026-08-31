import torch
import torch.nn as nn
from omegaconf import OmegaConf

from provenance import train
from provenance.fuse import DegradationAwareGate, degradation_features


def test_degradation_features_are_finite_and_bounded():
    pixels = torch.rand(3, 3, 32, 40)
    sizes = torch.tensor([[224, 224], [1024, 768], [4096, 2160]])
    features = degradation_features(pixels, sizes)
    assert features.shape == (3, 3)
    assert torch.isfinite(features).all()
    assert ((0 <= features) & (features <= 1)).all()
    assert torch.all(features[1:, 2] > features[:-1, 2])


def test_block_boundaries_reduce_the_jpeg_quality_proxy():
    smooth = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 3, 32, 32)
    blocks = (torch.arange(32) // 8 % 2).float().view(1, 1, 1, 32)
    blocky = blocks.expand(1, 3, 32, 32)
    smooth_quality = degradation_features(smooth)[0, 1]
    blocky_quality = degradation_features(blocky)[0, 1]
    assert blocky_quality < smooth_quality


def test_laplacian_feature_distinguishes_flat_and_high_frequency_images():
    flat = torch.full((1, 3, 32, 32), 0.5)
    checker = ((torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2).float()
    checker = checker.view(1, 1, 32, 32).expand(1, 3, 32, 32)
    assert degradation_features(checker)[0, 0] > degradation_features(flat)[0, 0]


def test_gate_outputs_per_image_simplex_and_receives_gradients():
    gate = DegradationAwareGate(2, hidden=8)
    weights = gate(torch.rand(4, 3, 32, 32), torch.full((4, 2), 512.0))
    assert weights.shape == (4, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)
    weights[:, 0].sum().backward()
    assert all(parameter.grad is not None for parameter in gate.parameters())


class _ClipHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens):
        score = tokens.mean(dim=(1, 2)) * self.scale
        return torch.stack((-score, score), dim=1)


class _SRMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixels):
        score = pixels.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack((score, -score), dim=1)


def test_branch_ensemble_uses_gate_and_can_freeze_only_branches(monkeypatch):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.model.gating.enabled = True
    cfg.model.gating.hidden = 8
    monkeypatch.setattr(train, "build_probe", lambda _: _ClipHead())
    monkeypatch.setattr(train.srm_branch, "build", lambda _: _SRMHead())

    model = train.BranchEnsemble(cfg, ["clip", "srm"])
    pixels = torch.rand(3, 3, 32, 32)
    sizes = torch.full((3, 2), 512.0)
    logits = model(pixels, tokens=torch.rand(3, 4, 6), image_sizes=sizes)
    weights = model.fusion_weights(pixels, sizes)
    assert logits.shape == (3, 2)
    assert weights.shape == (3, 2)
    assert not torch.allclose(weights[0], weights.new_tensor([0.5, 0.5]))

    model.freeze_branches()
    assert not model.probe.scale.requires_grad
    assert not model.srm.scale.requires_grad
    assert model.gate.mlp[1].weight.requires_grad
