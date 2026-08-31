"""Degradation-aware fusion from cheap, content-light image measurements.

The gate sees three scalars rather than semantic features: Laplacian variance
for sharpness, an 8x8-boundary blockiness ratio mapped to a JPEG-quality proxy,
and source image area. It cannot become a second content classifier; it can only
learn which forensic branch is reliable under the observed image condition.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def degradation_features(pixels, image_sizes=None):
    """Return [sharpness, JPEG-quality proxy, image-size] in stable ranges.

    ``pixels`` is Bx3xHxW in [0, 1]. ``image_sizes`` is Bx2 source width/height;
    when unavailable, the tensor dimensions are used. Measurements are detached
    because the gate should adapt to degradation, not alter pixels to game its
    own estimator.
    """
    x = pixels.detach().float().clamp(0, 1)
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    laplacian = gray.new_tensor([[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    response = F.conv2d(gray, laplacian, padding=1)
    variance = response.var(dim=(-2, -1), unbiased=False).squeeze(1)
    sharpness = (torch.log1p(variance * 1000.0) / math.log(1001.0)).clamp(0, 1)

    dx = (gray[..., 1:] - gray[..., :-1]).abs()
    dy = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
    bx, by = dx[..., 7::8], dy[..., 7::8, :]
    boundary_n = bx[0].numel() + by[0].numel() if len(x) else 1
    boundary = (bx.sum(dim=(1, 2, 3)) + by.sum(dim=(1, 2, 3))) / max(boundary_n, 1)

    mask_x = torch.ones(dx.shape[-1], dtype=torch.bool, device=dx.device)
    mask_y = torch.ones(dy.shape[-2], dtype=torch.bool, device=dy.device)
    mask_x[7::8] = False
    mask_y[7::8] = False
    ix, iy = dx[..., mask_x], dy[..., mask_y, :]
    interior_n = ix[0].numel() + iy[0].numel() if len(x) else 1
    interior = (ix.sum(dim=(1, 2, 3)) + iy.sum(dim=(1, 2, 3))) / max(interior_n, 1)
    blockiness = boundary / (interior + 1e-6)
    quality = (1.0 / (1.0 + 4.0 * F.relu(blockiness - 1.0))).clamp(0, 1)

    if image_sizes is None:
        sizes = x.new_tensor([x.shape[-1], x.shape[-2]]).expand(len(x), 2)
    else:
        sizes = image_sizes.detach().to(device=x.device, dtype=torch.float32)
    linear_size = sizes.prod(dim=1).clamp_min(1).sqrt()
    size = ((torch.log2(linear_size / 224.0) + 2.0) / 6.0).clamp(0, 1)
    return torch.stack((sharpness, quality, size), dim=1)


class DegradationAwareGate(nn.Module):
    """Map degradation measurements to per-image softmax branch weights."""

    def __init__(self, n_branches, hidden=64):
        super().__init__()
        if n_branches < 2:
            raise ValueError("degradation-aware fusion requires at least two branches")
        self.n_branches = int(n_branches)
        self.mlp = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.n_branches),
        )

    def forward(self, pixels, image_sizes=None):
        return torch.softmax(self.mlp(degradation_features(pixels, image_sizes)), dim=-1)


def build(cfg, n_branches):
    return DegradationAwareGate(n_branches, hidden=int(cfg.model.gating.hidden))
