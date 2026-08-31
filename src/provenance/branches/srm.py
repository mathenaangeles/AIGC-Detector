"""SRM high-pass residual CNN trained from scratch.

The first layer is the Spatial Rich Model's high-pass filter bank, fixed and
non-trainable. Every kernel sums to zero, so the layer discards image content
and keeps only the local noise residual -- the part of a photograph written by
the sensor and the demosaic, and the part a generator has to synthesise rather
than record. Freezing it matters: left trainable, these filters drift toward
low-pass content detectors within an epoch and the branch stops being a noise
model at all.

Sources:

    J. Fridrich and J. Kodovsky, "Rich Models for Steganalysis of Digital
    Images", IEEE TIFS 7(3):868-882, 2012.  Section III and Table 1 define the
    residual classes and the normalisation constants c used here.

    J. Ye, J. Ni, Y. Yi, "Deep Learning Hierarchical Representations for Image
    Steganalysis", IEEE TIFS 12(11):2545-2557, 2017.  Source of the practice of
    seeding a CNN's first layer with the SRM bank and of the Truncated Linear
    Unit (TLU) applied to the residuals.

    P. Zhou, X. Han, V. Morariu, L. Davis, "Learning Rich Features for Image
    Manipulation Detection", CVPR 2018.  Precedent for an SRM stream in a
    forensics network rather than a steganalysis one.

The bank is the standard 30: the linear residual classes of Table 1 at their
canonical orientations.

    class   n   support   c    description
    D1      8   5x5       1    first-order differences, 8 neighbours
    D2      4   5x5       2    second-order [1 -2 1], 4 axes
    D3      8   5x5       3    third-order [1 -3 3 -1], 8 directions
    S3      1   3x3       4    3x3 square
    E3      4   3x3       4    3x3 square with one row zeroed, 4 rotations
    S5      1   5x5      12    5x5 square
    E5      4   5x5      12    5x5 square with two rows zeroed, 4 rotations

Each kernel is divided by its c so residual magnitudes are comparable across
classes; this is what makes a single TLU threshold meaningful for all 30.

Two deviations from steganalysis practice, both because this is a forensics
problem and not a payload-detection one:

    Colour is kept. Steganalysis works on greyscale; demosaicing correlates
    neighbouring pixels differently per channel, and that correlation is a
    camera-pipeline signature, so the bank runs depthwise over R, G and B
    separately -- 30 kernels x 3 channels = 90 residual maps.

    Downsampling is delayed. The signal lives at the pixel scale, so the first
    two convolutions run at full resolution and pooling only starts afterwards,
    following SRNet's argument that early subsampling averages the residual
    away before anything has read it.

The network is fully convolutional and ends in global pooling, so it consumes
native-resolution crops of any size at or above the 32px floor. No resizing
happens here or anywhere upstream.

The TLU is implemented but disabled by default, which is a deliberate break
with the steganalysis practice cited above. Clamping to +/-t throws away
residual magnitude and leaves the output dominated by how often the residual
saturates -- on bias-matched SID_Set crops roughly 47% of real residuals
against 28% of synthetic ones at t=3. "How often does the local residual exceed
a threshold" is close to a description of where a JPEG encoder spends its bits,
and it shows: the clamped residual energy correlates with bytes-per-pixel at
r=0.974, and scores AUC 0.779 on its own, which is above what bytes-per-pixel
itself scores. The branch would start training with the dataset's
compressibility shortcut already handed to it in a highly learnable form.

Measured across thresholds -- ridge R^2 predicting bpp from the 90 per-map
energies, and logistic AUC on the bpp-matched val split -- every non-zero
threshold is worse on both axes than no threshold at all, so it is not a value
to tune. Setting `tlu_threshold: 0` disables the clamp. Instance-normalising the
residual instead was also tried and was worse than leaving it alone.

Unclamped residuals are heavy-tailed. The BatchNorm in the first conv block
absorbs the scale, and the training loop clips gradients; if a run does go
unstable, a large threshold is the escape hatch, at the cost above.
"""

import numpy as np
import torch
import torch.nn as nn

NEIGHBOURS_8 = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
AXES_4 = ((0, 1), (1, 1), (1, 0), (1, -1))

SQUARE_3 = np.array([[-1, 2, -1],
                     [2, -4, 2],
                     [-1, 2, -1]], dtype=np.float64)

SQUARE_5 = np.array([[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]], dtype=np.float64)


def _embed(kernel, size=5):
    """Centre a 3x3 kernel in a 5x5 window."""
    out = np.zeros((size, size), dtype=np.float64)
    offset = (size - kernel.shape[0]) // 2
    out[offset:offset + kernel.shape[0], offset:offset + kernel.shape[1]] = kernel
    return out


def _first_order():
    """D1: x[n+1] - x[n] along each of the 8 neighbour directions. c = 1."""
    kernels = []
    for dr, dc in NEIGHBOURS_8:
        k = np.zeros((5, 5))
        k[2, 2] = -1.0
        k[2 + dr, 2 + dc] = 1.0
        kernels.append(k)
    return kernels


def _second_order():
    """D2: [1 -2 1] along each of the 4 axes. c = 2."""
    kernels = []
    for dr, dc in AXES_4:
        k = np.zeros((5, 5))
        k[2, 2] = -2.0
        k[2 + dr, 2 + dc] = 1.0
        k[2 - dr, 2 - dc] = 1.0
        kernels.append(k / 2.0)
    return kernels


def _third_order():
    """D3: [1 -3 3 -1] along each of the 8 directions. c = 3."""
    kernels = []
    for dr, dc in NEIGHBOURS_8:
        k = np.zeros((5, 5))
        k[2 - dr, 2 - dc] = 1.0
        k[2, 2] = -3.0
        k[2 + dr, 2 + dc] = 3.0
        k[2 + 2 * dr, 2 + 2 * dc] = -1.0
        kernels.append(k / 3.0)
    return kernels


def _edges(square, n_zero_rows, c):
    """The square kernel with its trailing rows zeroed, in 4 rotations.

    Fridrich's EDGE residuals are the square filters restricted to a half
    plane, which is what gives them their orientation selectivity.
    """
    base = square.copy()
    base[base.shape[0] - n_zero_rows:, :] = 0.0
    return [_embed(np.rot90(base, k) / c) for k in range(4)]


def srm_kernels():
    """The 30 kernels as a (30, 5, 5) array, each normalised by its c."""
    kernels = (
        _first_order()
        + _second_order()
        + _third_order()
        + [_embed(SQUARE_3 / 4.0)]
        + _edges(SQUARE_3, 1, 4.0)
        + [SQUARE_5 / 12.0]
        + _edges(SQUARE_5, 2, 12.0)
    )
    bank = np.stack(kernels).astype(np.float32)
    if bank.shape != (30, 5, 5):
        raise AssertionError(f"expected 30 kernels, built {bank.shape}")
    return bank


KERNEL_CLASSES = ("D1",) * 8 + ("D2",) * 4 + ("D3",) * 8 + ("S3",) + ("E3",) * 4 + ("S5",) + ("E5",) * 4


class SRMFilterBank(nn.Module):
    """Fixed high-pass front end. Never trains.

    Registered as a buffer rather than a Parameter, so it cannot pick up an
    optimiser by accident and is not counted as a trainable parameter. Input is
    [0,1]; the bank is rescaled to the 0-255 range the SRM constants were
    derived for, which is also what makes the TLU threshold comparable to the
    steganalysis literature's.
    """

    def __init__(self, in_channels=3, threshold=3.0):
        super().__init__()
        bank = torch.from_numpy(srm_kernels()).unsqueeze(1)
        self.register_buffer("weight", bank.repeat(in_channels, 1, 1, 1) * 255.0)
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels * bank.shape[0]
        self.threshold = float(threshold)

    def train(self, mode=True):
        """The bank has nothing to train; keep it out of train-mode semantics."""
        return super().train(False)

    def forward(self, x):
        x = nn.functional.conv2d(x, self.weight, padding=2, groups=self.in_channels)
        if self.threshold:
            # TLU (Ye et al. 2017): the informative residuals are small, and
            # clipping stops a few saturated edge pixels dominating the scale.
            x = torch.clamp(x, -self.threshold, self.threshold)
        return x


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SRMNet(nn.Module):
    """Fixed SRM bank, then a small CNN to 2 logits.

    Pooling starts after the second block, so the two convolutions that see the
    residual at its native scale see it undamaged.
    """

    MIN_SIZE = 32

    def __init__(self, channels=(64, 128, 256, 256, 512, 512), in_channels=3,
                 threshold=3.0, dropout=0.1, n_classes=2, pool_after=2):
        super().__init__()
        self.srm = SRMFilterBank(in_channels, threshold)

        blocks, prev = [], self.srm.out_channels
        for i, out_ch in enumerate(channels):
            blocks.append(conv_block(prev, out_ch))
            if i >= int(pool_after) and i < len(channels) - 1:
                blocks.append(nn.AvgPool2d(2))
            prev = out_ch
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(float(dropout))
        # Mean and standard deviation: a residual field is characterised as
        # much by its dispersion as by its average, and mean alone is close to
        # zero by construction for a zero-sum filter bank.
        self.classifier = nn.Linear(prev * 2, int(n_classes))

    def forward(self, x, return_embedding=False):
        if min(x.shape[-2:]) < self.MIN_SIZE:
            raise ValueError(f"crop is {tuple(x.shape[-2:])}, minimum is {self.MIN_SIZE}")
        feats = self.features(self.srm(x))
        flat = feats.flatten(2)
        pooled = torch.cat([flat.mean(-1), flat.std(-1, unbiased=False)], dim=1)
        logits = self.classifier(self.dropout(pooled))
        return (logits, pooled) if return_embedding else logits

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(cfg):
    srm = cfg.model.srm
    return SRMNet(
        channels=tuple(int(c) for c in srm.channels),
        threshold=float(srm.tlu_threshold),
        dropout=float(srm.dropout),
        pool_after=int(srm.pool_after),
    )
