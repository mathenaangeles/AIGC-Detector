"""The six competition transforms, exactly parameterised.

Semantics (the brief leaves these implicit; fixed here so evaluate.py, the
consistency loss and TTA all agree):

jpeg           -- re-encode at the given PIL quality and decode back, keeping
                  PIL's default chroma subsampling.
gaussian_blur  -- Gaussian kernel with that sigma; kernel size 2*ceil(3*sigma)+1.
resize         -- downscale by scale, then upscale back to the original
                  dimensions. Bilinear both directions.
gaussian_noise -- additive Gaussian, sigma on a [0,1] pixel scale, clip to [0,1].
color_jitter   -- brightness, contrast and saturation each multiplied by a factor
                  drawn from [1-strength, 1+strength]. Deterministic evaluation
                  uses the two extremes, applied jointly to all three, recorded
                  in the grid label as _lo / _hi.
center_crop    -- retain `fraction` of each linear dimension, no resize after.

apply_named() is the deterministic path used by evaluate.py's grid.
sample_random() is the training path: continuous ranges, wider than the eval
values, composing one or two transforms.
"""

import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageEnhance

NAMES = ("jpeg", "gaussian_blur", "resize", "gaussian_noise", "color_jitter", "center_crop")

DEFAULT_TRAIN_RANGES = {
    "jpeg": (20.0, 100.0),
    "gaussian_blur": (0.0, 2.5),
    "resize": (0.2, 1.0),
    "gaussian_noise": (0.0, 0.12),
    "color_jitter": (0.0, 0.25),
    "center_crop": (0.7, 1.0),
}

_LABEL_KEY = {
    "jpeg": "q",
    "gaussian_blur": "s",
    "resize": "s",
    "gaussian_noise": "s",
    "color_jitter": "s",
    "center_crop": "f",
}


def _to_float(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr):
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")


def gaussian_kernel_size(sigma):
    return 2 * int(math.ceil(3.0 * sigma)) + 1


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gaussian_blur(img, sigma):
    if sigma <= 0:
        return img.convert("RGB")
    ksize = gaussian_kernel_size(sigma)
    radius = ksize // 2
    offsets = np.arange(ksize, dtype=np.float64) - radius
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    arr = _to_float(img).astype(np.float64)
    padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    rows = np.tensordot(np.lib.stride_tricks.sliding_window_view(padded, ksize, axis=1), kernel, axes=([3], [0]))
    out = np.tensordot(np.lib.stride_tricks.sliding_window_view(rows, ksize, axis=0), kernel, axes=([3], [0]))
    return _to_image(out)


def _resize(img, scale):
    w, h = img.size
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    down = img.convert("RGB").resize((dw, dh), Image.BILINEAR)
    return down.resize((w, h), Image.BILINEAR)


def _gaussian_noise(img, sigma, rng):
    arr = _to_float(img)
    return _to_image(arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32))


def _color_jitter(img, factor):
    out = img.convert("RGB")
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        out = enhancer(out).enhance(factor)
    return out


def _center_crop(img, fraction):
    w, h = img.size
    cw, ch = max(1, int(w * fraction)), max(1, int(h * fraction))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.convert("RGB").crop((left, top, left + cw, top + ch))


def _derived_rng(name, value, size):
    """Stable seed from (transform, value, image size) so the grid reproduces run to run."""
    digest = hashlib.sha256(f"{name}:{value!r}:{size[0]}x{size[1]}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def apply_named(img, name, value, variant=None):
    """Apply one transform deterministically. `variant` is required by color_jitter only."""
    if name not in NAMES:
        raise ValueError(f"unknown transform {name!r}; expected one of {NAMES}")
    if name == "color_jitter":
        if variant not in ("lo", "hi"):
            raise ValueError("color_jitter requires variant='lo' or 'hi'")
    elif variant is not None:
        raise ValueError(f"{name} does not take a variant")

    if name == "jpeg":
        return _jpeg(img, value)
    if name == "gaussian_blur":
        return _gaussian_blur(img, float(value))
    if name == "resize":
        return _resize(img, float(value))
    if name == "gaussian_noise":
        return _gaussian_noise(img, float(value), _derived_rng(name, value, img.size))
    if name == "color_jitter":
        factor = 1.0 - float(value) if variant == "lo" else 1.0 + float(value)
        return _color_jitter(img, factor)
    return _center_crop(img, float(value))


def eval_grid(cfg):
    """Enumerate (label, name, value, variant) for every point in the robustness table."""
    spec = cfg["transforms"]["eval"]
    grid = []
    for name in NAMES:
        key = _LABEL_KEY[name]
        for value in spec[name]["values"]:
            label = f"{name}_{key}{float(value):g}"
            if name == "color_jitter":
                grid.extend((f"{label}_{v}", name, value, v) for v in ("lo", "hi"))
            else:
                grid.append((label, name, value, None))
    return grid


def train_ranges(cfg):
    """Pull the continuous training ranges out of cfg.transforms.train."""
    spec = cfg["transforms"]["train"]
    ranges = {}
    for name in NAMES:
        bounds = next(iter(spec[name].values()))
        ranges[name] = (float(bounds[0]), float(bounds[1]))
    return ranges, tuple(int(n) for n in spec["n_compose"])


def apply_records(img, records, rng):
    """Replay the transform sample_random recorded, onto a different image.

    This is how one T is shared across a batch: sample_random draws the spec
    once, and every image in the batch is put through that same spec. The draw
    is replayed from the record rather than re-sampled, so `value` -- and
    color_jitter's realised `factor`, which is a second draw inside the
    transform -- are identical across the batch.

    gaussian_noise is the deliberate exception. sigma is shared, the noise field
    is drawn per image from `rng`. A batch sharing one noise realisation would
    put an identical additive pattern on every crop in the batch, which is a
    correlation across the batch that no real degradation produces and that the
    consistency loss would happily learn to undo.
    """
    out = img.convert("RGB")
    for record in records:
        name, value = record["name"], record["value"]
        if name not in NAMES:
            raise ValueError(f"unknown transform {name!r}; expected one of {NAMES}")
        if name == "jpeg":
            out = _jpeg(out, int(value))
        elif name == "gaussian_blur":
            out = _gaussian_blur(out, float(value))
        elif name == "resize":
            out = _resize(out, float(value))
        elif name == "gaussian_noise":
            out = _gaussian_noise(out, float(value), rng)
        elif name == "color_jitter":
            out = _color_jitter(out, float(record["factor"]))
        else:
            out = _center_crop(out, float(value))
    return out


def sample_random(img, rng, cfg=None):
    """Compose 1-2 transforms with continuously sampled parameters. Returns (image, records)."""
    if cfg is None:
        ranges, n_compose = dict(DEFAULT_TRAIN_RANGES), (1, 2)
    else:
        ranges, n_compose = train_ranges(cfg)

    k = int(rng.integers(min(n_compose), max(n_compose) + 1))
    chosen = [str(n) for n in rng.choice(NAMES, size=k, replace=False)]
    # center_crop last, so the other transform sees the full frame.
    chosen.sort(key=lambda n: n == "center_crop")

    out = img.convert("RGB")
    records = []
    for name in chosen:
        lo, hi = ranges[name]
        value = float(rng.uniform(lo, hi))
        record = {"name": name, "value": value}
        if name == "jpeg":
            value = int(round(value))
            record["value"] = value
            out = _jpeg(out, value)
        elif name == "gaussian_blur":
            out = _gaussian_blur(out, value)
        elif name == "resize":
            out = _resize(out, value)
        elif name == "gaussian_noise":
            out = _gaussian_noise(out, value, rng)
        elif name == "color_jitter":
            factor = float(rng.uniform(1.0 - value, 1.0 + value))
            record["factor"] = factor
            out = _color_jitter(out, factor)
        else:
            out = _center_crop(out, value)
        records.append(record)
    return out, records
