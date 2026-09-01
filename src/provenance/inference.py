"""Real-checkpoint multi-crop, transformation-time-augmentation inference."""

import hashlib
import json
import math
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .branches.clip_probe import FrozenCLIP, resolve_device
from .data import bias_match, reflect_pad_to
from .transforms import apply_named

TTA_NAMES = ("identity", "jpeg90", "resize0.5", "crop80")


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def conform(image, size):
    image = reflect_pad_to(image.convert("RGB"), int(size))
    width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _axis_positions(length, size, overlap):
    length = max(int(length), int(size))
    if length == size:
        return [0]
    stride = max(1, int(round(size * (1.0 - float(overlap)))))
    positions = list(range(0, length - size + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def overlapping_crop_boxes(width, height, size=224, overlap=0.5, max_crops=8):
    """Cover the image with an overlapping grid, deterministically capped."""
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if int(size) < 1 or int(max_crops) < 1:
        raise ValueError("size and max_crops must be positive")
    xs = _axis_positions(width, size, overlap)
    ys = _axis_positions(height, size, overlap)
    boxes = [(x, y) for y in ys for x in xs]
    if len(boxes) <= int(max_crops):
        return boxes

    # Even indices preserve all parts of the row-major spatial grid instead of
    # taking only the top-left crops of a large image.
    selected = np.linspace(0, len(boxes) - 1, int(max_crops)).round().astype(int)
    return [boxes[index] for index in selected]


def apply_tta(image, name, size):
    if name == "identity":
        transformed = image.convert("RGB")
    elif name == "jpeg90":
        transformed = apply_named(image, "jpeg", 90)
    elif name == "resize0.5":
        transformed = apply_named(image, "resize", 0.5)
    elif name == "crop80":
        transformed = apply_named(image, "center_crop", 0.8)
    else:
        raise ValueError(f"unknown TTA view {name!r}; expected one of {TTA_NAMES}")
    return conform(transformed, size)


def trimmed_mean(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        raise ValueError("cannot aggregate an empty score collection")
    if len(values) >= 3:
        values = values[1:-1]
    return float(values.mean())


def probabilities_from_margins(margins, temperature=1.0):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    margins = np.asarray(margins, dtype=np.float64) / temperature
    # Stable sigmoid without requiring scipy.
    probabilities = np.empty_like(margins)
    positive = margins >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-margins[positive]))
    exp_margin = np.exp(margins[~positive])
    probabilities[~positive] = exp_margin / (1.0 + exp_margin)
    return probabilities


def prediction_from_margins(margins, temperature=1.0):
    probabilities = probabilities_from_margins(margins, temperature)
    return trimmed_mean(probabilities), float(np.var(probabilities))


class TTAImageDataset(Dataset):
    """All overlapping crop/TTA views, with indices for image aggregation."""

    def __init__(self, paths, size=224, overlap=0.5, max_crops=8,
                 protocol="bias_matched", match_quality=90):
        self.paths = [os.fspath(path) for path in paths]
        self.size = int(size)
        self.protocol = str(protocol)
        self.match_quality = int(match_quality)
        if self.protocol not in ("raw", "bias_matched"):
            raise ValueError("protocol must be raw or bias_matched")

        self.image_sizes, self.items, self.crop_counts = [], [], []
        for image_index, path in enumerate(self.paths):
            with Image.open(path) as image:
                width, height = image.size
            boxes = overlapping_crop_boxes(
                width, height, self.size, float(overlap), int(max_crops))
            self.image_sizes.append((width, height))
            self.crop_counts.append(len(boxes))
            for crop_index, box in enumerate(boxes):
                for tta_index, tta_name in enumerate(TTA_NAMES):
                    self.items.append((image_index, crop_index, box, tta_index, tta_name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_index, _, box, tta_index, tta_name = self.items[index]
        path = self.paths[image_index]
        try:
            with Image.open(path) as image:
                image.load()
                image = reflect_pad_to(image.convert("RGB"), self.size)
                left, top = box
                crop = image.crop((left, top, left + self.size, top + self.size))
        except Exception as error:
            raise RuntimeError(f"could not read image {path}: {error}") from error
        if self.protocol == "bias_matched":
            crop = bias_match(crop, self.match_quality)
        view = apply_tta(crop, tta_name, self.size)
        width, height = self.image_sizes[image_index]
        return (to_tensor(view), image_index, tta_index,
                torch.tensor([float(width), float(height)]))


class CheckpointPredictor:
    """A loaded branch ensemble and its frozen CLIP tower, reusable per process."""

    def __init__(self, checkpoint, device="cpu", calibration=None,
                 protocol="bias_matched"):
        from .train import BranchEnsemble

        self.checkpoint = os.path.abspath(os.fspath(checkpoint))
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(self.checkpoint)
        self.device = resolve_device(device)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.cfg = OmegaConf.create(saved["config"])
        self.branches = list(saved["branches"])
        self.protocol = str(protocol)
        if self.protocol not in ("raw", "bias_matched"):
            raise ValueError("protocol must be raw or bias_matched")

        self.model = BranchEnsemble(self.cfg, self.branches, backbone=None)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.backbone = None
        if "clip" in self.branches:
            self.backbone = FrozenCLIP(
                str(self.cfg.model.clip.arch), str(self.cfg.model.clip.pretrained),
                device=self.device)
        self.temperature, self.threshold, self.calibration = 1.0, None, None
        if calibration is not None:
            self.load_calibration(calibration)

    def load_calibration(self, path):
        with open(path) as handle:
            calibration = json.load(handle)
        expected = calibration.get("checkpoint_sha256")
        actual = checkpoint_sha256(self.checkpoint)
        if expected and expected != actual:
            raise ValueError("calibration belongs to a different checkpoint")
        calibrated_protocol = calibration.get("protocol")
        if calibrated_protocol and calibrated_protocol != self.protocol:
            raise ValueError(
                f"calibration protocol {calibrated_protocol!r} does not match "
                f"inference protocol {self.protocol!r}")
        self.temperature = float(calibration["temperature"])
        self.threshold = float(calibration["threshold"])
        self.calibration = calibration

    @torch.inference_mode()
    def score_margins(self, paths, batch_size=4, max_crops=8, overlap=0.5,
                      num_workers=0):
        dataset = TTAImageDataset(
            paths, size=int(self.cfg.data.crop_size), overlap=overlap,
            max_crops=max_crops, protocol=self.protocol,
            match_quality=int(self.cfg.data.match_quality))
        loader = DataLoader(
            dataset, batch_size=int(batch_size), shuffle=False,
            num_workers=int(num_workers), pin_memory=self.device.type == "cuda")
        grouped = [[] for _ in paths]
        grouped_tta = [{name: [] for name in TTA_NAMES} for _ in paths]
        use_amp = self.device.type == "cuda"
        dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

        for pixels, image_indices, tta_indices, image_sizes in loader:
            pixels = pixels.to(self.device, non_blocking=True)
            image_sizes = image_sizes.to(self.device, non_blocking=True)
            with torch.autocast(
                    device_type=self.device.type, enabled=use_amp, dtype=dtype):
                tokens = self.backbone(pixels)[1] if self.backbone is not None else None
                logits = self.model(pixels, tokens, image_sizes=image_sizes)
            margins = (logits[:, 1] - logits[:, 0]).float().cpu()
            if not torch.isfinite(margins).all():
                raise FloatingPointError("model produced a non-finite inference logit")
            for margin, image_index, tta_index in zip(margins, image_indices, tta_indices):
                image_index, tta_index = int(image_index), int(tta_index)
                value = float(margin)
                grouped[image_index].append(value)
                grouped_tta[image_index][TTA_NAMES[tta_index]].append(value)
        if any(not values for values in grouped):
            raise RuntimeError("at least one image received no inference views")
        return grouped, grouped_tta, dataset.crop_counts

    def predict_paths(self, paths, batch_size=4, max_crops=8, overlap=0.5,
                      num_workers=0):
        if self.calibration is not None:
            expected_tta = tuple(self.calibration.get("tta", TTA_NAMES))
            if expected_tta != TTA_NAMES:
                raise ValueError("calibration uses a different TTA set")
            if int(self.calibration.get("max_crops", max_crops)) != int(max_crops):
                raise ValueError("max_crops does not match the calibration artifact")
            if not math.isclose(
                    float(self.calibration.get("overlap", overlap)), float(overlap),
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("overlap does not match the calibration artifact")
        margins, tta_margins, crop_counts = self.score_margins(
            paths, batch_size, max_crops, overlap, num_workers)
        details = []
        for image_margins, per_tta, n_crops in zip(margins, tta_margins, crop_counts):
            prediction, variance = prediction_from_margins(
                image_margins, self.temperature)
            record = {
                "pred": prediction,
                # Stability is the requested predictive variance: lower means
                # the decision survives crop and degradation changes better.
                "stability": variance,
                "n_crops": int(n_crops),
                "n_views": len(image_margins),
                "tta_predictions": {
                    name: trimmed_mean(probabilities_from_margins(values, self.temperature))
                    for name, values in per_tta.items()
                },
                "temperature": self.temperature,
            }
            if self.threshold is not None:
                record["threshold"] = self.threshold
                record["decision"] = int(prediction >= self.threshold)
            details.append(record)
        return details
