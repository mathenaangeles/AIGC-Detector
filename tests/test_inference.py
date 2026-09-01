import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from provenance.inference import (
    TTA_NAMES,
    CheckpointPredictor,
    TTAImageDataset,
    apply_tta,
    checkpoint_sha256,
    overlapping_crop_boxes,
    prediction_from_margins,
    safe_amp_dtype,
    trimmed_mean,
)
from provenance.train import BranchEnsemble


def test_overlapping_boxes_cover_boundaries_and_respect_cap():
    assert overlapping_crop_boxes(224, 224) == [(0, 0)]
    boxes = overlapping_crop_boxes(448, 448, size=224, overlap=0.5, max_crops=8)
    assert len(boxes) == 8
    assert boxes[0] == (0, 0)
    assert boxes[-1] == (224, 224)
    assert len(set(boxes)) == len(boxes)


def test_overlapping_boxes_validate_settings():
    with pytest.raises(ValueError, match="overlap"):
        overlapping_crop_boxes(300, 300, overlap=1.0)
    with pytest.raises(ValueError, match="positive"):
        overlapping_crop_boxes(300, 300, max_crops=0)


def test_all_required_tta_views_return_model_sized_images():
    array = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    image = Image.fromarray(array, "RGB")
    views = {name: apply_tta(image, name, 224) for name in TTA_NAMES}
    assert tuple(views) == ("identity", "jpeg90", "resize0.5", "crop80")
    assert all(view.size == (224, 224) for view in views.values())
    assert not np.array_equal(np.asarray(views["identity"]), np.asarray(views["jpeg90"]))


def test_prediction_uses_trimmed_mean_and_population_variance():
    margins = np.log(np.asarray([0.1, 0.2, 0.8, 0.9]) / np.asarray([0.9, 0.8, 0.2, 0.1]))
    prediction, stability = prediction_from_margins(margins)
    assert prediction == pytest.approx(0.5)
    assert stability == pytest.approx(np.var([0.1, 0.2, 0.8, 0.9]))
    assert trimmed_mean([0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.5)


def test_tta_dataset_has_four_views_per_crop(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (448, 224), (100, 120, 140)).save(path)
    dataset = TTAImageDataset([path], max_crops=8, protocol="bias_matched")
    assert dataset.crop_counts == [3]
    assert len(dataset) == 3 * len(TTA_NAMES)
    pixels, image_index, tta_index, size = dataset[0]
    assert pixels.shape == (3, 224, 224)
    assert (image_index, tta_index) == (0, 0)
    assert size.tolist() == [448.0, 224.0]


def make_srm_checkpoint(tmp_path):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.data.crop_size = 32
    cfg.data.match_quality = 90
    cfg.model.srm.channels = [4, 8]
    cfg.model.srm.pool_after = 2
    cfg.model.srm.dropout = 0.0
    model = BranchEnsemble(cfg, ["srm"])
    path = tmp_path / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "branches": ["srm"],
        "epoch": 0,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)
    return path


def test_checkpoint_predictor_runs_real_cpu_multicrop_tta(tmp_path):
    checkpoint = make_srm_checkpoint(tmp_path)
    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(3).integers(
        0, 256, (48, 64, 3), dtype=np.uint8), "RGB").save(image)
    predictor = CheckpointPredictor(checkpoint, device="cpu", protocol="bias_matched")
    details = predictor.predict_paths([image], batch_size=2, max_crops=3)
    result = details[0]
    assert 0.0 <= result["pred"] <= 1.0
    assert result["stability"] >= 0.0
    assert result["n_crops"] == 3
    assert result["n_views"] == 3 * len(TTA_NAMES)
    assert set(result["tta_predictions"]) == set(TTA_NAMES)


def test_old_cuda_devices_do_not_enable_unsafe_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert safe_amp_dtype(torch.device("cuda")) is None
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert safe_amp_dtype(torch.device("cuda")) == torch.bfloat16
    assert safe_amp_dtype(torch.device("cpu")) is None


def test_calibration_is_bound_to_checkpoint_protocol_and_crop_grid(tmp_path):
    checkpoint = make_srm_checkpoint(tmp_path)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "protocol": "bias_matched",
        "temperature": 2.0,
        "threshold": 0.75,
        "max_crops": 8,
        "overlap": 0.5,
        "tta": list(TTA_NAMES),
    }))
    predictor = CheckpointPredictor(
        checkpoint, device="cpu", calibration=calibration,
        protocol="bias_matched")
    assert predictor.temperature == 2.0
    assert predictor.threshold == 0.75
    with pytest.raises(ValueError, match="max_crops"):
        predictor.predict_paths([], max_crops=4)
    with pytest.raises(ValueError, match="protocol"):
        CheckpointPredictor(
            checkpoint, device="cpu", calibration=calibration, protocol="raw")
