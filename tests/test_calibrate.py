import numpy as np

from provenance.calibrate import (
    binary_nll,
    calibrated_predictions,
    fit_temperature,
    write_report,
)


def test_temperature_scaling_reduces_overconfident_nll():
    groups = [
        [8.0, 9.0, 10.0, 11.0],
        [8.0, 9.0, 10.0, 11.0],
        [-11.0, -10.0, -9.0, -8.0],
        [-11.0, -10.0, -9.0, -8.0],
    ]
    labels = np.asarray([1, 0, 0, 1])
    before = binary_nll(calibrated_predictions(groups, 1.0), labels)
    temperature = fit_temperature(groups, labels)
    after = binary_nll(calibrated_predictions(groups, temperature), labels)
    assert temperature > 1.0
    assert after < before


def test_calibration_report_contains_fixed_operating_point(tmp_path):
    result = {
        "checkpoint": "runs/gated/model.pt",
        "protocol": "bias_matched",
        "split": "val_matched",
        "n_images": 904,
        "temperature": 1.234567,
        "nll_before": 0.2,
        "nll_after": 0.1,
        "auc": 0.999,
        "target_fpr": 0.01,
        "threshold": 0.876543,
        "achieved_fpr": 0.0088,
        "accuracy": 0.98,
    }
    path = tmp_path / "calibration.md"
    write_report(path, result)
    text = path.read_text()
    assert "Temperature: `1.234567`" in text
    assert "Target FPR: `1.00%`" in text
    assert "Threshold: `0.876543`" in text

