import numpy as np
import pytest

from provenance.evaluate import (
    accuracy_at,
    aggregate_crops,
    fixed_fpr_threshold,
    parse_run_spec,
    write_report,
)


def test_parse_run_spec_accepts_named_directory_and_checkpoint(tmp_path):
    run = tmp_path / "p7-both-kl1"
    run.mkdir()
    checkpoint = run / "model.pt"
    checkpoint.touch()

    assert parse_run_spec(str(run)) == ("p7-both-kl1", str(checkpoint))
    assert parse_run_spec(f"ours={checkpoint}") == ("ours", str(checkpoint))


def test_parse_run_spec_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_run_spec(str(tmp_path / "missing"))


def test_trimmed_crop_aggregation_is_per_image():
    scores = [0.0, 0.5, 1.0, 0.2, 0.4, 0.6]
    indices = [0, 0, 0, 1, 1, 1]
    assert np.allclose(aggregate_crops(scores, indices, 2), [0.5, 0.4])
    assert np.allclose(aggregate_crops(scores, indices, 2, "mean"), [0.5, 0.4])


def test_fixed_fpr_threshold_is_fitted_on_negatives_and_respects_budget():
    negatives = np.linspace(0.0, 0.99, 100)
    positives = np.linspace(0.5, 1.0, 100)
    scores = np.concatenate((negatives, positives))
    labels = np.concatenate((np.zeros(100), np.ones(100)))
    threshold = fixed_fpr_threshold(scores, labels, 0.01)

    assert np.mean(negatives >= threshold) == pytest.approx(0.01)
    assert 0.0 <= accuracy_at(scores, labels, threshold) <= 1.0


def test_fixed_fpr_threshold_moves_above_a_tie():
    scores = np.asarray([0.9, 0.9, 0.8, 0.1, 0.95])
    labels = np.asarray([0, 0, 0, 0, 1])
    threshold = fixed_fpr_threshold(scores, labels, 0.25)
    assert np.mean(scores[labels == 0] >= threshold) <= 0.25


def test_report_contains_auc_accuracy_mean_and_operating_point(tmp_path):
    result = {
        "protocol": "bias_matched",
        "split": "val_matched",
        "n_images": 10,
        "crops_per_image": 8,
        "target_fpr": 0.01,
        "conditions": ["clean", "jpeg_q30", "gaussian_blur_s2"],
        "methods": {"gated": {"path": "runs/gated/model.pt"}},
        "auc": {"gated": {"clean": 0.9, "jpeg_q30": 0.8, "gaussian_blur_s2": 0.6}},
        "accuracy": {
            "gated": {"clean": 0.75, "jpeg_q30": 0.70, "gaussian_blur_s2": 0.65}
        },
        "operating_points": {"gated": {"threshold": 0.8123456, "clean_fpr": 0.01}},
    }
    path = tmp_path / "reports" / "robustness_table.md"
    write_report(path, result)
    text = path.read_text()

    assert "Protocol: `bias_matched`" in text
    assert "| gated | 0.9000 | 0.8000 | 0.6000 | 0.7000 |" in text
    assert "Accuracy at the clean fixed-FPR threshold" in text
    assert "| gated | 0.812346 | 0.0100 |" in text
