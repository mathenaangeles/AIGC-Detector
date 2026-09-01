import json
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "aigc_predict", Path(__file__).parents[1] / "predict.py")
predict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict)


def test_predict_writes_minimal_contract_and_detailed_sidecar(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "b.png").write_bytes(b"fixture")
    (image_dir / "a.jpg").write_bytes(b"fixture")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    out = tmp_path / "predictions.json"

    monkeypatch.setattr(predict, "predict_paths", lambda paths, **kwargs: [
        {"pred": 0.25, "stability": 0.01},
        {"pred": 0.75, "stability": 0.02},
    ])
    assert predict.main([
        "--image_dir", str(image_dir),
        "--checkpoint", str(checkpoint),
        "--calibration", "none",
        "--device", "cpu",
        "--out", str(out),
    ]) == 0

    minimal = json.loads(out.read_text())
    detailed = json.loads((tmp_path / "predictions_detailed.json").read_text())
    assert minimal == [
        {"image_path": "a.jpg", "pred": 0.25},
        {"image_path": "b.png", "pred": 0.75},
    ]
    assert set(minimal[0]) == {"image_path", "pred"}
    assert detailed[0] == {
        "image_path": "a.jpg", "pred": 0.25, "stability": 0.01}
