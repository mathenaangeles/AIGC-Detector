import importlib.util
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    "error_analysis", Path(__file__).parents[1] / "scripts/error_analysis.py")
error_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(error_analysis)


def test_transform_type_preserves_the_six_named_families():
    assert error_analysis.transform_type("clean") == "clean"
    assert error_analysis.transform_type("jpeg_q30") == "jpeg"
    assert error_analysis.transform_type("gaussian_blur_s2") == "gaussian_blur"
    assert error_analysis.transform_type("gaussian_noise_s0.1") == "gaussian_noise"
    assert error_analysis.transform_type("color_jitter_s0.2_lo") == "color_jitter"
    assert error_analysis.transform_type("center_crop_f0.8") == "center_crop"


def test_generator_grouping_reports_unknown_instead_of_inventing_sid_metadata():
    cfg = OmegaConf.load("configs/default.yaml")
    assert error_analysis.generator_name(
        {"label": 1, "source": "sid_set", "path": "fake/a.png"}, cfg) == "unknown"
    assert error_analysis.generator_name(
        {"label": 1, "source": "dalle_advanced", "path": "fake/a.png"}, cfg
    ) == "dalle_advanced"
    assert error_analysis.generator_name(
        {"label": 0, "source": "coco_val2017", "path": "real/a.jpg"}, cfg
    ) == "not_applicable"


def test_grouped_counts_uses_class_specific_error_denominators():
    entries = [
        {"transform": "jpeg", "label": 0, "error": "fp"},
        {"transform": "jpeg", "label": 0, "error": ""},
        {"transform": "jpeg", "label": 1, "error": "fn"},
        {"transform": "jpeg", "label": 1, "error": ""},
    ]
    grouped = error_analysis.grouped_counts(entries, "transform")
    assert grouped["jpeg"] == {
        "evaluated": 4, "real": 2, "synthetic": 2, "fp": 1, "fn": 1}
    table = error_analysis.table_for_groups("Transforms", grouped)
    assert "| jpeg | 4 | 1 | 1 | 0.5000 | 0.5000 |" in table


def test_contact_sheet_and_report_are_written(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (48, 48), (80, 100, 120)).save(image_path)
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.paths.root = str(tmp_path)
    cfg.data.crop_size = 32
    rows = [{"path": "sample.png"}]
    ranked = [{
        "row_index": 0, "path": "sample.png", "error": "fp",
        "condition": "clean", "score": 0.9, "source": "fixture",
        "generator": "unknown", "confidence": 0.4,
    }]
    sheet = tmp_path / "sheet.png"
    error_analysis.contact_sheet(
        sheet, ranked, rows, cfg, "raw", [("clean", None)])
    assert sheet.is_file()
    with Image.open(sheet) as image:
        assert image.size == (1560, 324)

    groups = {"fixture": {
        "evaluated": 2, "real": 1, "synthetic": 1, "fp": 1, "fn": 0}}
    result = {
        "protocol": "raw", "split": "val", "n_images": 2,
        "checkpoint": "runs/model.pt", "threshold": 0.5,
        "achieved_clean_fpr": 0.01, "n_conditions": 1,
        "crops_per_image": 8, "contact_sheet": str(sheet),
        "groups": {"transform": groups, "source": groups,
                   "generator": {"unknown": groups["fixture"]}},
        "ranked": ranked,
    }
    report = tmp_path / "report.md"
    error_analysis.write_report(report, result)
    text = report.read_text()
    assert "## Measured failure modes" in text
    assert "Highest-confidence errors" in text
