from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from uniscan.office_lens.adapter import (
    OfficeLensOnnx,
    _expand_quad,
    _quad_area,
    _quad_image_score,
    _quad_iou,
    choose_quad,
    detect_bright_document_quad,
    detect_edge_document_quad,
    detect_image_document_quad,
    enhance_page,
    mask_to_quad,
    mode_from_classification,
    result_to_report,
    save_overlay,
    save_pipeline_outputs,
    warp_document,
    warp_document_rgb,
)
from uniscan.office_lens.cli import main as office_lens_main


def _scene() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((180, 240, 3), 28, dtype=np.uint8)
    quad = np.float32([[35, 20], [210, 30], [220, 160], [25, 150]])
    cv2.fillConvexPoly(image, quad.astype(np.int32), (245, 245, 245))
    cv2.line(image, (55, 70), (185, 75), (45, 45, 45), 3)
    cv2.line(image, (50, 100), (190, 105), (45, 45, 45), 3)
    return image, quad


def test_quad_helpers_cover_selection_and_error_cases() -> None:
    image, quad = _scene()
    shifted = quad + np.float32([3, 2])

    assert _quad_area(None) == 0.0
    assert _quad_area(quad) > 20_000
    assert _quad_iou(None, quad) == 0.0
    assert _quad_iou(quad, shifted) > 0.8
    assert _quad_image_score(None, image) == -1.0
    assert _quad_image_score(quad, image) > 0.0
    assert np.array_equal(_expand_quad(quad, 240, 180, 0), quad)
    assert np.all(_expand_quad(quad, 240, 180, 0.1) >= 0)
    assert np.array_equal(choose_quad(None, quad, 240, 180), quad)
    assert np.array_equal(choose_quad(quad, None, 240, 180), quad)
    assert choose_quad(quad, shifted, 240, 180) is not None


def test_mask_and_image_detectors_find_synthetic_document() -> None:
    image, _quad = _scene()
    mask = np.zeros((256, 256), dtype=np.float32)
    cv2.rectangle(mask, (30, 25), (225, 225), 0.9, -1)

    detected, threshold = mask_to_quad(mask, image_width=240, image_height=180)

    assert detected is not None
    assert threshold > 0
    assert mask_to_quad(np.zeros((0, 0), dtype=np.float32), 100, 100) == (None, 0.0)
    assert detect_bright_document_quad(image) is not None
    assert detect_edge_document_quad(image, max_edge=120) is not None
    assert detect_image_document_quad(image) is not None
    assert detect_bright_document_quad(np.zeros((80, 80, 3), dtype=np.uint8)) is None


def test_enhancement_modes_and_perspective_warp() -> None:
    image, quad = _scene()
    for mode, variants in (
        ("document", {"enhanced", "gray", "bw"}),
        ("whiteboard", {"enhanced", "gray"}),
        ("photo", {"enhanced"}),
    ):
        result = enhance_page(image, mode)
        assert result.image.size > 0
        assert set(result.variants) == variants
    with pytest.raises(ValueError, match="Unsupported"):
        enhance_page(image, "unknown")
    assert mode_from_classification(" Photo ") == "photo"
    assert mode_from_classification("other") == "document"

    warped, size = warp_document_rgb(image, quad, padding_percent=0.02)
    assert warped.shape[1::-1] == size
    assert min(size) > 100


def test_real_model_file_flow_reports_and_saves_all_outputs(tmp_path) -> None:
    source = tmp_path / "scene.png"
    rgb, quad = _scene()
    cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    runner = OfficeLensOnnx()

    result = runner.process_file(source, mode="document")
    report = result_to_report(source, result)
    outputs = save_pipeline_outputs(source, result, tmp_path / "out")

    assert report["quadMask"]["shape"] == [256, 256]
    assert outputs["outputs"]["mask"].endswith("_quad_mask.png")
    assert (tmp_path / "out" / "scene_quad_overlay.png").is_file()
    assert "cleanup" in outputs["outputs"]

    warped_path = tmp_path / "manual-warp.png"
    width, height = warp_document(source, quad, warped_path)
    assert warped_path.is_file()
    assert width > 100 and height > 100


def test_overlay_without_quad_and_office_lens_cli(tmp_path, capsys) -> None:
    source = tmp_path / "scene.png"
    image, _quad = _scene()
    cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    no_quad = tmp_path / "no-quad.png"
    save_overlay(source, None, no_quad)
    assert no_quad.is_file()
    with pytest.raises(ValueError, match="Could not read"):
        save_overlay(tmp_path / "missing.png", None, tmp_path / "missing-out.png")

    out_dir = tmp_path / "cli-out"
    assert office_lens_main([str(source), "--out", str(out_dir), "--mode", "photo"]) == 0
    report_path = out_dir / "scene_onnx_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["mode"] == "photo"
    assert "Saved:" in capsys.readouterr().out
