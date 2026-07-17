from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from uniscan.office_lens.adapter import (
    CLASSIFIER_MODEL,
    OfficeLensOnnx,
    QUAD_MODEL,
    _expand_quad,
    _normalize_luminance,
    _read_rgb,
    _quad_area,
    _quad_image_score,
    _quad_iou,
    _write_image,
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


class _TensorInfo:
    name = "input"


class _FakeSession:
    def __init__(self, path: str, providers) -> None:
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [_TensorInfo()]

    def get_outputs(self):
        return [_TensorInfo()]

    def run(self, _outputs, _inputs):
        if self.path.endswith(QUAD_MODEL.name):
            mask = np.zeros((1, 256, 256, 1), dtype=np.float32)
            cv2.rectangle(mask[0, :, :, 0], (25, 25), (230, 230), 0.9, -1)
            return [mask]
        return [np.array([[0.8, 0.1, 0.1]], dtype=np.float32)]


class _FakeRuntime:
    InferenceSession = _FakeSession


def _configure_fake_byom(tmp_path, monkeypatch) -> None:
    (tmp_path / QUAD_MODEL.name).write_bytes(b"user supplied quad")
    (tmp_path / CLASSIFIER_MODEL.name).write_bytes(b"user supplied classifier")
    monkeypatch.setenv("UNISCAN_OFFICE_LENS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        "uniscan.office_lens.adapter._load_onnxruntime",
        lambda: _FakeRuntime(),
    )


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

    disjoint = np.float32([[5, 5], [80, 5], [80, 80], [5, 80]])
    same_area_elsewhere = disjoint + np.float32([120, 90])
    np.testing.assert_array_equal(
        choose_quad(disjoint, same_area_elsewhere, 240, 180),
        disjoint,
    )


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


def test_office_lens_identity_warp_preserves_inclusive_size() -> None:
    image = np.arange(37 * 53 * 3, dtype=np.uint8).reshape(37, 53, 3)
    quad = np.array(
        [[0, 0], [52, 0], [52, 36], [0, 36]],
        dtype=np.float32,
    )

    warped, size = warp_document_rgb(image, quad)

    assert size == (53, 37)
    assert warped.shape == image.shape
    np.testing.assert_array_equal(warped, image)


@pytest.mark.parametrize("level", [0, 127, 255])
def test_luminance_normalization_preserves_constant_pages(level: int) -> None:
    page = np.full((32, 48), level, dtype=np.uint8)

    normalized = _normalize_luminance(page)

    assert normalized.dtype == np.uint8
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, page)


@pytest.mark.parametrize("delta", [1, 2, 3, 4, 8])
def test_luminance_normalization_preserves_nearly_flat_page(delta: int) -> None:
    page = np.full((32, 48), 255 - delta, dtype=np.uint8)
    page[::2, ::2] = 255

    normalized = _normalize_luminance(page)

    np.testing.assert_array_equal(normalized, page)


@pytest.mark.parametrize("base", [0, 16, 64, 127, 246])
def test_luminance_normalization_bounds_near_flat_contrast_gain(base: int) -> None:
    delta = 9
    page = np.full((64, 64), base, dtype=np.uint8)
    page[::2, ::2] = base + delta

    normalized = _normalize_luminance(page)

    assert int(np.ptp(normalized)) <= delta * 4 + 1
    assert float(normalized.std()) <= float(page.std()) * 4.1
    assert abs(float(normalized.mean()) - float(page.mean())) <= 5.0


@pytest.mark.parametrize("outlier_count", [90, 100, 110])
@pytest.mark.parametrize("base", [16, 64, 127, 200, 252])
def test_luminance_normalization_ignores_sparse_extreme_outliers(
    outlier_count: int, base: int
) -> None:
    page = np.full((100, 100), base, dtype=np.uint8)
    pixels = page.reshape(-1)
    pixels[:outlier_count] = 0
    pixels[outlier_count : outlier_count * 2] = 255

    normalized = _normalize_luminance(page)

    np.testing.assert_array_equal(normalized, page)


@pytest.mark.parametrize("base", [64, 127, 200])
def test_luminance_normalization_is_smooth_at_dominant_mass_boundary(base: int) -> None:
    means: list[float] = []
    for dominant_count in (9_499, 9_500, 9_501):
        page = np.full((100, 100), base, dtype=np.uint8)
        pixels = page.reshape(-1)
        outlier_count = pixels.size - dominant_count
        low_count = outlier_count // 2
        pixels[:low_count] = 0
        pixels[low_count:outlier_count] = 255

        normalized = _normalize_luminance(page)
        means.append(float(normalized.mean()))
        assert abs(float(normalized.mean()) - float(page.mean())) <= 1.0

    assert abs(means[0] - means[1]) <= 1.0
    assert abs(means[1] - means[2]) <= 1.0


def test_luminance_normalization_is_smooth_across_adjacent_noise_spans() -> None:
    shifts: list[float] = []
    for level_count in (17, 18):
        levels = np.arange(64, 64 + level_count, dtype=np.uint8)
        page = np.tile(levels, (100, 1))

        normalized = _normalize_luminance(page)
        shifts.append(float(normalized.mean()) - float(page.mean()))

    assert max(abs(shift) for shift in shifts) <= 10.0
    assert abs(shifts[0] - shifts[1]) <= 3.0


@pytest.mark.parametrize("outlier_fraction", [0.0, 0.05])
def test_luminance_normalization_is_smooth_across_bimodal_distances(
    outlier_fraction: float,
) -> None:
    means: list[float] = []
    shifts: list[float] = []
    for second_level in (79, 80, 81, 82):
        pixels = np.empty(10_000, dtype=np.uint8)
        outlier_count = int(pixels.size * outlier_fraction)
        core_count = pixels.size - outlier_count
        midpoint = core_count // 2
        pixels[:midpoint] = 64
        pixels[midpoint:core_count] = second_level
        low_outliers = outlier_count // 2
        pixels[core_count : core_count + low_outliers] = 0
        pixels[core_count + low_outliers :] = 255
        page = pixels.reshape(100, 100)

        normalized = _normalize_luminance(page)
        means.append(float(normalized.mean()))
        shifts.append(float(normalized.mean()) - float(page.mean()))

    adjacent_changes = np.abs(np.diff(means))
    assert float(adjacent_changes.max()) <= 4.0
    assert max(abs(shift) for shift in shifts) <= 30.0


def test_luminance_normalization_is_smooth_across_corrected_tail_rank() -> None:
    rng = np.random.default_rng(17)
    core = rng.integers(64, 201, size=(100, 100), dtype=np.uint8)
    input_means: list[float] = []
    output_means: list[float] = []
    for zero_count in range(98, 103):
        page = core.copy()
        page.flat[:zero_count] = 0
        input_means.append(float(page.mean()))
        output_means.append(float(_normalize_luminance(page).mean()))

    assert float(np.abs(np.diff(input_means)).max()) <= 0.02
    assert float(np.abs(np.diff(output_means)).max()) <= 1.0


@pytest.mark.parametrize("base", [64, 127])
def test_luminance_normalization_is_smooth_across_raw_tail_rank(base: int) -> None:
    input_means: list[float] = []
    output_means: list[float] = []
    for zero_count in range(498, 503):
        page = np.full((100, 100), base, dtype=np.uint8)
        pixels = page.reshape(-1)
        pixels[:zero_count] = 0
        pixels[zero_count : zero_count + 100] = 255
        input_means.append(float(page.mean()))
        output_means.append(float(_normalize_luminance(page).mean()))

    assert float(np.abs(np.diff(input_means)).max()) <= 0.02
    assert float(np.abs(np.diff(output_means)).max()) <= 1.0


@pytest.mark.parametrize(
    "page",
    [
        np.empty((0, 8), dtype=np.uint8),
        np.zeros((8, 8), dtype=np.float32),
    ],
    ids=("empty", "float32"),
)
def test_luminance_normalization_rejects_invalid_image(page: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _normalize_luminance(page)


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -float("inf")])
def test_luminance_normalization_rejects_non_finite_strength(strength: float) -> None:
    page = np.tile(np.arange(32, dtype=np.uint8), (32, 1))

    with pytest.raises(ValueError, match="strength must be finite"):
        _normalize_luminance(page, strength=strength)


def test_luminance_normalization_non_flat_result_is_finite_uint8() -> None:
    page = np.tile(np.arange(256, dtype=np.uint8), (32, 1))

    normalized = _normalize_luminance(page)

    assert normalized.shape == page.shape
    assert normalized.dtype == np.uint8
    assert np.isfinite(normalized).all()


@pytest.mark.parametrize("start", [0, 8, 16, 32, 48, 63])
@pytest.mark.parametrize("samples", [257, 513, 1_025])
def test_luminance_normalization_avoids_false_edges_in_dark_ramps(start: int, samples: int) -> None:
    page = np.tile(np.linspace(start, 255, samples, dtype=np.uint8), (101, 1))
    normalized = _normalize_luminance(page)
    row = normalized[normalized.shape[0] // 2]
    steps = np.diff(row.astype(np.int16))

    assert int(steps.min()) >= -3
    assert int(steps.max()) <= 6
    duplicate_spread = max(int(np.ptp(row[page[0] == level])) for level in np.unique(page[0]))
    assert duplicate_spread <= 3


@pytest.mark.parametrize(
    ("start", "stop", "samples"),
    [
        (40, 112, 257),
        (44, 112, 127),
        (44, 224, 1_025),
        (46, 160, 127),
        (52, 128, 513),
        (56, 128, 1_025),
    ],
)
def test_luminance_normalization_preserves_midrange_ramp_order(
    start: int,
    stop: int,
    samples: int,
) -> None:
    page = np.tile(np.linspace(start, stop, samples, dtype=np.uint8), (101, 1))
    row = _normalize_luminance(page)[page.shape[0] // 2]
    steps = np.diff(row.astype(np.int16))

    assert int(steps.min()) >= -3
    assert int(steps.max()) <= 7
    duplicate_spread = max(int(np.ptp(row[page[0] == level])) for level in np.unique(page[0]))
    assert duplicate_spread <= 3


def test_byom_file_flow_reports_and_saves_all_outputs(tmp_path, monkeypatch) -> None:
    _configure_fake_byom(tmp_path, monkeypatch)
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


def test_overlay_without_quad_and_office_lens_cli(tmp_path, capsys, monkeypatch) -> None:
    _configure_fake_byom(tmp_path, monkeypatch)
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


def test_office_lens_direct_io_keeps_common_loader_safety(tmp_path, monkeypatch) -> None:
    source = tmp_path / "oversized.png"
    monkeypatch.setattr(
        "uniscan.office_lens.adapter.imread_unicode",
        lambda _path: (_ for _ in ()).throw(RuntimeError("safe input limit exceeded")),
    )

    with pytest.raises(RuntimeError, match="safe input limit"):
        _read_rgb(source)
    with pytest.raises(RuntimeError, match="safe input limit"):
        save_overlay(source, None, tmp_path / "overlay.png")

    destination = tmp_path / "страница.png"
    written: list[tuple[object, np.ndarray]] = []
    monkeypatch.setattr(
        "uniscan.office_lens.adapter.imwrite_unicode",
        lambda path, image: written.append((path, image.copy())) or True,
    )
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)

    _write_image(destination, rgb)

    assert written[0][0] == destination
    np.testing.assert_array_equal(written[0][1], [[[0, 0, 255]]])
