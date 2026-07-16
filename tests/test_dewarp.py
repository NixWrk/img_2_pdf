from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core.dewarp import (
    DEWARP_METHOD_AUTO,
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
    DEWARP_METHOD_TEXTLINE,
    DewarpDiagnostics,
    DewarpModel,
    apply_dewarp_model,
    dewarp_document,
    estimate_textline_dewarp_model,
    measure_dewarp_quality,
    normalize_control_curves,
    normalize_control_points,
)
from uniscan.core.scanner_adapter import ScanOutput


def _curved_text_page(*, amplitude: float = 18.0) -> tuple[np.ndarray, list[int]]:
    height, width = 700, 900
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    character_x = list(range(60, 840, 22))
    for baseline in range(90, 620, 65):
        for x in character_x:
            normalized_x = (2.0 * x / (width - 1)) - 1.0
            displacement = int(round(amplitude * ((normalized_x**2) - 0.35)))
            cv2.rectangle(
                image,
                (x, baseline + displacement),
                (x + 12, baseline + displacement + 11),
                (0, 0, 0),
                -1,
            )
    return image, character_x


def _first_line_residual_rms(image: np.ndarray, character_x: list[int]) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centers: list[float] = []
    for x in character_x:
        ys = np.flatnonzero(np.min(gray[:150, x : x + 13], axis=1) < 128)
        assert ys.size
        centers.append(float(np.median(ys)))
    xs = np.asarray(character_x, dtype=np.float64)
    center_arr = np.asarray(centers, dtype=np.float64)
    residual = center_arr - np.polyval(np.polyfit(xs, center_arr, 1), xs)
    return float(np.sqrt(np.mean(np.square(residual))))


def test_textline_dewarp_reduces_synthetic_page_curvature() -> None:
    image, character_x = _curved_text_page()

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_TEXTLINE)

    assert corrected.shape == image.shape
    assert diagnostics.applied is True
    assert diagnostics.line_count >= 3
    assert diagnostics.max_displacement_px > 2.0
    assert diagnostics.selected_method == DEWARP_METHOD_TEXTLINE
    assert diagnostics.curvature_after_px < diagnostics.curvature_before_px
    assert diagnostics.duration_ms > 0.0
    assert _first_line_residual_rms(corrected, character_x) < (
        _first_line_residual_rms(image, character_x) * 0.35
    )


def test_automatic_model_can_be_replayed_and_user_adjusted() -> None:
    image, _character_x = _curved_text_page()
    model, diagnostics = estimate_textline_dewarp_model(image)

    assert model is not None
    assert diagnostics.applied is True
    replayed = apply_dewarp_model(image, model)
    direct, _direct_diagnostics = dewarp_document(image, method=DEWARP_METHOD_TEXTLINE)
    np.testing.assert_array_equal(replayed, direct)

    adjusted_points = list(model.control_points)
    middle_x, middle_y = adjusted_points[len(adjusted_points) // 2]
    adjusted_points[len(adjusted_points) // 2] = (middle_x, middle_y + 0.01)
    adjusted = DewarpModel(
        method=DEWARP_METHOD_TEXTLINE,
        control_points=tuple(adjusted_points),
        source="user",
    )
    corrected, adjusted_diagnostics = dewarp_document(
        image,
        method=DEWARP_METHOD_TEXTLINE,
        model=adjusted,
    )

    assert adjusted_diagnostics.reason == "user_adjusted_model"
    assert not np.array_equal(corrected, replayed)


def test_three_dewarp_curves_vary_correction_over_page_height() -> None:
    height, width = 200, 120
    image = np.broadcast_to(np.arange(height, dtype=np.uint8)[:, None], (height, width)).copy()
    flat = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))
    upward = ((0.0, 0.05), (0.5, 0.05), (1.0, 0.05))
    downward = ((0.0, -0.05), (0.5, -0.05), (1.0, -0.05))
    model = DewarpModel(
        method=DEWARP_METHOD_TEXTLINE,
        control_points=flat,
        source="user",
        control_curves=((0.1, upward), (0.5, flat), (0.9, downward)),
    )

    corrected = apply_dewarp_model(image, model)

    assert int(corrected[20, width // 2]) == pytest.approx(30, abs=1)
    assert int(corrected[100, width // 2]) == pytest.approx(100, abs=1)
    assert int(corrected[180, width // 2]) == pytest.approx(170, abs=1)


def test_identical_multi_curve_model_matches_legacy_single_curve() -> None:
    image, _character_x = _curved_text_page()
    points = ((0.0, 0.0), (0.5, 0.03), (1.0, 0.0))
    legacy = DewarpModel(method=DEWARP_METHOD_TEXTLINE, control_points=points)
    multi = DewarpModel(
        method=DEWARP_METHOD_TEXTLINE,
        control_points=points,
        control_curves=((0.2, points), (0.5, points), (0.8, points)),
    )

    np.testing.assert_array_equal(
        apply_dewarp_model(image, multi),
        apply_dewarp_model(image, legacy),
    )


def test_control_curves_validate_and_sort_anchors() -> None:
    points = [(0.0, 0.0), (0.5, 0.01), (1.0, 0.0)]
    curves = normalize_control_curves([(0.8, points), (0.2, points), (0.5, points)])

    assert [curve[0] for curve in curves] == [0.2, 0.5, 0.8]
    with pytest.raises(ValueError, match="unique"):
        normalize_control_curves([(0.5, points), (0.5, points)])


def test_textline_dewarp_keeps_straight_page_unchanged() -> None:
    image, _character_x = _curved_text_page(amplitude=0.0)

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_TEXTLINE)

    assert diagnostics.applied is False
    assert diagnostics.reason == "curvature_below_threshold"
    np.testing.assert_array_equal(corrected, image)


def test_auto_dewarp_selects_validated_textline_candidate() -> None:
    image, character_x = _curved_text_page()

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_AUTO)

    assert diagnostics.applied is True
    assert diagnostics.selected_method == DEWARP_METHOD_TEXTLINE
    assert diagnostics.curvature_after_px < diagnostics.curvature_before_px * 0.5
    assert diagnostics.blank_border_after <= diagnostics.blank_border_before + 0.01
    assert diagnostics.aspect_change == 0.0
    assert _first_line_residual_rms(corrected, character_x) < 2.0


def test_auto_dewarp_keeps_straight_page_and_does_not_start_uvdoc(monkeypatch) -> None:
    image, _character_x = _curved_text_page(amplitude=0.0)
    uvdoc_calls = 0

    def unexpected_uvdoc(*_args, **_kwargs):
        nonlocal uvdoc_calls
        uvdoc_calls += 1
        raise AssertionError("UVDoc must be explicitly enabled")

    monkeypatch.setattr("uniscan.core.dewarp._uvdoc_dewarp", unexpected_uvdoc)
    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_AUTO)

    assert diagnostics.applied is False
    assert diagnostics.reason == "curvature_below_threshold"
    assert uvdoc_calls == 0
    np.testing.assert_array_equal(corrected, image)


def test_auto_dewarp_rejects_candidate_without_curvature_improvement(monkeypatch) -> None:
    image, _character_x = _curved_text_page()

    monkeypatch.setattr(
        "uniscan.core.dewarp._textline_dewarp",
        lambda source, *, model: (
            source.copy(),
            DewarpDiagnostics(
                method=DEWARP_METHOD_TEXTLINE,
                applied=True,
                line_count=8,
                max_displacement_px=12.0,
            ),
        ),
    )

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_AUTO)

    assert diagnostics.applied is False
    assert diagnostics.selected_method == DEWARP_METHOD_NONE
    assert diagnostics.reason == "textline_rejected:curvature_not_improved"
    np.testing.assert_array_equal(corrected, image)


def test_auto_dewarp_can_use_explicit_uvdoc_fallback(monkeypatch) -> None:
    image = np.full((200, 160, 3), 230, dtype=np.uint8)
    expected = image.copy()
    expected[40:160, 50:110] = 220

    monkeypatch.setattr(
        "uniscan.core.dewarp._uvdoc_dewarp",
        lambda _image, *, cache_home: (
            expected,
            DewarpDiagnostics(
                method=DEWARP_METHOD_PADDLEOCR_UVDOC,
                applied=True,
            ),
        ),
    )

    corrected, diagnostics = dewarp_document(
        image,
        method=DEWARP_METHOD_AUTO,
        auto_use_uvdoc=True,
    )

    np.testing.assert_array_equal(corrected, expected)
    assert diagnostics.applied is True
    assert diagnostics.selected_method == DEWARP_METHOD_PADDLEOCR_UVDOC
    assert diagnostics.reason == "textline_fallback:insufficient_text_lines"


def test_measure_dewarp_quality_reports_curvature() -> None:
    curved, _character_x = _curved_text_page()
    straight, _character_x = _curved_text_page(amplitude=0.0)

    curved_metrics = measure_dewarp_quality(curved)
    straight_metrics = measure_dewarp_quality(straight)

    assert curved_metrics.line_count >= 3
    assert curved_metrics.curvature_rms_px > straight_metrics.curvature_rms_px + 1.0


def test_dewarp_none_and_invalid_method() -> None:
    image = np.full((100, 120, 3), 200, dtype=np.uint8)
    unchanged, diagnostics = dewarp_document(image, method=DEWARP_METHOD_NONE)

    assert diagnostics.reason == "disabled"
    np.testing.assert_array_equal(unchanged, image)
    with pytest.raises(ValueError, match="Unsupported dewarp method"):
        dewarp_document(image, method="missing")
    with pytest.raises(ValueError, match="3..32"):
        normalize_control_points([(0.0, 0.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="unique"):
        normalize_control_points([(0.0, 0.0), (0.0, 0.1), (1.0, 0.0)])


def test_uvdoc_is_available_as_independent_dewarp_stage(monkeypatch) -> None:
    image = np.full((100, 120, 3), 200, dtype=np.uint8)
    expected = np.full((90, 110, 3), 180, dtype=np.uint8)

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._uvdoc_document_detector",
        lambda _image, *, cache_home: ScanOutput(
            warped=expected,
            contour=None,
            backend=DEWARP_METHOD_PADDLEOCR_UVDOC,
            detected=True,
            raw_result={"cache": cache_home},
        ),
    )

    corrected, diagnostics = dewarp_document(
        image,
        method=DEWARP_METHOD_PADDLEOCR_UVDOC,
    )

    np.testing.assert_array_equal(corrected, expected)
    assert diagnostics.applied is True
    assert diagnostics.method == DEWARP_METHOD_PADDLEOCR_UVDOC
