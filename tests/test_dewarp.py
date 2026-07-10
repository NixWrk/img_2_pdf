from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core.dewarp import (
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
    DEWARP_METHOD_TEXTLINE,
    dewarp_document,
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
    assert _first_line_residual_rms(corrected, character_x) < (
        _first_line_residual_rms(image, character_x) * 0.35
    )


def test_textline_dewarp_keeps_straight_page_unchanged() -> None:
    image, _character_x = _curved_text_page(amplitude=0.0)

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_TEXTLINE)

    assert diagnostics.applied is False
    assert diagnostics.reason == "curvature_below_threshold"
    np.testing.assert_array_equal(corrected, image)


def test_dewarp_none_and_invalid_method() -> None:
    image = np.full((100, 120, 3), 200, dtype=np.uint8)
    unchanged, diagnostics = dewarp_document(image, method=DEWARP_METHOD_NONE)

    assert diagnostics.reason == "disabled"
    np.testing.assert_array_equal(unchanged, image)
    with pytest.raises(ValueError, match="Unsupported dewarp method"):
        dewarp_document(image, method="missing")


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
