from __future__ import annotations

import cv2
import numpy as np

from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    scan_with_document_detector,
)


def _perspective_doc() -> np.ndarray:
    image = np.full((700, 900, 3), 35, dtype=np.uint8)
    quad = np.array([[170, 90], [760, 130], [700, 610], [130, 560]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (245, 245, 245))
    cv2.polylines(image, [quad], isClosed=True, color=(15, 15, 15), thickness=8)
    for y in range(170, 540, 40):
        cv2.line(image, (230, y), (640, y), (40, 40, 40), 4)
    return image


def test_scanner_adapter_detects_quad_with_opencv_fallback() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
    )

    assert result.backend == DETECTOR_BACKEND_OPENCV
    assert result.detected is True
    assert result.contour is not None
    assert result.warped is not None
    assert result.warped.shape[0] > 350
    assert result.warped.shape[1] > 350


def test_scanner_adapter_disabled_returns_original() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(image, enabled=False)

    assert result.contour is None
    assert result.backend is None
    assert result.detected is False
    assert np.array_equal(result.warped, image)


def test_scanner_adapter_gracefully_returns_no_contour() -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    result = scan_with_document_detector(
        blank,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
    )

    assert result.contour is None
    assert result.backend is None
    assert result.detected is False
    assert np.array_equal(result.warped, blank)


def test_scanner_adapter_supports_uvdoc_backend_without_contour(monkeypatch) -> None:
    image = _perspective_doc()
    expected = np.full((320, 220, 3), 210, dtype=np.uint8)

    class _FakeModel:
        def predict(self, _input):
            return [{"doctr_img": expected}]

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._load_uvdoc_model",
        lambda _cache_home=None: _FakeModel(),
    )

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_UVDOC,),
    )

    assert result.backend == DETECTOR_BACKEND_UVDOC
    assert result.detected is True
    assert result.contour is None
    assert np.array_equal(result.warped, expected)


def test_scanner_adapter_detects_quad_with_cv_hybrid() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_CV_HYBRID,),
    )

    assert result.backend == DETECTOR_BACKEND_CV_HYBRID
    assert result.detected is True
    assert result.contour is not None
    assert result.warped is not None


def test_scanner_adapter_detects_quad_with_hough_backend() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV_HOUGH,),
    )

    assert result.backend == DETECTOR_BACKEND_OPENCV_HOUGH
    assert result.detected is True
    assert result.contour is not None
    assert result.warped is not None


def test_scanner_adapter_detects_quad_with_minrect_backend() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV_MINRECT,),
    )

    assert result.backend == DETECTOR_BACKEND_OPENCV_MINRECT
    assert result.detected is True
    assert result.contour is not None
    assert result.warped is not None


def test_scanner_adapter_supports_paddleocr_uvdoc_alias(monkeypatch) -> None:
    image = _perspective_doc()
    expected = np.full((280, 210, 3), 180, dtype=np.uint8)

    class _FakeModel:
        def predict(self, _input):
            return [{"doctr_img": expected}]

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._load_uvdoc_model",
        lambda _cache_home=None: _FakeModel(),
    )

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_PADDLEOCR_UVDOC,),
    )

    assert result.backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC
    assert result.detected is True
    assert result.contour is None
    assert np.array_equal(result.warped, expected)
