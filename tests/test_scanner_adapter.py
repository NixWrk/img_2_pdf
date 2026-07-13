from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_OFFICE_LENS_ONNX,
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


def test_scanner_adapter_defaults_to_redistributable_cv_hybrid(monkeypatch) -> None:
    image = _perspective_doc()
    expected = np.full((300, 240, 3), 195, dtype=np.uint8)
    quad = np.array([[10, 20], [230, 20], [230, 280], [10, 280]], dtype=np.float32)

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._opencv_hybrid_document_detector",
        lambda _image: SimpleNamespace(
            warped=expected,
            contour=quad,
            backend=DETECTOR_BACKEND_CV_HYBRID,
            detected=True,
            raw_result=None,
        ),
    )
    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._load_office_lens_model",
        lambda: (_ for _ in ()).throw(AssertionError("Office Lens must be opt-in")),
    )

    result = scan_with_document_detector(image, enabled=True)

    assert result.backend == DETECTOR_BACKEND_CV_HYBRID
    assert result.detected is True
    assert result.contour is not None
    assert np.array_equal(result.warped, expected)


def test_scanner_adapter_office_lens_detection_does_not_run_classifier(monkeypatch) -> None:
    image = _perspective_doc()
    quad = np.array([[10, 20], [230, 20], [230, 280], [10, 280]], dtype=np.float32)

    class _FakeRunner:
        def predict_quad_mask(self, image_rgb):
            assert image_rgb.shape == image.shape
            return SimpleNamespace(quad=quad)

        def classify(self, _image_rgb):
            raise AssertionError("boundary detection must not run classification")

        def process_image(self, *_args, **_kwargs):
            raise AssertionError("boundary detection must not run enhancement")

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._load_office_lens_model",
        lambda: _FakeRunner(),
    )

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OFFICE_LENS_ONNX,),
    )

    assert result.backend == DETECTOR_BACKEND_OFFICE_LENS_ONNX
    assert result.detected is True
    assert result.warped is not None


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
        allow_dewarp_backends=True,
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
        allow_dewarp_backends=True,
    )

    assert result.backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC
    assert result.detected is True
    assert result.contour is None
    assert np.array_equal(result.warped, expected)


def test_scanner_adapter_skips_uvdoc_during_boundary_detection(monkeypatch) -> None:
    image = _perspective_doc()
    model_calls = 0

    def unexpected_model(*_args, **_kwargs):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("UVDoc must not run as a boundary detector")

    monkeypatch.setattr("uniscan.core.scanner_adapter._load_uvdoc_model", unexpected_model)

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_PADDLEOCR_UVDOC,),
    )

    assert result.detected is False
    assert result.backend is None
    assert np.array_equal(result.warped, image)
    assert model_calls == 0
    assert "use it as a dewarp method" in result.raw_result["errors"][0]
