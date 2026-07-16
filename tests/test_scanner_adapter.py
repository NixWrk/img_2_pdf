from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_OFFICE_LENS_ONNX,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    ScanOutput,
    _detection_roi,
    _is_image_frame,
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


def _document_inside_white_canvas() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((800, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (28, 20), (971, 779), (45, 45, 45), -1)
    document = np.array(
        [[115, 80], [895, 105], [855, 715], [130, 690]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, document, (242, 242, 242))
    cv2.polylines(image, [document], isClosed=True, color=(25, 25, 25), thickness=7)
    for y in range(170, 650, 45):
        cv2.line(image, (210, y), (780, y + 8), (55, 55, 55), 3)
    return image, document.astype(np.float32)


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


def test_proposal_only_detection_does_not_build_perspective_warp(monkeypatch) -> None:
    image = _perspective_doc()

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter.warp_perspective_from_points",
        lambda *_args, **_kwargs: pytest.fail("proposal detection must not build a warp"),
    )

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
        proposal_only=True,
    )

    assert result.backend == DETECTOR_BACKEND_OPENCV
    assert result.detected is True
    assert result.contour is not None
    assert result.warped is image


def test_scanner_adapter_prefers_document_inside_artificial_white_canvas() -> None:
    image, document = _document_inside_white_canvas()

    for backend in (DETECTOR_BACKEND_OPENCV, DETECTOR_BACKEND_CV_HYBRID):
        result = scan_with_document_detector(image, enabled=True, backends=(backend,))

        assert result.detected is True
        assert result.contour is not None
        detected_area = abs(float(cv2.contourArea(result.contour)))
        document_area = abs(float(cv2.contourArea(document)))
        assert 0.85 <= detected_area / document_area <= 1.15
        assert float(result.contour[:, 0].min()) > 80
        assert float(result.contour[:, 1].min()) > 55


def test_scanner_adapter_rejects_white_canvas_frame_from_external_backend(
    monkeypatch,
) -> None:
    image, _document = _document_inside_white_canvas()
    canvas_frame = np.array(
        [[28, 20], [971, 20], [971, 779], [28, 779]],
        dtype=np.float32,
    )
    framed = ScanOutput(
        warped=image[20:780, 28:972],
        contour=canvas_frame,
        backend=DETECTOR_BACKEND_OFFICE_LENS_ONNX,
        detected=True,
        raw_result=None,
    )
    monkeypatch.setattr(
        "uniscan.core.scanner_adapter._office_lens_document_detector",
        lambda _image: framed,
    )

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OFFICE_LENS_ONNX,),
    )

    assert result.detected is False
    assert result.contour is None
    assert "artificial white canvas frame" in result.raw_result["errors"][0]


def test_large_axis_aligned_page_on_dark_surface_is_not_treated_as_canvas() -> None:
    gray = np.full((800, 1000), 35, dtype=np.uint8)
    page = np.array(
        [[30, 25], [969, 25], [969, 774], [30, 774]],
        dtype=np.float32,
    )
    cv2.fillConvexPoly(gray, page.astype(np.int32), 245)

    assert _is_image_frame(page, gray.shape, gray) is False


def test_detector_ignores_large_white_letterbox_bands() -> None:
    image = np.full((842, 595, 3), 255, dtype=np.uint8)
    photo_top = 253
    photo_bottom = 589
    image[photo_top:photo_bottom, :] = 45
    document = np.array(
        [[70, 270], [535, 278], [520, 565], [82, 570]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, document, (242, 242, 242))
    cv2.polylines(image, [document], True, (20, 20, 20), 5)
    for y in range(325, 535, 32):
        cv2.line(image, (135, y), (465, y + 3), (50, 50, 50), 2)

    roi, offset_x, offset_y = _detection_roi(image)
    result = scan_with_document_detector(image)

    assert (offset_x, offset_y) == (0, photo_top)
    assert roi.shape[:2] == (photo_bottom - photo_top, image.shape[1])
    assert result.contour is not None
    detected_area = abs(float(cv2.contourArea(result.contour)))
    expected_area = abs(float(cv2.contourArea(document.astype(np.float32))))
    assert 0.80 <= detected_area / expected_area <= 1.20
    assert float(result.contour[:, 1].min()) >= photo_top
    assert float(result.contour[:, 1].max()) < photo_bottom


def test_detection_roi_rechecks_top_and_bottom_after_removing_side_bands() -> None:
    image = np.full((900, 700, 3), 255, dtype=np.uint8)
    image[:, 90:610] = 255
    image[160:740, 90:610] = 40

    roi, offset_x, offset_y = _detection_roi(image)

    assert (offset_x, offset_y) == (90, 160)
    assert roi.shape[:2] == (580, 520)


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


def test_contour_detector_checks_variance_on_bounded_proxy(monkeypatch) -> None:
    from uniscan.core import scanner_adapter

    image = np.zeros((1800, 2400, 3), dtype=np.uint8)
    checked_shapes: list[tuple[int, ...]] = []

    def low_variance(candidate) -> bool:
        checked_shapes.append(candidate.shape)
        return True

    monkeypatch.setattr(scanner_adapter, "_is_low_variance", low_variance)

    result = scanner_adapter._contour_detector_output(
        image,
        backend=DETECTOR_BACKEND_OPENCV,
        contour_finder=lambda _image: pytest.fail("low-variance input must skip contour search"),
    )

    assert result.detected is False
    assert checked_shapes == [(1200, 1600, 3)]


def test_detector_validates_canvas_frame_on_bounded_proxy(monkeypatch) -> None:
    from uniscan.core import scanner_adapter

    image = np.zeros((1800, 2400, 3), dtype=np.uint8)
    contour = np.array(
        [[0, 0], [2399, 0], [2399, 1799], [0, 1799]],
        dtype=np.float32,
    )
    validation_calls: list[tuple[np.ndarray, tuple[int, int]]] = []

    monkeypatch.setattr(
        scanner_adapter,
        "_opencv_document_detector",
        lambda _image: ScanOutput(
            warped=image,
            contour=contour,
            backend=DETECTOR_BACKEND_OPENCV,
            detected=True,
            raw_result=None,
        ),
    )

    def image_frame(candidate, shape, _gray) -> bool:
        validation_calls.append((candidate.copy(), shape))
        return False

    monkeypatch.setattr(scanner_adapter, "_is_image_frame", image_frame)

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
    )

    assert result.detected is True
    validated_contour, validated_shape = validation_calls[0]
    assert validated_shape == (1200, 1600)
    np.testing.assert_allclose(validated_contour, contour * (2.0 / 3.0))
