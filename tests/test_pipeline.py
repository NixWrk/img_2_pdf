from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from uniscan.core.pipeline import (
    PipelineOptions,
    build_pdf_from_images,
    process_loaded_items,
    split_spread,
)
from uniscan.core.geometry import render_backward_map
from uniscan.core.scanner_adapter import ScanOutput
from uniscan.io import imwrite_unicode


def _img() -> np.ndarray:
    out = np.zeros((20, 40, 3), dtype=np.uint8)
    out[:, :20] = (10, 20, 30)
    out[:, 20:] = (40, 50, 60)
    return out


def test_split_spread_returns_two_pages() -> None:
    image = _img()
    pages = split_spread(image)

    assert len(pages) == 2
    assert pages[0].shape == (20, 20, 3)
    assert pages[1].shape == (20, 20, 3)


def test_process_loaded_items_without_detector() -> None:
    loaded = [("sample.png", _img())]
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=True,
        postprocess_name="None",
    )
    pages = process_loaded_items(loaded, options=options)
    assert len(pages) == 2


def test_process_loaded_items_returns_page_results_with_raw() -> None:
    loaded = [("sample.png", _img())]
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=False,
        postprocess_name="None",
    )
    pages = process_loaded_items(loaded, options=options)
    assert len(pages) == 1
    page = pages[0]
    assert page.name == "sample.png"
    assert page.raw is not None
    assert page.warped is not None
    assert page.current is not None
    assert page.raw.shape == _img().shape
    assert page.detected is False
    assert page.fallback_reason is None
    assert page.geometry_source is not None
    assert page.geometry_map is not None
    assert page.geometry_was_resampled is False
    np.testing.assert_array_equal(
        render_backward_map(page.geometry_source, page.geometry_map),
        page.warped,
    )


def test_process_loaded_items_checks_cancellation_after_native_detector(monkeypatch) -> None:
    detector_finished = False

    def detector(image, **_kwargs):
        nonlocal detector_finished
        detector_finished = True
        return ScanOutput(
            warped=image,
            contour=None,
            backend="fake",
            detected=True,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        process_loaded_items(
            [("sample.png", _img())],
            options=PipelineOptions(detect_document=True),
            cancel_cb=lambda: detector_finished,
        )


def test_pipeline_options_default_is_non_destructive(monkeypatch) -> None:
    source = _img()
    monkeypatch.setattr(
        "uniscan.core.pipeline.scan_with_document_detector",
        lambda *_args, **_kwargs: pytest.fail("default options must not run detection"),
    )

    pages = process_loaded_items([("source.png", source)], options=PipelineOptions())

    assert len(pages) == 1
    np.testing.assert_array_equal(pages[0].raw, source)
    np.testing.assert_array_equal(pages[0].warped, source)
    np.testing.assert_array_equal(pages[0].current, source)
    assert pages[0].detected is False


def test_proposal_only_pipeline_keeps_source_pixels_with_detected_contour(monkeypatch) -> None:
    source = np.full((700, 900, 3), 35, dtype=np.uint8)
    contour = np.array(
        [[170, 90], [760, 130], [700, 610], [130, 560]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(source, contour, (245, 245, 245))
    cv2.polylines(source, [contour], isClosed=True, color=(15, 15, 15), thickness=8)

    monkeypatch.setattr(
        "uniscan.core.scanner_adapter.warp_perspective_from_points",
        lambda *_args, **_kwargs: pytest.fail("proposal pipeline must not build a warp"),
    )

    pages = process_loaded_items(
        [("capture.png", source)],
        options=PipelineOptions(
            detect_document=True,
            detect_proposal_only=True,
            two_page_mode=False,
        ),
    )

    assert len(pages) == 1
    assert pages[0].contour is not None
    assert pages[0].detected is True
    np.testing.assert_array_equal(pages[0].raw, source)
    np.testing.assert_array_equal(pages[0].warped, source)
    np.testing.assert_array_equal(pages[0].current, source)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (PipelineOptions(detect_proposal_only=True), "requires document detection"),
        (
            PipelineOptions(
                detect_document=True,
                two_page_mode=True,
                detect_proposal_only=True,
            ),
            "incompatible with two-page spread mode",
        ),
    ],
)
def test_pipeline_rejects_invalid_proposal_only_combinations(
    options: PipelineOptions, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        process_loaded_items([("source.png", _img())], options=options)


def test_pipeline_rejects_strict_detection_when_detection_is_disabled() -> None:
    with pytest.raises(ValueError, match="requires document detection"):
        process_loaded_items(
            [("source.png", _img())],
            options=PipelineOptions(strict_detect=True),
        )


def _spread_image() -> np.ndarray:
    width, height = 800, 500
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    image[:, 395:405] = 30  # dark gutter
    return image


def test_process_loaded_items_two_page_mode_splits_at_gutter() -> None:
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=True,
        postprocess_name="None",
    )
    pages = process_loaded_items([("spread.png", _spread_image())], options=options)
    assert len(pages) == 2
    # Left half should be roughly half the width, plus or minus the gutter offset.
    assert 350 < pages[0].warped.shape[1] < 450
    assert pages[0].name.endswith("[L]")
    assert pages[1].name.endswith("[R]")
    assert all(page.spread_detected for page in pages)
    assert all(page.spread_reason == "gutter_detected" for page in pages)


def test_pre_split_rotation_keeps_sideways_single_page_whole() -> None:
    upright = np.full((900, 600, 3), 245, dtype=np.uint8)
    for y in range(80, 840, 55):
        cv2.line(upright, (70, y), (530, y), (45, 45, 45), 5)
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)

    pages = process_loaded_items(
        [("single.jpg", sideways)],
        options=PipelineOptions(
            detect_document=False,
            two_page_mode=True,
            pre_split_rotation_degrees=270,
        ),
    )

    assert len(pages) == 1
    assert pages[0].current.shape == upright.shape
    assert pages[0].spread_detected is False
    assert pages[0].spread_reason == "source_aspect_not_spread"


def test_pre_split_rotation_still_splits_real_spread() -> None:
    spread = _spread_image()
    sideways = cv2.rotate(spread, cv2.ROTATE_90_CLOCKWISE)

    pages = process_loaded_items(
        [("spread.jpg", sideways)],
        options=PipelineOptions(
            detect_document=False,
            two_page_mode=True,
            pre_split_rotation_degrees=270,
        ),
    )

    assert len(pages) == 2
    assert all(page.spread_detected for page in pages)
    assert all(page.spread_confidence >= 0.5 for page in pages)


def test_portrait_source_rejects_false_gutter_from_bad_landscape_crop(monkeypatch) -> None:
    upright = np.full((900, 600, 3), 245, dtype=np.uint8)
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    bad_crop = np.full((900, 100, 3), 235, dtype=np.uint8)
    bad_crop[445:455, :] = 25

    monkeypatch.setattr(
        "uniscan.core.pipeline.scan_with_document_detector",
        lambda _image, **_kwargs: ScanOutput(
            warped=bad_crop,
            contour=None,
            backend="fake",
            detected=True,
            raw_result=None,
        ),
    )

    pages = process_loaded_items(
        [("table.jpg", sideways)],
        options=PipelineOptions(
            detect_document=True,
            two_page_mode=True,
            pre_split_rotation_degrees=270,
        ),
    )

    assert len(pages) == 1
    assert pages[0].spread_detected is False
    assert pages[0].spread_reason == "source_aspect_not_spread"
    assert pages[0].current.shape == upright.shape
    assert pages[0].detected is False
    assert pages[0].fallback_reason == "rejected landscape crop from portrait source"


def test_strict_detection_rejects_bad_landscape_crop_from_portrait(monkeypatch) -> None:
    upright = np.full((900, 600, 3), 245, dtype=np.uint8)
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    bad_crop = np.full((900, 100, 3), 235, dtype=np.uint8)
    monkeypatch.setattr(
        "uniscan.core.pipeline.scan_with_document_detector",
        lambda _image, **_kwargs: ScanOutput(
            warped=bad_crop,
            contour=None,
            backend="fake",
            detected=True,
            raw_result=None,
        ),
    )

    with pytest.raises(RuntimeError, match="rejected landscape crop from portrait source"):
        process_loaded_items(
            [("table.jpg", sideways)],
            options=PipelineOptions(
                detect_document=True,
                strict_detect=True,
                two_page_mode=True,
                pre_split_rotation_degrees=270,
            ),
        )


def test_large_landscape_crop_inside_portrait_canvas_can_split(monkeypatch) -> None:
    upright_source = np.full((900, 600, 3), 245, dtype=np.uint8)
    sideways_source = cv2.rotate(upright_source, cv2.ROTATE_90_CLOCKWISE)
    upright_spread = _spread_image()
    sideways_spread = cv2.rotate(upright_spread, cv2.ROTATE_90_CLOCKWISE)
    monkeypatch.setattr(
        "uniscan.core.pipeline.scan_with_document_detector",
        lambda _image, **_kwargs: ScanOutput(
            warped=sideways_spread,
            contour=None,
            backend="fake",
            detected=True,
            raw_result=None,
        ),
    )

    pages = process_loaded_items(
        [("pdf-page.png", sideways_source)],
        options=PipelineOptions(
            detect_document=True,
            two_page_mode=True,
            pre_split_rotation_degrees=270,
        ),
    )

    assert len(pages) == 2
    assert all(page.spread_detected for page in pages)
    assert all(page.detected for page in pages)


def test_bad_hybrid_landscape_crop_retries_hough_before_embedded_split(monkeypatch) -> None:
    source = np.full((900, 600, 3), 245, dtype=np.uint8)
    spread = _spread_image()
    attempted_backends: list[str] = []

    def detector(_image, *, backends, **_kwargs):
        backend = backends[0]
        attempted_backends.append(backend)
        if backend == "cv_hybrid":
            return ScanOutput(
                warped=np.full((100, 600, 3), 235, dtype=np.uint8),
                contour=None,
                backend=backend,
                detected=True,
                raw_result=None,
            )
        return ScanOutput(
            warped=spread,
            contour=None,
            backend=backend,
            detected=True,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    pages = process_loaded_items(
        [("embedded-spread.png", source)],
        options=PipelineOptions(
            detect_document=True,
            two_page_mode=True,
            detector_backends=("cv_hybrid",),
            rectify_split_pages=False,
        ),
    )

    assert len(pages) == 2
    assert all(page.detected for page in pages)
    assert all(page.backend == "opencv_hough" for page in pages)
    assert attempted_backends == ["cv_hybrid", "opencv_hough"]


def test_embedded_spread_inside_portrait_canvas_can_split() -> None:
    canvas = np.full((900, 600, 3), 255, dtype=np.uint8)
    photo = np.full((360, 600, 3), 35, dtype=np.uint8)
    cv2.rectangle(photo, (20, 15), (294, 345), (235, 235, 235), -1)
    cv2.rectangle(photo, (306, 15), (580, 345), (235, 235, 235), -1)
    for y in range(60, 320, 40):
        cv2.line(photo, (45, y), (270, y), (80, 80, 80), 3)
        cv2.line(photo, (330, y), (555, y), (80, 80, 80), 3)
    canvas[270:630, :] = photo

    pages = process_loaded_items(
        [("embedded-spread.png", canvas)],
        options=PipelineOptions(detect_document=False, two_page_mode=True),
    )

    assert len(pages) == 2
    assert all(page.spread_detected for page in pages)
    assert all(page.current.shape[0] < 400 for page in pages)


def test_embedded_spread_is_checked_after_rejecting_bad_detector_crop(monkeypatch) -> None:
    canvas = np.full((900, 600, 3), 255, dtype=np.uint8)
    photo = np.full((360, 600, 3), 35, dtype=np.uint8)
    cv2.rectangle(photo, (20, 15), (294, 345), (235, 235, 235), -1)
    cv2.rectangle(photo, (306, 15), (580, 345), (235, 235, 235), -1)
    for y in range(60, 320, 40):
        cv2.line(photo, (45, y), (270, y), (80, 80, 80), 3)
        cv2.line(photo, (330, y), (555, y), (80, 80, 80), 3)
    canvas[270:630, :] = photo
    bad_crop = np.full((100, 600, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        "uniscan.core.pipeline.scan_with_document_detector",
        lambda _image, **_kwargs: ScanOutput(
            warped=bad_crop,
            contour=None,
            backend="fake",
            detected=True,
            raw_result=None,
        ),
    )

    pages = process_loaded_items(
        [("embedded-spread.png", canvas)],
        options=PipelineOptions(detect_document=True, two_page_mode=True),
    )

    assert len(pages) == 2
    assert all(page.spread_detected for page in pages)
    assert all(page.detected is False for page in pages)
    assert all(
        page.fallback_reason == "rejected landscape crop from portrait source" for page in pages
    )


def test_split_pages_receive_individual_perspective_rectification(monkeypatch) -> None:
    spread = _spread_image()

    def detector(image, **_kwargs):
        height, width = image.shape[:2]
        contour = np.array(
            [[20, 20], [width - 20, 20], [width - 20, height - 20], [20, height - 20]],
            dtype=np.float32,
        )
        if width > 600:
            warped = image
        else:
            warped = image[20:-20, 20:-20]
        return ScanOutput(
            warped=warped,
            contour=contour,
            backend="fake",
            detected=True,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    pages = process_loaded_items(
        [("spread.png", spread)],
        options=PipelineOptions(detect_document=True, two_page_mode=True),
    )

    assert len(pages) == 2
    assert all(page.warped.shape[0] == spread.shape[0] - 40 for page in pages)
    assert all(page.warped.shape[1] < spread.shape[1] // 2 for page in pages)


def test_split_page_rectification_rejects_internal_column_crop(monkeypatch) -> None:
    spread = _spread_image()

    def detector(image, **_kwargs):
        height, width = image.shape[:2]
        if width > 600:
            return ScanOutput(
                warped=image,
                contour=np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float32,
                ),
                backend="fake",
                detected=True,
                raw_result=None,
            )
        strip_width = max(2, width // 5)
        return ScanOutput(
            warped=image[:, :strip_width],
            contour=np.array(
                [[0, 0], [strip_width, 0], [strip_width, height - 1], [0, height - 1]],
                dtype=np.float32,
            ),
            backend="fake",
            detected=True,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    pages = process_loaded_items(
        [("spread.png", spread)],
        options=PipelineOptions(detect_document=True, two_page_mode=True),
    )

    assert len(pages) == 2
    assert all(page.warped.shape[1] > 350 for page in pages)


def test_split_page_rectification_rejects_crop_that_loses_page_bottom(monkeypatch) -> None:
    spread = _spread_image()

    def detector(image, **_kwargs):
        height, width = image.shape[:2]
        if width > 600:
            contour = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            return ScanOutput(image, contour, "fake", True, None)
        contour = np.array(
            [[10, 0], [width - 10, 0], [width - 10, height * 0.90], [10, height * 0.90]],
            dtype=np.float32,
        )
        return ScanOutput(image[: int(height * 0.90), 10:-10], contour, "fake", True, None)

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    pages = process_loaded_items(
        [("spread.png", spread)],
        options=PipelineOptions(detect_document=True, two_page_mode=True),
    )

    assert len(pages) == 2
    assert all(page.warped.shape[0] == spread.shape[0] for page in pages)


def test_split_page_rectification_respects_requested_detector_policy(monkeypatch) -> None:
    attempted_backends: list[str] = []

    def detector(image, *, backends, **_kwargs):
        attempted_backends.extend(backends)
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    from uniscan.core.pipeline import _rectify_split_page

    result = _rectify_split_page(
        _img(),
        detector_backends=("office_lens_onnx",),
        scanner_root=None,
        uvdoc_cache_home=None,
    )

    assert result is None
    assert attempted_backends == ["office_lens_onnx"]


def test_split_page_rectification_continues_hybrid_after_untrusted_quad(monkeypatch) -> None:
    attempted_backends: list[str] = []

    def detector(image, *, backends, **_kwargs):
        backend = backends[0]
        attempted_backends.append(backend)
        height, width = image.shape[:2]
        if backend == "cv_hybrid":
            strip_width = width // 4
            contour = np.array(
                [[0, 0], [strip_width, 0], [strip_width, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            return ScanOutput(image[:, :strip_width], contour, backend, True, None)
        contour = np.array(
            [[10, 10], [width - 10, 10], [width - 10, height - 1], [10, height - 1]],
            dtype=np.float32,
        )
        return ScanOutput(image, contour, backend, True, None)

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", detector)

    from uniscan.core.pipeline import _rectify_split_page

    result = _rectify_split_page(
        np.full((500, 400, 3), 220, dtype=np.uint8),
        detector_backends=("cv_hybrid",),
        scanner_root=None,
        uvdoc_cache_home=None,
    )

    assert result is not None
    assert result.backend == "opencv_hough"
    assert attempted_backends == ["cv_hybrid", "opencv_hough"]


def test_build_pdf_uses_fixed_dpi_layout_and_atomically_publishes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output.pdf"
    output.write_bytes(b"previous")
    calls: list[dict[str, object]] = []

    def fake_convert(paths: list[str], **kwargs) -> None:
        assert paths == [str(Path("page.png"))]
        calls.append(kwargs)
        kwargs["outputstream"].write(b"%PDF-streamed")

    monkeypatch.setattr("uniscan.core.pipeline.img2pdf.convert", fake_convert)
    monkeypatch.setattr(
        "uniscan.core.pipeline.img2pdf.get_fixed_dpi_layout_fun",
        lambda dpi: ("layout", dpi),
    )

    build_pdf_from_images([Path("page.png")], output, 240)

    assert output.read_bytes() == b"%PDF-streamed"
    assert len(calls) == 1
    assert calls[0]["layout_fun"] == ("layout", (240, 240))
    assert "dpi" not in calls[0]


def test_build_pdf_failure_preserves_existing_target(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output.pdf"
    output.write_bytes(b"previous")

    def fail_convert(*_args, **kwargs) -> None:
        kwargs["outputstream"].write(b"partial")
        raise RuntimeError("forced")

    monkeypatch.setattr("uniscan.core.pipeline.img2pdf.convert", fail_convert)

    try:
        build_pdf_from_images([Path("page.png")], output, 300)
    except RuntimeError as exc:
        assert str(exc) == "forced"
    else:
        raise AssertionError("Expected RuntimeError")

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output.pdf.stage-*"))


def test_build_pdf_jpeg_compression_reduces_photographic_pdf_size(tmp_path) -> None:
    rng = np.random.default_rng(1234)
    image = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    source = tmp_path / "photo.png"
    assert imwrite_unicode(source, image)
    lossless = tmp_path / "lossless.pdf"
    compressed = tmp_path / "compressed.pdf"

    build_pdf_from_images([source], lossless, 300)
    build_pdf_from_images([source], compressed, 300, jpeg_quality=80)

    assert compressed.stat().st_size < lossless.stat().st_size * 0.5
    lossless_doc = pdfium.PdfDocument(lossless)
    compressed_doc = pdfium.PdfDocument(compressed)
    try:
        assert len(lossless_doc) == len(compressed_doc) == 1
        assert compressed_doc[0].get_size() == pytest.approx(lossless_doc[0].get_size())
    finally:
        lossless_doc.close()
        compressed_doc.close()


def test_build_pdf_rejects_invalid_jpeg_quality(tmp_path) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        build_pdf_from_images([Path("page.png")], tmp_path / "invalid.pdf", 300, jpeg_quality=0)
