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
            options=PipelineOptions(),
            cancel_cb=lambda: detector_finished,
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
    bad_crop = np.full((900, 300, 3), 235, dtype=np.uint8)
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
    bad_crop = np.full((900, 300, 3), 235, dtype=np.uint8)
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
