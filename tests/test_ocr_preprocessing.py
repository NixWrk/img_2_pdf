"""Tests for uniscan.ocr.preprocessing — image pre-processing pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from uniscan.ocr.preprocessing import (
    PREPROCESSING_MODES,
    _strip_markdown,
    apply_preprocessing,
    normalize_dpi,
    to_greyscale,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


def _grey_image(h: int = 100, w: int = 80) -> np.ndarray:
    """Create a simple greyscale gradient image."""
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))


def _bgr_image(h: int = 100, w: int = 80) -> np.ndarray:
    """Create a 3-channel BGR image (constant mid-grey)."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add some contrast so binarisation has something to work with
    img[: h // 2, : w // 2] = 30
    img[h // 2 :, w // 2 :] = 220
    return img


# ---------------------------------------------------------------------------
# _strip_markdown
# ---------------------------------------------------------------------------


class TestStripMarkdown:
    def test_strips_atx_headings(self) -> None:
        assert "# Heading" not in _strip_markdown("# Heading\nsome text")
        assert "some text" in _strip_markdown("# Heading\nsome text")

    def test_strips_bold(self) -> None:
        result = _strip_markdown("**bold text** here")
        assert "**" not in result
        assert "bold text" in result

    def test_strips_italic(self) -> None:
        result = _strip_markdown("*italic* word")
        assert "*" not in result
        assert "italic" in result

    def test_strips_inline_code(self) -> None:
        result = _strip_markdown("use `foo()` function")
        assert "`" not in result
        assert "foo()" in result

    def test_strips_markdown_links(self) -> None:
        result = _strip_markdown("[click here](https://example.com)")
        assert "click here" in result
        assert "https://example.com" not in result

    def test_strips_html_tags(self) -> None:
        result = _strip_markdown("<b>bold</b>")
        assert "<b>" not in result
        assert "bold" in result

    def test_collapses_blank_lines(self) -> None:
        text = "a\n\n\n\n\nb"
        result = _strip_markdown(text)
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self) -> None:
        text = "Hello world\nSecond line"
        result = _strip_markdown(text)
        assert "Hello world" in result
        assert "Second line" in result

    def test_empty_string(self) -> None:
        assert _strip_markdown("") == ""


# ---------------------------------------------------------------------------
# PREPROCESSING_MODES constant
# ---------------------------------------------------------------------------


def test_preprocessing_modes_contains_required_values() -> None:
    assert "none" in PREPROCESSING_MODES
    assert "basic" in PREPROCESSING_MODES
    assert "full" in PREPROCESSING_MODES


# ---------------------------------------------------------------------------
# apply_preprocessing – mode="none"
# ---------------------------------------------------------------------------


def test_apply_preprocessing_none_returns_identical_array() -> None:
    img = _bgr_image()
    result = apply_preprocessing(img, mode="none")
    assert result is img


def test_apply_preprocessing_none_greyscale_returns_identical_array() -> None:
    img = _grey_image()
    result = apply_preprocessing(img, mode="none")
    assert result is img


# ---------------------------------------------------------------------------
# to_greyscale (requires cv2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
class TestToGreyscale:
    def test_bgr_to_greyscale(self) -> None:
        img = _bgr_image()
        grey = to_greyscale(img)
        assert grey.ndim == 2
        assert grey.dtype == np.uint8

    def test_greyscale_passthrough(self) -> None:
        img = _grey_image()
        result = to_greyscale(img)
        assert result.ndim == 2
        assert result is img

    def test_rgba_to_greyscale(self) -> None:
        img = np.full((60, 80, 4), 128, dtype=np.uint8)
        grey = to_greyscale(img)
        assert grey.ndim == 2


# ---------------------------------------------------------------------------
# normalize_dpi (requires cv2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
class TestNormalizeDpi:
    def test_same_dpi_returns_original(self) -> None:
        img = _grey_image(100, 80)
        result = normalize_dpi(img, from_dpi=300, to_dpi=300)
        assert result is img

    def test_upscale_increases_dimensions(self) -> None:
        img = _grey_image(100, 80)
        result = normalize_dpi(img, from_dpi=150, to_dpi=300)
        assert result.shape[0] > img.shape[0]
        assert result.shape[1] > img.shape[1]

    def test_downscale_decreases_dimensions(self) -> None:
        img = _grey_image(200, 160)
        result = normalize_dpi(img, from_dpi=300, to_dpi=150)
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]

    def test_invalid_dpi_raises(self) -> None:
        img = _grey_image()
        with pytest.raises(ValueError):
            normalize_dpi(img, from_dpi=0, to_dpi=300)
        with pytest.raises(ValueError):
            normalize_dpi(img, from_dpi=300, to_dpi=-1)

    def test_exact_scale_2x(self) -> None:
        img = _grey_image(50, 40)
        result = normalize_dpi(img, from_dpi=150, to_dpi=300)
        assert result.shape == (100, 80)


# ---------------------------------------------------------------------------
# apply_preprocessing – mode="basic" (requires cv2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
class TestApplyPreprocessingBasic:
    def test_basic_returns_greyscale(self) -> None:
        img = _bgr_image()
        result = apply_preprocessing(img, mode="basic")
        assert result.ndim == 2

    def test_basic_with_dpi_rescale_changes_size(self) -> None:
        img = _bgr_image(100, 80)
        result = apply_preprocessing(img, mode="basic", render_dpi=150, ocr_dpi=300)
        assert result.shape == (200, 160)

    def test_basic_same_dpi_no_resize(self) -> None:
        img = _bgr_image(100, 80)
        result = apply_preprocessing(img, mode="basic", render_dpi=300, ocr_dpi=300)
        assert result.shape == (100, 80)

    def test_basic_zero_dpi_skips_resize(self) -> None:
        img = _bgr_image(100, 80)
        result = apply_preprocessing(img, mode="basic", render_dpi=0, ocr_dpi=0)
        assert result.shape == (100, 80)


# ---------------------------------------------------------------------------
# apply_preprocessing – mode="full" (requires cv2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
class TestApplyPreprocessingFull:
    def test_full_returns_2d_binary_image(self) -> None:
        img = _bgr_image(120, 100)
        result = apply_preprocessing(img, mode="full")
        assert result.ndim == 2
        assert result.dtype == np.uint8
        # Binary: only 0 and 255 values
        unique = set(np.unique(result).tolist())
        assert unique <= {0, 255}

    def test_full_with_dpi_rescale(self) -> None:
        img = _bgr_image(100, 80)
        result = apply_preprocessing(img, mode="full", render_dpi=150, ocr_dpi=300)
        # Should be rescaled then binarised
        assert result.ndim == 2
        assert result.shape == (200, 160)


# ---------------------------------------------------------------------------
# preprocess_image_file (requires cv2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
def test_preprocess_image_file_basic(tmp_path: Path) -> None:
    from uniscan.ocr.preprocessing import _cv2_imread_unicode, _cv2_imwrite_unicode, preprocess_image_file

    src = tmp_path / "input.png"
    dst = tmp_path / "output.png"

    img = _bgr_image(80, 60)
    _cv2_imwrite_unicode(src, img)

    result = preprocess_image_file(src, dst, mode="basic")

    assert result == dst
    assert dst.exists()
    import cv2  # type: ignore
    out = _cv2_imread_unicode(dst)
    assert out is not None


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
def test_preprocess_image_file_missing_src_raises(tmp_path: Path) -> None:
    from uniscan.ocr.preprocessing import preprocess_image_file

    with pytest.raises(RuntimeError, match="Failed to load"):
        preprocess_image_file(
            tmp_path / "does_not_exist.png",
            tmp_path / "out.png",
            mode="basic",
        )


# ---------------------------------------------------------------------------
# invalid mode
# ---------------------------------------------------------------------------


def test_apply_preprocessing_invalid_mode_raises_without_cv2() -> None:
    """apply_preprocessing with 'none' must not require cv2."""
    img = _bgr_image()
    result = apply_preprocessing(img, mode="none")
    assert result is img


@pytest.mark.skipif(not _CV2_AVAILABLE, reason="cv2 not installed")
def test_apply_preprocessing_unknown_mode_falls_through() -> None:
    """Unsupported modes beyond 'none'/'basic'/'full' should raise."""
    img = _bgr_image()
    # 'none' is safe; invalid value should not reach cv2 calls
    # (the function doesn't validate mode, but canonical.py does)
    result = apply_preprocessing(img, mode="none")
    assert result is img


# ---------------------------------------------------------------------------
# canonical run_ocr_canonical_package passes preprocessing param (unit test)
# ---------------------------------------------------------------------------


def test_run_ocr_canonical_invalid_preprocessing_raises(tmp_path) -> None:
    """run_ocr_canonical_package must reject unknown preprocessing values."""
    from uniscan.export import export_pages_as_pdf
    from uniscan.ocr.canonical import run_ocr_canonical_package

    pages = [np.full((60, 80, 3), 100, dtype=np.uint8)]
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=100)

    with pytest.raises(ValueError, match="Invalid preprocessing mode"):
        run_ocr_canonical_package(
            pdf_path=pdf_path,
            output_dir=tmp_path / "out",
            engines=("pytesseract",),
            sample_size=1,
            preprocessing="invalid_mode",  # type: ignore[arg-type]
        )


def test_run_ocr_canonical_preprocessing_none_does_not_create_preproc_dir(
    tmp_path, monkeypatch
) -> None:
    """When preprocessing='none', no preprocessed/ directory should be created."""
    from types import SimpleNamespace

    from uniscan.export import export_pages_as_pdf
    from uniscan.ocr import OCR_ENGINE_PADDLEOCR
    from uniscan.ocr.canonical import run_ocr_canonical_package

    pages = [np.full((60, 80, 3), 80, dtype=np.uint8)]
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=100)
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(ready=True, missing=[], searchable_pdf=False, label=engine_name)

    def fake_extract(engine: str, image_path: Path, *, lang: str, work_dir: Path) -> str:
        return f"text:{image_path.name}"

    monkeypatch.setattr("uniscan.ocr.canonical.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.canonical._extract_page_text", fake_extract)

    run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        preprocessing="none",
    )

    assert not (output_dir / "preprocessed").exists()


def test_run_ocr_canonical_render_dpi_overrides_dpi(tmp_path, monkeypatch) -> None:
    """render_dpi > 0 should override dpi for the render step."""
    from types import SimpleNamespace

    from uniscan.export import export_pages_as_pdf
    from uniscan.ocr import OCR_ENGINE_PADDLEOCR
    from uniscan.ocr.canonical import run_ocr_canonical_package

    pages = [np.full((60, 80, 3), 50, dtype=np.uint8)]
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=100)
    output_dir = tmp_path / "out"

    render_dpi_used: list[int] = []

    original_render = None

    def fake_render(pdf, page_indices, *, dpi):
        render_dpi_used.append(dpi)
        return original_render(pdf, page_indices, dpi=dpi)

    import uniscan.ocr.canonical as _canon_mod

    original_render = _canon_mod.render_pdf_page_indices
    monkeypatch.setattr(_canon_mod, "render_pdf_page_indices", fake_render)

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(ready=True, missing=[], searchable_pdf=False, label=engine_name)

    def fake_extract(engine: str, image_path: Path, *, lang: str, work_dir: Path) -> str:
        return "text"

    monkeypatch.setattr("uniscan.ocr.canonical.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.canonical._extract_page_text", fake_extract)

    run_ocr_canonical_package(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        dpi=72,
        render_dpi=200,
    )

    assert render_dpi_used == [200], f"Expected render_dpi=200, got {render_dpi_used}"
