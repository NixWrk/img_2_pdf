"""Processing pipeline for document pages."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterator

import cv2
import img2pdf
import numpy as np

from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.orientation import rotate_right_angle
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    _find_quad_contour,
    scan_with_document_detector,
)
from uniscan.core.spread import SpreadSplitResult, split_spread_analyzed
from uniscan.io.loaders import imread_unicode, imwrite_unicode

LoadedItem = tuple[str, np.ndarray]
ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def _check_cancelled(cancel_cb: CancelCb | None) -> None:
    if cancel_cb is not None and cancel_cb():
        raise RuntimeError("Cancelled by user.")


@dataclass(slots=True)
class PipelineOptions:
    detect_document: bool = True
    two_page_mode: bool = False
    postprocess_name: str = "None"
    detector_backends: tuple[str, ...] | None = None
    strict_detect: bool = False
    pre_split_rotation_degrees: int = 0


@dataclass(slots=True)
class PageResult:
    """One page produced by the pipeline."""

    name: str
    raw: np.ndarray
    warped: np.ndarray
    current: np.ndarray
    contour: np.ndarray | None
    backend: str | None
    detected: bool
    fallback_reason: str | None
    spread_detected: bool = False
    spread_confidence: float = 0.0
    spread_reason: str | None = None


def _detection_fallback_reason(scan_output) -> str:
    raw_result = scan_output.raw_result
    if isinstance(raw_result, dict):
        errors = raw_result.get("errors")
        if isinstance(errors, list) and errors:
            return "; ".join(str(error) for error in errors)
    return "no document boundary detected"


def split_spread(image: np.ndarray) -> list[np.ndarray]:
    """Naive 50/50 spread split, kept for backwards compatibility."""
    _, width = image.shape[:2]
    if width < 2:
        return [image]
    midpoint = width // 2
    left = image[:, :midpoint]
    right = image[:, midpoint:]
    if left.size == 0 or right.size == 0:
        return [image]
    return [left, right]


def _safe_split_proportional(image: np.ndarray, *, ratio: float) -> list[np.ndarray]:
    """Split an image at a horizontal ratio (used to keep raw aligned with warped split)."""
    height, width = image.shape[:2]
    if width < 2:
        return [image]
    cut = max(1, min(width - 1, int(round(width * ratio))))
    left = image[:, :cut]
    right = image[:, cut:]
    if left.size == 0 or right.size == 0:
        return [image]
    return [left, right]


def _augment_overlay_contour(scan_output, raw_image: np.ndarray) -> np.ndarray | None:
    """
    When UVDoc rectified the page without producing a contour, run a fast CV
    detector on the raw frame so the UI can still draw a boundary overlay.
    """
    if scan_output.contour is not None:
        return scan_output.contour
    if not scan_output.detected:
        return None
    try:
        return _find_quad_contour(raw_image)
    except Exception:
        return None


def process_loaded_items(
    loaded_items: list[LoadedItem],
    *,
    options: PipelineOptions,
    scanner_root: Path | None = None,
    uvdoc_cache_home: Path | None = None,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> list[PageResult]:
    """Process loaded input items and return PageResult list in order."""
    if options.postprocess_name not in POSTPROCESSING_OPTIONS:
        raise ValueError(f"Unsupported postprocess mode: {options.postprocess_name}")
    pre_split_rotation = int(options.pre_split_rotation_degrees) % 360
    if pre_split_rotation not in {0, 90, 180, 270}:
        raise ValueError("Pre-split rotation must be a multiple of 90 degrees.")

    postprocess_fn = POSTPROCESSING_OPTIONS[options.postprocess_name]
    pages: list[PageResult] = []

    total = len(loaded_items)
    for index, (name, image) in enumerate(loaded_items, start=1):
        _check_cancelled(cancel_cb)

        scan_output = scan_with_document_detector(
            image,
            enabled=options.detect_document,
            backends=options.detector_backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
            scanner_root=scanner_root,
            uvdoc_cache_home=uvdoc_cache_home,
        )
        _check_cancelled(cancel_cb)
        fallback_reason = None
        if options.detect_document and not scan_output.detected:
            fallback_reason = _detection_fallback_reason(scan_output)
            if options.strict_detect:
                raise RuntimeError(f"Document detection failed for {name}: {fallback_reason}")
        warped = scan_output.warped if scan_output.warped is not None else image
        contour = _augment_overlay_contour(scan_output, image)
        _check_cancelled(cancel_cb)
        backend = scan_output.backend

        oriented_raw = rotate_right_angle(image, pre_split_rotation)
        oriented_warped = rotate_right_angle(warped, pre_split_rotation)
        if pre_split_rotation:
            contour = None
        detected = scan_output.detected

        raw_height, raw_width = oriented_raw.shape[:2]
        warped_height, warped_width = oriented_warped.shape[:2]
        source_aspect = raw_width / max(1, raw_height)
        warped_aspect = warped_width / max(1, warped_height)
        if options.two_page_mode and source_aspect < 1.3 <= warped_aspect:
            # A table or other internal rectangle can fool boundary detection
            # into turning a portrait source into a wide strip. Keep the full
            # source rather than splitting or publishing a destructive crop.
            oriented_warped = oriented_raw
            contour = None
            detected = False
            fallback_reason = "rejected landscape crop from portrait source"
            if options.strict_detect:
                raise RuntimeError(f"Document detection failed for {name}: {fallback_reason}")

        if options.two_page_mode:
            if source_aspect < 1.3:
                spread = SpreadSplitResult(
                    (oriented_warped,),
                    None,
                    "source_aspect_not_spread",
                )
            else:
                spread = split_spread_analyzed(oriented_warped, fallback="none")
            warped_halves = spread.pages
            _check_cancelled(cancel_cb)
            if len(warped_halves) == 2:
                # Estimate split ratio from the warped result and replay it on raw
                warped_width = oriented_warped.shape[1]
                left_warped_width = warped_halves[0].shape[1]
                ratio = left_warped_width / max(1, warped_width)
                raw_halves = _safe_split_proportional(oriented_raw, ratio=ratio)
                for half_index, (raw_half, warped_half) in enumerate(
                    zip(raw_halves, warped_halves)
                ):
                    current_half = postprocess_fn(warped_half)
                    _check_cancelled(cancel_cb)
                    suffix = "L" if half_index == 0 else "R"
                    pages.append(
                        PageResult(
                            name=f"{name} [{suffix}]",
                            raw=raw_half,
                            warped=warped_half,
                            current=current_half,
                            contour=None,
                            backend=backend,
                            detected=detected,
                            fallback_reason=fallback_reason,
                            spread_detected=True,
                            spread_confidence=spread.candidate.confidence,
                            spread_reason=spread.reason,
                        )
                    )
            else:
                current = postprocess_fn(oriented_warped)
                _check_cancelled(cancel_cb)
                pages.append(
                    PageResult(
                        name=name,
                        raw=oriented_raw,
                        warped=oriented_warped,
                        current=current,
                        contour=contour,
                        backend=backend,
                        detected=detected,
                        fallback_reason=fallback_reason,
                        spread_reason=spread.reason,
                    )
                )
        else:
            current = postprocess_fn(oriented_warped)
            _check_cancelled(cancel_cb)
            pages.append(
                PageResult(
                    name=name,
                    raw=oriented_raw,
                    warped=oriented_warped,
                    current=current,
                    contour=contour,
                    backend=backend,
                    detected=detected,
                    fallback_reason=fallback_reason,
                )
            )

        if on_progress is not None:
            on_progress(index, total, name)

    return pages


def write_pages_to_dir(
    pages: list[np.ndarray],
    out_dir: Path,
    *,
    start_index: int = 1,
) -> list[Path]:
    """Persist pages as sequential PNG files and return path list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    index = start_index
    for page in pages:
        out_path = out_dir / f"{index:05d}.png"
        if not imwrite_unicode(out_path, page):
            raise RuntimeError(f"Failed writing page: {out_path}")
        output_paths.append(out_path)
        index += 1

    return output_paths


def build_pdf_from_images(
    image_paths: list[Path],
    out_pdf: Path,
    dpi: int,
    *,
    jpeg_quality: int | None = None,
    cancel_cb: CancelCb | None = None,
) -> None:
    """Build a merged PDF at an exact image DPI and publish it atomically.

    When ``jpeg_quality`` is set, each source is converted to an optimized JPEG
    before PDF assembly. This keeps photographic scans compact while preserving
    the requested physical DPI.
    """
    if not image_paths:
        raise ValueError("No image paths to export.")
    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError("PDF DPI must be positive.")
    if jpeg_quality is not None and not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("PDF JPEG quality must be between 1 and 100.")

    out_pdf = Path(out_pdf)
    if out_pdf.exists() and out_pdf.is_dir():
        raise ValueError(f"PDF output path is a directory: {out_pdf}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with _pdf_image_sources(
        image_paths,
        jpeg_quality=jpeg_quality,
        cancel_cb=cancel_cb,
    ) as pdf_image_paths:
        descriptor, raw_staged_path = tempfile.mkstemp(
            prefix=f".{out_pdf.name}.stage-",
            suffix=out_pdf.suffix or ".pdf",
            dir=out_pdf.parent,
        )
        staged_path = Path(raw_staged_path)
        try:
            _check_cancelled(cancel_cb)
            with os.fdopen(descriptor, "wb") as file:
                layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
                img2pdf.convert(
                    [str(p) for p in pdf_image_paths],
                    layout_fun=layout,
                    outputstream=file,
                )
                file.flush()
                os.fsync(file.fileno())
            _check_cancelled(cancel_cb)
            os.replace(staged_path, out_pdf)
        except Exception:
            # ``os.fdopen`` owns the descriptor after it succeeds. If it failed,
            # closing an already-closed descriptor is harmlessly avoided here.
            try:
                os.close(descriptor)
            except OSError:
                pass
            if staged_path.exists():
                staged_path.unlink()
            raise


@contextmanager
def _pdf_image_sources(
    image_paths: list[Path],
    *,
    jpeg_quality: int | None,
    cancel_cb: CancelCb | None,
) -> Iterator[list[Path]]:
    """Yield original paths or temporary JPEG-compressed PDF sources."""
    if jpeg_quality is None:
        yield image_paths
        return

    quality = int(jpeg_quality)
    with tempfile.TemporaryDirectory(prefix="uniscan_pdf_jpeg_") as temporary:
        temporary_dir = Path(temporary)
        compressed_paths: list[Path] = []
        encode_params = (
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        )
        for index, source_path in enumerate(image_paths, start=1):
            _check_cancelled(cancel_cb)
            image = imread_unicode(Path(source_path))
            if image is None:
                raise RuntimeError(f"Cannot read PDF source image: {source_path}")
            compressed_path = temporary_dir / f"{index:05d}.jpg"
            if not imwrite_unicode(compressed_path, image, params=encode_params):
                raise RuntimeError(f"Failed to compress PDF source image: {source_path}")
            compressed_paths.append(compressed_path)
        _check_cancelled(cancel_cb)
        yield compressed_paths
