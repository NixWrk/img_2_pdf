"""Processing pipeline for document pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import img2pdf
import numpy as np

from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    _find_quad_contour,
    scan_with_document_detector,
)
from uniscan.core.spread import split_spread_accurate
from uniscan.io.loaders import imwrite_unicode

LoadedItem = tuple[str, np.ndarray]
ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


@dataclass(slots=True)
class PipelineOptions:
    detect_document: bool = True
    two_page_mode: bool = False
    postprocess_name: str = "None"
    detector_backends: tuple[str, ...] | None = None
    strict_detect: bool = False


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

    postprocess_fn = POSTPROCESSING_OPTIONS[options.postprocess_name]
    pages: list[PageResult] = []

    total = len(loaded_items)
    for index, (name, image) in enumerate(loaded_items, start=1):
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Cancelled by user.")

        scan_output = scan_with_document_detector(
            image,
            enabled=options.detect_document,
            backends=options.detector_backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
            scanner_root=scanner_root,
            uvdoc_cache_home=uvdoc_cache_home,
        )
        fallback_reason = None
        if options.detect_document and not scan_output.detected:
            fallback_reason = _detection_fallback_reason(scan_output)
            if options.strict_detect:
                raise RuntimeError(f"Document detection failed for {name}: {fallback_reason}")
        warped = scan_output.warped if scan_output.warped is not None else image
        contour = _augment_overlay_contour(scan_output, image)
        backend = scan_output.backend

        if options.two_page_mode:
            warped_halves = split_spread_accurate(warped, fallback="midpoint")
            if len(warped_halves) == 2:
                # Estimate split ratio from the warped result and replay it on raw
                warped_width = warped.shape[1]
                left_warped_width = warped_halves[0].shape[1]
                ratio = left_warped_width / max(1, warped_width)
                raw_halves = _safe_split_proportional(image, ratio=ratio)
                for half_index, (raw_half, warped_half) in enumerate(
                    zip(raw_halves, warped_halves)
                ):
                    current_half = postprocess_fn(warped_half)
                    suffix = "L" if half_index == 0 else "R"
                    pages.append(
                        PageResult(
                            name=f"{name} [{suffix}]",
                            raw=raw_half,
                            warped=warped_half,
                            current=current_half,
                            contour=None,
                            backend=backend,
                            detected=scan_output.detected,
                            fallback_reason=fallback_reason,
                        )
                    )
            else:
                current = postprocess_fn(warped)
                pages.append(
                    PageResult(
                        name=name,
                        raw=image,
                        warped=warped,
                        current=current,
                        contour=contour,
                        backend=backend,
                        detected=scan_output.detected,
                        fallback_reason=fallback_reason,
                    )
                )
        else:
            current = postprocess_fn(warped)
            pages.append(
                PageResult(
                    name=name,
                    raw=image,
                    warped=warped,
                    current=current,
                    contour=contour,
                    backend=backend,
                    detected=scan_output.detected,
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


def build_pdf_from_images(image_paths: list[Path], out_pdf: Path, dpi: int) -> None:
    """Build a merged PDF directly into its output stream."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as file:
        try:
            img2pdf.convert(
                [str(p) for p in image_paths],
                dpi=dpi,
                outputstream=file,
            )
        except TypeError:
            file.seek(0)
            file.truncate()
            layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
            img2pdf.convert(
                [str(p) for p in image_paths],
                layout_fun=layout,
                outputstream=file,
            )
