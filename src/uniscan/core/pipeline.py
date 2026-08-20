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

from uniscan.core.boundary_review import assess_boundary_review
from uniscan.core.geometry import (
    BackwardMap,
    compose_backward_maps,
    identity_backward_map,
    perspective_backward_map,
    slice_backward_map,
)
from uniscan.core.layout import detect_content_box
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.orientation import rotate_right_angle
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
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
    detect_document: bool = False
    two_page_mode: bool = False
    postprocess_name: str = "None"
    detector_backends: tuple[str, ...] | None = None
    strict_detect: bool = False
    pre_split_rotation_degrees: int = 0
    rectify_split_pages: bool = True
    detect_proposal_only: bool = False


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
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()
    boundary_dark_border_fraction: float = 0.0
    geometry_source: np.ndarray | None = None
    geometry_map: BackwardMap | None = None
    geometry_was_resampled: bool = False


def _initial_geometry_map(
    source: np.ndarray,
    warped: np.ndarray,
    contour: np.ndarray | None,
    *,
    backend: str | None,
    detected: bool,
) -> tuple[np.ndarray, BackwardMap, bool]:
    """Recover a detector homography, or make already-rendered pixels authoritative."""
    if contour is not None and backend not in {
        DETECTOR_BACKEND_UVDOC,
        DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    }:
        try:
            backward_map = perspective_backward_map(source, contour)
            if backward_map.output_size == (warped.shape[1], warped.shape[0]):
                return source, backward_map, True
        except ValueError:
            pass
    if not detected and source.shape == warped.shape:
        return source, identity_backward_map((source.shape[1], source.shape[0])), False
    return warped, identity_backward_map((warped.shape[1], warped.shape[0])), True


def _boundary_review_fields(
    image: np.ndarray,
    *,
    options: PipelineOptions,
    detected: bool,
) -> dict[str, object]:
    diagnostics = assess_boundary_review(
        image,
        detection_enabled=options.detect_document,
        detected=detected,
        proposal_only=options.detect_proposal_only,
    )
    return {
        "needs_review": diagnostics.needs_review,
        "review_reasons": diagnostics.reasons,
        "boundary_dark_border_fraction": diagnostics.dark_border_fraction,
    }


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


def _is_trusted_page_rectification(source: np.ndarray, scan_output) -> bool:
    """Reject internal tables and strips found during split-page rectification."""
    if not scan_output.detected or scan_output.warped is None or scan_output.contour is None:
        return False
    source_height, source_width = source.shape[:2]
    warped_height, warped_width = scan_output.warped.shape[:2]
    contour_area_ratio = abs(float(cv2.contourArea(scan_output.contour))) / max(
        1, source_width * source_height
    )
    x, y, box_width, box_height = cv2.boundingRect(
        np.asarray(scan_output.contour, dtype=np.float32)
    )
    horizontal_coverage = box_width / max(1, source_width)
    top_position = y / max(1, source_height)
    bottom_position = (y + box_height) / max(1, source_height)
    warped_aspect = warped_width / max(1, warped_height)
    return (
        contour_area_ratio >= 0.60
        and horizontal_coverage >= 0.80
        and top_position <= 0.12
        and bottom_position >= 0.95
        and 0.45 <= warped_aspect <= 1.25
    )


def _rectify_split_page(
    image: np.ndarray,
    *,
    detector_backends: tuple[str, ...],
    scanner_root: Path | None,
    uvdoc_cache_home: Path | None,
):
    attempted: set[str] = set()
    for backend in detector_backends:
        # The hybrid detector returns its first available contour before this
        # split-page trust gate sees it. If that quad is an internal table,
        # continue through the hybrid's weaker full-page estimators instead of
        # silently keeping an avoidable unrectified fallback.
        variants = (backend,)
        if backend == DETECTOR_BACKEND_CV_HYBRID:
            variants += (
                DETECTOR_BACKEND_OPENCV_HOUGH,
                DETECTOR_BACKEND_OPENCV_MINRECT,
            )
        for variant in variants:
            if variant in attempted:
                continue
            attempted.add(variant)
            scan_output = scan_with_document_detector(
                image,
                enabled=True,
                backends=(variant,),
                scanner_root=scanner_root,
                uvdoc_cache_home=uvdoc_cache_home,
            )
            if _is_trusted_page_rectification(image, scan_output):
                return scan_output
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
    if options.strict_detect and not options.detect_document:
        raise ValueError("Strict detection requires document detection to be enabled.")
    if options.detect_proposal_only and not options.detect_document:
        raise ValueError("Proposal-only detection requires document detection to be enabled.")
    if options.detect_proposal_only and options.two_page_mode:
        raise ValueError("Proposal-only detection is incompatible with two-page spread mode.")

    postprocess_fn = POSTPROCESSING_OPTIONS[options.postprocess_name]
    pages: list[PageResult] = []

    total = len(loaded_items)
    for index, (name, image) in enumerate(loaded_items, start=1):
        _check_cancelled(cancel_cb)

        fallback_reason = None
        if options.detect_document:
            scan_output = scan_with_document_detector(
                image,
                enabled=True,
                backends=options.detector_backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
                scanner_root=scanner_root,
                uvdoc_cache_home=uvdoc_cache_home,
                proposal_only=options.detect_proposal_only,
            )
            _check_cancelled(cancel_cb)
            if not scan_output.detected:
                fallback_reason = _detection_fallback_reason(scan_output)
                if options.strict_detect:
                    raise RuntimeError(f"Document detection failed for {name}: {fallback_reason}")
            warped = scan_output.warped if scan_output.warped is not None else image
            contour = _augment_overlay_contour(scan_output, image)
            backend = scan_output.backend
            detected = scan_output.detected
        else:
            warped = image
            contour = None
            backend = None
            detected = False
        _check_cancelled(cancel_cb)

        oriented_raw = rotate_right_angle(image, pre_split_rotation)
        oriented_warped = rotate_right_angle(warped, pre_split_rotation)
        if pre_split_rotation:
            contour = None

        raw_height, raw_width = oriented_raw.shape[:2]
        warped_height, warped_width = oriented_warped.shape[:2]
        source_aspect = raw_width / max(1, raw_height)
        warped_aspect = warped_width / max(1, warped_height)
        crop_area_ratio = (warped_width * warped_height) / max(1, raw_width * raw_height)
        landscape_crop_from_portrait = source_aspect < 1.3 <= warped_aspect
        trusted_landscape_crop = landscape_crop_from_portrait and crop_area_ratio >= 0.30
        if options.two_page_mode and landscape_crop_from_portrait and not trusted_landscape_crop:
            detector_policy = options.detector_backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS
            if DETECTOR_BACKEND_CV_HYBRID in detector_policy:
                retry_output = scan_with_document_detector(
                    oriented_raw,
                    enabled=True,
                    backends=(DETECTOR_BACKEND_OPENCV_HOUGH,),
                    scanner_root=scanner_root,
                    uvdoc_cache_home=uvdoc_cache_home,
                )
                retry_warped = (
                    retry_output.warped if retry_output.warped is not None else oriented_raw
                )
                retry_height, retry_width = retry_warped.shape[:2]
                retry_aspect = retry_width / max(1, retry_height)
                retry_area_ratio = (retry_width * retry_height) / max(1, raw_width * raw_height)
                if retry_output.detected and retry_aspect >= 1.3 and retry_area_ratio >= 0.30:
                    oriented_warped = retry_warped
                    contour = retry_output.contour
                    backend = retry_output.backend
                    detected = True
                    fallback_reason = None
                    warped_height, warped_width = retry_height, retry_width
                    warped_aspect = retry_aspect
                    trusted_landscape_crop = True

        if options.two_page_mode and landscape_crop_from_portrait and not trusted_landscape_crop:
            # A table or other internal rectangle can fool boundary detection
            # into turning a portrait source into a wide strip. Keep the full
            # source rather than splitting or publishing a destructive crop.
            oriented_warped = oriented_raw
            contour = None
            detected = False
            fallback_reason = "rejected landscape crop from portrait source"
            if options.strict_detect:
                raise RuntimeError(f"Document detection failed for {name}: {fallback_reason}")
            warped_height, warped_width = oriented_warped.shape[:2]
            warped_aspect = warped_width / max(1, warped_height)

        geometry_source, spread_geometry_map, geometry_was_resampled = _initial_geometry_map(
            oriented_raw,
            oriented_warped,
            contour,
            backend=backend,
            detected=detected,
        )
        whole_geometry_map = spread_geometry_map
        spread_input = oriented_warped
        raw_split_input = oriented_raw
        embedded_landscape_crop = False
        if options.two_page_mode and source_aspect < 1.3 and warped_aspect < 1.3:
            content_box, content_confidence, _content_reason = detect_content_box(oriented_warped)
            content_aspect = content_box.width / max(1, content_box.height)
            content_area_ratio = (content_box.width * content_box.height) / max(
                1, warped_width * warped_height
            )
            if content_confidence >= 0.5 and content_aspect >= 1.3 and content_area_ratio >= 0.30:
                x0 = content_box.x
                y0 = content_box.y
                x1 = x0 + content_box.width
                y1 = y0 + content_box.height
                spread_input = oriented_warped[y0:y1, x0:x1]
                spread_geometry_map = slice_backward_map(
                    spread_geometry_map,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )

                raw_x0 = int(round(x0 * raw_width / max(1, warped_width)))
                raw_y0 = int(round(y0 * raw_height / max(1, warped_height)))
                raw_x1 = int(round(x1 * raw_width / max(1, warped_width)))
                raw_y1 = int(round(y1 * raw_height / max(1, warped_height)))
                raw_split_input = oriented_raw[raw_y0:raw_y1, raw_x0:raw_x1]
                embedded_landscape_crop = True

        if options.two_page_mode:
            if source_aspect < 1.3 and not trusted_landscape_crop and not embedded_landscape_crop:
                spread = SpreadSplitResult(
                    (oriented_warped,),
                    None,
                    "source_aspect_not_spread",
                )
            else:
                spread = split_spread_analyzed(spread_input, fallback="none")
            warped_halves = spread.pages
            _check_cancelled(cancel_cb)
            if len(warped_halves) == 2:
                # Estimate split ratio from the warped result and replay it on raw
                warped_width = spread_input.shape[1]
                left_warped_width = warped_halves[0].shape[1]
                ratio = left_warped_width / max(1, warped_width)
                raw_halves = _safe_split_proportional(raw_split_input, ratio=ratio)
                geometry_halves = (
                    slice_backward_map(
                        spread_geometry_map,
                        x0=0,
                        y0=0,
                        x1=left_warped_width,
                        y1=spread_input.shape[0],
                    ),
                    slice_backward_map(
                        spread_geometry_map,
                        x0=left_warped_width,
                        y0=0,
                        x1=spread_input.shape[1],
                        y1=spread_input.shape[0],
                    ),
                )
                for half_index, (raw_half, warped_half, half_geometry_map) in enumerate(
                    zip(raw_halves, warped_halves, geometry_halves)
                ):
                    half_geometry_source = geometry_source
                    half_geometry_was_resampled = geometry_was_resampled
                    half_backend = backend
                    half_detected = detected
                    half_fallback_reason = fallback_reason
                    if options.detect_document and options.rectify_split_pages:
                        half_scan = _rectify_split_page(
                            warped_half,
                            detector_backends=(
                                options.detector_backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS
                            ),
                            scanner_root=scanner_root,
                            uvdoc_cache_home=uvdoc_cache_home,
                        )
                        if half_scan is not None:
                            try:
                                correction_map = perspective_backward_map(
                                    warped_half,
                                    half_scan.contour,
                                )
                                if correction_map.output_size != (
                                    half_scan.warped.shape[1],
                                    half_scan.warped.shape[0],
                                ):
                                    raise ValueError("split rectification size mismatch")
                                half_geometry_map = compose_backward_maps(
                                    half_geometry_map,
                                    correction_map,
                                )
                                half_geometry_was_resampled = True
                            except ValueError:
                                half_geometry_source = half_scan.warped
                                half_geometry_was_resampled = True
                                half_geometry_map = identity_backward_map(
                                    (
                                        half_scan.warped.shape[1],
                                        half_scan.warped.shape[0],
                                    )
                                )
                            warped_half = half_scan.warped
                            half_backend = half_scan.backend
                            half_detected = True
                            half_fallback_reason = None
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
                            backend=half_backend,
                            detected=half_detected,
                            fallback_reason=half_fallback_reason,
                            spread_detected=True,
                            spread_confidence=spread.candidate.confidence,
                            spread_reason=spread.reason,
                            geometry_source=half_geometry_source,
                            geometry_map=half_geometry_map,
                            geometry_was_resampled=half_geometry_was_resampled,
                            **_boundary_review_fields(
                                warped_half,
                                options=options,
                                detected=half_detected,
                            ),
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
                        geometry_source=geometry_source,
                        geometry_map=whole_geometry_map,
                        geometry_was_resampled=geometry_was_resampled,
                        **_boundary_review_fields(
                            oriented_warped,
                            options=options,
                            detected=detected,
                        ),
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
                    geometry_source=geometry_source,
                    geometry_map=whole_geometry_map,
                    geometry_was_resampled=geometry_was_resampled,
                    **_boundary_review_fields(
                        oriented_warped,
                        options=options,
                        detected=detected,
                    ),
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
