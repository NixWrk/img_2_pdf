"""Adapter layer for document detection/scanner integration."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from .geometry import order_quad_points, warp_perspective_from_points

DETECTOR_BACKEND_CAMSCAN = "camscan"
DETECTOR_BACKEND_OPENCV = "opencv_quad"
DETECTOR_BACKEND_CV_HYBRID = "cv_hybrid"
DETECTOR_BACKEND_OPENCV_HOUGH = "opencv_hough"
DETECTOR_BACKEND_OPENCV_MINRECT = "opencv_minrect"
DETECTOR_BACKEND_UVDOC = "uvdoc"
DETECTOR_BACKEND_PADDLEOCR_UVDOC = "paddleocr_uvdoc"
DETECTOR_BACKEND_OFFICE_LENS_ONNX = "office_lens_onnx"
DEFAULT_ACTIVE_DOCUMENT_BACKENDS = (DETECTOR_BACKEND_CV_HYBRID,)


class ScanAdapterError(RuntimeError):
    """Raised when scanner backend cannot be loaded or used."""


@dataclass(slots=True)
class ScanOutput:
    """Normalized scanner output."""

    warped: np.ndarray | None
    contour: np.ndarray | None
    backend: str | None
    detected: bool
    raw_result: Any


def _import_scanner_with_optional_root(optional_root: Path | None = None) -> ModuleType:
    if optional_root is None:
        try:
            return importlib.import_module("camscan.scanner")
        except Exception as exc:  # pragma: no cover - import is environment-dependent
            raise ScanAdapterError(
                "Cannot import camscan.scanner. Ensure camscan is installed or vendored."
            ) from exc

    root_str = str(optional_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        return importlib.import_module("camscan.scanner")
    except Exception as exc:  # pragma: no cover - import is environment-dependent
        raise ScanAdapterError(
            f"Cannot import camscan.scanner from optional root: {optional_root}"
        ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_paddlex_cache_home() -> Path:
    return _repo_root() / ".paddlex_cache"


def _configure_uvdoc_environment(cache_home: Path | None = None) -> Path:
    resolved = cache_home or _default_paddlex_cache_home()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(resolved))
    # This check adds startup delay on every run and is not needed for local benchmarking.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return Path(os.environ["PADDLE_PDX_CACHE_HOME"])


@lru_cache(maxsize=1)
def _load_uvdoc_model(cache_home_str: str | None = None) -> Any:
    cache_home = Path(cache_home_str) if cache_home_str else None
    _configure_uvdoc_environment(cache_home)
    try:
        from paddleocr import TextImageUnwarping
    except Exception as exc:  # pragma: no cover - import depends on optional runtime deps
        raise ScanAdapterError(
            "Cannot import PaddleOCR UVDoc. Install paddleocr and paddlepaddle first."
        ) from exc
    return TextImageUnwarping()


@lru_cache(maxsize=1)
def _load_office_lens_model() -> Any:
    try:
        from uniscan.office_lens import OfficeLensOnnx

        return OfficeLensOnnx()
    except Exception as exc:  # pragma: no cover - import depends on optional runtime/models
        raise ScanAdapterError(
            "Cannot initialize Office Lens ONNX. Install the 'office-lens' extra and set "
            "UNISCAN_OFFICE_LENS_MODEL_DIR to licensed model files."
        ) from exc


def _image_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _resize_for_detection(image: np.ndarray, *, max_side: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_side / max(1, height), max_side / max(1, width), 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _white_padding_run(flags: np.ndarray, *, from_start: bool) -> int:
    ordered = flags if from_start else flags[::-1]
    non_padding = np.flatnonzero(~ordered)
    return int(non_padding[0]) if non_padding.size else int(ordered.size)


def _has_photo_transition(gray: np.ndarray, *, axis: int, start: int, from_start: bool) -> bool:
    length = gray.shape[axis]
    probe = max(3, int(round(length * 0.025)))
    if from_start:
        inner_start = start
        inner_stop = min(length, start + probe)
    else:
        inner_start = max(0, length - start - probe)
        inner_stop = length - start
    if inner_stop <= inner_start:
        return False
    region = gray[inner_start:inner_stop, :] if axis == 0 else gray[:, inner_start:inner_stop]
    return bool(float(np.count_nonzero(region < 230)) / max(1, region.size) >= 0.15)


def _detection_roi(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Exclude synthetic white letterbox bands while retaining source coordinates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    top = bottom = left = right = 0
    for _pass in range(2):
        region = gray[top : height - bottom, left : width - right]
        if region.size == 0:
            return image, 0, 0
        row_padding = np.mean(region >= 245, axis=1) >= 0.995
        trim_top = _white_padding_run(row_padding, from_start=True)
        trim_bottom = _white_padding_run(row_padding, from_start=False)
        if trim_top < max(2, int(region.shape[0] * 0.02)) or not _has_photo_transition(
            region, axis=0, start=trim_top, from_start=True
        ):
            trim_top = 0
        if trim_bottom < max(2, int(region.shape[0] * 0.02)) or not _has_photo_transition(
            region, axis=0, start=trim_bottom, from_start=False
        ):
            trim_bottom = 0
        top += trim_top
        bottom += trim_bottom

        region = gray[top : height - bottom, left : width - right]
        if region.size == 0:
            return image, 0, 0
        column_padding = np.mean(region >= 245, axis=0) >= 0.995
        trim_left = _white_padding_run(column_padding, from_start=True)
        trim_right = _white_padding_run(column_padding, from_start=False)
        if trim_left < max(2, int(region.shape[1] * 0.02)) or not _has_photo_transition(
            region, axis=1, start=trim_left, from_start=True
        ):
            trim_left = 0
        if trim_right < max(2, int(region.shape[1] * 0.02)) or not _has_photo_transition(
            region, axis=1, start=trim_right, from_start=False
        ):
            trim_right = 0
        left += trim_left
        right += trim_right

        if not any((trim_top, trim_bottom, trim_left, trim_right)):
            break

    roi = image[top : height - bottom, left : width - right]
    if roi.shape[0] < max(32, int(height * 0.20)) or roi.shape[1] < max(32, int(width * 0.20)):
        return image, 0, 0
    return roi, left, top


def _candidate_maps(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    edges = cv2.Canny(blurred, 60, 180)

    kernel = np.ones((5, 5), dtype=np.uint8)
    closed_maps = []
    for candidate in (adaptive, adaptive_inv, otsu, otsu_inv, edges):
        closed_maps.append(cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2))
    return closed_maps


def _is_low_variance(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.std(gray)) < 5.0


def _contour_score(contour: np.ndarray, image_area: float) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return -1.0
    x, y, width, height = cv2.boundingRect(contour)
    rect_area = float(max(1, width * height))
    fill_ratio = area / rect_area
    coverage = area / max(1.0, image_area)
    return (coverage * 10.0) + fill_ratio


def _is_image_frame(
    contour: np.ndarray,
    image_shape: tuple[int, int],
    gray: np.ndarray | None = None,
) -> bool:
    """Reject raster/canvas borders masquerading as a document quadrilateral."""
    height, width = image_shape
    normalized = order_quad_points(np.asarray(contour, dtype=np.float32).reshape(-1, 2))
    x, y, box_width, box_height = cv2.boundingRect(normalized.astype(np.int32))
    coverage = abs(float(cv2.contourArea(normalized))) / max(1.0, float(width * height))
    touches_every_edge = (
        x <= 1 and y <= 1 and x + box_width >= width - 1 and y + box_height >= height - 1
    )
    if coverage >= 0.9 and touches_every_edge:
        return True

    # Imported scans sometimes contain an axis-aligned white canvas around the
    # photographed page. Its inner edge is a stronger and larger rectangle than
    # the real, perspective-skewed sheet, so area-only ranking picks the canvas.
    if gray is None or coverage < 0.72:
        return False
    near_x = max(2.0, width * 0.06)
    near_y = max(2.0, height * 0.06)
    near_every_edge = (
        float(normalized[:, 0].min()) <= near_x
        and float(normalized[:, 1].min()) <= near_y
        and float(normalized[:, 0].max()) >= (width - 1) - near_x
        and float(normalized[:, 1].max()) >= (height - 1) - near_y
    )
    if not near_every_edge:
        return False

    tl, tr, br, bl = normalized
    axis_tolerance = max(3.0, min(width, height) * 0.025)
    axis_aligned = (
        max(
            abs(float(tl[1] - tr[1])),
            abs(float(bl[1] - br[1])),
            abs(float(tl[0] - bl[0])),
            abs(float(tr[0] - br[0])),
        )
        <= axis_tolerance
    )
    if not axis_aligned:
        return False

    outside_mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.fillConvexPoly(outside_mask, normalized.astype(np.int32), 0)
    outside = np.asarray(gray)[outside_mask > 0]
    if outside.size < max(16, int(width * height * 0.01)):
        return False
    bright_ratio = float(np.count_nonzero(outside >= 235)) / float(outside.size)
    return bool(bright_ratio >= 0.80)


def _find_quad_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = image_area * 0.12
    best_quad: np.ndarray | None = None
    best_score = -1.0
    candidate_maps = _candidate_maps(gray)

    for candidate_map in candidate_maps:
        contours, _hierarchy = cv2.findContours(
            candidate_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            if _is_image_frame(approx, gray.shape[:2], gray):
                continue
            score = _contour_score(approx, image_area)
            if score > best_score:
                best_score = score
                best_quad = approx.reshape(4, 2).astype(np.float32)

    best_quad_coverage = (
        abs(float(cv2.contourArea(best_quad))) / image_area if best_quad is not None else 0.0
    )
    if best_quad is not None and best_quad_coverage >= 0.20:
        return order_quad_points(best_quad)

    best_rect: np.ndarray | None = None
    best_rect_score = -1.0
    for candidate_map in candidate_maps:
        contours, _hierarchy = cv2.findContours(
            candidate_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            rect = cv2.minAreaRect(contour)
            points = order_quad_points(cv2.boxPoints(rect).astype(np.float32))
            if _is_image_frame(points, gray.shape[:2], gray):
                continue
            score = _contour_score(points.reshape(-1, 1, 2), image_area)
            if score > best_rect_score:
                best_rect_score = score
                best_rect = points
    if best_quad is None:
        return best_rect
    if best_rect is None:
        return order_quad_points(best_quad)
    rect_area = abs(float(cv2.contourArea(best_rect)))
    quad_area = abs(float(cv2.contourArea(best_quad)))
    if rect_area >= quad_area * 2.0:
        return best_rect
    return order_quad_points(best_quad)


def _find_minrect_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = image_area * 0.12
    best_rect: np.ndarray | None = None
    best_score = -1.0

    for candidate_map in _candidate_maps(gray):
        contours, _hierarchy = cv2.findContours(
            candidate_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            hull = cv2.convexHull(contour)
            rect = cv2.minAreaRect(hull)
            points = order_quad_points(cv2.boxPoints(rect).astype(np.float32))
            if _is_image_frame(points, gray.shape[:2], gray):
                continue
            score = _contour_score(points.reshape(-1, 1, 2), image_area)
            if score > best_score:
                best_score = score
                best_rect = points
    return best_rect


def _intersection_from_hough_lines(
    line_a: tuple[float, float], line_b: tuple[float, float]
) -> np.ndarray | None:
    rho1, theta1 = line_a
    rho2, theta2 = line_b
    matrix = np.array(
        [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]],
        dtype=np.float32,
    )
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    vector = np.array([[rho1], [rho2]], dtype=np.float32)
    x, y = np.linalg.solve(matrix, vector).flatten()
    return np.array([x, y], dtype=np.float32)


def _line_x_at_y(line: tuple[float, float], y: float) -> float | None:
    rho, theta = line
    cos_theta = float(np.cos(theta))
    if abs(cos_theta) < 1e-6:
        return None
    return float((rho - (y * np.sin(theta))) / cos_theta)


def _line_y_at_x(line: tuple[float, float], x: float) -> float | None:
    rho, theta = line
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-6:
        return None
    return float((rho - (x * np.cos(theta))) / sin_theta)


def _find_hough_quad_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    height, width = gray.shape[:2]
    min_side = min(height, width)

    lines: np.ndarray | None = None
    thresholds = [
        max(60, int(min_side * 0.35)),
        max(45, int(min_side * 0.25)),
        max(30, int(min_side * 0.18)),
    ]
    for threshold in thresholds:
        candidate = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        if candidate is not None and len(candidate) >= 4:
            lines = candidate.reshape(-1, 2)
            break
    if lines is None:
        return None

    vertical: list[tuple[float, float]] = []
    horizontal: list[tuple[float, float]] = []
    for rho, theta in lines:
        angle = float(theta % np.pi)
        if angle < np.pi / 6 or angle > 5 * np.pi / 6:
            vertical.append((float(rho), float(theta)))
        elif np.pi / 3 < angle < 2 * np.pi / 3:
            horizontal.append((float(rho), float(theta)))

    if len(vertical) < 2 or len(horizontal) < 2:
        return None

    center_x = width / 2.0
    center_y = height / 2.0
    vertical_positions = [(line, _line_x_at_y(line, center_y)) for line in vertical]
    horizontal_positions = [(line, _line_y_at_x(line, center_x)) for line in horizontal]
    vertical_positions = [(line, x_pos) for line, x_pos in vertical_positions if x_pos is not None]
    horizontal_positions = [
        (line, y_pos) for line, y_pos in horizontal_positions if y_pos is not None
    ]
    if len(vertical_positions) < 2 or len(horizontal_positions) < 2:
        return None

    left = min(vertical_positions, key=lambda item: item[1])[0]
    right = max(vertical_positions, key=lambda item: item[1])[0]
    top = min(horizontal_positions, key=lambda item: item[1])[0]
    bottom = max(horizontal_positions, key=lambda item: item[1])[0]

    corners = [
        _intersection_from_hough_lines(top, left),
        _intersection_from_hough_lines(top, right),
        _intersection_from_hough_lines(bottom, right),
        _intersection_from_hough_lines(bottom, left),
    ]
    if any(corner is None for corner in corners):
        return None

    contour = order_quad_points(np.vstack(corners).astype(np.float32))
    x_values = contour[:, 0]
    y_values = contour[:, 1]
    if x_values.min() < -0.15 * width or x_values.max() > 1.15 * width:
        return None
    if y_values.min() < -0.15 * height or y_values.max() > 1.15 * height:
        return None
    if float(cv2.contourArea(contour.reshape(-1, 1, 2))) < (height * width * 0.12):
        return None
    if _is_image_frame(contour, gray.shape[:2], gray):
        return None
    return contour


def _select_best_contour(image: np.ndarray, *contours: np.ndarray | None) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_area = float(gray.shape[0] * gray.shape[1])
    best_contour: np.ndarray | None = None
    best_score = -1.0

    for contour in contours:
        if contour is None:
            continue
        normalized = order_quad_points(np.asarray(contour, dtype=np.float32))
        if _is_image_frame(normalized, gray.shape[:2], gray):
            continue
        score = _contour_score(normalized.reshape(-1, 1, 2), image_area)
        if score > best_score:
            best_score = score
            best_contour = normalized
    return best_contour


def _find_hybrid_contour(image: np.ndarray) -> np.ndarray | None:
    """Prefer a real four-edge quad, then fall back to weaker estimators."""
    quad = _find_quad_contour(image)
    if quad is not None:
        return quad
    hough = _find_hough_quad_contour(image)
    if hough is not None:
        return hough
    return _find_minrect_contour(image)


def _contour_detector_output(
    image: np.ndarray,
    *,
    backend: str,
    contour_finder,
) -> ScanOutput:
    resized, scale = _resize_for_detection(image)
    if _is_low_variance(resized):
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result={backend: "low_variance"},
        )

    detection_image, offset_x, offset_y = _detection_roi(resized)
    contour = contour_finder(detection_image)
    if contour is None:
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result={backend: "no_contour"},
        )

    if offset_x or offset_y:
        contour = contour + np.array([offset_x, offset_y], dtype=np.float32)
    if scale != 1.0:
        contour = contour / scale
    contour = order_quad_points(contour.astype(np.float32))
    warped = warp_perspective_from_points(image, contour)
    return ScanOutput(
        warped=warped,
        contour=contour,
        backend=backend,
        detected=True,
        raw_result=None,
    )


def _opencv_document_detector(image: np.ndarray) -> ScanOutput:
    return _contour_detector_output(
        image,
        backend=DETECTOR_BACKEND_OPENCV,
        contour_finder=_find_quad_contour,
    )


def _opencv_minrect_document_detector(image: np.ndarray) -> ScanOutput:
    return _contour_detector_output(
        image,
        backend=DETECTOR_BACKEND_OPENCV_MINRECT,
        contour_finder=_find_minrect_contour,
    )


def _opencv_hough_document_detector(image: np.ndarray) -> ScanOutput:
    return _contour_detector_output(
        image,
        backend=DETECTOR_BACKEND_OPENCV_HOUGH,
        contour_finder=_find_hough_quad_contour,
    )


def _opencv_hybrid_document_detector(image: np.ndarray) -> ScanOutput:
    return _contour_detector_output(
        image,
        backend=DETECTOR_BACKEND_CV_HYBRID,
        contour_finder=_find_hybrid_contour,
    )


def _camscan_document_detector(
    image: np.ndarray, *, scanner_root: Path | None = None
) -> ScanOutput:
    scanner_module = _import_scanner_with_optional_root(optional_root=scanner_root)
    result = scanner_module.main(image)
    warped = getattr(result, "warped", None)
    contour = getattr(result, "contour", None)
    if contour is not None:
        contour_arr = np.array(contour, dtype=np.float32).reshape(-1, 2)
        if contour_arr.shape[0] == 4:
            contour = order_quad_points(contour_arr)
        elif contour_arr.shape[0] > 4:
            rect = cv2.minAreaRect(contour_arr.astype(np.float32))
            contour = order_quad_points(cv2.boxPoints(rect).astype(np.float32))
        else:
            contour = None
    if warped is None and contour is not None:
        warped = warp_perspective_from_points(image, contour)
    if warped is None and contour is None:
        return ScanOutput(
            warped=image, contour=None, backend=None, detected=False, raw_result=result
        )
    return ScanOutput(
        warped=warped,
        contour=contour,
        backend=DETECTOR_BACKEND_CAMSCAN,
        detected=True,
        raw_result=result,
    )


def _office_lens_document_detector(image: np.ndarray) -> ScanOutput:
    runner = _load_office_lens_model()
    # Boundary detection only needs the quad session.  Running classification
    # and enhancement here was expensive and its enhanced output was discarded.
    mask_result = runner.predict_quad_mask(_image_bgr_to_rgb(image))
    quad = mask_result.quad
    if quad is None:
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result=mask_result,
        )

    return ScanOutput(
        warped=warp_perspective_from_points(image, quad),
        contour=quad.astype(np.float32),
        backend=DETECTOR_BACKEND_OFFICE_LENS_ONNX,
        detected=True,
        raw_result=mask_result,
    )


def _uvdoc_document_detector(image: np.ndarray, *, cache_home: Path | None = None) -> ScanOutput:
    model = _load_uvdoc_model(str(cache_home) if cache_home is not None else None)
    result_list = model.predict(image)
    if not result_list:
        return ScanOutput(warped=image, contour=None, backend=None, detected=False, raw_result=None)

    raw_result = result_list[0]
    warped = raw_result.get("doctr_img")
    if warped is None:
        return ScanOutput(
            warped=image, contour=None, backend=None, detected=False, raw_result=raw_result
        )

    warped_arr = np.asarray(warped)
    if warped_arr.size == 0:
        return ScanOutput(
            warped=image, contour=None, backend=None, detected=False, raw_result=raw_result
        )
    if warped_arr.dtype != np.uint8:
        warped_arr = np.clip(warped_arr, 0, 255).astype(np.uint8)
    if warped_arr.ndim == 2:
        warped_arr = cv2.cvtColor(warped_arr, cv2.COLOR_GRAY2BGR)
    elif warped_arr.ndim == 3 and warped_arr.shape[2] == 4:
        warped_arr = cv2.cvtColor(warped_arr, cv2.COLOR_RGBA2BGR)

    return ScanOutput(
        warped=warped_arr,
        contour=None,
        backend=DETECTOR_BACKEND_UVDOC,
        detected=True,
        raw_result=raw_result,
    )


def _paddleocr_uvdoc_document_detector(
    image: np.ndarray, *, cache_home: Path | None = None
) -> ScanOutput:
    result = _uvdoc_document_detector(image, cache_home=cache_home)
    if result.detected:
        result.backend = DETECTOR_BACKEND_PADDLEOCR_UVDOC
    return result


def probe_detector_backend(
    backend: str,
    *,
    scanner_root: Path | None = None,
    uvdoc_cache_home: Path | None = None,
) -> None:
    """Raise ScanAdapterError if the requested backend is unavailable."""
    if backend == DETECTOR_BACKEND_CAMSCAN:
        _import_scanner_with_optional_root(optional_root=scanner_root)
        return
    if backend in (
        DETECTOR_BACKEND_OPENCV,
        DETECTOR_BACKEND_CV_HYBRID,
        DETECTOR_BACKEND_OPENCV_HOUGH,
        DETECTOR_BACKEND_OPENCV_MINRECT,
    ):
        return
    if backend == DETECTOR_BACKEND_OFFICE_LENS_ONNX:
        _load_office_lens_model()
        return
    if backend in (DETECTOR_BACKEND_UVDOC, DETECTOR_BACKEND_PADDLEOCR_UVDOC):
        _load_uvdoc_model(str(uvdoc_cache_home) if uvdoc_cache_home is not None else None)
        return
    raise ScanAdapterError(f"Unsupported detector backend: {backend}")


def scan_with_document_detector(
    image: np.ndarray,
    *,
    enabled: bool = True,
    scanner_root: Path | None = None,
    backends: tuple[str, ...] | None = None,
    uvdoc_cache_home: Path | None = None,
    allow_dewarp_backends: bool = False,
) -> ScanOutput:
    """
    Run document detector and return normalized output.

    If detection is disabled, returns the input image as warped. UVDoc is a
    geometric dewarp model rather than a boundary detector and is skipped by
    default; opt in only for dedicated legacy/benchmark calls.
    """
    if not enabled:
        return ScanOutput(warped=image, contour=None, backend=None, detected=False, raw_result=None)

    selected_backends = backends or DEFAULT_ACTIVE_DOCUMENT_BACKENDS
    errors: list[str] = []

    for backend in selected_backends:
        if backend in (DETECTOR_BACKEND_UVDOC, DETECTOR_BACKEND_PADDLEOCR_UVDOC) and not (
            allow_dewarp_backends
        ):
            errors.append(
                f"{backend}: skipped during boundary detection; use it as a dewarp method"
            )
            continue
        try:
            if backend == DETECTOR_BACKEND_OFFICE_LENS_ONNX:
                result = _office_lens_document_detector(image)
            elif backend == DETECTOR_BACKEND_CAMSCAN:
                result = _camscan_document_detector(image, scanner_root=scanner_root)
            elif backend == DETECTOR_BACKEND_OPENCV:
                result = _opencv_document_detector(image)
            elif backend == DETECTOR_BACKEND_CV_HYBRID:
                result = _opencv_hybrid_document_detector(image)
            elif backend == DETECTOR_BACKEND_OPENCV_HOUGH:
                result = _opencv_hough_document_detector(image)
            elif backend == DETECTOR_BACKEND_OPENCV_MINRECT:
                result = _opencv_minrect_document_detector(image)
            elif backend == DETECTOR_BACKEND_UVDOC:
                result = _uvdoc_document_detector(image, cache_home=uvdoc_cache_home)
            elif backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC:
                result = _paddleocr_uvdoc_document_detector(image, cache_home=uvdoc_cache_home)
            else:
                raise ScanAdapterError(f"Unsupported detector backend: {backend}")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            continue

        if result.detected and result.contour is not None:
            validation_image, validation_scale = _resize_for_detection(image)
            gray = (
                cv2.cvtColor(validation_image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3
                else validation_image
            )
            validation_contour = np.asarray(result.contour, dtype=np.float32) * validation_scale
            if _is_image_frame(validation_contour, gray.shape[:2], gray):
                errors.append(f"{backend}: rejected artificial white canvas frame")
                continue
        if result.detected:
            return result
        if isinstance(result.raw_result, dict):
            reasons = ", ".join(f"{key}={value}" for key, value in result.raw_result.items())
            if reasons:
                errors.append(f"{backend}: {reasons}")

    return ScanOutput(
        warped=image,
        contour=None,
        backend=None,
        detected=False,
        raw_result={"errors": errors},
    )
