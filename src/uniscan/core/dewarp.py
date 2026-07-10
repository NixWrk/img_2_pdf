"""Local page-surface correction independent from boundary detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np

DEWARP_METHOD_NONE = "none"
DEWARP_METHOD_TEXTLINE = "textline"
DEWARP_METHOD_PADDLEOCR_UVDOC = "paddleocr_uvdoc"
DEWARP_METHOD_CHOICES = (
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_TEXTLINE,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
)


@dataclass(slots=True, frozen=True)
class DewarpDiagnostics:
    """Explain whether and how local page correction was applied."""

    method: str
    applied: bool
    line_count: int = 0
    max_displacement_px: float = 0.0
    curvature_rms_px: float = 0.0
    reason: str | None = None


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_for_analysis(image: np.ndarray, *, max_width: int = 1400) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(1, width))
    if scale >= 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    min_side = min(gray.shape[:2])
    block_size = max(21, min(61, (min_side // 20) | 1))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        15,
    )


def _robust_baseline(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    if xs.size < 20 or np.unique(xs).size < 12:
        return None
    degree = min(3, np.unique(xs).size - 1)
    coefficients = np.polyfit(xs, ys, degree)
    fitted = np.polyval(coefficients, xs)
    residual = ys - fitted
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    tolerance = max(2.0, 3.5 * 1.4826 * mad)
    keep = np.abs(residual - median) <= tolerance
    if int(keep.sum()) < 16:
        return None
    coefficients = np.polyfit(xs[keep], ys[keep], degree)
    return np.polyval(coefficients, xs)


def _line_curves(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape[:2]
    join_width = max(15, width // 35)
    connected = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, 3)),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    curves: list[np.ndarray] = []
    for contour in contours:
        x, y, line_width, line_height = cv2.boundingRect(contour)
        if line_width < width * 0.18:
            continue
        if line_height < 3 or line_height > height * 0.09:
            continue
        if y <= 1 or y + line_height >= height - 1:
            continue

        crop = mask[y : y + line_height, x : x + line_width]
        sample_x: list[float] = []
        sample_y: list[float] = []
        for local_x in range(line_width):
            ink_y = np.flatnonzero(crop[:, local_x])
            if ink_y.size:
                sample_x.append(float(x + local_x))
                sample_y.append(float(y + np.median(ink_y)))
        if len(sample_x) < max(20, int(line_width * 0.15)):
            continue

        xs = np.asarray(sample_x, dtype=np.float64)
        ys = np.asarray(sample_y, dtype=np.float64)
        fitted = _robust_baseline(xs, ys)
        if fitted is None:
            continue
        linear = np.polyval(np.polyfit(xs, fitted, 1), xs)
        residual_curve = fitted - linear
        full_curve = np.full(width, np.nan, dtype=np.float32)
        start = max(0, int(np.ceil(xs.min())))
        end = min(width - 1, int(np.floor(xs.max())))
        target_x = np.arange(start, end + 1, dtype=np.float64)
        full_curve[start : end + 1] = np.interp(target_x, xs, residual_curve).astype(np.float32)
        curves.append(full_curve)
    return curves


def _aggregate_curve(curves: list[np.ndarray], height: int) -> np.ndarray | None:
    if len(curves) < 3:
        return None
    stack = np.vstack(curves)
    support = np.isfinite(stack).sum(axis=0)
    min_support = max(2, min(4, len(curves) // 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        curve = np.nanmedian(stack, axis=0)
    valid = np.isfinite(curve) & (support >= min_support)
    if int(valid.sum()) < max(30, curve.size // 5):
        return None

    x_coords = np.arange(curve.size, dtype=np.float32)
    curve = np.interp(x_coords, x_coords[valid], curve[valid]).astype(np.float32)
    smooth_sigma = max(3.0, curve.size / 120.0)
    curve = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), sigmaX=smooth_sigma).reshape(-1)
    curve -= np.polyval(np.polyfit(x_coords, curve, 1), x_coords).astype(np.float32)

    max_allowed = max(2.0, height * 0.06)
    peak = float(np.max(np.abs(curve)))
    if peak > max_allowed:
        curve *= max_allowed / peak
    return curve


def _textline_dewarp(image: np.ndarray) -> tuple[np.ndarray, DewarpDiagnostics]:
    if image.size == 0 or min(image.shape[:2]) < 80:
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            reason="image_too_small",
        )

    analysis, scale = _resize_for_analysis(image)
    mask = _foreground_mask(_to_gray(analysis))
    curves = _line_curves(mask)
    curve = _aggregate_curve(curves, analysis.shape[0])
    if curve is None:
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            line_count=len(curves),
            reason="insufficient_text_lines",
        )

    curvature_rms = float(np.sqrt(np.mean(np.square(curve))))
    peak = float(np.max(np.abs(curve)))
    if peak < 0.75 or curvature_rms < 0.25:
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            line_count=len(curves),
            max_displacement_px=round(peak / scale, 3),
            curvature_rms_px=round(curvature_rms / scale, 3),
            reason="curvature_below_threshold",
        )

    height, width = image.shape[:2]
    analysis_x = np.linspace(0.0, curve.size - 1, width, dtype=np.float32)
    displacement = np.interp(
        analysis_x,
        np.arange(curve.size, dtype=np.float32),
        curve,
    ).astype(np.float32)
    displacement /= scale

    map_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    map_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)).copy()
    map_y += displacement[None, :]
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, DewarpDiagnostics(
        method=DEWARP_METHOD_TEXTLINE,
        applied=True,
        line_count=len(curves),
        max_displacement_px=round(float(np.max(np.abs(displacement))), 3),
        curvature_rms_px=round(float(np.sqrt(np.mean(np.square(displacement)))), 3),
    )


def _uvdoc_dewarp(
    image: np.ndarray,
    *,
    cache_home: Path | None,
) -> tuple[np.ndarray, DewarpDiagnostics]:
    # Import lazily: PaddleOCR is intentionally an optional, heavyweight runtime.
    from .scanner_adapter import _uvdoc_document_detector

    result = _uvdoc_document_detector(image, cache_home=cache_home)
    if not result.detected or result.warped is None:
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_PADDLEOCR_UVDOC,
            applied=False,
            reason="uvdoc_no_result",
        )
    return result.warped, DewarpDiagnostics(
        method=DEWARP_METHOD_PADDLEOCR_UVDOC,
        applied=True,
    )


def dewarp_document(
    image: np.ndarray,
    *,
    method: str = DEWARP_METHOD_TEXTLINE,
    uvdoc_cache_home: Path | None = None,
) -> tuple[np.ndarray, DewarpDiagnostics]:
    """Straighten local page curvature without changing boundary detection policy."""
    normalized = method.strip().lower()
    if normalized == DEWARP_METHOD_NONE:
        return image, DewarpDiagnostics(method=normalized, applied=False, reason="disabled")
    if normalized == DEWARP_METHOD_TEXTLINE:
        return _textline_dewarp(image)
    if normalized == DEWARP_METHOD_PADDLEOCR_UVDOC:
        return _uvdoc_dewarp(image, cache_home=uvdoc_cache_home)
    raise ValueError(f"Unsupported dewarp method: {method}")
