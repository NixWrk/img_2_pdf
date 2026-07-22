"""Local page-surface correction independent from boundary detection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
import warnings

import cv2
import numpy as np

DEWARP_METHOD_NONE = "none"
DEWARP_METHOD_AUTO = "auto"
DEWARP_METHOD_TEXTLINE = "textline"
DEWARP_METHOD_UVDOC = "uvdoc"
DEWARP_METHOD_PADDLEOCR_UVDOC = "paddleocr_uvdoc"
DEWARP_METHOD_CHOICES = (
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_AUTO,
    DEWARP_METHOD_TEXTLINE,
    DEWARP_METHOD_UVDOC,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
)


@dataclass(slots=True, frozen=True)
class DewarpDiagnostics:
    """Explain whether and how local page correction was applied."""

    method: str
    applied: bool
    selected_method: str = DEWARP_METHOD_NONE
    line_count: int = 0
    max_displacement_px: float = 0.0
    curvature_rms_px: float = 0.0
    curvature_before_px: float = 0.0
    curvature_after_px: float = 0.0
    blank_border_before: float = 0.0
    blank_border_after: float = 0.0
    edge_ink_before: float = 0.0
    edge_ink_after: float = 0.0
    aspect_change: float = 0.0
    duration_ms: float = 0.0
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class DewarpQualityMetrics:
    """Image-only geometry evidence used to accept or reject a dewarp candidate."""

    curvature_rms_px: float
    line_count: int
    blank_border_ratio: float
    edge_ink_ratio: float
    aspect_ratio: float


@dataclass(slots=True, frozen=True)
class DewarpModel:
    """Editable vertical displacement curves that can be replayed."""

    method: str
    control_points: tuple[tuple[float, float], ...]
    source: str = "automatic"
    line_count: int = 0
    control_curves: tuple[tuple[float, tuple[tuple[float, float], ...]], ...] | None = None


def normalize_control_points(
    control_points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate, sort, and normalize persisted dewarp control points."""
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 3 <= points.shape[0] <= 32:
        raise ValueError("Dewarp control points must contain 3..32 (x, displacement) pairs.")
    if not np.isfinite(points).all():
        raise ValueError("Dewarp control points must be finite.")
    points = points[np.argsort(points[:, 0])]
    if np.any(np.diff(points[:, 0]) <= 0.0):
        raise ValueError("Dewarp control point x coordinates must be unique.")
    if points[0, 0] < 0.0 or points[-1, 0] > 1.0:
        raise ValueError("Dewarp control point x coordinates must be between 0 and 1.")
    if np.any(np.abs(points[:, 1]) > 0.25):
        raise ValueError("Dewarp control point displacement exceeds 25% of page height.")
    return tuple((float(x), float(y)) for x, y in points)


def interpolate_control_curve(
    control_points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    normalized_x: np.ndarray,
) -> np.ndarray:
    """Shape-preserving cubic interpolation with cubic end continuation."""
    points = normalize_control_points(control_points)
    point_x = np.asarray([point[0] for point in points], dtype=np.float64)
    point_y = np.asarray([point[1] for point in points], dtype=np.float64)
    interval_widths = np.diff(point_x)
    interval_slopes = np.diff(point_y) / interval_widths
    derivatives = np.zeros_like(point_y)

    for index in range(1, len(points) - 1):
        left_slope = interval_slopes[index - 1]
        right_slope = interval_slopes[index]
        if left_slope == 0.0 or right_slope == 0.0 or np.sign(left_slope) != np.sign(right_slope):
            derivatives[index] = 0.0
            continue
        left_weight = 2.0 * interval_widths[index] + interval_widths[index - 1]
        right_weight = interval_widths[index] + 2.0 * interval_widths[index - 1]
        derivatives[index] = (left_weight + right_weight) / (
            left_weight / left_slope + right_weight / right_slope
        )

    def endpoint_derivative(
        first_width: float,
        second_width: float,
        first_slope: float,
        second_slope: float,
    ) -> float:
        derivative = (
            (2.0 * first_width + second_width) * first_slope - first_width * second_slope
        ) / (first_width + second_width)
        if np.sign(derivative) != np.sign(first_slope):
            return 0.0
        if np.sign(first_slope) != np.sign(second_slope) and abs(derivative) > 3.0 * abs(
            first_slope
        ):
            return 3.0 * first_slope
        return float(derivative)

    derivatives[0] = endpoint_derivative(
        interval_widths[0],
        interval_widths[1],
        interval_slopes[0],
        interval_slopes[1],
    )
    derivatives[-1] = endpoint_derivative(
        interval_widths[-1],
        interval_widths[-2],
        interval_slopes[-1],
        interval_slopes[-2],
    )

    targets = np.clip(np.asarray(normalized_x, dtype=np.float64), 0.0, 1.0)
    target_shape = targets.shape
    targets = targets.reshape(-1)
    intervals = np.searchsorted(point_x, targets, side="right") - 1
    intervals = np.clip(intervals, 0, len(points) - 2)
    widths = interval_widths[intervals]
    relative = (targets - point_x[intervals]) / widths
    relative_squared = relative * relative
    relative_cubed = relative_squared * relative
    result = (
        (2.0 * relative_cubed - 3.0 * relative_squared + 1.0) * point_y[intervals]
        + (relative_cubed - 2.0 * relative_squared + relative) * widths * derivatives[intervals]
        + (-2.0 * relative_cubed + 3.0 * relative_squared) * point_y[intervals + 1]
        + (relative_cubed - relative_squared) * widths * derivatives[intervals + 1]
    )
    return np.clip(result, -0.25, 0.25).astype(np.float32).reshape(target_shape)


def normalize_control_curves(
    control_curves,
) -> tuple[tuple[float, tuple[tuple[float, float], ...]], ...]:
    """Validate and sort vertical anchors with independent horizontal curves."""
    if not isinstance(control_curves, (tuple, list)) or not 1 <= len(control_curves) <= 8:
        raise ValueError("Dewarp control curves must contain 1..8 anchored curves.")
    normalized = []
    for item in control_curves:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each dewarp curve must contain an anchor and control points.")
        anchor = float(item[0])
        if not np.isfinite(anchor) or not 0.0 <= anchor <= 1.0:
            raise ValueError("Dewarp curve anchors must be between 0 and 1.")
        normalized.append((anchor, normalize_control_points(item[1])))
    normalized.sort(key=lambda item: item[0])
    anchors = np.asarray([item[0] for item in normalized], dtype=np.float64)
    if np.any(np.diff(anchors) <= 0.0):
        raise ValueError("Dewarp curve anchors must be unique.")
    return tuple(normalized)


def _model_control_curves(
    model: DewarpModel,
) -> tuple[tuple[float, tuple[tuple[float, float], ...]], ...]:
    if model.control_curves is not None:
        return normalize_control_curves(model.control_curves)
    return ((0.5, normalize_control_points(model.control_points)),)


def _model_displacement_values(model: DewarpModel) -> np.ndarray:
    return np.asarray(
        [point[1] for _anchor, points in _model_control_curves(model) for point in points],
        dtype=np.float64,
    )


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


def _border_values(gray: np.ndarray, border: int) -> np.ndarray:
    return np.concatenate(
        (
            gray[:border, :].reshape(-1),
            gray[-border:, :].reshape(-1),
            gray[border:-border, :border].reshape(-1),
            gray[border:-border, -border:].reshape(-1),
        )
    )


def measure_dewarp_quality(image: np.ndarray) -> DewarpQualityMetrics:
    """Measure local curvature and common warp artifacts without recognizing text."""
    if image.size == 0 or len(image.shape) < 2:
        return DewarpQualityMetrics(0.0, 0, 1.0, 0.0, 0.0)

    analysis, scale = _resize_for_analysis(image)
    gray = _to_gray(analysis)
    mask = _foreground_mask(gray)
    curves = _line_curves(mask)
    curve = _aggregate_curve(curves, analysis.shape[0])
    curvature = 0.0
    if curve is not None:
        curvature = float(np.sqrt(np.mean(np.square(curve)))) / scale

    height, width = gray.shape[:2]
    border = max(1, min(height, width) // 50)
    border_values = _border_values(gray, border)
    blank_border = float(np.count_nonzero((border_values <= 4) | (border_values >= 251))) / max(
        1, border_values.size
    )
    edge_mask = _border_values(mask, border)
    edge_ink = float(np.count_nonzero(edge_mask)) / max(1, edge_mask.size)
    return DewarpQualityMetrics(
        curvature_rms_px=round(curvature, 3),
        line_count=len(curves),
        blank_border_ratio=round(blank_border, 6),
        edge_ink_ratio=round(edge_ink, 6),
        aspect_ratio=round(width / max(1.0, float(height)), 6),
    )


def _candidate_rejection_reason(
    before: DewarpQualityMetrics,
    after: DewarpQualityMetrics,
    *,
    require_curvature_improvement: bool,
    allow_reframing: bool = False,
) -> str | None:
    """Reject a candidate whose measurable geometry did not actually improve.

    ``allow_reframing`` loosens the framing limits for whole-page rectifiers
    that legitimately drop the photographed background: their output is the
    page alone, so content sits closer to the edge and the aspect ratio
    changes by design. Curvature evidence stays the deciding factor.
    """
    if after.aspect_ratio <= 0.0:
        return "invalid_output_size"
    aspect_change = abs(float(np.log(after.aspect_ratio / max(1e-6, before.aspect_ratio))))
    if aspect_change > (0.45 if allow_reframing else 0.15):
        return "excessive_aspect_change"
    if after.blank_border_ratio > before.blank_border_ratio + 0.15:
        return "new_blank_borders"
    if not allow_reframing and after.edge_ink_ratio > before.edge_ink_ratio + 0.04:
        return "content_moved_to_edge"
    if require_curvature_improvement and before.line_count >= 3:
        if after.line_count < 3:
            return "textline_evidence_lost"
        required_improvement = max(0.15, before.curvature_rms_px * 0.05)
        if after.curvature_rms_px > before.curvature_rms_px - required_improvement:
            return "curvature_not_improved"
    return None


def _quality_diagnostics(
    diagnostics: DewarpDiagnostics,
    *,
    before: DewarpQualityMetrics,
    after: DewarpQualityMetrics,
    started: float,
) -> DewarpDiagnostics:
    aspect_change = 0.0
    if before.aspect_ratio > 0.0 and after.aspect_ratio > 0.0:
        aspect_change = abs(float(np.log(after.aspect_ratio / before.aspect_ratio)))
    return replace(
        diagnostics,
        curvature_before_px=before.curvature_rms_px,
        curvature_after_px=after.curvature_rms_px,
        blank_border_before=before.blank_border_ratio,
        blank_border_after=after.blank_border_ratio,
        edge_ink_before=before.edge_ink_ratio,
        edge_ink_after=after.edge_ink_ratio,
        aspect_change=round(aspect_change, 6),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


def estimate_textline_dewarp_model(
    image: np.ndarray,
    *,
    control_point_count: int = 9,
) -> tuple[DewarpModel | None, DewarpDiagnostics]:
    """Estimate an editable normalized curve from several agreeing text lines."""
    if not 3 <= control_point_count <= 32:
        raise ValueError("control_point_count must be between 3 and 32.")
    if image.size == 0 or min(image.shape[:2]) < 80:
        return None, DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            reason="image_too_small",
        )

    analysis, scale = _resize_for_analysis(image)
    mask = _foreground_mask(_to_gray(analysis))
    curves = _line_curves(mask)
    curve = _aggregate_curve(curves, analysis.shape[0])
    if curve is None:
        return None, DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            line_count=len(curves),
            reason="insufficient_text_lines",
        )

    curvature_rms = float(np.sqrt(np.mean(np.square(curve))))
    peak = float(np.max(np.abs(curve)))
    if peak < 0.75 or curvature_rms < 0.25:
        return None, DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=False,
            line_count=len(curves),
            max_displacement_px=round(peak / scale, 3),
            curvature_rms_px=round(curvature_rms / scale, 3),
            reason="curvature_below_threshold",
        )

    point_x = np.linspace(0.0, 1.0, control_point_count, dtype=np.float32)
    curve_x = np.linspace(0.0, 1.0, curve.size, dtype=np.float32)
    point_y = np.interp(point_x, curve_x, curve).astype(np.float32) / analysis.shape[0]
    model = DewarpModel(
        method=DEWARP_METHOD_TEXTLINE,
        control_points=normalize_control_points(list(zip(point_x, point_y))),
        source="automatic",
        line_count=len(curves),
    )
    return model, DewarpDiagnostics(
        method=DEWARP_METHOD_TEXTLINE,
        applied=True,
        line_count=len(curves),
        max_displacement_px=round(peak / scale, 3),
        curvature_rms_px=round(curvature_rms / scale, 3),
    )


def apply_dewarp_model(image: np.ndarray, model: DewarpModel) -> np.ndarray:
    """Apply one curve uniformly or interpolate several curves over page height."""
    curves = _model_control_curves(model)
    height, width = image.shape[:2]
    normalized_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    smooth_sigma = max(1.0, width / 180.0)
    profiles = []
    for _anchor, points in curves:
        displacement = interpolate_control_curve(points, normalized_x) * height
        profiles.append(
            cv2.GaussianBlur(
                displacement.reshape(1, -1),
                (0, 0),
                sigmaX=smooth_sigma,
            ).reshape(-1)
        )
    profile_array = np.stack(profiles, axis=0)

    if len(curves) == 1:
        displacement_field = np.broadcast_to(profile_array[0], (height, width)).copy()
    else:
        anchors = np.asarray([curve[0] for curve in curves], dtype=np.float32)
        normalized_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        displacement_field = np.empty((height, width), dtype=np.float32)
        displacement_field[normalized_y <= anchors[0]] = profile_array[0]
        displacement_field[normalized_y >= anchors[-1]] = profile_array[-1]
        for curve_index in range(len(curves) - 1):
            lower = anchors[curve_index]
            upper = anchors[curve_index + 1]
            rows = (normalized_y >= lower) & (normalized_y <= upper)
            weights = ((normalized_y[rows] - lower) / (upper - lower))[:, None]
            displacement_field[rows] = (
                profile_array[curve_index][None, :] * (1.0 - weights)
                + profile_array[curve_index + 1][None, :] * weights
            )

    map_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    map_y = displacement_field
    map_y += np.arange(height, dtype=np.float32)[:, None]
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected


def _textline_dewarp(
    image: np.ndarray,
    *,
    model: DewarpModel | None = None,
) -> tuple[np.ndarray, DewarpDiagnostics]:
    diagnostics: DewarpDiagnostics
    if model is None:
        model, diagnostics = estimate_textline_dewarp_model(image)
        if model is None:
            return image.copy(), diagnostics
    else:
        displacements = _model_displacement_values(model)
        diagnostics = DewarpDiagnostics(
            method=DEWARP_METHOD_TEXTLINE,
            applied=True,
            line_count=model.line_count,
            max_displacement_px=round(float(np.max(np.abs(displacements))) * image.shape[0], 3),
            curvature_rms_px=round(
                float(np.sqrt(np.mean(np.square(displacements)))) * image.shape[0],
                3,
            ),
            reason="user_adjusted_model" if model.source == "user" else None,
        )
    return apply_dewarp_model(image, model), diagnostics


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


def _uvdoc_grid_dewarp(image: np.ndarray) -> tuple[np.ndarray, DewarpDiagnostics]:
    """Rectify the whole page with the bundled UVDoc grid model."""
    # Imported lazily so a missing optional runtime cannot break module import.
    from uniscan.core import uvdoc

    if not uvdoc.is_available():
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_UVDOC,
            applied=False,
            reason="uvdoc_model_unavailable",
        )
    corrected = uvdoc.dewarp(image)
    if corrected is None or corrected.size == 0:
        return image.copy(), DewarpDiagnostics(
            method=DEWARP_METHOD_UVDOC,
            applied=False,
            reason="uvdoc_no_result",
        )
    return corrected, DewarpDiagnostics(method=DEWARP_METHOD_UVDOC, applied=True)


def _try_uvdoc_grid_candidate(
    image: np.ndarray,
    *,
    before: DewarpQualityMetrics,
) -> tuple[np.ndarray, DewarpDiagnostics, DewarpQualityMetrics] | str:
    """Return an accepted UVDoc result, or the reason it was not used."""
    try:
        candidate, diagnostics = _uvdoc_grid_dewarp(image)
    # Automatic mode must stay a safe no-op whatever the optional runtime does.
    except Exception as exc:
        return f"uvdoc_unavailable:{type(exc).__name__}"
    if not diagnostics.applied:
        return diagnostics.reason or "uvdoc_no_result"
    after = measure_dewarp_quality(candidate)
    rejection = _candidate_rejection_reason(
        before,
        after,
        require_curvature_improvement=True,
        # UVDoc returns the page alone, so background loss is expected.
        allow_reframing=True,
    )
    if rejection is None and before.line_count < 3 and after.line_count < 3:
        # Neither side offers text-line evidence, so nothing can confirm the
        # warp helped. Photographs and near-blank pages land here, and forcing
        # a page model on them is worse than leaving them untouched. A genuine
        # rectification of an unreadable page reveals its lines instead.
        rejection = "no_text_line_evidence"
    if rejection is not None:
        return f"uvdoc_rejected:{rejection}"
    return candidate, diagnostics, after


def _automatic_dewarp(
    image: np.ndarray,
    *,
    before: DewarpQualityMetrics,
    uvdoc_cache_home: Path | None,
    auto_use_uvdoc: bool,
    auto_use_uvdoc_grid: bool,
    model: DewarpModel | None,
) -> tuple[np.ndarray, DewarpDiagnostics, DewarpQualityMetrics]:
    textline_candidate, textline_diagnostics = _textline_dewarp(image, model=model)
    textline_after: DewarpQualityMetrics | None = None
    if textline_diagnostics.applied:
        textline_after = measure_dewarp_quality(textline_candidate)
        rejection = None
        if model is None:
            rejection = _candidate_rejection_reason(
                before,
                textline_after,
                require_curvature_improvement=True,
            )
        if rejection is None:
            if model is not None:
                # A user-adjusted model is an explicit decision: no model gets
                # to overrule it, and no extra inference is worth running.
                return (
                    textline_candidate,
                    replace(
                        textline_diagnostics,
                        method=DEWARP_METHOD_AUTO,
                        selected_method=DEWARP_METHOD_TEXTLINE,
                    ),
                    textline_after,
                )
            textline_reason = None
        else:
            textline_after = None
            textline_reason = f"textline_rejected:{rejection}"
    else:
        textline_reason = textline_diagnostics.reason or "textline_no_result"

    uvdoc_grid_reason: str | None = None
    if auto_use_uvdoc_grid:
        outcome = _try_uvdoc_grid_candidate(image, before=before)
        if isinstance(outcome, tuple):
            grid_candidate, grid_diagnostics, grid_after = outcome
            # Both candidates passed their gates, so let the measured geometry
            # decide: the text-line model wins on pure vertical waves, UVDoc on
            # perspective and whole-surface deformation.
            if (
                textline_after is None
                or grid_after.curvature_rms_px < textline_after.curvature_rms_px
            ):
                return (
                    grid_candidate,
                    replace(
                        grid_diagnostics,
                        method=DEWARP_METHOD_AUTO,
                        selected_method=DEWARP_METHOD_UVDOC,
                        reason=(
                            None if textline_reason is None else f"textline:{textline_reason}"
                        ),
                    ),
                    grid_after,
                )
            uvdoc_grid_reason = "uvdoc_not_flatter"
        else:
            uvdoc_grid_reason = outcome

    if textline_after is not None:
        return (
            textline_candidate,
            replace(
                textline_diagnostics,
                method=DEWARP_METHOD_AUTO,
                selected_method=DEWARP_METHOD_TEXTLINE,
                reason=uvdoc_grid_reason,
            ),
            textline_after,
        )
    if uvdoc_grid_reason:
        textline_reason = f"{uvdoc_grid_reason};{textline_reason}"

    if auto_use_uvdoc:
        try:
            uvdoc_candidate, uvdoc_diagnostics = _uvdoc_dewarp(
                image,
                cache_home=uvdoc_cache_home,
            )
        # Optional third-party runtimes may surface backend-specific exception types. Automatic
        # mode must remain a safe no-op; explicit UVDoc mode still propagates its error.
        except Exception as exc:
            return (
                image.copy(),
                DewarpDiagnostics(
                    method=DEWARP_METHOD_AUTO,
                    applied=False,
                    line_count=textline_diagnostics.line_count,
                    curvature_rms_px=textline_diagnostics.curvature_rms_px,
                    reason=f"{textline_reason};uvdoc_unavailable:{type(exc).__name__}",
                ),
                before,
            )
        if uvdoc_diagnostics.applied:
            uvdoc_after = measure_dewarp_quality(uvdoc_candidate)
            rejection = _candidate_rejection_reason(
                before,
                uvdoc_after,
                require_curvature_improvement=before.line_count >= 3,
            )
            if rejection is None:
                return (
                    uvdoc_candidate,
                    replace(
                        uvdoc_diagnostics,
                        method=DEWARP_METHOD_AUTO,
                        selected_method=DEWARP_METHOD_PADDLEOCR_UVDOC,
                        reason=f"textline_fallback:{textline_reason}",
                    ),
                    uvdoc_after,
                )
            uvdoc_reason = f"uvdoc_rejected:{rejection}"
        else:
            uvdoc_reason = uvdoc_diagnostics.reason or "uvdoc_no_result"
        textline_reason = f"{textline_reason};{uvdoc_reason}"

    return (
        image.copy(),
        DewarpDiagnostics(
            method=DEWARP_METHOD_AUTO,
            applied=False,
            line_count=textline_diagnostics.line_count,
            max_displacement_px=textline_diagnostics.max_displacement_px,
            curvature_rms_px=textline_diagnostics.curvature_rms_px,
            reason=textline_reason,
        ),
        before,
    )


def dewarp_document(
    image: np.ndarray,
    *,
    method: str = DEWARP_METHOD_TEXTLINE,
    uvdoc_cache_home: Path | None = None,
    auto_use_uvdoc: bool = False,
    auto_use_uvdoc_grid: bool = True,
    model: DewarpModel | None = None,
) -> tuple[np.ndarray, DewarpDiagnostics]:
    """Straighten local page curvature without changing boundary detection policy."""
    normalized = method.strip().lower()
    started = time.perf_counter()
    if normalized == DEWARP_METHOD_NONE:
        return image, DewarpDiagnostics(method=normalized, applied=False, reason="disabled")
    before = measure_dewarp_quality(image)
    if normalized == DEWARP_METHOD_AUTO:
        corrected, diagnostics, after = _automatic_dewarp(
            image,
            before=before,
            uvdoc_cache_home=uvdoc_cache_home,
            auto_use_uvdoc=auto_use_uvdoc,
            auto_use_uvdoc_grid=auto_use_uvdoc_grid,
            model=model,
        )
        return corrected, _quality_diagnostics(
            diagnostics,
            before=before,
            after=after,
            started=started,
        )
    if normalized == DEWARP_METHOD_TEXTLINE:
        corrected, diagnostics = _textline_dewarp(image, model=model)
        diagnostics = replace(
            diagnostics,
            selected_method=DEWARP_METHOD_TEXTLINE if diagnostics.applied else DEWARP_METHOD_NONE,
        )
        return corrected, _quality_diagnostics(
            diagnostics,
            before=before,
            after=measure_dewarp_quality(corrected),
            started=started,
        )
    if normalized == DEWARP_METHOD_UVDOC:
        corrected, diagnostics = _uvdoc_grid_dewarp(image)
        if model is not None and diagnostics.applied:
            # An explicit user model refines the rectified page.
            corrected = apply_dewarp_model(corrected, model)
            diagnostics = replace(diagnostics, reason="uvdoc_with_user_adjustment")
        diagnostics = replace(
            diagnostics,
            selected_method=DEWARP_METHOD_UVDOC if diagnostics.applied else DEWARP_METHOD_NONE,
        )
        return corrected, _quality_diagnostics(
            diagnostics,
            before=before,
            after=measure_dewarp_quality(corrected),
            started=started,
        )
    if normalized == DEWARP_METHOD_PADDLEOCR_UVDOC:
        corrected, diagnostics = _uvdoc_dewarp(image, cache_home=uvdoc_cache_home)
        if model is not None:
            corrected = apply_dewarp_model(corrected, model)
            diagnostics = DewarpDiagnostics(
                method=normalized,
                applied=True,
                line_count=model.line_count,
                max_displacement_px=round(
                    float(np.max(np.abs(_model_displacement_values(model)))) * corrected.shape[0],
                    3,
                ),
                reason="uvdoc_with_user_adjustment",
            )
        diagnostics = replace(
            diagnostics,
            selected_method=(
                DEWARP_METHOD_PADDLEOCR_UVDOC if diagnostics.applied else DEWARP_METHOD_NONE
            ),
        )
        return corrected, _quality_diagnostics(
            diagnostics,
            before=before,
            after=measure_dewarp_quality(corrected),
            started=started,
        )
    raise ValueError(f"Unsupported dewarp method: {method}")
