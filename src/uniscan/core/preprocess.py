"""Preprocessing presets and enhancement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DESKEW_METHOD_NONE = "none"
DESKEW_METHOD_HYBRID = "hybrid"
DESKEW_METHOD_HOUGH = "hough"
DESKEW_METHOD_MIN_AREA = "min_area"
DESKEW_METHOD_CHOICES = (
    DESKEW_METHOD_NONE,
    DESKEW_METHOD_HYBRID,
    DESKEW_METHOD_HOUGH,
    DESKEW_METHOD_MIN_AREA,
)


@dataclass(slots=True)
class PreprocessSettings:
    contrast: float = 1.0
    brightness: int = 0
    denoise: int = 0
    threshold: int = 170
    apply_threshold: bool = False
    correct_illumination: bool = False


@dataclass(slots=True, frozen=True)
class LensModeProfile:
    preset_name: str
    postprocess_name: str


@dataclass(slots=True, frozen=True)
class SkewEstimate:
    """One deskew estimate with enough diagnostics to compare algorithms."""

    angle_degrees: float
    method: str
    confidence: float
    line_count: int = 0


PREPROCESS_PRESETS: dict[str, PreprocessSettings] = {
    "Custom": PreprocessSettings(),
    "Document": PreprocessSettings(
        contrast=1.25, brightness=10, denoise=4, threshold=170, apply_threshold=False
    ),
    "Whiteboard": PreprocessSettings(
        contrast=1.35, brightness=20, denoise=5, threshold=185, apply_threshold=False
    ),
    "Photo": PreprocessSettings(
        contrast=1.05, brightness=0, denoise=2, threshold=170, apply_threshold=False
    ),
    "B/W High Contrast": PreprocessSettings(
        contrast=1.45,
        brightness=8,
        denoise=4,
        threshold=165,
        apply_threshold=True,
    ),
}


LENS_MODE_PROFILES: dict[str, LensModeProfile] = {
    "Document": LensModeProfile(preset_name="Document", postprocess_name="Grayscale"),
    "Whiteboard": LensModeProfile(preset_name="Whiteboard", postprocess_name="Grayscale"),
    "Photo": LensModeProfile(preset_name="Photo", postprocess_name="None"),
    "B/W": LensModeProfile(preset_name="B/W High Contrast", postprocess_name="Black and White"),
}

LENS_MODE_CUSTOM = "Custom"
LENS_MODE_VALUES: tuple[str, ...] = tuple([*LENS_MODE_PROFILES.keys(), LENS_MODE_CUSTOM])


def resolve_lens_mode_profile(mode_name: str) -> LensModeProfile | None:
    if mode_name == LENS_MODE_CUSTOM:
        return None
    return LENS_MODE_PROFILES.get(mode_name)


def infer_lens_mode(preset_name: str, postprocess_name: str) -> str:
    for mode_name, profile in LENS_MODE_PROFILES.items():
        if profile.preset_name == preset_name and profile.postprocess_name == postprocess_name:
            return mode_name
    return LENS_MODE_CUSTOM


def correct_illumination(image: np.ndarray) -> np.ndarray:
    """Reduce smooth shadows and compress highlights without OCR or content rotation."""
    if image.size == 0:
        return image.copy()

    is_gray = image.ndim == 2
    if is_gray:
        lightness = image
        color = None
    else:
        color = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness = color[:, :, 0]

    min_side = min(lightness.shape[:2])
    if min_side < 5:
        return image.copy()
    sigma = max(5.0, min(45.0, min_side / 10.0))
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 1)
    corrected = cv2.divide(lightness, background, scale=245)
    corrected = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(corrected)

    if is_gray:
        return corrected
    assert color is not None
    color[:, :, 0] = corrected
    return cv2.cvtColor(color, cv2.COLOR_LAB2BGR)


def apply_enhancements(image: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    """Apply denoise, contrast/brightness, and optional binary threshold."""
    out = correct_illumination(image) if settings.correct_illumination else image
    denoise = max(0, int(settings.denoise))
    if denoise > 0:
        if len(out.shape) == 2:
            out = cv2.fastNlMeansDenoising(out, None, h=float(denoise))
        else:
            out = cv2.fastNlMeansDenoisingColored(
                out,
                None,
                h=float(denoise),
                hColor=float(denoise),
                templateWindowSize=7,
                searchWindowSize=21,
            )

    out = cv2.convertScaleAbs(out, alpha=float(settings.contrast), beta=int(settings.brightness))

    if settings.apply_threshold:
        if len(out.shape) == 3:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        else:
            gray = out
        _, out = cv2.threshold(gray, int(settings.threshold), 255, cv2.THRESH_BINARY)

    return out


def _deskew_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _min_area_skew(gray: np.ndarray, *, max_angle: float) -> SkewEstimate:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = cv2.bitwise_not(threshold)
    coords = np.column_stack(np.where(foreground > 0))
    if coords.size == 0:
        return SkewEstimate(0.0, DESKEW_METHOD_MIN_AREA, 0.0)

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = float(rect[-1])
    if angle < -45.0:
        angle = 90.0 + angle
    elif angle > 45.0:
        angle -= 90.0
    if abs(angle) > max_angle:
        return SkewEstimate(0.0, DESKEW_METHOD_MIN_AREA, 0.0)
    foreground_ratio = float(np.count_nonzero(foreground)) / max(1, foreground.size)
    confidence = min(1.0, foreground_ratio / 0.08)
    return SkewEstimate(angle, DESKEW_METHOD_MIN_AREA, confidence)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = float(weights.sum()) * 0.5
    return float(
        sorted_values[min(len(sorted_values) - 1, int(np.searchsorted(cumulative, cutoff)))]
    )


def _hough_skew(gray: np.ndarray, *, max_angle: float) -> SkewEstimate:
    height, width = gray.shape[:2]
    scale = min(1.0, 1600.0 / max(1, height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(25, width // 24),
        minLineLength=max(30, int(width * 0.12)),
        maxLineGap=max(8, int(width * 0.025)),
    )
    if lines is None:
        return SkewEstimate(0.0, DESKEW_METHOD_HOUGH, 0.0)

    angles: list[float] = []
    lengths: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        if abs(angle) <= max_angle:
            angles.append(angle)
            lengths.append(length)

    if len(angles) < 3:
        return SkewEstimate(0.0, DESKEW_METHOD_HOUGH, 0.0, line_count=len(angles))
    angle_arr = np.asarray(angles, dtype=np.float64)
    length_arr = np.asarray(lengths, dtype=np.float64)
    estimate = _weighted_median(angle_arr, length_arr)
    median_deviation = _weighted_median(np.abs(angle_arr - estimate), length_arr)
    coverage = min(1.0, float(length_arr.sum()) / max(1.0, width * 4.0))
    agreement = max(0.0, 1.0 - (median_deviation / 5.0))
    return SkewEstimate(
        estimate,
        DESKEW_METHOD_HOUGH,
        coverage * agreement,
        line_count=len(angles),
    )


def estimate_document_skew(
    image: np.ndarray,
    *,
    method: str = DESKEW_METHOD_HYBRID,
    max_angle: float = 20.0,
) -> SkewEstimate:
    """Estimate small document rotation using selectable, comparable algorithms."""
    normalized = method.strip().lower()
    if normalized not in DESKEW_METHOD_CHOICES:
        raise ValueError(f"Unsupported deskew method: {method}")
    if normalized == DESKEW_METHOD_NONE or image.size == 0:
        return SkewEstimate(0.0, normalized, 1.0)
    gray = _deskew_gray(image)
    if normalized == DESKEW_METHOD_MIN_AREA:
        return _min_area_skew(gray, max_angle=max_angle)
    if normalized == DESKEW_METHOD_HOUGH:
        return _hough_skew(gray, max_angle=max_angle)

    hough = _hough_skew(gray, max_angle=max_angle)
    if hough.confidence >= 0.12 and hough.line_count >= 3:
        return SkewEstimate(
            hough.angle_degrees,
            DESKEW_METHOD_HYBRID,
            hough.confidence,
            hough.line_count,
        )
    min_area = _min_area_skew(gray, max_angle=max_angle)
    return SkewEstimate(
        min_area.angle_degrees,
        DESKEW_METHOD_HYBRID,
        min_area.confidence,
        hough.line_count,
    )


def deskew_document(
    image: np.ndarray,
    *,
    method: str = DESKEW_METHOD_HYBRID,
    max_angle: float = 20.0,
) -> tuple[np.ndarray, float]:
    """
    Try to estimate and correct document skew.

    Returns `(deskewed_image, applied_angle_degrees)`.
    """
    estimate = estimate_document_skew(image, method=method, max_angle=max_angle)
    angle = estimate.angle_degrees
    if abs(angle) < 0.05:
        return image, 0.0

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)
