"""Document binarization, conservative despeckle, and lighting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BINARIZATION_NONE = "none"
BINARIZATION_FIXED = "fixed"
BINARIZATION_OTSU = "otsu"
BINARIZATION_SAUVOLA = "sauvola"
BINARIZATION_WOLF = "wolf"
BINARIZATION_CHOICES = (
    BINARIZATION_NONE,
    BINARIZATION_FIXED,
    BINARIZATION_OTSU,
    BINARIZATION_SAUVOLA,
    BINARIZATION_WOLF,
)

DESPECKLE_NONE = "none"
DESPECKLE_CONSERVATIVE = "conservative"
DESPECKLE_NORMAL = "normal"
DESPECKLE_STRONG = "strong"
DESPECKLE_CHOICES = (
    DESPECKLE_NONE,
    DESPECKLE_CONSERVATIVE,
    DESPECKLE_NORMAL,
    DESPECKLE_STRONG,
)


@dataclass(slots=True, frozen=True)
class DespeckleDiagnostics:
    strength: str
    applied: bool
    candidate_components: int = 0
    removed_components: int = 0
    removed_pixels: int = 0
    protected_components: int = 0
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class LightingDiagnostics:
    shadow_fraction: float
    glare_fraction: float
    clipped_pixel_fraction: float
    illumination_range: float
    unevenness: float
    warnings: tuple[str, ...] = ()


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _validated_window(window_size: int, shape: tuple[int, int]) -> int:
    window = int(window_size)
    if window < 3:
        raise ValueError("Binarization window must be >= 3.")
    if window % 2 == 0:
        window += 1
    maximum = min(shape)
    if maximum < 3:
        return 1
    if window > maximum:
        window = maximum if maximum % 2 == 1 else maximum - 1
    return max(3, window)


def _local_mean_std(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    source = gray.astype(np.float32)
    mean = cv2.boxFilter(source, cv2.CV_32F, (window, window), normalize=True)
    square_mean = cv2.boxFilter(source * source, cv2.CV_32F, (window, window), normalize=True)
    deviation = cv2.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    return mean, deviation


def binarize_document(
    image: np.ndarray,
    *,
    method: str,
    threshold: int = 170,
    window_size: int = 31,
    k: float | None = None,
) -> np.ndarray:
    """Convert a document to binary using selectable, reproducible algorithms."""
    normalized = method.strip().lower()
    if normalized not in BINARIZATION_CHOICES:
        raise ValueError(f"Unsupported binarization method: {method}")
    if normalized == BINARIZATION_NONE:
        return image
    if image.size == 0:
        return image.copy()
    gray = _gray(image)
    if normalized == BINARIZATION_FIXED:
        _value, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        return binary
    if normalized == BINARIZATION_OTSU:
        _value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    window = _validated_window(window_size, gray.shape[:2])
    mean, deviation = _local_mean_std(gray, window)
    if normalized == BINARIZATION_SAUVOLA:
        coefficient = 0.2 if k is None else float(k)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("Sauvola k must be between 0 and 1.")
        local_threshold = mean * (1.0 + coefficient * (deviation / 128.0 - 1.0))
    else:
        coefficient = 0.5 if k is None else float(k)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("Wolf k must be between 0 and 1.")
        maximum_deviation = max(1.0, float(deviation.max()))
        minimum_gray = float(gray.min())
        local_threshold = mean + coefficient * (deviation / maximum_deviation - 1.0) * (
            mean - minimum_gray
        )
    return np.where(gray.astype(np.float32) > local_threshold, 255, 0).astype(np.uint8)


def despeckle_document(
    image: np.ndarray,
    *,
    strength: str = DESPECKLE_CONSERVATIVE,
) -> tuple[np.ndarray, DespeckleDiagnostics]:
    """Remove only tiny isolated dark components, preserving punctuation near other ink."""
    normalized = strength.strip().lower()
    if normalized not in DESPECKLE_CHOICES:
        raise ValueError(f"Unsupported despeckle strength: {strength}")
    if normalized == DESPECKLE_NONE:
        return image, DespeckleDiagnostics(normalized, False, reason="disabled")
    if image.size == 0 or min(image.shape[:2]) < 8:
        return image.copy(), DespeckleDiagnostics(normalized, False, reason="image_too_small")

    gray = _gray(image)
    unique = np.unique(gray)
    if unique.size <= 2 and set(unique.tolist()).issubset({0, 255}):
        foreground = (gray < 128).astype(np.uint8)
    else:
        _value, foreground = cv2.threshold(
            gray,
            0,
            1,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    base_area = {
        DESPECKLE_CONSERVATIVE: 2,
        DESPECKLE_NORMAL: 5,
        DESPECKLE_STRONG: 10,
    }[normalized]
    resolution_scale = max(0.75, np.sqrt(gray.size / (1200.0 * 1600.0)))
    max_area = max(1, int(round(base_area * resolution_scale)))
    proximity = max(5, int(round(np.sqrt(max_area) * 3.0)))

    remove_labels: list[int] = []
    candidates = 0
    protected = 0
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area > max_area:
            continue
        candidates += 1
        left = max(0, x - proximity)
        top = max(0, y - proximity)
        right = min(gray.shape[1], x + width + proximity)
        bottom = min(gray.shape[0], y + height + proximity)
        neighbours = np.unique(labels[top:bottom, left:right])
        has_other_ink = any(value not in {0, label} for value in neighbours.tolist())
        if has_other_ink:
            protected += 1
        else:
            remove_labels.append(label)

    if not remove_labels:
        return image.copy(), DespeckleDiagnostics(
            normalized,
            False,
            candidate_components=candidates,
            protected_components=protected,
            reason="no_isolated_specks",
        )
    remove_mask = np.isin(labels, np.asarray(remove_labels, dtype=labels.dtype))
    result = image.copy()
    result[remove_mask] = 255
    return result, DespeckleDiagnostics(
        normalized,
        True,
        candidate_components=candidates,
        removed_components=len(remove_labels),
        removed_pixels=int(np.count_nonzero(remove_mask)),
        protected_components=protected,
    )


def analyze_lighting(image: np.ndarray) -> LightingDiagnostics:
    """Measure smooth shadows and anomalous clipped highlights without claiming recovery."""
    if image.size == 0:
        return LightingDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, ("empty_image",))
    gray = _gray(image)
    height, width = gray.shape[:2]
    scale = min(1.0, 900.0 / max(1, height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    source = gray.astype(np.float32)
    sigma = max(7.0, min(45.0, min(gray.shape[:2]) / 10.0))
    illumination = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    p10, median, p90 = (float(value) for value in np.percentile(illumination, (10, 50, 90)))
    shadow_threshold = median - max(18.0, median * 0.12)
    shadow_mask = (illumination < shadow_threshold) & (source > 60.0)

    highlight_context = cv2.GaussianBlur(source, (0, 0), sigmaX=20.0, sigmaY=20.0)
    glare_mask = (source >= 250.0) & (highlight_context <= 245.0)
    clipped_mask = source >= 252.0
    illumination_range = max(0.0, p90 - p10)
    unevenness = illumination_range / max(1.0, median)
    shadow_fraction = float(np.count_nonzero(shadow_mask)) / max(1, shadow_mask.size)
    glare_fraction = float(np.count_nonzero(glare_mask)) / max(1, glare_mask.size)
    clipped_fraction = float(np.count_nonzero(clipped_mask)) / max(1, clipped_mask.size)
    warnings: list[str] = []
    if shadow_fraction >= 0.05 or unevenness >= 0.28:
        warnings.append("uneven_shadow")
    if glare_fraction >= 0.001:
        warnings.append("possible_glare")
    return LightingDiagnostics(
        shadow_fraction=round(shadow_fraction, 6),
        glare_fraction=round(glare_fraction, 6),
        clipped_pixel_fraction=round(clipped_fraction, 6),
        illumination_range=round(illumination_range, 3),
        unevenness=round(unevenness, 6),
        warnings=tuple(warnings),
    )
