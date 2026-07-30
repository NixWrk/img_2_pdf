"""Conservative diagnostics for suspicious automatic page boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


LARGE_DARK_BORDER_REASON = "large_dark_border_region"
BOUNDARY_NOT_DETECTED_REASON = "boundary_not_detected"
EMPTY_OUTPUT_REASON = "empty_output"

_MAX_ANALYSIS_DIMENSION = 900
_MIN_COMPONENT_AREA_FRACTION = 0.003
_DARK_BORDER_REVIEW_THRESHOLD = 0.12


@dataclass(frozen=True, slots=True)
class BoundaryReviewDiagnostics:
    """Explain why a produced page should receive manual boundary review."""

    needs_review: bool = False
    reasons: tuple[str, ...] = ()
    dark_border_fraction: float = 0.0


def _analysis_gray(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, _MAX_ANALYSIS_DIMENSION / max(1, height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _dark_border_fraction(image: np.ndarray) -> float:
    gray = _analysis_gray(image)
    height, width = gray.shape[:2]
    if height < 2 or width < 2:
        return 0.0

    sigma = max(3.0, min(height, width) / 120.0)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    y_margin = int(round(height * 0.20))
    x_margin = int(round(width * 0.20))
    center = smooth[
        y_margin : max(y_margin + 1, height - y_margin),
        x_margin : max(x_margin + 1, width - x_margin),
    ]
    paper_level = float(np.percentile(center, 75))
    dark_threshold = max(25.0, paper_level - 55.0)
    dark = np.asarray(smooth < dark_threshold, dtype=np.uint8)

    kernel_size = max(3, int(round(min(height, width) / 100.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    image_area = max(1, height * width)
    min_component_area = image_area * _MIN_COMPONENT_AREA_FRACTION
    border_area = 0
    for label in range(1, component_count):
        x, y, box_width, box_height, area = stats[label]
        touches_border = x == 0 or y == 0 or x + box_width >= width or y + box_height >= height
        if touches_border and area >= min_component_area:
            border_area += int(area)
    return border_area / image_area


def assess_boundary_review(
    image: np.ndarray,
    *,
    detection_enabled: bool,
    detected: bool,
    proposal_only: bool = False,
) -> BoundaryReviewDiagnostics:
    """Assess an automatically committed crop without changing its pixels.

    Crop proposals are already explicitly pending operator approval in the GUI,
    while disabled detection is an intentional policy choice. Neither receives
    this additional automatic quality flag.
    """
    if not detection_enabled or proposal_only:
        return BoundaryReviewDiagnostics()
    if image is None or image.size == 0:
        return BoundaryReviewDiagnostics(True, (EMPTY_OUTPUT_REASON,))

    reasons: list[str] = []
    if not detected:
        reasons.append(BOUNDARY_NOT_DETECTED_REASON)
    dark_border_fraction = _dark_border_fraction(image)
    if dark_border_fraction >= _DARK_BORDER_REVIEW_THRESHOLD:
        reasons.append(LARGE_DARK_BORDER_REASON)
    return BoundaryReviewDiagnostics(
        needs_review=bool(reasons),
        reasons=tuple(reasons),
        dark_border_fraction=round(dark_border_fraction, 6),
    )
