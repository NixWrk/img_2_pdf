"""Two-page spread detection and accurate gutter-aware splitting.

The detector combines three column-wise profiles in the central band of the
image — darkness, vertical-edge density, and content (ink) density — into a
single cost. The minimum of the smoothed cost inside the search band marks
the candidate gutter. Confidence is then derived from three checks:

1. **darkness contrast** — the gutter must be noticeably darker than the
   rest of the band.
2. **content balance** — the page split at the gutter should leave
   comparable amounts of ink on either side.
3. **edge continuity** — long vertical gradients along the gutter line raise
   confidence (an actual binding usually produces a long contiguous edge).

If the resulting confidence falls below ``min_confidence`` the detector
returns ``None`` and the caller decides whether to fall back to a naive
midpoint split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class GutterCandidate:
    """One candidate gutter location and its confidence components."""

    x: int
    confidence: float
    darkness_score: float
    edge_score: float
    balance_score: float


@dataclass(slots=True, frozen=True)
class SpreadSplitResult:
    """Pages and evidence produced by one automatic spread decision."""

    pages: tuple[np.ndarray, ...]
    candidate: GutterCandidate | None
    reason: str


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalized(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; zero array stays zero."""
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    if span <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / span).astype(np.float32)


def _gaussian_blur_1d(values: np.ndarray, *, sigma: float) -> np.ndarray:
    if values.size == 0 or sigma <= 0:
        return values.astype(np.float32, copy=False)
    kernel_size = max(3, int(round(sigma * 6)) | 1)
    source = values.astype(np.float32, copy=False).reshape(1, -1)
    return cv2.GaussianBlur(
        source,
        (kernel_size, 1),
        sigmaX=sigma,
        sigmaY=0,
        borderType=cv2.BORDER_CONSTANT,
    ).reshape(-1)


def detect_spread_gutter(
    image: np.ndarray,
    *,
    search_band: tuple[float, float] = (0.30, 0.70),
    min_aspect: float = 1.3,
    min_confidence: float = 0.5,
) -> GutterCandidate | None:
    """Detect the gutter column inside a two-page spread.

    Returns ``None`` if no confident candidate is found, leaving the caller
    free to fall back to a midpoint split or refuse to split altogether.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    if width < 16 or height < 16:
        return None

    aspect = width / max(1, height)
    if aspect < min_aspect:
        return None

    low_frac, high_frac = search_band
    low_frac = max(0.05, min(0.95, low_frac))
    high_frac = max(low_frac + 0.05, min(0.99, high_frac))
    low = int(round(width * low_frac))
    high = int(round(width * high_frac))
    if high - low < 4:
        return None

    gray = _to_gray(image)
    # CLAHE normalizes uneven page lighting so the binding shadow is visible.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    # Vertical-edge response over the full image height (used for edge-continuity scoring).
    sobel_full = cv2.Sobel(blurred, cv2.CV_32F, dx=1, dy=0, ksize=3)
    abs_sobel_full = np.abs(sobel_full)

    # Restrict column profiles to the vertical middle so binding shadow / page curl don't dominate.
    top = int(round(height * 0.10))
    bottom = int(round(height * 0.90))
    band = blurred[top:bottom, :]
    if band.shape[0] < 8:
        return None

    # 1. Darkness per column (gutter columns are darker on average).
    column_means = band.astype(np.float32).mean(axis=0)
    darkness_full = 255.0 - column_means

    # 2. Vertical-edge density per column within the central band.
    edge_band = abs_sobel_full[top:bottom, :]
    edge_full = edge_band.mean(axis=0)

    # 3. Content (ink) density per column — Otsu binarization then ratio of ink pixels.
    _, otsu = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    content_full = otsu.astype(np.float32).mean(axis=0) / 255.0  # in [0, 1]

    band_darkness = darkness_full[low:high]
    band_edge = edge_full[low:high]
    n = band_darkness.size
    norm_darkness = _normalized(band_darkness)
    norm_edge = _normalized(band_edge)
    norm_content = _normalized(content_full[low:high])

    # Cost: prefer dark + vertical-edge columns with low ink (the binding seam itself).
    weights = (0.5, 0.3, 0.2)
    cost = -weights[0] * norm_darkness - weights[1] * norm_edge + weights[2] * norm_content
    smoothed = _gaussian_blur_1d(cost, sigma=max(2.0, width / 200.0))

    # Mild centring bias so two equally-strong candidates resolve toward the middle.
    centre = (n - 1) / 2.0
    if centre > 0:
        distance = np.abs(np.arange(n, dtype=np.float32) - centre) / centre
        smoothed = smoothed + 0.05 * distance

    local_index = int(np.argmin(smoothed))
    x = low + local_index

    # Confidence components -------------------------------------------------
    band_median = float(np.median(band_darkness))
    band_std = float(np.std(band_darkness)) + 1e-6
    darkness_contrast = (float(band_darkness[local_index]) - band_median) / band_std
    darkness_score = float(np.clip(darkness_contrast / 2.5, 0.0, 1.0))

    left_content = float(content_full[:x].mean()) if x > 0 else 0.0
    right_content = float(content_full[x:].mean()) if x < width else 0.0
    denom = max(left_content, right_content, 1e-6)
    balance_score = float(np.clip(1.0 - abs(left_content - right_content) / denom, 0.0, 1.0))

    # Edge continuity: how much of the vertical extent near x has strong vertical edges.
    window = max(3, int(round(width * 0.01)))
    x_lo = max(0, x - window)
    x_hi = min(width, x + window + 1)
    edge_max_per_row = abs_sobel_full[:, x_lo:x_hi].max(axis=1)
    edge_max_normalized = edge_max_per_row / (edge_max_per_row.max() + 1e-6)
    edge_continuity = float((edge_max_normalized > 0.35).mean())
    edge_score = float(np.clip(edge_continuity, 0.0, 1.0))

    confidence = float(
        np.clip(0.5 * darkness_score + 0.3 * edge_score + 0.2 * balance_score, 0.0, 1.0)
    )

    if confidence < min_confidence:
        return None

    return GutterCandidate(
        x=int(x),
        confidence=confidence,
        darkness_score=darkness_score,
        edge_score=edge_score,
        balance_score=balance_score,
    )


def split_spread_accurate(
    image: np.ndarray,
    *,
    fallback: Literal["midpoint", "none"] = "midpoint",
    min_confidence: float = 0.5,
    search_band: tuple[float, float] = (0.30, 0.70),
    min_aspect: float = 1.3,
) -> list[np.ndarray]:
    """Split a two-page spread at the detected gutter.

    Falls back according to ``fallback`` if no confident gutter is found.
    """
    return list(
        split_spread_analyzed(
            image,
            fallback=fallback,
            min_confidence=min_confidence,
            search_band=search_band,
            min_aspect=min_aspect,
        ).pages
    )


def split_spread_analyzed(
    image: np.ndarray,
    *,
    fallback: Literal["midpoint", "none"] = "none",
    min_confidence: float = 0.5,
    search_band: tuple[float, float] = (0.30, 0.70),
    min_aspect: float = 1.3,
) -> SpreadSplitResult:
    """Split only with gutter evidence and return the decision diagnostics."""
    candidate = detect_spread_gutter(
        image,
        search_band=search_band,
        min_aspect=min_aspect,
        min_confidence=min_confidence,
    )
    if candidate is not None:
        cut = candidate.x
    elif fallback == "none":
        return SpreadSplitResult((image,), None, "no_confident_gutter")
    else:
        if image is None or image.size == 0:
            pages = (image,) if image is not None else ()
            return SpreadSplitResult(pages, None, "empty_image")
        _, width = image.shape[:2]
        if width < 2:
            return SpreadSplitResult((image,), None, "image_too_narrow")
        cut = width // 2

    if cut <= 0 or cut >= image.shape[1]:
        return SpreadSplitResult((image,), candidate, "invalid_gutter")
    left = image[:, :cut]
    right = image[:, cut:]
    if left.size == 0 or right.size == 0:
        return SpreadSplitResult((image,), candidate, "empty_half")
    reason = "gutter_detected" if candidate is not None else "midpoint_fallback"
    return SpreadSplitResult((left, right), candidate, reason)
