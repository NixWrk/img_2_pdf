"""Content-box detection and consistent output page layout."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAGE_LAYOUT_NONE = "none"
PAGE_LAYOUT_A4 = "a4"
PAGE_LAYOUT_LETTER = "letter"
PAGE_LAYOUT_CHOICES = (PAGE_LAYOUT_NONE, PAGE_LAYOUT_A4, PAGE_LAYOUT_LETTER)
HORIZONTAL_ALIGNMENTS = ("left", "center", "right")
VERTICAL_ALIGNMENTS = ("top", "center", "bottom")
DEFAULT_MAX_LAYOUT_PIXELS = 150_000_000

_PAGE_SIZE_MM = {
    PAGE_LAYOUT_A4: (210.0, 297.0),
    PAGE_LAYOUT_LETTER: (215.9, 279.4),
}


@dataclass(slots=True, frozen=True)
class ContentBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True, frozen=True)
class PageLayoutDiagnostics:
    method: str
    applied: bool
    content_box: ContentBox
    content_confidence: float = 0.0
    scale: float = 1.0
    target_width: int = 0
    target_height: int = 0
    reason: str | None = None


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_content_box(image: np.ndarray) -> tuple[ContentBox, float, str | None]:
    """Find visible document content independently from the physical page boundary."""
    height, width = image.shape[:2]
    full = ContentBox(0, 0, width, height)
    if image.size == 0 or min(height, width) < 20:
        return full, 0.0, "image_too_small"

    gray = _gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(21, min(81, (min(height, width) // 16) | 1))
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        15,
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    min_area = max(4, int(image.shape[0] * image.shape[1] * 0.000002))
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if area < min_area:
            continue
        if box_width > width * 0.94 and box_height > height * 0.94:
            continue
        boxes.append((x, y, box_width, box_height, area))
    if not boxes:
        return full, 0.0, "no_content"

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    padding = max(2, min(height, width) // 200)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    ink_ratio = sum(box[4] for box in boxes) / max(1.0, float(width * height))
    confidence = min(1.0, len(boxes) / 20.0) * 0.5 + min(1.0, ink_ratio / 0.03) * 0.5
    return (
        ContentBox(left, top, max(1, right - left), max(1, bottom - top)),
        round(float(confidence), 3),
        None,
    )


def _aligned_offset(extra: int, alignment: str) -> int:
    if alignment in {"left", "top"}:
        return 0
    if alignment in {"right", "bottom"}:
        return extra
    return extra // 2


def layout_document_page(
    image: np.ndarray,
    *,
    method: str = PAGE_LAYOUT_NONE,
    dpi: int = 300,
    margin_mm: float = 10.0,
    horizontal_alignment: str = "center",
    vertical_alignment: str = "center",
    max_pixels: int = DEFAULT_MAX_LAYOUT_PIXELS,
) -> tuple[np.ndarray, PageLayoutDiagnostics]:
    """Place detected content on a standard page with reproducible margins and alignment."""
    normalized = method.strip().lower()
    if normalized not in PAGE_LAYOUT_CHOICES:
        raise ValueError(f"Unsupported page layout: {method}")
    if horizontal_alignment not in HORIZONTAL_ALIGNMENTS:
        raise ValueError(f"Unsupported horizontal alignment: {horizontal_alignment}")
    if vertical_alignment not in VERTICAL_ALIGNMENTS:
        raise ValueError(f"Unsupported vertical alignment: {vertical_alignment}")
    dpi = int(dpi)
    if dpi < 72:
        raise ValueError("Page layout DPI must be >= 72.")
    margin_mm = float(margin_mm)
    if not 0.0 <= margin_mm <= 80.0:
        raise ValueError("Page margin must be between 0 and 80 mm.")

    if normalized == PAGE_LAYOUT_NONE:
        height, width = image.shape[:2]
        return image, PageLayoutDiagnostics(
            method=normalized,
            applied=False,
            content_box=ContentBox(0, 0, width, height),
            target_width=width,
            target_height=height,
            reason="disabled",
        )

    page_width_mm, page_height_mm = _PAGE_SIZE_MM[normalized]
    target_width = int(round(page_width_mm * dpi / 25.4))
    target_height = int(round(page_height_mm * dpi / 25.4))
    pixel_limit = int(max_pixels)
    if pixel_limit < 1:
        raise ValueError("Maximum output layout pixel count must be positive.")
    target_pixels = target_width * target_height
    if target_pixels > pixel_limit:
        raise ValueError(
            f"{normalized.upper()} layout at {dpi} DPI is {target_width}x{target_height} "
            f"({target_pixels:,} pixels), above the safe output limit of "
            f"{pixel_limit:,} pixels. Reduce output PDF DPI."
        )
    content_box, confidence, content_reason = detect_content_box(image)
    margin_px = int(round(margin_mm * dpi / 25.4))
    available_width = target_width - 2 * margin_px
    available_height = target_height - 2 * margin_px
    if available_width < 1 or available_height < 1:
        raise ValueError("Page margin leaves no printable content area.")

    crop = image[
        content_box.y : content_box.y + content_box.height,
        content_box.x : content_box.x + content_box.width,
    ]
    scale = min(available_width / crop.shape[1], available_height / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    is_binary = image.ndim == 2 and np.unique(crop).size <= 2
    if is_binary:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)

    if image.ndim == 2:
        canvas = np.full((target_height, target_width), 255, dtype=image.dtype)
    else:
        canvas = np.full((target_height, target_width, image.shape[2]), 255, dtype=image.dtype)
    x = margin_px + _aligned_offset(available_width - resized_width, horizontal_alignment)
    y = margin_px + _aligned_offset(available_height - resized_height, vertical_alignment)
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas, PageLayoutDiagnostics(
        method=normalized,
        applied=True,
        content_box=content_box,
        content_confidence=confidence,
        scale=round(float(scale), 6),
        target_width=target_width,
        target_height=target_height,
        reason=content_reason,
    )
