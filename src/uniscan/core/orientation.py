"""Conservative page orientation without OCR or text recognition."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ORIENTATION_METHOD_NONE = "none"
ORIENTATION_METHOD_AUTO = "auto"
ORIENTATION_METHOD_CHOICES = (
    ORIENTATION_METHOD_NONE,
    ORIENTATION_METHOD_AUTO,
    "90",
    "180",
    "270",
)


@dataclass(slots=True, frozen=True)
class OrientationDiagnostics:
    """Explain an automatic right-angle orientation decision."""

    method: str
    applied: bool
    angle_degrees: int = 0
    confidence: float = 0.0
    line_count: int = 0
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class _LayoutEvidence:
    angle_degrees: int
    axis_score: float
    upright_score: float
    line_count: int


def _rotate_right_angle(image: np.ndarray, angle_degrees: int) -> np.ndarray:
    normalized = angle_degrees % 360
    if normalized == 0:
        return image
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Page orientation angle must be a multiple of 90 degrees.")


def _analysis_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    scale = min(1.0, 1200.0 / max(1, height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, dark = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    light = cv2.bitwise_not(dark)
    dark_ratio = float(np.count_nonzero(dark)) / max(1, dark.size)
    light_ratio = 1.0 - dark_ratio
    mask = dark if dark_ratio <= light_ratio else light

    # Page borders and camera shadows are layout noise, not text direction evidence.
    margin = max(1, min(mask.shape[:2]) // 100)
    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    return mask


def _character_boxes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(3, int(mask.size * 0.000002))
    max_area = max(min_area + 1, int(mask.size * 0.004))
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if not min_area <= area <= max_area:
            continue
        if box_width < 2 or box_height < 3:
            continue
        if box_width > width * 0.08 or box_height > height * 0.08:
            continue
        boxes.append((x, y, box_width, box_height))
    return np.asarray(boxes, dtype=np.int32).reshape(-1, 4)


def _line_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape[:2]
    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(11, width // 45), 3)),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.12 or box_width < box_height * 2.5:
            continue
        if box_height < 4 or box_height > height * 0.11:
            continue
        boxes.append((x, y, box_width, box_height))
    return boxes


def _line_upright_score(
    mask: np.ndarray,
    line: tuple[int, int, int, int],
    characters: np.ndarray,
) -> tuple[float, int]:
    x, y, width, height = line
    if characters.size == 0:
        return 0.0, 0
    centers_x = characters[:, 0] + characters[:, 2] * 0.5
    centers_y = characters[:, 1] + characters[:, 3] * 0.5
    inside = (
        (centers_x >= x) & (centers_x <= x + width) & (centers_y >= y) & (centers_y <= y + height)
    )
    members = characters[inside]
    if len(members) < 5:
        return 0.0, len(members)

    tops = members[:, 1].astype(np.float64)
    bottoms = (members[:, 1] + members[:, 3]).astype(np.float64)
    top_mad = float(np.median(np.abs(tops - np.median(tops))))
    bottom_mad = float(np.median(np.abs(bottoms - np.median(bottoms))))
    alignment = (top_mad - bottom_mad) / max(1.0, top_mad + bottom_mad)

    row_ink = np.count_nonzero(mask[y : y + height, x : x + width], axis=1)
    if row_ink.size == 0 or int(row_ink.max()) == 0:
        density_bias = 0.0
    else:
        peak_position = float(np.argmax(row_ink)) / max(1, height - 1)
        density_bias = float(np.clip((peak_position - 0.5) * 2.0, -1.0, 1.0))
    # In an upright line, glyph bodies concentrate above the baseline. The raw top/bottom
    # alignment and density terms therefore point toward the inverse (upside-down) candidate.
    upright = -(0.65 * alignment + 0.35 * density_bias)
    return float(np.clip(upright, -1.0, 1.0)), len(members)


def _layout_evidence(image: np.ndarray, angle_degrees: int) -> _LayoutEvidence:
    rotated = _rotate_right_angle(image, angle_degrees)
    gray = _analysis_gray(rotated)
    mask = _foreground_mask(gray)
    foreground_ratio = float(np.count_nonzero(mask)) / max(1, mask.size)
    if not 0.002 <= foreground_ratio <= 0.45:
        return _LayoutEvidence(angle_degrees, 0.0, 0.0, 0)

    lines = _line_boxes(mask)
    characters = _character_boxes(mask)
    if not lines:
        return _LayoutEvidence(angle_degrees, 0.0, 0.0, 0)

    width = mask.shape[1]
    line_coverage = sum(line[2] for line in lines) / max(1.0, float(width))
    axis_score = min(1.0, len(lines) / 5.0) * 0.45 + min(1.0, line_coverage / 2.5) * 0.55

    weighted_scores: list[tuple[float, int]] = []
    for line in lines:
        score, member_count = _line_upright_score(mask, line, characters)
        if member_count >= 5:
            weighted_scores.append((score, member_count))
    if weighted_scores:
        total_weight = sum(weight for _score, weight in weighted_scores)
        upright_score = sum(score * weight for score, weight in weighted_scores) / total_weight
    else:
        upright_score = 0.0
    return _LayoutEvidence(
        angle_degrees=angle_degrees,
        axis_score=round(float(axis_score), 6),
        upright_score=round(float(upright_score), 6),
        line_count=len(lines),
    )


def estimate_page_orientation(
    image: np.ndarray,
    *,
    min_axis_confidence: float = 0.22,
    min_direction_confidence: float = 0.08,
) -> OrientationDiagnostics:
    """Estimate a safe right-angle correction from layout and baseline evidence only."""
    if image.size == 0 or min(image.shape[:2]) < 80:
        return OrientationDiagnostics(
            method=ORIENTATION_METHOD_AUTO,
            applied=False,
            reason="image_too_small",
        )

    evidence = [_layout_evidence(image, angle) for angle in (0, 90, 180, 270)]
    original_axis = max(evidence[0].axis_score, evidence[2].axis_score)
    sideways_axis = max(evidence[1].axis_score, evidence[3].axis_score)
    best_axis = max(original_axis, sideways_axis)
    if best_axis < 0.18:
        return OrientationDiagnostics(
            method=ORIENTATION_METHOD_AUTO,
            applied=False,
            line_count=max(item.line_count for item in evidence),
            reason="insufficient_layout",
        )

    axis_confidence = abs(original_axis - sideways_axis) / max(0.05, original_axis + sideways_axis)
    if axis_confidence < min_axis_confidence:
        return OrientationDiagnostics(
            method=ORIENTATION_METHOD_AUTO,
            applied=False,
            confidence=round(float(axis_confidence), 3),
            line_count=max(item.line_count for item in evidence),
            reason="ambiguous_text_axis",
        )

    pair = (
        (evidence[0], evidence[2]) if original_axis > sideways_axis else (evidence[1], evidence[3])
    )
    best = max(pair, key=lambda item: item.upright_score)
    other = min(pair, key=lambda item: item.upright_score)
    direction_confidence = max(0.0, (best.upright_score - other.upright_score) * 0.5)
    confidence = float(np.sqrt(axis_confidence * direction_confidence))
    if direction_confidence < min_direction_confidence:
        return OrientationDiagnostics(
            method=ORIENTATION_METHOD_AUTO,
            applied=False,
            confidence=round(confidence, 3),
            line_count=best.line_count,
            reason="ambiguous_reading_direction",
        )

    angle = best.angle_degrees
    return OrientationDiagnostics(
        method=ORIENTATION_METHOD_AUTO,
        applied=angle != 0,
        angle_degrees=angle,
        confidence=round(confidence, 3),
        line_count=best.line_count,
        reason="already_upright" if angle == 0 else None,
    )


def orient_document(
    image: np.ndarray,
    *,
    method: str = ORIENTATION_METHOD_AUTO,
) -> tuple[np.ndarray, OrientationDiagnostics]:
    """Apply a conservative 0/90/180/270 correction without recognizing text."""
    normalized = method.strip().lower()
    if normalized not in ORIENTATION_METHOD_CHOICES:
        raise ValueError(f"Unsupported orientation method: {method}")
    if normalized == ORIENTATION_METHOD_NONE:
        return image, OrientationDiagnostics(
            method=normalized,
            applied=False,
            reason="disabled",
        )
    if normalized in {"90", "180", "270"}:
        angle = int(normalized)
        return _rotate_right_angle(image, angle), OrientationDiagnostics(
            method=normalized,
            applied=True,
            angle_degrees=angle,
            confidence=1.0,
            reason="forced",
        )

    diagnostics = estimate_page_orientation(image)
    if not diagnostics.applied:
        return image, diagnostics
    return _rotate_right_angle(image, diagnostics.angle_degrees), diagnostics
