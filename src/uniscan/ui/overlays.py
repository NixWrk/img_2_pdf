"""Drawing helpers for the document-boundary overlay."""

from __future__ import annotations

import cv2
import numpy as np


_DEFAULT_LINE_COLOR = (0, 255, 102)
_DEFAULT_CORNER_COLOR = (255, 51, 85)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_quad_overlay(
    image: np.ndarray,
    contour: np.ndarray,
    *,
    line_color: tuple[int, int, int] = _DEFAULT_LINE_COLOR,
    corner_color: tuple[int, int, int] = _DEFAULT_CORNER_COLOR,
    thickness: int = 2,
    corner_radius: int = 6,
) -> np.ndarray:
    """Return a copy of ``image`` with the document quad drawn over it.

    ``contour`` is expected as a (4, 2) array of (x, y) points in image
    coordinates. Anything else is returned unchanged.
    """
    if contour is None:
        return image.copy()

    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return image.copy()

    canvas = _ensure_bgr(image).copy()
    pts_int = np.round(points).astype(np.int32)
    cv2.polylines(
        canvas,
        [pts_int.reshape(-1, 1, 2)],
        isClosed=True,
        color=line_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    for point in pts_int:
        center = (int(point[0]), int(point[1]))
        cv2.circle(canvas, center, corner_radius, corner_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(
            canvas, center, corner_radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )

    return canvas


def scale_contour(
    contour: np.ndarray,
    *,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Rescale a contour from one image size to another (both (h, w))."""
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    scale_x = dst_w / max(1, src_w)
    scale_y = dst_h / max(1, src_h)
    out = np.asarray(contour, dtype=np.float32).copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out
