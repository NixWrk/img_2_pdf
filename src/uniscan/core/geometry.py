"""Geometry utilities for manual page corner correction."""

from __future__ import annotations

import numpy as np
import cv2


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Input shape: (4, 2).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise ValueError("Quad points must contain only finite coordinates.")
    if np.unique(pts, axis=0).shape[0] != 4:
        raise ValueError("Quad points must contain four distinct coordinates.")
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if hull.shape[0] != 4:
        raise ValueError("Quad points must form a convex four-corner polygon.")

    # The traditional sum/difference shortcut selects the same point twice for
    # symmetric 45-degree quads.  Ordering around the centroid preserves every
    # vertex.  Start at the best top-left candidate; y/x break exact sum ties,
    # which makes a diamond start at its top point rather than duplicating it.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    cyclic = pts[np.argsort(angles)]
    start = min(
        range(4),
        key=lambda index: (
            float(cyclic[index].sum()),
            float(cyclic[index, 1]),
            float(cyclic[index, 0]),
        ),
    )
    ordered = np.roll(cyclic, -start, axis=0)

    # A valid perspective transform needs a non-degenerate convex quad.  The
    # angular order gives positive signed area in image coordinates.
    x = ordered[:, 0]
    y = ordered[:, 1]
    edges = np.roll(ordered, -1, axis=0) - ordered
    crosses = np.array(
        [
            edges[index, 0] * edges[(index + 1) % 4, 1]
            - edges[index, 1] * edges[(index + 1) % 4, 0]
            for index in range(4)
        ],
        dtype=np.float32,
    )
    if np.any(np.abs(crosses) < 1e-3) or not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise ValueError("Quad points must form a strictly convex polygon.")
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if abs(signed_area) < 1e-3:
        raise ValueError("Quad points must form a non-degenerate polygon.")
    if signed_area < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32, copy=False)


def warp_perspective_from_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply perspective transform using 4 corner points."""
    quad = order_quad_points(points)
    (tl, tr, br, bl) = quad

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    max_width = max(1, max_width)
    max_height = max(1, max_height)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
