"""Geometry utilities for manual page corner correction."""

from __future__ import annotations

import math

import numpy as np
import cv2

MAX_PERSPECTIVE_IMAGE_DIMENSION = 32_766
MAX_PERSPECTIVE_OUTPUT_PIXELS = 150_000_000
MAX_PERSPECTIVE_ASPECT_RATIO = 100.0


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


def _inclusive_pixel_count(distance: float) -> int:
    """Convert a distance between pixel centres to an inclusive pixel count."""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Perspective edge lengths must be finite and non-negative.")
    return max(1, math.floor(distance + 0.5) + 1)


def prepare_perspective_warp(
    image: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Validate and plan a bounded perspective warp without allocating its output.

    Coordinates identify pixel centres. Consequently, an edge from ``0`` to
    ``W - 1`` spans ``W`` output pixels, including both endpoints.
    """
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("Perspective source must be an image array with at least two dimensions.")
    image_height, image_width = image.shape[:2]
    if image_width < 1 or image_height < 1:
        raise ValueError("Perspective source dimensions must be positive.")
    if (
        image_width > MAX_PERSPECTIVE_IMAGE_DIMENSION
        or image_height > MAX_PERSPECTIVE_IMAGE_DIMENSION
    ):
        raise ValueError(
            f"Perspective source {image_width}x{image_height} exceeds OpenCV's safe "
            f"per-dimension limit of {MAX_PERSPECTIVE_IMAGE_DIMENSION:,} pixels."
        )

    try:
        raw_points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Perspective points must contain exactly four x/y coordinates.") from exc
    if not np.isfinite(raw_points).all():
        raise ValueError("Perspective points must contain only finite coordinates.")

    x_coordinates = raw_points[:, 0]
    y_coordinates = raw_points[:, 1]
    if (
        np.any(x_coordinates < 0)
        or np.any(x_coordinates > image_width - 1)
        or np.any(y_coordinates < 0)
        or np.any(y_coordinates > image_height - 1)
    ):
        raise ValueError(
            "Perspective points must stay within source pixel-centre bounds "
            f"x=0..{image_width - 1}, y=0..{image_height - 1}."
        )

    quad = order_quad_points(raw_points)
    precise_quad = quad.astype(np.float64)
    tl, tr, br, bl = precise_quad
    width_extent = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height_extent = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    if width_extent < 0.5 or height_extent < 0.5:
        raise ValueError(
            "Perspective output must be at least 2x2 pixels; a one-pixel dimension "
            "would create a degenerate homography."
        )
    geometric_aspect = max(width_extent / height_extent, height_extent / width_extent)
    if geometric_aspect > MAX_PERSPECTIVE_ASPECT_RATIO:
        raise ValueError(
            f"Perspective quad aspect ratio {geometric_aspect:.2f}:1 exceeds the safe limit "
            f"of {MAX_PERSPECTIVE_ASPECT_RATIO:g}:1."
        )

    width = _inclusive_pixel_count(width_extent)
    height = _inclusive_pixel_count(height_extent)

    if width > MAX_PERSPECTIVE_IMAGE_DIMENSION or height > MAX_PERSPECTIVE_IMAGE_DIMENSION:
        raise ValueError(
            "Perspective output dimensions "
            f"{width}x{height} exceed the safe per-dimension limit of "
            f"{MAX_PERSPECTIVE_IMAGE_DIMENSION:,} pixels."
        )
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > MAX_PERSPECTIVE_ASPECT_RATIO:
        raise ValueError(
            f"Perspective output aspect ratio {aspect_ratio:.2f}:1 exceeds the safe limit of "
            f"{MAX_PERSPECTIVE_ASPECT_RATIO:g}:1."
        )
    output_pixels = width * height
    if output_pixels > MAX_PERSPECTIVE_OUTPUT_PIXELS:
        raise ValueError(
            f"Perspective output {width}x{height} requires {output_pixels:,} pixels, "
            f"exceeding the safe limit of {MAX_PERSPECTIVE_OUTPUT_PIXELS:,}."
        )

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    return quad, destination, (width, height)


def warp_perspective_from_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply perspective transform using 4 corner points."""
    quad, destination, output_size = prepare_perspective_warp(image, points)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(image, matrix, output_size)
