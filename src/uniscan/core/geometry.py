"""Geometry utilities for manual page corner correction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import cv2

MAX_PERSPECTIVE_IMAGE_DIMENSION = 32_766
MAX_PERSPECTIVE_OUTPUT_PIXELS = 150_000_000
MAX_PERSPECTIVE_ASPECT_RATIO = 100.0


@dataclass(slots=True, frozen=True)
class BackwardMap:
    """Dense output-to-input coordinates for one geometric operation."""

    map_x: np.ndarray
    map_y: np.ndarray

    def __post_init__(self) -> None:
        map_x = np.asarray(self.map_x, dtype=np.float32)
        map_y = np.asarray(self.map_y, dtype=np.float32)
        if map_x.ndim != 2 or map_y.shape != map_x.shape or not map_x.size:
            raise ValueError("Backward maps must be non-empty equally sized 2-D arrays.")
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValueError("Backward maps must contain only finite coordinates.")
        object.__setattr__(self, "map_x", np.ascontiguousarray(map_x))
        object.__setattr__(self, "map_y", np.ascontiguousarray(map_y))

    @property
    def output_size(self) -> tuple[int, int]:
        height, width = self.map_x.shape
        return width, height


def identity_backward_map(size: tuple[int, int]) -> BackwardMap:
    """Return an exact pixel-centre identity map for ``(width, height)``."""
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Backward-map dimensions must be positive.")
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return BackwardMap(map_x, map_y)


def rotate_points_right_angle(
    points: np.ndarray,
    *,
    source_size: tuple[int, int],
    angle_degrees: int,
) -> np.ndarray:
    """Move source pixel-centre coordinates through an exact right-angle rotation."""
    width, height = source_size
    normalized = int(angle_degrees) % 360
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = array[:, 0]
    y = array[:, 1]
    if normalized == 0:
        return array.copy()
    if normalized == 90:
        return np.column_stack((height - 1 - y, x)).astype(np.float32)
    if normalized == 180:
        return np.column_stack((width - 1 - x, height - 1 - y)).astype(np.float32)
    if normalized == 270:
        return np.column_stack((y, width - 1 - x)).astype(np.float32)
    raise ValueError("Point rotation angle must be a multiple of 90 degrees.")


def right_angle_backward_map(
    size: tuple[int, int],
    angle_degrees: int,
) -> BackwardMap:
    """Map an exact right-angle rotated image back to its input coordinates."""
    width, height = size
    normalized = int(angle_degrees) % 360
    if normalized == 0:
        return identity_backward_map(size)
    if normalized in {90, 270}:
        output_width, output_height = height, width
    elif normalized == 180:
        output_width, output_height = width, height
    else:
        raise ValueError("Right-angle map requires a multiple of 90 degrees.")
    output_x, output_y = np.meshgrid(
        np.arange(output_width, dtype=np.float32),
        np.arange(output_height, dtype=np.float32),
    )
    if normalized == 90:
        return BackwardMap(output_y, height - 1 - output_x)
    if normalized == 180:
        return BackwardMap(width - 1 - output_x, height - 1 - output_y)
    return BackwardMap(width - 1 - output_y, output_x)


def slice_backward_map(
    backward_map: BackwardMap,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> BackwardMap:
    """Slice an intermediate image and its output-to-source map identically."""
    width, height = backward_map.output_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Backward-map slice must stay inside its output bounds.")
    return BackwardMap(
        backward_map.map_x[y0:y1, x0:x1],
        backward_map.map_y[y0:y1, x0:x1],
    )


def perspective_backward_map(image: np.ndarray, points: np.ndarray) -> BackwardMap:
    """Plan the same four-corner crop as ``warpPerspective`` without sampling pixels."""
    quad, destination, output_size = prepare_perspective_warp(image, points)
    output_to_source = cv2.getPerspectiveTransform(destination, quad)
    width, height = output_size
    output_x, output_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    homogeneous = output_to_source @ np.stack(
        (output_x.reshape(-1), output_y.reshape(-1), np.ones(width * height)),
        axis=0,
    )
    denominator = homogeneous[2]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("Perspective transform has points at infinity.")
    map_x = (homogeneous[0] / denominator).reshape(height, width)
    map_y = (homogeneous[1] / denominator).reshape(height, width)
    return BackwardMap(map_x, map_y)


def rotation_backward_map(
    size: tuple[int, int],
    angle_degrees: float,
) -> BackwardMap:
    """Map the expanded deskew output back to the unrotated input."""
    width, height = size
    if width < 1 or height < 1 or not math.isfinite(angle_degrees):
        raise ValueError("Rotation map requires positive dimensions and a finite angle.")
    if abs(angle_degrees) < 0.05:
        return identity_backward_map(size)
    center = (width / 2.0, height / 2.0)
    input_to_output = cv2.getRotationMatrix2D(center, float(angle_degrees), 1.0)
    cosine = abs(float(input_to_output[0, 0]))
    sine = abs(float(input_to_output[0, 1]))
    output_width = max(1, int(np.ceil(width * cosine + height * sine)))
    output_height = max(1, int(np.ceil(height * cosine + width * sine)))
    input_to_output[0, 2] += output_width / 2.0 - center[0]
    input_to_output[1, 2] += output_height / 2.0 - center[1]
    output_to_input = cv2.invertAffineTransform(input_to_output)
    output_x, output_y = np.meshgrid(
        np.arange(output_width, dtype=np.float32),
        np.arange(output_height, dtype=np.float32),
    )
    map_x = (
        output_to_input[0, 0] * output_x
        + output_to_input[0, 1] * output_y
        + output_to_input[0, 2]
    )
    map_y = (
        output_to_input[1, 0] * output_x
        + output_to_input[1, 1] * output_y
        + output_to_input[1, 2]
    )
    return BackwardMap(map_x, map_y)


def compose_backward_maps(
    input_map: BackwardMap,
    output_to_input: BackwardMap,
) -> BackwardMap:
    """Compose ``intermediate -> source`` with ``output -> intermediate``."""
    map_x = cv2.remap(
        input_map.map_x,
        output_to_input.map_x,
        output_to_input.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    map_y = cv2.remap(
        input_map.map_y,
        output_to_input.map_x,
        output_to_input.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return BackwardMap(map_x, map_y)


def render_backward_map(
    source: np.ndarray,
    backward_map: BackwardMap,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Sample authoritative pixels once through an already composed map."""
    if not isinstance(source, np.ndarray) or source.size == 0:
        raise ValueError("Backward-map source must be a non-empty image.")
    return cv2.remap(
        source,
        backward_map.map_x,
        backward_map.map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


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
