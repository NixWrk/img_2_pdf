import cv2
import numpy as np
import pytest

from uniscan.core.geometry import (
    order_quad_points,
    prepare_perspective_warp,
    warp_perspective_from_points,
)


def test_order_quad_points_returns_consistent_order() -> None:
    points = np.array(
        [
            [90, 10],  # tr
            [15, 120],  # bl
            [100, 130],  # br
            [10, 20],  # tl
        ],
        dtype=np.float32,
    )
    ordered = order_quad_points(points)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0]
    assert bl[1] > tl[1]
    assert br[0] > bl[0]


def test_order_quad_points_keeps_all_vertices_for_45_degree_diamond() -> None:
    points = np.array(
        [[50, 0], [100, 50], [50, 100], [0, 50]],
        dtype=np.float32,
    )

    ordered = order_quad_points(points)

    np.testing.assert_array_equal(
        ordered,
        np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32),
    )
    assert np.unique(ordered, axis=0).shape[0] == 4

    image = np.zeros((101, 101), dtype=np.uint8)
    cv2.fillConvexPoly(image, points.astype(np.int32), 255)
    warped = warp_perspective_from_points(image, points)
    assert float(np.mean(warped)) > 240.0


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [100, 0], [100, 100], [50, 50]],
        [[0, 0], [50, 0], [100, 0], [150, 0]],
    ],
)
def test_order_quad_points_rejects_concave_or_degenerate_polygon(points) -> None:
    with pytest.raises(ValueError, match="convex"):
        order_quad_points(np.asarray(points, dtype=np.float32))


def test_warp_perspective_from_points_outputs_non_empty_image() -> None:
    image = np.zeros((120, 140, 3), dtype=np.uint8)
    image[20:100, 30:110] = (255, 255, 255)
    points = np.array(
        [
            [30, 20],
            [110, 20],
            [110, 100],
            [30, 100],
        ],
        dtype=np.float32,
    )
    warped = warp_perspective_from_points(image, points)
    assert warped.size > 0
    assert warped.shape[0] >= 70
    assert warped.shape[1] >= 70


def test_identity_quad_preserves_every_source_pixel() -> None:
    image = np.arange(37 * 53 * 3, dtype=np.uint8).reshape(37, 53, 3)
    points = np.array(
        [[0, 0], [52, 0], [52, 36], [0, 36]],
        dtype=np.float32,
    )

    warped = warp_perspective_from_points(image, points)

    assert warped.shape == image.shape
    np.testing.assert_array_equal(warped, image)


@pytest.mark.parametrize(
    ("image_shape", "points", "expected_size"),
    [
        ((37, 53), [[0, 0], [52, 0], [52, 36], [0, 36]], (53, 37)),
        ((101, 101), [[50, 0], [100, 50], [50, 100], [0, 50]], (72, 72)),
        (
            (20, 20),
            [[1.25, 2.25], [10.75, 2.25], [10.75, 12.75], [1.25, 12.75]],
            (11, 12),
        ),
    ],
    ids=("axis-aligned", "rotated", "fractional"),
)
def test_perspective_size_uses_inclusive_pixel_centres(image_shape, points, expected_size) -> None:
    image = np.zeros(image_shape, dtype=np.uint8)

    _quad, _destination, output_size = prepare_perspective_warp(
        image,
        np.asarray(points, dtype=np.float32),
    )

    assert output_size == expected_size


def test_perspective_rejects_singleton_dimension_before_opencv(monkeypatch) -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    points = np.asarray(
        [[1, 1], [1.4, 1], [1.4, 11], [1, 11]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "uniscan.core.geometry.cv2.warpPerspective",
        lambda *_args, **_kwargs: pytest.fail("degenerate warp was attempted"),
    )

    with pytest.raises(ValueError, match="at least 2x2"):
        warp_perspective_from_points(image, points)


def _virtual_image(shape: tuple[int, int]) -> np.ndarray:
    """Expose a large shape without allocating its pixels."""
    return np.lib.stride_tricks.as_strided(
        np.zeros(1, dtype=np.uint8),
        shape=shape,
        strides=(0, 0),
    )


@pytest.mark.parametrize(
    ("image", "points", "message"),
    [
        (
            _virtual_image((200, 32_767)),
            [[0, 0], [100, 0], [100, 100], [0, 100]],
            "Perspective source",
        ),
        (
            _virtual_image((32_766, 32_766)),
            [[0, 0], [32_765, 32_265], [32_765, 32_765], [0, 500]],
            "output dimensions",
        ),
        (
            _virtual_image((20_000, 20_000)),
            [[0, 0], [19_999, 0], [19_999, 10_000], [0, 10_000]],
            "safe limit",
        ),
        (
            _virtual_image((4, 1_002)),
            [[0, 0], [1_001, 0], [1_001, 1], [0, 1]],
            "aspect ratio",
        ),
    ],
    ids=("source-dimension", "output-dimension", "pixels", "aspect"),
)
def test_perspective_limits_fail_before_opencv_output_allocation(
    image, points, message, monkeypatch
) -> None:
    monkeypatch.setattr(
        "uniscan.core.geometry.cv2.warpPerspective",
        lambda *_args, **_kwargs: pytest.fail("unsafe warp allocation was attempted"),
    )

    with pytest.raises(ValueError, match=message):
        warp_perspective_from_points(image, np.asarray(points, dtype=np.float32))


def test_perspective_accepts_largest_safe_opencv_source_dimension() -> None:
    image = _virtual_image((329, 32_766))
    points = np.asarray(
        [[0, 0], [32_765, 0], [32_765, 328], [0, 328]],
        dtype=np.float32,
    )

    _quad, _destination, output_size = prepare_perspective_warp(image, points)

    assert output_size == (32_766, 329)


def test_perspective_checks_aspect_before_rounding() -> None:
    image = np.zeros((120, 10), dtype=np.uint8)
    points = np.asarray(
        [[1, 1], [1.6, 1], [1.6, 101], [1, 101]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="quad aspect ratio"):
        prepare_perspective_warp(image, points)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (
            [[0, 0], [53, 0], [52, 36], [0, 36]],
            "source pixel-centre bounds",
        ),
        (
            [[0, 0], [52, 0], [52, float("inf")], [0, 36]],
            "finite",
        ),
    ],
)
def test_perspective_rejects_invalid_coordinates(points, message) -> None:
    image = np.zeros((37, 53), dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        prepare_perspective_warp(image, np.asarray(points, dtype=np.float64))
