import cv2
import numpy as np
import pytest

from uniscan.core.geometry import order_quad_points, warp_perspective_from_points


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
