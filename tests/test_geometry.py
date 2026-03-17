import numpy as np

from uniscan.core.geometry import order_quad_points, warp_perspective_from_points


def test_order_quad_points_returns_consistent_order() -> None:
    points = np.array(
        [
            [90, 10],   # tr
            [15, 120],  # bl
            [100, 130], # br
            [10, 20],   # tl
        ],
        dtype=np.float32,
    )
    ordered = order_quad_points(points)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0]
    assert bl[1] > tl[1]
    assert br[0] > bl[0]


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
