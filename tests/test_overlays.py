import numpy as np

from uniscan.ui.overlays import draw_quad_overlay, scale_contour


def _blank(width: int = 100, height: int = 80) -> np.ndarray:
    return np.full((height, width, 3), 30, dtype=np.uint8)


def test_draw_quad_overlay_preserves_shape() -> None:
    image = _blank()
    contour = np.array([[10, 10], [80, 10], [80, 60], [10, 60]], dtype=np.float32)
    out = draw_quad_overlay(image, contour)
    assert out.shape == image.shape


def test_draw_quad_overlay_writes_green_pixels() -> None:
    image = _blank()
    contour = np.array([[10, 10], [80, 10], [80, 60], [10, 60]], dtype=np.float32)
    out = draw_quad_overlay(image, contour, thickness=3)
    # Original was uniform [30, 30, 30]; expect some pixels to have a strong green channel.
    green_dominant = (out[..., 1].astype(int) - out[..., 0].astype(int)) > 80
    assert green_dominant.any()


def test_draw_quad_overlay_handles_grayscale_input() -> None:
    image = np.full((50, 70), 100, dtype=np.uint8)
    contour = np.array([[5, 5], [65, 5], [65, 45], [5, 45]], dtype=np.float32)
    out = draw_quad_overlay(image, contour)
    assert out.ndim == 3
    assert out.shape == (50, 70, 3)


def test_scale_contour_maps_between_sizes() -> None:
    contour = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    scaled = scale_contour(contour, src_shape=(50, 100), dst_shape=(25, 50))
    assert scaled.shape == contour.shape
    assert np.isclose(scaled[1, 0], 50.0)
    assert np.isclose(scaled[2, 1], 25.0)
