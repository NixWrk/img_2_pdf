from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core import uvdoc
from uniscan.core.dewarp import (
    DEWARP_METHOD_AUTO,
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_UVDOC,
    DewarpDiagnostics,
    dewarp_document,
    measure_dewarp_quality,
)


def _identity_grid(rows: int = 45, columns: int = 31) -> np.ndarray:
    """A grid that samples every output pixel from its own position."""
    ys = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, columns, dtype=np.float32)
    grid_x = np.repeat(xs[None, :], rows, axis=0)
    grid_y = np.repeat(ys[:, None], columns, axis=1)
    return np.stack([grid_x, grid_y])[None]


def _text_page(width: int = 320, height: int = 440) -> np.ndarray:
    page = np.full((height, width, 3), 250, np.uint8)
    for y in range(40, height - 40, 24):
        cv2.rectangle(page, (30, y), (width - 30, y + 9), (30, 30, 30), -1)
    return page


class _FakeSession:
    """Stands in for an ONNX session so tests need no model file."""

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [type("Spec", (), {"name": "image"})()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["image"])
        return [self.grid]


def test_identity_grid_becomes_the_identity_map() -> None:
    image = _text_page()
    height, width = image.shape[:2]

    map_x, map_y = uvdoc.grid_to_backward_map(_identity_grid(), size=(width, height))

    # Corner alignment: the outermost grid samples must land exactly on the
    # outermost pixels, or the rectified page drifts off its own edges.
    assert map_x.shape == (height, width)
    expected_x = np.tile(np.linspace(0.0, width - 1, width, dtype=np.float32), (height, 1))
    expected_y = np.repeat(
        np.linspace(0.0, height - 1, height, dtype=np.float32)[:, None], width, axis=1
    )
    np.testing.assert_allclose(map_x, expected_x, atol=0.01)
    np.testing.assert_allclose(map_y, expected_y, atol=0.01)

    # Sampling through it returns the same page.
    remapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    assert float(np.abs(remapped.astype(np.int16) - image.astype(np.int16)).mean()) < 1.0


def test_grid_maps_normalized_coordinates_onto_pixels() -> None:
    grid = np.zeros((1, 2, 4, 4), np.float32)  # every sample at the image centre
    map_x, map_y = uvdoc.grid_to_backward_map(grid, size=(101, 201))

    assert np.allclose(map_x, 50.0)
    assert np.allclose(map_y, 100.0)


def test_grid_shape_and_size_are_validated() -> None:
    with pytest.raises(ValueError, match="Unexpected UVDoc grid shape"):
        uvdoc.grid_to_backward_map(np.zeros((3, 4, 4), np.float32), size=(10, 10))
    with pytest.raises(ValueError, match="at least 2x2"):
        uvdoc.grid_to_backward_map(_identity_grid(), size=(1, 10))


def test_dewarp_feeds_a_normalized_rgb_tensor_and_remaps(monkeypatch) -> None:
    image = _text_page()
    session = _FakeSession(_identity_grid())
    monkeypatch.setattr(uvdoc, "_load_session", lambda: session)

    result = uvdoc.dewarp(image)

    tensor = session.inputs[0]
    assert tensor.shape == (1, 3, uvdoc.UVDOC_INPUT_SIZE[1], uvdoc.UVDOC_INPUT_SIZE[0])
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    # Channel order is RGB while OpenCV images are BGR.
    assert float(tensor[0, 0].mean()) == pytest.approx(float(tensor[0, 2].mean()), abs=0.02)
    assert result.shape == image.shape


def test_dewarp_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty image"):
        uvdoc.predict_grid(np.zeros((0, 0, 3), np.uint8))


def test_model_path_honours_the_environment_override(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(uvdoc.MODEL_ENV_VAR, raising=False)
    assert uvdoc.model_path().name == uvdoc.MODEL_FILENAME

    override = tmp_path / "custom.onnx"
    monkeypatch.setenv(uvdoc.MODEL_ENV_VAR, str(override))
    assert uvdoc.model_path() == override
    assert uvdoc.is_available() is False  # the override does not exist yet

    override.write_bytes(b"not really a model")
    uvdoc.reset_session_cache()
    with pytest.raises(Exception):
        uvdoc.predict_grid(np.zeros((10, 10, 3), np.uint8))


def test_missing_model_is_reported_not_raised(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(uvdoc.MODEL_ENV_VAR, str(tmp_path / "absent.onnx"))
    image = _text_page()

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_UVDOC)

    assert diagnostics.applied is False
    assert diagnostics.reason == "uvdoc_model_unavailable"
    assert diagnostics.selected_method == DEWARP_METHOD_NONE
    np.testing.assert_array_equal(corrected, image)


def test_auto_prefers_the_flatter_candidate(monkeypatch) -> None:
    image = _text_page()
    flat = _text_page()
    barely = _text_page()
    cv2.rectangle(barely, (30, 100), (barely.shape[1] - 30, 109), (200, 200, 200), -1)

    monkeypatch.setattr(
        "uniscan.core.dewarp._textline_dewarp",
        lambda source, *, model: (
            flat,
            DewarpDiagnostics(method="textline", applied=True, line_count=9),
        ),
    )
    monkeypatch.setattr(
        "uniscan.core.dewarp._uvdoc_grid_dewarp",
        lambda source: (barely, DewarpDiagnostics(method=DEWARP_METHOD_UVDOC, applied=True)),
    )
    monkeypatch.setattr(
        "uniscan.core.dewarp._candidate_rejection_reason",
        lambda *_args, **_kwargs: None,
    )

    flat_curvature = measure_dewarp_quality(flat).curvature_rms_px
    barely_curvature = measure_dewarp_quality(barely).curvature_rms_px
    expected = DEWARP_METHOD_UVDOC if barely_curvature < flat_curvature else "textline"

    _corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_AUTO)

    assert diagnostics.selected_method == expected


def test_auto_keeps_a_user_model_without_running_uvdoc(monkeypatch) -> None:
    from uniscan.core.dewarp import DewarpModel

    image = _text_page()
    calls: list[bool] = []

    def unexpected(_image):
        calls.append(True)
        raise AssertionError("a user-adjusted model must win outright")

    monkeypatch.setattr("uniscan.core.dewarp._uvdoc_grid_dewarp", unexpected)
    model = DewarpModel(
        method="textline",
        control_points=((0.0, 0.0), (0.5, 0.02), (1.0, 0.0)),
        source="user",
    )

    _corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_AUTO, model=model)

    assert calls == []
    assert diagnostics.selected_method == "textline"


def test_auto_can_disable_the_grid_model(monkeypatch) -> None:
    image = _text_page()
    calls: list[bool] = []

    monkeypatch.setattr(
        "uniscan.core.dewarp._uvdoc_grid_dewarp",
        lambda _image: calls.append(True) or (image, DewarpDiagnostics("uvdoc", True)),
    )

    dewarp_document(image, method=DEWARP_METHOD_AUTO, auto_use_uvdoc_grid=False)

    assert calls == []
