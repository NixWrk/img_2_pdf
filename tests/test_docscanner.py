from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core import docscanner
from uniscan.core.dewarp import (
    DEWARP_METHOD_DOCSCANNER,
    DEWARP_METHOD_NONE,
    dewarp_document,
)


def _constant_grid(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    grid = np.empty((1, 2, 288, 288), dtype=np.float32)
    grid[:, 0] = x
    grid[:, 1] = y
    return grid


class _FakeSession:
    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [type("Spec", (), {"name": "image"})()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["image"])
        return [self.grid]


def _colour_page() -> np.ndarray:
    image = np.zeros((31, 41, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 1] = 80
    image[:, :, 2] = 220
    cv2.rectangle(image, (10, 8), (30, 22), (120, 140, 160), -1)
    return image


def test_predict_grid_prepares_the_official_rgb_tensor(monkeypatch) -> None:
    session = _FakeSession(_constant_grid())
    monkeypatch.setattr(docscanner, "_load_session", lambda: session)

    grid = docscanner.predict_grid(_colour_page())

    assert grid.shape == (1, 2, 288, 288)
    tensor = session.inputs[0]
    assert tensor.shape == (1, 3, 288, 288)
    assert tensor.dtype == np.float32
    assert float(tensor[0, 0].mean()) > float(tensor[0, 2].mean())


def test_constant_grid_samples_the_page_centre(monkeypatch) -> None:
    image = _colour_page()
    session = _FakeSession(_constant_grid())
    monkeypatch.setattr(docscanner, "_load_session", lambda: session)

    result = docscanner.dewarp(image)

    expected = image[image.shape[0] // 2, image.shape[1] // 2]
    assert result.shape == image.shape
    np.testing.assert_allclose(result, np.broadcast_to(expected, result.shape), atol=1)


def test_grid_shape_and_target_size_are_validated() -> None:
    with pytest.raises(ValueError, match="Unexpected DocScanner-L grid shape"):
        docscanner.grid_to_backward_map(np.zeros((1, 3, 4, 4)), size=(20, 20))
    with pytest.raises(ValueError, match="at least 2x2"):
        docscanner.grid_to_backward_map(_constant_grid(), size=(1, 20))


def test_environment_override_is_still_hash_pinned(monkeypatch, tmp_path) -> None:
    path = tmp_path / "custom.onnx"
    path.write_bytes(b"not the pinned graph")
    monkeypatch.setenv(docscanner.MODEL_ENV_VAR, str(path))
    docscanner.reset_session_cache()

    with pytest.raises(RuntimeError, match="size mismatch"):
        docscanner.verify_model(path)


def test_missing_docscanner_is_a_safe_explicit_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(docscanner.MODEL_ENV_VAR, str(tmp_path / "missing.onnx"))
    image = _colour_page()

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_DOCSCANNER)

    np.testing.assert_array_equal(corrected, image)
    assert diagnostics.applied is False
    assert diagnostics.reason == "docscanner_model_unavailable"
    assert diagnostics.selected_method == DEWARP_METHOD_NONE


def test_docscanner_is_an_explicit_dewarp_stage(monkeypatch) -> None:
    image = _colour_page()
    expected = np.full_like(image, 177)
    monkeypatch.setattr(docscanner, "is_available", lambda: True)
    monkeypatch.setattr(docscanner, "dewarp", lambda _image: expected)

    corrected, diagnostics = dewarp_document(image, method=DEWARP_METHOD_DOCSCANNER)

    np.testing.assert_array_equal(corrected, expected)
    assert diagnostics.applied is True
    assert diagnostics.method == DEWARP_METHOD_DOCSCANNER
    assert diagnostics.selected_method == DEWARP_METHOD_DOCSCANNER
