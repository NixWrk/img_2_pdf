from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core import docshadow


class _FakeSession:
    """Stands in for an ONNX session so tests need no model file."""

    def __init__(self, output: np.ndarray, *, shape=(1, 3, 64, 64)) -> None:
        self.output = output
        self.shape = shape
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [type("Spec", (), {"name": "image", "shape": list(self.shape)})()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["image"])
        return [self.output]


def _shadowed_page(width: int = 320, height: int = 440) -> np.ndarray:
    """A text page darkened by a smooth gradient down its left side."""
    page = np.full((height, width, 3), 245, np.uint8)
    for y in range(30, height - 30, 24):
        cv2.rectangle(page, (25, y), (width - 25, y + 9), (30, 30, 30), -1)
    ramp = np.linspace(0.45, 1.0, width, dtype=np.float32)[None, :, None]
    return np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def test_input_size_reads_the_graph_and_falls_back_when_dynamic() -> None:
    fixed = _FakeSession(np.zeros((1, 3, 8, 8), np.float32), shape=(1, 3, 256, 192))
    assert docshadow.input_size(fixed) == (192, 256)

    dynamic = _FakeSession(np.zeros((1, 3, 8, 8), np.float32), shape=(1, 3, "h", "w"))
    assert docshadow.input_size(dynamic) == docshadow.DEFAULT_INPUT_SIZE


def test_gain_map_is_the_ratio_the_model_would_apply(monkeypatch) -> None:
    size = 16
    # The fake model returns its input brightened by exactly 1.5x.
    session = _FakeSession(np.zeros((1, 3, size, size), np.float32), shape=(1, 3, size, size))

    def run(_outputs, feed):
        session.inputs.append(feed["image"])
        return [np.clip(feed["image"] * 1.5, 0.0, 1.0)]

    session.run = run
    monkeypatch.setattr(docshadow, "_load_session", lambda: session)

    page = np.full((40, 30, 3), 100, np.uint8)
    gain = docshadow.gain_map(page)

    # A scalar map: illumination is achromatic, so there is one gain per pixel.
    assert gain.shape == (size, size)
    # 100/255 * 1.5 stays below 1.0, so the ratio is the full 1.5.
    np.testing.assert_allclose(gain, 1.5, rtol=0.02)


def test_gain_map_is_clamped_to_a_safe_range(monkeypatch) -> None:
    size = 8
    session = _FakeSession(np.ones((1, 3, size, size), np.float32), shape=(1, 3, size, size))
    monkeypatch.setattr(docshadow, "_load_session", lambda: session)

    # A near-black input with a white prediction would ask for a huge gain.
    gain = docshadow.gain_map(np.zeros((20, 20, 3), np.uint8))

    assert float(gain.max()) <= docshadow._MAX_GAIN
    assert float(gain.min()) >= docshadow._MIN_GAIN


def test_apply_gain_map_keeps_full_resolution_and_hue() -> None:
    page = _shadowed_page()
    gain = np.full((8, 8), 1.3, np.float32)

    result = docshadow.apply_gain_map(page, gain)

    assert result.shape == page.shape  # no downscaling to the model input
    # One scalar gain lifts every channel by the same factor, so the page gets
    # brighter without any channel drifting relative to the others.
    for channel in range(3):
        assert float(result[..., channel].mean()) > float(page[..., channel].mean())
    before_spread = float(np.ptp(page.reshape(-1, 3).mean(axis=0)))
    after_spread = float(np.ptp(result.reshape(-1, 3).mean(axis=0)))
    assert after_spread == pytest.approx(before_spread, abs=1.5)


def test_apply_gain_map_tolerates_a_per_channel_map() -> None:
    page = _shadowed_page()
    gain = np.full((8, 8, 3), 1.2, np.float32)

    result = docshadow.apply_gain_map(page, gain)

    assert result.shape == page.shape
    assert float(result.mean()) > float(page.mean())


def test_apply_gain_map_preserves_grayscale_input() -> None:
    page = cv2.cvtColor(_shadowed_page(), cv2.COLOR_BGR2GRAY)
    gain = np.full((8, 8), 1.2, np.float32)

    result = docshadow.apply_gain_map(page, gain)

    assert result.ndim == 2
    assert result.shape == page.shape


def test_remove_shadows_flattens_a_gradient_without_losing_detail(monkeypatch) -> None:
    page = _shadowed_page()
    size = 64

    def run(_outputs, feed):
        # An oracle model: undo the known ramp exactly.
        tensor = feed["image"]
        ramp = np.linspace(0.45, 1.0, tensor.shape[3], dtype=np.float32)[None, None, :]
        return [np.clip(tensor / ramp, 0.0, 1.0)]

    session = _FakeSession(np.zeros(1, np.float32), shape=(1, 3, size, size))
    session.run = run
    monkeypatch.setattr(docshadow, "_load_session", lambda: session)

    result = docshadow.remove_shadows(page)

    assert result.shape == page.shape

    def paper_unevenness(image: np.ndarray) -> float:
        """Spread of the paper level across columns.

        Measured on a high percentile so the answer describes illumination
        only: a plain column mean would mostly track how much text each
        column happens to carry.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        paper = np.percentile(gray, 95, axis=0)
        return float(paper.max() - paper.min())

    # The left-to-right brightness ramp is what should disappear.
    assert paper_unevenness(result) < paper_unevenness(page) * 0.25
    # Text is still text: the page has not been flattened into a single tone.
    assert float(result.std()) > 20.0


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty image"):
        docshadow.gain_map(np.zeros((0, 0, 3), np.uint8))


def test_output_conversion_accepts_the_common_layouts() -> None:
    hwc = np.full((4, 4, 3), 0.5, np.float32)
    np.testing.assert_allclose(docshadow._output_to_rgb(hwc), 0.5)

    nchw = np.full((1, 3, 4, 4), 0.25, np.float32)
    assert docshadow._output_to_rgb(nchw).shape == (4, 4, 3)

    # Exports that emit 0-255 are normalized back to 0-1.
    scaled = np.full((1, 3, 4, 4), 128.0, np.float32)
    np.testing.assert_allclose(docshadow._output_to_rgb(scaled), 128.0 / 255.0, rtol=1e-5)

    # DocShadow overshoots past 1.0 on bright pixels; that is still a [0, 1]
    # result and must be clipped, not rescaled as if it were 8-bit.
    overshoot = np.full((1, 3, 4, 4), 0.8, np.float32)
    overshoot[0, :, 0, 0] = 2.9
    converted = docshadow._output_to_rgb(overshoot)
    assert float(converted.max()) == pytest.approx(1.0)
    np.testing.assert_allclose(converted[1:, 1:], 0.8, rtol=1e-5)

    single = np.full((1, 1, 4, 4), 0.5, np.float32)
    assert docshadow._output_to_rgb(single).shape == (4, 4, 3)

    with pytest.raises(ValueError, match="Unexpected DocShadow output shape"):
        docshadow._output_to_rgb(np.zeros((4, 4), np.float32))


def test_model_path_honours_the_environment_override(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(docshadow.MODEL_ENV_VAR, raising=False)
    assert docshadow.model_path().name == docshadow.MODEL_FILENAME

    override = tmp_path / "custom.onnx"
    monkeypatch.setenv(docshadow.MODEL_ENV_VAR, str(override))
    assert docshadow.model_path() == override
    assert docshadow.is_available() is False
