"""DocShadow (FSENet) shadow removal through ONNX Runtime.

The network runs at its own small input size, which is far below a scanned
page. Rather than upscaling its output — that would throw away the page's real
resolution — this module keeps only what the model is actually good at: the
illumination it removed. Document shadows are largely multiplicative, so the
ratio between the model's shadow-free result and its input is a smooth gain
map. Upsampling that map and applying it to the full-resolution page transfers
the correction while every glyph keeps its original pixels.

Weights are the MIT-licensed DocShadow model exported to ONNX (also MIT); see
``src/uniscan/models/README.md``. The runtime is optional: everything here
degrades to "unavailable" when onnxruntime or the model file is missing.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from uniscan.model_assets import model_file_identity, verify_model_asset

MODEL_FILENAME = "docshadow_sd7k.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_ENV_VAR = "UNISCAN_DOCSHADOW_MODEL"

# Used when the exported graph declares a dynamic input size, which the
# bundled export does. Only a smooth illumination map is kept, so a small
# input loses nothing: measured against the project's lighting metrics this
# was both the cleanest result and the fastest (~350 ms vs ~1.3 s at 512).
DEFAULT_INPUT_SIZE = (256, 256)
# The gain map is smooth by nature; clamping keeps a mispredicted patch from
# blowing out highlights or crushing ink into black.
_MIN_GAIN = 0.5
_MAX_GAIN = 3.0
_EPSILON = 1e-3
# Gaussian sigma for the gain map, as a fraction of its shorter side. Enough
# to erase text structure from the ratio (measured: eight times less
# high-frequency content than half this value) without over-smoothing, which
# would flatten the map against the page border and weaken the correction
# exactly where shadows are usually strongest.
_GAIN_SMOOTH_RATIO = 0.02

_session_lock = threading.Lock()
_session_cache: dict[str, object] = {}


def model_path() -> Path:
    """Configured or bundled model location (may not exist)."""
    override = os.environ.get(MODEL_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return MODEL_DIR / MODEL_FILENAME


def is_available() -> bool:
    """True when the model file and the ONNX runtime can both be used."""
    if not model_path().is_file():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def model_identity() -> str:
    """Content identity used by persistent processing-cache keys."""
    return "docshadow:" + model_file_identity(model_path())


def _load_session():
    """Return a cached inference session for the configured model."""
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"DocShadow model is missing: {path}. Set {MODEL_ENV_VAR} to a DocShadow ONNX file."
        )
    if not os.environ.get(MODEL_ENV_VAR):
        verify_model_asset("docshadow_sd7k", path)
    key = str(path.resolve())
    with _session_lock:
        session = _session_cache.get(key)
        if session is not None:
            return session
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("DocShadow shadow removal requires onnxruntime.") from exc
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _session_cache[key] = session
        return session


def reset_session_cache() -> None:
    """Drop cached sessions so a changed model path is picked up."""
    with _session_lock:
        _session_cache.clear()


def input_size(session) -> tuple[int, int]:
    """Network input as (width, height), falling back when it is dynamic."""
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or height <= 0:
        height = DEFAULT_INPUT_SIZE[1]
    if not isinstance(width, int) or width <= 0:
        width = DEFAULT_INPUT_SIZE[0]
    return int(width), int(height)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _prepare_input(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to the network input and return a normalized RGB NCHW tensor."""
    resized = cv2.resize(_to_bgr(image), size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


def _output_to_rgb(output: np.ndarray) -> np.ndarray:
    """Convert the network output to an HxWx3 RGB float image in [0, 1]."""
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unexpected DocShadow output shape: {np.asarray(output).shape}")
    if array.shape[0] in (1, 3):  # NCHW
        array = np.transpose(array, (1, 2, 0))
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] != 3:
        raise ValueError(f"Unexpected DocShadow output shape: {np.asarray(output).shape}")
    # Some exports emit [0, 255] instead of [0, 1]. Decide on the mean, not
    # the maximum: this network overshoots well past 1.0 on bright pixels, so
    # a maximum-based test would misread a [0, 1] result as an 8-bit one.
    if float(array.mean()) > 2.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def gain_map(image: np.ndarray) -> np.ndarray:
    """Scalar illumination gain the model would apply, at model resolution.

    The gain is taken from luminance only and smoothed, because illumination
    is achromatic and low-frequency. A per-channel ratio would instead fold
    the network's small colour differences into a cast, and its text-edge
    ringing into haloes around every glyph.
    """
    if image is None or image.size == 0 or image.ndim < 2:
        raise ValueError("DocShadow requires a non-empty image.")
    session = _load_session()
    size = input_size(session)
    tensor = _prepare_input(image, size)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    if not outputs:
        raise RuntimeError("DocShadow returned no output tensor.")
    clean = _output_to_rgb(outputs[0])
    shadowed = np.transpose(tensor[0], (1, 2, 0))
    if clean.shape[:2] != shadowed.shape[:2]:
        clean = cv2.resize(
            clean, (shadowed.shape[1], shadowed.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    gain = (_luminance(clean) + _EPSILON) / (_luminance(shadowed) + _EPSILON)
    gain = np.clip(gain, _MIN_GAIN, _MAX_GAIN)
    # Strip whatever page content survived the ratio: an illumination map must
    # carry no text structure at all.
    sigma = max(1.0, min(gain.shape[:2]) * _GAIN_SMOOTH_RATIO)
    smoothed = cv2.GaussianBlur(gain, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return smoothed.astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an HxWx3 RGB float image."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def apply_gain_map(image: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply a model-resolution scalar gain map to a full-resolution page."""
    source = _to_bgr(image)
    height, width = source.shape[:2]
    dense = np.asarray(gain, dtype=np.float32)
    if dense.ndim == 3:  # tolerate a per-channel map by averaging it
        dense = dense.mean(axis=2)
    # The map is smooth, so a plain bilinear upsample carries no blockiness,
    # and the page keeps every pixel of its own detail.
    dense = cv2.resize(dense, (width, height), interpolation=cv2.INTER_LINEAR)
    corrected = source.astype(np.float32) * dense[:, :, None]
    result = np.clip(corrected, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """Remove document shadows while keeping the source resolution."""
    return apply_gain_map(image, gain_map(image))
