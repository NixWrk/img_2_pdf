"""UVDoc grid-based page rectification through ONNX Runtime.

UVDoc predicts a coarse backward sampling grid for the whole page instead of
tracking text lines, so it also straightens pages with little or no text. The
network runs on a fixed small input; the grid it returns is resolution
independent and is upsampled to remap the original full-resolution image.

Weights are the MIT-licensed UVDoc model exported to ONNX (Apache-2.0); see
``src/uniscan/models/README.md``. The runtime is optional: everything here
degrades to "unavailable" when onnxruntime or the model file is missing.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "UVDoc_grid.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_ENV_VAR = "UNISCAN_UVDOC_MODEL"

# The exported graph has a fixed input; (width, height) in pixels.
UVDOC_INPUT_SIZE = (496, 720)
# Output grid axes, verified against the exported model: (2, rows, columns).
_GRID_CHANNELS = 2

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


def _load_session():
    """Return a cached inference session for the configured model."""
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"UVDoc model is missing: {path}. Set {MODEL_ENV_VAR} to a UVDoc ONNX file."
        )
    key = str(path.resolve())
    with _session_lock:
        session = _session_cache.get(key)
        if session is not None:
            return session
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("UVDoc rectification requires onnxruntime.") from exc
        # Sessions are expensive to build and safe to share across threads.
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        _session_cache[key] = session
        return session


def reset_session_cache() -> None:
    """Drop cached sessions so a changed model path is picked up."""
    with _session_lock:
        _session_cache.clear()


def _prepare_input(image: np.ndarray) -> np.ndarray:
    """Resize to the network input and return a normalized RGB NCHW tensor."""
    resized = cv2.resize(image, UVDOC_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[None]


def grid_to_backward_map(
    grid: np.ndarray,
    *,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a normalized UVDoc grid into full-resolution remap tables.

    ``grid`` is (2, rows, columns) with x and y in [-1, 1]; ``size`` is the
    (width, height) of the image being sampled, which is also the output size.
    """
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != _GRID_CHANNELS:
        raise ValueError(f"Unexpected UVDoc grid shape: {np.asarray(grid).shape}")

    width, height = size
    if width < 2 or height < 2:
        raise ValueError("Target size must be at least 2x2 pixels.")

    rows, columns = array.shape[1], array.shape[2]
    map_x = _upsample_channel(array[0], width=width, height=height, rows=rows, columns=columns)
    map_y = _upsample_channel(array[1], width=width, height=height, rows=rows, columns=columns)
    map_x = (map_x + 1.0) * 0.5 * float(width - 1)
    map_y = (map_y + 1.0) * 0.5 * float(height - 1)
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(
        map_y, dtype=np.float32
    )


def _upsample_channel(
    channel: np.ndarray,
    *,
    width: int,
    height: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Bilinear upsample of one grid channel with the corners pinned.

    UVDoc is trained with corner-aligned sampling: the first and last grid
    values must land on the first and last output pixels. OpenCV's resize
    assumes half-pixel centres and extrapolates past the grid, and its remap
    quantizes coordinates to 1/32 px, so both bend the map near the page
    border. Interpolating here keeps the mapping exact, and doing it one
    channel and one axis at a time keeps the temporaries small.
    """
    grid_x = np.linspace(0.0, columns - 1, width, dtype=np.float32)
    left = np.floor(grid_x).astype(np.intp)
    right = np.minimum(left + 1, columns - 1)
    weight_x = (grid_x - left).astype(np.float32)
    # (rows, width): still tiny, one row per grid sample.
    along_x = channel[:, left] * (1.0 - weight_x) + channel[:, right] * weight_x

    grid_y = np.linspace(0.0, rows - 1, height, dtype=np.float32)
    top = np.floor(grid_y).astype(np.intp)
    bottom = np.minimum(top + 1, rows - 1)
    weight_y = (grid_y - top).astype(np.float32)[:, None]
    return along_x[top] * (1.0 - weight_y) + along_x[bottom] * weight_y


def predict_grid(image: np.ndarray) -> np.ndarray:
    """Run UVDoc and return its raw (2, rows, columns) sampling grid."""
    if image is None or image.size == 0 or image.ndim < 2:
        raise ValueError("UVDoc requires a non-empty image.")
    session = _load_session()
    inputs = {session.get_inputs()[0].name: _prepare_input(image)}
    outputs = session.run(None, inputs)
    if not outputs:
        raise RuntimeError("UVDoc returned no output tensor.")
    return np.asarray(outputs[0], dtype=np.float32)


def dewarp(image: np.ndarray) -> np.ndarray:
    """Rectify a page with UVDoc, keeping the source resolution."""
    grid = predict_grid(image)
    height, width = image.shape[:2]
    map_x, map_y = grid_to_backward_map(grid, size=(width, height))
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
