"""DocScanner-L page rectification through a pinned ONNX grid model.

The graph predicts the same full 288x288 backward grid as the official
PyTorch runner.  UniScan applies that grid to the original page directly, so
the network never downsamples the text pixels that are ultimately exported.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from uniscan.model_assets import file_sha256, model_file_identity

MODEL_FILENAME = "DocScanner-L-grid-opset17.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_ENV_VAR = "UNISCAN_DOCSCANNER_MODEL"
MODEL_SIZE = 34_100_351
MODEL_SHA256 = "9fdebcb4067afb09d66b6637f3fd1036ba7952bbf0656778d44c2bd1c2c067f4"
INPUT_SIZE = (288, 288)

_session_lock = threading.Lock()
_session_cache: dict[str, object] = {}


def model_path() -> Path:
    """Configured or external-release model location (which may not exist)."""
    override = os.environ.get(MODEL_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return MODEL_DIR / MODEL_FILENAME


def verify_model(path: Path) -> Path:
    """Accept only the graph reproduced and validated from official weights."""
    if not path.is_file():
        raise FileNotFoundError(f"DocScanner-L model is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != MODEL_SIZE:
        raise RuntimeError(
            f"DocScanner-L model size mismatch for {path}: "
            f"expected {MODEL_SIZE}, got {actual_size}."
        )
    actual_hash = file_sha256(path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(
            f"DocScanner-L model SHA-256 mismatch for {path}: "
            f"expected {MODEL_SHA256}, got {actual_hash}."
        )
    return path


def is_available() -> bool:
    """True when the graph exists and ONNX Runtime can execute it."""
    if not model_path().is_file():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def model_identity() -> str:
    """Content identity used by persistent processing-cache keys."""
    return "docscanner-l:" + model_file_identity(model_path())


def _load_session():
    path = verify_model(model_path())
    key = str(path.resolve())
    with _session_lock:
        session = _session_cache.get(key)
        if session is not None:
            return session
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("DocScanner-L rectification requires onnxruntime.") from exc
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _session_cache[key] = session
        return session


def reset_session_cache() -> None:
    with _session_lock:
        _session_cache.clear()


def _prepare_input(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preserve the official runner's order: normalize the original first,
    # then resize its floating-point pixels. Resizing uint8 first quantizes the
    # model input enough to move the predicted full-page grid visibly.
    resized = cv2.resize(rgb.astype(np.float64) / 255.0, INPUT_SIZE)
    return np.transpose(resized, (2, 0, 1))[None].astype(np.float32)


def predict_grid(image: np.ndarray) -> np.ndarray:
    """Return DocScanner-L's normalized `(1, 2, 288, 288)` backward grid."""
    if image is None or image.size == 0 or image.ndim < 2:
        raise ValueError("DocScanner-L requires a non-empty image.")
    session = _load_session()
    outputs = session.run(None, {session.get_inputs()[0].name: _prepare_input(image)})
    if not outputs:
        raise RuntimeError("DocScanner-L returned no output tensor.")
    grid = np.asarray(outputs[0], dtype=np.float32)
    if grid.shape != (1, 2, INPUT_SIZE[1], INPUT_SIZE[0]):
        raise ValueError(f"Unexpected DocScanner-L grid shape: {grid.shape}")
    return grid


def grid_to_backward_map(
    grid: np.ndarray,
    *,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the official OpenCV resize/blur grid postprocessing."""
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.shape != (2, INPUT_SIZE[1], INPUT_SIZE[0]):
        raise ValueError(f"Unexpected DocScanner-L grid shape: {np.asarray(grid).shape}")
    width, height = size
    if width < 2 or height < 2:
        raise ValueError("Target size must be at least 2x2 pixels.")
    normalized_x = cv2.blur(cv2.resize(array[0], (width, height)), (3, 3))
    normalized_y = cv2.blur(cv2.resize(array[1], (width, height)), (3, 3))
    map_x = (normalized_x + 1.0) * 0.5 * float(width - 1)
    map_y = (normalized_y + 1.0) * 0.5 * float(height - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def _bilinear_sample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """NumPy equivalent of PyTorch grid_sample(align_corners=True)."""
    source = np.asarray(image)
    scalar = source.ndim == 2
    if scalar:
        source = source[:, :, None]
    height, width, channels = source.shape
    source_float = source.astype(np.float32) / 255.0
    output = np.empty((height, width, channels), dtype=np.uint8)

    # Four full-resolution float32 neighbourhood tensors can exceed several
    # hundred MiB on a phone photo. Row chunks preserve PyTorch's exact
    # zero-padded bilinear math while bounding production peak memory.
    for start in range(0, height, 256):
        stop = min(start + 256, height)
        sample_x = map_x[start:stop]
        sample_y = map_y[start:stop]
        x0 = np.floor(sample_x).astype(np.intp)
        y0 = np.floor(sample_y).astype(np.intp)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = (sample_x - x0).astype(np.float32)
        wy = (sample_y - y0).astype(np.float32)

        def gather(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            values = source_float[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
            return values * valid[:, :, None]

        result = (
            gather(x0, y0) * ((1.0 - wx) * (1.0 - wy))[:, :, None]
            + gather(x1, y0) * (wx * (1.0 - wy))[:, :, None]
            + gather(x0, y1) * ((1.0 - wx) * wy)[:, :, None]
            + gather(x1, y1) * (wx * wy)[:, :, None]
        )
        output[start:stop] = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
    return output[:, :, 0] if scalar else output


def sample_backward_map(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Apply an arbitrary composed map with the official PyTorch sampling semantics."""
    if map_x.shape != map_y.shape:
        raise ValueError("DocScanner-L backward maps must have equal shapes.")
    return _bilinear_sample(image, map_x, map_y)


def dewarp(image: np.ndarray) -> np.ndarray:
    """Rectify a page while sampling its original full-resolution pixels."""
    grid = predict_grid(image)
    height, width = image.shape[:2]
    map_x, map_y = grid_to_backward_map(grid, size=(width, height))
    return _bilinear_sample(image, map_x, map_y)
