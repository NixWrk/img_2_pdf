import time

import cv2
import numpy as np

from uniscan.core.scanner_adapter import DETECTOR_BACKEND_OPENCV
from uniscan.ui.live_detect import LiveContourDetector


def _make_document_frame(width: int = 480, height: int = 360) -> np.ndarray:
    """Synthetic camera frame with a white document quad on dark background."""
    frame = np.full((height, width, 3), 20, dtype=np.uint8)
    pts = np.array(
        [[80, 60], [width - 80, 50], [width - 60, height - 70], [60, height - 50]],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], color=(245, 245, 245))
    return frame


def _wait_for_detection(
    detector: LiveContourDetector, timeout_sec: float = 2.0
) -> tuple[np.ndarray | None, float]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        contour, age_ms = detector.latest()
        if contour is not None:
            return contour, age_ms
        time.sleep(0.02)
    return detector.latest()


def test_submit_then_latest_returns_recent_contour() -> None:
    detector = LiveContourDetector(backend=DETECTOR_BACKEND_OPENCV)
    detector.start()
    try:
        frame = _make_document_frame()
        detector.submit(frame)
        contour, age_ms = _wait_for_detection(detector)
        assert contour is not None
        assert contour.shape == (4, 2)
        assert age_ms < 600
    finally:
        detector.stop()


def test_latest_returns_none_when_no_frame_submitted() -> None:
    detector = LiveContourDetector(backend=DETECTOR_BACKEND_OPENCV)
    detector.start()
    try:
        contour, age_ms = detector.latest()
        assert contour is None
        assert age_ms == float("inf")
    finally:
        detector.stop()


def test_set_backend_rejects_unknown_value() -> None:
    detector = LiveContourDetector()
    try:
        detector.set_backend("nonexistent_backend")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_smoothing_blends_two_close_detections() -> None:
    detector = LiveContourDetector(backend=DETECTOR_BACKEND_OPENCV)
    first = np.array([[10, 10], [100, 10], [100, 60], [10, 60]], dtype=np.float32)
    second = np.array([[12, 12], [102, 12], [102, 62], [12, 62]], dtype=np.float32)
    blended = detector._smooth(first, second, frame_shape=(360, 480))
    assert blended.shape == (4, 2)
    # EMA with alpha=0.4: blended_x = 0.4*second + 0.6*first → between the two
    assert 10.5 < float(blended[0, 0]) < 11.7


def test_smoothing_resets_on_large_jump() -> None:
    detector = LiveContourDetector(backend=DETECTOR_BACKEND_OPENCV)
    first = np.array([[10, 10], [100, 10], [100, 60], [10, 60]], dtype=np.float32)
    second = np.array([[300, 300], [400, 300], [400, 350], [300, 350]], dtype=np.float32)
    result = detector._smooth(first, second, frame_shape=(360, 480))
    # On large jump the EMA should snap to the latest, not blend.
    assert np.allclose(result, second)
