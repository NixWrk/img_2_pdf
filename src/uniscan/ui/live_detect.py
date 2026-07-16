"""Background document-edge detector for the camera preview.

Design: a single worker thread consumes the most recent frame from a
single-slot inbox (older frames are dropped). The latest detected contour is
exposed with a timestamp; the UI thread reads it on every preview tick and
draws the overlay. EMA smoothing damps jitter between detections; large jumps
reset the EMA so the overlay snaps to a freshly-placed document.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    ScanAdapterError,
    scan_with_document_detector,
)

DEFAULT_LIVE_BACKEND = DETECTOR_BACKEND_OPENCV
LIVE_BACKEND_CHOICES: tuple[str, ...] = (
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_CV_HYBRID,
)


@dataclass(slots=True)
class DetectionState:
    contour: np.ndarray | None = None
    backend: str | None = None
    timestamp: float = 0.0
    last_frame_shape: tuple[int, int] | None = None
    error: str | None = None


class LiveContourDetector:
    """Single-slot, worker-threaded document-contour detector for live preview."""

    def __init__(
        self,
        *,
        backend: str = DEFAULT_LIVE_BACKEND,
        ttl_ms: float = 500.0,
        ema_alpha: float = 0.4,
        reset_jump_ratio: float = 0.3,
    ) -> None:
        self._backend = backend
        self._ttl_ms = float(ttl_ms)
        self._ema_alpha = float(ema_alpha)
        self._reset_jump_ratio = float(reset_jump_ratio)

        self._lock = threading.Lock()
        self._inbox: np.ndarray | None = None
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._state = DetectionState()
        self._thread: threading.Thread | None = None

    # Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="LiveContourDetector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            self._inbox = None
            self._state = DetectionState()

    # Public API ---------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    def set_backend(self, backend: str) -> None:
        if backend not in LIVE_BACKEND_CHOICES:
            raise ValueError(f"Unsupported live detector backend: {backend}")
        with self._lock:
            self._backend = backend
            self._state = DetectionState()  # reset EMA on backend change

    def submit(self, frame: np.ndarray) -> None:
        """Hand a new frame to the detector. Older queued frames are dropped."""
        if frame is None or frame.size == 0:
            return
        with self._lock:
            self._inbox = frame
        self._wake_event.set()

    def latest(self) -> tuple[np.ndarray | None, float]:
        """Return (contour, age_ms). Contour is None if no recent detection."""
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            if self._state.contour is None:
                return None, float("inf")
            age = now_ms - self._state.timestamp
            if age > self._ttl_ms:
                return None, age
            return self._state.contour.copy(), age

    def status(self) -> DetectionState:
        with self._lock:
            return DetectionState(
                contour=None if self._state.contour is None else self._state.contour.copy(),
                backend=self._state.backend,
                timestamp=self._state.timestamp,
                last_frame_shape=self._state.last_frame_shape,
                error=self._state.error,
            )

    # Worker loop --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.2)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                frame = self._inbox
                self._inbox = None
                backend = self._backend
            if frame is None:
                continue
            try:
                contour = self._detect(frame, backend)
                error_text: str | None = None
            except Exception as exc:  # detection is best-effort
                contour = None
                error_text = str(exc)

            now_ms = time.monotonic() * 1000.0
            with self._lock:
                if contour is not None:
                    smoothed = self._smooth(self._state.contour, contour, frame.shape[:2])
                    self._state = DetectionState(
                        contour=smoothed,
                        backend=backend,
                        timestamp=now_ms,
                        last_frame_shape=frame.shape[:2],
                        error=None,
                    )
                else:
                    self._state = DetectionState(
                        contour=None,
                        backend=backend,
                        timestamp=self._state.timestamp,
                        last_frame_shape=frame.shape[:2],
                        error=error_text,
                    )

    def _detect(self, frame: np.ndarray, backend: str) -> np.ndarray | None:
        try:
            scan_output = scan_with_document_detector(
                frame,
                enabled=True,
                backends=(backend,) if backend else DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
                proposal_only=True,
            )
        except ScanAdapterError:
            return None
        contour = scan_output.contour
        if contour is None:
            return None
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 4:
            return None
        return points[:4].copy()

    def _smooth(
        self,
        previous: np.ndarray | None,
        latest: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        latest = np.asarray(latest, dtype=np.float32).reshape(-1, 2)
        if previous is None or previous.shape != latest.shape:
            return latest.copy()
        height, width = frame_shape
        diagonal = float(np.hypot(width, height))
        if diagonal <= 1.0:
            return latest.copy()
        max_shift = float(np.max(np.linalg.norm(latest - previous, axis=1)))
        if max_shift / diagonal > self._reset_jump_ratio:
            return latest.copy()
        alpha = self._ema_alpha
        return (alpha * latest + (1.0 - alpha) * previous).astype(np.float32)
