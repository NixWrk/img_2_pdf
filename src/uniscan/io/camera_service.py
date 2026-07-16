"""Camera abstraction for both live and burst capture modes."""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import cv2
import numpy as np

CancelCb = Callable[[], bool]
ProgressCb = Callable[[int, int], None]
MAX_BURST_SHOTS = 20


class CaptureDevice(Protocol):
    """Small cv2.VideoCapture-compatible surface used by CameraService."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...


CaptureFactory = Callable[[int, int | None], CaptureDevice]


def _opencv_capture_factory(index: int, api_preference: int | None) -> CaptureDevice:
    if api_preference is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, api_preference)


def default_api_preference() -> int | None:
    """Select platform-specific OpenCV camera backend."""
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return None


class CameraService:
    """Thin wrapper around cv2.VideoCapture."""

    MAX_BURST_SHOTS = MAX_BURST_SHOTS

    def __init__(
        self,
        *,
        index: int = 0,
        resolution: tuple[int, int] | None = None,
        target_fps: int | None = None,
        api_preference: int | None = None,
        capture_factory: CaptureFactory = _opencv_capture_factory,
    ) -> None:
        self.index = index
        self.resolution = resolution
        self.target_fps = target_fps
        self.api_preference = default_api_preference() if api_preference is None else api_preference
        self.capture_factory = capture_factory
        self._capture: CaptureDevice | None = None
        self._capture_lock = threading.RLock()

    def open(self) -> None:
        """Open underlying VideoCapture."""
        with self._capture_lock:
            self._release_locked()
            capture = self.capture_factory(self.index, self.api_preference)
            self._capture = capture

            try:
                if self.resolution is not None:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                if self.target_fps is not None:
                    capture.set(cv2.CAP_PROP_FPS, self.target_fps)
                opened = capture.isOpened()
            except Exception:
                self._release_locked()
                raise

            if not opened:
                self._release_locked()
                raise RuntimeError(f"Cannot open camera index {self.index}.")

    def _release_locked(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def release(self) -> None:
        """Release capture handle."""
        with self._capture_lock:
            self._release_locked()

    def set_index(self, index: int) -> None:
        """Switch camera index and re-open."""
        with self._capture_lock:
            self.index = index
            self.open()

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """Switch camera resolution and re-open."""
        with self._capture_lock:
            self.resolution = resolution
            self.open()

    def read_frame(self) -> np.ndarray | None:
        """Read one frame."""
        with self._capture_lock:
            if self._capture is None:
                self.open()
            capture = self._capture
            if capture is None:  # Defensive: open() either publishes a handle or raises.
                return None
            ok, frame = capture.read()
            if not ok:
                return None
            return frame

    def capture_burst(
        self,
        *,
        shots: int,
        delay_sec: float,
        warmup_reads: int = 4,
        cancel_cb: CancelCb | None = None,
        on_progress: ProgressCb | None = None,
    ) -> list[np.ndarray]:
        """Capture burst of frames with optional delay and cancellation."""
        return [
            frame
            for _index, frame in self.iter_burst(
                shots=shots,
                delay_sec=delay_sec,
                warmup_reads=warmup_reads,
                cancel_cb=cancel_cb,
                on_progress=on_progress,
            )
        ]

    def iter_burst(
        self,
        *,
        shots: int,
        delay_sec: float,
        warmup_reads: int = 4,
        cancel_cb: CancelCb | None = None,
        on_progress: ProgressCb | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield burst frames one at a time so callers can stage them to disk."""
        if shots < 1 or shots > MAX_BURST_SHOTS:
            raise ValueError(f"shots must be between 1 and {MAX_BURST_SHOTS}")
        if delay_sec < 0:
            raise ValueError("delay_sec must be >= 0")

        with self._capture_lock:
            if self._capture is None:
                self.open()

        for i in range(1, shots + 1):
            if cancel_cb and cancel_cb():
                raise RuntimeError("Cancelled by user.")

            # Hold the lock across one shot only. release() can then interrupt
            # the burst between shots without racing a VideoCapture.read call.
            with self._capture_lock:
                capture = self._capture
                if capture is None:
                    raise RuntimeError("Camera was closed during burst capture.")
                for _ in range(max(0, warmup_reads)):
                    capture.read()
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to capture frame {i}/{shots}.")

            captured_at = time.monotonic()
            if on_progress is not None:
                on_progress(i, shots)
            yield i, frame

            if i < shots and delay_sec > 0:
                remaining = delay_sec - (time.monotonic() - captured_at)
                while remaining > 0:
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("Cancelled by user.")
                    time.sleep(min(0.1, remaining))
                    remaining = delay_sec - (time.monotonic() - captured_at)

    @classmethod
    def get_available_device_indices(
        cls,
        *,
        max_indices: int = 10,
        api_preference: int | None = None,
        capture_factory: CaptureFactory = _opencv_capture_factory,
    ) -> list[int]:
        """Probe camera indices and return opened ones."""
        pref = default_api_preference() if api_preference is None else api_preference
        found: list[int] = []
        for index in range(max_indices):
            cap = capture_factory(index, pref)
            if cap.isOpened():
                found.append(index)
            cap.release()
        return found
