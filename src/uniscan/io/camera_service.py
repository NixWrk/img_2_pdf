"""Camera abstraction for both live and burst capture modes."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

CancelCb = Callable[[], bool]
ProgressCb = Callable[[int, int], None]


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

    def open(self) -> None:
        """Open underlying VideoCapture."""
        self.release()
        self._capture = self.capture_factory(self.index, self.api_preference)

        if self.resolution is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        if self.target_fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Cannot open camera index {self.index}.")

    def release(self) -> None:
        """Release capture handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def set_index(self, index: int) -> None:
        """Switch camera index and re-open."""
        self.index = index
        self.open()

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """Switch camera resolution and re-open."""
        self.resolution = resolution
        self.open()

    def read_frame(self) -> np.ndarray | None:
        """Read one frame."""
        if self._capture is None:
            self.open()
        ok, frame = self._capture.read()
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
        if shots < 1:
            raise ValueError("shots must be >= 1")
        if delay_sec < 0:
            raise ValueError("delay_sec must be >= 0")

        if self._capture is None:
            self.open()

        frames: list[np.ndarray] = []
        for i in range(1, shots + 1):
            if cancel_cb and cancel_cb():
                raise RuntimeError("Cancelled by user.")

            for _ in range(max(0, warmup_reads)):
                self._capture.read()
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to capture frame {i}/{shots}.")

            frames.append(frame)
            if on_progress is not None:
                on_progress(i, shots)

            if i < shots and delay_sec > 0:
                wait_total = int(max(1, delay_sec / 0.1))
                for _ in range(wait_total):
                    if cancel_cb and cancel_cb():
                        raise RuntimeError("Cancelled by user.")
                    time.sleep(0.1)

        return frames

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
