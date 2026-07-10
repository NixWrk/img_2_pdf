from __future__ import annotations

import numpy as np
import pytest

from uniscan.io.camera_service import CameraService


class FakeCapture:
    def __init__(self, frames: list[np.ndarray] | None = None, *, opened: bool = True) -> None:
        self.frames = list(frames or [])
        self.opened = opened
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True

    def set(self, prop_id: int, value: float) -> bool:
        self.settings.append((prop_id, value))
        return True


def _frame(value: int) -> np.ndarray:
    return np.full((8, 12, 3), value, dtype=np.uint8)


def test_open_applies_settings_and_release() -> None:
    capture = FakeCapture([_frame(1)])
    calls: list[tuple[int, int | None]] = []

    def factory(index: int, api: int | None):
        calls.append((index, api))
        return capture

    service = CameraService(
        index=3,
        resolution=(1920, 1080),
        target_fps=30,
        api_preference=700,
        capture_factory=factory,
    )
    service.open()

    assert calls == [(3, 700)]
    assert len(capture.settings) == 3
    assert service.read_frame() is not None
    service.release()
    assert capture.released is True


def test_open_rejects_unavailable_camera() -> None:
    service = CameraService(capture_factory=lambda _index, _api: FakeCapture(opened=False))
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        service.open()


def test_capture_burst_progress_and_validation() -> None:
    capture = FakeCapture([_frame(value) for value in range(6)])
    service = CameraService(capture_factory=lambda _index, _api: capture)
    progress: list[tuple[int, int]] = []

    frames = service.capture_burst(
        shots=2,
        delay_sec=0,
        warmup_reads=1,
        on_progress=lambda current, total: progress.append((current, total)),
    )

    assert [int(frame.mean()) for frame in frames] == [1, 3]
    assert progress == [(1, 2), (2, 2)]
    with pytest.raises(ValueError, match="shots"):
        service.capture_burst(shots=0, delay_sec=0)
    with pytest.raises(ValueError, match="delay_sec"):
        service.capture_burst(shots=1, delay_sec=-1)


def test_capture_burst_cancellation_and_failed_frame() -> None:
    service = CameraService(capture_factory=lambda _index, _api: FakeCapture([_frame(1)]))
    with pytest.raises(RuntimeError, match="Cancelled"):
        service.capture_burst(shots=1, delay_sec=0, cancel_cb=lambda: True)

    failed = CameraService(capture_factory=lambda _index, _api: FakeCapture())
    with pytest.raises(RuntimeError, match="Failed to capture"):
        failed.capture_burst(shots=1, delay_sec=0, warmup_reads=0)


def test_get_available_device_indices_uses_factory() -> None:
    captures: list[FakeCapture] = []

    def factory(index: int, _api: int | None):
        capture = FakeCapture(opened=index in {1, 3})
        captures.append(capture)
        return capture

    assert CameraService.get_available_device_indices(max_indices=4, capture_factory=factory) == [
        1,
        3,
    ]
    assert all(capture.released for capture in captures)
