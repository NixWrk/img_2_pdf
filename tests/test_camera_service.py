from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

from uniscan.io import camera_service
from uniscan.io.camera_service import CameraMode, CameraService, FrameInfo, best_realtime_mode


class FakeCapture:
    def __init__(self, frames: list[np.ndarray] | None = None, *, opened: bool = True) -> None:
        self.frames = list(frames or [])
        self.opened = opened
        self.released = False
        self.settings: list[tuple[int, float]] = []
        self.props: dict[int, float] = {}

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
        self.props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)


class EndlessCapture(FakeCapture):
    """Streams a new frame on every read, stamped with the read counter."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.read_gate = threading.Event()
        self.read_gate.set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_gate.wait(timeout=1.0)
        if self.released:
            return False, None
        time.sleep(0.001)  # pace the synthetic stream like a (fast) device
        self.reads += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[0, 0] = (self.reads & 255, (self.reads >> 8) & 255, (self.reads >> 16) & 255)
        return True, frame

    @staticmethod
    def stamp(frame: np.ndarray) -> int:
        low, mid, high = (int(value) for value in frame[0, 0])
        return low | (mid << 8) | (high << 16)


def _frame(value: int) -> np.ndarray:
    return np.full((8, 12, 3), value, dtype=np.uint8)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met in time")


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
    applied = [prop for prop, _value in capture.settings]
    assert set(applied) == {
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
    }
    # DirectShow only honours MJPG when it is requested after the frame size,
    # and a later CAP_PROP_FPS silently reverts the format to YUY2.
    assert applied.index(cv2.CAP_PROP_FOURCC) > applied.index(cv2.CAP_PROP_FRAME_HEIGHT)
    assert applied.index(cv2.CAP_PROP_FOURCC) > applied.index(cv2.CAP_PROP_FPS)
    assert service.effective_resolution == (1920, 1080)
    assert service.read_frame() is not None
    service.release()
    assert capture.released is True


def test_default_backend_prefers_msmf_with_dshow_fallback(monkeypatch) -> None:
    monkeypatch.setattr(camera_service.platform, "system", lambda: "Windows")
    assert camera_service.default_api_preference() == cv2.CAP_MSMF
    assert camera_service.fallback_api_preferences(cv2.CAP_MSMF) == (cv2.CAP_DSHOW, None)

    monkeypatch.setattr(camera_service.platform, "system", lambda: "Linux")
    assert camera_service.default_api_preference() is None
    assert camera_service.fallback_api_preferences(None) == ()


def test_open_falls_back_to_the_next_backend(monkeypatch) -> None:
    monkeypatch.setattr(camera_service.platform, "system", lambda: "Windows")
    attempted: list[int | None] = []

    def factory(_index: int, api: int | None):
        attempted.append(api)
        # Only DirectShow works on this (simulated) machine.
        return FakeCapture([_frame(1)], opened=api == cv2.CAP_DSHOW)

    service = CameraService(capture_factory=factory)
    service.open()

    assert attempted == [cv2.CAP_MSMF, cv2.CAP_DSHOW]
    assert service.api_preference == cv2.CAP_DSHOW
    service.release()


def test_open_without_mjpg_skips_fourcc() -> None:
    capture = FakeCapture()
    service = CameraService(prefer_mjpg=False, capture_factory=lambda _i, _a: capture)
    service.open()
    assert cv2.CAP_PROP_FOURCC not in {prop for prop, _value in capture.settings}


def test_open_rejects_unavailable_camera() -> None:
    service = CameraService(capture_factory=lambda _index, _api: FakeCapture(opened=False))
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        service.open()


def test_msmf_hardware_transforms_are_disabled_before_cv2_loads() -> None:
    # Media Foundation otherwise spends ~28 s per open negotiating hardware
    # transforms; OpenCV reads this only while importing, so uniscan/__init__
    # must set it and cv2 must already be imported by the time tests run.
    import uniscan

    assert "cv2" in sys.modules
    if sys.platform == "win32":
        assert os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] == "0"
    assert uniscan.__version__


def test_open_releases_handle_when_configuration_raises() -> None:
    class FailingCapture(FakeCapture):
        def set(self, prop_id: int, value: float) -> bool:
            raise OSError("driver failed")

    capture = FailingCapture()
    service = CameraService(
        resolution=(640, 480),
        capture_factory=lambda _index, _api: capture,
    )

    with pytest.raises(OSError, match="driver failed"):
        service.open()

    assert capture.released is True


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
    with pytest.raises(ValueError, match="between 1 and 20"):
        service.capture_burst(shots=21, delay_sec=0)
    with pytest.raises(ValueError, match="delay_sec"):
        service.capture_burst(shots=1, delay_sec=-1)


def test_capture_burst_cancellation_and_failed_frame() -> None:
    service = CameraService(capture_factory=lambda _index, _api: FakeCapture([_frame(1)]))
    with pytest.raises(RuntimeError, match="Cancelled"):
        service.capture_burst(shots=1, delay_sec=0, cancel_cb=lambda: True)

    failed = CameraService(capture_factory=lambda _index, _api: FakeCapture())
    with pytest.raises(RuntimeError, match="Failed to capture"):
        failed.capture_burst(shots=1, delay_sec=0, warmup_reads=0)


def test_iter_burst_yields_frames_incrementally() -> None:
    capture = FakeCapture([_frame(1), _frame(2)])
    service = CameraService(capture_factory=lambda _index, _api: capture)

    frames = service.iter_burst(shots=2, delay_sec=0, warmup_reads=0)
    first_index, first = next(frames)

    assert first_index == 1
    assert int(first.mean()) == 1
    assert len(capture.frames) == 1
    assert [(index, int(frame.mean())) for index, frame in frames] == [(2, 2)]


def test_releasing_incremental_burst_does_not_reopen_camera() -> None:
    captures = [FakeCapture([_frame(1), _frame(2)])]
    factory_calls = 0

    def factory(_index: int, _api: int | None) -> FakeCapture:
        nonlocal factory_calls
        factory_calls += 1
        return captures[0]

    service = CameraService(capture_factory=factory)
    frames = service.iter_burst(shots=2, delay_sec=0, warmup_reads=0)
    assert next(frames)[0] == 1

    service.release()

    with pytest.raises(RuntimeError, match="closed during burst"):
        next(frames)
    assert factory_calls == 1


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


# Streaming ---------------------------------------------------------------


def test_stream_publishes_latest_frame_and_fps() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        assert service.is_streaming() is True

        _wait_until(lambda: service.latest_frame_info() is not None)
        info = service.latest_frame_info()
        assert isinstance(info, FrameInfo)
        assert info.seq >= 1
        assert service.latest_frame() is not None

        _wait_until(lambda: service.measured_fps is not None)
        assert service.measured_fps > 0
    finally:
        service.release()
    assert service.is_streaming() is False
    assert service.latest_frame() is None
    assert capture.released is True


def test_read_frame_uses_stream_when_running() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame() is not None)
        reads_before = capture.reads
        assert service.read_frame() is not None
        # The caller thread must not touch the device directly.
        assert capture.reads >= reads_before
    finally:
        service.release()


def test_next_fresh_frame_waits_for_post_call_frame() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame_info() is not None)

        capture.read_gate.clear()  # pause the grabber mid-stream
        time.sleep(0.02)
        stale = service.latest_frame_info()
        assert stale is not None

        results: list[FrameInfo | None] = []

        def wait_fresh() -> None:
            frame = service.next_fresh_frame(timeout_sec=2.0)
            results.append(service.latest_frame_info() if frame is not None else None)

        waiter = threading.Thread(target=wait_fresh, daemon=True)
        waiter.start()
        time.sleep(0.05)
        assert not results  # still waiting: no frame captured after the call yet
        capture.read_gate.set()
        waiter.join(timeout=2.0)

        assert results and results[0] is not None
        assert results[0].seq > stale.seq
    finally:
        service.release()


def test_next_fresh_frame_cancellation_and_timeout() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame_info() is not None)
        capture.read_gate.clear()
        time.sleep(0.02)  # let any in-flight read publish before measuring

        with pytest.raises(RuntimeError, match="Cancelled"):
            service.next_fresh_frame(timeout_sec=2.0, cancel_cb=lambda: True)
        with pytest.raises(RuntimeError, match="Timed out"):
            service.next_fresh_frame(timeout_sec=0.05)
    finally:
        capture.read_gate.set()
        service.release()


def test_iter_burst_uses_fresh_stream_frames_without_warmup_reads() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame_info() is not None)

        frames = list(service.iter_burst(shots=3, delay_sec=0, warmup_reads=4))

        assert [index for index, _frame in frames] == [1, 2, 3]
        values = [EndlessCapture.stamp(frame) for _index, frame in frames]
        # Fresh frames are strictly newer captures, never the same stale buffer.
        assert values == sorted(set(values))
        assert len(values) == 3
    finally:
        service.release()


def _counting_factory(captures: list[EndlessCapture]):
    def factory(_index: int, _api: int | None) -> EndlessCapture:
        capture = EndlessCapture()
        captures.append(capture)
        return capture

    return factory


def test_capture_still_takes_the_on_screen_frame_without_reconfiguring() -> None:
    captures: list[EndlessCapture] = []
    service = CameraService(resolution=(1920, 1080), capture_factory=_counting_factory(captures))
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame() is not None)
        on_screen = service.latest_frame_info()

        frame = service.capture_still(timeout_sec=2.0)

        assert frame is not None
        # The shot is the frame the user was looking at: no device re-open,
        # no waiting, and the stream keeps running at the same resolution.
        assert EndlessCapture.stamp(frame) == EndlessCapture.stamp(on_screen.frame)
        assert len(captures) == 1
        assert service.resolution == (1920, 1080)
        assert service.is_streaming() is True
    finally:
        service.release()


def test_capture_still_without_stream_uses_warmup_reads() -> None:
    capture = FakeCapture([_frame(value) for value in range(6)])
    service = CameraService(capture_factory=lambda _i, _a: capture)

    frame = service.capture_still(warmup_reads=2)

    assert frame is not None
    assert int(frame.mean()) == 2  # two stale frames flushed first
    service.release()


def test_iter_burst_pins_the_first_shot_to_the_press_time_frame() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame() is not None)
        pressed = service.latest_frame()
        _wait_until(lambda: EndlessCapture.stamp(service.latest_frame()) > stamp_of(pressed))

        frames = list(service.iter_burst(shots=2, delay_sec=0, first_frame=pressed))

        # Shot one is the caller's frame even though newer ones have arrived.
        assert EndlessCapture.stamp(frames[0][1]) == stamp_of(pressed)
        assert EndlessCapture.stamp(frames[1][1]) > stamp_of(pressed)
    finally:
        service.release()


def stamp_of(frame: np.ndarray) -> int:
    return EndlessCapture.stamp(frame)


def test_iter_burst_keeps_one_resolution_and_starts_on_the_live_frame() -> None:
    captures: list[EndlessCapture] = []
    service = CameraService(resolution=(1920, 1080), capture_factory=_counting_factory(captures))
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame() is not None)
        on_screen = service.latest_frame_info()

        frames = list(service.iter_burst(shots=3, delay_sec=0))

        assert [index for index, _frame in frames] == [1, 2, 3]
        stamps = [EndlessCapture.stamp(frame) for _index, frame in frames]
        # First shot is the on-screen frame; the rest are newer captures.
        assert stamps[0] == EndlessCapture.stamp(on_screen.frame)
        assert stamps == sorted(set(stamps))
        assert len(captures) == 1  # the device is never reconfigured
        assert service.resolution == (1920, 1080)
    finally:
        service.release()


# Capture-mode detection --------------------------------------------------


def test_probe_modes_measures_rate_and_deduplicates_granted_sizes() -> None:
    class ModeCapture(FakeCapture):
        """Grants a fixed size and paces reads to a target frame rate."""

        def __init__(self, granted: tuple[int, int], interval: float) -> None:
            super().__init__()
            self.props[cv2.CAP_PROP_FRAME_WIDTH] = granted[0]
            self.props[cv2.CAP_PROP_FRAME_HEIGHT] = granted[1]
            self.interval = interval

        def set(self, prop_id: int, value: float) -> bool:
            self.settings.append((prop_id, value))
            return True  # requested size is ignored, like a real driver

        def read(self):
            time.sleep(self.interval)
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

    granted_for = {
        (3264, 2448): ((2304, 1536), 0.15),  # ~7 fps: below the real-time bar
        (2592, 1944): ((2304, 1536), 0.15),  # same granted size: probed once
        (1920, 1080): ((1920, 1080), 0.005),
        (1600, 1200): ((1600, 896), 0.005),
        (1280, 720): ((1280, 720), 0.005),
        (640, 480): ((640, 480), 0.005),
    }
    requested_order: list[tuple[int, int]] = []
    pending: list[tuple[int, int]] = []

    def factory(_index: int, _api: int | None):
        requested = pending[-1]
        granted, interval = granted_for[requested]
        return ModeCapture(granted, interval)

    original_init = CameraService.__init__

    def tracking_init(self, **kwargs):
        pending.append(kwargs["resolution"])
        requested_order.append(kwargs["resolution"])
        original_init(self, **kwargs)

    CameraService.__init__ = tracking_init
    try:
        progress: list[tuple[int, int]] = []
        modes = CameraService.probe_modes(
            capture_factory=factory,
            sample_frames=6,
            on_progress=lambda done, total: progress.append((done, total)),
        )
    finally:
        CameraService.__init__ = original_init

    granted_sizes = [mode.granted for mode in modes]
    assert granted_sizes == [(2304, 1536), (1920, 1080), (1600, 896), (1280, 720), (640, 480)]
    assert progress[0] == (1, 6) and progress[-1] == (6, 6)

    slow, fast = modes[0], modes[1]
    assert slow.is_realtime is False
    assert "slow" in slow.label
    assert fast.is_realtime is True
    assert fast.label.startswith("1920x1080")
    assert best_realtime_mode(modes) is fast


def test_best_realtime_mode_falls_back_to_the_largest_slow_mode() -> None:
    slow_large = CameraMode(requested=(3264, 2448), granted=(2304, 1536), fps=2.0)
    slow_small = CameraMode(requested=(640, 480), granted=(640, 480), fps=3.0)

    assert best_realtime_mode([slow_small, slow_large]) is slow_large
    assert best_realtime_mode([]) is None


def test_stream_reports_device_loss() -> None:
    capture = EndlessCapture()
    service = CameraService(capture_factory=lambda _i, _a: capture)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame_info() is not None)
        capture.released = True  # simulate unplug: reads start failing

        _wait_until(lambda: service.stream_error is not None, timeout=5.0)
        assert "Camera stopped returning frames" in (service.stream_error or "")
        with pytest.raises(RuntimeError, match="stopped returning frames"):
            service.next_fresh_frame(timeout_sec=1.0)
    finally:
        service.release()


def test_reopen_preserves_streaming_state() -> None:
    captures: list[EndlessCapture] = []

    def factory(_index: int, _api: int | None) -> EndlessCapture:
        capture = EndlessCapture()
        captures.append(capture)
        return capture

    service = CameraService(resolution=(640, 480), capture_factory=factory)
    try:
        service.start_stream()
        _wait_until(lambda: service.latest_frame() is not None)

        service.set_resolution((1280, 720))

        assert len(captures) == 2
        assert captures[0].released is True
        assert service.is_streaming() is True
        _wait_until(lambda: service.latest_frame() is not None)
    finally:
        service.release()
