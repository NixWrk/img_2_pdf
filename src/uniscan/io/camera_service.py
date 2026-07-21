"""Camera abstraction for live preview and capture.

Frame acquisition is built around a background grabber thread
(:meth:`CameraService.start_stream`): it continuously drains the driver so the
newest frame is always decoded and ready.  Consumers never block on camera I/O:

- ``latest_frame()`` returns the most recent frame immediately (preview path);
- ``next_fresh_frame()`` waits for the first frame captured *after* the call,
  which is what a shutter press needs — no stale driver-buffer frames and no
  blind warm-up reads.

The legacy synchronous path (``read_frame``/``iter_burst`` without a running
stream) is kept for the CLI ``doctor`` command and headless use.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

CancelCb = Callable[[], bool]
ProgressCb = Callable[[int, int], None]
MAX_BURST_SHOTS = 20

# Fresh-frame waits fail after this long; covers cameras as slow as ~0.3 fps.
FRESH_FRAME_TIMEOUT_SEC = 5.0
# Consecutive failed reads before the stream declares the device lost.
_STREAM_FAILURE_LIMIT = 20
_STREAM_FAILURE_SLEEP_SEC = 0.05
_STREAM_LOST_MESSAGE = "Camera stopped returning frames (disconnected or in use elsewhere)."
_FPS_WINDOW = 30

# Sizes worth measuring when detecting what a device can actually deliver.
# Drivers collapse unsupported requests onto their nearest real mode, so this
# short list covers the usual sensor maximum down to VGA without probing
# near-duplicates.
MODE_PROBE_CANDIDATES: tuple[tuple[int, int], ...] = (
    (3264, 2448),
    (2592, 1944),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (640, 480),
)
MODE_PROBE_FRAMES = 20
MODE_PROBE_BUDGET_SEC = 0.6
_MODE_PROBE_DISCARD = 3
# Below this rate the preview is a slideshow and a shot lags behind the
# shutter press by a visible fraction of a second.
REALTIME_FPS_THRESHOLD = 10.0


class CaptureDevice(Protocol):
    """Small cv2.VideoCapture-compatible surface used by CameraService."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...


CaptureFactory = Callable[[int, int | None], CaptureDevice]


def _opencv_capture_factory(index: int, api_preference: int | None) -> CaptureDevice:
    if api_preference is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, api_preference)


# Windows enumerates video capture interfaces under this device-interface
# class, in the order DirectShow and Media Foundation hand them out, which is
# the order OpenCV device indices follow.
_KSCATEGORY_VIDEO_CAMERA = "{e5323777-f976-4f5b-9b55-b94699c46e44}"
_DEVICE_CLASSES_KEY = r"SYSTEM\CurrentControlSet\Control\DeviceClasses"
_DEVICE_ENUM_KEY = r"SYSTEM\CurrentControlSet\Enum"


def _clean_device_name(raw: str) -> str:
    """Strip the ``@oem58.inf,%PID_085C_DD%;`` prefix INF-supplied names carry."""
    if raw.startswith("@") and ";" in raw:
        raw = raw.split(";", 1)[1]
    return raw.strip()


def list_camera_device_names() -> list[str]:
    """Friendly names of the video capture devices, in enumeration order.

    Returns an empty list off Windows or when the registry cannot be read;
    callers then fall back to plain device indices.
    """
    if platform.system() != "Windows":
        return []
    try:
        import winreg
    except ImportError:
        return []

    names: list[str] = []
    seen: set[str] = set()
    try:
        class_path = f"{_DEVICE_CLASSES_KEY}\\{_KSCATEGORY_VIDEO_CAMERA}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_path) as class_key:
            subkey_count = winreg.QueryInfoKey(class_key)[0]
            for position in range(subkey_count):
                subkey_name = winreg.EnumKey(class_key, position)
                try:
                    with winreg.OpenKey(class_key, subkey_name) as device_key:
                        instance = winreg.QueryValueEx(device_key, "DeviceInstance")[0]
                except OSError:
                    continue  # "Properties" and other non-device entries
                if instance in seen:
                    continue  # one device can expose several interfaces
                seen.add(instance)
                names.append(_device_friendly_name(winreg, instance))
    except OSError:
        return []
    return names


def _device_friendly_name(winreg, instance: str) -> str:
    """Registry name for a device instance, falling back to its instance id."""
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, f"{_DEVICE_ENUM_KEY}\\{instance}"
        ) as enum_key:
            for value_name in ("FriendlyName", "DeviceDesc"):
                try:
                    return _clean_device_name(winreg.QueryValueEx(enum_key, value_name)[0])
                except OSError:
                    continue
    except OSError:
        pass
    return instance


def default_api_preference() -> int | None:
    """Preferred OpenCV camera backend for this platform.

    Windows: Media Foundation. It streams the same webcam at full frame rate
    where DirectShow's uncompressed YUY2 path collapses (measured 30 fps vs
    10 fps at 720p), and `uniscan/__init__` disables its slow hardware
    transforms so it opens in a fraction of a second. DirectShow remains the
    automatic fallback in :func:`fallback_api_preferences`.
    """
    if platform.system() == "Windows":
        return cv2.CAP_MSMF
    return None


def fallback_api_preferences(api_preference: int | None) -> tuple[int | None, ...]:
    """Backends to try, in order, when ``api_preference`` fails to open."""
    if platform.system() != "Windows":
        return ()
    ordered = [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]
    return tuple(candidate for candidate in ordered if candidate != api_preference)


@dataclass(slots=True, frozen=True)
class FrameInfo:
    """One published stream frame with its monotonic sequence number."""

    frame: np.ndarray
    seq: int
    timestamp: float


@dataclass(slots=True, frozen=True)
class CameraMode:
    """A capture size the device grants, with its measured frame rate."""

    requested: tuple[int, int]
    granted: tuple[int, int]
    fps: float

    @property
    def pixels(self) -> int:
        return self.granted[0] * self.granted[1]

    @property
    def is_realtime(self) -> bool:
        """True when shots come off the live stream without a visible lag."""
        return self.fps >= REALTIME_FPS_THRESHOLD

    @property
    def label(self) -> str:
        """Menu text: the size plus what it actually costs the user."""
        size = f"{self.granted[0]}x{self.granted[1]}"
        if self.is_realtime:
            return f"{size} - {self.fps:.0f} fps"
        return f"{size} - {self.fps:.1f} fps (slow)"


def best_realtime_mode(modes: Sequence[CameraMode]) -> CameraMode | None:
    """Largest mode that still streams in real time, else the largest one."""
    if not modes:
        return None
    realtime = [mode for mode in modes if mode.is_realtime]
    pool = realtime or list(modes)
    return max(pool, key=lambda mode: mode.pixels)


class CameraService:
    """cv2.VideoCapture wrapper with an optional background frame stream."""

    MAX_BURST_SHOTS = MAX_BURST_SHOTS

    def __init__(
        self,
        *,
        index: int = 0,
        resolution: tuple[int, int] | None = None,
        target_fps: int | None = None,
        api_preference: int | None = None,
        prefer_mjpg: bool = True,
        capture_factory: CaptureFactory = _opencv_capture_factory,
    ) -> None:
        self.index = index
        self.resolution = resolution
        self.target_fps = target_fps
        self.api_preference = default_api_preference() if api_preference is None else api_preference
        self.prefer_mjpg = prefer_mjpg
        self.capture_factory = capture_factory
        self._capture: CaptureDevice | None = None
        self._capture_lock = threading.RLock()
        self._effective_resolution: tuple[int, int] | None = None

        # Stream state. _frame_cond guards every field below it; the grabber
        # thread never takes _capture_lock, so joining it under _capture_lock
        # cannot deadlock.
        self._frame_cond = threading.Condition()
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._latest: FrameInfo | None = None
        self._frame_seq = 0
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._stream_error: str | None = None

    # Device lifecycle ----------------------------------------------------

    def open(self) -> None:
        """Open the underlying VideoCapture, restarting the stream if it ran.

        The preferred backend is tried first; if it cannot open the device the
        remaining platform backends are tried before giving up.
        """
        restart_stream = self.is_streaming()
        self._stop_stream()
        with self._capture_lock:
            self._release_locked()
            candidates = (self.api_preference, *fallback_api_preferences(self.api_preference))
            failure: Exception | None = None
            for candidate in candidates:
                try:
                    if self._open_with_backend(candidate):
                        self.api_preference = candidate
                        break
                except Exception as exc:  # try the next backend, report the first
                    failure = failure or exc
                    self._release_locked()
            else:
                if failure is not None:
                    raise failure
                raise RuntimeError(f"Cannot open camera index {self.index}.")
        if restart_stream:
            self.start_stream()

    def _open_with_backend(self, api_preference: int | None) -> bool:
        """Open and configure one backend; True when the device is usable."""
        capture = self.capture_factory(self.index, api_preference)
        self._capture = capture
        # Order matters on DirectShow: FOURCC must be applied *after* the frame
        # size, and setting CAP_PROP_FPS afterwards silently reverts the format
        # back to uncompressed YUY2 (measured: MJPG 19 fps -> YUY2 10 fps).
        if self.resolution is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        if self.target_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        if self.prefer_mjpg:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Shallow driver queue keeps the non-stream read path fresher.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self._release_locked()
            return False
        self._effective_resolution = self._query_resolution(capture)
        return True

    @staticmethod
    def _query_resolution(capture: CaptureDevice) -> tuple[int, int] | None:
        getter = getattr(capture, "get", None)
        if getter is None:
            return None
        try:
            width = int(getter(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(getter(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return (width, height)

    @property
    def effective_resolution(self) -> tuple[int, int] | None:
        """Resolution the driver actually granted, when it reports one."""
        return self._effective_resolution

    def _release_locked(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def release(self) -> None:
        """Stop the stream and release the capture handle."""
        # Join politely first; if a driver read is stuck the handle release
        # below is what actually breaks it loose, and the thread then exits.
        self._stop_stream()
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

    # Streaming -----------------------------------------------------------

    def start_stream(self) -> None:
        """Start the background grabber; opens the device on demand."""
        with self._capture_lock:
            if self._stream_thread is not None and self._stream_thread.is_alive():
                return
            if self._capture is None:
                self.open()
            capture = self._capture
            if capture is None:  # Defensive: open() either publishes or raises.
                raise RuntimeError(f"Cannot open camera index {self.index}.")
            self._stream_stop = threading.Event()
            with self._frame_cond:
                self._stream_error = None
                self._frame_times.clear()
            thread = threading.Thread(
                target=self._stream_loop,
                args=(capture, self._stream_stop),
                name=f"CameraStream-{self.index}",
                daemon=True,
            )
            self._stream_thread = thread
            thread.start()

    def _stop_stream(self, *, join_timeout: float = 1.0) -> None:
        self._stream_stop.set()
        thread = self._stream_thread
        self._stream_thread = None
        with self._frame_cond:
            self._frame_cond.notify_all()
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=join_timeout)
        with self._frame_cond:
            self._latest = None
            self._frame_times.clear()

    def stop_stream(self) -> None:
        """Stop the background grabber, keeping the device open."""
        self._stop_stream()

    def is_streaming(self) -> bool:
        thread = self._stream_thread
        return thread is not None and thread.is_alive()

    @property
    def stream_error(self) -> str | None:
        """Error text once the stream has declared the device lost."""
        with self._frame_cond:
            return self._stream_error

    def _stream_loop(self, capture: CaptureDevice, stop_event: threading.Event) -> None:
        failures = 0
        while not stop_event.is_set():
            ok, frame = capture.read()
            if stop_event.is_set():
                break
            if not ok or frame is None:
                failures += 1
                if failures >= _STREAM_FAILURE_LIMIT:
                    with self._frame_cond:
                        self._stream_error = _STREAM_LOST_MESSAGE
                        self._frame_cond.notify_all()
                    break
                time.sleep(_STREAM_FAILURE_SLEEP_SEC)
                continue
            failures = 0
            # perf_counter: time.monotonic has 15.6 ms granularity on Windows
            # before Python 3.13, which is too coarse for frame intervals.
            now = time.perf_counter()
            with self._frame_cond:
                self._frame_seq += 1
                self._latest = FrameInfo(frame=frame, seq=self._frame_seq, timestamp=now)
                self._frame_times.append(now)
                self._frame_cond.notify_all()

    def latest_frame_info(self) -> FrameInfo | None:
        """Newest streamed frame with metadata, without blocking."""
        with self._frame_cond:
            return self._latest

    def latest_frame(self) -> np.ndarray | None:
        """Newest streamed frame, without blocking."""
        info = self.latest_frame_info()
        return None if info is None else info.frame

    @property
    def measured_fps(self) -> float | None:
        """Rolling capture rate of the stream, or None while warming up."""
        with self._frame_cond:
            if len(self._frame_times) < 2:
                return None
            span = self._frame_times[-1] - self._frame_times[0]
            if span <= 0:
                return None
            return (len(self._frame_times) - 1) / span

    def next_fresh_frame(
        self,
        *,
        timeout_sec: float = FRESH_FRAME_TIMEOUT_SEC,
        cancel_cb: CancelCb | None = None,
    ) -> np.ndarray:
        """Wait for the first frame captured after this call (shutter press).

        Raises RuntimeError on cancellation, stream loss, or timeout.
        """
        deadline = time.monotonic() + timeout_sec
        with self._frame_cond:
            target_seq = self._frame_seq + 1
            while True:
                latest = self._latest
                if latest is not None and latest.seq >= target_seq:
                    return latest.frame
                if self._stream_error is not None:
                    raise RuntimeError(self._stream_error)
                if cancel_cb is not None and cancel_cb():
                    raise RuntimeError("Cancelled by user.")
                if not self.is_streaming():
                    raise RuntimeError("Camera stream is not running.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for a fresh camera frame.")
                # Short slices keep cancellation responsive even if no frame
                # ever arrives to notify the condition.
                self._frame_cond.wait(timeout=min(0.1, remaining))

    # Single-shot and burst capture ---------------------------------------

    def capture_still(
        self,
        *,
        timeout_sec: float = FRESH_FRAME_TIMEOUT_SEC,
        cancel_cb: CancelCb | None = None,
        warmup_reads: int = 4,
    ) -> np.ndarray | None:
        """One still frame at the configured resolution.

        With a running stream this is the newest streamed frame, taken with no
        wait and no device reconfiguration. Without one it is a direct read
        preceded by warm-up reads that flush stale driver-buffer frames.
        """
        if self.is_streaming():
            frame = self.latest_frame()
            if frame is not None:
                return frame
            return self.next_fresh_frame(timeout_sec=timeout_sec, cancel_cb=cancel_cb)
        with self._capture_lock:
            if self._capture is None:
                self.open()
            capture = self._capture
            if capture is None:  # Defensive: open() either publishes a handle or raises.
                return None
            for _ in range(max(0, warmup_reads)):
                capture.read()
            ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return frame

    def read_frame(self) -> np.ndarray | None:
        """Read one frame: newest stream frame, or a direct blocking read."""
        if self.is_streaming():
            return self.latest_frame()
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
        first_frame: np.ndarray | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield burst frames one at a time so callers can stage them to disk.

        Preview and capture share one resolution, so a running stream supplies
        every shot directly: the first is the frame already on screen when the
        shutter was pressed, the rest are guaranteed-fresh frames and
        ``warmup_reads`` is ignored. Pass ``first_frame`` to pin shot one to a
        frame the caller grabbed at the exact moment of the press, before this
        generator started. Without a stream it falls back to warm-up reads that
        flush stale frames from the driver queue.
        """
        if shots < 1 or shots > MAX_BURST_SHOTS:
            raise ValueError(f"shots must be between 1 and {MAX_BURST_SHOTS}")
        if delay_sec < 0:
            raise ValueError("delay_sec must be >= 0")

        if not self.is_streaming():
            with self._capture_lock:
                if self._capture is None:
                    self.open()

        for i in range(1, shots + 1):
            if cancel_cb and cancel_cb():
                raise RuntimeError("Cancelled by user.")

            if i == 1 and first_frame is not None:
                frame = first_frame
            elif self.is_streaming():
                # The first shot is the frame already on screen, so pressing
                # the shutter captures exactly what the user was looking at;
                # later shots must be new frames, not that same one.
                frame = self.latest_frame() if i == 1 else None
                if frame is None:
                    frame = self.next_fresh_frame(cancel_cb=cancel_cb)
            else:
                # Hold the lock across one shot only. release() can then
                # interrupt the burst between shots without racing a
                # VideoCapture.read call.
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

    @classmethod
    def probe_modes(
        cls,
        *,
        index: int = 0,
        candidates: Sequence[tuple[int, int]] = MODE_PROBE_CANDIDATES,
        sample_frames: int = MODE_PROBE_FRAMES,
        api_preference: int | None = None,
        capture_factory: CaptureFactory = _opencv_capture_factory,
        cancel_cb: CancelCb | None = None,
        on_progress: ProgressCb | None = None,
    ) -> list[CameraMode]:
        """Measure what the device really delivers for each requested size.

        Drivers silently substitute sizes and some modes run at a few frames
        per second, which decides whether a shot can come straight off the
        live stream. Duplicate granted sizes are probed once. Returned modes
        are ordered from largest to smallest.
        """
        modes: dict[tuple[int, int], CameraMode] = {}
        total = len(candidates)
        for position, requested in enumerate(candidates, start=1):
            if cancel_cb is not None and cancel_cb():
                break
            if on_progress is not None:
                on_progress(position, total)
            service = cls(
                index=index,
                resolution=requested,
                api_preference=api_preference,
                capture_factory=capture_factory,
            )
            try:
                service.open()
                granted = service.effective_resolution or requested
                if granted in modes:
                    continue
                fps = service._sample_read_rate(sample_frames)
                if fps is None:
                    continue
                modes[granted] = CameraMode(requested=requested, granted=granted, fps=fps)
            except Exception:
                continue  # unsupported mode: leave it out of the menu
            finally:
                service.release()
        return sorted(modes.values(), key=lambda mode: mode.pixels, reverse=True)

    def _sample_read_rate(self, sample_frames: int) -> float | None:
        """Time direct reads and return the sustained frame rate.

        Frames buffered during device start-up arrive back to back and would
        overstate the rate, so the first reads are discarded. Sampling is
        time-boxed: a slow mode yields one frame and is still classified
        correctly instead of holding the probe for seconds.
        """
        with self._capture_lock:
            capture = self._capture
            if capture is None:
                return None
            for _ in range(_MODE_PROBE_DISCARD):
                if not capture.read()[0]:
                    return None
            start = time.perf_counter()
            captured = 0
            while captured < max(1, sample_frames):
                if not capture.read()[0]:
                    break
                captured += 1
                if time.perf_counter() - start >= MODE_PROBE_BUDGET_SEC:
                    break
            elapsed = time.perf_counter() - start
        if captured == 0 or elapsed <= 0:
            return None
        return captured / elapsed
