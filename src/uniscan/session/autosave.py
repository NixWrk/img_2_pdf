"""Persistent session helpers used by the desktop application."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO

from uniscan.storage import PageStore

from .capture_session import CaptureSession


class SessionInUseError(RuntimeError):
    """Raised when another UniScan process owns the shared autosave session."""


class UnsafeSessionLockError(RuntimeError):
    """Raised when the autosave lock path is not a safe regular file."""


class AutosaveSessionLock:
    """Non-blocking, process-scoped lock for one autosave state directory."""

    def __init__(self, path: Path, stream: BinaryIO) -> None:
        self.path = Path(path)
        self._stream: BinaryIO | None = stream

    @staticmethod
    def _validate_path(path: Path) -> os.stat_result:
        info = path.lstat()
        is_junction = getattr(path, "is_junction", lambda: False)()
        attributes = int(getattr(info, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        is_reparse_point = bool(attributes & reparse_flag)
        if (
            path.is_symlink()
            or is_junction
            or is_reparse_point
            or not stat.S_ISREG(info.st_mode)
            or int(getattr(info, "st_nlink", 1)) != 1
        ):
            raise UnsafeSessionLockError(f"Unsafe autosave lock path: {path}")
        return info

    @classmethod
    def acquire(cls, path: Path) -> "AutosaveSessionLock":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            cls._validate_path(path)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            opened = os.fstat(stream.fileno())
            linked = cls._validate_path(path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(getattr(opened, "st_nlink", 1)) != 1
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            ):
                raise UnsafeSessionLockError(f"Autosave lock path changed while opening: {path}")
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
        except Exception:
            stream.close()
            raise
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise SessionInUseError(
                f"Another UniScan process is using the autosave session: {path.parent}"
            ) from exc
        return cls(path, stream)

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "AutosaveSessionLock":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


def default_state_dir() -> Path:
    override = os.environ.get("UNISCAN_STATE_DIR")
    if override:
        return Path(override).expanduser()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "UniScan"
    return Path.home() / ".uniscan"


def default_autosave_path() -> Path:
    return default_state_dir() / "autosave.json"


def acquire_autosave_lock(manifest_path: Path | None = None) -> AutosaveSessionLock:
    """Acquire the single-writer lock associated with an autosave manifest."""
    manifest = Path(manifest_path) if manifest_path is not None else default_autosave_path()
    return AutosaveSessionLock.acquire(manifest.with_name("autosave.lock"))


def create_persistent_session(state_dir: Path | None = None) -> CaptureSession:
    root = Path(state_dir) if state_dir is not None else default_state_dir()
    store = PageStore(root_dir=root / "sessions", keep_on_close=True)
    return CaptureSession(store=store)


def load_or_create_session(manifest_path: Path | None = None) -> tuple[CaptureSession, bool]:
    manifest = Path(manifest_path) if manifest_path is not None else default_autosave_path()
    if manifest.is_file():
        return CaptureSession.restore_manifest(
            manifest,
            allowed_sessions_root=manifest.parent / "sessions",
        ), True
    return create_persistent_session(manifest.parent), False


def discard_autosave(session: CaptureSession, manifest_path: Path | None = None) -> None:
    manifest = Path(manifest_path) if manifest_path is not None else default_autosave_path()
    # Remove the durable reference first.  A crash afterwards can leave only
    # harmless orphaned assets, never a manifest pointing at deleted files.
    manifest.unlink(missing_ok=True)
    session.close(preserve=False)
