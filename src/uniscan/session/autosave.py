"""Persistent session helpers used by the desktop application."""

from __future__ import annotations

import os
from pathlib import Path

from uniscan.storage import PageStore

from .capture_session import CaptureSession


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
    session.close(preserve=False)
    manifest.unlink(missing_ok=True)
