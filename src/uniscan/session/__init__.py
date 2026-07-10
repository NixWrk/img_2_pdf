"""Session layer for in-memory page management."""

from .capture_session import CaptureEntry, CaptureSession
from .autosave import (
    create_persistent_session,
    default_autosave_path,
    default_state_dir,
    discard_autosave,
    load_or_create_session,
)

__all__ = [
    "CaptureEntry",
    "CaptureSession",
    "create_persistent_session",
    "default_autosave_path",
    "default_state_dir",
    "discard_autosave",
    "load_or_create_session",
]
