"""Session layer for in-memory page management."""

from .capture_session import (
    CROP_STATE_APPLIED,
    CROP_STATE_NONE,
    CROP_STATE_PROPOSED,
    CaptureEntry,
    CaptureSession,
    CommittedPageProcessing,
    ProcessingRecipe,
)
from .autosave import (
    AutosaveSessionLock,
    SessionInUseError,
    UnsafeSessionLockError,
    acquire_autosave_lock,
    create_persistent_session,
    default_autosave_path,
    default_state_dir,
    discard_autosave,
    load_or_create_session,
)

__all__ = [
    "CROP_STATE_APPLIED",
    "CROP_STATE_NONE",
    "CROP_STATE_PROPOSED",
    "UnsafeSessionLockError",
    "CommittedPageProcessing",
    "ProcessingRecipe",
    "AutosaveSessionLock",
    "SessionInUseError",
    "acquire_autosave_lock",
    "CaptureEntry",
    "CaptureSession",
    "create_persistent_session",
    "default_autosave_path",
    "default_state_dir",
    "discard_autosave",
    "load_or_create_session",
]
