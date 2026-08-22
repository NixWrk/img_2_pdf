from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.session import CaptureSession
from uniscan.storage import PageStore
from uniscan.ui.app import UnifiedScanApp
from uniscan.ui.stage_history import StageHistory


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Button:
    def __init__(self) -> None:
        self.state = None

    def configure(self, *, state) -> None:
        self.state = state


def test_stage_undo_redo_refreshes_and_respects_busy_and_editor_guards(tmp_path) -> None:
    app = object.__new__(UnifiedScanApp)
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = app.session.add_image(
        name="page",
        image=np.full((5, 7, 3), (11, 37, 203), dtype=np.uint8),
    )
    app.stage_history = StageHistory(tmp_path / "history")
    app._pending_stage_history_notice = None
    app.status_var = _Var()
    app.job_thread = None
    app.inline_editor_close_callback = None
    refreshed: list[tuple[str, ...]] = []
    app.refresh_page_list = lambda *, keep_entry_ids: refreshed.append(tuple(keep_entry_ids))

    before = entry.current_image.copy()
    original = entry.original_image.copy()
    contour = np.array([[0, 0], [6, 0], [6, 4], [0, 4]], np.float32)
    entry.detected_contour = contour.copy()
    entry.detected_backend = "detector"
    entry.committed_processing = {"generation": "before"}
    capture = app.stage_history.capture((entry,), action="Edit waves", stage="Waves")
    after = np.full_like(before, (241, 89, 7))
    entry.current_image = after
    entry.committed_processing = {"generation": "after"}
    app.stage_history.record(capture, (entry,))

    app.job_thread = object()
    app.undo_stage_edit()
    assert app.stage_history.can_undo
    assert "background work" in app.status_var.get()
    assert refreshed == []

    app.job_thread = None
    app.inline_editor_close_callback = lambda: None
    app.undo_stage_edit()
    assert app.stage_history.can_undo
    assert "open editor" in app.status_var.get()
    assert refreshed == []

    app.inline_editor_close_callback = None
    assert app._run_shortcut(app.undo_stage_edit) == "break"
    np.testing.assert_array_equal(entry.current_image, before)
    assert entry.committed_processing == {"generation": "before"}
    np.testing.assert_array_equal(entry.original_image, original)
    np.testing.assert_array_equal(entry.detected_contour, contour)
    assert entry.detected_backend == "detector"
    assert refreshed == [(entry.entry_id,)]
    assert app.stage_history.can_redo

    assert app._run_shortcut(app.redo_stage_edit) == "break"
    np.testing.assert_array_equal(entry.current_image, after)
    assert entry.committed_processing == {"generation": "after"}
    np.testing.assert_array_equal(entry.original_image, original)
    np.testing.assert_array_equal(entry.detected_contour, contour)
    assert entry.detected_backend == "detector"
    assert refreshed[-1] == (entry.entry_id,)
    assert app.stage_history.can_undo


def test_history_record_failure_does_not_turn_durable_edit_into_failure() -> None:
    app = object.__new__(UnifiedScanApp)
    app._pending_stage_history_notice = None
    app.status_var = _Var()

    class BrokenHistory:
        @staticmethod
        def record(_capture, _entries):
            raise OSError("snapshot disk unavailable")

    class Capture:
        discarded = False

        def discard(self):
            self.discarded = True
            raise OSError("snapshot cleanup unavailable")

    capture = Capture()
    app.stage_history = BrokenHistory()
    app._stage_history_record(capture, ())

    assert capture.discarded is True
    assert app._pending_stage_history_notice is not None
    app._set_status("Applied processing candidate.")
    assert app.status_var.get().startswith("Applied processing candidate.")
    assert "Undo is unavailable" in app.status_var.get()
    assert app._pending_stage_history_notice is None


def test_record_snapshot_failure_keeps_durable_edit_and_deletes_capture(
    tmp_path, monkeypatch
) -> None:
    app = object.__new__(UnifiedScanApp)
    app._pending_stage_history_notice = None
    app.status_var = _Var()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = app.session.add_image(name="page", image=np.full((5, 7, 3), 20, dtype=np.uint8))
    app.stage_history = StageHistory(tmp_path / "history")
    capture = app.stage_history.capture((entry,), action="Edit", stage="Cleanup")
    after = np.full_like(entry.current_image, 155)
    entry.current_image = after
    monkeypatch.setattr(
        app.stage_history,
        "record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("snapshot disk unavailable")),
    )

    app._stage_history_record(capture, (entry,))

    np.testing.assert_array_equal(entry.current_image, after)
    assert capture.closed is True
    assert app.stage_history.undo_depth == 0
    assert list(app.stage_history.root.iterdir()) == []
    app._set_status("Applied processing candidate.")
    assert "Undo is unavailable" in app.status_var.get()


def test_constructor_failure_after_history_init_closes_owned_root(tmp_path, monkeypatch) -> None:
    history = StageHistory(tmp_path / "history")
    root = history.root

    def fail_after_history(self) -> None:
        self.stage_history = history
        raise RuntimeError("failure after history init")

    monkeypatch.setattr(UnifiedScanApp, "_initialize", fail_after_history)
    with pytest.raises(RuntimeError, match="after history init"):
        UnifiedScanApp()

    assert not root.exists()
    history.close()


def test_stage_history_buttons_follow_their_own_stack() -> None:
    app = object.__new__(UnifiedScanApp)
    app.session = SimpleNamespace(entries=[], can_undo_deletion=False)
    app.stage_history = SimpleNamespace(can_undo=True, can_redo=False)
    app._selected_entry_indices = lambda: []
    app._deskew_restore_available = lambda: False
    app._lighting_restore_available = lambda: False
    app._cleanup_restore_available = lambda: False
    app._layout_restore_available = lambda: False
    app.undo_stage_button = _Button()
    app.redo_stage_button = _Button()
    app.pending_split_entry_id = None
    app.pending_split_ratio = None
    app.pending_split_revision = None

    app._update_page_action_states()
    assert app.undo_stage_button.state == "normal"
    assert app.redo_stage_button.state == "disabled"

    app.stage_history.can_undo = False
    app.stage_history.can_redo = True
    app._update_page_action_states()
    assert app.undo_stage_button.state == "disabled"
    assert app.redo_stage_button.state == "normal"
