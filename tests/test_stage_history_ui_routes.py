from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
    capture = app.stage_history.capture((entry,), action="Edit waves", stage="Waves")
    after = np.full_like(before, (241, 89, 7))
    entry.current_image = after
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
    app.undo_stage_edit()
    np.testing.assert_array_equal(entry.current_image, before)
    assert refreshed == [(entry.entry_id,)]
    assert app.stage_history.can_redo

    app.redo_stage_edit()
    np.testing.assert_array_equal(entry.current_image, after)
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
