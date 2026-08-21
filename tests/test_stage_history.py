"""Regression tests for atomic stage-edit history."""

from __future__ import annotations

import numpy as np
import pytest

from uniscan.io import imwrite_unicode
from uniscan.session.capture_session import CROP_STATE_APPLIED, CaptureEntry
from uniscan.storage import PageStore
from uniscan.ui.stage_history import (
    StageHistory,
    StageHistoryBlocked,
    StageHistoryError,
    StageHistoryRollbackError,
)


def _entry(store, value=1):
    return CaptureEntry.from_image(
        name="page", image=np.full((3, 4, 3), value, np.uint8), store=store
    )


def _color_image(blue: int, green: int, red: int) -> np.ndarray:
    image = np.empty((3, 4, 3), np.uint8)
    image[:, :, 0] = blue
    image[:, :, 1] = green
    image[:, :, 2] = red
    return image


def _edit(history, entry, value, action="edit", perspective=False):
    capture = history.capture(
        (entry,), action=action, stage="Waves", include_perspective=perspective
    )
    entry.current_image = np.full((3, 4, 3), value, np.uint8)
    history.record(capture, (entry,))


def test_color_round_trip_and_monotonic_revision(tmp_path):
    store = PageStore(tmp_path / "pages")
    before = _color_image(11, 37, 203)
    after = _color_image(241, 89, 7)
    entry = CaptureEntry.from_image(name="color", image=before, store=store)
    history = StageHistory(tmp_path / "missing" / "history", max_records=3)
    capture = history.capture((entry,), action="color edit", stage="Cleanup")
    entry.current_image = after
    history.record(capture, (entry,))
    revision = entry.revision
    history.undo(lambda _: entry)
    assert np.array_equal(entry.current_image, before)
    history.redo(lambda _: entry)
    assert np.array_equal(entry.current_image, after)
    assert entry.revision == revision + 2
    owned_root = history.root
    history.close()
    assert not owned_root.exists()
    assert (tmp_path / "missing" / "history").is_dir()


def test_unique_eviction_and_undo_new_clears_redo_files(tmp_path):
    store = PageStore(tmp_path / "pages")
    entry = _entry(store)
    history = StageHistory(tmp_path, max_records=1)
    for value in (2, 3, 4):
        _edit(history, entry, value, action=str(value))
    paths = list(history.root.glob("*.png"))
    assert len(paths) == len({path.name for path in paths})
    assert len(paths) == 2
    redo_record = history.undo(lambda _: entry)
    redo_paths = {item.current_path for item in (*redo_record.before, *redo_record.after)}
    assert all(path.exists() for path in redo_paths)
    _edit(history, entry, 8, action="new")
    assert not history.can_redo
    assert all(not path.exists() for path in redo_paths)
    assert len(list(history.root.glob("*.png"))) == 2
    history.close()


def test_perspective_roundtrip_restores_original_and_review_metadata(tmp_path):
    store = PageStore(tmp_path / "pages")
    entry = _entry(store)
    before_original = entry.original_image.copy()
    before_current = _color_image(3, 17, 91)
    entry.current_image = before_current
    before_contour = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], np.float32)
    entry.detected_contour = before_contour.copy()
    entry.detected_backend = "test"
    entry.crop_state = CROP_STATE_APPLIED
    entry.needs_review = True
    entry.review_reasons = ("reason",)
    entry.dewarp_control_points = ((0.0, 0.1), (1.0, 0.1))
    entry.dewarp_control_curves = ((0.5, ((0.0, 0.1), (1.0, 0.1))),)
    entry.committed_processing = {"generation": "before"}
    history = StageHistory(tmp_path, max_records=2)
    capture = history.capture(
        (entry,), action="geometry", stage="Perspective", include_perspective=True
    )
    after_original = _color_image(13, 71, 199)
    after_current = _color_image(211, 43, 5)
    after_contour = np.array([[1, 0], [3, 1], [2, 2], [0, 1]], np.float32)
    entry.original_image = after_original
    entry.current_image = after_current
    entry.detected_contour = after_contour.copy()
    entry.detected_backend = "changed"
    entry.crop_state = CROP_STATE_APPLIED
    entry.needs_review = False
    entry.review_reasons = ()
    entry.dewarp_control_points = ((0.0, -0.2), (1.0, -0.2))
    entry.dewarp_control_curves = ((0.5, ((0.0, -0.2), (1.0, -0.2))),)
    entry.committed_processing = {"generation": "after"}
    history.record(capture, (entry,))
    history.undo(lambda _: entry)
    assert np.array_equal(entry.original_image, before_original)
    assert np.array_equal(entry.current_image, before_current)
    assert np.array_equal(entry.detected_contour, before_contour)
    assert entry.detected_backend == "test"
    assert entry.crop_state == CROP_STATE_APPLIED
    assert entry.needs_review and entry.review_reasons == ("reason",)
    assert entry.dewarp_control_points == ((0.0, 0.1), (1.0, 0.1))
    assert entry.dewarp_control_curves == ((0.5, ((0.0, 0.1), (1.0, 0.1))),)
    assert entry.committed_processing == {"generation": "before"}
    history.redo(lambda _: entry)
    assert np.array_equal(entry.original_image, after_original)
    assert np.array_equal(entry.current_image, after_current)
    assert np.array_equal(entry.detected_contour, after_contour)
    assert entry.detected_backend == "changed"
    assert not entry.needs_review and entry.review_reasons == ()
    assert entry.dewarp_control_points == ((0.0, -0.2), (1.0, -0.2))
    assert entry.dewarp_control_curves == ((0.5, ((0.0, -0.2), (1.0, -0.2))),)
    assert entry.committed_processing == {"generation": "after"}
    history.close()


@pytest.mark.parametrize("side", ["source", "target"])
def test_corrupted_valid_snapshot_is_blocked_before_writes(tmp_path, side):
    store = PageStore(tmp_path / "pages")
    entry = _entry(store)
    history = StageHistory(tmp_path)
    _edit(history, entry, 2)
    record = history._undo[-1]
    snapshot = record.after[0] if side == "source" else record.before[0]
    assert imwrite_unicode(snapshot.current_path, _color_image(9, 19, 29))
    revision = entry.revision
    current = entry.current_image.copy()
    with pytest.raises(StageHistoryBlocked):
        history.undo(lambda _: entry)
    assert entry.revision == revision
    assert np.array_equal(entry.current_image, current)
    assert history.can_undo and not history.can_redo
    history.close()


def test_batch_second_write_failure_rolls_back_and_keeps_stack(tmp_path, monkeypatch):
    store = PageStore(tmp_path / "pages")
    first, second = _entry(store, 1), _entry(store, 2)
    history = StageHistory(tmp_path, max_records=2)
    capture = history.capture((first, second), action="batch", stage="Deskew")
    first.current_image = np.full((3, 4, 3), 7, np.uint8)
    second.current_image = np.full((3, 4, 3), 8, np.uint8)
    history.record(capture, (first, second))
    before = (
        first.current_image.copy(),
        second.current_image.copy(),
        first.revision,
        second.revision,
    )
    original = history._restore_entry
    calls = {"count": 0}

    def fail_once(entry, *args):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected")
        return original(entry, *args)

    monkeypatch.setattr(history, "_restore_entry", fail_once)
    with pytest.raises(StageHistoryError):
        history.undo(lambda item_id: {first.entry_id: first, second.entry_id: second}.get(item_id))
    assert np.array_equal(first.current_image, before[0])
    assert np.array_equal(second.current_image, before[1])
    assert (first.revision, second.revision) == before[2:]
    assert history.can_undo and not history.can_redo
    history.close()


def test_preflight_missing_entry_and_duplicate_capture_do_not_write(tmp_path):
    store = PageStore(tmp_path / "pages")
    entry = _entry(store)
    history = StageHistory(tmp_path)
    with pytest.raises(ValueError):
        history.capture((entry, entry), action="x", stage="x")
    _edit(history, entry, 2)
    revision, image = entry.revision, entry.current_image.copy()
    with pytest.raises(StageHistoryBlocked):
        history.undo(lambda _: None)
    assert entry.revision == revision and np.array_equal(entry.current_image, image)
    history.close()
    history.close()


def test_partial_capture_and_mismatched_record_can_be_discarded_without_leaks(tmp_path):
    class BrokenCopy:
        def __deepcopy__(self, _memo):
            raise RuntimeError("cannot copy metadata")

    class FakeEntry:
        def __init__(self, entry_id, committed_processing=None):
            self.entry_id = entry_id
            self.revision = 0
            self.current_image = _color_image(1, 2, 3)
            self.committed_processing = committed_processing
            self.dewarp_control_points = None
            self.dewarp_control_curves = None

    history = StageHistory(tmp_path)
    with pytest.raises(RuntimeError, match="cannot copy metadata"):
        history.capture(
            (FakeEntry("first"), FakeEntry("second", BrokenCopy())),
            action="capture",
            stage="Cleanup",
        )
    assert list(history.root.iterdir()) == []

    entry = FakeEntry("retry")
    capture = history.capture((entry,), action="capture", stage="Cleanup")
    with pytest.raises(StageHistoryError, match="does not match"):
        history.record(capture, ())
    capture.discard()
    assert list(history.root.iterdir()) == []
    history.close()


def test_incomplete_batch_rollback_has_distinct_error_and_keeps_stacks(tmp_path, monkeypatch):
    store = PageStore(tmp_path / "pages")
    first, second = _entry(store, 1), _entry(store, 2)
    history = StageHistory(tmp_path)
    capture = history.capture((first, second), action="batch", stage="Deskew")
    first.current_image = np.full((3, 4, 3), 7, np.uint8)
    second.current_image = np.full((3, 4, 3), 8, np.uint8)
    history.record(capture, (first, second))
    original = history._restore_entry
    calls = {"count": 0}

    def fail_apply_and_rollback(entry, *args):
        calls["count"] += 1
        if calls["count"] == 2:
            original(entry, *args)
            raise OSError("failure after second-page write")
        if calls["count"] == 3:
            raise OSError("second-page rollback failed")
        return original(entry, *args)

    monkeypatch.setattr(history, "_restore_entry", fail_apply_and_rollback)
    with pytest.raises(StageHistoryRollbackError):
        history.undo(lambda item_id: {first.entry_id: first, second.entry_id: second}.get(item_id))
    assert history.can_undo and not history.can_redo
    history.close()
