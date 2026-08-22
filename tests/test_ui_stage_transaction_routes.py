from __future__ import annotations

import numpy as np
import pytest

from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.session import CaptureSession, CommittedPageProcessing
from uniscan.storage import PageStore
from uniscan.ui.app import UnifiedScanApp
from uniscan.ui.stage_history import StageHistory
from uniscan.ui.stage_transaction import StageTransactionError


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _app_with_pages(tmp_path):
    app = object.__new__(UnifiedScanApp)
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    app.session.add_image(name="one", image=np.full((30, 40, 3), 180, dtype=np.uint8))
    app.session.add_image(name="two", image=np.full((30, 40, 3), 160, dtype=np.uint8))
    app.stage_history = StageHistory(tmp_path / "history")
    app._pending_stage_history_notice = None
    app.deskew_method_var = _Var("Hybrid (recommended)")
    app.processing_cache = None
    app._last_processing_cache_hits = ()
    app._selected_entry_indices = lambda: [0, 1]
    app._processing_request = lambda **_kwargs: PageProcessingRequest(deskew_method="hybrid")
    app.refresh_page_list = lambda **_kwargs: None
    app.status = None
    app._set_status = lambda status: setattr(app, "status", status)
    return app


def _real_candidate(app, entry, request):
    result = process_document_page(entry.original_image, request)
    committed = CommittedPageProcessing.from_result(request, result.diagnostics, result.image)
    return result.image, committed, result.diagnostics


def test_auto_deskew_processing_failure_on_second_page_keeps_real_batch_unchanged(
    tmp_path, monkeypatch
) -> None:
    app = _app_with_pages(tmp_path)
    before = [
        (entry.current_image, entry.revision, entry.committed_processing)
        for entry in app.session.entries
    ]
    calls = 0

    def fail_second(entry, request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second page processing failed")
        return _real_candidate(app, entry, request)

    monkeypatch.setattr(app, "_compute_processing_candidate", fail_second)
    with pytest.raises(RuntimeError, match="second page"):
        app.auto_deskew_selected()

    assert calls == 2
    for entry, (pixels, revision, committed) in zip(app.session.entries, before):
        np.testing.assert_array_equal(entry.current_image, pixels)
        assert entry.revision == revision
        assert entry.committed_processing is committed
    assert app.stage_history.undo_depth == 0
    assert list(app.stage_history.root.iterdir()) == []


def test_auto_deskew_store_failure_on_second_page_rolls_back_real_batch(
    tmp_path, monkeypatch
) -> None:
    app = _app_with_pages(tmp_path)
    before = [entry.current_image for entry in app.session.entries]
    original_replace = app.session.entries[1].store.replace_page_set
    failed = False

    second_id = app.session.entries[1].entry_id

    def fail_second_store(entry_id, *args, **kwargs):
        nonlocal failed
        if entry_id == second_id and kwargs.get("current_image") is not None and not failed:
            failed = True
            raise OSError("second page store failed")
        return original_replace(entry_id, *args, **kwargs)

    monkeypatch.setattr(app.session.entries[1].store, "replace_page_set", fail_second_store)
    with pytest.raises(StageTransactionError, match="all entries were restored"):
        app.auto_deskew_selected()

    assert failed is True
    for entry, pixels in zip(app.session.entries, before):
        np.testing.assert_array_equal(entry.current_image, pixels)
        assert entry.revision == 0
        assert entry.committed_processing is None
    assert app.stage_history.undo_depth == 0
    assert list(app.stage_history.root.iterdir()) == []


def test_auto_deskew_success_is_one_batch_history_record(tmp_path, monkeypatch) -> None:
    app = _app_with_pages(tmp_path)
    before = [entry.current_image.copy() for entry in app.session.entries]
    calls = 0

    def candidate(entry, request):
        nonlocal calls
        calls += 1
        pixels = np.full_like(entry.current_image, 70 + calls)
        committed = CommittedPageProcessing(
            recipe=request,
            diagnostics={"deskew": calls},
            current_fingerprint=CommittedPageProcessing.fingerprint_image(pixels),
        )
        diagnostics = type(
            "Diagnostics",
            (),
            {"deskew_angle_degrees": float(calls), "cache_hits": ("deskew",)},
        )()
        return pixels, committed, diagnostics

    monkeypatch.setattr(app, "_compute_processing_candidate", candidate)
    app.auto_deskew_selected()

    assert app.stage_history.undo_depth == 1
    record = app.stage_history._undo[-1]
    assert record.stage == "Deskew"
    assert record.entry_ids == tuple(entry.entry_id for entry in app.session.entries)
    app.stage_history.undo(app._stage_history_lookup)
    for entry, pixels in zip(app.session.entries, before):
        np.testing.assert_array_equal(entry.current_image, pixels)
    app.stage_history.redo(app._stage_history_lookup)
    assert all(entry.committed_processing is not None for entry in app.session.entries)
