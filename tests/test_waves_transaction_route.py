from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.core.processing import PageProcessingRequest
from uniscan.session import CaptureSession, CommittedPageProcessing
from uniscan.storage import PageStore
from uniscan.ui.app import UnifiedScanApp
from uniscan.ui.stage_transaction import (
    StageEditTransaction,
    StaleStageRevisionError,
)


def _app_entry(tmp_path):
    app = object.__new__(UnifiedScanApp)
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="page", image=np.full((20, 30, 3), 180, dtype=np.uint8))
    app.session = session
    app._last_processing_cache_hits = ()
    return app, entry


def _candidate(image: np.ndarray):
    return (
        image,
        CommittedPageProcessing(
            recipe=SimpleNamespace(),
            diagnostics={"wave": "candidate"},
            current_fingerprint=CommittedPageProcessing.fingerprint_image(image),
        ),
        SimpleNamespace(max_displacement_px=4.0),
    )


def test_waves_processing_failure_after_draft_controls_keeps_real_entry_unchanged(
    tmp_path, monkeypatch
) -> None:
    app, entry = _app_entry(tmp_path)
    original_pixels = entry.current_image
    transaction = StageEditTransaction.begin((entry,))
    monkeypatch.setattr(
        app,
        "_compute_processing_candidate",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("wave processing failed")),
    )

    with pytest.raises(RuntimeError, match="wave processing"):
        app._commit_dewarp_candidate(
            entry,
            transaction,
            (
                (0.25, ((0.0, 0.1), (0.5, 0.1), (1.0, 0.1))),
                (0.5, ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
            ),
            PageProcessingRequest(),
        )

    transaction.discard()
    np.testing.assert_array_equal(entry.current_image, original_pixels)
    assert entry.dewarp_control_curves is None
    assert entry.dewarp_control_points is None
    assert entry.committed_processing is None
    assert entry.revision == 0


def test_waves_stale_revision_rejects_draft_without_publishing_controls(
    tmp_path, monkeypatch
) -> None:
    app, entry = _app_entry(tmp_path)
    original_pixels = entry.current_image
    transaction = StageEditTransaction.begin((entry,))
    candidate_pixels = np.full_like(original_pixels, 90)
    monkeypatch.setattr(
        app, "_compute_processing_candidate", lambda *_args: _candidate(candidate_pixels)
    )
    entry.revision += 1

    with pytest.raises(StaleStageRevisionError):
        app._commit_dewarp_candidate(
            entry,
            transaction,
            (
                (0.25, ((0.0, 0.1), (0.5, 0.1), (1.0, 0.1))),
                (0.5, ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
            ),
            PageProcessingRequest(),
        )

    np.testing.assert_array_equal(entry.current_image, original_pixels)
    assert entry.dewarp_control_curves is None
    assert entry.dewarp_control_points is None
    assert entry.committed_processing is None
    assert entry.revision == 1


def test_waves_success_returns_stage_diagnostics_and_publishes_normalized_controls(
    tmp_path, monkeypatch
) -> None:
    app, entry = _app_entry(tmp_path)
    transaction = StageEditTransaction.begin((entry,))
    candidate_pixels = np.full_like(entry.current_image, 90)
    dewarp_diagnostics = SimpleNamespace(max_displacement_px=4.0)
    full_diagnostics = SimpleNamespace(cache_hits=("waves",), dewarp=dewarp_diagnostics)
    committed = _candidate(candidate_pixels)[1]
    monkeypatch.setattr(
        app,
        "_compute_processing_candidate",
        lambda *_args: (candidate_pixels, committed, full_diagnostics),
    )

    returned = app._commit_dewarp_candidate(
        entry,
        transaction,
        (
            (0.75, ((1.0, 0.1), (0.5, 0.1), (0.0, 0.1))),
            (0.25, ((1.0, 0.0), (0.5, 0.0), (0.0, 0.0))),
        ),
        PageProcessingRequest(),
    )

    assert returned is dewarp_diagnostics
    assert app._last_processing_cache_hits == ("waves",)
    assert entry.dewarp_control_curves == (
        (0.25, ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
        (0.75, ((0.0, 0.1), (0.5, 0.1), (1.0, 0.1))),
    )
    assert entry.dewarp_control_points == ((0.0, 0.1), (0.5, 0.1), (1.0, 0.1))
    np.testing.assert_array_equal(entry.current_image, candidate_pixels)
    assert entry.revision == 1
