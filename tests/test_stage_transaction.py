from __future__ import annotations

import numpy as np
import pytest

from uniscan.core.processing import PageProcessingRequest
from uniscan.session.capture_session import (
    CaptureEntry,
    CommittedPageProcessing,
    ProcessingRecipe,
)
from uniscan.storage.page_store import PageStore
from uniscan.ui.stage_transaction import (
    IncompleteStageCandidateError,
    StageEditTransaction,
    StageTransactionClosedError,
    StageTransactionError,
    StaleStageRevisionError,
)


class FakeEntry:
    def __init__(self, entry_id: str, image: np.ndarray) -> None:
        self.entry_id = entry_id
        self.revision = 3
        self._image = image.copy()
        self.committed_processing = {"recipe": {"stage": "waves"}, "diagnostics": {"ok": True}}
        self.wave_curves = {"top": [(0.0, 1.0)]}
        self.fail_next_write = False

    @property
    def current_image(self) -> np.ndarray:
        return self._image.copy()

    @current_image.setter
    def current_image(self, image: np.ndarray) -> None:
        if self.fail_next_write:
            self.fail_next_write = False
            raise OSError("simulated page-store failure")
        self._image = image.copy()
        self.committed_processing = None
        self.revision += 1


def _tx(*entries: FakeEntry) -> StageEditTransaction:
    return StageEditTransaction.begin(entries, metadata_fields=("wave_curves",))


def _real_entry(store: PageStore, image: np.ndarray, name: str = "page") -> CaptureEntry:
    return CaptureEntry.from_image(name=name, image=image, store=store)


def _committed(image: np.ndarray) -> CommittedPageProcessing:
    return CommittedPageProcessing(
        recipe=ProcessingRecipe.from_request(PageProcessingRequest()),
        diagnostics={"ok": True},
        current_fingerprint=CommittedPageProcessing.fingerprint_image(image),
    )


def test_commit_publishes_copied_pixels_processing_and_metadata() -> None:
    entry = FakeEntry("page-1", np.array([[1, 2]], dtype=np.uint8))
    tx = _tx(entry)
    pixels = np.array([[9, 8]], dtype=np.uint8)
    processing = {"recipe": {"mode": "auto"}, "diagnostics": {"score": 0.8}}
    curves = {"top": [(0.0, 0.2), (1.0, 0.3)]}

    candidate = tx.stage(
        entry.entry_id,
        pixels=pixels,
        committed_processing=processing,
        metadata={"wave_curves": curves},
    )
    candidate.pixels[0, 0] = 3
    candidate.committed_processing["diagnostics"]["score"] = 0
    candidate.metadata["wave_curves"]["top"].append((2.0, 1.0))
    pixels[0, 0] = 0
    processing["diagnostics"]["score"] = 0
    curves["top"].append((2.0, 1.0))

    assert tx.commit() == (entry.entry_id,)
    assert np.array_equal(entry.current_image, [[9, 8]])
    assert entry.committed_processing["diagnostics"]["score"] == 0.8
    assert entry.wave_curves == {"top": [(0.0, 0.2), (1.0, 0.3)]}
    assert entry.revision == 4


def test_snapshot_is_independent_from_entry_and_isolated_on_read() -> None:
    entry = FakeEntry("page-1", np.array([[1]], dtype=np.uint8))
    tx = _tx(entry)
    first = tx.snapshots[0]
    first.pixels[0, 0] = 7
    first.committed_processing["diagnostics"]["ok"] = False
    first.metadata["wave_curves"]["top"].append((1.0, 1.0))

    second = tx.snapshots[0]
    assert second.pixels[0, 0] == 1
    assert second.committed_processing["diagnostics"]["ok"] is True
    assert second.metadata["wave_curves"] == {"top": [(0.0, 1.0)]}
    assert entry.current_image[0, 0] == 1


def test_reserved_metadata_is_rejected() -> None:
    entry = FakeEntry("page-1", np.array([[1]], dtype=np.uint8))
    with pytest.raises(ValueError, match="entry_id"):
        StageEditTransaction.begin((entry,), metadata_fields=("entry_id",))
    tx = _tx(entry)
    with pytest.raises(KeyError, match="current_image"):
        tx.stage(
            entry.entry_id,
            pixels=np.array([[2]], dtype=np.uint8),
            committed_processing=None,
            metadata={"current_image": np.array([[2]], dtype=np.uint8)},
        )


def test_stale_revision_is_rejected_before_any_batch_write_and_closes() -> None:
    first = FakeEntry("page-1", np.array([[1]], dtype=np.uint8))
    second = FakeEntry("page-2", np.array([[2]], dtype=np.uint8))
    tx = _tx(first, second)
    for entry, value in ((first, 8), (second, 9)):
        tx.stage(
            entry.entry_id, pixels=np.array([[value]], dtype=np.uint8), committed_processing=None
        )

    second.revision += 1
    with pytest.raises(StaleStageRevisionError) as error:
        tx.commit()

    assert error.value.entry_ids == (second.entry_id,)
    assert np.array_equal(first.current_image, [[1]])
    assert np.array_equal(second.current_image, [[2]])
    assert first.revision == 3
    with pytest.raises(StageTransactionClosedError):
        tx.discard()


def test_failed_batch_commit_restores_already_changed_entries() -> None:
    first = FakeEntry("page-1", np.array([[1]], dtype=np.uint8))
    second = FakeEntry("page-2", np.array([[2]], dtype=np.uint8))
    tx = _tx(first, second)
    tx.stage(
        first.entry_id, pixels=np.array([[8]], dtype=np.uint8), committed_processing={"new": 1}
    )
    tx.stage(
        second.entry_id, pixels=np.array([[9]], dtype=np.uint8), committed_processing={"new": 2}
    )
    second.fail_next_write = True

    with pytest.raises(StageTransactionError, match="all entries were restored"):
        tx.commit()

    assert np.array_equal(first.current_image, [[1]])
    assert np.array_equal(second.current_image, [[2]])
    assert first.committed_processing["recipe"]["stage"] == "waves"
    assert second.committed_processing["recipe"]["stage"] == "waves"
    assert first.revision == second.revision == 3


def test_real_capture_entry_success_persists_processing_fingerprint_and_revision(tmp_path) -> None:
    store = PageStore(tmp_path)
    original = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    entry = _real_entry(store, original)
    entry.dewarp_control_curves = ((0.5, ((0.0, 0.0), (1.0, 0.0))),)
    candidate_pixels = np.array([[9, 8], [7, 6]], dtype=np.uint8)
    candidate_processing = _committed(candidate_pixels)
    tx = StageEditTransaction.begin((entry,))
    tx.stage(
        entry.entry_id,
        pixels=candidate_pixels,
        committed_processing=candidate_processing,
        metadata={
            "dewarp_control_points": ((0.0, 0.1), (1.0, 0.1)),
            "dewarp_control_curves": ((0.5, ((0.0, 0.1), (1.0, 0.1))),),
        },
    )

    assert tx.commit() == (entry.entry_id,)
    assert np.array_equal(entry.current_image, candidate_pixels)
    assert isinstance(entry.committed_processing, CommittedPageProcessing)
    assert (
        entry.committed_processing.current_fingerprint
        == _committed(candidate_pixels).current_fingerprint
    )
    assert entry.revision == 1
    assert entry.dewarp_control_points == ((0.0, 0.1), (1.0, 0.1))
    assert entry.dewarp_control_curves == ((0.5, ((0.0, 0.1), (1.0, 0.1))),)


def test_real_capture_entry_cancel_has_no_mutation(tmp_path) -> None:
    store = PageStore(tmp_path)
    original = np.array([[1, 2]], dtype=np.uint8)
    entry = _real_entry(store, original)
    before = entry.current_image
    tx = StageEditTransaction.begin((entry,))
    tx.stage(entry.entry_id, pixels=np.array([[8, 9]], dtype=np.uint8), committed_processing=None)
    tx.discard()
    assert np.array_equal(entry.current_image, before)
    assert entry.revision == 0


def test_real_batch_failure_restores_pixels_metadata_controls_and_revisions(tmp_path) -> None:
    store = PageStore(tmp_path)
    first = _real_entry(store, np.array([[1]], dtype=np.uint8), "first")
    second = _real_entry(store, np.array([[2]], dtype=np.uint8), "second")
    first.dewarp_control_curves = ((0.5, ((0.0, 0.0), (1.0, 0.0))),)
    second.dewarp_control_curves = ((0.5, ((0.0, 0.2), (1.0, 0.2))),)
    first.dewarp_control_points = ((0.0, 0.0), (1.0, 0.0))
    second.dewarp_control_points = ((0.0, 0.2), (1.0, 0.2))
    first.committed_processing = _committed(first.current_image)
    second.committed_processing = _committed(second.current_image)
    original_first = first.current_image
    original_second = second.current_image
    original_first_processing = first.committed_processing
    original_second_processing = second.committed_processing
    original_first_controls = first.dewarp_control_curves
    original_second_controls = second.dewarp_control_curves
    original_first_points = first.dewarp_control_points
    original_second_points = second.dewarp_control_points

    original_replace = second.store.replace_page_set
    failed = False

    def fail_second(entry_id, *args, **kwargs):
        nonlocal failed
        if entry_id == second.entry_id and kwargs.get("current_image") is not None and not failed:
            failed = True
            raise OSError("simulated page-store failure")
        return original_replace(entry_id, *args, **kwargs)

    second.store.replace_page_set = fail_second
    tx = StageEditTransaction.begin((first, second))
    tx.stage(
        first.entry_id,
        pixels=np.array([[8]], dtype=np.uint8),
        committed_processing=None,
        metadata={
            "dewarp_control_points": ((0.0, 0.8), (1.0, 0.8)),
            "dewarp_control_curves": ((0.5, ((0.0, 0.8), (1.0, 0.8))),),
        },
    )
    tx.stage(
        second.entry_id,
        pixels=np.array([[9]], dtype=np.uint8),
        committed_processing=None,
        metadata={
            "dewarp_control_points": ((0.0, 0.9), (1.0, 0.9)),
            "dewarp_control_curves": ((0.5, ((0.0, 0.9), (1.0, 0.9))),),
        },
    )

    with pytest.raises(StageTransactionError, match="all entries were restored"):
        tx.commit()

    assert np.array_equal(first.current_image, original_first)
    assert np.array_equal(second.current_image, original_second)
    assert first.committed_processing == original_first_processing
    assert second.committed_processing == original_second_processing
    assert first.dewarp_control_curves == original_first_controls
    assert second.dewarp_control_curves == original_second_controls
    assert first.dewarp_control_points == original_first_points
    assert second.dewarp_control_points == original_second_points
    assert first.revision == second.revision == 0


def test_fingerprint_preflight_rejects_entire_batch_before_writes(tmp_path) -> None:
    store = PageStore(tmp_path)
    first = _real_entry(store, np.array([[1]], dtype=np.uint8), "first")
    second = _real_entry(store, np.array([[2]], dtype=np.uint8), "second")
    first_before = first.current_image
    second_before = second.current_image
    tx = StageEditTransaction.begin((first, second), metadata_fields=())
    tx.stage(
        first.entry_id,
        pixels=np.array([[8]], dtype=np.uint8),
        committed_processing=_committed(np.array([[8]], dtype=np.uint8)),
    )
    tx.stage(
        second.entry_id,
        pixels=np.array([[9]], dtype=np.uint8),
        committed_processing=_committed(np.array([[7]], dtype=np.uint8)),
    )

    with pytest.raises(StageTransactionError, match="fingerprint"):
        tx.commit()

    assert np.array_equal(first.current_image, first_before)
    assert np.array_equal(second.current_image, second_before)
    assert first.revision == second.revision == 0


def test_commit_requires_a_complete_batch() -> None:
    first = FakeEntry("page-1", np.array([[1]], dtype=np.uint8))
    second = FakeEntry("page-2", np.array([[2]], dtype=np.uint8))
    tx = _tx(first, second)
    tx.stage(first.entry_id, pixels=np.array([[8]], dtype=np.uint8), committed_processing=None)

    with pytest.raises(IncompleteStageCandidateError):
        tx.commit()

    assert first.revision == second.revision == 3
