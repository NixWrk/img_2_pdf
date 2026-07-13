from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.session import (
    AutosaveSessionLock,
    CommittedPageProcessing,
    SessionInUseError,
    UnsafeSessionLockError,
    acquire_autosave_lock,
    CaptureSession,
    create_persistent_session,
    discard_autosave,
    load_or_create_session,
)
from uniscan.core.processing import PageProcessingRequest, process_document_page


def _image(value: int) -> np.ndarray:
    return np.full((20, 30, 3), value, dtype=np.uint8)


def test_session_manifest_round_trip(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    first = session.add_image_with_contour(
        name="first",
        raw_image=_image(10),
        warped_image=_image(20),
        contour=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        backend="fake",
    )
    first.selected = True
    first.set_dewarp_control_points([(0.0, 0.0), (0.5, 0.015), (1.0, 0.0)])
    session.add_image(name="second", image=_image(30))
    session.save_manifest(manifest)
    assert json.loads(manifest.read_text(encoding="utf-8"))["schemaVersion"] == 2
    session.close(preserve=True)

    restored, was_restored = load_or_create_session(manifest)

    assert was_restored is True
    assert [entry.name for entry in restored.entries] == ["first", "second"]
    assert restored.entries[0].selected is True
    assert restored.entries[0].detected_backend == "fake"
    np.testing.assert_array_equal(restored.entries[0].detected_contour, first.detected_contour)
    assert restored.entries[0].dewarp_control_points == first.dewarp_control_points
    assert int(restored.entries[1].current_image.mean()) == 30
    discard_autosave(restored, manifest)
    assert not manifest.exists()
    assert not restored.store.session_dir.exists()


def test_corrupt_manifest_is_rejected(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    manifest.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read session manifest"):
        CaptureSession.restore_manifest(manifest)


def test_unidentifiable_manifest_entry_aborts_without_pruning_assets(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="source", image=_image(91))
    session.save_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["entryId"] = "not-a-page-id"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    source_dir = entry.original_path.parent
    session.close(preserve=True)

    with pytest.raises(ValueError, match="Cannot safely associate"):
        CaptureSession.restore_manifest(manifest)

    assert source_dir.is_dir()
    assert (source_dir / "raw.png").is_file()
    assert (source_dir / "original.png").is_file()


def test_incomplete_page_assets_are_skipped_without_losing_valid_pages(tmp_path) -> None:
    session = create_persistent_session(tmp_path)
    valid = session.add_image(name="valid", image=_image(42))
    entry_id = "0" * 32
    session_dir = session.store.session_dir
    (session_dir / "pages" / entry_id).mkdir(parents=True)
    manifest = tmp_path / "autosave.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sessionDir": str(session_dir),
                "entries": [
                    {"entryId": entry_id, "name": "broken"},
                    {"entryId": valid.entry_id, "name": "valid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    restored = CaptureSession.restore_manifest(manifest)

    assert [entry.name for entry in restored.entries] == ["valid"]
    assert len(restored.restore_warnings) == 1
    assert restored.quarantined_entry_ids == (entry_id,)
    assert restored.has_recoverable_state is True

    restored.save_manifest(manifest)
    assert (session_dir / "pages" / entry_id).is_dir()
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["quarantinedEntries"] == [{"entryId": entry_id, "name": "broken"}]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_restore_rebuilds_missing_or_corrupt_current_from_original(tmp_path, damage) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image_with_contour(
        name="recoverable",
        raw_image=_image(10),
        warped_image=_image(70),
        contour=None,
        backend=None,
    )
    request = PageProcessingRequest(postprocess_name="Grayscale")
    result = process_document_page(entry.original_image, request)
    entry.current_image = result.image
    entry.committed_processing = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )
    session.save_manifest(manifest)
    if damage == "missing":
        entry.current_path.unlink()
    else:
        entry.current_path.write_bytes(b"not an image")
    session.close(preserve=True)

    restored = CaptureSession.restore_manifest(manifest)

    assert [item.name for item in restored.entries] == ["recoverable"]
    assert restored.quarantined_entry_ids == ()
    np.testing.assert_array_equal(restored.entries[0].current_image, _image(70))
    assert restored.entries[0].committed_processing is None
    assert "Recovered session page" in restored.restore_warnings[0]
    paths = restored.entries[0].paths
    assert all(
        path.is_file()
        for path in (paths.preview_raw, paths.preview_original, paths.preview_current, paths.thumb)
    )


def test_skipped_valid_page_remains_quarantined_until_explicit_discard(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="metadata-damaged", image=_image(55))
    session.save_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["detectedContour"] = [[1, 2]]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    page_dir = entry.original_path.parent
    session.close(preserve=True)

    restored = CaptureSession.restore_manifest(manifest)
    assert len(restored) == 0
    assert restored.quarantined_entry_ids == (entry.entry_id,)
    assert restored.has_recoverable_state is True
    assert restored.store.read_image(page_dir / "raw.png").mean() == 55
    assert restored.store.read_image(page_dir / "original.png").mean() == 55

    restored.save_manifest(manifest)
    assert page_dir.is_dir()
    restored_again = CaptureSession.restore_manifest(manifest)
    assert restored_again.quarantined_entry_ids == (entry.entry_id,)
    assert page_dir.is_dir()

    discard_autosave(restored_again, manifest)
    assert not manifest.exists()
    assert not restored_again.store.session_dir.exists()


def test_remove_is_committed_manifest_first_then_assets_are_pruned(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    removed = session.add_image(name="remove", image=_image(10))
    kept = session.add_image(name="keep", image=_image(20))
    session.save_manifest(manifest)

    removed.selected = True
    assert session.remove_selected() == 1
    assert removed.original_path.exists()

    # A crash before the checkpoint still has the complete old generation.
    old_generation = CaptureSession.restore_manifest(manifest)
    assert [entry.name for entry in old_generation.entries] == ["remove", "keep"]

    session.save_manifest(manifest)
    assert not removed.original_path.exists()
    assert kept.original_path.exists()
    new_generation = CaptureSession.restore_manifest(manifest)
    assert [entry.name for entry in new_generation.entries] == ["keep"]


def test_manifest_restore_recovers_interrupted_page_generation_swap(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="stable", image=_image(30))
    session.save_manifest(manifest)
    page_dir, stage_dir, backup_dir = session.store._page_directories(entry.entry_id)
    session.store._write_page_set(
        stage_dir,
        raw_image=_image(100),
        warped_image=_image(110),
        current_image=_image(120),
    )
    os.replace(page_dir, backup_dir)
    session.close(preserve=True)

    restored, was_restored = load_or_create_session(manifest)

    assert was_restored is True
    assert len(restored.entries) == 1
    assert int(restored.entries[0].raw_image.mean()) == 30
    assert int(restored.entries[0].original_image.mean()) == 30
    assert int(restored.entries[0].current_image.mean()) == 30
    assert not stage_dir.exists()
    assert not backup_dir.exists()


def test_autosave_rejects_session_directory_outside_state_root(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "must-not-delete"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("safe", encoding="utf-8")
    manifest = state / "autosave.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sessionDir": str(outside),
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes autosave storage"):
        load_or_create_session(manifest)

    assert marker.read_text(encoding="utf-8") == "safe"


def test_discard_removes_manifest_before_deleting_assets(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    session.add_image(name="page", image=_image(10))
    session.save_manifest(manifest)

    def interrupted_close(*, preserve=False):
        del preserve
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(session, "close", interrupted_close)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        discard_autosave(session, manifest)

    assert not manifest.exists()
    assert session.store.session_dir.exists()


def test_autosave_lock_is_nonblocking_and_releasable(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    first = acquire_autosave_lock(manifest)
    try:
        with pytest.raises(SessionInUseError, match="already|using"):
            acquire_autosave_lock(manifest)
    finally:
        first.release()
    acquire_autosave_lock(manifest).release()


def test_autosave_lock_rejects_reparse_metadata() -> None:
    fake_path = SimpleNamespace(
        lstat=lambda: SimpleNamespace(
            st_mode=stat.S_IFREG,
            st_file_attributes=0x400,
        ),
        is_symlink=lambda: False,
        is_junction=lambda: False,
    )
    with pytest.raises(UnsafeSessionLockError, match="Unsafe autosave lock path"):
        AutosaveSessionLock._validate_path(fake_path)


def test_autosave_lock_rejects_symlink_without_touching_referent(tmp_path) -> None:
    referent = tmp_path / "referent.txt"
    referent.write_bytes(b"keep-me")
    lock_path = tmp_path / "autosave.lock"
    try:
        lock_path.symlink_to(referent)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this Windows host.")

    with pytest.raises(UnsafeSessionLockError):
        AutosaveSessionLock.acquire(lock_path)
    assert referent.read_bytes() == b"keep-me"


def test_autosave_lock_rejects_hardlink_without_touching_referent(tmp_path) -> None:
    referent = tmp_path / "referent.txt"
    referent.write_bytes(b"keep-hardlink")
    lock_path = tmp_path / "autosave.lock"
    os.link(referent, lock_path)

    with pytest.raises(UnsafeSessionLockError):
        AutosaveSessionLock.acquire(lock_path)
    assert referent.read_bytes() == b"keep-hardlink"


def test_manifest_v2_round_trips_committed_recipe_and_v1_migrates(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="processed", image=_image(40))
    request = PageProcessingRequest(postprocess_name="Grayscale", page_dpi=240)
    result = process_document_page(entry.original_image, request)
    entry.current_image = result.image
    entry.committed_processing = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )
    session.save_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 2
    session.close(preserve=True)

    restored = CaptureSession.restore_manifest(manifest)
    committed = restored.entries[0].committed_processing
    assert committed is not None
    assert committed.recipe.page_dpi == 240
    assert committed.recipe.postprocess_name == "Grayscale"
    assert "layout" in committed.diagnostics

    payload["schemaVersion"] = 1
    payload["entries"][0].pop("committedProcessing", None)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    restored_v1 = CaptureSession.restore_manifest(manifest)
    assert restored_v1.entries[0].committed_processing is None


def test_invalid_optional_recipe_is_dropped_without_quarantining_page(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="valid-page", image=_image(77))
    session.save_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["committedProcessing"] = {
        "schemaVersion": 999,
        "recipe": {},
        "diagnostics": {},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    session.close(preserve=True)

    restored = CaptureSession.restore_manifest(manifest)
    assert [item.entry_id for item in restored.entries] == [entry.entry_id]
    assert restored.entries[0].committed_processing is None
    assert restored.quarantined_entry_ids == ()
    assert any("Ignored processing metadata" in warning for warning in restored.restore_warnings)


def test_manifest_json_root_must_be_object(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported session manifest"):
        CaptureSession.restore_manifest(manifest)

    manifest.write_text('{"schemaVersion": true, "entries": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported session manifest"):
        CaptureSession.restore_manifest(manifest)


def test_restore_drops_recipe_if_current_pixels_changed_after_manifest(tmp_path) -> None:
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="crash-window", image=_image(20))
    request = PageProcessingRequest(postprocess_name="Grayscale", page_dpi=220)
    result = process_document_page(entry.original_image, request)
    entry.current_image = result.image
    entry.committed_processing = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )
    session.save_manifest(manifest)
    entry.store.write_image(entry.current_path, _image(230))
    session.close(preserve=True)

    restored = CaptureSession.restore_manifest(manifest)
    assert restored.entries[0].committed_processing is None
    assert any("current image fingerprint changed" in item for item in restored.restore_warnings)


def test_replace_raw_invalidates_committed_processing_and_revision(tmp_path) -> None:
    session = create_persistent_session(tmp_path)
    entry = session.add_image(name="raw", image=_image(20))
    request = PageProcessingRequest()
    result = process_document_page(entry.original_image, request)
    entry.committed_processing = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )
    revision = entry.revision

    entry.replace_raw(_image(90))

    assert entry.committed_processing is None
    assert entry.revision == revision + 1
