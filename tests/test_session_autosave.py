from __future__ import annotations

import json
import os

import numpy as np
import pytest

from uniscan.session import (
    CaptureSession,
    create_persistent_session,
    discard_autosave,
    load_or_create_session,
)


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
    entry.current_image = _image(190)
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
