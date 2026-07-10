from __future__ import annotations

import json

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


def test_incomplete_page_assets_are_rejected(tmp_path) -> None:
    entry_id = "0" * 32
    session_dir = tmp_path / "sessions" / "broken"
    (session_dir / "pages" / entry_id).mkdir(parents=True)
    manifest = tmp_path / "autosave.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sessionDir": str(session_dir),
                "entries": [{"entryId": entry_id, "name": "broken"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assets are incomplete"):
        CaptureSession.restore_manifest(manifest)


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
