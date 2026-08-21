import numpy as np
import pytest

from uniscan.session import CaptureSession
from uniscan.storage import PageStore


def _img(value: int = 0) -> np.ndarray:
    return np.full((10, 12, 3), value, dtype=np.uint8)


def test_session_add_move_select_remove(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    session.add_image(name="a", image=_img(10))
    session.add_image(name="b", image=_img(20))
    c = session.add_image(name="c", image=_img(30))

    assert len(session) == 3
    assert [x.name for x in session.entries] == ["a", "b", "c"]

    moved = session.move(c.entry_id, -1)
    assert moved
    assert [x.name for x in session.entries] == ["a", "c", "b"]

    session.select_all(True)
    removed = session.remove_selected()
    assert removed == 3
    assert len(session) == 0
    session.close()


def test_page_deletion_undo_restores_order_selection_and_assets(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entries = [session.add_image(name=name, image=_img(index)) for index, name in enumerate("abcd")]
    entries[1].selected = True
    entries[2].selected = True
    deleted_paths = (entries[1].original_path, entries[2].original_path)
    manifest_path = tmp_path / "session.json"

    assert session.remove_selected_for_undo() == 2
    assert session.can_undo_deletion is True
    assert [entry.name for entry in session.entries] == ["a", "d"]
    assert session.remove_selected_for_undo() == 0
    assert session.remove_entry("0" * 32) is False
    assert session.can_undo_deletion is True
    session.save_manifest(manifest_path)
    assert all(path.exists() for path in deleted_paths)

    session.add_image(name="e", image=_img(5))
    restored_ids = session.undo_last_deletion()

    assert restored_ids == (entries[1].entry_id, entries[2].entry_id)
    assert [entry.name for entry in session.entries] == ["a", "b", "c", "d", "e"]
    assert [entry.name for entry in session.selected_entries()] == ["b", "c"]
    assert session.can_undo_deletion is False
    session.close()


def test_finalized_page_deletion_is_pruned_after_manifest_save(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="delete-me", image=_img(1))
    entry.selected = True
    page_dir = entry.original_path.parent
    manifest_path = tmp_path / "session.json"

    assert session.remove_selected_for_undo() == 1
    session.save_manifest(manifest_path)
    assert page_dir.exists()
    assert session.finalize_pending_deletion() is True
    session.save_manifest(manifest_path)
    assert not page_dir.exists()
    assert session.undo_last_deletion() == ()
    session.close()


def test_second_page_deletion_replaces_the_undo_snapshot(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entries = [session.add_image(name=name, image=_img(index)) for index, name in enumerate("abc")]
    entries[1].selected = True
    assert session.remove_selected_for_undo() == 1

    entries[2].selected = True
    assert session.remove_selected_for_undo() == 1
    assert session.undo_last_deletion() == (entries[2].entry_id,)
    assert [entry.name for entry in session.entries] == ["a", "c"]
    assert session.undo_last_deletion() == ()
    session.close()


def test_session_moves_and_reorders_multiple_pages_as_a_stable_block(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entries = [session.add_image(name=name, image=_img(index)) for index, name in enumerate("abcd")]
    selected_ids = (entries[1].entry_id, entries[2].entry_id)

    assert session.move_many(selected_ids, 1) is True
    assert [entry.name for entry in session.entries] == ["a", "d", "b", "c"]
    assert session.move_many(selected_ids, -1) is True
    assert [entry.name for entry in session.entries] == ["a", "b", "c", "d"]

    assert session.reorder_entries(
        selected_ids,
        entries[3].entry_id,
        place_after=True,
    )
    assert [entry.name for entry in session.entries] == ["a", "d", "b", "c"]
    assert session.reorder_entries(
        selected_ids,
        entries[0].entry_id,
        place_after=False,
    )
    assert [entry.name for entry in session.entries] == ["b", "c", "a", "d"]
    session.close()


def test_session_apply_postprocess_uses_original(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="gray", image=_img(127))
    session.apply_postprocess("Grayscale")

    assert entry.current_image.ndim == 2
    assert entry.original_image.ndim == 3
    session.close()


def test_session_entries_are_disk_backed(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="disk", image=_img(64))

    assert entry.original_path.exists()
    assert entry.current_path.exists()
    assert entry.preview_original_path.exists()
    assert entry.preview_current_path.exists()
    assert entry.thumb_path.exists()
    session.close()


def test_entry_original_image_setter_writes_to_disk(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="orig", image=_img(10))
    entry.set_dewarp_control_points([(0.0, 0.0), (0.5, 0.02), (1.0, 0.0)])
    replacement = _img(200)
    entry.original_image = replacement

    reloaded = entry.original_image
    assert reloaded.shape == replacement.shape
    assert int(reloaded[0, 0, 0]) == 200
    preview = entry.preview_original_image
    assert preview.shape == replacement.shape
    assert int(entry.current_image[0, 0, 0]) == 200
    assert entry.dewarp_control_points is None
    session.close()


def test_entry_validates_dewarp_control_points(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="dewarp", image=_img(10))

    entry.set_dewarp_control_points([(1.0, 0.0), (0.5, 0.02), (0.0, 0.0)])

    assert entry.dewarp_control_points == ((0.0, 0.0), (0.5, 0.02), (1.0, 0.0))
    entry.clear_dewarp_control_points()
    assert entry.dewarp_control_points is None
    session.close()


def test_entry_persists_three_dewarp_control_curves(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="dewarp-curves", image=_img(10))
    points = [(0.0, 0.0), (0.5, 0.02), (1.0, 0.0)]

    entry.set_dewarp_control_curves([(0.75, points), (0.25, points), (0.5, points)])

    assert [curve[0] for curve in entry.dewarp_control_curves] == [0.25, 0.5, 0.75]
    assert entry.dewarp_control_points == entry.dewarp_control_curves[1][1]
    entry.clear_dewarp_control_points()
    assert entry.dewarp_control_curves is None
    session.close()


def test_replace_entry_image_updates_content_and_name(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="old", image=_img(10))
    original = _img(140)
    current = _img(220)

    ok = session.replace_entry_image(
        entry.entry_id,
        original_image=original,
        current_image=current,
        name="new_name",
    )

    assert ok
    assert entry.name == "new_name"
    assert int(entry.original_image[0, 0, 0]) == 140
    assert int(entry.current_image[0, 0, 0]) == 220
    session.close()


def test_replace_entry_image_clears_stale_detection_metadata(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image_with_contour(
        name="detected",
        raw_image=_img(10),
        warped_image=_img(20),
        contour=np.float32([[0, 0], [11, 0], [11, 9], [0, 9]]),
        backend="old-detector",
        needs_review=True,
        review_reasons=("large_dark_border_region",),
    )

    assert session.replace_entry_image(
        entry.entry_id,
        raw_image=_img(30),
        original_image=_img(30),
    )

    assert entry.detected_contour is None
    assert entry.detected_backend is None
    assert entry.needs_review is False
    assert entry.review_reasons == ()
    session.close()


def test_replace_entry_image_rolls_back_a_partial_multi_asset_failure(
    tmp_path, monkeypatch
) -> None:
    store = PageStore(root_dir=tmp_path)
    session = CaptureSession(store=store)
    old_contour = np.float32([[0, 0], [11, 0], [11, 9], [0, 9]])
    entry = session.add_image_with_contour(
        name="old",
        raw_image=_img(10),
        warped_image=_img(20),
        contour=old_contour,
        backend="old-detector",
    )
    real_atomic_write = store._atomic_image_write
    failures_left = 1

    def fail_once(path, image, *, kind):
        nonlocal failures_left
        if failures_left and path.name == "preview_original.jpg":
            failures_left -= 1
            raise RuntimeError("simulated preview failure")
        return real_atomic_write(path, image, kind=kind)

    monkeypatch.setattr(store, "_atomic_image_write", fail_once)
    with pytest.raises(RuntimeError, match="simulated preview failure"):
        session.replace_entry_image(
            entry.entry_id,
            name="new",
            raw_image=_img(100),
            original_image=_img(110),
            current_image=_img(120),
        )

    assert entry.name == "old"
    assert int(entry.raw_image[0, 0, 0]) == 10
    assert int(entry.original_image[0, 0, 0]) == 20
    assert int(entry.current_image[0, 0, 0]) == 20
    assert entry.detected_backend == "old-detector"
    np.testing.assert_array_equal(entry.detected_contour, old_contour)
    session.close()


def test_replace_entry_image_returns_false_for_unknown_id(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    session.add_image(name="only", image=_img(10))
    ok = session.replace_entry_image("missing", original_image=_img(50))
    assert not ok
    session.close()


def test_add_image_with_contour_stores_raw_and_contour(tmp_path) -> None:
    import numpy as np

    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    raw = _img(20)
    warped = _img(180)
    contour = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], dtype=np.float32)
    entry = session.add_image_with_contour(
        name="spread",
        raw_image=raw,
        warped_image=warped,
        contour=contour,
        backend="opencv_quad",
    )

    assert entry.raw_path.exists()
    assert int(entry.raw_image[0, 0, 0]) == 20
    assert int(entry.original_image[0, 0, 0]) == 180
    assert entry.detected_backend == "opencv_quad"
    assert entry.detected_contour is not None
    assert entry.detected_contour.shape == (4, 2)
    session.close()


def test_replace_raw_updates_raw_only(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="page", image=_img(50))
    new_raw = _img(120)
    entry.replace_raw(new_raw)
    assert int(entry.raw_image[0, 0, 0]) == 120
    # Original stays at the previous value because raw and original were separate copies.
    assert int(entry.original_image[0, 0, 0]) == 50
    session.close()
