import numpy as np

from uniscan.session import CaptureSession
from uniscan.storage import PageStore


def _img(value: int = 0) -> np.ndarray:
    return np.full((10, 12, 3), value, dtype=np.uint8)


def test_session_add_move_select_remove(tmp_path) -> None:
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    a = session.add_image(name="a", image=_img(10))
    b = session.add_image(name="b", image=_img(20))
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
    replacement = _img(200)
    entry.original_image = replacement

    reloaded = entry.original_image
    assert reloaded.shape == replacement.shape
    assert int(reloaded[0, 0, 0]) == 200
    preview = entry.preview_original_image
    assert preview.shape == replacement.shape
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
