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
    session.close()
