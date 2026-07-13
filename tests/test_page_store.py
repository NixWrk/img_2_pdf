import os

import numpy as np
import pytest

from uniscan.storage import PageStore


def _img(value: int = 0) -> np.ndarray:
    out = np.zeros((40, 60, 3), dtype=np.uint8)
    out[:, :] = (value, value + 10, value + 20)
    return out


def test_page_store_add_read_remove(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    entry_id = "entry_a"
    paths = store.add_page(entry_id, _img())

    assert paths.raw.exists()
    assert paths.original.exists()
    assert paths.current.exists()
    assert paths.preview_raw.exists()
    assert paths.preview_original.exists()
    assert paths.preview_current.exists()
    assert paths.thumb.exists()
    assert store.read_image(paths.current).shape == (40, 60, 3)
    assert store.read_image(paths.preview_original).shape == (40, 60, 3)

    store.remove_page(entry_id)
    assert not paths.original.exists()
    assert not paths.current.exists()
    assert not paths.raw.exists()


def test_page_store_cleanup_session(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    store.add_page("entry_b", _img())
    session_dir = store.session_dir
    assert session_dir.exists()

    store.close()
    assert not session_dir.exists()


def test_page_store_writes_raw_distinct_from_warped(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    raw = _img(10)
    warped = _img(200)
    paths = store.add_page("entry_pair", raw, warped)

    raw_back = store.read_image(paths.raw)
    warped_back = store.read_image(paths.original)
    assert int(raw_back[0, 0, 0]) == 10
    assert int(warped_back[0, 0, 0]) == 200


def test_page_store_failed_encode_keeps_previous_complete_file(tmp_path, monkeypatch) -> None:
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page("entry_atomic", _img(10))
    previous = paths.current.read_bytes()

    def fail_after_partial_write(path, _image):
        path.write_bytes(b"partial")
        return False

    monkeypatch.setattr("uniscan.storage.page_store.imwrite_unicode", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="Cannot write page image"):
        store.write_image(paths.current, _img(200))

    assert paths.current.read_bytes() == previous
    assert not list(paths.current.parent.glob(".*.stage-*"))


def test_page_store_recovers_old_generation_if_swap_stopped_before_publish(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page("entry_recover_old", _img(10), _img(20))
    page_dir, stage_dir, backup_dir = store._page_directories("entry_recover_old")
    store._write_page_set(
        stage_dir,
        raw_image=_img(100),
        warped_image=_img(110),
        current_image=_img(120),
    )
    os.replace(page_dir, backup_dir)  # process dies before stage publication

    assert int(store.read_image(paths.raw)[0, 0, 0]) == 10
    assert int(store.read_image(paths.original)[0, 0, 0]) == 20
    assert int(store.read_image(paths.current)[0, 0, 0]) == 20
    assert page_dir.is_dir()
    assert not stage_dir.exists()
    assert not backup_dir.exists()


def test_page_store_recovers_new_generation_if_cleanup_was_interrupted(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page("entry_recover_new", _img(10), _img(20))
    page_dir, stage_dir, backup_dir = store._page_directories("entry_recover_new")
    store._write_page_set(
        stage_dir,
        raw_image=_img(100),
        warped_image=_img(110),
        current_image=_img(120),
    )
    os.replace(page_dir, backup_dir)
    os.replace(stage_dir, page_dir)  # process dies before backup cleanup

    assert int(store.read_image(paths.raw)[0, 0, 0]) == 100
    assert int(store.read_image(paths.original)[0, 0, 0]) == 110
    assert int(store.read_image(paths.current)[0, 0, 0]) == 120
    assert page_dir.is_dir()
    assert not backup_dir.exists()
