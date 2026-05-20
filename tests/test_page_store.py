import numpy as np

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
