import os

import cv2
import numpy as np
import pytest

from uniscan.storage import PageStore


def _img(value: int = 0) -> np.ndarray:
    out = np.zeros((40, 60, 3), dtype=np.uint8)
    out[:, :] = (value, value + 10, value + 20)
    return out


def _entry_id(seed: int) -> str:
    return f"{seed:032x}"


def test_page_store_add_read_remove(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    entry_id = _entry_id(1)
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
    store.add_page(_entry_id(2), _img())
    session_dir = store.session_dir
    assert session_dir.exists()

    store.close()
    assert not session_dir.exists()


def test_page_store_writes_raw_distinct_from_warped(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    raw = _img(10)
    warped = _img(200)
    paths = store.add_page(_entry_id(3), raw, warped)

    raw_back = store.read_image(paths.raw)
    warped_back = store.read_image(paths.original)
    assert int(raw_back[0, 0, 0]) == 10
    assert int(warped_back[0, 0, 0]) == 200


def test_page_store_preview_resize_never_adds_letterbox_padding(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    source = np.full((1200, 2400, 3), 73, dtype=np.uint8)

    preview = store._resize_for_display(source, max_width=1000, max_height=1000)

    assert preview.shape == (500, 1000, 3)
    assert np.all(preview == 73)


def test_page_store_resizes_grayscale_before_expanding_channels(tmp_path, monkeypatch) -> None:
    store = PageStore(root_dir=tmp_path)
    source = np.full((1200, 2400), 73, dtype=np.uint8)
    converted_shapes: list[tuple[int, ...]] = []
    real_cvt_color = cv2.cvtColor

    def tracked_cvt_color(image, code):
        converted_shapes.append(image.shape)
        return real_cvt_color(image, code)

    monkeypatch.setattr("uniscan.storage.page_store.cv2.cvtColor", tracked_cvt_color)

    preview = store._resize_for_display(source, max_width=1000, max_height=1000)

    assert converted_shapes == [(500, 1000)]
    assert preview.shape == (500, 1000, 3)
    assert np.all(preview == 73)


def test_page_store_failed_encode_keeps_previous_complete_file(tmp_path, monkeypatch) -> None:
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page(_entry_id(4), _img(10))
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
    entry_id = _entry_id(5)
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page(entry_id, _img(10), _img(20))
    page_dir, stage_dir, backup_dir = store._page_directories(entry_id)
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
    entry_id = _entry_id(6)
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page(entry_id, _img(10), _img(20))
    page_dir, stage_dir, backup_dir = store._page_directories(entry_id)
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


def test_page_store_propagates_bounded_decoder_failure(tmp_path, monkeypatch) -> None:
    store = PageStore(root_dir=tmp_path)
    paths = store.add_page(_entry_id(7), _img(10), _img(20))

    def reject(_path, **_kwargs):
        raise RuntimeError("safe input limit: 150,000,000 pixels")

    monkeypatch.setattr("uniscan.storage.page_store.imread_unicode", reject)
    with pytest.raises(RuntimeError, match="safe input limit"):
        store.read_image(paths.current)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda store, entry_id: store.paths_for_entry(entry_id),
            id="paths_for_entry",
        ),
        pytest.param(
            lambda store, entry_id: store.replace_page_set(entry_id, current_image=_img()),
            id="replace_page_set",
        ),
        pytest.param(
            lambda store, entry_id: store.add_page(entry_id, _img()),
            id="add_page",
        ),
        pytest.param(
            lambda store, entry_id: store.remove_page(entry_id),
            id="remove_page",
        ),
        pytest.param(
            lambda store, entry_id: store.prune_pages({entry_id}),
            id="prune_pages",
        ),
        pytest.param(
            lambda store, entry_id: store.repair_page_assets(entry_id),
            id="repair_page_assets",
        ),
    ],
)
@pytest.mark.parametrize(
    "entry_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "..\\outside",
        "nested/page",
        "nested\\page",
        "/absolute",
        "C:\\absolute",
        "0" * 31,
        "A" * 32,
        "550e8400-e29b-41d4-a716-446655440000",
    ],
)
def test_page_store_rejects_invalid_entry_ids_before_page_resolution(
    tmp_path, monkeypatch, operation, entry_id
) -> None:
    store = PageStore(root_dir=tmp_path)
    monkeypatch.setattr(store, "_recover_page_locked", pytest.fail)
    monkeypatch.setattr(store, "_page_directories", pytest.fail)

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        operation(store, entry_id)
