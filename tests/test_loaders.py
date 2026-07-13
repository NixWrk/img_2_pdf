from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from uniscan.io.loaders import (
    _safe_render_dpi,
    _render_pdf_page,
    imread_unicode,
    imwrite_unicode,
    iter_pdf_pages,
    list_supported_in_folder,
    load_input_items,
    render_pdf_page_indices,
)


def _img(value: int) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _make_pdf(path) -> None:
    doc = pdfium.PdfDocument.new()
    try:
        for _index in range(2):
            page = doc.new_page(100, 100)
            page.close()
        doc.save(path)
    finally:
        doc.close()


def test_list_supported_in_folder_uses_natural_sort(tmp_path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    for name, value in [("page10.png", 10), ("page2.png", 20), ("page1.png", 30)]:
        ok = imwrite_unicode(folder / name, _img(value))
        assert ok

    paths = list_supported_in_folder(folder)
    assert [path.name for path in paths] == ["page1.png", "page2.png", "page10.png"]


def test_load_input_items_preserves_input_order_and_pdf_page_names(tmp_path) -> None:
    image_path = tmp_path / "image_a.png"
    pdf_path = tmp_path / "doc_b.pdf"
    ok = imwrite_unicode(image_path, _img(80))
    assert ok
    _make_pdf(pdf_path)

    items = load_input_items([pdf_path, image_path], pdf_dpi=72)

    assert [name for name, _image in items] == [
        "doc_b.pdf [p0001]",
        "doc_b.pdf [p0002]",
        "image_a.png",
    ]


def test_imread_unicode_scales_16bit_grayscale_without_saturation(tmp_path) -> None:
    path = tmp_path / "depth16.png"
    samples = np.array([[0, 1000, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(samples).save(path)

    loaded = imread_unicode(path)

    assert loaded is not None
    np.testing.assert_allclose(loaded[0, :, 0], [0, 4, 128, 255], atol=1)
    np.testing.assert_array_equal(loaded[:, :, 0], loaded[:, :, 1])
    np.testing.assert_array_equal(loaded[:, :, 1], loaded[:, :, 2])


def test_imread_unicode_composites_alpha_on_white(tmp_path) -> None:
    path = tmp_path / "alpha.png"
    rgba = np.array([[[255, 0, 0, 0], [255, 0, 0, 255]]], dtype=np.uint8)
    Image.fromarray(rgba).save(path)

    loaded = imread_unicode(path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(loaded[0, 1], [0, 0, 255])


def test_load_input_items_yields_every_multipage_tiff_frame(tmp_path) -> None:
    path = tmp_path / "scan.tiff"
    first = Image.fromarray(np.full((12, 16), 30, dtype=np.uint8))
    second = Image.fromarray(np.full((12, 16), 210, dtype=np.uint8))
    first.save(path, save_all=True, append_images=[second])

    items = load_input_items([path], pdf_dpi=72)

    assert [name for name, _image in items] == [
        "scan.tiff [p0001]",
        "scan.tiff [p0002]",
    ]
    assert [round(float(image.mean())) for _name, image in items] == [30, 210]


def test_imwrite_unicode_keeps_existing_target_when_replace_fails(tmp_path, monkeypatch) -> None:
    target = tmp_path / "atomic.png"
    target.write_bytes(b"existing")

    monkeypatch.setattr(
        "uniscan.io.loaders.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    assert imwrite_unicode(target, _img(90)) is False
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_pdfium_page_selection_and_streaming_cancellation(tmp_path) -> None:
    path = tmp_path / "two-pages.pdf"
    _make_pdf(path)

    selected = render_pdf_page_indices(path, [1], dpi=72)
    assert [name for name, _image in selected] == ["two-pages.pdf [p0002]"]
    assert selected[0][1].shape[:2] == (100, 100)

    calls = 0

    def cancel_after_first() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    pages = iter_pdf_pages(path, dpi=72, cancel_cb=cancel_after_first)
    assert next(pages)[0] == "two-pages.pdf [p0001]"
    with pytest.raises(RuntimeError, match="Cancelled"):
        next(pages)


def test_safe_render_dpi_calculates_limit_without_claiming_to_change_request() -> None:
    safe = _safe_render_dpi((720.0, 720.0), requested_dpi=600, max_pixels=1_000_000)
    assert safe == 100


def test_pdf_render_fails_closed_if_pixel_cap_would_change_physical_size(
    tmp_path,
    monkeypatch,
) -> None:
    class FakePage:
        def get_size(self):
            return 720.0, 720.0

        def render(self, **_kwargs):
            raise AssertionError("oversized page must not be rendered")

    monkeypatch.setattr("uniscan.io.loaders._safe_render_dpi", lambda *_args, **_kwargs: 100)

    with pytest.raises(RuntimeError, match="silently lowering DPI"):
        _render_pdf_page(
            FakePage(),
            pdf_path=tmp_path / "large.pdf",
            page_index=0,
            dpi=600,
        )
