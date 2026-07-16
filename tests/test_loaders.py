from __future__ import annotations

import warnings

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from uniscan.io.loaders import (
    _safe_render_dpi,
    _render_pdf_page,
    imread_unicode,
    imwrite_unicode,
    iter_input_items,
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


def test_jpeg_with_embedded_mpo_preview_yields_only_primary_frame(tmp_path) -> None:
    path = tmp_path / "camera.jpg"
    primary = Image.fromarray(np.full((24, 32, 3), 30, dtype=np.uint8))
    preview = Image.fromarray(np.full((8, 12, 3), 210, dtype=np.uint8))
    primary.save(path, format="MPO", save_all=True, append_images=[preview])

    items = load_input_items([path], pdf_dpi=72)

    assert len(items) == 1
    assert items[0][0] == "camera.jpg"
    assert items[0][1].shape == (24, 32, 3)
    assert round(float(items[0][1].mean())) == 30


def test_raster_input_fails_closed_above_configured_pixel_limit(tmp_path) -> None:
    path = tmp_path / "large-for-test.png"
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(path)

    with pytest.raises(RuntimeError, match=r"32x24 .*safe input limit: 700 pixels"):
        imread_unicode(path, max_pixels=700)


def test_each_tiff_frame_is_checked_before_it_is_decoded(tmp_path) -> None:
    path = tmp_path / "mixed-sizes.tiff"
    first = Image.fromarray(np.full((12, 16), 30, dtype=np.uint8))
    second = Image.fromarray(np.full((20, 30), 210, dtype=np.uint8))
    first.save(path, save_all=True, append_images=[second])

    pages = iter_input_items([path], pdf_dpi=72, max_input_pixels=300)
    assert next(pages)[0] == "mixed-sizes.tiff [p0001]"
    with pytest.raises(RuntimeError, match=r"frame 2: 30x20 .*safe input limit: 300 pixels"):
        next(pages)


def test_pillow_warning_is_handled_locally_without_weakening_global_guard(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "warning-threshold.png"
    Image.fromarray(np.zeros((11, 11), dtype=np.uint8)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = imread_unicode(path, max_pixels=150)

    assert loaded is not None
    assert Image.MAX_IMAGE_PIXELS == 100
    assert not any(isinstance(item.message, Image.DecompressionBombWarning) for item in caught)


def test_pillow_hard_decompression_bomb_guard_remains_enabled(tmp_path, monkeypatch) -> None:
    path = tmp_path / "hard-limit.png"
    Image.fromarray(np.zeros((15, 15), dtype=np.uint8)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(RuntimeError, match="Pillow's decompression-bomb safety limit"):
        imread_unicode(path, max_pixels=300)


def test_unidentified_raster_never_falls_back_to_unbounded_opencv_decode(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an image")
    monkeypatch.setattr(
        "uniscan.io.loaders.cv2.imdecode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe OpenCV fallback must not run")
        ),
    )

    assert imread_unicode(path) is None
    with pytest.raises(RuntimeError, match="Cannot safely read advertised raster"):
        load_input_items([path], pdf_dpi=72)


@pytest.mark.parametrize("max_pixels", (0, -1))
def test_raster_pixel_limit_must_be_positive(tmp_path, max_pixels: int) -> None:
    path = tmp_path / "small.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)

    with pytest.raises(ValueError, match="must be positive"):
        imread_unicode(path, max_pixels=max_pixels)


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


def test_imwrite_unicode_writes_encoded_buffer_without_tobytes_copy(tmp_path, monkeypatch) -> None:
    class EncodedBuffer(np.ndarray):
        def tobytes(self, *_args, **_kwargs):
            raise AssertionError("encoded buffer must be written without a full copy")

    encoded = np.arange(32, dtype=np.uint8).view(EncodedBuffer)
    monkeypatch.setattr("uniscan.io.loaders.cv2.imencode", lambda *_args: (True, encoded))
    target = tmp_path / "buffer.png"

    assert imwrite_unicode(target, _img(90)) is True
    assert target.read_bytes() == bytes(range(32))


def test_pdfium_page_selection_and_streaming_cancellation(tmp_path) -> None:
    path = tmp_path / "two-pages.pdf"
    _make_pdf(path)

    selected = render_pdf_page_indices(path, [1], dpi=72)
    assert [name for name, _image in selected] == ["two-pages.pdf [p0002]"]
    assert selected[0][1].shape[:2] == (100, 100)

    cancelled = False

    pages = iter_pdf_pages(path, dpi=72, cancel_cb=lambda: cancelled)
    assert next(pages)[0] == "two-pages.pdf [p0001]"
    cancelled = True
    with pytest.raises(RuntimeError, match="Cancelled"):
        next(pages)


def test_pdf_stream_checks_cancellation_immediately_after_native_render(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "one-page.pdf"
    _make_pdf(path)
    rendered = False
    real_render = _render_pdf_page

    def render_then_cancel(*args, **kwargs):
        nonlocal rendered
        item = real_render(*args, **kwargs)
        rendered = True
        return item

    monkeypatch.setattr("uniscan.io.loaders._render_pdf_page", render_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled"):
        next(iter_pdf_pages(path, dpi=72, cancel_cb=lambda: rendered))


def test_raster_stream_checks_cancellation_after_native_conversion(tmp_path, monkeypatch) -> None:
    path = tmp_path / "one-page.png"
    Image.fromarray(np.zeros((10, 12), dtype=np.uint8)).save(path)
    converted = False
    from uniscan.io import loaders

    real_convert = loaders._pil_frame_to_bgr

    def convert_then_cancel(frame):
        nonlocal converted
        image = real_convert(frame)
        converted = True
        return image

    monkeypatch.setattr(loaders, "_pil_frame_to_bgr", convert_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled"):
        next(iter_input_items([path], pdf_dpi=72, cancel_cb=lambda: converted))


def test_pdf_render_uses_options_supported_by_minimum_pdfium(tmp_path) -> None:
    class FakeBitmap:
        def to_numpy(self):
            return np.zeros((2, 3, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    class MinimumApiPage:
        def get_size(self):
            return 3.0, 2.0

        def render(self, *, scale, rev_byteorder, fill_color):
            assert scale == 1.0
            assert rev_byteorder is True
            assert fill_color == (255, 255, 255, 255)
            return FakeBitmap()

    name, rendered = _render_pdf_page(
        MinimumApiPage(),
        pdf_path=tmp_path / "minimum.pdf",
        page_index=0,
        dpi=72,
    )

    assert name == "minimum.pdf [p0001]"
    assert rendered.shape == (2, 3, 3)


def test_safe_render_dpi_calculates_limit_without_claiming_to_change_request() -> None:
    safe = _safe_render_dpi((720.0, 720.0), requested_dpi=600, max_pixels=1_000_000)
    assert safe == 100
    with pytest.raises(ValueError, match="must be positive"):
        _safe_render_dpi((720.0, 720.0), requested_dpi=600, max_pixels=0)


def test_pdf_render_checks_ceiled_dimensions_before_extreme_page_allocation(tmp_path) -> None:
    class SkinnyExtremePage:
        def get_size(self):
            return 0.01, 72_000_000.0

        def render(self, **_kwargs):
            raise AssertionError("oversized page must not be rendered")

    assert (
        _safe_render_dpi(
            SkinnyExtremePage().get_size(),
            requested_dpi=2,
            max_pixels=1_000_000,
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="exceeds the safe pixel limit"):
        _render_pdf_page(
            SkinnyExtremePage(),
            pdf_path=tmp_path / "skinny-extreme.pdf",
            page_index=0,
            dpi=2,
            max_pixels=1_000_000,
        )


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
