from pathlib import Path

import numpy as np

from uniscan.core.pipeline import (
    PipelineOptions,
    build_pdf_from_images,
    process_loaded_items,
    split_spread,
)


def _img() -> np.ndarray:
    out = np.zeros((20, 40, 3), dtype=np.uint8)
    out[:, :20] = (10, 20, 30)
    out[:, 20:] = (40, 50, 60)
    return out


def test_split_spread_returns_two_pages() -> None:
    image = _img()
    pages = split_spread(image)

    assert len(pages) == 2
    assert pages[0].shape == (20, 20, 3)
    assert pages[1].shape == (20, 20, 3)


def test_process_loaded_items_without_detector() -> None:
    loaded = [("sample.png", _img())]
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=True,
        postprocess_name="None",
    )
    pages = process_loaded_items(loaded, options=options)
    assert len(pages) == 2


def test_process_loaded_items_returns_page_results_with_raw() -> None:
    loaded = [("sample.png", _img())]
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=False,
        postprocess_name="None",
    )
    pages = process_loaded_items(loaded, options=options)
    assert len(pages) == 1
    page = pages[0]
    assert page.name == "sample.png"
    assert page.raw is not None
    assert page.warped is not None
    assert page.current is not None
    assert page.raw.shape == _img().shape
    assert page.detected is False
    assert page.fallback_reason is None


def _spread_image() -> np.ndarray:
    width, height = 800, 500
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    image[:, 395:405] = 30  # dark gutter
    return image


def test_process_loaded_items_two_page_mode_splits_at_gutter() -> None:
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=True,
        postprocess_name="None",
    )
    pages = process_loaded_items([("spread.png", _spread_image())], options=options)
    assert len(pages) == 2
    # Left half should be roughly half the width, plus or minus the gutter offset.
    assert 350 < pages[0].warped.shape[1] < 450
    assert pages[0].name.endswith("[L]")
    assert pages[1].name.endswith("[R]")


def test_build_pdf_writes_to_stream_and_truncates_before_compatibility_fallback(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output.pdf"
    calls: list[dict[str, object]] = []

    def fake_convert(paths: list[str], **kwargs) -> None:
        assert paths == [str(Path("page.png"))]
        calls.append(kwargs)
        stream = kwargs["outputstream"]
        if "dpi" in kwargs:
            stream.write(b"partial")
            raise TypeError("legacy dpi API")
        stream.write(b"%PDF-streamed")

    monkeypatch.setattr("uniscan.core.pipeline.img2pdf.convert", fake_convert)
    monkeypatch.setattr(
        "uniscan.core.pipeline.img2pdf.get_fixed_dpi_layout_fun",
        lambda dpi: ("layout", dpi),
    )

    build_pdf_from_images([Path("page.png")], output, 240)

    assert output.read_bytes() == b"%PDF-streamed"
    assert calls[0]["dpi"] == 240
    assert calls[1]["layout_fun"] == ("layout", (240, 240))
