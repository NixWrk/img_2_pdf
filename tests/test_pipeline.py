import numpy as np

from uniscan.core.pipeline import PipelineOptions, process_loaded_items, split_spread


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
