from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from uniscan.ui.import_sources import (
    clipboard_file_paths,
    clipboard_image_to_bgr,
    normalize_path_strings,
    paths_from_tk_drop,
)


def test_normalize_path_strings_deduplicates(tmp_path) -> None:
    path = tmp_path / "a.png"
    assert normalize_path_strings([str(path), str(path)]) == [path]


def test_paths_from_tk_drop_uses_tk_splitter() -> None:
    raw = r"{C:\My Files\a.png} C:\b.png"
    paths = paths_from_tk_drop(
        raw,
        lambda _raw: (r"C:\My Files\a.png", r"C:\b.png"),
    )
    assert [path.name for path in paths] == ["a.png", "b.png"]


def test_clipboard_adapters() -> None:
    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    bgr = clipboard_image_to_bgr(image)
    assert bgr is not None
    assert bgr.shape == (2, 3, 3)
    np.testing.assert_array_equal(bgr[0, 0], [30, 20, 10])

    files = clipboard_file_paths(["a.png", "b.pdf"])
    assert [path.name for path in files] == ["a.png", "b.pdf"]
    assert clipboard_image_to_bgr(["a.png"]) is None


def test_clipboard_image_pixel_cap_is_checked_before_conversion() -> None:
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    with pytest.raises(RuntimeError, match="Clipboard image.*safe input limit"):
        clipboard_image_to_bgr(image, max_pixels=11)
    with pytest.raises(ValueError, match="must be positive"):
        clipboard_image_to_bgr(image, max_pixels=0)
