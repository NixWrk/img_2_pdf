from __future__ import annotations

import numpy as np
from PIL import Image

from uniscan.io.loaders import imread_unicode


def test_imread_unicode_applies_exif_orientation(tmp_path) -> None:
    path = tmp_path / "oriented.jpg"
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = (255, 0, 0)
    image = Image.fromarray(pixels, mode="RGB")
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    image.save(path, quality=100, exif=exif)

    loaded = imread_unicode(path)

    assert loaded is not None
    assert loaded.shape[:2] == (40, 20)
    # RGB red becomes BGR red and moves to the top after clockwise rotation.
    assert float(loaded[:18, :, 2].mean()) > 220
    assert float(loaded[-18:, :, 2].mean()) < 30
