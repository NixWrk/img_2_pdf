"""GUI-independent adapters for clipboard and Tk drag-and-drop payloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from uniscan.io.loaders import DEFAULT_MAX_INPUT_PIXELS


def normalize_path_strings(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).expanduser()
        normalized = path.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(path)
    return paths


def paths_from_tk_drop(raw_data: str, splitlist: Callable[[str], tuple[str, ...]]) -> list[Path]:
    """Decode Tk's brace-aware DND_FILES payload."""
    return normalize_path_strings(splitlist(raw_data))


def clipboard_file_paths(payload: object) -> list[Path]:
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return normalize_path_strings(payload)
    return []


def clipboard_image_to_bgr(
    payload: object,
    *,
    max_pixels: int = DEFAULT_MAX_INPUT_PIXELS,
) -> np.ndarray | None:
    if not isinstance(payload, Image.Image):
        return None
    limit = int(max_pixels)
    if limit < 1:
        raise ValueError("Maximum input pixel count must be positive.")
    width, height = map(int, payload.size)
    pixels = width * height
    if width < 1 or height < 1 or pixels > limit:
        raise RuntimeError(
            f"Clipboard image: {width}x{height} ({pixels:,} pixels); "
            f"safe input limit: {limit:,} pixels."
        )
    rgb = np.asarray(payload.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
