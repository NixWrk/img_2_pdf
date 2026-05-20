"""Disk-backed page storage to keep memory usage low."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from uniscan.io.loaders import imwrite_unicode


@dataclass(slots=True, frozen=True)
class PagePaths:
    """File paths for a single page entry on disk."""

    raw: Path
    original: Path
    current: Path
    preview_raw: Path
    preview_original: Path
    preview_current: Path
    thumb: Path


class PageStore:
    """Manage per-session page files (raw/original/current/previews/thumbnail)."""

    def __init__(self, root_dir: Path | None = None, *, keep_on_close: bool = False) -> None:
        base = Path(root_dir) if root_dir is not None else Path(tempfile.gettempdir()) / "uniscan_cache"
        self.session_id = uuid4().hex
        self.session_dir = base / self.session_id
        self.pages_dir = self.session_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.keep_on_close = keep_on_close

    def paths_for_entry(self, entry_id: str) -> PagePaths:
        page_dir = self.pages_dir / entry_id
        page_dir.mkdir(parents=True, exist_ok=True)
        return PagePaths(
            raw=page_dir / "raw.png",
            original=page_dir / "original.png",
            current=page_dir / "current.png",
            preview_raw=page_dir / "preview_raw.jpg",
            preview_original=page_dir / "preview_original.jpg",
            preview_current=page_dir / "preview_current.jpg",
            thumb=page_dir / "thumb.jpg",
        )

    def read_image(self, path: Path) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read page image: {path}")
        return image

    def write_image(self, path: Path, image: np.ndarray) -> None:
        if not imwrite_unicode(path, image):
            raise RuntimeError(f"Cannot write page image: {path}")

    def _resize_for_display(
        self,
        image: np.ndarray,
        *,
        max_width: int,
        max_height: int,
    ) -> np.ndarray:
        if len(image.shape) == 2:
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            preview = image
        h, w = preview.shape[:2]
        scale = min(max_width / max(1, w), max_height / max(1, h), 1.0)
        if scale >= 1.0:
            return preview
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def write_preview(self, path: Path, image: np.ndarray, *, max_width: int = 1920, max_height: int = 1080) -> None:
        preview = self._resize_for_display(image, max_width=max_width, max_height=max_height)
        if not imwrite_unicode(path, preview):
            raise RuntimeError(f"Cannot write page preview: {path}")

    def write_thumbnail(self, path: Path, image: np.ndarray, *, max_side: int = 320) -> None:
        thumb = self._resize_for_display(image, max_width=max_side, max_height=max_side)
        if not imwrite_unicode(path, thumb):
            raise RuntimeError(f"Cannot write page thumbnail: {path}")

    def add_page(
        self,
        entry_id: str,
        raw_image: np.ndarray,
        warped_image: np.ndarray | None = None,
    ) -> PagePaths:
        """
        Persist a page to disk.

        `raw_image` is the immutable source. `warped_image` is the rectified
        result (after document detection / UVDoc). If `warped_image` is None,
        the raw image is used for both (no rectification was applied).
        """
        if warped_image is None:
            warped_image = raw_image
        paths = self.paths_for_entry(entry_id)
        self.write_image(paths.raw, raw_image)
        self.write_image(paths.original, warped_image)
        self.write_image(paths.current, warped_image)
        self.write_preview(paths.preview_raw, raw_image)
        self.write_preview(paths.preview_original, warped_image)
        self.write_preview(paths.preview_current, warped_image)
        self.write_thumbnail(paths.thumb, warped_image)
        return paths

    def remove_page(self, entry_id: str) -> None:
        page_dir = self.pages_dir / entry_id
        shutil.rmtree(page_dir, ignore_errors=True)

    def close(self) -> None:
        if self.keep_on_close:
            return
        shutil.rmtree(self.session_dir, ignore_errors=True)
