"""Disk-backed page storage to keep memory usage low."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from uniscan.io.loaders import imread_unicode, imwrite_unicode


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

    def __init__(
        self,
        root_dir: Path | None = None,
        *,
        keep_on_close: bool = False,
        session_id: str | None = None,
    ) -> None:
        base = (
            Path(root_dir)
            if root_dir is not None
            else Path(tempfile.gettempdir()) / "uniscan_cache"
        )
        self.session_id = session_id or uuid4().hex
        self.session_dir = base / self.session_id
        self.pages_dir = self.session_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.keep_on_close = keep_on_close
        self._lock = threading.RLock()

    @classmethod
    def from_session_dir(cls, session_dir: Path) -> "PageStore":
        """Open an existing persistent session directory."""
        session_dir = Path(session_dir)
        if not session_dir.is_dir():
            raise ValueError(f"Session directory does not exist: {session_dir}")
        return cls(
            root_dir=session_dir.parent,
            keep_on_close=True,
            session_id=session_dir.name,
        )

    def paths_for_entry(self, entry_id: str) -> PagePaths:
        with self._lock:
            page_dir = self._recover_page_locked(entry_id)
            return self._paths_for_page_dir(page_dir)

    def _page_directories(self, entry_id: str) -> tuple[Path, Path, Path]:
        return (
            self.pages_dir / entry_id,
            self.pages_dir / f".{entry_id}.stage",
            self.pages_dir / f".{entry_id}.backup",
        )

    @staticmethod
    def _paths_for_page_dir(page_dir: Path) -> PagePaths:
        return PagePaths(
            raw=page_dir / "raw.png",
            original=page_dir / "original.png",
            current=page_dir / "current.png",
            preview_raw=page_dir / "preview_raw.jpg",
            preview_original=page_dir / "preview_original.jpg",
            preview_current=page_dir / "preview_current.jpg",
            thumb=page_dir / "thumb.jpg",
        )

    @staticmethod
    def _read_image_file(path: Path) -> np.ndarray:
        image = imread_unicode(path, preserve_channels=True)
        if image is None:
            raise RuntimeError(f"Cannot read page image: {path}")
        return image

    def read_image(self, path: Path) -> np.ndarray:
        path = Path(path)
        with self._lock:
            if path.parent.parent == self.pages_dir and not path.parent.name.startswith("."):
                self._recover_page_locked(path.parent.name)
            return self._read_image_file(path)

    def write_image(self, path: Path, image: np.ndarray) -> None:
        with self._lock:
            self._atomic_image_write(path, image, kind="page image")

    @staticmethod
    def _atomic_image_write(path: Path, image: np.ndarray, *, kind: str) -> None:
        """Encode beside the destination, then atomically publish the complete file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = path.with_name(f".{path.stem}.stage-{uuid4().hex}{path.suffix}")
        try:
            if not imwrite_unicode(stage, image):
                raise RuntimeError(f"Cannot write {kind}: {path}")
            os.replace(stage, path)
        finally:
            stage.unlink(missing_ok=True)

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

    def write_preview(
        self, path: Path, image: np.ndarray, *, max_width: int = 1920, max_height: int = 1080
    ) -> None:
        preview = self._resize_for_display(image, max_width=max_width, max_height=max_height)
        self._atomic_image_write(path, preview, kind="page preview")

    def write_thumbnail(self, path: Path, image: np.ndarray, *, max_side: int = 320) -> None:
        thumb = self._resize_for_display(image, max_width=max_side, max_height=max_side)
        self._atomic_image_write(path, thumb, kind="page thumbnail")

    def _write_page_set(
        self,
        page_dir: Path,
        *,
        raw_image: np.ndarray,
        warped_image: np.ndarray,
        current_image: np.ndarray,
    ) -> PagePaths:
        paths = self._paths_for_page_dir(page_dir)
        self._atomic_image_write(paths.raw, raw_image, kind="page image")
        self._atomic_image_write(paths.original, warped_image, kind="page image")
        self._atomic_image_write(paths.current, current_image, kind="page image")
        self._atomic_image_write(
            paths.preview_raw,
            self._resize_for_display(raw_image, max_width=1920, max_height=1080),
            kind="page preview",
        )
        self._atomic_image_write(
            paths.preview_original,
            self._resize_for_display(warped_image, max_width=1920, max_height=1080),
            kind="page preview",
        )
        self._atomic_image_write(
            paths.preview_current,
            self._resize_for_display(current_image, max_width=1920, max_height=1080),
            kind="page preview",
        )
        self._atomic_image_write(
            paths.thumb,
            self._resize_for_display(current_image, max_width=320, max_height=320),
            kind="page thumbnail",
        )
        return paths

    @staticmethod
    def _link_or_copy_file(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def _stage_page_set_update(
        self,
        existing: PagePaths,
        stage_dir: Path,
        *,
        raw_image: np.ndarray | None,
        original_image: np.ndarray | None,
        current_image: np.ndarray | None,
    ) -> None:
        """Stage changed asset groups and link/copy every unchanged group."""
        staged = self._paths_for_page_dir(stage_dir)
        if raw_image is None:
            self._link_or_copy_file(existing.raw, staged.raw)
            self._link_or_copy_file(existing.preview_raw, staged.preview_raw)
        else:
            self._atomic_image_write(staged.raw, raw_image, kind="page image")
            self._atomic_image_write(
                staged.preview_raw,
                self._resize_for_display(raw_image, max_width=1920, max_height=1080),
                kind="page preview",
            )

        if original_image is None:
            self._link_or_copy_file(existing.original, staged.original)
            self._link_or_copy_file(existing.preview_original, staged.preview_original)
        else:
            self._atomic_image_write(staged.original, original_image, kind="page image")
            self._atomic_image_write(
                staged.preview_original,
                self._resize_for_display(original_image, max_width=1920, max_height=1080),
                kind="page preview",
            )

        if current_image is None:
            self._link_or_copy_file(existing.current, staged.current)
            self._link_or_copy_file(existing.preview_current, staged.preview_current)
            self._link_or_copy_file(existing.thumb, staged.thumb)
        else:
            self._atomic_image_write(staged.current, current_image, kind="page image")
            self._atomic_image_write(
                staged.preview_current,
                self._resize_for_display(current_image, max_width=1920, max_height=1080),
                kind="page preview",
            )
            self._atomic_image_write(
                staged.thumb,
                self._resize_for_display(current_image, max_width=320, max_height=320),
                kind="page thumbnail",
            )

    def _page_set_is_valid(self, page_dir: Path) -> bool:
        paths = self._paths_for_page_dir(page_dir)
        if not all(
            path.is_file()
            for path in (
                paths.raw,
                paths.original,
                paths.current,
                paths.preview_raw,
                paths.preview_original,
                paths.preview_current,
                paths.thumb,
            )
        ):
            return False
        try:
            self._read_image_file(paths.raw)
            self._read_image_file(paths.original)
            self._read_image_file(paths.current)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def _recover_page_locked(self, entry_id: str) -> Path:
        """Resolve an interrupted deterministic directory swap to one generation."""
        page_dir, stage_dir, backup_dir = self._page_directories(entry_id)
        if backup_dir.exists():
            if page_dir.exists() and self._page_set_is_valid(page_dir):
                # New generation was published; only backup cleanup was interrupted.
                shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                # Publication did not complete (or the new generation is corrupt).
                shutil.rmtree(page_dir, ignore_errors=True)
                os.replace(backup_dir, page_dir)
            shutil.rmtree(stage_dir, ignore_errors=True)
        elif stage_dir.exists():
            # Without a backup there is no committed update to finish.  The
            # manifest either references the intact target or no page yet.
            shutil.rmtree(stage_dir, ignore_errors=True)
        return page_dir

    def _replace_page_set_locked(
        self,
        entry_id: str,
        *,
        raw_image: np.ndarray | None,
        original_image: np.ndarray | None,
        current_image: np.ndarray | None,
    ) -> PagePaths:
        page_dir = self._recover_page_locked(entry_id)
        _page_dir, stage_dir, backup_dir = self._page_directories(entry_id)
        if not page_dir.is_dir():
            raise RuntimeError(f"Page does not exist: {entry_id}")
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
        try:
            self._stage_page_set_update(
                self._paths_for_page_dir(page_dir),
                stage_dir,
                raw_image=raw_image,
                original_image=original_image,
                current_image=current_image,
            )
            os.replace(page_dir, backup_dir)
            try:
                os.replace(stage_dir, page_dir)
            except Exception:
                os.replace(backup_dir, page_dir)
                raise
            shutil.rmtree(backup_dir, ignore_errors=True)
        finally:
            # If this is an ordinary exception (not process death), recover now.
            if backup_dir.exists():
                self._recover_page_locked(entry_id)
            shutil.rmtree(stage_dir, ignore_errors=True)
        return self._paths_for_page_dir(page_dir)

    def replace_page_set(
        self,
        entry_id: str,
        *,
        raw_image: np.ndarray | None = None,
        original_image: np.ndarray | None = None,
        current_image: np.ndarray | None = None,
    ) -> PagePaths:
        """Publish a complete page generation, recovering either side of a crash."""
        with self._lock:
            self._recover_page_locked(entry_id)
            return self._replace_page_set_locked(
                entry_id,
                raw_image=raw_image,
                original_image=original_image,
                current_image=current_image,
            )

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
        with self._lock:
            page_dir, stage_dir, backup_dir = self._page_directories(entry_id)
            self._recover_page_locked(entry_id)
            if page_dir.exists():
                raise RuntimeError(f"Page already exists: {entry_id}")
            shutil.rmtree(stage_dir, ignore_errors=True)
            shutil.rmtree(backup_dir, ignore_errors=True)
            try:
                self._write_page_set(
                    stage_dir,
                    raw_image=raw_image,
                    warped_image=warped_image,
                    current_image=warped_image,
                )
                # A manifest can only ever observe no page or a complete page set.
                os.replace(stage_dir, page_dir)
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
            return self._paths_for_page_dir(page_dir)

    def remove_page(self, entry_id: str) -> None:
        with self._lock:
            for page_dir in self._page_directories(entry_id):
                shutil.rmtree(page_dir, ignore_errors=True)

    def prune_pages(self, live_entry_ids: set[str]) -> None:
        """Remove pages only after a manifest no longer references them."""
        with self._lock:
            for entry_id in live_entry_ids:
                self._recover_page_locked(entry_id)
            for page_dir in self.pages_dir.iterdir():
                if not page_dir.is_dir():
                    continue
                if page_dir.name.startswith("."):
                    if page_dir.name.endswith((".stage", ".backup")):
                        shutil.rmtree(page_dir, ignore_errors=True)
                    continue
                if page_dir.name not in live_entry_ids:
                    shutil.rmtree(page_dir, ignore_errors=True)

    def ensure_derived_assets(self, paths: PagePaths) -> None:
        """Rebuild disposable previews after an interrupted write or old session restore."""
        self.repair_page_assets(paths.raw.parent.name)

    def repair_page_assets(self, entry_id: str) -> tuple[PagePaths, bool]:
        """Rebuild derived assets and recover `current` from a valid original."""
        with self._lock:
            page_dir = self._recover_page_locked(entry_id)
            recovered = self._paths_for_page_dir(page_dir)
            raw = self._read_image_file(recovered.raw)
            original = self._read_image_file(recovered.original)
            current_rebuilt = False
            try:
                current = self._read_image_file(recovered.current)
            except (OSError, RuntimeError, ValueError):
                current = original.copy()
                current_rebuilt = True
            paths = self._replace_page_set_locked(
                entry_id,
                raw_image=raw,
                original_image=original,
                current_image=current,
            )
            return paths, current_rebuilt

    def snapshot_image(self, source: Path, destination: Path) -> Path:
        """Create an immutable hard-link/copy snapshot of one committed image."""
        source = Path(source)
        destination = Path(destination)
        with self._lock:
            if source.parent.parent == self.pages_dir:
                self._recover_page_locked(source.parent.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        return destination

    def close(self) -> None:
        with self._lock:
            if self.keep_on_close:
                return
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def discard(self) -> None:
        """Remove the session directory even for persistent stores."""
        with self._lock:
            shutil.rmtree(self.session_dir, ignore_errors=True)
