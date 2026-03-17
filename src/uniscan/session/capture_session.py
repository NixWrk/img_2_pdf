"""Capture/session data model used by unified UI and export pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.storage import PageStore


@dataclass(slots=True)
class CaptureEntry:
    """Single page entry in a capture/import session."""

    name: str
    store: PageStore
    original_path: Path
    current_path: Path
    thumb_path: Path
    selected: bool = False
    entry_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_image(cls, *, name: str, image: np.ndarray, store: PageStore) -> "CaptureEntry":
        entry_id = uuid4().hex
        original_path, current_path, thumb_path = store.add_page(entry_id, image)
        return cls(
            name=name,
            store=store,
            original_path=original_path,
            current_path=current_path,
            thumb_path=thumb_path,
            entry_id=entry_id,
        )

    @property
    def original_image(self) -> np.ndarray:
        return self.store.read_image(self.original_path)

    @original_image.setter
    def original_image(self, image: np.ndarray) -> None:
        self.store.write_image(self.original_path, image)

    @property
    def current_image(self) -> np.ndarray:
        return self.store.read_image(self.current_path)

    @current_image.setter
    def current_image(self, image: np.ndarray) -> None:
        self.store.write_image(self.current_path, image)
        self.store.write_thumbnail(self.thumb_path, image)

    @property
    def thumbnail_image(self) -> np.ndarray:
        return self.store.read_image(self.thumb_path)


class CaptureSession:
    """Ordered page session with disk-backed image storage."""

    def __init__(self, store: PageStore | None = None) -> None:
        self.store = store or PageStore()
        self._entries: list[CaptureEntry] = []

    @property
    def entries(self) -> list[CaptureEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        for entry in self._entries:
            self.store.remove_page(entry.entry_id)
        self._entries.clear()

    def add_entry(self, entry: CaptureEntry) -> None:
        self._entries.append(entry)

    def add_image(self, *, name: str, image: np.ndarray) -> CaptureEntry:
        entry = CaptureEntry.from_image(name=name, image=image, store=self.store)
        self._entries.append(entry)
        return entry

    def add_images(self, items: list[tuple[str, np.ndarray]]) -> list[CaptureEntry]:
        added: list[CaptureEntry] = []
        for name, image in items:
            added.append(self.add_image(name=name, image=image))
        return added

    def move(self, entry_id: str, distance: int) -> bool:
        """Move entry up/down by distance and return whether move succeeded."""
        index = self._find_index(entry_id)
        if index is None:
            return False
        new_index = index + distance
        if new_index < 0 or new_index >= len(self._entries):
            return False
        self._entries[index], self._entries[new_index] = self._entries[new_index], self._entries[index]
        return True

    def select_all(self, selected: bool = True) -> None:
        for entry in self._entries:
            entry.selected = selected

    def remove_selected(self) -> int:
        before = len(self._entries)
        kept: list[CaptureEntry] = []
        for entry in self._entries:
            if entry.selected:
                self.store.remove_page(entry.entry_id)
            else:
                kept.append(entry)
        self._entries = kept
        return before - len(self._entries)

    def apply_postprocess(self, postprocess_name: str) -> None:
        if postprocess_name not in POSTPROCESSING_OPTIONS:
            raise ValueError(f"Unsupported postprocess mode: {postprocess_name}")
        post_fn = POSTPROCESSING_OPTIONS[postprocess_name]
        for entry in self._entries:
            entry.current_image = post_fn(entry.original_image)

    def replace_entry_image(
        self,
        entry_id: str,
        *,
        original_image: np.ndarray,
        current_image: np.ndarray | None = None,
        name: str | None = None,
    ) -> bool:
        """Replace entry images in-place while preserving ordering and identity."""
        index = self._find_index(entry_id)
        if index is None:
            return False

        entry = self._entries[index]
        entry.original_image = original_image
        entry.current_image = original_image if current_image is None else current_image
        if name is not None and name.strip():
            entry.name = name.strip()
        return True

    def selected_entries(self) -> list[CaptureEntry]:
        return [entry for entry in self._entries if entry.selected]

    def close(self) -> None:
        self.clear()
        self.store.close()

    def _find_index(self, entry_id: str) -> int | None:
        for idx, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return idx
        return None
