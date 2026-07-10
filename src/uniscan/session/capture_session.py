"""Capture/session data model used by unified UI and export pipeline."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from uniscan.core.dewarp import normalize_control_points
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.storage import PagePaths, PageStore


@dataclass(slots=True)
class CaptureEntry:
    """Single page entry in a capture/import session."""

    name: str
    store: PageStore
    paths: PagePaths
    detected_contour: np.ndarray | None = None
    detected_backend: str | None = None
    dewarp_control_points: tuple[tuple[float, float], ...] | None = None
    selected: bool = False
    entry_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_image(cls, *, name: str, image: np.ndarray, store: PageStore) -> "CaptureEntry":
        """Create an entry from a single image (raw == warped, no contour)."""
        return cls.from_raw_and_warped(
            name=name,
            raw_image=image,
            warped_image=image,
            contour=None,
            backend=None,
            store=store,
        )

    @classmethod
    def from_raw_and_warped(
        cls,
        *,
        name: str,
        raw_image: np.ndarray,
        warped_image: np.ndarray,
        contour: np.ndarray | None,
        backend: str | None,
        store: PageStore,
    ) -> "CaptureEntry":
        entry_id = uuid4().hex
        paths = store.add_page(entry_id, raw_image, warped_image)
        return cls(
            name=name,
            store=store,
            paths=paths,
            detected_contour=contour,
            detected_backend=backend,
            entry_id=entry_id,
        )

    # Path shortcuts (backwards-compatible attributes used by existing code/tests).

    @property
    def raw_path(self) -> Path:
        return self.paths.raw

    @property
    def original_path(self) -> Path:
        return self.paths.original

    @property
    def current_path(self) -> Path:
        return self.paths.current

    @property
    def preview_raw_path(self) -> Path:
        return self.paths.preview_raw

    @property
    def preview_original_path(self) -> Path:
        return self.paths.preview_original

    @property
    def preview_current_path(self) -> Path:
        return self.paths.preview_current

    @property
    def thumb_path(self) -> Path:
        return self.paths.thumb

    # Image accessors.

    @property
    def raw_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.raw)

    @property
    def preview_raw_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.preview_raw)

    @property
    def original_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.original)

    @original_image.setter
    def original_image(self, image: np.ndarray) -> None:
        self.store.write_image(self.paths.original, image)
        self.store.write_preview(self.paths.preview_original, image)
        self.dewarp_control_points = None

    @property
    def current_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.current)

    @current_image.setter
    def current_image(self, image: np.ndarray) -> None:
        self.store.write_image(self.paths.current, image)
        self.store.write_preview(self.paths.preview_current, image)
        self.store.write_thumbnail(self.paths.thumb, image)

    @property
    def preview_original_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.preview_original)

    @property
    def preview_current_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.preview_current)

    @property
    def thumbnail_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.thumb)

    def replace_raw(self, raw_image: np.ndarray) -> None:
        """Replace the immutable raw source (used when retaking or replacing a page)."""
        self.store.write_image(self.paths.raw, raw_image)
        self.store.write_preview(self.paths.preview_raw, raw_image)

    def set_dewarp_control_points(
        self,
        control_points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    ) -> None:
        self.dewarp_control_points = normalize_control_points(control_points)

    def clear_dewarp_control_points(self) -> None:
        self.dewarp_control_points = None


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

    def add_image_with_contour(
        self,
        *,
        name: str,
        raw_image: np.ndarray,
        warped_image: np.ndarray,
        contour: np.ndarray | None,
        backend: str | None,
    ) -> CaptureEntry:
        entry = CaptureEntry.from_raw_and_warped(
            name=name,
            raw_image=raw_image,
            warped_image=warped_image,
            contour=contour,
            backend=backend,
            store=self.store,
        )
        self._entries.append(entry)
        return entry

    def add_images(self, items: list[tuple[str, np.ndarray]]) -> list[CaptureEntry]:
        added: list[CaptureEntry] = []
        for name, image in items:
            added.append(self.add_image(name=name, image=image))
        return added

    def insert_entry_after(self, after_entry_id: str, entry: CaptureEntry) -> bool:
        index = self._find_index(after_entry_id)
        if index is None:
            return False
        self._entries.insert(index + 1, entry)
        return True

    def move(self, entry_id: str, distance: int) -> bool:
        """Move entry up/down by distance and return whether move succeeded."""
        index = self._find_index(entry_id)
        if index is None:
            return False
        new_index = index + distance
        if new_index < 0 or new_index >= len(self._entries):
            return False
        self._entries[index], self._entries[new_index] = (
            self._entries[new_index],
            self._entries[index],
        )
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

    def remove_entry(self, entry_id: str) -> bool:
        index = self._find_index(entry_id)
        if index is None:
            return False
        self.store.remove_page(entry_id)
        del self._entries[index]
        return True

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
        raw_image: np.ndarray | None = None,
        contour: np.ndarray | None = None,
        backend: str | None = None,
    ) -> bool:
        """Replace entry images in-place while preserving ordering and identity."""
        index = self._find_index(entry_id)
        if index is None:
            return False

        entry = self._entries[index]
        if raw_image is not None:
            entry.replace_raw(raw_image)
        entry.original_image = original_image
        entry.current_image = original_image if current_image is None else current_image
        if name is not None and name.strip():
            entry.name = name.strip()
        if contour is not None or backend is not None:
            entry.detected_contour = contour
            entry.detected_backend = backend
        return True

    def selected_entries(self) -> list[CaptureEntry]:
        return [entry for entry in self._entries if entry.selected]

    def save_manifest(self, manifest_path: Path) -> Path:
        """Atomically save enough metadata to reopen this disk-backed session."""
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": 1,
            "sessionDir": str(self.store.session_dir.resolve()),
            "entries": [
                {
                    "entryId": entry.entry_id,
                    "name": entry.name,
                    "selected": entry.selected,
                    "detectedBackend": entry.detected_backend,
                    "detectedContour": (
                        entry.detected_contour.tolist()
                        if entry.detected_contour is not None
                        else None
                    ),
                    "dewarpControlPoints": (
                        [list(point) for point in entry.dewarp_control_points]
                        if entry.dewarp_control_points is not None
                        else None
                    ),
                }
                for entry in self._entries
            ],
        }
        descriptor, raw_stage = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.stage-",
            suffix=".json",
            dir=manifest_path.parent,
        )
        os.close(descriptor)
        stage = Path(raw_stage)
        try:
            stage.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(stage, manifest_path)
        finally:
            if stage.exists():
                stage.unlink()
        return manifest_path

    @classmethod
    def restore_manifest(
        cls,
        manifest_path: Path,
        *,
        allowed_sessions_root: Path | None = None,
    ) -> "CaptureSession":
        """Restore entry order and metadata from a persistent session manifest."""
        manifest_path = Path(manifest_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read session manifest: {manifest_path}") from exc
        if payload.get("schemaVersion") != 1 or not isinstance(payload.get("entries"), list):
            raise ValueError(f"Unsupported session manifest: {manifest_path}")

        session_dir_raw = payload.get("sessionDir")
        if not isinstance(session_dir_raw, str):
            raise ValueError(f"Invalid session directory in manifest: {manifest_path}")
        session_dir = Path(session_dir_raw).resolve()
        if allowed_sessions_root is not None:
            allowed_root = Path(allowed_sessions_root).resolve()
            if session_dir.parent != allowed_root:
                raise ValueError(f"Session directory escapes autosave storage: {session_dir}")

        store = PageStore.from_session_dir(session_dir)
        session = cls(store=store)
        for item in payload["entries"]:
            if not isinstance(item, dict):
                raise ValueError("Invalid session entry in manifest.")
            entry_id = str(item.get("entryId", ""))
            if re.fullmatch(r"[0-9a-f]{32}", entry_id) is None:
                raise ValueError(f"Invalid session entry id: {entry_id}")
            paths = store.paths_for_entry(entry_id)
            required = (paths.raw, paths.original, paths.current)
            if not all(path.is_file() for path in required):
                raise ValueError(f"Session page assets are incomplete: {entry_id}")
            contour_raw = item.get("detectedContour")
            contour = np.asarray(contour_raw, dtype=np.float32) if contour_raw is not None else None
            if contour is not None and (contour.shape != (4, 2) or not np.isfinite(contour).all()):
                raise ValueError(f"Invalid detected contour for session entry: {entry_id}")
            control_points_raw = item.get("dewarpControlPoints")
            try:
                control_points = (
                    normalize_control_points(control_points_raw)
                    if control_points_raw is not None
                    else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid dewarp control points for session entry: {entry_id}"
                ) from exc
            session.add_entry(
                CaptureEntry(
                    name=str(item.get("name", entry_id)),
                    store=store,
                    paths=paths,
                    detected_contour=contour,
                    detected_backend=item.get("detectedBackend"),
                    dewarp_control_points=control_points,
                    selected=bool(item.get("selected", False)),
                    entry_id=entry_id,
                )
            )
        return session

    def close(self, *, preserve: bool = False) -> None:
        if preserve:
            self.store.close()
            return
        self.clear()
        self.store.discard()

    def _find_index(self, entry_id: str) -> int | None:
        for idx, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return idx
        return None
