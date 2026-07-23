"""Capture/session data model used by unified UI and export pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np

from uniscan.core.dewarp import DewarpModel, normalize_control_curves, normalize_control_points
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.preprocess import PreprocessSettings
from uniscan.core.processing import PageProcessingDiagnostics, PageProcessingRequest
from uniscan.storage import PagePaths, PageStore


CROP_STATE_NONE = "none"
CROP_STATE_PROPOSED = "proposed"
CROP_STATE_APPLIED = "applied"
CROP_STATES = frozenset({CROP_STATE_NONE, CROP_STATE_PROPOSED, CROP_STATE_APPLIED})


def _resolve_crop_state(
    crop_state: str | None, contour: np.ndarray | None, backend: str | None
) -> str:
    resolved = crop_state
    if resolved is None:
        resolved = (
            CROP_STATE_APPLIED if contour is not None or backend is not None else CROP_STATE_NONE
        )
    if resolved not in CROP_STATES:
        raise ValueError(f"Invalid crop state: {resolved}")
    if resolved == CROP_STATE_PROPOSED and contour is None:
        raise ValueError("Proposed crop state requires a contour.")
    if resolved == CROP_STATE_NONE and (contour is not None or backend is not None):
        raise ValueError("Empty crop state cannot retain contour or backend metadata.")
    return resolved


@dataclass(slots=True, frozen=True)
class ProcessingRecipe:
    """Serializable settings required to reproduce one committed page."""

    schema_version: int
    orientation_method: str
    deskew_method: str
    dewarp_method: str
    dewarp_model: dict[str, object] | None
    dewarp_already_applied: bool
    uvdoc_cache_home: str | None
    auto_dewarp_uvdoc: bool
    auto_dewarp_uvdoc_grid: bool
    shadow_method: str
    postprocess_name: str
    preprocess_settings: dict[str, object] | None
    page_layout: str
    page_dpi: int
    page_margin_mm: float
    horizontal_alignment: str
    vertical_alignment: str
    lighting_diagnostics: bool
    source_fingerprint: str | None

    @classmethod
    def from_request(cls, request: PageProcessingRequest) -> "ProcessingRecipe":
        return cls(
            schema_version=2,
            orientation_method=request.orientation_method,
            deskew_method=request.deskew_method,
            dewarp_method=request.dewarp_method,
            dewarp_model=asdict(request.dewarp_model) if request.dewarp_model else None,
            dewarp_already_applied=request.dewarp_already_applied,
            uvdoc_cache_home=str(request.uvdoc_cache_home) if request.uvdoc_cache_home else None,
            auto_dewarp_uvdoc=request.auto_dewarp_uvdoc,
            auto_dewarp_uvdoc_grid=request.auto_dewarp_uvdoc_grid,
            shadow_method=request.shadow_method,
            postprocess_name=request.postprocess_name,
            preprocess_settings=(
                asdict(request.preprocess_settings) if request.preprocess_settings else None
            ),
            page_layout=request.page_layout,
            page_dpi=request.page_dpi,
            page_margin_mm=request.page_margin_mm,
            horizontal_alignment=request.horizontal_alignment,
            vertical_alignment=request.vertical_alignment,
            lighting_diagnostics=request.lighting_diagnostics,
            source_fingerprint=request.source_fingerprint,
        )

    def to_request(self) -> PageProcessingRequest:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported processing recipe: {self.schema_version}")
        model = DewarpModel(**self.dewarp_model) if self.dewarp_model is not None else None
        settings = (
            PreprocessSettings(**self.preprocess_settings)
            if self.preprocess_settings is not None
            else None
        )
        return PageProcessingRequest(
            orientation_method=self.orientation_method,
            deskew_method=self.deskew_method,
            dewarp_method=self.dewarp_method,
            dewarp_model=model,
            dewarp_already_applied=self.dewarp_already_applied,
            uvdoc_cache_home=Path(self.uvdoc_cache_home) if self.uvdoc_cache_home else None,
            auto_dewarp_uvdoc=self.auto_dewarp_uvdoc,
            auto_dewarp_uvdoc_grid=self.auto_dewarp_uvdoc_grid,
            shadow_method=self.shadow_method,
            postprocess_name=self.postprocess_name,
            preprocess_settings=settings,
            page_layout=self.page_layout,
            page_dpi=self.page_dpi,
            page_margin_mm=self.page_margin_mm,
            horizontal_alignment=self.horizontal_alignment,
            vertical_alignment=self.vertical_alignment,
            lighting_diagnostics=self.lighting_diagnostics,
            source_fingerprint=self.source_fingerprint,
        )

    @classmethod
    def from_payload(cls, payload: object) -> "ProcessingRecipe":
        if not isinstance(payload, dict):
            raise ValueError("Processing recipe is not an object.")
        normalized = dict(payload)
        if normalized.get("schema_version") == 1:
            # Version 1 predates the bundled page model and shadow stage.
            normalized.setdefault("auto_dewarp_uvdoc_grid", True)
            normalized.setdefault("shadow_method", "none")
        try:
            recipe = cls(**normalized)
        except TypeError as exc:
            raise ValueError("Processing recipe fields are invalid.") from exc
        string_fields = (
            recipe.orientation_method,
            recipe.deskew_method,
            recipe.dewarp_method,
            recipe.shadow_method,
            recipe.postprocess_name,
            recipe.page_layout,
            recipe.horizontal_alignment,
            recipe.vertical_alignment,
        )
        if not all(isinstance(value, str) for value in string_fields):
            raise ValueError("Processing recipe string fields are invalid.")
        if not isinstance(recipe.page_dpi, int) or isinstance(recipe.page_dpi, bool):
            raise ValueError("Processing recipe DPI is invalid.")
        recipe.to_request()
        return recipe

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class CommittedPageProcessing:
    """Durable recipe and evidence for the pixels in current.png."""

    recipe: ProcessingRecipe
    diagnostics: dict[str, object]
    current_fingerprint: str

    @classmethod
    def from_result(
        cls,
        request: PageProcessingRequest,
        diagnostics: PageProcessingDiagnostics,
        current_image: np.ndarray,
    ) -> "CommittedPageProcessing":
        payload = json.loads(json.dumps(asdict(diagnostics), allow_nan=False))
        if not isinstance(payload, dict):
            raise ValueError("Processing diagnostics are not an object.")
        return cls(
            recipe=ProcessingRecipe.from_request(request),
            diagnostics=payload,
            current_fingerprint=cls.fingerprint_image(current_image),
        )

    @staticmethod
    def fingerprint_image(image: np.ndarray) -> str:
        canonical = np.asarray(image)
        if canonical.ndim == 3 and canonical.shape[2] == 1:
            canonical = canonical[:, :, 0]
        elif canonical.ndim == 3 and canonical.shape[2] == 3:
            if np.array_equal(canonical[:, :, 0], canonical[:, :, 1]) and np.array_equal(
                canonical[:, :, 0], canonical[:, :, 2]
            ):
                canonical = canonical[:, :, 0]
        contiguous = np.ascontiguousarray(canonical)
        digest = hashlib.sha256()
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(memoryview(contiguous))
        return digest.hexdigest()

    @classmethod
    def from_payload(cls, payload: object) -> "CommittedPageProcessing":
        if (
            not isinstance(payload, dict)
            or type(payload.get("schemaVersion")) is not int
            or payload.get("schemaVersion") != 1
        ):
            raise ValueError("Committed processing metadata is invalid.")
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Committed processing diagnostics are invalid.")
        fingerprint = payload.get("currentFingerprint")
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise ValueError("Committed current image fingerprint is invalid.")
        json.dumps(diagnostics, allow_nan=False)
        return cls(
            recipe=ProcessingRecipe.from_payload(payload.get("recipe")),
            diagnostics=diagnostics,
            current_fingerprint=fingerprint,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "recipe": self.recipe.to_payload(),
            "diagnostics": self.diagnostics,
            "currentFingerprint": self.current_fingerprint,
        }


@dataclass(slots=True)
class CaptureEntry:
    """Single page entry in a capture/import session."""

    name: str
    store: PageStore
    paths: PagePaths
    detected_contour: np.ndarray | None = None
    detected_backend: str | None = None
    crop_state: str = CROP_STATE_NONE
    dewarp_control_points: tuple[tuple[float, float], ...] | None = None
    dewarp_control_curves: tuple[tuple[float, tuple[tuple[float, float], ...]], ...] | None = None
    committed_processing: CommittedPageProcessing | None = None
    selected: bool = False
    revision: int = field(default=0, repr=False)
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
        crop_state: str | None = None,
    ) -> "CaptureEntry":
        resolved_crop_state = _resolve_crop_state(crop_state, contour, backend)
        entry_id = uuid4().hex
        paths = store.add_page(entry_id, raw_image, warped_image)
        return cls(
            name=name,
            store=store,
            paths=paths,
            detected_contour=contour,
            detected_backend=backend,
            crop_state=resolved_crop_state,
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
        self.paths = self.store.replace_page_set(
            self.entry_id,
            original_image=image,
            # Until downstream processing publishes its own generation, the
            # safe current representation is the new original itself.
            current_image=image,
        )
        self.dewarp_control_points = None
        self.dewarp_control_curves = None
        self.committed_processing = None
        self.revision += 1

    @property
    def current_image(self) -> np.ndarray:
        return self.store.read_image(self.paths.current)

    @current_image.setter
    def current_image(self, image: np.ndarray) -> None:
        self.paths = self.store.replace_page_set(
            self.entry_id,
            current_image=image,
        )
        self.committed_processing = None
        self.revision += 1

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
        self.paths = self.store.replace_page_set(
            self.entry_id,
            raw_image=raw_image,
        )
        self.committed_processing = None
        self.revision += 1

    def set_dewarp_control_points(
        self,
        control_points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    ) -> None:
        self.dewarp_control_points = normalize_control_points(control_points)
        self.dewarp_control_curves = None
        self.revision += 1

    def set_dewarp_control_curves(self, control_curves) -> None:
        curves = normalize_control_curves(control_curves)
        self.dewarp_control_curves = curves
        self.dewarp_control_points = min(curves, key=lambda item: abs(item[0] - 0.5))[1]
        self.revision += 1

    def clear_dewarp_control_points(self) -> None:
        if self.dewarp_control_points is not None or self.dewarp_control_curves is not None:
            self.revision += 1
        self.dewarp_control_points = None
        self.dewarp_control_curves = None


class CaptureSession:
    """Ordered page session with disk-backed image storage."""

    def __init__(self, store: PageStore | None = None) -> None:
        self.store = store or PageStore()
        self._entries: list[CaptureEntry] = []
        self._quarantined_entries: list[dict[str, object]] = []
        self._document_order: list[str] = []
        self.restore_warnings: list[str] = []

    @property
    def entries(self) -> list[CaptureEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def quarantined_entry_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item["entryId"])
            for item in self._quarantined_entries
            if isinstance(item.get("entryId"), str)
        )

    @property
    def has_recoverable_state(self) -> bool:
        """Whether autosave must retain live or quarantined page assets."""
        return bool(self._entries or self._quarantined_entries)

    def clear(self) -> None:
        # Keep assets until the next manifest checkpoint.  If the process dies
        # before that checkpoint, the previous manifest remains fully restorable.
        removed_ids = {entry.entry_id for entry in self._entries}
        self._entries.clear()
        self._document_order = [
            entry_id for entry_id in self._document_order if entry_id not in removed_ids
        ]

    def add_entry(self, entry: CaptureEntry) -> None:
        self._entries.append(entry)
        if entry.entry_id not in self._document_order:
            self._document_order.append(entry.entry_id)

    def _sync_document_order(self) -> None:
        """Apply visible-page ordering while retaining quarantined page slots."""
        active_ids = [entry.entry_id for entry in self._entries]
        active_set = set(active_ids)
        quarantined_ids = [
            str(item["entryId"])
            for item in self._quarantined_entries
            if isinstance(item.get("entryId"), str)
        ]
        quarantined_set = set(quarantined_ids)
        known_ids = active_set | quarantined_set
        template = [entry_id for entry_id in self._document_order if entry_id in known_ids]
        for entry_id in quarantined_ids:
            if entry_id not in template:
                template.append(entry_id)

        active_iter = iter(active_ids)
        ordered: list[str] = []
        for entry_id in template:
            if entry_id in quarantined_set:
                ordered.append(entry_id)
                continue
            replacement = next(active_iter, None)
            if replacement is not None:
                ordered.append(replacement)
        ordered.extend(active_iter)
        self._document_order = ordered

    def _remove_from_document_order(self, entry_ids: Iterable[str]) -> None:
        removed = set(entry_ids)
        self._document_order = [
            entry_id for entry_id in self._document_order if entry_id not in removed
        ]

    def add_image(self, *, name: str, image: np.ndarray) -> CaptureEntry:
        entry = CaptureEntry.from_image(name=name, image=image, store=self.store)
        self.add_entry(entry)
        return entry

    def add_image_with_contour(
        self,
        *,
        name: str,
        raw_image: np.ndarray,
        warped_image: np.ndarray,
        contour: np.ndarray | None,
        backend: str | None,
        crop_state: str | None = None,
    ) -> CaptureEntry:
        entry = CaptureEntry.from_raw_and_warped(
            name=name,
            raw_image=raw_image,
            warped_image=warped_image,
            contour=contour,
            backend=backend,
            store=self.store,
            crop_state=crop_state,
        )
        self.add_entry(entry)
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
        try:
            order_index = self._document_order.index(after_entry_id)
        except ValueError:
            order_index = len(self._document_order) - 1
        self._document_order.insert(order_index + 1, entry.entry_id)
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
        self._sync_document_order()
        return True

    def move_many(self, entry_ids: Iterable[str], distance: int) -> bool:
        """Move selected entries one position while preserving their relative order."""
        if distance not in {-1, 1}:
            raise ValueError("Multiple pages can only move one position at a time.")
        selected = set(entry_ids)
        if not selected:
            return False
        moved = False
        indexes = (
            range(1, len(self._entries)) if distance < 0 else range(len(self._entries) - 2, -1, -1)
        )
        for index in indexes:
            adjacent = index + distance
            if (
                self._entries[index].entry_id in selected
                and self._entries[adjacent].entry_id not in selected
            ):
                self._entries[index], self._entries[adjacent] = (
                    self._entries[adjacent],
                    self._entries[index],
                )
                moved = True
        if moved:
            self._sync_document_order()
        return moved

    def reorder_entries(
        self,
        entry_ids: Iterable[str],
        target_entry_id: str,
        *,
        place_after: bool,
    ) -> bool:
        """Place selected entries beside a target while preserving selected-page order."""
        selected = set(entry_ids)
        if not selected or target_entry_id in selected:
            return False
        moving = [entry for entry in self._entries if entry.entry_id in selected]
        if not moving:
            return False
        remaining = [entry for entry in self._entries if entry.entry_id not in selected]
        target_index = next(
            (index for index, entry in enumerate(remaining) if entry.entry_id == target_entry_id),
            None,
        )
        if target_index is None:
            return False
        insertion_index = target_index + int(place_after)
        reordered = remaining[:insertion_index] + moving + remaining[insertion_index:]
        if reordered == self._entries:
            return False
        self._entries = reordered
        self._sync_document_order()
        return True

    def select_all(self, selected: bool = True) -> None:
        for entry in self._entries:
            entry.selected = selected

    def remove_selected(self) -> int:
        before = len(self._entries)
        kept: list[CaptureEntry] = []
        removed_ids: list[str] = []
        for entry in self._entries:
            if not entry.selected:
                kept.append(entry)
            else:
                removed_ids.append(entry.entry_id)
        self._entries = kept
        self._remove_from_document_order(removed_ids)
        return before - len(self._entries)

    def remove_entry(self, entry_id: str) -> bool:
        index = self._find_index(entry_id)
        if index is None:
            return False
        self._remove_from_document_order((entry_id,))
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
        crop_state: str | None = None,
    ) -> bool:
        """Replace entry images in-place while preserving ordering and identity."""
        index = self._find_index(entry_id)
        if index is None:
            return False

        entry = self._entries[index]
        resolved_crop_state = _resolve_crop_state(crop_state, contour, backend)
        entry.paths = self.store.replace_page_set(
            entry.entry_id,
            raw_image=raw_image,
            original_image=original_image,
            current_image=original_image if current_image is None else current_image,
        )
        entry.dewarp_control_points = None
        entry.dewarp_control_curves = None
        entry.committed_processing = None
        entry.revision += 1
        if name is not None and name.strip():
            entry.name = name.strip()
        # Replacement is authoritative.  A detector miss must clear metadata
        # from the previous source instead of retaining a stale overlay/backend.
        entry.detected_contour = contour
        entry.detected_backend = backend
        entry.crop_state = resolved_crop_state
        return True

    def selected_entries(self) -> list[CaptureEntry]:
        return [entry for entry in self._entries if entry.selected]

    def save_manifest(self, manifest_path: Path) -> Path:
        """Atomically save enough metadata to reopen this disk-backed session."""
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._sync_document_order()
        document_positions = {
            entry_id: index for index, entry_id in enumerate(self._document_order)
        }
        payload = {
            "schemaVersion": 5,
            "sessionDir": str(self.store.session_dir.resolve()),
            "entries": [
                {
                    "entryId": entry.entry_id,
                    "documentPosition": document_positions[entry.entry_id],
                    "name": entry.name,
                    "selected": entry.selected,
                    "cropState": entry.crop_state,
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
                    "dewarpControlCurves": (
                        [
                            [anchor, [list(point) for point in points]]
                            for anchor, points in entry.dewarp_control_curves
                        ]
                        if entry.dewarp_control_curves is not None
                        else None
                    ),
                    "committedProcessing": (
                        entry.committed_processing.to_payload()
                        if entry.committed_processing is not None
                        else None
                    ),
                }
                for entry in self._entries
            ],
            # Skipped pages remain referenced until an explicit session discard.
            # A future restore can recover them if their assets are repaired.
            "quarantinedEntries": [
                {
                    **item,
                    "cropState": item.get(
                        "cropState",
                        (
                            CROP_STATE_APPLIED
                            if item.get("detectedContour") is not None
                            or item.get("detectedBackend") is not None
                            else CROP_STATE_NONE
                        ),
                    ),
                    "documentPosition": document_positions[str(item["entryId"])],
                }
                for item in self._quarantined_entries
            ],
        }
        descriptor, raw_stage = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.stage-",
            suffix=".json",
            dir=manifest_path.parent,
        )
        stage = Path(raw_stage)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(stage, manifest_path)
            try:
                parent_descriptor = os.open(manifest_path.parent, os.O_RDONLY)
            except OSError:
                pass
            else:
                try:
                    os.fsync(parent_descriptor)
                except OSError:
                    pass
                finally:
                    os.close(parent_descriptor)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        finally:
            if stage.exists():
                stage.unlink()
        # Delete removed page directories only after the durable manifest no
        # longer references them.  A crash on either side is recoverable.
        protected_ids = {
            *[entry.entry_id for entry in self._entries],
            *self.quarantined_entry_ids,
        }
        self.store.prune_pages(protected_ids)
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
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported session manifest: {manifest_path}")
        manifest_version = payload.get("schemaVersion")
        if (
            type(manifest_version) is not int
            or manifest_version not in {1, 2, 3, 4, 5}
            or not isinstance(payload.get("entries"), list)
        ):
            raise ValueError(f"Unsupported session manifest: {manifest_path}")
        quarantined_entries = payload.get("quarantinedEntries", [])
        if not isinstance(quarantined_entries, list):
            raise ValueError(f"Invalid quarantined entries in manifest: {manifest_path}")

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
        seen_ids: set[str] = set()
        restore_items = [*payload["entries"], *quarantined_entries]
        if manifest_version >= 4:
            positions = [
                item.get("documentPosition") if isinstance(item, dict) else None
                for item in restore_items
            ]
            if any(type(position) is not int or position < 0 for position in positions) or sorted(
                positions
            ) != list(range(len(restore_items))):
                raise ValueError(f"Invalid document order in manifest: {manifest_path}")
            restore_items.sort(key=lambda item: item["documentPosition"])
        for position, item in enumerate(restore_items, start=1):
            entry_id = "unknown"
            can_quarantine = False
            try:
                if not isinstance(item, dict):
                    raise ValueError("entry metadata is not an object")
                entry_id = str(item.get("entryId", ""))
                if re.fullmatch(r"[0-9a-f]{32}", entry_id) is None:
                    raise ValueError("invalid entry id")
                if entry_id in seen_ids:
                    raise ValueError("duplicate entry id")
                # Claim the id before touching its assets.  A malformed duplicate
                # must never be allowed to shadow or delete the first page.
                seen_ids.add(entry_id)
                session._document_order.append(entry_id)
                can_quarantine = True
                # Raw and original are authoritative.  Current and all display
                # derivatives are disposable and can be rebuilt after a crash.
                paths, current_rebuilt = store.repair_page_assets(entry_id)
                contour_raw = item.get("detectedContour")
                contour = (
                    np.asarray(contour_raw, dtype=np.float32) if contour_raw is not None else None
                )
                if contour is not None and (
                    contour.shape != (4, 2) or not np.isfinite(contour).all()
                ):
                    raise ValueError("invalid detected contour")
                control_points_raw = item.get("dewarpControlPoints")
                control_points = (
                    normalize_control_points(control_points_raw)
                    if control_points_raw is not None
                    else None
                )
                control_curves_raw = (
                    item.get("dewarpControlCurves") if manifest_version >= 3 else None
                )
                control_curves = (
                    normalize_control_curves(control_curves_raw)
                    if control_curves_raw is not None
                    else None
                )
                backend_raw = item.get("detectedBackend")
                if backend_raw is not None and not isinstance(backend_raw, str):
                    raise ValueError("invalid detected backend")
                if manifest_version >= 5:
                    crop_state_raw = item.get("cropState")
                    crop_state = _resolve_crop_state(crop_state_raw, contour, backend_raw)
                else:
                    # Before schema 5, contour/backend described geometry that
                    # was already committed to original.png.  Treating it as a
                    # proposal would apply legacy crops a second time.
                    crop_state = (
                        CROP_STATE_APPLIED
                        if contour is not None or backend_raw is not None
                        else CROP_STATE_NONE
                    )
                committed_payload = (
                    item.get("committedProcessing") if manifest_version >= 2 else None
                )
                committed_processing = None
                if committed_payload is not None:
                    try:
                        committed_processing = CommittedPageProcessing.from_payload(
                            committed_payload
                        )
                    except (TypeError, ValueError) as exc:
                        session.restore_warnings.append(
                            f"Ignored processing metadata for session page {position} "
                            f"({entry_id}): {exc}"
                        )
                if current_rebuilt:
                    committed_processing = None
                elif committed_processing is not None:
                    current_image = store.read_image(paths.current)
                    actual_fingerprint = CommittedPageProcessing.fingerprint_image(current_image)
                    if actual_fingerprint != committed_processing.current_fingerprint:
                        session.restore_warnings.append(
                            f"Ignored stale processing metadata for session page {position} "
                            f"({entry_id}): current image fingerprint changed."
                        )
                        committed_processing = None
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if not can_quarantine and entry_id not in seen_ids:
                    # Without a valid UUID there is no safe way to associate
                    # this manifest record with a page directory.  Abort the
                    # restore instead of returning a partial session whose next
                    # autosave could prune the unknown source assets.
                    raise ValueError(
                        f"Cannot safely associate session page {position} with its assets."
                    ) from exc
                session.restore_warnings.append(
                    f"Skipped session page {position} ({entry_id}): {exc}"
                )
                if can_quarantine:
                    # Keep the durable reference across the next autosave.  The
                    # page can be retried on a later launch and is deleted only
                    # by an explicit session discard.
                    session._quarantined_entries.append(dict(item))
                continue

            if current_rebuilt:
                session.restore_warnings.append(
                    f"Recovered session page {position} ({entry_id}) from its original image."
                )
            session.add_entry(
                CaptureEntry(
                    name=str(item.get("name", entry_id)),
                    store=store,
                    paths=paths,
                    detected_contour=contour,
                    detected_backend=backend_raw,
                    crop_state=crop_state,
                    dewarp_control_points=control_points,
                    dewarp_control_curves=control_curves,
                    committed_processing=committed_processing,
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
