"""Capture/session data model used by unified UI and export pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from uniscan.core.dewarp import DewarpModel, normalize_control_points
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.preprocess import PreprocessSettings
from uniscan.core.processing import PageProcessingDiagnostics, PageProcessingRequest
from uniscan.storage import PagePaths, PageStore


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
            schema_version=1,
            orientation_method=request.orientation_method,
            deskew_method=request.deskew_method,
            dewarp_method=request.dewarp_method,
            dewarp_model=asdict(request.dewarp_model) if request.dewarp_model else None,
            dewarp_already_applied=request.dewarp_already_applied,
            uvdoc_cache_home=str(request.uvdoc_cache_home) if request.uvdoc_cache_home else None,
            auto_dewarp_uvdoc=request.auto_dewarp_uvdoc,
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
        if type(self.schema_version) is not int or self.schema_version != 1:
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
        try:
            recipe = cls(**payload)
        except TypeError as exc:
            raise ValueError("Processing recipe fields are invalid.") from exc
        string_fields = (
            recipe.orientation_method,
            recipe.deskew_method,
            recipe.dewarp_method,
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
    dewarp_control_points: tuple[tuple[float, float], ...] | None = None
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
        self.paths = self.store.replace_page_set(
            self.entry_id,
            original_image=image,
            # Until downstream processing publishes its own generation, the
            # safe current representation is the new original itself.
            current_image=image,
        )
        self.dewarp_control_points = None
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
        self.revision += 1

    def clear_dewarp_control_points(self) -> None:
        if self.dewarp_control_points is not None:
            self.revision += 1
        self.dewarp_control_points = None


class CaptureSession:
    """Ordered page session with disk-backed image storage."""

    def __init__(self, store: PageStore | None = None) -> None:
        self.store = store or PageStore()
        self._entries: list[CaptureEntry] = []
        self._quarantined_entries: list[dict[str, object]] = []
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
            if not entry.selected:
                kept.append(entry)
        self._entries = kept
        return before - len(self._entries)

    def remove_entry(self, entry_id: str) -> bool:
        index = self._find_index(entry_id)
        if index is None:
            return False
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
        entry.paths = self.store.replace_page_set(
            entry.entry_id,
            raw_image=raw_image,
            original_image=original_image,
            current_image=original_image if current_image is None else current_image,
        )
        entry.dewarp_control_points = None
        entry.committed_processing = None
        entry.revision += 1
        if name is not None and name.strip():
            entry.name = name.strip()
        # Replacement is authoritative.  A detector miss must clear metadata
        # from the previous source instead of retaining a stale overlay/backend.
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
            "schemaVersion": 2,
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
            "quarantinedEntries": self._quarantined_entries,
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
            or manifest_version not in {1, 2}
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
                backend_raw = item.get("detectedBackend")
                if backend_raw is not None and not isinstance(backend_raw, str):
                    raise ValueError("invalid detected backend")
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
                    dewarp_control_points=control_points,
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
