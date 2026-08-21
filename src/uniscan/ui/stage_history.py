"""Bounded process-local stage-edit history with atomic image restoration."""

from __future__ import annotations

import copy
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from uniscan.io import imread_unicode, imwrite_unicode

__all__ = [
    "PageSnapshot",
    "StageHistory",
    "StageHistoryBlocked",
    "StageHistoryCapture",
    "StageHistoryError",
    "StageHistoryRecord",
    "StageHistoryRollbackError",
]


class StageHistoryError(RuntimeError):
    """Invalid, failed, or unavailable history operation."""


class StageHistoryBlocked(StageHistoryError):
    """CAS, identity, fingerprint, or snapshot preflight rejected the operation."""


class StageHistoryRollbackError(StageHistoryError):
    """A failed restore could not be rolled back completely."""


@dataclass(slots=True)
class PageSnapshot:
    entry_id: str
    revision: int
    current_path: Path
    current_fingerprint: str
    original_path: Path | None
    original_fingerprint: str | None
    committed_processing: Any
    dewarp_control_points: Any
    dewarp_control_curves: Any
    perspective: dict[str, Any]


@dataclass(slots=True)
class StageHistoryRecord:
    action: str
    stage: str
    before: tuple[PageSnapshot, ...]
    after: tuple[PageSnapshot, ...]

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(item.entry_id for item in self.before)


class StageHistoryCapture:
    def __init__(self, history: "StageHistory", record: StageHistoryRecord) -> None:
        self.history = history
        self.record = record
        self.closed = False

    def discard(self) -> None:
        if not self.closed:
            self.history._delete_record_files(self.record)
            self.closed = True


def _fingerprint(image: np.ndarray) -> str:
    image = np.ascontiguousarray(image)
    return hashlib.sha256(
        str(image.dtype).encode() + repr(image.shape).encode() + image.tobytes()
    ).hexdigest()


class StageHistory:
    """Bounded metadata stack and uniquely named disk snapshots, not persisted."""

    def __init__(self, root: Path | None = None, *, max_records: int = 20) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        parent = Path(root) if root is not None else Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="uniscan_stage_history_", dir=parent))
        self.max_records = max_records
        self._serial = count(1)
        self._undo: list[StageHistoryRecord] = []
        self._redo: list[StageHistoryRecord] = []
        self._closed = False

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def redo_depth(self) -> int:
        return len(self._redo)

    def capture(
        self, entries: Iterable[Any], *, action: str, stage: str, include_perspective: bool = False
    ) -> StageHistoryCapture:
        if self._closed:
            raise StageHistoryError("history is closed")
        if not action.strip() or not stage.strip():
            raise ValueError("action/stage required")
        items = tuple(entries)
        ids = tuple(item.entry_id for item in items)
        if not items:
            raise ValueError("at least one entry required")
        if len(set(ids)) != len(ids):
            raise ValueError("entry ids must be unique")
        snapshots: list[PageSnapshot] = []
        try:
            for item in items:
                snapshots.append(self._snapshot(item, include_perspective, "before"))
        except Exception:
            for snapshot in snapshots:
                self._delete_snapshot_files(snapshot)
            raise
        return StageHistoryCapture(self, StageHistoryRecord(action, stage, tuple(snapshots), ()))

    def record(self, capture: StageHistoryCapture, entries: Iterable[Any]) -> StageHistoryRecord:
        if capture.history is not self or capture.closed:
            raise StageHistoryError("capture closed")
        items = tuple(entries)
        by_id = {item.entry_id: item for item in items}
        expected = {item.entry_id for item in capture.record.before}
        if len(by_id) != len(items) or set(by_id) != expected:
            raise StageHistoryError("record entry set does not match capture")
        try:
            after = tuple(
                self._snapshot(by_id[item.entry_id], bool(item.original_path), "after")
                for item in capture.record.before
            )
        except Exception:
            capture.discard()
            raise
        record = StageHistoryRecord(
            capture.record.action, capture.record.stage, capture.record.before, after
        )
        for old in tuple(self._redo):
            self._delete_record_files(old)
        self._redo.clear()
        self._undo.append(record)
        capture.closed = True
        while len(self._undo) > self.max_records:
            self._delete_record_files(self._undo.pop(0))
        return record

    def undo(self, lookup: Callable[[str], Any | None]) -> StageHistoryRecord:
        if not self._undo:
            raise StageHistoryError("no stage edit to undo")
        record = self._undo[-1]
        self._apply(record, record.after, record.before, lookup)
        self._undo.pop()
        self._redo.append(record)
        return record

    def redo(self, lookup: Callable[[str], Any | None]) -> StageHistoryRecord:
        if not self._redo:
            raise StageHistoryError("no stage edit to redo")
        record = self._redo[-1]
        self._apply(record, record.before, record.after, lookup)
        self._redo.pop()
        self._undo.append(record)
        return record

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.root, ignore_errors=True)
        self._undo.clear()
        self._redo.clear()

    def _snapshot(self, entry: Any, perspective: bool, label: str) -> PageSnapshot:
        token = f"{next(self._serial):08d}_{entry.entry_id}_{label}"
        current = np.asarray(entry.current_image)
        current_path = self.root / f"{token}.png"
        original_path: Path | None = None
        try:
            if hasattr(entry, "store") and hasattr(entry, "paths"):
                entry.store.snapshot_image(entry.paths.current, current_path)
            elif not imwrite_unicode(current_path, current):
                raise StageHistoryError("cannot write history snapshot")
            original_fingerprint = None
            metadata: dict[str, Any] = {}
            if perspective:
                if not hasattr(entry, "store") or not hasattr(entry, "paths"):
                    raise StageHistoryError("perspective history requires a persistent store")
                original = np.asarray(entry.original_image)
                original_fingerprint = _fingerprint(original)
                original_path = self.root / f"{token}_original.png"
                entry.store.snapshot_image(entry.paths.original, original_path)
                for name in (
                    "detected_contour",
                    "detected_backend",
                    "crop_state",
                    "needs_review",
                    "review_reasons",
                ):
                    metadata[name] = copy.deepcopy(getattr(entry, name, None))
            return PageSnapshot(
                entry.entry_id,
                int(entry.revision),
                current_path,
                _fingerprint(current),
                original_path,
                original_fingerprint,
                copy.deepcopy(getattr(entry, "committed_processing", None)),
                copy.deepcopy(getattr(entry, "dewarp_control_points", None)),
                copy.deepcopy(getattr(entry, "dewarp_control_curves", None)),
                metadata,
            )
        except Exception:
            self._delete_snapshot_files(
                PageSnapshot(
                    entry.entry_id, 0, current_path, "", original_path, None, None, None, None, {}
                )
            )
            raise

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        image = imread_unicode(path, preserve_channels=True)
        if image is None:
            raise StageHistoryBlocked(f"invalid history snapshot: {path}")
        return np.asarray(image).copy()

    def _apply(
        self,
        record: StageHistoryRecord,
        source: tuple[PageSnapshot, ...],
        target: tuple[PageSnapshot, ...],
        lookup: Callable[[str], Any | None],
    ) -> None:
        if len(source) != len(target):
            raise StageHistoryError("malformed history record")
        entry_ids = tuple(item.entry_id for item in source)
        if len(set(entry_ids)) != len(entry_ids) or any(
            source_item.entry_id != target_item.entry_id
            for source_item, target_item in zip(source, target)
        ):
            raise StageHistoryError("malformed history record entry ids")
        entries: list[Any] = []
        source_data: list[tuple[np.ndarray, np.ndarray | None]] = []
        target_data: list[tuple[np.ndarray, np.ndarray | None]] = []
        for source_item, target_item in zip(source, target):
            entry = lookup(source_item.entry_id)
            if entry is None or entry.entry_id != source_item.entry_id:
                raise StageHistoryBlocked("missing/replaced history entry")
            if (
                int(entry.revision) != source_item.revision
                or _fingerprint(np.asarray(entry.current_image)) != source_item.current_fingerprint
            ):
                raise StageHistoryBlocked("history CAS/fingerprint mismatch")
            entries.append(entry)
            source_current = self._read(source_item.current_path)
            if _fingerprint(source_current) != source_item.current_fingerprint:
                raise StageHistoryBlocked("source snapshot fingerprint mismatch")
            source_original = (
                self._read(source_item.original_path) if source_item.original_path else None
            )
            if (
                source_original is not None
                and _fingerprint(source_original) != source_item.original_fingerprint
            ):
                raise StageHistoryBlocked("source original fingerprint mismatch")
            target_current = self._read(target_item.current_path)
            if _fingerprint(target_current) != target_item.current_fingerprint:
                raise StageHistoryBlocked("target snapshot fingerprint mismatch")
            target_original = (
                self._read(target_item.original_path) if target_item.original_path else None
            )
            if (
                target_original is not None
                and _fingerprint(target_original) != target_item.original_fingerprint
            ):
                raise StageHistoryBlocked("target original fingerprint mismatch")
            source_data.append((source_current, source_original))
            target_data.append((target_current, target_original))
        attempted: list[int] = []
        try:
            for index, (entry, item, data) in enumerate(zip(entries, target, target_data)):
                attempted.append(index)
                self._restore_entry(entry, item, data[0], data[1], int(entry.revision) + 1)
        except Exception as exc:
            rollback_errors: list[BaseException] = []
            for index in reversed(attempted):
                try:
                    entry = entries[index]
                    item = source[index]
                    data = source_data[index]
                    self._restore_entry(entry, item, data[0], data[1], item.revision)
                except Exception as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if rollback_errors:
                raise StageHistoryRollbackError("rollback incomplete") from exc
            raise StageHistoryError("stage history restore failed") from exc
        for item, entry in zip(target, entries):
            item.revision = int(entry.revision)

    @staticmethod
    def _restore_entry(
        entry: Any,
        item: PageSnapshot,
        current: np.ndarray,
        original: np.ndarray | None,
        revision: int,
    ) -> None:
        if hasattr(entry, "store"):
            entry.paths = entry.store.replace_page_set(
                entry.entry_id, original_image=original, current_image=current
            )
        else:
            entry.current_image = current
        entry.committed_processing = copy.deepcopy(item.committed_processing)
        entry.dewarp_control_points = copy.deepcopy(item.dewarp_control_points)
        entry.dewarp_control_curves = copy.deepcopy(item.dewarp_control_curves)
        for name, value in item.perspective.items():
            setattr(entry, name, copy.deepcopy(value))
        entry.revision = revision

    def _delete_snapshot_files(self, snapshot: PageSnapshot) -> None:
        snapshot.current_path.unlink(missing_ok=True)
        if snapshot.original_path:
            snapshot.original_path.unlink(missing_ok=True)

    def _delete_record_files(self, record: StageHistoryRecord) -> None:
        for item in (*record.before, *record.after):
            self._delete_snapshot_files(item)
