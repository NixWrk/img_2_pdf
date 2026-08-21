"""Small transaction boundary for computed stage edits.

The adapter deliberately knows nothing about image processing or the UI. A
caller computes a complete result first, then stages it here. The only
domain-specific mutation it performs is the existing ``CaptureEntry``
``current_image`` setter; optional entry metadata is copied field-by-field.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_METADATA_FIELDS = (
    "dewarp_control_points",
    "dewarp_control_curves",
)
RESERVED_METADATA_FIELDS = frozenset(
    {"entry_id", "revision", "current_image", "committed_processing", "paths", "store"}
)


class StageTransactionError(RuntimeError):
    """Base error for an invalid or failed stage transaction."""


class StageTransactionClosedError(StageTransactionError):
    """Raised when a transaction is used after commit or discard."""


class StaleStageRevisionError(StageTransactionError):
    """Raised before writes when any entry changed after the snapshot."""

    def __init__(self, entry_ids: Iterable[str]) -> None:
        self.entry_ids = tuple(entry_ids)
        joined = ", ".join(self.entry_ids)
        super().__init__(f"Stage transaction snapshot is stale for entry(s): {joined}")


class IncompleteStageCandidateError(StageTransactionError):
    """Raised when a batch does not contain exactly one candidate per entry."""


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Copied committed baseline captured at transaction start."""

    entry_id: str
    expected_revision: int
    pixels: np.ndarray
    committed_processing: Any
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StageCandidate:
    """A fully computed result ready to publish.

    ``committed_processing`` is supplied by the caller. It may contain the
    stage recipe and diagnostics, but this adapter never invokes a processor
    or constructs processing metadata.
    """

    entry_id: str
    pixels: np.ndarray
    committed_processing: Any
    metadata: Mapping[str, Any]


def _copy(value: Any) -> Any:
    """Copy arrays and nested result metadata without sharing mutable state."""

    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


def _clone_candidate(candidate: StageCandidate) -> StageCandidate:
    return StageCandidate(
        entry_id=candidate.entry_id,
        pixels=_copy(candidate.pixels),
        committed_processing=_copy(candidate.committed_processing),
        metadata={key: _copy(value) for key, value in candidate.metadata.items()},
    )


class StageEditTransaction:
    """Toolkit-independent all-or-nothing publisher for stage results."""

    def __init__(self, entries: tuple[Any, ...], metadata_fields: tuple[str, ...]) -> None:
        self._entries = {entry.entry_id: entry for entry in entries}
        self._snapshots = {
            entry.entry_id: StageSnapshot(
                entry_id=entry.entry_id,
                expected_revision=entry.revision,
                pixels=_copy(entry.current_image),
                committed_processing=_copy(entry.committed_processing),
                metadata={field: _copy(getattr(entry, field)) for field in metadata_fields},
            )
            for entry in entries
        }
        self._metadata_fields = metadata_fields
        self._candidates: dict[str, StageCandidate] = {}
        self._closed = False

    @classmethod
    def begin(
        cls,
        entries: Iterable[Any],
        *,
        metadata_fields: Iterable[str] = DEFAULT_METADATA_FIELDS,
    ) -> "StageEditTransaction":
        """Capture entry ids, revisions, pixels, and durable metadata."""

        entry_tuple = tuple(entries)
        if not entry_tuple:
            raise ValueError("A stage transaction needs at least one entry.")
        ids = [entry.entry_id for entry in entry_tuple]
        if len(set(ids)) != len(ids):
            raise ValueError("Stage transaction entries must have unique entry ids.")
        fields = tuple(metadata_fields)
        if len(set(fields)) != len(fields):
            raise ValueError("Stage transaction metadata fields must be unique.")
        reserved = set(fields).intersection(RESERVED_METADATA_FIELDS)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"Reserved stage metadata field(s): {names}")
        for entry in entry_tuple:
            for field in fields:
                if not hasattr(entry, field):
                    raise AttributeError(f"Entry has no stage metadata field {field!r}.")
        return cls(entry_tuple, fields)

    @property
    def snapshots(self) -> tuple[StageSnapshot, ...]:
        """Return fresh copies of the baseline snapshots."""

        return tuple(
            StageSnapshot(
                entry_id=snapshot.entry_id,
                expected_revision=snapshot.expected_revision,
                pixels=_copy(snapshot.pixels),
                committed_processing=_copy(snapshot.committed_processing),
                metadata={key: _copy(value) for key, value in snapshot.metadata.items()},
            )
            for snapshot in self._snapshots.values()
        )

    def stage(
        self,
        entry_id: str,
        *,
        pixels: np.ndarray,
        committed_processing: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageCandidate:
        """Copy and hold one complete computed result for later commit."""

        self._ensure_open()
        if entry_id not in self._entries:
            raise KeyError(f"Entry {entry_id!r} is not part of this transaction.")
        if not isinstance(pixels, np.ndarray):
            raise TypeError("Stage candidate pixels must be a numpy.ndarray.")
        candidate_metadata = dict(metadata or {})
        reserved = set(candidate_metadata).intersection(RESERVED_METADATA_FIELDS)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise KeyError(f"Reserved stage metadata field(s): {names}")
        unknown = set(candidate_metadata).difference(self._metadata_fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported stage metadata field(s): {names}")
        candidate = StageCandidate(
            entry_id=entry_id,
            pixels=_copy(pixels),
            committed_processing=_copy(committed_processing),
            metadata={key: _copy(value) for key, value in candidate_metadata.items()},
        )
        # The returned object is deliberately not the object retained for a
        # future commit: frozen dataclasses still contain mutable arrays/maps.
        self._candidates[entry_id] = _clone_candidate(candidate)
        return candidate

    def commit(self) -> tuple[str, ...]:
        """Publish every staged result, rolling back all writes on failure."""

        self._ensure_open()
        expected_ids = set(self._snapshots)
        if set(self._candidates) != expected_ids:
            missing = expected_ids.difference(self._candidates)
            extra = set(self._candidates).difference(expected_ids)
            raise IncompleteStageCandidateError(
                f"Stage candidate batch mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        # This is the complete preflight pass: no entry has been written until
        # every revision and candidate has been checked.
        stale_ids = tuple(
            entry_id
            for entry_id, snapshot in self._snapshots.items()
            if self._entries[entry_id].revision != snapshot.expected_revision
        )
        if stale_ids:
            self._closed = True
            self._candidates.clear()
            raise StaleStageRevisionError(stale_ids)
        for candidate in self._candidates.values():
            self._validate_candidate(candidate)

        changed: list[str] = []
        current_entry_id: str | None = None
        try:
            for entry_id, snapshot in self._snapshots.items():
                current_entry_id = entry_id
                entry = self._entries[entry_id]
                candidate = self._candidates[entry_id]
                entry.current_image = _copy(candidate.pixels)
                entry.committed_processing = _copy(candidate.committed_processing)
                for field, value in candidate.metadata.items():
                    setattr(entry, field, _copy(value))
                changed.append(entry_id)
        except Exception as exc:
            rollback_errors: list[Exception] = []
            rollback_ids = list(changed)
            if current_entry_id is not None and current_entry_id not in rollback_ids:
                rollback_ids.append(current_entry_id)
            for entry_id in reversed(rollback_ids):
                try:
                    self._restore(self._entries[entry_id], self._snapshots[entry_id])
                except Exception as rollback_error:  # pragma: no cover - defensive path
                    rollback_errors.append(rollback_error)
            self._closed = True
            if rollback_errors:
                raise StageTransactionError(
                    "Stage commit failed and rollback was incomplete."
                ) from exc
            raise StageTransactionError("Stage commit failed; all entries were restored.") from exc

        self._closed = True
        return tuple(self._snapshots)

    def discard(self) -> None:
        """Close without mutating any entry."""

        self._ensure_open()
        self._closed = True
        self._candidates.clear()

    def _restore(self, entry: Any, snapshot: StageSnapshot) -> None:
        entry.current_image = _copy(snapshot.pixels)
        entry.committed_processing = _copy(snapshot.committed_processing)
        for field, value in snapshot.metadata.items():
            setattr(entry, field, _copy(value))
        entry.revision = snapshot.expected_revision

    def _validate_candidate(self, candidate: StageCandidate) -> None:
        processing = candidate.committed_processing
        if processing is None:
            return
        expected_fingerprint = getattr(processing, "current_fingerprint", None)
        fingerprint_image = getattr(type(processing), "fingerprint_image", None)
        if expected_fingerprint is not None and callable(fingerprint_image):
            actual_fingerprint = fingerprint_image(candidate.pixels)
            if actual_fingerprint != expected_fingerprint:
                raise StageTransactionError(
                    "Committed processing fingerprint does not match candidate pixels."
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise StageTransactionClosedError("Stage transaction is already closed.")


__all__ = [
    "DEFAULT_METADATA_FIELDS",
    "IncompleteStageCandidateError",
    "StageCandidate",
    "StageEditTransaction",
    "StageSnapshot",
    "StageTransactionClosedError",
    "StageTransactionError",
    "StaleStageRevisionError",
]
