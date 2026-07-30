"""Headless input-to-PDF pipeline built from the production scanner primitives."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from uniscan.core.pipeline import PipelineOptions, process_loaded_items
from uniscan.core.orientation import ORIENTATION_METHOD_CHOICES, OrientationDiagnostics
from uniscan.core.layout import (
    HORIZONTAL_ALIGNMENTS,
    PAGE_LAYOUT_CHOICES,
    VERTICAL_ALIGNMENTS,
)
from uniscan.core.lighting import (
    SHADOW_METHOD_CHOICES,
    SHADOW_METHOD_CLASSICAL,
    SHADOW_METHOD_NONE,
)
from uniscan.core.cleanup import (
    BINARIZATION_CHOICES,
    BINARIZATION_NONE,
    DESPECKLE_CHOICES,
    DESPECKLE_NONE,
)
from uniscan.core.dewarp import (
    DEWARP_METHOD_AUTO,
    DEWARP_METHOD_CHOICES,
)
from uniscan.core.preprocess import (
    DESKEW_METHOD_CHOICES,
    DESKEW_METHOD_MANUAL,
    PREPROCESS_PRESETS,
    PreprocessSettings,
    resolve_lens_mode_profile,
)
from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OFFICE_LENS_ONNX,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
)
from uniscan.export import export_image_paths_as_files, export_image_paths_as_pdf
from uniscan.export.exporters import (
    _canonical_export_target,
    _clone_directory_for_staging,
    _directory_export_lock,
    _directory_export_lock_path,
    _exclusive_export_lock,
    _file_export_lock,
    _file_export_lock_path,
    _is_link_like,
)
from uniscan.io import (
    DEFAULT_MAX_INPUT_PIXELS,
    IMG_EXTS,
    PDF_EXTS,
    imwrite_unicode,
    iter_input_items,
    list_supported_in_folder,
)
from uniscan.storage import ProcessingStageCache


LENS_MODE_CHOICES = ("none", "document", "grayscale", "whiteboard", "photo", "b/w")
IMAGE_FORMAT_CHOICES = ("png", "jpg", "jpeg", "webp", "tif", "tiff")
DETECTOR_POLICY_CHOICES = (
    "auto",
    "office_lens_onnx",
    "cv_hybrid",
    "opencv_quad",
    "opencv_hough",
    "opencv_minrect",
)
CancelCb = Callable[[], bool]
ProgressCb = Callable[[int, int, str], None]

_DETECTOR_POLICIES: dict[str, tuple[str, ...]] = {
    "auto": DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    "office_lens_onnx": (DETECTOR_BACKEND_OFFICE_LENS_ONNX,),
    "cv_hybrid": (DETECTOR_BACKEND_CV_HYBRID,),
    "opencv_quad": (DETECTOR_BACKEND_OPENCV,),
    "opencv_hough": (DETECTOR_BACKEND_OPENCV_HOUGH,),
    "opencv_minrect": (DETECTOR_BACKEND_OPENCV_MINRECT,),
}


@dataclass(slots=True, frozen=True)
class PageRunReport:
    """Detection and timing details for one exported page."""

    index: int
    name: str
    detected: bool
    backend: str | None
    fallback_reason: str | None
    duration_ms: float
    orientation_method: str = "none"
    orientation_applied: bool = False
    orientation_angle_degrees: int = 0
    orientation_confidence: float = 0.0
    orientation_reason: str | None = None
    deskew_method: str = "none"
    deskew_angle_degrees: float = 0.0
    deskew_selected_method: str = "none"
    deskew_confidence: float = 0.0
    deskew_line_count: int = 0
    deskew_reason: str | None = None
    dewarp_method: str = "none"
    dewarp_applied: bool = False
    dewarp_selected_method: str = "none"
    dewarp_line_count: int = 0
    dewarp_max_displacement_px: float = 0.0
    dewarp_curvature_before_px: float = 0.0
    dewarp_curvature_after_px: float = 0.0
    dewarp_perspective_before: float = 0.0
    dewarp_perspective_after: float = 0.0
    dewarp_blank_border_before: float = 0.0
    dewarp_blank_border_after: float = 0.0
    dewarp_edge_ink_before: float = 0.0
    dewarp_edge_ink_after: float = 0.0
    dewarp_aspect_change: float = 0.0
    dewarp_duration_ms: float = 0.0
    dewarp_reason: str | None = None
    shadow_method: str = "none"
    shadow_applied: bool = False
    shadow_selected_method: str = "none"
    shadow_unevenness_before: float = 0.0
    shadow_unevenness_after: float = 0.0
    shadow_before: float = 0.0
    shadow_after: float = 0.0
    shadow_glare_after: float = 0.0
    shadow_duration_ms: float = 0.0
    shadow_reason: str | None = None
    page_layout: str = "none"
    layout_applied: bool = False
    content_box: tuple[int, int, int, int] | None = None
    content_confidence: float = 0.0
    layout_scale: float = 1.0
    layout_reason: str | None = None
    binarization_method: str = "none"
    despeckle_strength: str = "none"
    despeckle_removed_components: int = 0
    despeckle_removed_pixels: int = 0
    despeckle_protected_components: int = 0
    shadow_fraction: float | None = None
    glare_fraction: float | None = None
    clipped_pixel_fraction: float | None = None
    lighting_unevenness: float | None = None
    lighting_warnings: tuple[str, ...] = ()
    processing_stage_durations_ms: dict[str, float] | None = None
    processing_cache_hits: tuple[str, ...] = ()
    spread_detected: bool = False
    spread_confidence: float = 0.0
    spread_reason: str | None = None
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()
    boundary_dark_border_fraction: float = 0.0


@dataclass(slots=True, frozen=True)
class BatchPipelineResult:
    """Summary of one completed headless conversion."""

    output_pdf: Path
    report_path: Path
    input_files: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    total_pages: int
    detected_pages: int
    fallback_pages: int
    review_pages: int
    pages: tuple[PageRunReport, ...]


@dataclass(slots=True, frozen=True)
class _StagedTarget:
    staged: Path
    target: Path


def resolve_input_paths(inputs: Sequence[Path], *, output_pdf: Path) -> tuple[Path, ...]:
    """Expand files and folders while preserving argument and natural folder order."""
    if not inputs:
        raise ValueError("At least one input file or folder is required.")

    output_resolved = output_pdf.with_suffix(".pdf").resolve()
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")

        if path.is_dir():
            candidates = list_supported_in_folder(path)
        elif path.is_file():
            if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS):
                raise ValueError(f"Unsupported input: {path}")
            if path.resolve() == output_resolved:
                raise ValueError("Output PDF cannot also be an explicit input file.")
            candidates = [path]
        else:
            raise ValueError(f"Input is neither a file nor a folder: {path}")

        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if candidate_resolved == output_resolved:
                raise ValueError("Output PDF cannot also be an input discovered in a folder.")
            if candidate_resolved in seen:
                continue
            seen.add(candidate_resolved)
            resolved.append(candidate)

    if not resolved:
        raise ValueError("No supported image or PDF inputs were found.")
    return tuple(resolved)


def _resolve_processing(mode: str):
    normalized = mode.strip().lower()
    if normalized == "none":
        return "None", None, None

    profiles_by_key = {
        name.lower(): profile
        for name, profile in (
            (name, resolve_lens_mode_profile(name))
            for name in ("Document", "Grayscale", "Whiteboard", "Photo", "B/W")
        )
    }
    profile = profiles_by_key.get(normalized)
    if profile is None:
        raise ValueError(f"Unsupported lens mode: {mode}")
    return profile.postprocess_name, profile.preset_name, PREPROCESS_PRESETS[profile.preset_name]


def _resolve_detector_policy(policy: str) -> tuple[str, ...]:
    try:
        return _DETECTOR_POLICIES[policy.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector policy: {policy}") from exc


def _validate_output_targets(
    *,
    output_pdf: Path,
    report_path: Path,
    images_dir: Path | None,
    input_files: Sequence[Path],
) -> None:
    if _is_link_like(output_pdf):
        raise ValueError("PDF output path cannot be a link or junction.")
    if _is_link_like(report_path):
        raise ValueError("JSON report path cannot be a link or junction.")
    output_resolved = output_pdf.resolve()
    report_resolved = report_path.resolve()
    input_resolved = tuple(path.resolve() for path in input_files)

    if output_pdf.exists() and not output_pdf.is_file():
        raise ValueError("PDF output path exists and is not a file.")
    if report_path.exists() and not report_path.is_file():
        raise ValueError("JSON report path exists and is not a file.")
    if output_resolved == report_resolved:
        raise ValueError("PDF output and JSON report must use different paths.")
    if output_resolved in input_resolved:
        raise ValueError("PDF output cannot also be an input file.")
    if report_resolved in input_resolved:
        raise ValueError("JSON report cannot also be an input file.")

    # A file target used as another target's parent would make publication order
    # destructive or impossible (for example ``out.pdf/report.json``).
    if output_resolved.is_relative_to(report_resolved) or report_resolved.is_relative_to(
        output_resolved
    ):
        raise ValueError("PDF output and JSON report cannot contain one another.")

    if images_dir is None:
        return

    images_resolved = images_dir.resolve()
    if _is_link_like(images_dir):
        raise ValueError("Images output path cannot be a link or junction.")
    if images_dir.exists() and not images_dir.is_dir():
        raise ValueError("Images output path exists and is not a directory.")
    if output_resolved.is_relative_to(images_resolved):
        raise ValueError("PDF output cannot be inside the replaceable images directory.")
    if report_resolved.is_relative_to(images_resolved):
        raise ValueError("JSON report cannot be inside the replaceable images directory.")
    if images_resolved.is_relative_to(output_resolved) or images_resolved.is_relative_to(
        report_resolved
    ):
        raise ValueError("Images output directory cannot be nested below a file output path.")
    if any(path.is_relative_to(images_resolved) for path in input_resolved):
        raise ValueError("Images output directory cannot contain input files.")
    if images_resolved in input_resolved:
        raise ValueError("Images output directory cannot also be an input file.")


def _new_stage_file(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.stage-",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _remove_path(path: Path) -> None:
    if _is_link_like(path):
        try:
            path.unlink()
        except (IsADirectoryError, PermissionError):
            path.rmdir()
    elif path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


_TRANSACTION_VERSION = 1
_TRANSACTION_SUFFIX = ".uniscan-transaction.json"


def _path_exists(path: Path) -> bool:
    return path.exists() or _is_link_like(path)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _path_key(path: Path) -> str:
    return os.path.normcase(str(_absolute_path(path)))


def _cleanup_path_best_effort(path: Path) -> bool:
    try:
        if _path_exists(path):
            _remove_path(path)
    except OSError:
        return False
    return not _path_exists(path)


def _fsync_parent(path: Path) -> None:
    """Best-effort directory flush; Windows does not expose it on every filesystem."""
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    if _is_link_like(path) or (path.exists() and not path.is_file()):
        raise ValueError(f"Transaction journal path is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_staged = tempfile.mkstemp(
        prefix=f".{path.name}.stage-",
        suffix=".json",
        dir=path.parent,
    )
    staged = Path(raw_staged)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        _fsync_parent(path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _cleanup_path_best_effort(staged)
        raise


def _transaction_journal_path(output_pdf: Path) -> Path:
    output_pdf = _canonical_export_target(output_pdf, description="PDF output path")
    return output_pdf.parent / f".{output_pdf.name}{_TRANSACTION_SUFFIX}"


@contextmanager
def _transaction_lock(journal_path: Path):
    """Hold a non-blocking inter-process lock for recovery/publication."""
    lock_path = journal_path.with_name(f"{journal_path.name}.lock")
    with _exclusive_export_lock(
        lock_path,
        invalid_path_message="Invalid transaction lock path",
        conflict_message="Another UniScan process is publishing or recovering these outputs.",
    ):
        yield


def _transaction_target_specs(
    *,
    output_pdf: Path,
    images_dir: Path | None,
    report_path: Path,
) -> tuple[tuple[Path, str], ...]:
    specs: list[tuple[Path, str]] = [
        (_canonical_export_target(output_pdf, description="PDF output path"), "file")
    ]
    if images_dir is not None:
        specs.append(
            (_canonical_export_target(images_dir, description="Images output path"), "directory")
        )
    specs.append((_canonical_export_target(report_path, description="JSON report path"), "file"))
    return tuple(specs)


@contextmanager
def _transaction_output_locks(
    journal_path: Path,
    *,
    expected_targets: Sequence[tuple[Path, str]],
):
    """Lock every canonical final target in stable order, then the batch journal."""
    output_targets = {
        _path_key(path): (_absolute_path(path), kind) for path, kind in expected_targets
    }
    with ExitStack() as stack:
        for key in sorted(output_targets):
            path, kind = output_targets[key]
            lock = _directory_export_lock if kind == "directory" else _file_export_lock
            stack.enter_context(lock(path))
        stack.enter_context(_transaction_lock(journal_path))
        yield


def _validate_transaction_path_type(path: Path, *, kind: str, role: str) -> None:
    if not _path_exists(path):
        return
    if _is_link_like(path):
        raise ValueError(f"Invalid transaction journal: {role} path is a link: {path}")
    if kind == "directory":
        if _is_link_like(path) or not path.is_dir():
            raise ValueError(f"Invalid transaction journal directory path: {path}")
    elif kind == "file":
        if not path.is_file():
            raise ValueError(f"Invalid transaction journal file path: {path}")
    else:  # pragma: no cover - guarded by strict journal schema validation
        raise ValueError(f"Invalid transaction target kind: {kind}")


def _validate_transaction_payload(
    payload: object,
    *,
    expected_targets: Sequence[tuple[Path, str]],
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "transactionId",
        "state",
        "entries",
    }:
        raise ValueError("Invalid UniScan transaction journal schema.")
    transaction_id = payload.get("transactionId")
    entries = payload.get("entries")
    if (
        type(payload.get("schemaVersion")) is not int
        or payload["schemaVersion"] != _TRANSACTION_VERSION
        or not isinstance(transaction_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
        or payload.get("state") not in {"prepared", "committed"}
        or not isinstance(entries, list)
        or len(entries) != len(expected_targets)
    ):
        raise ValueError("Invalid UniScan transaction journal schema.")

    seen_paths: set[str] = set()
    for raw_entry, (expected_target, expected_kind) in zip(entries, expected_targets, strict=True):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "target",
            "staged",
            "backup",
            "kind",
            "hadTarget",
        }:
            raise ValueError("Invalid UniScan transaction journal entry.")
        if (
            not all(isinstance(raw_entry.get(key), str) for key in ("target", "staged", "backup"))
            or raw_entry.get("kind") != expected_kind
            or type(raw_entry.get("hadTarget")) is not bool
        ):
            raise ValueError("Invalid UniScan transaction journal entry.")

        target = Path(raw_entry["target"])
        staged = Path(raw_entry["staged"])
        backup = Path(raw_entry["backup"])
        if not all(path.is_absolute() for path in (target, staged, backup)):
            raise ValueError("Invalid UniScan transaction journal: paths must be absolute.")
        expected_target = _absolute_path(expected_target)
        if _path_key(target) != _path_key(expected_target):
            raise ValueError("Transaction journal targets do not match this invocation.")
        if _path_key(staged.parent) != _path_key(
            expected_target.parent
        ) or not staged.name.startswith(f".{expected_target.name}.stage-"):
            raise ValueError("Invalid UniScan transaction staged path.")
        expected_backup_name = f".{expected_target.name}.backup-{transaction_id}"
        if (
            _path_key(backup.parent) != _path_key(expected_target.parent)
            or backup.name != expected_backup_name
        ):
            raise ValueError("Invalid UniScan transaction backup path.")
        entry_paths = {_path_key(target), _path_key(staged), _path_key(backup)}
        if len(entry_paths) != 3 or seen_paths.intersection(entry_paths):
            raise ValueError("Invalid UniScan transaction journal: duplicate paths.")
        seen_paths.update(entry_paths)

        _validate_transaction_path_type(target, kind=expected_kind, role="target")
        _validate_transaction_path_type(staged, kind=expected_kind, role="staged")
        _validate_transaction_path_type(backup, kind=expected_kind, role="backup")
        target_exists = _path_exists(target)
        staged_exists = _path_exists(staged)
        backup_exists = _path_exists(backup)
        had_target = raw_entry["hadTarget"]
        if payload["state"] == "committed":
            if not target_exists or staged_exists:
                raise ValueError("Invalid committed UniScan transaction state.")
        elif had_target:
            if not target_exists and not backup_exists:
                raise ValueError("Invalid prepared UniScan transaction state.")
            if target_exists and staged_exists and backup_exists:
                raise ValueError("Ambiguous prepared UniScan transaction state.")
        else:
            if backup_exists or (target_exists and staged_exists):
                raise ValueError("Invalid prepared UniScan transaction state.")
    return payload


def _read_transaction_journal(journal_path: Path) -> object:
    if _is_link_like(journal_path) or not journal_path.is_file():
        raise ValueError("Invalid UniScan transaction journal path.")
    if journal_path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid UniScan transaction journal: file is too large.")
    try:
        return json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UniScan transaction journal.") from exc


def _load_transaction_journal(
    journal_path: Path,
    *,
    expected_targets: Sequence[tuple[Path, str]],
) -> dict[str, object]:
    payload = _read_transaction_journal(journal_path)
    return _validate_transaction_payload(payload, expected_targets=expected_targets)


def _discover_owned_transaction(
    journal_path: Path,
    *,
    output_pdf: Path,
) -> tuple[dict[str, object], tuple[tuple[Path, str], ...]]:
    """Validate an output-owned journal and return its recorded final targets."""
    payload = _read_transaction_journal(journal_path)
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "transactionId",
        "state",
        "entries",
    }:
        raise ValueError("Invalid UniScan transaction journal schema.")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) not in {2, 3}:
        raise ValueError("Invalid UniScan transaction journal schema.")

    expected_kinds = (
        ("file", "file")
        if len(entries) == 2
        else (
            "file",
            "directory",
            "file",
        )
    )
    targets: list[tuple[Path, str]] = []
    for entry, expected_kind in zip(entries, expected_kinds, strict=True):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("target"), str)
            or entry.get("kind") != expected_kind
        ):
            raise ValueError("Invalid UniScan transaction journal entry.")
        recorded_target = Path(entry["target"])
        if not recorded_target.is_absolute():
            raise ValueError("Invalid UniScan transaction journal: paths must be absolute.")
        try:
            canonical_target = _canonical_export_target(
                recorded_target,
                description="Recorded transaction target",
            )
        except ValueError as exc:
            raise ValueError("Invalid UniScan transaction target path.") from exc
        if recorded_target != canonical_target:
            raise ValueError("Invalid UniScan transaction target path.")
        targets.append((canonical_target, expected_kind))

    output_pdf = _canonical_export_target(output_pdf, description="PDF output path")
    if _path_key(targets[0][0]) != _path_key(output_pdf):
        raise ValueError("Transaction journal targets do not match this PDF output ownership.")
    expected_journal = _transaction_journal_path(targets[0][0])
    if _path_key(journal_path) != _path_key(expected_journal):
        raise ValueError("Transaction journal targets do not match this PDF output ownership.")

    recorded_images = targets[1][0] if len(targets) == 3 else None
    _validate_output_targets(
        output_pdf=targets[0][0],
        report_path=targets[-1][0],
        images_dir=recorded_images,
        input_files=(),
    )
    validated = _validate_transaction_payload(payload, expected_targets=targets)
    return validated, tuple(targets)


def _validate_recovery_compatibility(
    payload: dict[str, object],
    *,
    journal_path: Path,
    recorded_targets: Sequence[tuple[Path, str]],
    current_targets: Sequence[tuple[Path, str]],
) -> None:
    """Reject only path ownership overlaps that could mutate current outputs."""

    def overlaps(first: Path, second: Path) -> bool:
        first = _absolute_path(first)
        second = _absolute_path(second)
        return (
            _path_key(first) == _path_key(second)
            or first.is_relative_to(second)
            or second.is_relative_to(first)
        )

    for current_path, current_kind in current_targets:
        for recorded_path, recorded_kind in recorded_targets:
            if _path_key(current_path) == _path_key(recorded_path):
                if current_kind != recorded_kind:
                    raise ValueError(
                        "Current output targets conflict with recorded transaction ownership."
                    )
                continue
            if overlaps(current_path, recorded_path):
                raise ValueError(
                    "Current output targets conflict with recorded transaction ownership."
                )

    protected_paths = [
        journal_path,
        journal_path.with_name(f"{journal_path.name}.lock"),
    ]
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        protected_paths.extend((Path(entry["staged"]), Path(entry["backup"])))
    for recorded_path, recorded_kind in recorded_targets:
        lock_path = (
            _directory_export_lock_path(recorded_path)
            if recorded_kind == "directory"
            else _file_export_lock_path(recorded_path)
        )
        protected_paths.append(lock_path)

    for current_path, _current_kind in current_targets:
        if any(overlaps(current_path, protected_path) for protected_path in protected_paths):
            raise ValueError("Current output targets conflict with recorded transaction ownership.")


def _rollback_prepared_transaction(payload: dict[str, object]) -> bool:
    entries = payload["entries"]
    assert isinstance(entries, list)
    cleanup_complete = True
    for raw_entry in reversed(entries):
        assert isinstance(raw_entry, dict)
        target = Path(raw_entry["target"])
        staged = Path(raw_entry["staged"])
        backup = Path(raw_entry["backup"])
        if _path_exists(backup):
            if _path_exists(target):
                _remove_path(target)
            os.replace(backup, target)
            _fsync_parent(target)
        elif not raw_entry["hadTarget"] and _path_exists(target):
            _remove_path(target)
            _fsync_parent(target)
        cleanup_complete = _cleanup_path_best_effort(staged) and cleanup_complete
        cleanup_complete = not _path_exists(backup) and cleanup_complete
    return cleanup_complete


def _recover_batch_transaction(
    journal_path: Path,
    *,
    expected_targets: Sequence[tuple[Path, str]],
    locks_held: bool = False,
) -> None:
    if not locks_held:
        with _transaction_output_locks(
            journal_path,
            expected_targets=expected_targets,
        ):
            _recover_batch_transaction(
                journal_path,
                expected_targets=expected_targets,
                locks_held=True,
            )
        return
    if not _path_exists(journal_path):
        return
    payload = _load_transaction_journal(journal_path, expected_targets=expected_targets)
    if payload["state"] == "prepared":
        cleanup_complete = _rollback_prepared_transaction(payload)
    else:
        entries = payload["entries"]
        assert isinstance(entries, list)
        cleanup_complete = True
        for raw_entry in entries:
            assert isinstance(raw_entry, dict)
            cleanup_complete = (
                _cleanup_path_best_effort(Path(raw_entry["backup"])) and cleanup_complete
            )
            cleanup_complete = (
                _cleanup_path_best_effort(Path(raw_entry["staged"])) and cleanup_complete
            )
    if not cleanup_complete:
        raise RuntimeError(
            "A previous UniScan transaction is committed/recovered, but locked debris "
            "could not be cleaned; close programs using the output and retry."
        )
    if not _cleanup_path_best_effort(journal_path):
        raise RuntimeError("Recovered transaction journal is locked; retry after closing it.")


def _publish_staged_targets(
    targets: Sequence[_StagedTarget],
    *,
    journal_path: Path,
) -> None:
    expected_targets = tuple(
        (
            _absolute_path(item.target),
            "directory" if item.staged.is_dir() else "file",
        )
        for item in targets
    )
    try:
        with _transaction_output_locks(
            journal_path,
            expected_targets=expected_targets,
        ):
            _recover_batch_transaction(
                journal_path,
                expected_targets=expected_targets,
                locks_held=True,
            )
            _publish_staged_targets_locked(targets, journal_path=journal_path)
    except Exception:
        for item in targets:
            _cleanup_path_best_effort(item.staged)
        raise


def _publish_staged_targets_locked(
    targets: Sequence[_StagedTarget],
    *,
    journal_path: Path,
) -> None:
    """Durably publish a recoverable multi-target transaction."""
    for item in targets:
        if not item.staged.exists():
            raise RuntimeError(f"Staged output is missing: {item.staged}")
        if item.target.exists() and item.target.is_dir() != item.staged.is_dir():
            raise ValueError(f"Output target has the wrong type: {item.target}")
    if _path_exists(journal_path):
        raise ValueError(f"Unrecovered transaction journal already exists: {journal_path}")

    transaction_id = uuid.uuid4().hex
    entries: list[dict[str, object]] = []
    for item in targets:
        target = _absolute_path(item.target)
        staged = _absolute_path(item.staged)
        backup = target.with_name(f".{target.name}.backup-{transaction_id}")
        if _path_exists(backup):
            raise ValueError(f"Transaction backup path already exists: {backup}")
        entries.append(
            {
                "target": str(target),
                "staged": str(staged),
                "backup": str(backup),
                "kind": "directory" if staged.is_dir() else "file",
                "hadTarget": _path_exists(target),
            }
        )
    payload: dict[str, object] = {
        "schemaVersion": _TRANSACTION_VERSION,
        "transactionId": transaction_id,
        "state": "prepared",
        "entries": entries,
    }
    try:
        _atomic_write_json(journal_path, payload)
        for entry in entries:
            target = Path(entry["target"])
            staged = Path(entry["staged"])
            backup = Path(entry["backup"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["hadTarget"]:
                os.replace(target, backup)
            os.replace(staged, target)
            _fsync_parent(target)
        payload["state"] = "committed"
        _atomic_write_json(journal_path, payload)
    except Exception:
        if _path_exists(journal_path):
            try:
                rollback_clean = _rollback_prepared_transaction(payload)
                if rollback_clean:
                    _cleanup_path_best_effort(journal_path)
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"Output publication failed and rollback is incomplete: {rollback_exc}"
                ) from rollback_exc
        else:
            for item in targets:
                _cleanup_path_best_effort(item.staged)
        raise

    cleanup_complete = True
    for entry in entries:
        cleanup_complete = _cleanup_path_best_effort(Path(entry["backup"])) and cleanup_complete
        cleanup_complete = _cleanup_path_best_effort(Path(entry["staged"])) and cleanup_complete
    if cleanup_complete:
        _cleanup_path_best_effort(journal_path)


def _report_payload(
    *,
    output_pdf: Path,
    report_path: Path,
    images_dir: Path | None,
    image_outputs: Sequence[Path],
    image_format: str,
    input_pdf_dpi: int,
    output_pdf_dpi: int,
    pdf_jpeg_quality: int | None,
    max_input_pixels: int,
    input_files: Sequence[Path],
    pages: Sequence[PageRunReport],
    detect_document: bool,
    detector_policy: str,
    detector_backends: Sequence[str],
    strict_detect: bool,
    two_page_mode: bool,
    lens_mode: str,
    preprocess_preset: str | None,
    postprocess_name: str,
    preprocess_settings: PreprocessSettings | None,
    illumination_correction: bool,
    legacy_illumination_correction: bool,
    orientation_method: str,
    deskew_method: str,
    deskew_angle_degrees: float | None,
    dewarp_method: str,
    auto_dewarp_uvdoc: bool,
    auto_dewarp_uvdoc_grid: bool,
    shadow_method: str,
    page_layout: str,
    page_margin_mm: float,
    horizontal_alignment: str,
    vertical_alignment: str,
    binarization_method: str,
    binarization_window: int,
    binarization_k: float | None,
    despeckle_strength: str,
    lighting_diagnostics: bool,
    stage_cache_enabled: bool,
    stage_cache_dir: Path | None,
    stage_cache_max_mb: int,
    stage_cache_stats: dict[str, int],
    uvdoc_cache_home: Path | None,
) -> dict[str, object]:
    detected_pages = sum(page.detected for page in pages)
    fallback_pages = sum(page.fallback_reason is not None for page in pages)
    review_pages = sum(page.needs_review for page in pages)
    effective_preprocess = preprocess_settings or PreprocessSettings()
    return {
        "schemaVersion": 5,
        "outputPdf": str(output_pdf),
        "reportPath": str(report_path),
        "imagesDirectory": str(images_dir) if images_dir is not None else None,
        "imageOutputs": [str(path) for path in image_outputs],
        "imageFormat": image_format,
        # Kept for report consumers written before the two DPI roles were split.
        "pdfDpi": output_pdf_dpi,
        "inputPdfDpi": input_pdf_dpi,
        "outputPdfDpi": output_pdf_dpi,
        "pdfJpegQuality": pdf_jpeg_quality,
        "maxInputPixels": max_input_pixels,
        "inputFiles": [str(path) for path in input_files],
        "detectionEnabled": detect_document,
        "detectorPolicy": detector_policy,
        "detectorBackends": list(detector_backends),
        "strictDetect": strict_detect,
        "twoPageMode": two_page_mode,
        "lensMode": lens_mode,
        "preprocessPreset": preprocess_preset,
        "postprocessName": postprocess_name,
        "preprocessEnabled": preprocess_settings is not None,
        "contrast": effective_preprocess.contrast,
        "brightness": effective_preprocess.brightness,
        "denoise": effective_preprocess.denoise,
        "threshold": effective_preprocess.threshold,
        "applyThreshold": effective_preprocess.apply_threshold,
        "illuminationCorrection": illumination_correction,
        "legacyIlluminationCorrectionRequested": legacy_illumination_correction,
        "orientationMethod": orientation_method,
        "deskewMethod": deskew_method,
        "deskewRequestedAngleDegrees": deskew_angle_degrees,
        "dewarpMethod": dewarp_method,
        "autoDewarpUvdoc": auto_dewarp_uvdoc,
        "autoDewarpPageModel": auto_dewarp_uvdoc_grid,
        "shadowMethod": shadow_method,
        "pageLayout": page_layout,
        "pageMarginMm": page_margin_mm,
        "horizontalAlignment": horizontal_alignment,
        "verticalAlignment": vertical_alignment,
        "binarizationMethod": binarization_method,
        "binarizationWindow": binarization_window,
        "binarizationK": binarization_k,
        "despeckleStrength": despeckle_strength,
        "lightingDiagnostics": lighting_diagnostics,
        "stageCacheEnabled": stage_cache_enabled,
        "stageCacheDir": str(stage_cache_dir) if stage_cache_dir is not None else None,
        "stageCacheMaxMb": stage_cache_max_mb,
        "stageCacheStats": stage_cache_stats,
        "uvdocCacheHome": str(uvdoc_cache_home) if uvdoc_cache_home is not None else None,
        "totalPages": len(pages),
        "detectedPages": detected_pages,
        "fallbackPages": fallback_pages,
        "needsReviewPages": review_pages,
        "pages": [
            {
                "index": page.index,
                "name": page.name,
                "detected": page.detected,
                "backend": page.backend,
                "fallbackReason": page.fallback_reason,
                "durationMs": page.duration_ms,
                "orientationMethod": page.orientation_method,
                "orientationApplied": page.orientation_applied,
                "orientationAngleDegrees": page.orientation_angle_degrees,
                "orientationConfidence": page.orientation_confidence,
                "orientationReason": page.orientation_reason,
                "deskewMethod": page.deskew_method,
                "deskewAngleDegrees": page.deskew_angle_degrees,
                "deskewSelectedMethod": page.deskew_selected_method,
                "deskewConfidence": page.deskew_confidence,
                "deskewLineCount": page.deskew_line_count,
                "deskewReason": page.deskew_reason,
                "dewarpMethod": page.dewarp_method,
                "dewarpApplied": page.dewarp_applied,
                "dewarpSelectedMethod": page.dewarp_selected_method,
                "dewarpLineCount": page.dewarp_line_count,
                "dewarpMaxDisplacementPx": page.dewarp_max_displacement_px,
                "dewarpCurvatureBeforePx": page.dewarp_curvature_before_px,
                "dewarpCurvatureAfterPx": page.dewarp_curvature_after_px,
                "dewarpPerspectiveBefore": page.dewarp_perspective_before,
                "dewarpPerspectiveAfter": page.dewarp_perspective_after,
                "dewarpBlankBorderBefore": page.dewarp_blank_border_before,
                "dewarpBlankBorderAfter": page.dewarp_blank_border_after,
                "dewarpEdgeInkBefore": page.dewarp_edge_ink_before,
                "dewarpEdgeInkAfter": page.dewarp_edge_ink_after,
                "dewarpAspectChange": page.dewarp_aspect_change,
                "dewarpDurationMs": page.dewarp_duration_ms,
                "dewarpReason": page.dewarp_reason,
                "shadowMethod": page.shadow_method,
                "shadowApplied": page.shadow_applied,
                "shadowSelectedMethod": page.shadow_selected_method,
                "shadowUnevennessBefore": page.shadow_unevenness_before,
                "shadowUnevennessAfter": page.shadow_unevenness_after,
                "shadowBefore": page.shadow_before,
                "shadowAfter": page.shadow_after,
                "shadowGlareAfter": page.shadow_glare_after,
                "shadowDurationMs": page.shadow_duration_ms,
                "shadowReason": page.shadow_reason,
                "pageLayout": page.page_layout,
                "layoutApplied": page.layout_applied,
                "contentBox": list(page.content_box) if page.content_box is not None else None,
                "contentConfidence": page.content_confidence,
                "layoutScale": page.layout_scale,
                "layoutReason": page.layout_reason,
                "binarizationMethod": page.binarization_method,
                "despeckleStrength": page.despeckle_strength,
                "despeckleRemovedComponents": page.despeckle_removed_components,
                "despeckleRemovedPixels": page.despeckle_removed_pixels,
                "despeckleProtectedComponents": page.despeckle_protected_components,
                "shadowFraction": page.shadow_fraction,
                "glareFraction": page.glare_fraction,
                "clippedPixelFraction": page.clipped_pixel_fraction,
                "lightingUnevenness": page.lighting_unevenness,
                "lightingWarnings": list(page.lighting_warnings),
                "processingStageDurationsMs": page.processing_stage_durations_ms or {},
                "processingCacheHits": list(page.processing_cache_hits),
                "spreadDetected": page.spread_detected,
                "spreadConfidence": page.spread_confidence,
                "spreadReason": page.spread_reason,
                "needsReview": page.needs_review,
                "reviewReasons": list(page.review_reasons),
                "boundaryDarkBorderFraction": page.boundary_dark_border_fraction,
            }
            for page in pages
        ],
    }


def _stage_outputs(
    *,
    staged_page_paths: Sequence[Path],
    output_pdf: Path,
    images_dir: Path | None,
    image_format: str,
    report_path: Path,
    report_payload: dict[str, object],
    dpi: int,
    pdf_jpeg_quality: int | None,
    cancel_cb: CancelCb | None,
) -> tuple[list[_StagedTarget], tuple[Path, ...]]:
    """Prepare every output beside its target and clean all stages on failure."""
    targets: list[_StagedTarget] = []
    final_image_paths: tuple[Path, ...] = ()
    try:
        staged_pdf = _new_stage_file(output_pdf)
        targets.append(_StagedTarget(staged=staged_pdf, target=output_pdf))
        staging_pdf_lock = _file_export_lock_path(staged_pdf)
        try:
            export_image_paths_as_pdf(
                staged_page_paths,
                out_pdf=staged_pdf,
                dpi=dpi,
                jpeg_quality=pdf_jpeg_quality,
                cancel_cb=cancel_cb,
            )
        finally:
            _cleanup_path_best_effort(staging_pdf_lock)

        if images_dir is not None:
            images_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_images_dir = Path(
                tempfile.mkdtemp(prefix=f".{images_dir.name}.stage-", dir=images_dir.parent)
            )
            targets.append(_StagedTarget(staged=staged_images_dir, target=images_dir))
            if images_dir.exists():
                # The images directory may contain user-owned neighbours.  Seed
                # the transaction stage from the live directory, then let the
                # ownership-aware exporter replace only files from its manifest.
                _clone_directory_for_staging(images_dir, staged_images_dir)
            staging_lock = _directory_export_lock_path(staged_images_dir)
            try:
                staged_images = export_image_paths_as_files(
                    staged_page_paths,
                    output_dir=staged_images_dir,
                    ext=image_format,
                    base_name="page",
                    cancel_cb=cancel_cb,
                )
            finally:
                _cleanup_path_best_effort(staging_lock)
            final_image_paths = tuple(images_dir / path.name for path in staged_images)

        report_payload["imageOutputs"] = [str(path) for path in final_image_paths]
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Cancelled by user.")
        staged_report = _new_stage_file(report_path)
        targets.append(_StagedTarget(staged=staged_report, target=report_path))
        with staged_report.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report_payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return targets, final_image_paths
    except Exception:
        for item in targets:
            if item.staged.exists():
                _remove_path(item.staged)
        raise


def run_batch_pipeline(
    *,
    inputs: Sequence[Path],
    output_pdf: Path,
    images_dir: Path | None = None,
    image_format: str = "png",
    report_path: Path | None = None,
    pdf_dpi: int = 300,
    input_pdf_dpi: int | None = None,
    output_pdf_dpi: int | None = None,
    pdf_jpeg_quality: int | None = None,
    max_input_pixels: int = DEFAULT_MAX_INPUT_PIXELS,
    detect_document: bool = False,
    detector_policy: str = "auto",
    strict_detect: bool = False,
    two_page_mode: bool = False,
    lens_mode: str = "none",
    illumination_correction: bool = False,
    orientation_method: str = "none",
    deskew_method: str = "none",
    deskew_angle_degrees: float | None = None,
    dewarp_method: str = "none",
    auto_dewarp_uvdoc: bool = False,
    auto_dewarp_uvdoc_grid: bool = True,
    shadow_method: str = "none",
    page_layout: str = "none",
    page_margin_mm: float = 10.0,
    horizontal_alignment: str = "center",
    vertical_alignment: str = "center",
    binarization_method: str = BINARIZATION_NONE,
    binarization_window: int = 31,
    binarization_k: float | None = None,
    despeckle_strength: str = DESPECKLE_NONE,
    lighting_diagnostics: bool = False,
    stage_cache_dir: Path | None = None,
    stage_cache_max_mb: int = 512,
    uvdoc_cache_home: Path | None = None,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> BatchPipelineResult:
    """Run the complete streaming pre-OCR pipeline and atomically publish its outputs."""
    legacy_illumination_correction = bool(illumination_correction)
    legacy_dpi = int(pdf_dpi)
    input_dpi = int(input_pdf_dpi) if input_pdf_dpi is not None else legacy_dpi
    output_dpi = int(output_pdf_dpi) if output_pdf_dpi is not None else legacy_dpi
    max_input_pixels = int(max_input_pixels)
    if pdf_jpeg_quality == 0:
        pdf_jpeg_quality = None
    elif pdf_jpeg_quality is not None:
        pdf_jpeg_quality = int(pdf_jpeg_quality)
    if input_dpi < 72:
        raise ValueError("Input PDF DPI must be >= 72.")
    if output_dpi < 72:
        raise ValueError("Output PDF DPI must be >= 72.")
    if max_input_pixels < 1:
        raise ValueError("Maximum input pixel count must be positive.")
    if pdf_jpeg_quality is not None and not 1 <= pdf_jpeg_quality <= 100:
        raise ValueError("PDF JPEG quality must be between 1 and 100, or 0 for lossless.")
    if strict_detect and not detect_document:
        raise ValueError("strict_detect cannot be used when document detection is disabled.")
    image_format = image_format.strip().lower().lstrip(".")
    lens_mode = lens_mode.strip().lower()
    detector_policy = detector_policy.strip().lower()
    orientation_method = orientation_method.strip().lower()
    deskew_method = deskew_method.strip().lower()
    dewarp_method = dewarp_method.strip().lower()
    shadow_method = shadow_method.strip().lower()
    # The old flag is now only a compatibility spelling for the classical
    # method. An explicit --shadow value wins, and only one stage ever runs.
    if illumination_correction and shadow_method == SHADOW_METHOD_NONE:
        shadow_method = SHADOW_METHOD_CLASSICAL
    page_layout = page_layout.strip().lower()
    binarization_method = binarization_method.strip().lower()
    despeckle_strength = despeckle_strength.strip().lower()
    if image_format not in IMAGE_FORMAT_CHOICES:
        raise ValueError(f"Unsupported image format: {image_format}")
    if lens_mode not in LENS_MODE_CHOICES:
        raise ValueError(f"Unsupported lens mode: {lens_mode}")
    if detector_policy == "paddleocr_uvdoc":
        raise ValueError(
            "UVDoc is a dewarp method, not a boundary detector; "
            "use dewarp_method='paddleocr_uvdoc'."
        )
    if detector_policy not in DETECTOR_POLICY_CHOICES:
        raise ValueError(f"Unsupported detector policy: {detector_policy}")
    if orientation_method not in ORIENTATION_METHOD_CHOICES:
        raise ValueError(f"Unsupported orientation method: {orientation_method}")
    if deskew_method not in DESKEW_METHOD_CHOICES:
        raise ValueError(f"Unsupported deskew method: {deskew_method}")
    if deskew_method == DESKEW_METHOD_MANUAL:
        if deskew_angle_degrees is None:
            raise ValueError("Manual deskew requires deskew_angle_degrees.")
        deskew_angle_degrees = float(deskew_angle_degrees)
        if not -20.0 <= deskew_angle_degrees <= 20.0:
            raise ValueError("Manual deskew angle must be within +/-20 degrees.")
    elif deskew_angle_degrees is not None:
        raise ValueError("deskew_angle_degrees requires deskew_method='manual'.")
    if dewarp_method not in DEWARP_METHOD_CHOICES:
        raise ValueError(f"Unsupported dewarp method: {dewarp_method}")
    if shadow_method not in SHADOW_METHOD_CHOICES:
        raise ValueError(f"Unsupported shadow removal method: {shadow_method}")
    if auto_dewarp_uvdoc and dewarp_method != DEWARP_METHOD_AUTO:
        raise ValueError("auto_dewarp_uvdoc requires dewarp_method='auto'.")
    if page_layout not in PAGE_LAYOUT_CHOICES:
        raise ValueError(f"Unsupported page layout: {page_layout}")
    if horizontal_alignment not in HORIZONTAL_ALIGNMENTS:
        raise ValueError(f"Unsupported horizontal alignment: {horizontal_alignment}")
    if vertical_alignment not in VERTICAL_ALIGNMENTS:
        raise ValueError(f"Unsupported vertical alignment: {vertical_alignment}")
    if binarization_method not in BINARIZATION_CHOICES:
        raise ValueError(f"Unsupported binarization method: {binarization_method}")
    if int(binarization_window) < 3:
        raise ValueError("Binarization window must be >= 3.")
    if binarization_k is not None and not 0.0 <= float(binarization_k) <= 1.0:
        raise ValueError("Binarization k must be between 0 and 1.")
    if despeckle_strength not in DESPECKLE_CHOICES:
        raise ValueError(f"Unsupported despeckle strength: {despeckle_strength}")
    if int(stage_cache_max_mb) < 1:
        raise ValueError("Stage cache size must be at least 1 MiB.")

    output_pdf = _canonical_export_target(
        Path(output_pdf).with_suffix(".pdf"),
        description="PDF output path",
    )
    report_path = _canonical_export_target(
        Path(report_path) if report_path else output_pdf.with_suffix(".pdf.report.json"),
        description="JSON report path",
    )
    images_dir = (
        _canonical_export_target(Path(images_dir), description="Images output path")
        if images_dir is not None
        else None
    )
    # Validate relationships before acquiring recovery locks: a nested target's
    # lock directory could otherwise create an invalid file target path.
    _validate_output_targets(
        output_pdf=output_pdf,
        report_path=report_path,
        images_dir=images_dir,
        input_files=(),
    )
    transaction_journal = _transaction_journal_path(output_pdf)
    transaction_targets = _transaction_target_specs(
        output_pdf=output_pdf,
        images_dir=images_dir,
        report_path=report_path,
    )
    # Recovery depends only on the durable journal and its output targets.  Do
    # it before resolving inputs so a disconnected source drive cannot strand
    # an otherwise recoverable publication transaction.
    if _path_exists(transaction_journal):
        recorded_payload, recorded_targets = _discover_owned_transaction(
            transaction_journal,
            output_pdf=output_pdf,
        )
        _validate_recovery_compatibility(
            recorded_payload,
            journal_path=transaction_journal,
            recorded_targets=recorded_targets,
            current_targets=transaction_targets,
        )
        _recover_batch_transaction(
            transaction_journal,
            expected_targets=recorded_targets,
        )
    input_files = resolve_input_paths(inputs, output_pdf=output_pdf)
    _validate_output_targets(
        output_pdf=output_pdf,
        report_path=report_path,
        images_dir=images_dir,
        input_files=input_files,
    )

    postprocess_name, preprocess_preset, preprocess_settings = _resolve_processing(lens_mode)
    if preprocess_settings is not None:
        preprocess_settings = replace(
            preprocess_settings,
            correct_illumination=False,
            binarization_method=binarization_method,
            binarization_window=int(binarization_window),
            binarization_k=binarization_k,
            despeckle_strength=despeckle_strength,
        )
    elif binarization_method != BINARIZATION_NONE or despeckle_strength != DESPECKLE_NONE:
        preprocess_settings = PreprocessSettings(
            correct_illumination=False,
            binarization_method=binarization_method,
            binarization_window=int(binarization_window),
            binarization_k=binarization_k,
            despeckle_strength=despeckle_strength,
        )
    detector_backends = _resolve_detector_policy(detector_policy)
    stage_cache = (
        ProcessingStageCache(
            Path(stage_cache_dir),
            max_bytes=int(stage_cache_max_mb) * 1024 * 1024,
            max_entries=256,
        )
        if stage_cache_dir is not None
        else None
    )
    pre_split_rotation = (
        int(orientation_method)
        if two_page_mode and orientation_method in {"90", "180", "270"}
        else 0
    )
    processing_orientation_method = "none" if pre_split_rotation else orientation_method
    options = PipelineOptions(
        detect_document=bool(detect_document),
        two_page_mode=bool(two_page_mode),
        postprocess_name="None",
        detector_backends=detector_backends,
        strict_detect=bool(strict_detect),
        pre_split_rotation_degrees=pre_split_rotation,
    )

    staged_targets: list[_StagedTarget] = []
    page_reports: list[PageRunReport] = []
    final_image_paths: tuple[Path, ...] = ()
    with tempfile.TemporaryDirectory(prefix="uniscan_pages_") as tmp:
        page_stage_dir = Path(tmp)
        staged_page_paths: list[Path] = []

        for loaded_item in iter_input_items(
            input_files,
            pdf_dpi=input_dpi,
            max_input_pixels=max_input_pixels,
            on_progress=on_progress,
            cancel_cb=cancel_cb,
        ):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Cancelled by user.")
            detection_started = time.perf_counter()
            page_results = process_loaded_items(
                [loaded_item],
                options=options,
                uvdoc_cache_home=uvdoc_cache_home,
                cancel_cb=cancel_cb,
            )
            detection_share_ms = (
                (time.perf_counter() - detection_started) * 1000.0 / max(1, len(page_results))
            )
            for page in page_results:
                page_started = time.perf_counter()
                processed = process_document_page(
                    page.current,
                    PageProcessingRequest(
                        orientation_method=processing_orientation_method,
                        deskew_method=deskew_method,
                        deskew_angle_degrees=deskew_angle_degrees,
                        dewarp_method=dewarp_method,
                        uvdoc_cache_home=uvdoc_cache_home,
                        auto_dewarp_uvdoc=auto_dewarp_uvdoc,
                        auto_dewarp_uvdoc_grid=auto_dewarp_uvdoc_grid,
                        shadow_method=shadow_method,
                        postprocess_name=postprocess_name,
                        preprocess_settings=preprocess_settings,
                        page_layout=page_layout,
                        page_dpi=output_dpi,
                        page_margin_mm=page_margin_mm,
                        horizontal_alignment=horizontal_alignment,
                        vertical_alignment=vertical_alignment,
                        lighting_diagnostics=lighting_diagnostics,
                        stage_cache=stage_cache,
                        cancel_cb=cancel_cb,
                    ),
                )
                current = processed.image
                processing_diagnostics = processed.diagnostics
                orientation_diagnostics = processing_diagnostics.orientation
                if pre_split_rotation:
                    orientation_diagnostics = OrientationDiagnostics(
                        method=orientation_method,
                        applied=True,
                        angle_degrees=pre_split_rotation,
                        confidence=1.0,
                        reason="forced_before_spread_detection",
                    )
                deskew_angle = processing_diagnostics.deskew_angle_degrees
                dewarp_diagnostics = processing_diagnostics.dewarp
                shadow_diagnostics = processing_diagnostics.shadow
                despeckle_diagnostics = processing_diagnostics.despeckle
                layout_diagnostics = processing_diagnostics.layout
                lighting = processing_diagnostics.lighting
                page_path = page_stage_dir / f"{len(staged_page_paths) + 1:05d}.png"
                if not imwrite_unicode(page_path, current):
                    raise RuntimeError(f"Failed to write processed page: {page_path}")
                if cancel_cb is not None and cancel_cb():
                    raise RuntimeError("Cancelled by user.")
                staged_page_paths.append(page_path)
                page_reports.append(
                    PageRunReport(
                        index=len(staged_page_paths),
                        name=page.name,
                        detected=page.detected,
                        backend=page.backend,
                        fallback_reason=page.fallback_reason,
                        duration_ms=round(
                            detection_share_ms + (time.perf_counter() - page_started) * 1000.0,
                            3,
                        ),
                        orientation_method=orientation_method,
                        orientation_applied=orientation_diagnostics.applied,
                        orientation_angle_degrees=orientation_diagnostics.angle_degrees,
                        orientation_confidence=orientation_diagnostics.confidence,
                        orientation_reason=orientation_diagnostics.reason,
                        deskew_method=deskew_method,
                        deskew_angle_degrees=round(float(deskew_angle), 3),
                        deskew_selected_method=processing_diagnostics.deskew_selected_method,
                        deskew_confidence=processing_diagnostics.deskew_confidence,
                        deskew_line_count=processing_diagnostics.deskew_line_count,
                        deskew_reason=processing_diagnostics.deskew_reason,
                        dewarp_method=dewarp_method,
                        dewarp_applied=dewarp_diagnostics.applied,
                        dewarp_selected_method=dewarp_diagnostics.selected_method,
                        dewarp_line_count=dewarp_diagnostics.line_count,
                        dewarp_max_displacement_px=dewarp_diagnostics.max_displacement_px,
                        dewarp_curvature_before_px=dewarp_diagnostics.curvature_before_px,
                        dewarp_curvature_after_px=dewarp_diagnostics.curvature_after_px,
                        dewarp_perspective_before=dewarp_diagnostics.perspective_before,
                        dewarp_perspective_after=dewarp_diagnostics.perspective_after,
                        dewarp_blank_border_before=dewarp_diagnostics.blank_border_before,
                        dewarp_blank_border_after=dewarp_diagnostics.blank_border_after,
                        dewarp_edge_ink_before=dewarp_diagnostics.edge_ink_before,
                        dewarp_edge_ink_after=dewarp_diagnostics.edge_ink_after,
                        dewarp_aspect_change=dewarp_diagnostics.aspect_change,
                        dewarp_duration_ms=dewarp_diagnostics.duration_ms,
                        dewarp_reason=dewarp_diagnostics.reason,
                        shadow_method=shadow_diagnostics.method,
                        shadow_applied=shadow_diagnostics.applied,
                        shadow_selected_method=shadow_diagnostics.selected_method,
                        shadow_unevenness_before=shadow_diagnostics.unevenness_before,
                        shadow_unevenness_after=shadow_diagnostics.unevenness_after,
                        shadow_before=shadow_diagnostics.shadow_before,
                        shadow_after=shadow_diagnostics.shadow_after,
                        shadow_glare_after=shadow_diagnostics.glare_after,
                        shadow_duration_ms=shadow_diagnostics.duration_ms,
                        shadow_reason=shadow_diagnostics.reason,
                        page_layout=page_layout,
                        layout_applied=layout_diagnostics.applied,
                        content_box=(
                            layout_diagnostics.content_box.x,
                            layout_diagnostics.content_box.y,
                            layout_diagnostics.content_box.width,
                            layout_diagnostics.content_box.height,
                        ),
                        content_confidence=layout_diagnostics.content_confidence,
                        layout_scale=layout_diagnostics.scale,
                        layout_reason=layout_diagnostics.reason,
                        binarization_method=binarization_method,
                        despeckle_strength=despeckle_strength,
                        despeckle_removed_components=despeckle_diagnostics.removed_components,
                        despeckle_removed_pixels=despeckle_diagnostics.removed_pixels,
                        despeckle_protected_components=despeckle_diagnostics.protected_components,
                        shadow_fraction=(
                            lighting.shadow_fraction if lighting is not None else None
                        ),
                        glare_fraction=(lighting.glare_fraction if lighting is not None else None),
                        clipped_pixel_fraction=(
                            lighting.clipped_pixel_fraction if lighting is not None else None
                        ),
                        lighting_unevenness=(lighting.unevenness if lighting is not None else None),
                        lighting_warnings=(lighting.warnings if lighting is not None else ()),
                        processing_stage_durations_ms=processing_diagnostics.stage_durations_ms,
                        processing_cache_hits=processing_diagnostics.cache_hits,
                        spread_detected=page.spread_detected,
                        spread_confidence=page.spread_confidence,
                        spread_reason=page.spread_reason,
                        needs_review=page.needs_review,
                        review_reasons=page.review_reasons,
                        boundary_dark_border_fraction=page.boundary_dark_border_fraction,
                    )
                )

        if not staged_page_paths:
            raise ValueError("The input did not produce any pages.")

        report_payload = _report_payload(
            output_pdf=output_pdf,
            report_path=report_path,
            images_dir=images_dir,
            image_outputs=(),
            image_format=image_format,
            input_pdf_dpi=input_dpi,
            output_pdf_dpi=output_dpi,
            pdf_jpeg_quality=pdf_jpeg_quality,
            max_input_pixels=max_input_pixels,
            input_files=input_files,
            pages=page_reports,
            detect_document=bool(detect_document),
            detector_policy="disabled" if not detect_document else detector_policy,
            detector_backends=detector_backends if detect_document else (),
            strict_detect=bool(strict_detect),
            two_page_mode=bool(two_page_mode),
            lens_mode=lens_mode,
            preprocess_preset=preprocess_preset,
            postprocess_name=postprocess_name,
            preprocess_settings=preprocess_settings,
            illumination_correction=shadow_method != SHADOW_METHOD_NONE,
            legacy_illumination_correction=legacy_illumination_correction,
            orientation_method=orientation_method,
            deskew_method=deskew_method,
            deskew_angle_degrees=deskew_angle_degrees,
            dewarp_method=dewarp_method,
            auto_dewarp_uvdoc=bool(auto_dewarp_uvdoc),
            auto_dewarp_uvdoc_grid=bool(auto_dewarp_uvdoc_grid),
            shadow_method=shadow_method,
            page_layout=page_layout,
            page_margin_mm=float(page_margin_mm),
            horizontal_alignment=horizontal_alignment,
            vertical_alignment=vertical_alignment,
            binarization_method=binarization_method,
            binarization_window=int(binarization_window),
            binarization_k=(float(binarization_k) if binarization_k is not None else None),
            despeckle_strength=despeckle_strength,
            lighting_diagnostics=bool(lighting_diagnostics),
            stage_cache_enabled=stage_cache is not None,
            stage_cache_dir=Path(stage_cache_dir) if stage_cache_dir is not None else None,
            stage_cache_max_mb=int(stage_cache_max_mb),
            stage_cache_stats=(
                {
                    "hits": stage_cache.stats.hits,
                    "misses": stage_cache.stats.misses,
                    "writes": stage_cache.stats.writes,
                    "evictions": stage_cache.stats.evictions,
                }
                if stage_cache is not None
                else {"hits": 0, "misses": 0, "writes": 0, "evictions": 0}
            ),
            uvdoc_cache_home=(Path(uvdoc_cache_home) if uvdoc_cache_home is not None else None),
        )
        with _transaction_output_locks(
            transaction_journal,
            expected_targets=transaction_targets,
        ):
            _recover_batch_transaction(
                transaction_journal,
                expected_targets=transaction_targets,
                locks_held=True,
            )
            try:
                staged_targets, final_image_paths = _stage_outputs(
                    staged_page_paths=staged_page_paths,
                    output_pdf=output_pdf,
                    images_dir=images_dir,
                    image_format=image_format,
                    report_path=report_path,
                    report_payload=report_payload,
                    dpi=output_dpi,
                    pdf_jpeg_quality=pdf_jpeg_quality,
                    cancel_cb=cancel_cb,
                )
                if cancel_cb is not None and cancel_cb():
                    raise RuntimeError("Cancelled by user.")
                _publish_staged_targets_locked(
                    staged_targets,
                    journal_path=transaction_journal,
                )
            except Exception:
                for item in staged_targets:
                    _cleanup_path_best_effort(item.staged)
                raise

    detected_pages = sum(page.detected for page in page_reports)
    fallback_pages = sum(page.fallback_reason is not None for page in page_reports)
    review_pages = sum(page.needs_review for page in page_reports)
    return BatchPipelineResult(
        output_pdf=output_pdf,
        report_path=report_path,
        input_files=input_files,
        image_outputs=final_image_paths,
        total_pages=len(page_reports),
        detected_pages=detected_pages,
        fallback_pages=fallback_pages,
        review_pages=review_pages,
        pages=tuple(page_reports),
    )
