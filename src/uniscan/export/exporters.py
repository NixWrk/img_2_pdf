"""Export page images to PDF and separate image files."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from uniscan.core.pipeline import build_pdf_from_images
from uniscan.io.loaders import imwrite_unicode

CancelCb = Callable[[], bool]
_EXPORT_MANIFEST_NAME = ".uniscan-export-manifest.json"
_EXPORT_MANIFEST_OWNER = "uniscan.page-export"
_EXPORT_MANIFEST_VERSION = 1
_DIRECTORY_TRANSACTION_OWNER = "uniscan.directory-export"
_DIRECTORY_TRANSACTION_VERSION = 1
_DIRECTORY_TRANSACTION_SUFFIX = ".uniscan-directory-transaction.json"


def _check_cancelled(cancel_cb: CancelCb | None) -> None:
    if cancel_cb is not None and cancel_cb():
        raise RuntimeError("Cancelled by user.")


def _is_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or _is_junction(path)


def _remove_path(path: Path) -> None:
    if _is_junction(path):
        path.rmdir()
    elif path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _cleanup_path_best_effort(path: Path) -> bool:
    """Remove post-commit debris without changing a successful result into failure."""
    try:
        if path.exists() or _is_link_like(path):
            _remove_path(path)
    except OSError:
        # A virus scanner or Explorer preview can temporarily hold a file on
        # Windows.  The live output is already committed; a later export will
        # retry cleanup of the recoverable backup.
        return False
    return not (path.exists() or _is_link_like(path))


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _path_key(path: Path) -> str:
    return os.path.normcase(str(_absolute_path(path)))


def _directory_transaction_path(output_dir: Path) -> Path:
    output_dir = _absolute_path(output_dir)
    return output_dir.parent / f".{output_dir.name}{_DIRECTORY_TRANSACTION_SUFFIX}"


def _directory_export_lock_path(output_dir: Path) -> Path:
    journal = _directory_transaction_path(output_dir)
    return journal.with_name(f"{journal.name}.lock")


@contextmanager
def _directory_export_lock(output_dir: Path):
    """Serialize recovery, staging, and publication for one image directory."""
    lock_path = _directory_export_lock_path(output_dir)
    if _is_link_like(lock_path) or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError(f"Invalid image-directory export lock path: {lock_path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised by Linux CI
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(
                "Another UniScan process is exporting to this image directory."
            ) from exc
        yield
    finally:
        if locked:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised by Linux CI
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        stream.close()


def _write_directory_transaction(path: Path, payload: dict[str, object]) -> None:
    if _is_link_like(path) or (path.exists() and not path.is_file()):
        raise ValueError(f"Invalid directory-export transaction path: {path}")
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
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _cleanup_path_best_effort(staged)
        raise


def _load_directory_transaction(path: Path, *, output_dir: Path) -> dict[str, object]:
    if _is_link_like(path) or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid UniScan directory-export transaction journal.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UniScan directory-export transaction journal.") from exc
    expected_keys = {
        "schemaVersion",
        "owner",
        "transactionId",
        "state",
        "target",
        "staged",
        "backup",
        "hadTarget",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Invalid UniScan directory-export transaction schema.")
    transaction_id = payload.get("transactionId")
    if (
        type(payload.get("schemaVersion")) is not int
        or payload["schemaVersion"] != _DIRECTORY_TRANSACTION_VERSION
        or payload.get("owner") != _DIRECTORY_TRANSACTION_OWNER
        or not isinstance(transaction_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
        or payload.get("state") not in {"prepared", "committed"}
        or type(payload.get("hadTarget")) is not bool
        or not all(isinstance(payload.get(key), str) for key in ("target", "staged", "backup"))
    ):
        raise ValueError("Invalid UniScan directory-export transaction schema.")

    target = Path(payload["target"])
    staged = Path(payload["staged"])
    backup = Path(payload["backup"])
    output_dir = _absolute_path(output_dir)
    if not all(item.is_absolute() for item in (target, staged, backup)):
        raise ValueError("Invalid directory-export transaction paths.")
    if _path_key(target) != _path_key(output_dir):
        raise ValueError("Directory-export transaction target does not match this output.")
    if _path_key(staged.parent) != _path_key(output_dir.parent) or not staged.name.startswith(
        f".{output_dir.name}.stage-"
    ):
        raise ValueError("Invalid directory-export staged path.")
    if (
        _path_key(backup.parent) != _path_key(output_dir.parent)
        or backup.name != f".{output_dir.name}.backup-{transaction_id}"
    ):
        raise ValueError("Invalid directory-export backup path.")
    if len({_path_key(target), _path_key(staged), _path_key(backup)}) != 3:
        raise ValueError("Invalid directory-export transaction paths.")
    for candidate in (target, staged, backup):
        if (candidate.exists() or _is_link_like(candidate)) and (
            _is_link_like(candidate) or not candidate.is_dir()
        ):
            raise ValueError(f"Invalid directory-export transaction directory: {candidate}")

    target_exists = target.exists()
    staged_exists = staged.exists()
    backup_exists = backup.exists()
    if payload["state"] == "committed":
        if not target_exists or staged_exists:
            raise ValueError("Invalid committed directory-export transaction state.")
    elif payload["hadTarget"]:
        if not target_exists and not backup_exists:
            raise ValueError("Invalid prepared directory-export transaction state.")
        if target_exists and staged_exists and backup_exists:
            raise ValueError("Ambiguous prepared directory-export transaction state.")
    elif backup_exists or (target_exists and staged_exists):
        raise ValueError("Invalid prepared directory-export transaction state.")
    return payload


def _rollback_directory_transaction(payload: dict[str, object]) -> bool:
    target = Path(payload["target"])
    staged = Path(payload["staged"])
    backup = Path(payload["backup"])
    if backup.exists():
        if target.exists():
            _remove_path(target)
        os.replace(backup, target)
    elif not payload["hadTarget"] and target.exists():
        _remove_path(target)
    return _cleanup_path_best_effort(staged) and not backup.exists()


def _recover_directory_backup(output_dir: Path) -> None:
    """Recover only the exact backup named by a validated durable journal."""
    journal = _directory_transaction_path(output_dir)
    if not (journal.exists() or _is_link_like(journal)):
        return
    payload = _load_directory_transaction(journal, output_dir=output_dir)
    if payload["state"] == "prepared":
        cleanup_complete = _rollback_directory_transaction(payload)
    else:
        cleanup_complete = _cleanup_path_best_effort(Path(payload["backup"]))
        cleanup_complete = _cleanup_path_best_effort(Path(payload["staged"])) and cleanup_complete
    if not cleanup_complete:
        raise RuntimeError(
            "A prior image-directory export is recovered/committed, but its exact backup "
            "is locked; close programs using it and retry."
        )
    if not _cleanup_path_best_effort(journal):
        raise RuntimeError("Recovered image-directory transaction journal is locked.")


def _replace_directory_atomically(
    staged_dir: Path,
    output_dir: Path,
    *,
    cancel_cb: CancelCb | None,
) -> None:
    """Replace one directory through a durable, exact-path transaction journal."""
    transaction_id = uuid.uuid4().hex
    backup = output_dir.with_name(f".{output_dir.name}.backup-{transaction_id}")
    journal = _directory_transaction_path(output_dir)
    if journal.exists() or _is_link_like(journal):
        raise ValueError(f"Unrecovered image-directory transaction exists: {journal}")
    if backup.exists() or _is_link_like(backup):
        raise ValueError(f"Image-directory transaction backup already exists: {backup}")
    payload: dict[str, object] = {
        "schemaVersion": _DIRECTORY_TRANSACTION_VERSION,
        "owner": _DIRECTORY_TRANSACTION_OWNER,
        "transactionId": transaction_id,
        "state": "prepared",
        "target": str(_absolute_path(output_dir)),
        "staged": str(_absolute_path(staged_dir)),
        "backup": str(_absolute_path(backup)),
        "hadTarget": output_dir.exists() or _is_link_like(output_dir),
    }
    try:
        _check_cancelled(cancel_cb)
        _write_directory_transaction(journal, payload)
        if payload["hadTarget"]:
            os.replace(output_dir, backup)
        _check_cancelled(cancel_cb)
        os.replace(staged_dir, output_dir)
        _check_cancelled(cancel_cb)
        payload["state"] = "committed"
        _write_directory_transaction(journal, payload)
    except Exception:
        if journal.exists():
            rollback_clean = _rollback_directory_transaction(payload)
            if rollback_clean:
                _cleanup_path_best_effort(journal)
        raise

    cleanup_complete = _cleanup_path_best_effort(backup)
    cleanup_complete = _cleanup_path_best_effort(staged_dir) and cleanup_complete
    if cleanup_complete:
        _cleanup_path_best_effort(journal)


def _validate_base_name(base_name: str) -> None:
    if not base_name or base_name in {".", ".."} or Path(base_name).name != base_name:
        raise ValueError("Image base_name must be one plain file-name component.")


def _validate_owned_file_name(name: str, *, base_name: str, index: int) -> None:
    prefix = f"{base_name}_{index:05d}."
    if (
        not name.startswith(prefix)
        or Path(name).name != name
        or re.fullmatch(r"[a-z0-9]+", name[len(prefix) :]) is None
    ):
        raise ValueError(f"Invalid UniScan image-export manifest entry: {name!r}")


def _load_owned_files(output_dir: Path) -> tuple[str | None, tuple[str, ...]]:
    manifest_path = output_dir / _EXPORT_MANIFEST_NAME
    if not (manifest_path.exists() or _is_link_like(manifest_path)):
        return None, ()
    if _is_link_like(manifest_path) or not manifest_path.is_file():
        raise ValueError("Invalid UniScan image-export manifest path.")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid UniScan image-export manifest: file is too large.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UniScan image-export manifest.") from exc
    expected_keys = {"schemaVersion", "owner", "baseName", "files"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Invalid UniScan image-export manifest schema.")
    if (
        type(payload["schemaVersion"]) is not int
        or payload["schemaVersion"] != _EXPORT_MANIFEST_VERSION
        or payload["owner"] != _EXPORT_MANIFEST_OWNER
        or not isinstance(payload["baseName"], str)
        or not isinstance(payload["files"], list)
        or not all(isinstance(name, str) for name in payload["files"])
    ):
        raise ValueError("Invalid UniScan image-export manifest schema.")

    stored_base_name = payload["baseName"]
    _validate_base_name(stored_base_name)
    owned_files = tuple(payload["files"])
    if len(set(owned_files)) != len(owned_files):
        raise ValueError("Invalid UniScan image-export manifest: duplicate entries.")
    for index, name in enumerate(owned_files, start=1):
        _validate_owned_file_name(name, base_name=stored_base_name, index=index)
        owned_path = output_dir / name
        if (owned_path.exists() or _is_link_like(owned_path)) and (
            _is_link_like(owned_path) or not owned_path.is_file()
        ):
            raise ValueError(f"Invalid owned image-export path: {owned_path}")
    return stored_base_name, owned_files


def _write_owned_files_manifest(
    staged_dir: Path,
    *,
    base_name: str,
    files: Sequence[str],
) -> None:
    manifest_path = staged_dir / _EXPORT_MANIFEST_NAME
    payload = {
        "schemaVersion": _EXPORT_MANIFEST_VERSION,
        "owner": _EXPORT_MANIFEST_OWNER,
        "baseName": base_name,
        "files": list(files),
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_staged_directory(
    output_dir: Path,
    *,
    desired_names: Sequence[str],
) -> tuple[Path, Path]:
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir == output_dir.parent:
        raise ValueError("The filesystem root cannot be used as an images output directory.")
    _recover_directory_backup(output_dir)
    if _is_link_like(output_dir):
        raise ValueError(f"Images output path cannot be a link or junction: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Images output path exists and is not a directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        if output_dir.exists():
            shutil.copytree(output_dir, staged_dir, dirs_exist_ok=True, symlinks=True)
        _stored_base_name, owned_files = _load_owned_files(staged_dir)
        owned_set = set(owned_files)
        for name in desired_names:
            candidate = staged_dir / name
            if (candidate.exists() or _is_link_like(candidate)) and name not in owned_set:
                raise ValueError(
                    f"Refusing to overwrite unowned image-export collision: {candidate}"
                )
        for name in owned_files:
            owned_path = staged_dir / name
            if owned_path.exists() or _is_link_like(owned_path):
                _remove_path(owned_path)
        manifest_path = staged_dir / _EXPORT_MANIFEST_NAME
        if manifest_path.exists():
            manifest_path.unlink()
        return output_dir, staged_dir
    except Exception:
        _remove_path(staged_dir)
        raise


def _normalize_extension(ext: str) -> str:
    normalized = ext.lower().lstrip(".") or "png"
    if re.fullmatch(r"[a-z0-9]+", normalized) is None:
        raise ValueError("Image extension must contain only letters and digits.")
    return normalized


def export_pages_as_pdf(
    pages: Sequence[np.ndarray],
    *,
    out_pdf: Path,
    dpi: int = 300,
    cancel_cb: CancelCb | None = None,
) -> Path:
    """Export pages to one merged PDF."""
    if len(pages) == 0:
        raise ValueError("No pages to export.")

    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="uniscan_pdf_") as tmp:
        tmp_dir = Path(tmp)
        image_paths: list[Path] = []
        for idx, page in enumerate(pages, start=1):
            _check_cancelled(cancel_cb)
            page_path = tmp_dir / f"{idx:05d}.png"
            if not imwrite_unicode(page_path, page):
                raise RuntimeError(f"Failed to write temporary page image: {page_path}")
            image_paths.append(page_path)
        _check_cancelled(cancel_cb)
        build_pdf_from_images(
            image_paths,
            out_pdf=out_pdf,
            dpi=int(dpi),
            cancel_cb=cancel_cb,
        )
    return out_pdf


def export_pages_as_files(
    pages: Sequence[np.ndarray],
    *,
    output_dir: Path,
    ext: str = "png",
    base_name: str = "page",
    cancel_cb: CancelCb | None = None,
) -> list[Path]:
    """Atomically replace a directory with separately exported page images."""
    if len(pages) == 0:
        raise ValueError("No pages to export.")

    ext = _normalize_extension(ext)
    _validate_base_name(base_name)
    output_names = tuple(f"{base_name}_{idx:05d}.{ext}" for idx in range(1, len(pages) + 1))
    output_dir = _absolute_path(Path(output_dir))
    if output_dir == output_dir.parent:
        raise ValueError("The filesystem root cannot be used as an images output directory.")
    with _directory_export_lock(output_dir):
        output_dir, staged_dir = _new_staged_directory(
            output_dir,
            desired_names=output_names,
        )

        output_paths: list[Path] = []
        try:
            for page, output_name in zip(pages, output_names, strict=True):
                _check_cancelled(cancel_cb)
                staged_path = staged_dir / output_name
                if not imwrite_unicode(staged_path, page):
                    raise RuntimeError(f"Failed to write page image: {staged_path}")
                output_paths.append(output_dir / staged_path.name)
            _write_owned_files_manifest(staged_dir, base_name=base_name, files=output_names)
            _replace_directory_atomically(staged_dir, output_dir, cancel_cb=cancel_cb)
            return output_paths
        except Exception:
            if staged_dir.exists():
                _remove_path(staged_dir)
            raise


def export_image_paths_as_pdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    dpi: int = 300,
    cancel_cb: CancelCb | None = None,
) -> Path:
    """Export image file paths to merged PDF without loading all images in memory."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to export.")
    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf_from_images(
        [Path(p) for p in image_paths],
        out_pdf=out_pdf,
        dpi=int(dpi),
        cancel_cb=cancel_cb,
    )
    return out_pdf


def export_image_paths_as_files(
    image_paths: Sequence[Path],
    *,
    output_dir: Path,
    ext: str = "png",
    base_name: str = "page",
    cancel_cb: CancelCb | None = None,
) -> list[Path]:
    """Atomically replace a directory with converted copies of image paths."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to export.")
    ext = _normalize_extension(ext)
    _validate_base_name(base_name)
    output_names = tuple(f"{base_name}_{idx:05d}.{ext}" for idx in range(1, len(image_paths) + 1))
    output_dir = _absolute_path(Path(output_dir))
    if output_dir == output_dir.parent:
        raise ValueError("The filesystem root cannot be used as an images output directory.")
    with _directory_export_lock(output_dir):
        output_dir, staged_dir = _new_staged_directory(
            output_dir,
            desired_names=output_names,
        )

        out_paths: list[Path] = []
        try:
            for src, output_name in zip(image_paths, output_names, strict=True):
                _check_cancelled(cancel_cb)
                staged_path = staged_dir / output_name
                src_path = Path(src)
                if src_path.suffix.lower().lstrip(".") == ext:
                    shutil.copy2(src_path, staged_path)
                else:
                    data = np.fromfile(str(src_path), dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise RuntimeError(f"Cannot read source image: {src_path}")
                    if not imwrite_unicode(staged_path, image):
                        raise RuntimeError(f"Failed to write page image: {staged_path}")
                out_paths.append(output_dir / staged_path.name)
            _write_owned_files_manifest(staged_dir, base_name=base_name, files=output_names)
            _replace_directory_atomically(staged_dir, output_dir, cancel_cb=cancel_cb)
            return out_paths
        except Exception:
            if staged_dir.exists():
                _remove_path(staged_dir)
            raise
