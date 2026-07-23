"""Pinned model-asset metadata, verification, and atomic downloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen


MODEL_DIR = Path(__file__).resolve().parent / "models"
MANIFEST_PATH = MODEL_DIR / "manifest.json"
_CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True, frozen=True)
class ModelAsset:
    name: str
    filename: str
    size: int
    sha256: str
    license: str
    source: str
    url: str | None = None


@lru_cache(maxsize=1)
def model_assets() -> dict[str, ModelAsset]:
    """Load and strictly validate the repository's pinned model manifest."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("assets"), dict):
        raise RuntimeError("Unsupported model asset manifest.")
    result: dict[str, ModelAsset] = {}
    for name, raw in payload["assets"].items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise RuntimeError("Invalid model asset manifest entry.")
        asset = ModelAsset(name=name, **raw)
        if (
            not asset.filename
            or Path(asset.filename).name != asset.filename
            or type(asset.size) is not int
            or asset.size <= 0
            or len(asset.sha256) != 64
            or any(character not in "0123456789abcdef" for character in asset.sha256)
        ):
            raise RuntimeError(f"Invalid model asset manifest entry: {name}")
        result[name] = asset
    return result


def model_asset(name: str) -> ModelAsset:
    try:
        return model_assets()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model asset: {name}") from exc


@lru_cache(maxsize=32)
def _sha256_for_stat(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_for_stat(str(resolved), stat.st_size, stat.st_mtime_ns)


def verify_model_asset(name: str, path: Path | None = None) -> Path:
    """Return the asset path only if its exact size and SHA-256 match."""
    asset = model_asset(name)
    candidate = Path(path) if path is not None else MODEL_DIR / asset.filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Model asset is missing: {candidate}")
    actual_size = candidate.stat().st_size
    if actual_size != asset.size:
        raise RuntimeError(
            f"Model asset size mismatch for {candidate}: expected {asset.size}, got {actual_size}."
        )
    actual_hash = file_sha256(candidate)
    if actual_hash != asset.sha256:
        raise RuntimeError(
            f"Model asset SHA-256 mismatch for {candidate}: "
            f"expected {asset.sha256}, got {actual_hash}."
        )
    return candidate


def model_file_identity(path: Path) -> str:
    """Stable cache identity for a configured model, including missing files."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return f"missing:{candidate.resolve()}"
    return f"sha256:{file_sha256(candidate)}:{candidate.stat().st_size}"


def download_model_asset(name: str, target_dir: Path = MODEL_DIR) -> Path:
    """Download a pinned release asset and publish it only after verification."""
    asset = model_asset(name)
    if asset.url is None:
        raise ValueError(f"Model asset has no downloadable release URL: {name}")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / asset.filename
    if destination.is_file():
        try:
            return verify_model_asset(name, destination)
        except RuntimeError:
            pass

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{asset.filename}.",
            suffix=".part",
            dir=target_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(asset.url, headers={"User-Agent": "UniScan-model-assets/1"})
            digest = hashlib.sha256()
            downloaded = 0
            with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS URL
                while chunk := response.read(_CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > asset.size:
                        raise RuntimeError(f"Model asset is larger than declared: {name}")
                    digest.update(chunk)
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if downloaded != asset.size:
            raise RuntimeError(
                f"Model asset size mismatch for {name}: expected {asset.size}, got {downloaded}."
            )
        actual_hash = digest.hexdigest()
        if actual_hash != asset.sha256:
            raise RuntimeError(
                f"Model asset SHA-256 mismatch for {name}: "
                f"expected {asset.sha256}, got {actual_hash}."
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return verify_model_asset(name, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "MANIFEST_PATH",
    "MODEL_DIR",
    "ModelAsset",
    "download_model_asset",
    "file_sha256",
    "model_asset",
    "model_assets",
    "model_file_identity",
    "verify_model_asset",
]
