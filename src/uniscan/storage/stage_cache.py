"""Atomic bounded disk cache for deterministic document-processing stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from uuid import uuid4

import cv2
import numpy as np


@dataclass(slots=True)
class StageCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0


class ProcessingStageCache:
    """Store lossless stage images and JSON metadata with LRU-style bounds."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root_dir: Path,
        *,
        max_bytes: int = 512 * 1024 * 1024,
        max_entries: int = 256,
    ) -> None:
        if int(max_bytes) < 1024 * 1024:
            raise ValueError("Stage cache max_bytes must be at least 1 MiB.")
        if int(max_entries) < 1:
            raise ValueError("Stage cache max_entries must be positive.")
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.stats = StageCacheStats()
        self._lock = threading.RLock()

    @staticmethod
    def fingerprint_image(image: np.ndarray) -> str:
        """Hash pixels plus shape/dtype so equal sources share downstream stage keys."""
        contiguous = np.ascontiguousarray(image)
        digest = hashlib.sha256()
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(memoryview(contiguous))
        return digest.hexdigest()

    @staticmethod
    def stage_key(upstream_key: str, stage: str, options: dict[str, object]) -> str:
        encoded = json.dumps(
            {
                "schema": ProcessingStageCache.SCHEMA_VERSION,
                "upstream": upstream_key,
                "stage": stage,
                "options": options,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Invalid stage cache key.")
        return self.root_dir / f"{key}.png", self.root_dir / f"{key}.json"

    def get(self, key: str) -> tuple[np.ndarray, dict[str, object]] | None:
        image_path, metadata_path = self._paths(key)
        with self._lock:
            if not image_path.is_file() or not metadata_path.is_file():
                self.stats.misses += 1
                return None
            try:
                data = np.fromfile(str(image_path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    image is None
                    or not isinstance(metadata, dict)
                    or metadata.get("schemaVersion") != self.SCHEMA_VERSION
                ):
                    raise ValueError("invalid cache entry")
                payload = metadata.get("metadata")
                if not isinstance(payload, dict):
                    raise ValueError("invalid cache metadata")
                os.utime(image_path, None)
                os.utime(metadata_path, None)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                image_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return image, payload

    def put(self, key: str, image: np.ndarray, metadata: dict[str, object]) -> bool:
        if image.dtype != np.uint8 or image.ndim not in {2, 3}:
            return False
        image_path, metadata_path = self._paths(key)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return False
        token = uuid4().hex
        temporary_image = self.root_dir / f".{key}.{token}.png.tmp"
        temporary_metadata = self.root_dir / f".{key}.{token}.json.tmp"
        payload = {
            "schemaVersion": self.SCHEMA_VERSION,
            "metadata": metadata,
        }
        with self._lock:
            try:
                temporary_image.write_bytes(encoded.tobytes())
                temporary_metadata.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                os.replace(temporary_image, image_path)
                os.replace(temporary_metadata, metadata_path)
            except (OSError, TypeError, ValueError):
                return False
            finally:
                temporary_image.unlink(missing_ok=True)
                temporary_metadata.unlink(missing_ok=True)
            self.stats.writes += 1
            self._prune_locked()
        return True

    def _prune_locked(self) -> None:
        entries: list[tuple[float, Path, Path, int]] = []
        for image_path in self.root_dir.glob("*.png"):
            metadata_path = image_path.with_suffix(".json")
            try:
                size = image_path.stat().st_size
                modified = image_path.stat().st_mtime
                if metadata_path.exists():
                    size += metadata_path.stat().st_size
            except OSError:
                continue
            entries.append((modified, image_path, metadata_path, size))
        entries.sort(key=lambda item: item[0])
        total_bytes = sum(item[3] for item in entries)
        while entries and (len(entries) > self.max_entries or total_bytes > self.max_bytes):
            _modified, image_path, metadata_path, size = entries.pop(0)
            try:
                image_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            except OSError:
                continue
            total_bytes -= size
            self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            for pattern in ("*.png", "*.json", ".*.tmp"):
                for path in self.root_dir.glob(pattern):
                    path.unlink(missing_ok=True)
