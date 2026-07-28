"""Fail a Windows portable build containing unsafe, stale, or unrelated payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = (
    "uniscan.exe",
    "LICENSE.txt",
    "README.md",
    "CHANGELOG.md",
    "THIRD_PARTY_LICENSES/INDEX.txt",
    "THIRD_PARTY_LICENSES/FROZEN_PAYLOAD.txt",
    "THIRD_PARTY_LICENSES/RUNTIME/PYTHON-PSF-LICENSE.txt",
    "THIRD_PARTY_LICENSES/RUNTIME/TCL-LICENSE.txt",
    "THIRD_PARTY_LICENSES/RUNTIME/TK-LICENSE.txt",
    "docs/manual_smoke_checklist.md",
    "docs/windows_release.md",
)
FORBIDDEN_NAMES = {".ds_store"}
FORBIDDEN_SUFFIXES = {
    ".lib",
    ".ort",
    ".pdmodel",
    ".pdparams",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
}
FORBIDDEN_MODEL_SUFFIXES = (
    ".onnx",
    ".onnx.data",
    ".ort",
    ".pdmodel",
    ".pdparams",
    ".pt",
    ".pth",
    ".safetensors",
)
ALLOWED_TKDND_PLATFORMS = {"win-x64", "win-x64-tcl9"}
RUNTIME_NOTICE_MARKERS = {
    "THIRD_PARTY_LICENSES/RUNTIME/PYTHON-PSF-LICENSE.txt": "Python Software Foundation",
    "THIRD_PARTY_LICENSES/RUNTIME/TCL-LICENSE.txt": "This software is copyrighted by",
    "THIRD_PARTY_LICENSES/RUNTIME/TK-LICENSE.txt": "This software is copyrighted by",
}
ROBOTO_ASSET_PREFIX = "customtkinter/assets/fonts/roboto/"
ROBOTO_ASSET_DESTINATIONS = frozenset(
    {
        f"{ROBOTO_ASSET_PREFIX}roboto-medium.ttf",
        f"{ROBOTO_ASSET_PREFIX}roboto-regular.ttf",
    }
)
ROBOTO_NOTICE_PATH = "THIRD_PARTY_LICENSES/ASSETS/Roboto-Apache-2.0.txt"
ROBOTO_NOTICE_MARKERS = (
    "Roboto font files",
    "Copyright 2011 Google Inc.",
    "Apache License",
    "Version 2.0",
)
MODEL_NOTICE_PATHS = {
    "uvdoc": "THIRD_PARTY_LICENSES/ASSETS/UVDoc-ONNX-Apache-2.0.txt",
    "docshadow": "THIRD_PARTY_LICENSES/ASSETS/DocShadow-MIT.txt",
}
MODEL_NOTICE_MARKERS = {
    "uvdoc": ("Apache License", "Version 2.0"),
    "docshadow": ("MIT License", "Copyright (c) 2023"),
}


def _approved_model_assets() -> dict[str, tuple[int, str]]:
    manifest = json.loads((ROOT / "src/uniscan/models/manifest.json").read_text(encoding="utf-8"))
    return {
        f"uniscan/models/{entry['filename']}".lower(): (entry["size"], entry["sha256"])
        for entry in manifest["assets"].values()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_payload_destination(relative: str) -> str:
    return relative.lower().removeprefix("_internal/")


def audit_portable_contents(
    root: Path,
    *,
    approved_model_assets: dict[str, tuple[int, str]] | None = None,
) -> None:
    root = Path(root)
    approved_models = (
        _approved_model_assets() if approved_model_assets is None else approved_model_assets
    )
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Portable artifact is missing: {', '.join(missing)}")

    forbidden: list[str] = []
    tkdnd_platforms: set[str] = set()
    payload_destinations: set[str] = set()
    payload_paths: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        relative_lower = relative.lower()
        normalized_destination = _normalized_payload_destination(relative)
        payload_destinations.add(normalized_destination)
        payload_paths[normalized_destination] = path
        if path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative)
        if relative_lower.endswith(FORBIDDEN_MODEL_SUFFIXES) and (
            normalized_destination not in approved_models
        ):
            forbidden.append(relative)
        parts = tuple(part.lower() for part in path.parts)
        if "tkdnd" in parts:
            index = parts.index("tkdnd")
            if index + 1 < len(parts):
                tkdnd_platforms.add(parts[index + 1])
    if forbidden:
        raise RuntimeError(f"Portable artifact contains forbidden files: {', '.join(forbidden)}")
    unexpected = sorted(tkdnd_platforms - ALLOWED_TKDND_PLATFORMS)
    if unexpected:
        raise RuntimeError(f"Portable artifact contains unrelated TkDND platforms: {unexpected}")
    if "win-x64" not in tkdnd_platforms:
        raise RuntimeError("Portable artifact is missing the Windows x64 TkDND runtime")

    missing_models = sorted(set(approved_models) - payload_destinations)
    if missing_models:
        raise RuntimeError(
            "Portable artifact is missing approved model assets: " + ", ".join(missing_models)
        )
    for destination, (expected_size, expected_hash) in approved_models.items():
        path = payload_paths[destination]
        if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
            raise RuntimeError(f"Portable model asset failed SHA-256 verification: {destination}")
    if approved_models:
        for model_name, relative in MODEL_NOTICE_PATHS.items():
            notice_path = root / relative
            if not notice_path.is_file():
                raise RuntimeError(f"Portable {model_name} model is missing its license notice")
            notice = notice_path.read_text(encoding="utf-8")
            if any(marker not in notice for marker in MODEL_NOTICE_MARKERS[model_name]):
                raise RuntimeError(f"Portable {model_name} model license notice is invalid")
        inventory = (
            (root / "THIRD_PARTY_LICENSES/FROZEN_PAYLOAD.txt").read_text(encoding="utf-8").lower()
        )
        if any(destination not in inventory for destination in approved_models) or (
            "uvdoc onnx export; apache-2.0" not in inventory
            or "docshadow onnx export; mit" not in inventory
        ):
            raise RuntimeError("Portable frozen inventory does not license its model assets")

    roboto_candidates = {
        destination
        for destination in payload_destinations
        if destination.startswith(ROBOTO_ASSET_PREFIX)
    }
    unexpected_roboto = sorted(roboto_candidates - ROBOTO_ASSET_DESTINATIONS)
    if unexpected_roboto:
        raise RuntimeError(
            "Portable artifact contains unreviewed Roboto assets: " + ", ".join(unexpected_roboto)
        )
    if roboto_candidates and roboto_candidates != ROBOTO_ASSET_DESTINATIONS:
        missing_roboto = sorted(ROBOTO_ASSET_DESTINATIONS - roboto_candidates)
        raise RuntimeError("Portable Roboto asset set is incomplete: " + ", ".join(missing_roboto))
    if roboto_candidates:
        notice_path = root / ROBOTO_NOTICE_PATH
        if not notice_path.is_file():
            raise RuntimeError("Portable Roboto assets are missing their Apache-2.0 license notice")
        notice = notice_path.read_text(encoding="utf-8")
        if any(marker not in notice for marker in ROBOTO_NOTICE_MARKERS):
            raise RuntimeError("Portable Roboto Apache-2.0 license notice is invalid")
        inventory = (root / "THIRD_PARTY_LICENSES/FROZEN_PAYLOAD.txt").read_text(encoding="utf-8")
        inventory_lower = inventory.lower()
        if any(asset not in inventory_lower for asset in ROBOTO_ASSET_DESTINATIONS) or (
            "roboto; apache-2.0" not in inventory_lower
        ):
            raise RuntimeError("Portable frozen inventory does not license its Roboto assets")

    for relative, marker in RUNTIME_NOTICE_MARKERS.items():
        notice = (root / relative).read_text(encoding="utf-8")
        if marker not in notice:
            raise RuntimeError(f"Portable runtime notice is invalid: {relative}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    stale_instructions = [
        token for token in ("run_uniscan.cmd", ".venv", "camscan_hybrid_tool") if token in readme
    ]
    if stale_instructions:
        raise RuntimeError(
            f"Portable README references unavailable development files: {stale_instructions}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    audit_portable_contents(args.dist_dir)
    print(f"Portable content audit OK: {args.dist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
