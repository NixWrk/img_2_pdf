"""Verify that a built wheel contains production code and no unlicensed model assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path


REQUIRED_SUFFIXES = {
    "uniscan/tools/batch_pipeline.py",
    "uniscan/office_lens/models/README.md",
    "uniscan/models/manifest.json",
    "uniscan/models/UVDoc_grid.onnx",
    "uniscan/models/UVDoc_grid.onnx.data",
}
FORBIDDEN_MODEL_SUFFIXES = {
    ".onnx",
    ".ort",
    ".pdmodel",
    ".pdparams",
    ".pt",
    ".pth",
    ".safetensors",
}
ROOT = Path(__file__).resolve().parents[1]


def _approved_wheel_models() -> dict[str, tuple[int, str]]:
    manifest = json.loads((ROOT / "src/uniscan/models/manifest.json").read_text(encoding="utf-8"))
    return {
        f"uniscan/models/{entry['filename']}": (entry["size"], entry["sha256"])
        for name, entry in manifest["assets"].items()
        if name in {"uvdoc_graph", "uvdoc_data"}
    }


def _archived_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("uniscan-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one UniScan wheel in {directory}, found {len(wheels)}"
        )

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        approved_models = _approved_wheel_models()
        missing = sorted(REQUIRED_SUFFIXES - names)
        if missing:
            raise RuntimeError(f"Wheel is missing required files: {', '.join(missing)}")
        forbidden_models = sorted(
            name
            for name in names
            if (
                Path(name).suffix.lower() in FORBIDDEN_MODEL_SUFFIXES
                or name.lower().endswith(".onnx.data")
            )
            and name not in approved_models
        )
        if forbidden_models:
            raise RuntimeError(
                f"Wheel contains model weights without redistribution approval: "
                f"{', '.join(forbidden_models)}"
            )
        for name, (expected_size, expected_hash) in approved_models.items():
            info = archive.getinfo(name)
            if info.file_size != expected_size or _archived_sha256(archive, name) != expected_hash:
                raise RuntimeError(f"Wheel model asset failed SHA-256 verification: {name}")
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_points) != 1:
            raise RuntimeError("Wheel must contain exactly one entry_points.txt")
        raw_entry_points = archive.read(entry_points[0]).decode("utf-8")
        for script_name in ("uniscan =", "uniscan-office-lens ="):
            if script_name not in raw_entry_points:
                raise RuntimeError(f"Wheel entry points do not define {script_name.rstrip(' =')}")

        with tempfile.TemporaryDirectory(prefix="uniscan_wheel_verify_") as tmp:
            archive.extractall(tmp)
            sys.path.insert(0, tmp)
            try:
                for module_name in tuple(sys.modules):
                    if module_name == "uniscan" or module_name.startswith("uniscan."):
                        del sys.modules[module_name]
                importlib.import_module("uniscan")
                importlib.import_module("uniscan.core.scanner_adapter")
            finally:
                sys.path.remove(tmp)

    return wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args(argv)
    wheel = verify_wheel(args.wheel_dir)
    print(f"Verified wheel: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
