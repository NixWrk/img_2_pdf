"""Verify that a built wheel contains and can load the production model assets."""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import zipfile
from pathlib import Path


REQUIRED_SUFFIXES = {
    "uniscan/tools/batch_pipeline.py",
    "uniscan/office_lens/models/mnv2_ep42_wb_quant.ort",
    "uniscan/office_lens/models/triclass_doc_classifier.ort",
}


def verify_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("uniscan-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one UniScan wheel in {directory}, found {len(wheels)}"
        )

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_SUFFIXES - names)
        if missing:
            raise RuntimeError(f"Wheel is missing required files: {', '.join(missing)}")
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
                package = importlib.import_module("uniscan.office_lens")
                package.OfficeLensOnnx()
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
