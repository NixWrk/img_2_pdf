"""Smoke-test a frozen Windows directory without importing the source tree."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import pypdfium2 as pdfium


def _run(executable: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _write_test_png(path: Path) -> None:
    width = height = 16
    rows = b"".join(b"\x00" + (b"\xf0\xf0\xf0" * width) for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _first_page_size(path: Path) -> tuple[float, float]:
    document = pdfium.PdfDocument(path)
    try:
        if len(document) != 1:
            raise RuntimeError(f"expected one page in {path}, found {len(document)}")
        page = document[0]
        try:
            return tuple(float(value) for value in page.get_size())
        finally:
            page.close()
    finally:
        document.close()


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: smoke_windows_artifact.py DIST_DIR", file=sys.stderr)
        return 2
    executable = Path(args[0]).resolve() / "uniscan.exe"
    if not executable.is_file():
        print(f"missing executable: {executable}", file=sys.stderr)
        return 2

    version = _run(executable, "--version")
    if version.returncode != 0 or not version.stdout.strip():
        print(version.stderr or "version smoke failed", file=sys.stderr)
        return 1
    doctor = _run(executable, "doctor", "--json")
    try:
        report = json.loads(doctor.stdout)
    except json.JSONDecodeError:
        print(doctor.stderr or doctor.stdout or "doctor did not return JSON", file=sys.stderr)
        return 1
    if doctor.returncode != 0 or not report.get("ok"):
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    gui_runtime = _run(executable, "doctor", "--gui-runtime", "--json")
    try:
        gui_report = json.loads(gui_runtime.stdout)
    except json.JSONDecodeError:
        print(
            gui_runtime.stderr or gui_runtime.stdout or "GUI doctor did not return JSON",
            file=sys.stderr,
        )
        return 1
    if gui_runtime.returncode != 0 or not gui_report.get("ok"):
        print(json.dumps(gui_report, indent=2), file=sys.stderr)
        return 1

    help_result = _run(executable, "--help")
    if help_result.returncode != 0 or "benchmark-geometry" not in help_result.stdout:
        print(help_result.stderr or "frozen CLI is missing current commands", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="uniscan_frozen_smoke_") as directory:
        root = Path(directory)
        source = root / "page.png"
        output = root / "output.pdf"
        _write_test_png(source)
        convert = _run(
            executable,
            "convert",
            "--input",
            str(source),
            "--output",
            str(output),
            "--no-detect",
            "--mode",
            "none",
            "--pdf-dpi",
            "72",
        )
        report_path = output.with_suffix(".pdf.report.json")
        if convert.returncode != 0 or not output.is_file() or not report_path.is_file():
            print(convert.stderr or convert.stdout or "conversion smoke failed", file=sys.stderr)
            return 1
        run_report = json.loads(report_path.read_text(encoding="utf-8"))
        if run_report.get("totalPages") != 1 or not output.read_bytes().startswith(b"%PDF"):
            print("conversion smoke produced invalid outputs", file=sys.stderr)
            return 1
        output_size = _first_page_size(output)
        if any(abs(value - 16.0) > 0.25 for value in output_size):
            print(f"conversion smoke produced wrong PDF page size: {output_size}", file=sys.stderr)
            return 1
        roundtrip = root / "roundtrip.pdf"
        pdf_import = _run(
            executable,
            "convert",
            "--input",
            str(output),
            "--output",
            str(roundtrip),
            "--pdf-dpi",
            "72",
            "--no-detect",
            "--mode",
            "none",
        )
        if pdf_import.returncode != 0 or not roundtrip.read_bytes().startswith(b"%PDF"):
            print(
                pdf_import.stderr or pdf_import.stdout or "PDF import smoke failed", file=sys.stderr
            )
            return 1
        roundtrip_size = _first_page_size(roundtrip)
        if any(
            abs(actual - expected) > 0.25 for actual, expected in zip(roundtrip_size, output_size)
        ):
            print(
                f"PDF roundtrip changed physical size: {output_size} -> {roundtrip_size}",
                file=sys.stderr,
            )
            return 1
    print(f"Frozen UniScan {version.stdout.strip()}: diagnostics OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
