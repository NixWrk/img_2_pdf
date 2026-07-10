from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from uniscan.cli import main
from uniscan.tools.batch_pipeline import resolve_input_paths, run_batch_pipeline


def _write_image(path: Path, value: int) -> None:
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    buffer.tofile(str(path))


def test_resolve_input_paths_expands_folders_in_natural_order(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_image(input_dir / "page10.png", 100)
    _write_image(input_dir / "page2.png", 20)

    resolved = resolve_input_paths([input_dir], output_pdf=input_dir / "result.pdf")

    assert [path.name for path in resolved] == ["page2.png", "page10.png"]


def test_run_batch_pipeline_writes_pdf_and_images(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_image(input_dir / "page1.png", 20)
    _write_image(input_dir / "page2.png", 100)

    result = run_batch_pipeline(
        inputs=[input_dir],
        output_pdf=tmp_path / "result",
        images_dir=tmp_path / "images",
        pdf_dpi=200,
        detect_document=False,
        lens_mode="none",
    )

    assert result.output_pdf == tmp_path / "result.pdf"
    assert result.output_pdf.exists()
    assert result.output_pdf.stat().st_size > 0
    assert result.total_pages == 2
    assert result.detected_pages == 0
    assert [path.name for path in result.image_outputs] == ["page_00001.png", "page_00002.png"]
    assert all(path.exists() for path in result.image_outputs)


def test_cli_convert_runs_end_to_end(tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_image(input_dir / "page.png", 80)
    output_pdf = tmp_path / "cli-result.pdf"

    exit_code = main(
        [
            "convert",
            "--input",
            str(input_dir),
            "--output",
            str(output_pdf),
            "--no-detect",
            "--mode",
            "none",
        ]
    )

    assert exit_code == 0
    assert output_pdf.exists()
    assert "Wrote 1 page(s)" in capsys.readouterr().out


def test_output_pdf_cannot_be_explicit_input(tmp_path) -> None:
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"not read")

    try:
        resolve_input_paths([input_pdf], output_pdf=input_pdf)
    except ValueError as exc:
        assert "cannot also be an explicit input" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_cli_convert_reports_input_errors_without_traceback(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "convert",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out.pdf"),
        ]
    )

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert "uniscan: error: Input does not exist" in stderr
    assert "Traceback" not in stderr
