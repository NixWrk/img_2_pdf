from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from uniscan.cli import main
from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    ScanAdapterError,
    ScanOutput,
)
from uniscan.tools.crop_benchmark import run_crop_benchmark


def _write_image(path: Path, value: int) -> None:
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    buf.tofile(str(path))


def test_run_crop_benchmark_preserves_natural_input_order(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_image(input_dir / "page10.png", 90)
    _write_image(input_dir / "page2.png", 30)

    seen_means: list[tuple[str, int]] = []
    exported: dict[str, list[str]] = {}

    def fake_probe(*_args, **_kwargs) -> None:
        return None

    def fake_scan(image, *, backends, **_kwargs):
        backend = backends[0]
        seen_means.append((backend, int(image.mean())))
        return ScanOutput(
            warped=image,
            contour=None,
            backend=backend,
            detected=True,
            raw_result=None,
        )

    def fake_export(image_paths, *, out_pdf, dpi):
        exported[out_pdf.stem] = [Path(path).name for path in image_paths]
        out_pdf.write_bytes(f"pdf dpi={dpi}".encode("utf-8"))
        return out_pdf

    monkeypatch.setattr("uniscan.tools.crop_benchmark.probe_detector_backend", fake_probe)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.scan_with_document_detector", fake_scan)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.export_image_paths_as_pdf", fake_export)

    results = run_crop_benchmark(
        input_dir=input_dir,
        output_dir=output_dir,
        backends=(DETECTOR_BACKEND_OPENCV, DETECTOR_BACKEND_PADDLEOCR_UVDOC),
        pdf_dpi=220,
    )

    assert [(backend, mean) for backend, mean in seen_means] == [
        (DETECTOR_BACKEND_OPENCV, 30),
        (DETECTOR_BACKEND_OPENCV, 90),
        (DETECTOR_BACKEND_PADDLEOCR_UVDOC, 30),
        (DETECTOR_BACKEND_PADDLEOCR_UVDOC, 90),
    ]
    assert [result.backend for result in results] == [
        DETECTOR_BACKEND_OPENCV,
        DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    ]
    assert all(result.output_pdf is not None for result in results)
    assert all(result.detected_pages == 2 for result in results)
    assert all(result.total_pages == 2 for result in results)
    assert exported == {
        "input_opencv_quad": ["00001.png", "00002.png"],
        "input_paddleocr_uvdoc": ["00001.png", "00002.png"],
    }


def test_run_crop_benchmark_defaults_to_paddleocr_uvdoc(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_image(input_dir / "page1.png", 77)

    seen_backends: list[str] = []

    def fake_probe(*_args, **_kwargs) -> None:
        return None

    def fake_scan(image, *, backends, **_kwargs):
        backend = backends[0]
        seen_backends.append(backend)
        return ScanOutput(
            warped=image,
            contour=None,
            backend=backend,
            detected=True,
            raw_result=None,
        )

    def fake_export(image_paths, *, out_pdf, dpi):
        out_pdf.write_bytes(f"{len(image_paths)}:{dpi}".encode("utf-8"))
        return out_pdf

    monkeypatch.setattr("uniscan.tools.crop_benchmark.probe_detector_backend", fake_probe)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.scan_with_document_detector", fake_scan)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.export_image_paths_as_pdf", fake_export)

    results = run_crop_benchmark(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert seen_backends == [DETECTOR_BACKEND_PADDLEOCR_UVDOC]
    assert [result.backend for result in results] == [DETECTOR_BACKEND_PADDLEOCR_UVDOC]


def test_run_crop_benchmark_keeps_other_backends_when_one_is_unavailable(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_image(input_dir / "page1.png", 120)

    def fake_probe(backend, **_kwargs) -> None:
        if backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC:
            raise ScanAdapterError("uvdoc missing")

    def fake_scan(image, *, backends, **_kwargs):
        backend = backends[0]
        return ScanOutput(
            warped=image,
            contour=None,
            backend=backend,
            detected=True,
            raw_result=None,
        )

    def fake_export(image_paths, *, out_pdf, dpi):
        out_pdf.write_bytes(f"{len(image_paths)}:{dpi}".encode("utf-8"))
        return out_pdf

    monkeypatch.setattr("uniscan.tools.crop_benchmark.probe_detector_backend", fake_probe)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.scan_with_document_detector", fake_scan)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.export_image_paths_as_pdf", fake_export)

    results = run_crop_benchmark(
        input_dir=input_dir,
        output_dir=output_dir,
        backends=(DETECTOR_BACKEND_OPENCV, DETECTOR_BACKEND_PADDLEOCR_UVDOC),
    )

    ok_result, failed_result = results
    assert ok_result.backend == DETECTOR_BACKEND_OPENCV
    assert ok_result.output_pdf is not None
    assert ok_result.error is None
    assert failed_result.backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC
    assert failed_result.output_pdf is None
    assert failed_result.error == "uvdoc missing"


def test_run_crop_benchmark_ignores_its_own_output_pdfs_in_input_folder(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_image(input_dir / "page1.png", 40)
    (input_dir / "input_opencv_quad.pdf").write_bytes(b"old")
    (input_dir / "input_paddleocr_uvdoc.pdf").write_bytes(b"old")

    seen_means: list[int] = []

    def fake_probe(*_args, **_kwargs) -> None:
        return None

    def fake_scan(image, *, backends, **_kwargs):
        seen_means.append(int(image.mean()))
        backend = backends[0]
        return ScanOutput(
            warped=image,
            contour=None,
            backend=backend,
            detected=True,
            raw_result=None,
        )

    def fake_export(image_paths, *, out_pdf, dpi):
        out_pdf.write_bytes(f"{len(image_paths)}:{dpi}".encode("utf-8"))
        return out_pdf

    monkeypatch.setattr("uniscan.tools.crop_benchmark.probe_detector_backend", fake_probe)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.scan_with_document_detector", fake_scan)
    monkeypatch.setattr("uniscan.tools.crop_benchmark.export_image_paths_as_pdf", fake_export)

    results = run_crop_benchmark(
        input_dir=input_dir,
        output_dir=input_dir,
        backends=(DETECTOR_BACKEND_OPENCV, DETECTOR_BACKEND_PADDLEOCR_UVDOC),
    )

    assert seen_means == [40, 40]
    assert all(result.output_pdf is not None for result in results)


def test_cli_benchmark_crop_uses_runner_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    def fake_run_crop_benchmark(**kwargs):
        assert kwargs["input_dir"] == input_dir
        assert kwargs["output_dir"] == output_dir
        assert kwargs["pdf_dpi"] == 300
        return [
            type(
                "Result",
                (),
                {
                    "backend": DETECTOR_BACKEND_PADDLEOCR_UVDOC,
                    "output_pdf": output_dir / "input_paddleocr_uvdoc.pdf",
                    "detected_pages": 2,
                    "total_pages": 2,
                    "error": None,
                },
            )()
        ]

    monkeypatch.setattr("uniscan.cli.run_crop_benchmark", fake_run_crop_benchmark)

    exit_code = main(["benchmark-crop", "--input", str(input_dir), "--output", str(output_dir)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "paddleocr_uvdoc" in stdout
