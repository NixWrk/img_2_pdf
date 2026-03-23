from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr import (
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_SURYA,
    resolve_pdf_page_indices,
    run_ocr_benchmark,
    sample_pdf_page_indices,
)

FIXTURE_PDF = Path(r"J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf")
ALL_ENGINES = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_CHANDRA,
)
SEARCHABLE_ENGINES = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PYMUPDF,
)
EXTRACTION_ENGINES = (
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_CHANDRA,
)


def _build_sample_pdf(tmp_path: Path, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((120, 180, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=150)
    return pdf_path


def _ready_status(engine_name: str, *, searchable_pdf: bool) -> SimpleNamespace:
    return SimpleNamespace(
        engine_name=engine_name,
        ready=True,
        missing=[],
        searchable_pdf=searchable_pdf,
        label=engine_name,
    )


def test_sample_pdf_page_indices_returns_evenly_distributed_pages() -> None:
    assert sample_pdf_page_indices(12, sample_size=3) == [0, 6, 11]
    assert sample_pdf_page_indices(12, sample_size=2) == [0, 11]
    assert sample_pdf_page_indices(12, sample_size=1) == [0]
    assert sample_pdf_page_indices(2, sample_size=5) == [0, 1]
    assert sample_pdf_page_indices(0, sample_size=5) == []


def test_resolve_pdf_page_indices_with_explicit_pages() -> None:
    assert resolve_pdf_page_indices(12, page_numbers=[3, 9, 3]) == [2, 8]
    assert resolve_pdf_page_indices(12, sample_size=2) == [0, 11]

    with pytest.raises(ValueError, match=">= 1"):
        resolve_pdf_page_indices(12, page_numbers=[0])
    with pytest.raises(ValueError, match="valid range is 1..12"):
        resolve_pdf_page_indices(12, page_numbers=[13])


def test_run_ocr_benchmark_writes_report_and_artifacts(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=engine_name in SEARCHABLE_ENGINES)

    def fake_searchable_pdf(image_paths, *, out_pdf, lang, engine_name):
        out_pdf.write_text(f"{engine_name}:{lang}:{len(image_paths)}", encoding="utf-8")
        return out_pdf

    def fake_extract_chars(_pdf_path: Path) -> int:
        return 321

    def fake_paddleocr(image_paths, *, lang):
        return f"paddle:{lang}:{len(image_paths)}", 12

    def fake_surya(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert work_dir.name == "surya_work"
        return f"surya:{lang}:{len(image_paths)}", 13

    def fake_mineru(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert work_dir.name == "mineru_work"
        return f"mineru:{lang}:{len(image_paths)}", 14

    def fake_chandra(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert work_dir.name == "chandra_work"
        return f"chandra:{lang}:{len(image_paths)}", 15

    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark.image_paths_to_searchable_pdf", fake_searchable_pdf)
    monkeypatch.setattr("uniscan.ocr.benchmark._extract_pdf_text_chars", fake_extract_chars)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_surya_direct", fake_surya)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_mineru_direct", fake_mineru)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_chandra_direct", fake_chandra)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=ALL_ENGINES,
        sample_size=2,
        dpi=120,
        lang="eng",
    )

    assert [result.engine for result in results] == list(ALL_ENGINES)
    assert all(result.status == "ok" for result in results)
    assert all(result.artifact_path and Path(result.artifact_path).exists() for result in results)
    for result in results:
        assert result.sample_pages == [1, 3]
    assert {result.text_chars for result in results if result.engine in SEARCHABLE_ENGINES} == {321}
    assert {result.text_chars for result in results if result.engine in EXTRACTION_ENGINES} == {12, 13, 14, 15}

    report_path = output_dir / "fixture_ocr_benchmark.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["pdf_path"] == str(pdf_path)
    assert payload["sample_pages"] == [1, 3]
    assert len(payload["results"]) == len(ALL_ENGINES)


def test_run_ocr_benchmark_uses_explicit_page_numbers(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_sample(_page_count: int, *, sample_size: int = 5) -> list[int]:
        raise AssertionError(f"sample_pdf_page_indices should not be called, sample_size={sample_size}")

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=False)

    def fake_paddleocr(image_paths, *, lang):
        return f"paddle:{lang}:{len(image_paths)}", 7

    monkeypatch.setattr("uniscan.ocr.benchmark.sample_pdf_page_indices", fake_sample)
    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=99,
        page_numbers=(2,),
        dpi=120,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [2]
    report = json.loads((output_dir / "fixture_ocr_benchmark.json").read_text(encoding="utf-8"))
    assert report["sample_pages"] == [2]


def test_run_ocr_benchmark_unready_engine_is_error(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=engine_name,
            ready=False,
            missing=["dependency-x"],
            searchable_pdf=False,
            label=engine_name,
        )

    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_SURYA,),
        sample_size=1,
        dpi=100,
    )

    assert len(results) == 1
    assert results[0].engine == OCR_ENGINE_SURYA
    assert results[0].status == "error"
    assert results[0].note == "missing: dependency-x"
    assert results[0].artifact_path is not None


@pytest.mark.skipif(not FIXTURE_PDF.exists(), reason="external OCR fixture is not available")
def test_run_ocr_benchmark_uses_external_fixture_smoke(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"

    def fake_sample(_page_count: int, *, sample_size: int = 5) -> list[int]:
        assert sample_size == 1
        return [0]

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=False)

    def fake_paddleocr(image_paths, *, lang):
        assert len(image_paths) == 1
        return f"{lang}:fixture", 7

    monkeypatch.setattr("uniscan.ocr.benchmark.sample_pdf_page_indices", fake_sample)
    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=FIXTURE_PDF,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        dpi=72,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [1]
    assert results[0].artifact_path is not None
    assert Path(results[0].artifact_path).exists()
    assert (output_dir / "Imaging Edge Mobile_paddleocr_uvdoc_ocr_benchmark.json").exists()


def test_cli_benchmark_ocr_uses_runner_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**kwargs):
        assert kwargs["pdf_path"] == pdf_path
        assert kwargs["output_dir"] == output_dir
        assert kwargs["sample_size"] == 5
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[1],
                elapsed_seconds=1.23,
                artifact_path=str(output_dir / "fixture_paddleocr.txt"),
                text_chars=7,
                memory_delta_mb=1.0,
                error=None,
                note=None,
            )
        ]

    def fake_summary(results):
        assert len(results) == 1
        return "paddleocr ok"

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", fake_summary)

    exit_code = main(["benchmark-ocr", "--pdf", str(pdf_path), "--output", str(output_dir)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "paddleocr ok" in stdout


def test_cli_benchmark_ocr_parses_pages(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**kwargs):
        assert kwargs["page_numbers"] == (3, 9)
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[3, 9],
                elapsed_seconds=1.23,
                artifact_path=str(output_dir / "fixture_paddleocr.txt"),
                text_chars=7,
                memory_delta_mb=1.0,
                error=None,
                note=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", lambda _results: "ok")

    exit_code = main(
        [
            "benchmark-ocr",
            "--pdf",
            str(pdf_path),
            "--output",
            str(output_dir),
            "--pages",
            "3,9",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "ok" in stdout


def test_cli_benchmark_ocr_strict_fails_when_any_engine_not_ok(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**_kwargs):
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[1],
                elapsed_seconds=1.0,
                artifact_path=str(output_dir / "ok.txt"),
                text_chars=10,
                memory_delta_mb=1.1,
                error=None,
                note=None,
            ),
            SimpleNamespace(
                engine=OCR_ENGINE_SURYA,
                status="error",
                sample_pages=[1],
                elapsed_seconds=1.0,
                artifact_path=str(output_dir / "err.txt"),
                text_chars=0,
                memory_delta_mb=1.2,
                error="broken",
                note=None,
            ),
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", lambda results: f"rows={len(results)}")

    exit_code = main(
        ["benchmark-ocr", "--pdf", str(pdf_path), "--output", str(output_dir), "--strict"]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 1
    assert "rows=2" in stdout
