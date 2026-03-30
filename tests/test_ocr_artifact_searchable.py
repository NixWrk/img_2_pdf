from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr.artifact_searchable import (
    _assign_lines_to_boxes,
    _build_searchable_pdf_from_text,
    _estimate_page_split_weights,
    _parse_artifact_filename,
    _split_lines_to_pages_by_weights,
    _split_text_to_pages,
    run_artifact_searchable_package,
)


def _build_sample_pdf(tmp_path: Path, name: str, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((200, 300, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / f"{name}.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=120)
    return pdf_path


def _extract_pdf_text(pdf_path: Path) -> str:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def test_parse_artifact_filename() -> None:
    document, engine = _parse_artifact_filename(Path("ГОСТ__chandra.txt"))
    assert document == "ГОСТ"
    assert engine == "chandra"

    with pytest.raises(ValueError):
        _parse_artifact_filename(Path("broken_name.txt"))


def test_split_text_to_pages_with_markers() -> None:
    text = "\n".join(
        [
            "[SOURCE PAGE 1]",
            "Page one text",
            "[SOURCE PAGE 2]",
            "Page two text",
        ]
    )
    pages = _split_text_to_pages(text, 2)
    assert pages == ["Page one text", "Page two text"]


def test_split_lines_to_pages_by_weights() -> None:
    lines = [f"L{i}" for i in range(12)]
    pages = _split_lines_to_pages_by_weights(lines, page_count=3, page_weights=[1.0, 5.0, 1.0])
    counts = [len(page.splitlines()) if page else 0 for page in pages]
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]
    assert sum(counts) == 12


def test_estimate_page_split_weights_clips_outliers() -> None:
    weights = _estimate_page_split_weights(
        [
            [(0.0, 0.0, 1.0, 1.0)] * 10,
            [(0.0, 0.0, 1.0, 1.0)] * 200,
            [],
        ]
    )
    assert len(weights) == 3
    assert weights[1] < 200.0
    assert weights[2] > 0.0


def test_assign_lines_to_boxes_balances_lines() -> None:
    lines = ["L1", "L2", "L3", "L4", "L5"]
    boxes = [(0.0, 0.0, 100.0, 20.0), (0.0, 25.0, 100.0, 45.0)]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 2
    assert "L1" in placements[0][1]
    assert "L5" in placements[1][1]


def test_assign_lines_to_boxes_merges_row_segments() -> None:
    lines = ["L1", "L2"]
    boxes = [
        (0.0, 0.0, 20.0, 10.0),
        (24.0, 1.0, 42.0, 11.0),
        (0.0, 20.0, 25.0, 30.0),
        (30.0, 21.0, 45.0, 31.0),
    ]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 2
    assert placements[0][0][2] >= 40.0
    assert placements[1][0][2] >= 40.0


def test_assign_lines_to_boxes_spreads_assignments_when_many_boxes() -> None:
    lines = ["L1", "L2", "L3"]
    boxes = [
        (0.0, 0.0, 30.0, 10.0),
        (0.0, 15.0, 30.0, 25.0),
        (0.0, 30.0, 30.0, 40.0),
        (0.0, 45.0, 30.0, 55.0),
        (0.0, 60.0, 30.0, 70.0),
        (0.0, 75.0, 30.0, 85.0),
    ]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 3
    y_positions = [item[0][1] for item in placements]
    assert y_positions[0] <= 1.0
    assert y_positions[-1] >= 70.0


def test_build_searchable_pdf_keeps_text_when_boxes_are_tiny(monkeypatch, tmp_path: Path) -> None:
    src_pdf = _build_sample_pdf(tmp_path, "tiny_box_fixture", [40])
    out_pdf = tmp_path / "tiny_box_out.pdf"
    text = "\n".join(f"Line {idx:03d}" for idx in range(120))

    monkeypatch.setattr(
        "uniscan.ocr.artifact_searchable._estimate_page_line_bboxes",
        lambda **_kwargs: [(0.0, 0.0, 40.0, 12.0)],
    )

    _build_searchable_pdf_from_text(
        source_pdf=src_pdf,
        text=text,
        out_pdf=out_pdf,
    )
    extracted = _extract_pdf_text(out_pdf)
    assert "Line 000" in extracted
    assert "Line 119" in extracted


def test_run_artifact_searchable_package_builds_pdfs(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    _build_sample_pdf(pdf_root, "ГОСТ с плохим качеством скана", [30, 90])
    (compare_dir / "ГОСТ с плохим качеством скана__chandra.txt").write_text(
        "Alpha line for page one.\n"
        "\u041f\u0440\u0438\u043c\u0435\u0440 "
        "\u0440\u0443\u0441\u0441\u043a\u043e\u0439 "
        "\u0441\u0442\u0440\u043e\u043a\u0438.",
        encoding="utf-8",
    )
    (compare_dir / "ГОСТ с плохим качеством скана__surya.txt").write_text(
        "Surya text content.",
        encoding="utf-8",
    )
    (compare_dir / "Missing Document__olmocr.txt").write_text("text", encoding="utf-8")

    results = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra", "surya", "olmocr"),
    )

    assert len(results) == 3
    ok_rows = [row for row in results if row.status == "ok"]
    err_rows = [row for row in results if row.status == "error"]
    assert len(ok_rows) == 2
    assert len(err_rows) == 1
    assert err_rows[0].engine == "olmocr"
    assert "not found" in (err_rows[0].error or "").lower()

    for row in ok_rows:
        assert row.searchable_pdf_path is not None
        pdf_path = Path(row.searchable_pdf_path)
        assert pdf_path.exists()
        extracted = _extract_pdf_text(pdf_path)
        assert extracted.strip()
        if row.engine == "chandra":
            assert any(0x0400 <= ord(ch) <= 0x04FF for ch in extracted)

    assert (output_dir / "artifact_searchable_summary.json").exists()
    assert (output_dir / "artifact_searchable_summary.csv").exists()


def test_cli_build_searchable_from_artifacts_success(monkeypatch, tmp_path: Path, capsys) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()
    output_dir.mkdir()

    def fake_run(**kwargs):
        assert kwargs["compare_dir"] == compare_dir
        assert kwargs["pdf_root"] == pdf_root
        assert kwargs["output_dir"] == output_dir
        assert kwargs["engines"] == ("chandra", "surya")
        return [
            SimpleNamespace(
                document="ГОСТ",
                engine="chandra",
                status="ok",
                source_pdf_path="x.pdf",
                text_artifact_path="x.txt",
                searchable_pdf_path="out.pdf",
                page_count=2,
                text_chars=123,
                elapsed_seconds=1.0,
                error=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_artifact_searchable_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_artifact_searchable_package", lambda rows: f"rows={len(rows)}")

    exit_code = main(
        [
            "build-searchable-from-artifacts",
            "--compare-dir",
            str(compare_dir),
            "--pdf-root",
            str(pdf_root),
            "--output",
            str(output_dir),
            "--engines",
            "chandra",
            "surya",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "rows=1" in stdout


def test_cli_build_searchable_from_artifacts_strict_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()
    output_dir.mkdir()

    def fake_run(**_kwargs):
        return [
            SimpleNamespace(
                document="ГОСТ",
                engine="chandra",
                status="ok",
                source_pdf_path="x.pdf",
                text_artifact_path="x.txt",
                searchable_pdf_path="ok.pdf",
                page_count=2,
                text_chars=123,
                elapsed_seconds=1.0,
                error=None,
            ),
            SimpleNamespace(
                document="ГОСТ",
                engine="surya",
                status="error",
                source_pdf_path="x.pdf",
                text_artifact_path="y.txt",
                searchable_pdf_path=None,
                page_count=0,
                text_chars=0,
                elapsed_seconds=1.0,
                error="broken",
            ),
        ]

    monkeypatch.setattr("uniscan.cli.run_artifact_searchable_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_artifact_searchable_package", lambda _rows: "summary")

    exit_code = main(
        [
            "build-searchable-from-artifacts",
            "--compare-dir",
            str(compare_dir),
            "--pdf-root",
            str(pdf_root),
            "--output",
            str(output_dir),
            "--strict",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "summary" in stdout
