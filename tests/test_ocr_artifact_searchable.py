from __future__ import annotations

import json
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
    _expand_lines_to_target_count,
    _geometry_boxes_in_reading_order,
    _geometry_lines_in_reading_order,
    _has_explicit_page_markers,
    _placements_from_geometry_text_with_linefit,
    _placements_from_surya_geometry,
    _parse_artifact_filename,
    _split_line_to_word_fragments,
    _split_lines_to_pages_by_weights,
    _split_text_to_pages,
    build_compare_txt_from_benchmark,
    run_artifact_searchable_package,
)


def _build_sample_pdf(tmp_path: Path, name: str, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((200, 300, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / f"{name}.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=120)
    return pdf_path


def _rotate_pdf_90(source_pdf: Path, out_pdf: Path) -> Path:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(90)
        writer.add_page(page)
    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return out_pdf


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


def test_split_line_to_word_fragments_splits_into_tokens() -> None:
    parts = _split_line_to_word_fragments(
        "ИНСТРУМЕНТЫ МЕДИЦИНСКИЕ МЕТАЛЛИЧЕСКИЕ",
        bbox=(10.0, 20.0, 210.0, 40.0),
    )
    assert len(parts) == 5
    assert parts[0][1].startswith("ИНСТРУМЕНТЫ")
    assert parts[1][1] == " "
    assert parts[-1][1].strip() == "МЕТАЛЛИЧЕСКИЕ"
    assert parts[0][0][0] < parts[1][0][0] < parts[2][0][0]


def test_split_line_to_word_fragments_keeps_single_token() -> None:
    parts = _split_line_to_word_fragments("ГОСТ19126", bbox=(0.0, 0.0, 100.0, 20.0))
    assert len(parts) == 1
    assert parts[0][1] == "ГОСТ19126"


def test_expand_lines_to_target_count_splits_long_lines() -> None:
    source = ["A B C D E F G H", "I J K L"]
    expanded = _expand_lines_to_target_count(source, target_count=5)
    assert len(expanded) == 5
    assert "A B" in " ".join(expanded)
    assert "K L" in " ".join(expanded)


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


def test_placements_from_surya_geometry_scales_and_cleans_text() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 2000.0,
        "lines": [
            {
                "text": "<b>ИНСТРУМЕНТЫ</b>",
                "bbox": [100.0, 200.0, 300.0, 260.0],
            }
        ],
    }
    placements = _placements_from_surya_geometry(
        page_data=payload,
        page_width=500.0,
        page_height=1000.0,
    )
    assert len(placements) == 1
    bbox, text = placements[0]
    assert text == "ИНСТРУМЕНТЫ"
    assert bbox == pytest.approx((50.0, 100.0, 150.0, 130.0))


def test_placements_from_surya_geometry_auto_spread_orders_left_then_right() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    placements = _placements_from_surya_geometry(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    ordered_texts = [text for _, text in placements]
    assert ordered_texts == ["L-top", "L-mid", "L-bottom", "R-top", "R-mid", "R-bottom"]


def test_geometry_lines_in_reading_order_auto_spread() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    lines = _geometry_lines_in_reading_order(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    assert lines == ["L-top", "L-mid", "L-bottom", "R-top", "R-mid", "R-bottom"]


def test_geometry_boxes_in_reading_order_auto_spread() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    boxes = _geometry_boxes_in_reading_order(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    assert len(boxes) == 6
    y_positions = [item[1] for item in boxes]
    assert y_positions[:3] == sorted(y_positions[:3])
    assert y_positions[3:] == sorted(y_positions[3:])


def test_placements_from_geometry_text_with_linefit_prefers_detected_boxes() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "LINE A", "bbox": [100.0, 100.0, 900.0, 160.0]},
            {"text": "LINE B", "bbox": [100.0, 200.0, 900.0, 260.0]},
        ],
    }
    line_boxes = [
        (10.0, 20.0, 120.0, 36.0),
        (12.0, 40.0, 125.0, 56.0),
    ]
    placements = _placements_from_geometry_text_with_linefit(
        page_data=payload,
        page_width=500.0,
        page_height=500.0,
        line_boxes=line_boxes,
    )
    assert len(placements) == 2
    assert placements[0][0] == pytest.approx(line_boxes[0])
    assert placements[1][0] == pytest.approx(line_boxes[1])
    assert [text for _, text in placements] == ["LINE A", "LINE B"]


def test_placements_from_geometry_text_with_linefit_falls_back_to_geometry() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 2000.0,
        "lines": [{"text": "SINGLE", "bbox": [100.0, 200.0, 300.0, 260.0]}],
    }
    placements = _placements_from_geometry_text_with_linefit(
        page_data=payload,
        page_width=500.0,
        page_height=1000.0,
        line_boxes=[],
    )
    assert len(placements) == 1
    bbox, text = placements[0]
    assert text == "SINGLE"
    assert bbox == pytest.approx((50.0, 100.0, 150.0, 130.0))


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


def test_build_searchable_pdf_normalizes_rotated_pages(tmp_path: Path) -> None:
    base_pdf = _build_sample_pdf(tmp_path, "rotated_fixture_base", [80])
    rotated_pdf = _rotate_pdf_90(base_pdf, tmp_path / "rotated_fixture.pdf")
    out_pdf = tmp_path / "rotated_out.pdf"

    _build_searchable_pdf_from_text(
        source_pdf=rotated_pdf,
        text="[SOURCE PAGE 1]\nROTATED PAGE TEXT\n",
        out_pdf=out_pdf,
        surya_geometry_by_page={
            1: {
                "image_width": 300.0,
                "image_height": 200.0,
                "lines": [{"text": "ROTATED PAGE TEXT", "bbox": [30.0, 40.0, 260.0, 85.0]}],
            }
        },
    )

    from pypdf import PdfReader

    reader = PdfReader(str(out_pdf))
    assert int(reader.pages[0].get("/Rotate", 0) or 0) == 0
    extracted = _extract_pdf_text(out_pdf)
    assert "ROTATED PAGE TEXT" in extracted


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


def test_run_artifact_searchable_package_uses_chandra_sidecar_geometry(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [40])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("", encoding="utf-8")

    chandra_dir = compare_dir.parent / "chandra"
    chandra_dir.mkdir()
    (chandra_dir / "pages.json").write_text(
        json.dumps(
            {
                "pdf_path": str((pdf_root / f"{doc_name}.pdf")),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (chandra_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "CHANDRA GEOMETRY LINE", "bbox": [20, 20, 280, 60]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "CHANDRA GEOMETRY LINE" in extracted


def test_run_artifact_searchable_package_reads_nested_chandra_pages_json(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [40])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("TXT LINE SHOULD WIN", encoding="utf-8")

    nested_dir = compare_dir.parent / "chandra" / "chandra"
    nested_dir.mkdir(parents=True)
    (nested_dir / "pages.json").write_text(
        json.dumps(
            {
                "pdf_path": str((pdf_root / f"{doc_name}.pdf")),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (nested_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "GEOMETRY LINE", "bbox": [20, 20, 280, 60]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "TXT LINE SHOULD WIN" in extracted


def test_run_artifact_searchable_package_uses_chandra_geometry_on_pdf_name_mismatch(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [50])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("", encoding="utf-8")

    chandra_dir = compare_dir.parent / "chandra"
    chandra_dir.mkdir()
    (chandra_dir / "pages.json").write_text(
        json.dumps(
            {
                # Intentionally mismatched stem (simulates mojibake path in JSON).
                "pdf_path": str(pdf_root / "Ð¤Ð°Ð¹Ð».pdf"),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (chandra_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "GEOMETRY STILL APPLIED", "bbox": [30, 30, 260, 70]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "GEOMETRY STILL APPLIED" in extracted


def test_run_artifact_searchable_package_require_markers(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    _build_sample_pdf(pdf_root, "fixture_doc", [30, 90])
    (compare_dir / "fixture_doc__chandra.txt").write_text("plain markerless text", encoding="utf-8")

    results = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
        require_page_markers=True,
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert "no explicit page markers" in (results[0].error or "").lower()


def test_build_compare_txt_from_benchmark(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()

    src_txt = benchmark_root / "fixture_doc_chandra.txt"
    src_txt.write_text("[SOURCE PAGE 0001]\nHello\n", encoding="utf-8")
    payload = [
        {
            "engine": "chandra",
            "status": "ok",
            "artifact_path": str(src_txt),
        },
        {
            "engine": "surya",
            "status": "error",
            "artifact_path": "",
        },
    ]
    (benchmark_root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = build_compare_txt_from_benchmark(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        engines=("chandra", "surya"),
    )
    assert len(rows) == 2
    ok_rows = [row for row in rows if row.status == "ok"]
    err_rows = [row for row in rows if row.status == "error"]
    assert len(ok_rows) == 1
    assert len(err_rows) == 1
    assert (output_dir / "fixture_doc__chandra.txt").exists()
    assert (output_dir / "sources_map.txt").exists()


def test_has_explicit_page_markers_detects_marker_in_multiline_text() -> None:
    text = "preface line\n[SOURCE PAGE 0001]\nbody line"
    assert _has_explicit_page_markers(text) is True


def test_build_compare_txt_from_reports_without_summary(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()

    surya_dir = benchmark_root / "surya"
    chandra_dir = benchmark_root / "chandra"
    surya_dir.mkdir()
    chandra_dir.mkdir()

    surya_txt = surya_dir / "fixture_doc_surya.txt"
    chandra_txt = chandra_dir / "fixture_doc_chandra.txt"
    surya_txt.write_text("[SOURCE PAGE 0001]\nS\n", encoding="utf-8")
    chandra_txt.write_text("[SOURCE PAGE 0001]\nC\n", encoding="utf-8")

    surya_report = {
        "pdf_path": "fixture_doc.pdf",
        "results": [{"engine": "surya", "status": "ok", "artifact_path": str(surya_txt)}],
    }
    chandra_report = {
        "pdf_path": "fixture_doc.pdf",
        "results": [{"engine": "chandra", "status": "ok", "artifact_path": str(chandra_txt)}],
    }
    (surya_dir / "fixture_doc_ocr_benchmark.json").write_text(json.dumps(surya_report), encoding="utf-8")
    (chandra_dir / "fixture_doc_ocr_benchmark.json").write_text(json.dumps(chandra_report), encoding="utf-8")

    rows = build_compare_txt_from_benchmark(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        engines=("surya", "chandra"),
    )
    assert len(rows) == 2
    assert all(row.status == "ok" for row in rows)
    assert (output_dir / "fixture_doc__surya.txt").exists()
    assert (output_dir / "fixture_doc__chandra.txt").exists()
    sources_map = (output_dir / "sources_map.txt").read_text(encoding="utf-8")
    assert "discovered_reports=2" in sources_map


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
        assert kwargs["require_page_markers"] is False
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
        assert _kwargs["require_page_markers"] is True
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


def test_cli_prepare_compare_txt_strict_fails(tmp_path: Path, capsys) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()
    src_txt = benchmark_root / "fixture_doc_chandra.txt"
    src_txt.write_text("[SOURCE PAGE 0001]\nHello\n", encoding="utf-8")
    payload = [
        {"engine": "chandra", "status": "ok", "artifact_path": str(src_txt)},
        {"engine": "surya", "status": "error", "artifact_path": ""},
    ]
    (benchmark_root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "prepare-compare-txt",
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(output_dir),
            "--engines",
            "chandra",
            "surya",
            "--strict",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "chandra: ok" in stdout
    assert "surya: error" in stdout
