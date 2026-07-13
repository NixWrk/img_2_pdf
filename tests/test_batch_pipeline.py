from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from uniscan.cli import main
from uniscan.core.scanner_adapter import DETECTOR_BACKEND_CV_HYBRID, ScanOutput
from uniscan.tools.batch_pipeline import resolve_input_paths, run_batch_pipeline


def _write_image(path: Path, value: int) -> None:
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    buffer.tofile(str(path))


def _write_pdf(path: Path, page_count: int) -> None:
    document = pdfium.PdfDocument.new()
    try:
        for _index in range(page_count):
            page = document.new_page(width=100, height=100)
            page.close()
        document.save(str(path))
    finally:
        document.close()


def _write_sized_pdf(path: Path, *, width: float, height: float) -> None:
    document = pdfium.PdfDocument.new()
    try:
        page = document.new_page(width=width, height=height)
        page.close()
        document.save(str(path))
    finally:
        document.close()


def _pdf_page_size(path: Path) -> tuple[float, float]:
    document = pdfium.PdfDocument(str(path))
    try:
        page = document[0]
        try:
            return page.get_size()
        finally:
            page.close()
    finally:
        document.close()


def _pdf_page_count(path: Path) -> int:
    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def _write_curved_text_page(path: Path) -> None:
    height, width = 700, 900
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for baseline in range(90, 620, 65):
        for x in range(60, 840, 22):
            normalized_x = (2.0 * x / (width - 1)) - 1.0
            displacement = int(round(18.0 * ((normalized_x**2) - 0.35)))
            cv2.rectangle(
                image,
                (x, baseline + displacement),
                (x + 12, baseline + displacement + 11),
                (0, 0, 0),
                -1,
            )
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    buffer.tofile(str(path))


def _write_sideways_text_page(path: Path) -> None:
    image = np.full((720, 520, 3), 255, dtype=np.uint8)
    for index, text in enumerate(
        (
            "Document page",
            "quickly aligns",
            "baseline glyphs",
            "properly oriented",
            "local processing",
            "keeps text safe",
            "without any OCR",
        )
    ):
        cv2.putText(
            image,
            text,
            (35, 100 + index * 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    sideways = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    ok, buffer = cv2.imencode(".png", sideways)
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
    assert result.fallback_pages == 0
    assert result.report_path.exists()
    assert [path.name for path in result.image_outputs] == ["page_00001.png", "page_00002.png"]
    assert all(path.exists() for path in result.image_outputs)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["totalPages"] == 2
    assert report["detectionEnabled"] is False
    assert report["orientationMethod"] == "none"
    assert report["deskewMethod"] == "none"
    assert report["dewarpMethod"] == "none"


def test_batch_images_export_preserves_unrelated_neighbours(tmp_path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    notes = images_dir / "notes.txt"
    notes.write_text("personal notes", encoding="utf-8")

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "result.pdf",
        images_dir=images_dir,
        detect_document=False,
        lens_mode="none",
    )

    assert result.image_outputs[0].is_file()
    assert notes.read_text(encoding="utf-8") == "personal notes"
    assert (images_dir / ".uniscan-export-manifest.json").is_file()


def test_report_records_complete_effective_processing_configuration(tmp_path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 100)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "configured.pdf",
        images_dir=tmp_path / "configured-pages",
        image_format="jpg",
        pdf_dpi=200,
        detect_document=False,
        two_page_mode=True,
        lens_mode="document",
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schemaVersion"] == 2
    assert report["pdfDpi"] == 200
    assert report["imageFormat"] == "jpg"
    assert report["imagesDirectory"] == str(tmp_path / "configured-pages")
    assert report["strictDetect"] is False
    assert report["twoPageMode"] is True
    assert report["lensMode"] == "document"
    assert report["preprocessPreset"] == "Document"
    assert report["postprocessName"] == "Grayscale"
    assert report["preprocessEnabled"] is True
    assert report["contrast"] == 1.25
    assert report["brightness"] == 10
    assert report["denoise"] == 4
    assert report["threshold"] == 170
    assert report["applyThreshold"] is False
    assert report["detectorBackends"] == []


@pytest.mark.parametrize("dpi", (72, 300))
def test_pdf_input_roundtrip_preserves_physical_page_size(tmp_path, dpi: int) -> None:
    source = tmp_path / f"source-{dpi}.pdf"
    _write_sized_pdf(source, width=144.0, height=288.0)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / f"roundtrip-{dpi}.pdf",
        pdf_dpi=dpi,
        detect_document=False,
        lens_mode="none",
    )

    width, height = _pdf_page_size(result.output_pdf)
    assert width == pytest.approx(144.0, abs=0.25)
    assert height == pytest.approx(288.0, abs=0.25)


@pytest.mark.parametrize(
    ("layout", "dpi", "expected_width", "expected_height"),
    (
        ("a4", 72, 595.28, 841.89),
        ("a4", 300, 595.28, 841.89),
        ("letter", 72, 612.0, 792.0),
        ("letter", 300, 612.0, 792.0),
    ),
)
def test_standard_layout_pdf_has_correct_physical_page_size(
    tmp_path,
    layout: str,
    dpi: int,
    expected_width: float,
    expected_height: float,
) -> None:
    source = tmp_path / f"source-{layout}-{dpi}.png"
    _write_image(source, 100)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / f"{layout}-{dpi}.pdf",
        pdf_dpi=dpi,
        detect_document=False,
        lens_mode="none",
        page_layout=layout,
    )

    width, height = _pdf_page_size(result.output_pdf)
    assert width == pytest.approx(expected_width, abs=0.5)
    assert height == pytest.approx(expected_height, abs=0.5)


def test_run_batch_pipeline_applies_and_reports_textline_dewarp(tmp_path) -> None:
    source = tmp_path / "curved.png"
    _write_curved_text_page(source)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "dewarped.pdf",
        images_dir=tmp_path / "dewarped-pages",
        detect_document=False,
        lens_mode="none",
        deskew_method="none",
        dewarp_method="textline",
    )

    assert result.pages[0].dewarp_applied is True
    assert result.pages[0].dewarp_selected_method == "textline"
    assert result.pages[0].dewarp_line_count >= 3
    assert result.pages[0].dewarp_max_displacement_px > 2.0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["dewarpMethod"] == "textline"
    assert report["pages"][0]["dewarpApplied"] is True
    assert report["pages"][0]["dewarpSelectedMethod"] == "textline"
    assert report["pages"][0]["dewarpReason"] is None


def test_run_batch_pipeline_auto_dewarp_reports_quality_metrics(tmp_path) -> None:
    source = tmp_path / "curved-auto.png"
    _write_curved_text_page(source)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "auto-dewarped.pdf",
        detect_document=False,
        lens_mode="none",
        dewarp_method="auto",
    )

    page = result.pages[0]
    assert page.dewarp_selected_method == "textline"
    assert page.dewarp_curvature_after_px < page.dewarp_curvature_before_px
    assert page.dewarp_duration_ms > 0.0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    report_page = report["pages"][0]
    assert report["autoDewarpUvdoc"] is False
    assert report_page["dewarpSelectedMethod"] == "textline"
    assert report_page["dewarpCurvatureAfterPx"] < report_page["dewarpCurvatureBeforePx"]


def test_run_batch_pipeline_places_content_on_standard_page(tmp_path) -> None:
    source = tmp_path / "content.png"
    image = np.full((300, 200, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "content",
        (35, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(source))

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "a4.pdf",
        images_dir=tmp_path / "a4-pages",
        pdf_dpi=100,
        detect_document=False,
        lens_mode="none",
        page_layout="a4",
        page_margin_mm=10,
        horizontal_alignment="left",
        vertical_alignment="top",
    )

    output = cv2.imread(str(result.image_outputs[0]))
    assert output.shape[:2] == (1169, 827)
    assert result.pages[0].layout_applied is True
    assert result.pages[0].content_box is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["pageLayout"] == "a4"
    assert report["pages"][0]["layoutApplied"] is True


def test_run_batch_pipeline_reports_cleanup_and_lighting_diagnostics(tmp_path) -> None:
    source = tmp_path / "uneven.png"
    image = np.full((320, 480, 3), 210, dtype=np.uint8)
    image[:, :150] = 115
    cv2.putText(
        image,
        "Document line",
        (45, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, (370, 90), 24, (255, 255, 255), -1)
    image[25, 25] = 0
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(source))

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "cleanup.pdf",
        images_dir=tmp_path / "cleanup-pages",
        detect_document=False,
        lens_mode="none",
        binarization_method="sauvola",
        binarization_window=31,
        despeckle_strength="conservative",
        lighting_diagnostics=True,
    )

    output = cv2.imread(str(result.image_outputs[0]), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(output).tolist()).issubset({0, 255})
    page = result.pages[0]
    assert page.binarization_method == "sauvola"
    assert page.despeckle_removed_components >= 1
    assert page.shadow_fraction is not None and page.shadow_fraction > 0.05
    assert page.glare_fraction is not None and page.glare_fraction > 0.001
    assert "uneven_shadow" in page.lighting_warnings
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["binarizationMethod"] == "sauvola"
    assert report["despeckleStrength"] == "conservative"
    assert report["lightingDiagnostics"] is True
    assert report["pages"][0]["despeckleRemovedComponents"] >= 1


def test_run_batch_pipeline_reuses_persistent_stage_cache(tmp_path) -> None:
    source = tmp_path / "cached.png"
    _write_curved_text_page(source)
    cache_dir = tmp_path / "stage-cache"

    first = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "first.pdf",
        detect_document=False,
        lens_mode="none",
        dewarp_method="textline",
        binarization_method="otsu",
        stage_cache_dir=cache_dir,
        stage_cache_max_mb=16,
    )
    second = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "second.pdf",
        detect_document=False,
        lens_mode="none",
        dewarp_method="textline",
        binarization_method="otsu",
        stage_cache_dir=cache_dir,
        stage_cache_max_mb=16,
    )

    assert first.pages[0].processing_cache_hits == ()
    assert second.pages[0].processing_cache_hits == ("dewarp", "cleanup")
    report = json.loads(second.report_path.read_text(encoding="utf-8"))
    assert report["stageCacheEnabled"] is True
    assert report["stageCacheMaxMb"] == 16
    assert report["stageCacheStats"]["hits"] >= 2
    assert report["pages"][0]["processingCacheHits"] == ["dewarp", "cleanup"]


def test_run_batch_pipeline_applies_and_reports_orientation(tmp_path) -> None:
    source = tmp_path / "sideways.png"
    _write_sideways_text_page(source)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "oriented.pdf",
        images_dir=tmp_path / "oriented-pages",
        detect_document=False,
        lens_mode="none",
        orientation_method="auto",
    )

    page = result.pages[0]
    assert page.orientation_applied is True
    assert page.orientation_angle_degrees == 270
    assert page.orientation_confidence > 0.2
    assert page.orientation_reason is None
    output = cv2.imread(str(result.image_outputs[0]))
    assert output.shape[:2] == (720, 520)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["orientationMethod"] == "auto"
    assert report["pages"][0]["orientationApplied"] is True
    assert report["pages"][0]["orientationAngleDegrees"] == 270


def test_run_batch_pipeline_reports_hybrid_deskew_fallback_evidence(tmp_path) -> None:
    source = tmp_path / "no-lines.png"
    _write_image(source, 255)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=tmp_path / "deskew-evidence.pdf",
        detect_document=False,
        lens_mode="none",
        deskew_method="hybrid",
    )

    page = result.pages[0]
    assert page.deskew_method == "hybrid"
    assert page.deskew_selected_method == "min_area"
    assert page.deskew_confidence == 0.0
    assert page.deskew_line_count == 0
    assert page.deskew_reason == "hough_no_lines;min_area_selected"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    report_page = report["pages"][0]
    assert report_page["deskewMethod"] == "hybrid"
    assert report_page["deskewSelectedMethod"] == "min_area"
    assert report_page["deskewConfidence"] == 0.0
    assert report_page["deskewLineCount"] == 0
    assert report_page["deskewReason"] == "hough_no_lines;min_area_selected"


def test_run_batch_pipeline_rejects_unknown_geometry_methods(tmp_path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)

    with pytest.raises(ValueError, match="Unsupported orientation method"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "orientation.pdf",
            orientation_method="missing",
        )

    with pytest.raises(ValueError, match="Unsupported deskew method"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "deskew.pdf",
            deskew_method="missing",
        )
    with pytest.raises(ValueError, match="Unsupported dewarp method"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "dewarp.pdf",
            dewarp_method="missing",
        )
    with pytest.raises(ValueError, match="requires dewarp_method='auto'"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "uvdoc-auto.pdf",
            dewarp_method="none",
            auto_dewarp_uvdoc=True,
        )
    with pytest.raises(ValueError, match="Unsupported binarization"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "binarization.pdf",
            binarization_method="missing",
        )
    with pytest.raises(ValueError, match="Unsupported despeckle"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "despeckle.pdf",
            despeckle_strength="missing",
        )
    with pytest.raises(ValueError, match="Stage cache size"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "cache.pdf",
            stage_cache_max_mb=0,
        )

    with pytest.raises(ValueError, match="UVDoc is a dewarp method"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "uvdoc-detector.pdf",
            detector_policy="paddleocr_uvdoc",
        )


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
            "--illumination-correction",
        ]
    )

    assert exit_code == 0
    assert output_pdf.exists()
    report = json.loads(output_pdf.with_suffix(".pdf.report.json").read_text(encoding="utf-8"))
    assert report["illuminationCorrection"] is True
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


def test_output_pdf_cannot_be_an_input_discovered_in_folder(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_pdf = input_dir / "result.pdf"
    _write_pdf(output_pdf, page_count=1)

    with pytest.raises(ValueError, match="input discovered in a folder"):
        resolve_input_paths([input_dir], output_pdf=output_pdf)

    assert _pdf_page_count(output_pdf) == 1


def test_output_targets_cannot_overlap_replaceable_images_directory(tmp_path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="PDF output cannot be inside"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_dir / "document.pdf",
            images_dir=output_dir,
            detect_document=False,
        )

    with pytest.raises(ValueError, match="different paths"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "same.pdf",
            report_path=tmp_path / "same.pdf",
            detect_document=False,
        )


def test_report_cannot_replace_an_input_file(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    original = source.read_bytes()

    def processing_must_not_start(*_args, **_kwargs):
        raise AssertionError("target validation must run before input processing")

    monkeypatch.setattr("uniscan.tools.batch_pipeline.iter_input_items", processing_must_not_start)

    with pytest.raises(ValueError, match="JSON report cannot also be an input"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "document.pdf",
            report_path=source,
            detect_document=False,
        )

    assert source.read_bytes() == original


@pytest.mark.parametrize("target_kind", ("pdf", "report", "images"))
def test_outputs_cannot_replace_targets_of_the_wrong_type(tmp_path, target_kind: str) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    wrong_target = tmp_path / ("wrong.pdf" if target_kind == "pdf" else "wrong")

    if target_kind in {"pdf", "report"}:
        wrong_target.mkdir()
        (wrong_target / "keep.txt").write_text("keep", encoding="utf-8")
    else:
        wrong_target.write_text("keep", encoding="utf-8")

    kwargs = {
        "inputs": [source],
        "output_pdf": wrong_target if target_kind == "pdf" else tmp_path / "document.pdf",
        "detect_document": False,
    }
    if target_kind == "report":
        kwargs["report_path"] = wrong_target
    elif target_kind == "images":
        kwargs["images_dir"] = wrong_target

    with pytest.raises(ValueError, match="exists and is not"):
        run_batch_pipeline(**kwargs)

    if target_kind in {"pdf", "report"}:
        assert (wrong_target / "keep.txt").read_text(encoding="utf-8") == "keep"
    else:
        assert wrong_target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("report_path", "images_dir", "message"),
    (
        (Path("document.pdf/report.json"), None, "cannot contain one another"),
        (None, Path("document.pdf/images"), "nested below a file output"),
    ),
)
def test_file_targets_cannot_be_ancestors_of_other_outputs(
    tmp_path, report_path: Path | None, images_dir: Path | None, message: str
) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    output_pdf = tmp_path / "document.pdf"

    with pytest.raises(ValueError, match=message):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_pdf,
            report_path=(tmp_path / report_path) if report_path is not None else None,
            images_dir=(tmp_path / images_dir) if images_dir is not None else None,
            detect_document=False,
        )


def test_images_output_directory_cannot_contain_inputs(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "page.png"
    _write_image(source, 90)

    with pytest.raises(ValueError, match="cannot contain input"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "document.pdf",
            images_dir=source_dir,
            detect_document=False,
        )


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


def test_strict_detection_rejects_disabled_detection(tmp_path) -> None:
    source = tmp_path / "source.png"
    _write_image(source, 90)
    with pytest.raises(ValueError, match="strict_detect"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=tmp_path / "output.pdf",
            detect_document=False,
            strict_detect=True,
        )


def test_multi_page_pdf_is_processed_one_page_at_a_time(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "many-pages.pdf"
    _write_pdf(pdf_path, page_count=24)
    batch_sizes: list[int] = []

    from uniscan.tools import batch_pipeline

    real_process = batch_pipeline.process_loaded_items

    def tracking_process(loaded_items, **kwargs):
        batch_sizes.append(len(loaded_items))
        return real_process(loaded_items, **kwargs)

    monkeypatch.setattr(batch_pipeline, "process_loaded_items", tracking_process)
    result = run_batch_pipeline(
        inputs=[pdf_path],
        output_pdf=tmp_path / "streamed.pdf",
        pdf_dpi=72,
        detect_document=False,
        lens_mode="none",
    )

    assert result.total_pages == 24
    assert batch_sizes == [1] * 24
    assert result.pages[0].name.endswith("[p0001]")
    assert result.pages[-1].name.endswith("[p0024]")


def test_failed_build_preserves_existing_outputs(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    _write_image(image_path, 80)
    output_pdf = tmp_path / "result.pdf"
    output_pdf.write_bytes(b"existing-pdf")
    report_path = tmp_path / "result.pdf.report.json"
    report_path.write_text("existing-report", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "keep.txt").write_text("existing-images", encoding="utf-8")

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("forced build failure")

    monkeypatch.setattr("uniscan.tools.batch_pipeline.export_image_paths_as_pdf", fail_export)

    with pytest.raises(RuntimeError, match="forced build failure"):
        run_batch_pipeline(
            inputs=[image_path],
            output_pdf=output_pdf,
            images_dir=images_dir,
            detect_document=False,
            lens_mode="none",
        )

    assert output_pdf.read_bytes() == b"existing-pdf"
    assert report_path.read_text(encoding="utf-8") == "existing-report"
    assert (images_dir / "keep.txt").read_text(encoding="utf-8") == "existing-images"
    assert not list(tmp_path.glob(".*.stage-*"))


def test_detector_policy_and_fallback_are_reported(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    _write_image(image_path, 80)
    seen_backends: list[tuple[str, ...]] = []

    def no_detection(image, *, backends, **_kwargs):
        seen_backends.append(backends)
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result={"errors": ["cv_hybrid: no candidate"]},
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", no_detection)
    result = run_batch_pipeline(
        inputs=[image_path],
        output_pdf=tmp_path / "fallback.pdf",
        detector_policy="cv_hybrid",
        lens_mode="none",
    )

    assert seen_backends == [(DETECTOR_BACKEND_CV_HYBRID,)]
    assert result.detected_pages == 0
    assert result.fallback_pages == 1
    assert result.pages[0].fallback_reason == "cv_hybrid: no candidate"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["detectorPolicy"] == "cv_hybrid"
    assert report["fallbackPages"] == 1
    assert report["pages"][0]["fallbackReason"] == "cv_hybrid: no candidate"
    assert "durationMs" in report["pages"][0]


def test_strict_detection_failure_preserves_existing_pdf(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    _write_image(image_path, 80)
    output_pdf = tmp_path / "strict.pdf"
    output_pdf.write_bytes(b"existing")

    def no_detection(image, **_kwargs):
        return ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result=None,
        )

    monkeypatch.setattr("uniscan.core.pipeline.scan_with_document_detector", no_detection)

    with pytest.raises(RuntimeError, match="Document detection failed"):
        run_batch_pipeline(
            inputs=[image_path],
            output_pdf=output_pdf,
            detector_policy="cv_hybrid",
            strict_detect=True,
            lens_mode="none",
        )

    assert output_pdf.read_bytes() == b"existing"


def test_cancellation_preserves_existing_pdf(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    _write_image(image_path, 80)
    output_pdf = tmp_path / "cancelled.pdf"
    output_pdf.write_bytes(b"existing")

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        run_batch_pipeline(
            inputs=[image_path],
            output_pdf=output_pdf,
            detect_document=False,
            lens_mode="none",
            cancel_cb=lambda: True,
        )

    assert output_pdf.read_bytes() == b"existing"


def test_interrupted_multi_target_publish_is_rolled_back_on_next_run(tmp_path, monkeypatch) -> None:
    from uniscan.tools import batch_pipeline

    source = tmp_path / "source.png"
    _write_image(source, 90)
    output_pdf = tmp_path / "result.pdf"
    report_path = tmp_path / "result.report.json"
    images_dir = tmp_path / "images"
    output_pdf.write_bytes(b"old-pdf")
    report_path.write_bytes(b"old-report")
    images_dir.mkdir()
    (images_dir / "notes.txt").write_text("old-images", encoding="utf-8")

    staged_pdf = tmp_path / ".result.pdf.stage-crash"
    staged_pdf.write_bytes(b"new-pdf")
    staged_images = tmp_path / ".images.stage-crash"
    staged_images.mkdir()
    (staged_images / "page_00001.png").write_bytes(b"new-image")
    staged_report = tmp_path / ".result.report.json.stage-crash"
    staged_report.write_bytes(b"new-report")
    targets = [
        batch_pipeline._StagedTarget(staged=staged_pdf, target=output_pdf),
        batch_pipeline._StagedTarget(staged=staged_images, target=images_dir),
        batch_pipeline._StagedTarget(staged=staged_report, target=report_path),
    ]
    journal = batch_pipeline._transaction_journal_path(output_pdf)
    real_replace = batch_pipeline.os.replace

    def crash_during_images_publish(source_path, target_path) -> None:
        if Path(source_path) == staged_images and Path(target_path) == images_dir:
            raise KeyboardInterrupt("simulated process termination")
        real_replace(source_path, target_path)

    monkeypatch.setattr(batch_pipeline.os, "replace", crash_during_images_publish)
    with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
        batch_pipeline._publish_staged_targets(targets, journal_path=journal)
    monkeypatch.setattr(batch_pipeline.os, "replace", real_replace)

    assert journal.is_file()
    expected_targets = batch_pipeline._transaction_target_specs(
        output_pdf=output_pdf,
        images_dir=images_dir,
        report_path=report_path,
    )
    from uniscan.export import exporters

    with exporters._directory_export_lock(images_dir):
        with pytest.raises(RuntimeError, match="Another UniScan process"):
            batch_pipeline._recover_batch_transaction(
                journal,
                expected_targets=expected_targets,
            )

    # Recovery must fail before touching any member of the partial transaction.
    assert output_pdf.read_bytes() == b"new-pdf"
    assert not images_dir.exists()
    assert report_path.read_bytes() == b"old-report"
    assert journal.is_file()

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_pdf,
            images_dir=images_dir,
            report_path=report_path,
            detect_document=False,
            lens_mode="none",
            cancel_cb=lambda: True,
        )

    assert output_pdf.read_bytes() == b"old-pdf"
    assert report_path.read_bytes() == b"old-report"
    assert (images_dir / "notes.txt").read_text(encoding="utf-8") == "old-images"
    assert not journal.exists()
    assert not list(tmp_path.glob(".*.backup-*"))


def test_invalid_transaction_journal_paths_fail_closed_before_processing(tmp_path) -> None:
    from uniscan.tools import batch_pipeline

    source = tmp_path / "source.png"
    _write_image(source, 90)
    output_pdf = tmp_path / "result.pdf"
    report_path = tmp_path / "result.report.json"
    victim = tmp_path / "victim.pdf"
    victim.write_bytes(b"personal")
    transaction_id = "a" * 32
    journal = batch_pipeline._transaction_journal_path(output_pdf)
    payload = {
        "schemaVersion": 1,
        "transactionId": transaction_id,
        "state": "prepared",
        "entries": [
            {
                "target": str(victim.resolve()),
                "staged": str(tmp_path / ".victim.pdf.stage-malicious"),
                "backup": str(tmp_path / f".victim.pdf.backup-{transaction_id}"),
                "kind": "file",
                "hadTarget": True,
            },
            {
                "target": str(report_path.resolve()),
                "staged": str(tmp_path / ".result.report.json.stage-malicious"),
                "backup": str(tmp_path / f".result.report.json.backup-{transaction_id}"),
                "kind": "file",
                "hadTarget": False,
            },
        ],
    }
    journal.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="targets do not match"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_pdf,
            report_path=report_path,
            detect_document=False,
        )

    assert victim.read_bytes() == b"personal"
    assert journal.exists()
    assert not output_pdf.exists()


def test_committed_transaction_journal_recovery_keeps_published_outputs(
    tmp_path, monkeypatch
) -> None:
    from uniscan.tools import batch_pipeline

    output_pdf = tmp_path / "result.pdf"
    report_path = tmp_path / "result.report.json"
    output_pdf.write_bytes(b"old-pdf")
    report_path.write_bytes(b"old-report")
    staged_pdf = tmp_path / ".result.pdf.stage-committed"
    staged_report = tmp_path / ".result.report.json.stage-committed"
    staged_pdf.write_bytes(b"new-pdf")
    staged_report.write_bytes(b"new-report")
    targets = [
        batch_pipeline._StagedTarget(staged=staged_pdf, target=output_pdf),
        batch_pipeline._StagedTarget(staged=staged_report, target=report_path),
    ]
    journal = batch_pipeline._transaction_journal_path(output_pdf)
    real_cleanup = batch_pipeline._cleanup_path_best_effort

    def leave_committed_journal(path: Path) -> None:
        if path == journal:
            return
        real_cleanup(path)

    monkeypatch.setattr(batch_pipeline, "_cleanup_path_best_effort", leave_committed_journal)
    batch_pipeline._publish_staged_targets(targets, journal_path=journal)
    monkeypatch.setattr(batch_pipeline, "_cleanup_path_best_effort", real_cleanup)

    assert journal.is_file()
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "committed"
    batch_pipeline._recover_batch_transaction(
        journal,
        expected_targets=batch_pipeline._transaction_target_specs(
            output_pdf=output_pdf,
            images_dir=None,
            report_path=report_path,
        ),
    )

    assert output_pdf.read_bytes() == b"new-pdf"
    assert report_path.read_bytes() == b"new-report"
    assert not journal.exists()


def test_active_transaction_lock_fails_safely(tmp_path) -> None:
    from uniscan.tools import batch_pipeline

    output_pdf = tmp_path / "result.pdf"
    report_path = tmp_path / "result.report.json"
    journal = batch_pipeline._transaction_journal_path(output_pdf)
    expected_targets = batch_pipeline._transaction_target_specs(
        output_pdf=output_pdf,
        images_dir=None,
        report_path=report_path,
    )

    with batch_pipeline._transaction_lock(journal):
        with pytest.raises(RuntimeError, match="Another UniScan process"):
            batch_pipeline._recover_batch_transaction(
                journal,
                expected_targets=expected_targets,
            )


def test_batches_with_different_pdf_journals_lock_shared_images_directory(tmp_path) -> None:
    from uniscan.tools import batch_pipeline

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    keep = images_dir / "keep.txt"
    keep.write_text("live-images", encoding="utf-8")

    first_pdf = tmp_path / "first.pdf"
    first_report = tmp_path / "first.report.json"
    first_journal = batch_pipeline._transaction_journal_path(first_pdf)
    first_targets = batch_pipeline._transaction_target_specs(
        output_pdf=first_pdf,
        images_dir=images_dir,
        report_path=first_report,
    )

    second_pdf = tmp_path / "second.pdf"
    second_report = tmp_path / "second.report.json"
    second_pdf.write_bytes(b"old-pdf")
    second_report.write_bytes(b"old-report")
    staged_pdf = tmp_path / ".second.pdf.stage-concurrent"
    staged_report = tmp_path / ".second.report.json.stage-concurrent"
    staged_images = tmp_path / ".images.stage-concurrent"
    staged_pdf.write_bytes(b"new-pdf")
    staged_report.write_bytes(b"new-report")
    staged_images.mkdir()
    (staged_images / "page_00001.png").write_bytes(b"new-image")
    second_staged_targets = [
        batch_pipeline._StagedTarget(staged=staged_pdf, target=second_pdf),
        batch_pipeline._StagedTarget(staged=staged_images, target=images_dir),
        batch_pipeline._StagedTarget(staged=staged_report, target=second_report),
    ]
    second_journal = batch_pipeline._transaction_journal_path(second_pdf)

    with batch_pipeline._transaction_output_locks(
        first_journal,
        expected_targets=first_targets,
    ):
        with pytest.raises(RuntimeError, match="Another UniScan process"):
            batch_pipeline._publish_staged_targets(
                second_staged_targets,
                journal_path=second_journal,
            )

    assert second_pdf.read_bytes() == b"old-pdf"
    assert second_report.read_bytes() == b"old-report"
    assert keep.read_text(encoding="utf-8") == "live-images"
    assert not second_journal.exists()
    assert not staged_pdf.exists()
    assert not staged_report.exists()
    assert not staged_images.exists()


def test_batch_backup_cleanup_failure_does_not_turn_commit_into_failure(
    tmp_path, monkeypatch
) -> None:
    from uniscan.tools import batch_pipeline

    source = tmp_path / "source.png"
    _write_image(source, 90)
    output_pdf = tmp_path / "result.pdf"
    report_path = tmp_path / "result.report.json"
    images_dir = tmp_path / "images"
    output_pdf.write_bytes(b"old-pdf")
    report_path.write_bytes(b"old-report")
    images_dir.mkdir()
    (images_dir / "notes.txt").write_text("personal", encoding="utf-8")
    real_remove = batch_pipeline._remove_path

    def locked_backup(path: Path) -> None:
        if ".backup-" in path.name:
            raise PermissionError("locked by scanner")
        real_remove(path)

    monkeypatch.setattr(batch_pipeline, "_remove_path", locked_backup)

    result = run_batch_pipeline(
        inputs=[source],
        output_pdf=output_pdf,
        images_dir=images_dir,
        report_path=report_path,
        detect_document=False,
        lens_mode="none",
    )

    assert result.output_pdf.read_bytes().startswith(b"%PDF")
    assert json.loads(result.report_path.read_text(encoding="utf-8"))["totalPages"] == 1
    assert (images_dir / "notes.txt").read_text(encoding="utf-8") == "personal"
    assert len(list(tmp_path.glob(".*.backup-*"))) == 3
    journal = batch_pipeline._transaction_journal_path(output_pdf)
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "committed"

    with pytest.raises(RuntimeError, match="locked debris"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_pdf,
            images_dir=images_dir,
            report_path=report_path,
            detect_document=False,
            lens_mode="none",
        )

    monkeypatch.setattr(batch_pipeline, "_remove_path", real_remove)
    with pytest.raises(RuntimeError, match="Cancelled by user"):
        run_batch_pipeline(
            inputs=[source],
            output_pdf=output_pdf,
            images_dir=images_dir,
            report_path=report_path,
            detect_document=False,
            lens_mode="none",
            cancel_cb=lambda: True,
        )
    assert not journal.exists()
    assert not list(tmp_path.glob(".*.backup-*"))
