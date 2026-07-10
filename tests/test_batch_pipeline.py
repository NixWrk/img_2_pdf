from __future__ import annotations

import json
from pathlib import Path

import cv2
import fitz
import numpy as np
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
    document = fitz.open()
    try:
        for index in range(page_count):
            page = document.new_page(width=100, height=100)
            page.insert_text((10, 40), f"Page {index + 1}")
        document.save(str(path))
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
    assert result.pages[0].dewarp_line_count >= 3
    assert result.pages[0].dewarp_max_displacement_px > 2.0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["dewarpMethod"] == "textline"
    assert report["pages"][0]["dewarpApplied"] is True
    assert report["pages"][0]["dewarpReason"] is None


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
