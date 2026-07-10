"""Headless input-to-PDF pipeline built from the production scanner primitives."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from uniscan.core.pipeline import PipelineOptions, process_loaded_items
from uniscan.core.orientation import ORIENTATION_METHOD_CHOICES, orient_document
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.dewarp import (
    DEWARP_METHOD_CHOICES,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
    DewarpDiagnostics,
    dewarp_document,
)
from uniscan.core.preprocess import (
    DESKEW_METHOD_CHOICES,
    PREPROCESS_PRESETS,
    apply_enhancements,
    deskew_document,
    PreprocessSettings,
    resolve_lens_mode_profile,
)
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OFFICE_LENS_ONNX,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
)
from uniscan.export import export_image_paths_as_files, export_image_paths_as_pdf
from uniscan.io import (
    IMG_EXTS,
    PDF_EXTS,
    imwrite_unicode,
    iter_input_items,
    list_supported_in_folder,
)


LENS_MODE_CHOICES = ("none", "document", "whiteboard", "photo", "b/w")
DETECTOR_POLICY_CHOICES = (
    "auto",
    "office_lens_onnx",
    "cv_hybrid",
    "opencv_quad",
    "opencv_hough",
    "opencv_minrect",
    "paddleocr_uvdoc",
)
CancelCb = Callable[[], bool]

_DETECTOR_POLICIES: dict[str, tuple[str, ...]] = {
    "auto": DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    "office_lens_onnx": (DETECTOR_BACKEND_OFFICE_LENS_ONNX,),
    "cv_hybrid": (DETECTOR_BACKEND_CV_HYBRID,),
    "opencv_quad": (DETECTOR_BACKEND_OPENCV,),
    "opencv_hough": (DETECTOR_BACKEND_OPENCV_HOUGH,),
    "opencv_minrect": (DETECTOR_BACKEND_OPENCV_MINRECT,),
    "paddleocr_uvdoc": (DETECTOR_BACKEND_PADDLEOCR_UVDOC,),
}


@dataclass(slots=True, frozen=True)
class PageRunReport:
    """Detection and timing details for one exported page."""

    index: int
    name: str
    detected: bool
    backend: str | None
    fallback_reason: str | None
    duration_ms: float
    orientation_method: str = "none"
    orientation_applied: bool = False
    orientation_angle_degrees: int = 0
    orientation_confidence: float = 0.0
    orientation_reason: str | None = None
    deskew_method: str = "none"
    deskew_angle_degrees: float = 0.0
    dewarp_method: str = "none"
    dewarp_applied: bool = False
    dewarp_line_count: int = 0
    dewarp_max_displacement_px: float = 0.0
    dewarp_reason: str | None = None


@dataclass(slots=True, frozen=True)
class BatchPipelineResult:
    """Summary of one completed headless conversion."""

    output_pdf: Path
    report_path: Path
    input_files: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    total_pages: int
    detected_pages: int
    fallback_pages: int
    pages: tuple[PageRunReport, ...]


@dataclass(slots=True, frozen=True)
class _StagedTarget:
    staged: Path
    target: Path


def resolve_input_paths(inputs: Sequence[Path], *, output_pdf: Path) -> tuple[Path, ...]:
    """Expand files and folders while preserving argument and natural folder order."""
    if not inputs:
        raise ValueError("At least one input file or folder is required.")

    output_resolved = output_pdf.with_suffix(".pdf").resolve()
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")

        if path.is_dir():
            candidates = list_supported_in_folder(path)
        elif path.is_file():
            if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS):
                raise ValueError(f"Unsupported input: {path}")
            if path.resolve() == output_resolved:
                raise ValueError("Output PDF cannot also be an explicit input file.")
            candidates = [path]
        else:
            raise ValueError(f"Input is neither a file nor a folder: {path}")

        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if candidate_resolved == output_resolved or candidate_resolved in seen:
                continue
            seen.add(candidate_resolved)
            resolved.append(candidate)

    if not resolved:
        raise ValueError("No supported image or PDF inputs were found.")
    return tuple(resolved)


def _resolve_processing(mode: str):
    normalized = mode.strip().lower()
    if normalized == "none":
        return POSTPROCESSING_OPTIONS["None"], None

    profiles_by_key = {
        name.lower(): profile
        for name, profile in (
            (name, resolve_lens_mode_profile(name))
            for name in ("Document", "Whiteboard", "Photo", "B/W")
        )
    }
    profile = profiles_by_key.get(normalized)
    if profile is None:
        raise ValueError(f"Unsupported lens mode: {mode}")
    return POSTPROCESSING_OPTIONS[profile.postprocess_name], PREPROCESS_PRESETS[profile.preset_name]


def _resolve_detector_policy(policy: str) -> tuple[str, ...]:
    try:
        return _DETECTOR_POLICIES[policy.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported detector policy: {policy}") from exc


def _validate_output_targets(
    *,
    output_pdf: Path,
    report_path: Path,
    images_dir: Path | None,
    input_files: Sequence[Path],
) -> None:
    output_resolved = output_pdf.resolve()
    report_resolved = report_path.resolve()
    if output_resolved == report_resolved:
        raise ValueError("PDF output and JSON report must use different paths.")
    if images_dir is None:
        return

    images_resolved = images_dir.resolve()
    if images_dir.exists() and not images_dir.is_dir():
        raise ValueError("Images output path exists and is not a directory.")
    if output_resolved.is_relative_to(images_resolved):
        raise ValueError("PDF output cannot be inside the replaceable images directory.")
    if report_resolved.is_relative_to(images_resolved):
        raise ValueError("JSON report cannot be inside the replaceable images directory.")
    if any(path.resolve().is_relative_to(images_resolved) for path in input_files):
        raise ValueError("Images output directory cannot contain input files.")


def _new_stage_file(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.stage-",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_staged_targets(targets: Sequence[_StagedTarget]) -> None:
    """Publish files/directories with rollback if any replacement fails."""
    published: list[tuple[Path, Path | None]] = []
    try:
        for item in targets:
            item.target.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if item.target.exists():
                backup = item.target.with_name(f".{item.target.name}.backup-{uuid.uuid4().hex}")
                os.replace(item.target, backup)
            try:
                os.replace(item.staged, item.target)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, item.target)
                raise
            published.append((item.target, backup))
    except Exception:
        for target, backup in reversed(published):
            _remove_path(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    else:
        for _target, backup in published:
            if backup is not None:
                _remove_path(backup)
    finally:
        for item in targets:
            if item.staged.exists():
                _remove_path(item.staged)


def _report_payload(
    *,
    output_pdf: Path,
    image_outputs: Sequence[Path],
    input_files: Sequence[Path],
    pages: Sequence[PageRunReport],
    detect_document: bool,
    detector_policy: str,
    illumination_correction: bool,
    orientation_method: str,
    deskew_method: str,
    dewarp_method: str,
) -> dict[str, object]:
    detected_pages = sum(page.detected for page in pages)
    fallback_pages = sum(page.fallback_reason is not None for page in pages)
    return {
        "schemaVersion": 1,
        "outputPdf": str(output_pdf),
        "imageOutputs": [str(path) for path in image_outputs],
        "inputFiles": [str(path) for path in input_files],
        "detectionEnabled": detect_document,
        "detectorPolicy": detector_policy,
        "illuminationCorrection": illumination_correction,
        "orientationMethod": orientation_method,
        "deskewMethod": deskew_method,
        "dewarpMethod": dewarp_method,
        "totalPages": len(pages),
        "detectedPages": detected_pages,
        "fallbackPages": fallback_pages,
        "pages": [
            {
                "index": page.index,
                "name": page.name,
                "detected": page.detected,
                "backend": page.backend,
                "fallbackReason": page.fallback_reason,
                "durationMs": page.duration_ms,
                "orientationMethod": page.orientation_method,
                "orientationApplied": page.orientation_applied,
                "orientationAngleDegrees": page.orientation_angle_degrees,
                "orientationConfidence": page.orientation_confidence,
                "orientationReason": page.orientation_reason,
                "deskewMethod": page.deskew_method,
                "deskewAngleDegrees": page.deskew_angle_degrees,
                "dewarpMethod": page.dewarp_method,
                "dewarpApplied": page.dewarp_applied,
                "dewarpLineCount": page.dewarp_line_count,
                "dewarpMaxDisplacementPx": page.dewarp_max_displacement_px,
                "dewarpReason": page.dewarp_reason,
            }
            for page in pages
        ],
    }


def _stage_outputs(
    *,
    staged_page_paths: Sequence[Path],
    output_pdf: Path,
    images_dir: Path | None,
    image_format: str,
    report_path: Path,
    report_payload: dict[str, object],
    dpi: int,
) -> tuple[list[_StagedTarget], tuple[Path, ...]]:
    """Prepare every output beside its target and clean all stages on failure."""
    targets: list[_StagedTarget] = []
    final_image_paths: tuple[Path, ...] = ()
    try:
        staged_pdf = _new_stage_file(output_pdf)
        targets.append(_StagedTarget(staged=staged_pdf, target=output_pdf))
        export_image_paths_as_pdf(staged_page_paths, out_pdf=staged_pdf, dpi=dpi)

        if images_dir is not None:
            images_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_images_dir = Path(
                tempfile.mkdtemp(prefix=f".{images_dir.name}.stage-", dir=images_dir.parent)
            )
            targets.append(_StagedTarget(staged=staged_images_dir, target=images_dir))
            staged_images = export_image_paths_as_files(
                staged_page_paths,
                output_dir=staged_images_dir,
                ext=image_format,
                base_name="page",
            )
            final_image_paths = tuple(images_dir / path.name for path in staged_images)

        report_payload["imageOutputs"] = [str(path) for path in final_image_paths]
        staged_report = _new_stage_file(report_path)
        targets.append(_StagedTarget(staged=staged_report, target=report_path))
        staged_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return targets, final_image_paths
    except Exception:
        for item in targets:
            if item.staged.exists():
                _remove_path(item.staged)
        raise


def run_batch_pipeline(
    *,
    inputs: Sequence[Path],
    output_pdf: Path,
    images_dir: Path | None = None,
    image_format: str = "png",
    report_path: Path | None = None,
    pdf_dpi: int = 300,
    detect_document: bool = True,
    detector_policy: str = "auto",
    strict_detect: bool = False,
    two_page_mode: bool = False,
    lens_mode: str = "document",
    illumination_correction: bool = False,
    orientation_method: str = "none",
    deskew_method: str = "none",
    dewarp_method: str = "none",
    uvdoc_cache_home: Path | None = None,
    cancel_cb: CancelCb | None = None,
) -> BatchPipelineResult:
    """Run the complete streaming pre-OCR pipeline and atomically publish its outputs."""
    dpi = int(pdf_dpi)
    if dpi < 72:
        raise ValueError("PDF DPI must be >= 72.")
    if strict_detect and not detect_document:
        raise ValueError("strict_detect cannot be used when document detection is disabled.")
    orientation_method = orientation_method.strip().lower()
    deskew_method = deskew_method.strip().lower()
    dewarp_method = dewarp_method.strip().lower()
    if orientation_method not in ORIENTATION_METHOD_CHOICES:
        raise ValueError(f"Unsupported orientation method: {orientation_method}")
    if deskew_method not in DESKEW_METHOD_CHOICES:
        raise ValueError(f"Unsupported deskew method: {deskew_method}")
    if dewarp_method not in DEWARP_METHOD_CHOICES:
        raise ValueError(f"Unsupported dewarp method: {dewarp_method}")

    output_pdf = Path(output_pdf).with_suffix(".pdf")
    report_path = Path(report_path) if report_path else output_pdf.with_suffix(".pdf.report.json")
    input_files = resolve_input_paths(inputs, output_pdf=output_pdf)
    images_dir = Path(images_dir) if images_dir is not None else None
    _validate_output_targets(
        output_pdf=output_pdf,
        report_path=report_path,
        images_dir=images_dir,
        input_files=input_files,
    )

    postprocess, preprocess_settings = _resolve_processing(lens_mode)
    if preprocess_settings is not None:
        preprocess_settings = replace(
            preprocess_settings,
            correct_illumination=bool(illumination_correction),
        )
    elif illumination_correction:
        preprocess_settings = PreprocessSettings(correct_illumination=True)
    detector_backends = _resolve_detector_policy(detector_policy)
    options = PipelineOptions(
        detect_document=bool(detect_document),
        two_page_mode=bool(two_page_mode),
        postprocess_name="None",
        detector_backends=detector_backends,
        strict_detect=bool(strict_detect),
    )

    staged_targets: list[_StagedTarget] = []
    page_reports: list[PageRunReport] = []
    final_image_paths: tuple[Path, ...] = ()
    with tempfile.TemporaryDirectory(prefix="uniscan_pages_") as tmp:
        page_stage_dir = Path(tmp)
        staged_page_paths: list[Path] = []

        for loaded_item in iter_input_items(
            input_files,
            pdf_dpi=dpi,
            cancel_cb=cancel_cb,
        ):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Cancelled by user.")
            started = time.perf_counter()
            page_results = process_loaded_items(
                [loaded_item],
                options=options,
                uvdoc_cache_home=uvdoc_cache_home,
                cancel_cb=cancel_cb,
            )
            for page in page_results:
                oriented, orientation_diagnostics = orient_document(
                    page.current,
                    method=orientation_method,
                )
                deskewed, deskew_angle = deskew_document(oriented, method=deskew_method)
                if (
                    dewarp_method == DEWARP_METHOD_PADDLEOCR_UVDOC
                    and page.backend == DETECTOR_BACKEND_PADDLEOCR_UVDOC
                ):
                    dewarped = deskewed
                    dewarp_diagnostics = DewarpDiagnostics(
                        method=dewarp_method,
                        applied=True,
                        reason="applied_by_detection_backend",
                    )
                else:
                    dewarped, dewarp_diagnostics = dewarp_document(
                        deskewed,
                        method=dewarp_method,
                        uvdoc_cache_home=uvdoc_cache_home,
                    )
                current = postprocess(dewarped)
                if preprocess_settings is not None:
                    current = apply_enhancements(current, preprocess_settings)
                page_path = page_stage_dir / f"{len(staged_page_paths) + 1:05d}.png"
                if not imwrite_unicode(page_path, current):
                    raise RuntimeError(f"Failed to write processed page: {page_path}")
                staged_page_paths.append(page_path)
                page_reports.append(
                    PageRunReport(
                        index=len(staged_page_paths),
                        name=page.name,
                        detected=page.detected,
                        backend=page.backend,
                        fallback_reason=page.fallback_reason,
                        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                        orientation_method=orientation_method,
                        orientation_applied=orientation_diagnostics.applied,
                        orientation_angle_degrees=orientation_diagnostics.angle_degrees,
                        orientation_confidence=orientation_diagnostics.confidence,
                        orientation_reason=orientation_diagnostics.reason,
                        deskew_method=deskew_method,
                        deskew_angle_degrees=round(float(deskew_angle), 3),
                        dewarp_method=dewarp_method,
                        dewarp_applied=dewarp_diagnostics.applied,
                        dewarp_line_count=dewarp_diagnostics.line_count,
                        dewarp_max_displacement_px=dewarp_diagnostics.max_displacement_px,
                        dewarp_reason=dewarp_diagnostics.reason,
                    )
                )

        if not staged_page_paths:
            raise ValueError("The input did not produce any pages.")

        report_payload = _report_payload(
            output_pdf=output_pdf,
            image_outputs=(),
            input_files=input_files,
            pages=page_reports,
            detect_document=bool(detect_document),
            detector_policy="disabled" if not detect_document else detector_policy,
            illumination_correction=bool(illumination_correction),
            orientation_method=orientation_method,
            deskew_method=deskew_method,
            dewarp_method=dewarp_method,
        )
        staged_targets, final_image_paths = _stage_outputs(
            staged_page_paths=staged_page_paths,
            output_pdf=output_pdf,
            images_dir=images_dir,
            image_format=image_format,
            report_path=report_path,
            report_payload=report_payload,
            dpi=dpi,
        )
        _publish_staged_targets(staged_targets)

    detected_pages = sum(page.detected for page in page_reports)
    fallback_pages = sum(page.fallback_reason is not None for page in page_reports)
    return BatchPipelineResult(
        output_pdf=output_pdf,
        report_path=report_path,
        input_files=input_files,
        image_outputs=final_image_paths,
        total_pages=len(page_reports),
        detected_pages=detected_pages,
        fallback_pages=fallback_pages,
        pages=tuple(page_reports),
    )
