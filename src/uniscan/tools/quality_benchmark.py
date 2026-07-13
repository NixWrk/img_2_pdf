"""Ground-truth benchmark for document crop quality and latency."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from uniscan.core.geometry import order_quad_points
from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    ScanAdapterError,
    probe_detector_backend,
    scan_with_document_detector,
)
from uniscan.io.loaders import imread_unicode

DEFAULT_QUALITY_BACKENDS = (DETECTOR_BACKEND_CV_HYBRID,)


@dataclass(slots=True, frozen=True)
class QualityPageResult:
    case_id: str
    category: str
    detected: bool
    backend: str | None
    crop_success: bool
    corner_error_px: float | None
    corner_error_ratio: float | None
    latency_ms: float


@dataclass(slots=True, frozen=True)
class QualityBackendResult:
    backend: str
    total_pages: int
    detected_pages: int
    crop_success_pages: int
    crop_success_rate: float
    fallback_rate: float
    mean_corner_error_px: float | None
    mean_corner_error_ratio: float | None
    median_latency_ms: float
    p95_latency_ms: float
    pages: tuple[QualityPageResult, ...]
    error: str | None = None


@dataclass(slots=True, frozen=True)
class QualityBenchmarkReport:
    schema_version: int
    corpus_version: str
    corner_tolerance_ratio: float
    backends: tuple[QualityBackendResult, ...]


def load_quality_manifest(corpus_dir: Path) -> dict[str, object]:
    """Load and minimally validate a versioned corpus manifest."""
    path = Path(corpus_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read quality corpus manifest: {path}: {exc}") from exc
    if payload.get("schemaVersion") != 1:
        raise ValueError("Unsupported quality corpus schema version.")
    if not isinstance(payload.get("version"), str) or not isinstance(payload.get("cases"), list):
        raise ValueError("Invalid quality corpus manifest fields.")
    return payload


def _corner_error(
    actual: np.ndarray,
    expected: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[float, float]:
    actual_quad = order_quad_points(np.asarray(actual, dtype=np.float32).reshape(4, 2))
    expected_quad = order_quad_points(np.asarray(expected, dtype=np.float32).reshape(4, 2))
    mean_error = float(np.linalg.norm(actual_quad - expected_quad, axis=1).mean())
    height, width = image_shape[:2]
    diagonal = max(1.0, float(np.hypot(width, height)))
    return mean_error, mean_error / diagonal


def _run_backend(
    *,
    corpus_dir: Path,
    cases: list[object],
    backend: str,
    corner_tolerance_ratio: float,
    scanner_root: Path | None,
    uvdoc_cache_home: Path | None,
) -> QualityBackendResult:
    probe_detector_backend(
        backend,
        scanner_root=scanner_root,
        uvdoc_cache_home=uvdoc_cache_home,
    )
    page_results: list[QualityPageResult] = []
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Quality corpus case must be an object.")
        case_id = str(raw_case.get("id", ""))
        category = str(raw_case.get("category", ""))
        image_name = str(raw_case.get("image", ""))
        expected = np.asarray(raw_case.get("corners"), dtype=np.float32)
        if not case_id or not image_name or expected.shape != (4, 2):
            raise ValueError(f"Invalid quality corpus case: {case_id or '<unnamed>'}")
        image = imread_unicode(corpus_dir / image_name)
        if image is None:
            raise ValueError(f"Cannot read quality corpus image: {image_name}")

        started = time.perf_counter()
        output = scan_with_document_detector(
            image,
            enabled=True,
            scanner_root=scanner_root,
            backends=(backend,),
            uvdoc_cache_home=uvdoc_cache_home,
            allow_dewarp_backends=backend
            in (DETECTOR_BACKEND_UVDOC, DETECTOR_BACKEND_PADDLEOCR_UVDOC),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        error_px: float | None = None
        error_ratio: float | None = None
        crop_success = bool(output.detected)
        if output.contour is not None:
            error_px, error_ratio = _corner_error(output.contour, expected, image.shape)
            crop_success = bool(output.detected and error_ratio <= corner_tolerance_ratio)
        page_results.append(
            QualityPageResult(
                case_id=case_id,
                category=category,
                detected=bool(output.detected),
                backend=output.backend,
                crop_success=crop_success,
                corner_error_px=round(error_px, 4) if error_px is not None else None,
                corner_error_ratio=round(error_ratio, 6) if error_ratio is not None else None,
                latency_ms=round(latency_ms, 3),
            )
        )

    total = len(page_results)
    detected = sum(page.detected for page in page_results)
    successes = sum(page.crop_success for page in page_results)
    corner_errors_px = [
        page.corner_error_px for page in page_results if page.corner_error_px is not None
    ]
    corner_errors_ratio = [
        page.corner_error_ratio for page in page_results if page.corner_error_ratio is not None
    ]
    latencies = [page.latency_ms for page in page_results]
    return QualityBackendResult(
        backend=backend,
        total_pages=total,
        detected_pages=detected,
        crop_success_pages=successes,
        crop_success_rate=round(successes / max(1, total), 6),
        fallback_rate=round((total - detected) / max(1, total), 6),
        mean_corner_error_px=(
            round(float(np.mean(corner_errors_px)), 4) if corner_errors_px else None
        ),
        mean_corner_error_ratio=(
            round(float(np.mean(corner_errors_ratio)), 6) if corner_errors_ratio else None
        ),
        median_latency_ms=round(float(np.median(latencies)), 3) if latencies else 0.0,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        pages=tuple(page_results),
    )


def _write_report_atomic(report: QualityBenchmarkReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.stage-",
        suffix=output_path.suffix,
        dir=output_path.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        payload: dict[str, object] = {
            "schemaVersion": report.schema_version,
            "corpusVersion": report.corpus_version,
            "cornerToleranceRatio": report.corner_tolerance_ratio,
            "backends": [
                {
                    "backend": result.backend,
                    "totalPages": result.total_pages,
                    "detectedPages": result.detected_pages,
                    "cropSuccessPages": result.crop_success_pages,
                    "cropSuccessRate": result.crop_success_rate,
                    "fallbackRate": result.fallback_rate,
                    "meanCornerErrorPx": result.mean_corner_error_px,
                    "meanCornerErrorRatio": result.mean_corner_error_ratio,
                    "medianLatencyMs": result.median_latency_ms,
                    "p95LatencyMs": result.p95_latency_ms,
                    "pages": [
                        {
                            "caseId": page.case_id,
                            "category": page.category,
                            "detected": page.detected,
                            "backend": page.backend,
                            "cropSuccess": page.crop_success,
                            "cornerErrorPx": page.corner_error_px,
                            "cornerErrorRatio": page.corner_error_ratio,
                            "latencyMs": page.latency_ms,
                        }
                        for page in result.pages
                    ],
                    "error": result.error,
                }
                for result in report.backends
            ],
        }
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_quality_benchmark(
    *,
    corpus_dir: Path,
    output_path: Path,
    backends: Sequence[str] = DEFAULT_QUALITY_BACKENDS,
    corner_tolerance_ratio: float = 0.08,
    scanner_root: Path | None = None,
    uvdoc_cache_home: Path | None = None,
) -> QualityBenchmarkReport:
    """Benchmark crop success, corner error, latency, and fallback rate."""
    if not 0.0 < corner_tolerance_ratio < 1.0:
        raise ValueError("Corner tolerance ratio must be between 0 and 1.")
    corpus_dir = Path(corpus_dir)
    manifest = load_quality_manifest(corpus_dir)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    results: list[QualityBackendResult] = []
    for backend in backends:
        try:
            result = _run_backend(
                corpus_dir=corpus_dir,
                cases=cases,
                backend=backend,
                corner_tolerance_ratio=corner_tolerance_ratio,
                scanner_root=scanner_root,
                uvdoc_cache_home=uvdoc_cache_home,
            )
        except (ScanAdapterError, RuntimeError, ValueError) as exc:
            result = QualityBackendResult(
                backend=backend,
                total_pages=0,
                detected_pages=0,
                crop_success_pages=0,
                crop_success_rate=0.0,
                fallback_rate=1.0,
                mean_corner_error_px=None,
                mean_corner_error_ratio=None,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                pages=(),
                error=str(exc),
            )
        results.append(result)
    report = QualityBenchmarkReport(
        schema_version=1,
        corpus_version=str(manifest["version"]),
        corner_tolerance_ratio=float(corner_tolerance_ratio),
        backends=tuple(results),
    )
    _write_report_atomic(report, Path(output_path))
    return report


def validate_quality_baseline(
    report: QualityBenchmarkReport,
    baseline_path: Path,
) -> list[str]:
    """Return human-readable regressions against committed baseline thresholds."""
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read quality baseline: {baseline_path}: {exc}") from exc
    thresholds = baseline.get("backends")
    if not isinstance(thresholds, dict):
        raise ValueError("Invalid quality baseline: missing backends object.")
    baseline_corpus = baseline.get("corpusVersion")
    if baseline_corpus is not None and baseline_corpus != report.corpus_version:
        raise ValueError(
            f"Quality baseline corpus {baseline_corpus} does not match {report.corpus_version}."
        )

    by_backend = {item.backend: item for item in report.backends}
    failures: list[str] = []
    for backend, raw_threshold in thresholds.items():
        if not isinstance(raw_threshold, dict):
            continue
        result = by_backend.get(backend)
        if result is None or result.error:
            failures.append(f"{backend}: benchmark unavailable")
            continue
        min_success = float(raw_threshold.get("minCropSuccessRate", 0.0))
        max_fallback = float(raw_threshold.get("maxFallbackRate", 1.0))
        max_latency = float(raw_threshold.get("maxP95LatencyMs", float("inf")))
        if result.crop_success_rate < min_success:
            failures.append(
                f"{backend}: crop success {result.crop_success_rate:.3f} < {min_success:.3f}"
            )
        if result.fallback_rate > max_fallback:
            failures.append(f"{backend}: fallback {result.fallback_rate:.3f} > {max_fallback:.3f}")
        if "maxMeanCornerErrorRatio" in raw_threshold:
            max_corner_error = float(raw_threshold["maxMeanCornerErrorRatio"])
            actual_corner = (
                "n/a"
                if result.mean_corner_error_ratio is None
                else f"{result.mean_corner_error_ratio:.3f}"
            )
            if (
                result.mean_corner_error_ratio is None
                or result.mean_corner_error_ratio > max_corner_error
            ):
                failures.append(
                    f"{backend}: mean corner error {actual_corner} > {max_corner_error:.3f}"
                )
        if result.p95_latency_ms > max_latency:
            failures.append(
                f"{backend}: p95 latency {result.p95_latency_ms:.1f} ms > {max_latency:.1f} ms"
            )
    return failures


def summarize_quality_report(report: QualityBenchmarkReport) -> str:
    """Format one concise line per detector backend."""
    lines: list[str] = []
    for result in report.backends:
        if result.error:
            lines.append(f"{result.backend}: failed - {result.error}")
            continue
        corner = (
            "n/a" if result.mean_corner_error_px is None else f"{result.mean_corner_error_px:.1f}px"
        )
        lines.append(
            f"{result.backend}: crop {result.crop_success_pages}/{result.total_pages}, "
            f"fallback {result.fallback_rate:.1%}, corners {corner}, "
            f"p95 {result.p95_latency_ms:.1f}ms"
        )
    return "\n".join(lines)
