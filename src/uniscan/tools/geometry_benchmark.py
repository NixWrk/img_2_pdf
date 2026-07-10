"""Deterministic regression benchmark for the automatic geometry stages."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from uniscan.core.dewarp import DEWARP_METHOD_AUTO, dewarp_document, measure_dewarp_quality
from uniscan.core.orientation import ORIENTATION_METHOD_AUTO, orient_document
from uniscan.core.preprocess import DESKEW_METHOD_HYBRID, deskew_document
from uniscan.io import imread_unicode


def load_geometry_manifest(corpus_dir: Path) -> dict[str, object]:
    """Load and minimally validate a versioned geometry corpus manifest."""
    path = Path(corpus_dir) / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read geometry corpus manifest: {path}: {exc}") from exc
    if payload.get("schemaVersion") != 1:
        raise ValueError("Unsupported geometry corpus schema version.")
    if not isinstance(payload.get("corpusVersion"), str) or not isinstance(
        payload.get("cases"), list
    ):
        raise ValueError("Invalid geometry corpus manifest fields.")
    return payload


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run_geometry_benchmark(*, corpus_dir: Path, output_path: Path) -> dict[str, object]:
    """Run orientation, deskew, and validated dewarp against one generated corpus."""
    corpus_dir = Path(corpus_dir)
    manifest = load_geometry_manifest(corpus_dir)
    page_results: list[dict[str, object]] = []

    for raw_case in manifest["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("Invalid geometry corpus case.")
        case_id = str(raw_case.get("id", ""))
        image_name = str(raw_case.get("image", ""))
        if not case_id or not image_name:
            raise ValueError("Geometry cases require id and image fields.")
        image = imread_unicode(corpus_dir / image_name)
        if image is None:
            raise ValueError(f"Cannot read geometry corpus image: {image_name}")

        started = time.perf_counter()
        oriented, orientation = orient_document(image, method=ORIENTATION_METHOD_AUTO)
        deskewed, deskew_angle = deskew_document(oriented, method=DESKEW_METHOD_HYBRID)
        before = measure_dewarp_quality(deskewed)
        corrected, dewarp = dewarp_document(deskewed, method=DEWARP_METHOD_AUTO)
        after = measure_dewarp_quality(corrected)
        duration_ms = (time.perf_counter() - started) * 1000.0

        expected_orientation = raw_case.get("expectedOrientationAngle")
        orientation_correct = (
            None
            if expected_orientation is None
            else orientation.angle_degrees == int(expected_orientation)
        )
        expected_deskew = raw_case.get("expectedDeskewAngle")
        deskew_tolerance = float(raw_case.get("deskewTolerance", 1.5))
        deskew_correct = (
            None
            if expected_deskew is None
            else abs(float(deskew_angle) - float(expected_deskew)) <= deskew_tolerance
        )

        expectation = str(raw_case.get("dewarpExpectation", "skip"))
        if expectation == "apply":
            min_improvement = float(raw_case.get("minCurvatureImprovementRatio", 0.5))
            required_after = before.curvature_rms_px * (1.0 - min_improvement)
            dewarp_correct = bool(
                dewarp.applied
                and before.curvature_rms_px > 0.0
                and after.curvature_rms_px <= required_after
            )
        elif expectation == "noop":
            dewarp_correct = not dewarp.applied
        elif expectation == "skip":
            dewarp_correct = None
        else:
            raise ValueError(f"Unsupported dewarp expectation in {case_id}: {expectation}")

        page_results.append(
            {
                "id": case_id,
                "image": image_name,
                "orientationAngle": orientation.angle_degrees,
                "orientationConfidence": orientation.confidence,
                "orientationReason": orientation.reason,
                "orientationCorrect": orientation_correct,
                "deskewAngle": round(float(deskew_angle), 3),
                "deskewCorrect": deskew_correct,
                "dewarpExpectation": expectation,
                "dewarpApplied": dewarp.applied,
                "dewarpSelectedMethod": dewarp.selected_method,
                "dewarpReason": dewarp.reason,
                "curvatureBeforePx": before.curvature_rms_px,
                "curvatureAfterPx": after.curvature_rms_px,
                "dewarpCorrect": dewarp_correct,
                "durationMs": round(duration_ms, 3),
            }
        )

    orientation_cases = [page for page in page_results if page["orientationCorrect"] is not None]
    deskew_cases = [page for page in page_results if page["deskewCorrect"] is not None]
    dewarp_cases = [page for page in page_results if page["dewarpCorrect"] is not None]
    durations = [float(page["durationMs"]) for page in page_results]
    report: dict[str, object] = {
        "schemaVersion": 1,
        "corpusVersion": manifest["corpusVersion"],
        "totalCases": len(page_results),
        "orientationCases": len(orientation_cases),
        "orientationAccuracy": round(
            sum(page["orientationCorrect"] is True for page in orientation_cases)
            / max(1, len(orientation_cases)),
            6,
        ),
        "deskewCases": len(deskew_cases),
        "deskewAccuracy": round(
            sum(page["deskewCorrect"] is True for page in deskew_cases) / max(1, len(deskew_cases)),
            6,
        ),
        "dewarpCases": len(dewarp_cases),
        "dewarpAccuracy": round(
            sum(page["dewarpCorrect"] is True for page in dewarp_cases) / max(1, len(dewarp_cases)),
            6,
        ),
        "p95LatencyMs": round(_percentile(durations, 95.0), 3),
        "pages": page_results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate_geometry_baseline(report: dict[str, object], baseline_path: Path) -> list[str]:
    """Return human-readable failures against committed minimum geometry thresholds."""
    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read geometry baseline: {baseline_path}: {exc}") from exc
    thresholds = baseline.get("thresholds")
    if baseline.get("schemaVersion") != 1 or not isinstance(thresholds, dict):
        raise ValueError("Invalid geometry baseline.")

    failures: list[str] = []
    comparisons = (
        ("orientationAccuracy", "minOrientationAccuracy", ">="),
        ("deskewAccuracy", "minDeskewAccuracy", ">="),
        ("dewarpAccuracy", "minDewarpAccuracy", ">="),
        ("p95LatencyMs", "maxP95LatencyMs", "<="),
    )
    for report_key, threshold_key, operator in comparisons:
        if threshold_key not in thresholds:
            continue
        actual = float(report[report_key])
        expected = float(thresholds[threshold_key])
        failed = actual < expected if operator == ">=" else actual > expected
        if failed:
            failures.append(f"{report_key}={actual:.3f} must be {operator} {expected:.3f}")
    return failures


def summarize_geometry_report(report: dict[str, object]) -> str:
    """Format the high-signal geometry metrics for CLI output."""
    return (
        f"geometry: orientation {float(report['orientationAccuracy']):.1%}, "
        f"deskew {float(report['deskewAccuracy']):.1%}, "
        f"dewarp {float(report['dewarpAccuracy']):.1%}, "
        f"p95 {float(report['p95LatencyMs']):.1f}ms"
    )
