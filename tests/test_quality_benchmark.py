from __future__ import annotations

import json

import cv2
import numpy as np

from uniscan.cli import main
from uniscan.core.scanner_adapter import ScanOutput
from uniscan.tools.quality_benchmark import (
    run_quality_benchmark,
    validate_quality_baseline,
)


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    cv2.rectangle(image, (10, 15), (105, 90), (20, 20, 20), 2)
    cv2.imwrite(str(corpus / "case.png"), image)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "version": "test-1",
                "cases": [
                    {
                        "id": "case",
                        "category": "document",
                        "image": "case.png",
                        "corners": [[10, 15], [105, 15], [105, 90], [10, 90]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return corpus


def test_quality_benchmark_measures_corners_and_writes_report(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    output = tmp_path / "report.json"

    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.probe_detector_backend", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.scan_with_document_detector",
        lambda image, **_kwargs: ScanOutput(
            warped=image,
            contour=np.float32([[11, 16], [104, 16], [104, 89], [11, 89]]),
            backend="fake",
            detected=True,
            raw_result=None,
        ),
    )

    report = run_quality_benchmark(
        corpus_dir=corpus,
        output_path=output,
        backends=("fake",),
        corner_tolerance_ratio=0.05,
    )

    result = report.backends[0]
    assert result.detected_pages == 1
    assert result.crop_success_rate == 1.0
    assert result.fallback_rate == 0.0
    assert result.mean_corner_error_px is not None
    assert result.mean_corner_error_px < 2.0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["corpusVersion"] == "test-1"
    assert payload["backends"][0]["cropSuccessRate"] == 1.0
    assert payload["backends"][0]["pages"][0]["cornerErrorPx"] < 2.0


def test_quality_baseline_reports_regressions(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.probe_detector_backend", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.scan_with_document_detector",
        lambda image, **_kwargs: ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result=None,
        ),
    )
    report = run_quality_benchmark(
        corpus_dir=corpus,
        output_path=tmp_path / "report.json",
        backends=("fake",),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"backends": {"fake": {"minCropSuccessRate": 0.5, "maxFallbackRate": 0.25}}}),
        encoding="utf-8",
    )

    failures = validate_quality_baseline(report, baseline)

    assert len(failures) == 2


def test_cli_quality_benchmark_returns_failure_for_baseline_regression(
    tmp_path, monkeypatch, capsys
) -> None:
    corpus = _corpus(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"backends": {"fake": {"minCropSuccessRate": 1.0, "maxFallbackRate": 0.0}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.probe_detector_backend", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "uniscan.tools.quality_benchmark.scan_with_document_detector",
        lambda image, **_kwargs: ScanOutput(
            warped=image,
            contour=None,
            backend=None,
            detected=False,
            raw_result=None,
        ),
    )

    code = main(
        [
            "benchmark-quality",
            "--input",
            str(corpus),
            "--output",
            str(tmp_path / "report.json"),
            "--backends",
            "fake",
            "--baseline",
            str(baseline),
        ]
    )

    assert code == 2
    assert "regressions" in capsys.readouterr().err
