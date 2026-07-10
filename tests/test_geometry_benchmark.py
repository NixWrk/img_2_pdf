from __future__ import annotations

import json
from pathlib import Path

from uniscan.cli import main
from uniscan.tools.geometry_benchmark import (
    load_geometry_manifest,
    run_geometry_benchmark,
    summarize_geometry_report,
    validate_geometry_baseline,
)


CORPUS = Path(__file__).parents[1] / "benchmarks" / "geometry_v1"


def test_geometry_corpus_passes_committed_baseline(tmp_path) -> None:
    manifest = load_geometry_manifest(CORPUS)

    report = run_geometry_benchmark(
        corpus_dir=CORPUS,
        output_path=tmp_path / "geometry-report.json",
    )

    assert manifest["corpusVersion"] == "1.0.0"
    assert report["totalCases"] == 10
    assert report["orientationAccuracy"] == 1.0
    assert report["deskewAccuracy"] == 1.0
    assert report["dewarpAccuracy"] == 1.0
    assert validate_geometry_baseline(report, CORPUS / "baseline.json") == []
    assert "orientation 100.0%" in summarize_geometry_report(report)


def test_geometry_baseline_and_cli_report_regressions(tmp_path, monkeypatch, capsys) -> None:
    baseline = tmp_path / "strict.json"
    baseline.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "thresholds": {"maxP95LatencyMs": 1.0},
            }
        ),
        encoding="utf-8",
    )
    fake_report = {
        "orientationAccuracy": 1.0,
        "deskewAccuracy": 1.0,
        "dewarpAccuracy": 1.0,
        "p95LatencyMs": 20.0,
    }
    monkeypatch.setattr("uniscan.cli.run_geometry_benchmark", lambda **_kwargs: fake_report)

    exit_code = main(
        [
            "benchmark-geometry",
            "--input",
            str(CORPUS),
            "--output",
            str(tmp_path / "report.json"),
            "--baseline",
            str(baseline),
        ]
    )

    assert exit_code == 2
    assert "Geometry baseline regressions" in capsys.readouterr().err
