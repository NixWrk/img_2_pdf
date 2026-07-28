from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uniscan.cli import main
from uniscan.io import imwrite_unicode
from uniscan.tools.model_tournament import (
    load_model_tournament_manifest,
    parse_candidate_specs,
    run_model_tournament,
)


ROOT = Path(__file__).parents[1]


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert imwrite_unicode(path, image)


def _paired_corpus(tmp_path: Path) -> tuple[Path, np.ndarray]:
    corpus = tmp_path / "corpus"
    reference = np.full((96, 128, 3), 242, dtype=np.uint8)
    cv2.rectangle(reference, (12, 10), (115, 85), (220, 220, 220), 2)
    cv2.putText(reference, "QUALITY", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.line(reference, (20, 65), (105, 65), (30, 30, 30), 2)
    _write(corpus / "inputs/source.png", reference)
    _write(corpus / "references/flat.png", reference)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "corpusVersion": "paired-test-v1",
                "task": "geometry",
                "metricWeights": {"ssim": 0.4, "edgeF1": 0.4, "psnr": 0.2},
                "cases": [
                    {
                        "id": "page-1",
                        "category": "curved",
                        "input": "inputs/source.png",
                        "reference": "references/flat.png",
                        "output": "page-1.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return corpus, reference


def test_quality_ranking_is_independent_of_license_family(tmp_path: Path) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    research = tmp_path / "research-model"
    permissive = tmp_path / "permissive-model"
    _write(research / "page-1.png", reference)
    degraded = cv2.GaussianBlur(reference, (11, 11), 3.0)
    _write(permissive / "page-1.png", degraded)
    (research / "candidate.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "license": "AGPL-3.0-only",
                "delivery": "external",
                "modelIdentity": "sha256:research-weights",
                "outputs": {"page-1": {"path": "page-1.png", "latencyMs": 42.0}},
            }
        ),
        encoding="utf-8",
    )
    (permissive / "candidate.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "license": "MIT",
                "delivery": "bundled",
                "modelIdentity": "sha256:permissive-weights",
                "outputs": {"page-1": "page-1.png"},
            }
        ),
        encoding="utf-8",
    )

    report = run_model_tournament(
        corpus_dir=corpus,
        output_path=tmp_path / "report.json",
        candidates={"research": research, "permissive": permissive},
    )

    assert report.winner == "research"
    assert report.ranking == ("research", "permissive")
    assert report.candidates[0].license == "AGPL-3.0-only"
    assert report.candidates[0].quality_score == pytest.approx(1.0)
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["selectionPolicy"] == "quality-first-license-agnostic"
    assert len(payload["candidates"][0]["cases"][0]["outputSha256"]) == 64


def test_incomplete_candidate_is_reported_but_does_not_abort_ranking(tmp_path: Path) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    complete = tmp_path / "complete"
    incomplete = tmp_path / "incomplete"
    _write(complete / "page-1.png", reference)
    incomplete.mkdir()

    report = run_model_tournament(
        corpus_dir=corpus,
        output_path=tmp_path / "report.json",
        candidates={"complete": complete, "incomplete": incomplete},
    )

    assert report.ranking == ("complete",)
    failed = next(candidate for candidate in report.candidates if candidate.name == "incomplete")
    assert not failed.eligible
    assert "output is missing" in str(failed.error)


def test_manifest_rejects_paths_outside_corpus(tmp_path: Path) -> None:
    corpus, _reference = _paired_corpus(tmp_path)
    payload = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    payload["cases"][0]["reference"] = "../outside.png"
    (corpus / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes its root"):
        load_model_tournament_manifest(corpus)


def test_candidate_specs_require_unique_name_and_path(tmp_path: Path) -> None:
    assert parse_candidate_specs([f"one={tmp_path}"]) == {"one": tmp_path}
    with pytest.raises(ValueError, match="expected NAME=OUTPUT_DIR"):
        parse_candidate_specs(["broken"])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_candidate_specs([f"one={tmp_path}", f"one={tmp_path}"])


def test_cli_benchmark_models_writes_quality_first_report(tmp_path: Path, capsys) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    candidate = tmp_path / "candidate"
    _write(candidate / "page-1.png", reference)
    report_path = tmp_path / "cli-report.json"

    code = main(
        [
            "benchmark-models",
            "--input",
            str(corpus),
            "--candidate",
            f"exact={candidate}",
            "--output",
            str(report_path),
        ]
    )

    assert code == 0
    assert "1. exact" in capsys.readouterr().out
    assert json.loads(report_path.read_text(encoding="utf-8"))["winner"] == "exact"


def test_candidate_registry_includes_restricted_models_in_quality_pool() -> None:
    registry = json.loads((ROOT / "benchmarks/model_candidates.json").read_text(encoding="utf-8"))
    candidates = {candidate["id"]: candidate for candidate in registry["candidates"]}

    assert registry["selectionPolicy"] == "quality-first-license-agnostic"
    assert candidates["dvd"]["license"] == "AGPL-3.0-only"
    assert candidates["dvd"]["status"] == "runnable-external"
    assert candidates["docscanner-l"]["priority"] == 1
    assert candidates["shadocnet"]["status"] == "runnable-external-release-assets"
    assert candidates["mmdir"]["license"] == "CC-BY-NC-ND-4.0"
