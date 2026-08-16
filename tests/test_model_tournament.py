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
    model_tournament_manifest_identity_sha256,
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
    assert payload["schemaVersion"] == 3
    assert payload["selectionPolicy"] == "quality-first-license-agnostic"
    assert len(payload["manifestSha256"]) == 64
    assert len(payload["manifestIdentitySha256"]) == 64
    case = payload["candidates"][0]["cases"][0]
    assert len(case["outputSha256"]) == 64
    assert case["referenceSize"] == [128, 96]
    assert case["candidateOriginalSize"] == [128, 96]
    assert case["aspectRatioScore"] == pytest.approx(1.0)
    assert case["alignment"] == "identity"


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


def test_manifest_rejects_changed_hash_bound_corpus_image(tmp_path: Path) -> None:
    corpus, _reference = _paired_corpus(tmp_path)
    payload = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    payload["cases"][0]["referenceSha256"] = "0" * 64
    (corpus / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="referenceSha256 mismatch"):
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
    assert candidates["dvd"]["status"] == "isolated-gpu-spike"
    assert candidates["docscanner-l"]["priority"] == 1
    assert candidates["shadocnet"]["status"] == "runnable-external-release-assets"
    assert candidates["mmdir"]["license"] == "CC-BY-NC-ND-4.0"


def test_manifest_accepts_explicit_dvd_common_docunet_profile(tmp_path: Path) -> None:
    corpus, _reference = _paired_corpus(tmp_path)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmarkProfile"] = {
        "id": "docunet-corrected-common-128",
        "protocolVersion": 1,
        "targetAreaPixels": 598400,
        "msSsimWeights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_model_tournament_manifest(corpus)

    assert loaded["benchmarkProfile"]["id"] == "docunet-corrected-common-128"


def test_standard_profile_adds_geometry_metrics_and_hash_bound_official_sidecar(
    tmp_path: Path,
) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmarkProfile"] = {
        "id": "docunet-corrected",
        "protocolVersion": 1,
        "targetAreaPixels": 598400,
        "msSsimWeights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = tmp_path / "candidate"
    _write(candidate / "page-1.png", reference)
    report_path = tmp_path / "report.json"

    first = run_model_tournament(
        corpus_dir=corpus,
        output_path=report_path,
        candidates={"exact": candidate},
    )
    result = first.candidates[0]
    assert result.geometry_metrics["docunetMsSsim"] == pytest.approx(1.0001)
    assert result.geometry_metrics["aadOpenCvDisProxy"] == pytest.approx(0.0)
    assert result.output_set_sha256 is not None

    (candidate / "official-metrics.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "benchmarkProfile": "docunet-corrected",
                "manifestSha256": first.manifest_sha256,
                "outputSetSha256": result.output_set_sha256,
                "implementation": {"matlab": "R2019a", "flow": "official SIFTflow"},
                "metrics": {"msSsim": 1.0, "ld": 0.0, "aad": 0.0},
            }
        ),
        encoding="utf-8",
    )
    second = run_model_tournament(
        corpus_dir=corpus,
        output_path=report_path,
        candidates={"exact": candidate},
    )
    assert second.candidates[0].official_evaluation is not None
    assert second.candidates[0].official_evaluation["metrics"]["ld"] == 0.0
    assert len(second.candidates[0].official_evaluation["sidecarSha256"]) == 64


def test_manifest_identity_and_schema2_sidecar_are_portable_across_locations(
    tmp_path: Path,
) -> None:
    corpora = []
    reference = None
    for index, source_path in enumerate(("C:/downloads/docunet", "D:/datasets/docunet")):
        parent = tmp_path / f"location-{index}"
        parent.mkdir()
        corpus, current_reference = _paired_corpus(parent)
        reference = current_reference
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sourceProvenance"] = {"distorted": {"path": source_path, "treeSha256": "a" * 64}}
        manifest_path.write_text(json.dumps(manifest, indent=index), encoding="utf-8")
        corpora.append(corpus)

    assert reference is not None
    candidate = tmp_path / "portable-candidate"
    _write(candidate / "page-1.png", reference)
    first = run_model_tournament(
        corpus_dir=corpora[0],
        output_path=tmp_path / "first-report.json",
        candidates={"exact": candidate},
    )
    first_result = first.candidates[0]
    assert first_result.output_set_sha256 is not None

    sidecar_path = candidate / "official-metrics.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "benchmarkProfile": None,
                "manifestIdentitySha256": first.manifest_identity_sha256,
                "outputSetSha256": first_result.output_set_sha256,
                "implementation": {"evaluator": "portable-test"},
                "metrics": {"msSsim": 1.0},
            }
        ),
        encoding="utf-8",
    )
    second = run_model_tournament(
        corpus_dir=corpora[1],
        output_path=tmp_path / "second-report.json",
        candidates={"exact": candidate},
    )

    first_manifest = load_model_tournament_manifest(corpora[0])
    second_manifest = load_model_tournament_manifest(corpora[1])
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.manifest_identity_sha256 == second.manifest_identity_sha256
    assert first.manifest_identity_sha256 == model_tournament_manifest_identity_sha256(
        first_manifest
    )
    assert first.manifest_identity_sha256 == model_tournament_manifest_identity_sha256(
        second_manifest
    )
    assert second.candidates[0].official_evaluation is not None

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["manifestIdentitySha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    rejected = run_model_tournament(
        corpus_dir=corpora[1],
        output_path=tmp_path / "rejected-report.json",
        candidates={"exact": candidate},
    )
    assert rejected.candidates[0].eligible is False
    assert "manifest identity does not match" in str(rejected.candidates[0].error)


def test_ocr_subset_requires_tesseract_and_known_subset(tmp_path: Path) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["subsets"] = ["ocr-test"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = tmp_path / "candidate"
    _write(candidate / "page-1.png", reference)

    with pytest.raises(ValueError, match="provided together"):
        run_model_tournament(
            corpus_dir=corpus,
            output_path=tmp_path / "report.json",
            candidates={"exact": candidate},
            ocr_subset="ocr-test",
        )


def test_wrong_aspect_ratio_is_preserved_and_penalized(tmp_path: Path) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    exact = tmp_path / "exact"
    wrong_aspect = tmp_path / "wrong-aspect"
    _write(exact / "page-1.png", reference)
    _write(
        wrong_aspect / "page-1.png",
        cv2.resize(reference, (96, 96), interpolation=cv2.INTER_AREA),
    )

    report = run_model_tournament(
        corpus_dir=corpus,
        output_path=tmp_path / "report.json",
        candidates={"exact": exact, "wrong-aspect": wrong_aspect},
    )

    result = next(candidate for candidate in report.candidates if candidate.name == "wrong-aspect")
    case = result.cases[0]
    assert report.winner == "exact"
    assert case.reference_size == (128, 96)
    assert case.candidate_original_size == (96, 96)
    assert case.aspect_ratio_log_error == pytest.approx(abs(np.log(0.75)))
    assert case.aspect_ratio_score == pytest.approx(0.75)
    assert case.alignment == "fit-preserve-aspect"
    assert case.resized_to_reference is True
    assert case.quality_score < 0.55


def test_quality_score_macro_averages_categories(tmp_path: Path) -> None:
    corpus, reference = _paired_corpus(tmp_path)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"] = [
        {
            "id": "easy-1",
            "category": "easy",
            "input": "inputs/source.png",
            "reference": "references/flat.png",
            "output": "easy-1.png",
        },
        {
            "id": "easy-2",
            "category": "easy",
            "input": "inputs/source.png",
            "reference": "references/flat.png",
            "output": "easy-2.png",
        },
        {
            "id": "hard-1",
            "category": "hard",
            "input": "inputs/source.png",
            "reference": "references/flat.png",
            "output": "hard-1.png",
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = tmp_path / "candidate"
    _write(candidate / "easy-1.png", reference)
    _write(candidate / "easy-2.png", reference)
    _write(candidate / "hard-1.png", np.full_like(reference, 255))

    report = run_model_tournament(
        corpus_dir=corpus,
        output_path=tmp_path / "report.json",
        candidates={"mixed": candidate},
    )

    result = report.candidates[0]
    macro_average = sum(result.categories.values()) / len(result.categories)
    case_average = sum(case.quality_score for case in result.cases) / len(result.cases)
    assert result.categories["easy"] == pytest.approx(1.0)
    assert result.quality_score == pytest.approx(macro_average)
    assert result.quality_score != pytest.approx(case_average)
    assert report.metric_implementations["categoryAggregation"]["caseWeights"] == "within-category"
