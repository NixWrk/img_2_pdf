from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uniscan.io import imread_unicode, imwrite_unicode
from uniscan.tools.document_metrics import (
    axis_aligned_distortion_from_flow,
    docunet_ms_ssim,
    levenshtein_distance,
)
from uniscan.tools.standard_geometry import (
    STANDARD_GEOMETRY_PROFILES,
    StandardGeometryProfile,
    import_standard_geometry_candidate,
    import_standard_geometry_corpus,
    sha256_tree,
)


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert imwrite_unicode(path, image)


def _small_docunet_profile(monkeypatch: pytest.MonkeyPatch) -> StandardGeometryProfile:
    profile = StandardGeometryProfile(
        id="docunet-corrected",
        case_ids=("1_1", "64_1"),
        input_templates=("{case}.png",),
        reference_templates=("{document}.png",),
        candidate_templates=("{case}_rec.png",),
        corrections={"64_1": "rotate180"},
        subsets={"docunet-ocr-setting-1": frozenset({"1_1"})},
    )
    monkeypatch.setitem(STANDARD_GEOMETRY_PROFILES, profile.id, profile)
    return profile


def test_published_dvd_common_profile_is_explicitly_limited_to_128_cases() -> None:
    profile = STANDARD_GEOMETRY_PROFILES["docunet-corrected-common-128"]

    assert len(profile.case_ids) == 128
    assert profile.case_ids[-1] == "64_2"
    assert "65_1" not in profile.case_ids
    assert profile.document_id("64_2") == "64"
    assert profile.corrections == {"64_1": "rotate180", "64_2": "rotate180"}


def test_standard_corpus_import_hashes_sources_and_applies_known_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_docunet_profile(monkeypatch)
    distorted = tmp_path / "distorted"
    references = tmp_path / "references"
    page = np.zeros((12, 16, 3), dtype=np.uint8)
    page[:4, :5] = (10, 80, 240)
    _write(distorted / "1_1.png", page)
    _write(distorted / "64_1.png", page)
    _write(references / "1.png", page)
    _write(references / "64.png", page)
    distorted_hash = sha256_tree(distorted)
    reference_hash = sha256_tree(references)

    manifest_path = import_standard_geometry_corpus(
        profile_id="docunet-corrected",
        distorted_dir=distorted,
        reference_dir=references,
        destination_dir=tmp_path / "corpus",
        expected_distorted_sha256=distorted_hash,
        expected_reference_sha256=reference_hash,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmarkProfile"]["knownCorrections"] == {"64_1": "rotate180"}
    assert manifest["sourceProvenance"]["distorted"]["expectedSha256Verified"] is True
    assert len(manifest["cases"][0]["inputSha256"]) == 64
    assert len(manifest["cases"][0]["referenceSha256"]) == 64
    corrected = imread_unicode(tmp_path / "corpus/inputs/64_1.png")
    assert np.array_equal(corrected, cv2.rotate(page, cv2.ROTATE_180))
    assert manifest["cases"][0]["subsets"] == ["docunet-ocr-setting-1"]


def test_candidate_import_normalizes_names_and_rejects_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _small_docunet_profile(monkeypatch)
    source = tmp_path / "published"
    image = np.full((10, 14, 3), 180, dtype=np.uint8)
    _write(source / "1_1_rec.png", image)
    _write(source / "64_1_rec.png", image)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_standard_geometry_candidate(
            profile_id="docunet-corrected",
            source_dir=source,
            destination_dir=tmp_path / "wrong",
            name="model",
            expected_source_sha256="0" * 64,
        )

    metadata_path = import_standard_geometry_candidate(
        profile_id="docunet-corrected",
        source_dir=source,
        destination_dir=tmp_path / "candidate",
        name="model",
        expected_source_sha256=sha256_tree(source),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["outputs"] == {"1_1": "1_1.png", "64_1": "64_1.png"}
    assert metadata["modelIdentity"].startswith("sha256:")


def test_document_geometry_metrics_have_clear_invariants() -> None:
    image = np.full((96, 128, 3), 245, dtype=np.uint8)
    cv2.putText(image, "TEXT", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # The five published weights sum to 1.0001, as does the upstream MATLAB code.
    assert docunet_ms_ssim(image, image, target_area=12_288) == pytest.approx(1.0001)
    uniform_flow = np.full((96, 128, 2), (4.0, -3.0), dtype=np.float32)
    assert axis_aligned_distortion_from_flow(image, uniform_flow) == pytest.approx(
        0.0, abs=1e-8
    )
    assert levenshtein_distance("document", "documant") == 1
