from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uniscan.io import imread_unicode, imwrite_unicode
from uniscan.tools.geometry_candidate import run_bundled_uvdoc_candidate


def test_bundled_uvdoc_candidate_records_model_manifest_and_latency(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    image = np.full((24, 32, 3), 180, dtype=np.uint8)
    assert imwrite_unicode(corpus / "input.png", image)
    assert imwrite_unicode(corpus / "reference.png", image)
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "corpusVersion": "test-v1",
                "task": "geometry",
                "cases": [
                    {
                        "id": "case-1",
                        "input": "input.png",
                        "reference": "reference.png",
                        "output": "case-1.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("uniscan.tools.geometry_candidate.uvdoc.dewarp", lambda value: value)
    monkeypatch.setattr(
        "uniscan.tools.geometry_candidate.uvdoc.model_identity", lambda: "uvdoc:sha256:test"
    )

    metadata_path = run_bundled_uvdoc_candidate(
        corpus_dir=corpus,
        output_dir=tmp_path / "candidate",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["modelIdentity"] == "uvdoc:sha256:test"
    assert len(metadata["sourceManifestSha256"]) == 64
    assert metadata["outputs"]["case-1"]["latencyMs"] >= 0
    assert np.array_equal(imread_unicode(tmp_path / "candidate/case-1.png"), image)
