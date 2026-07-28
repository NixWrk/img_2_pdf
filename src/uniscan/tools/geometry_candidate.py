"""Generate tournament outputs with UniScan's bundled geometry model."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from uniscan.core import uvdoc
from uniscan.io import imread_unicode, imwrite_unicode
from uniscan.tools.model_tournament import load_model_tournament_manifest
from uniscan.tools.standard_geometry import sha256_file


def run_bundled_uvdoc_candidate(
    *,
    corpus_dir: Path,
    output_dir: Path,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Run the pinned CPU UVDoc graph on every input in a paired corpus."""
    corpus = Path(corpus_dir).resolve(strict=True)
    manifest = load_model_tournament_manifest(corpus)
    if manifest.get("task") != "geometry":
        raise ValueError("Bundled UVDoc candidate generation requires a geometry corpus.")
    destination = Path(output_dir).resolve(strict=False)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"UVDoc candidate destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    cases = list(manifest["cases"])
    outputs: dict[str, dict[str, object]] = {}
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        source = corpus / str(case["input"])
        image = imread_unicode(source)
        if image is None:
            raise ValueError(f"Cannot decode UVDoc candidate input: {source}")
        started = perf_counter()
        result = uvdoc.dewarp(image)
        latency_ms = (perf_counter() - started) * 1000.0
        relative = f"{case_id}.png"
        if not imwrite_unicode(destination / relative, result):
            raise ValueError(f"Cannot write UVDoc candidate output: {destination / relative}")
        outputs[case_id] = {"path": relative, "latencyMs": latency_ms}
        if on_progress is not None:
            on_progress(index, len(cases), case_id)

    profile = manifest.get("benchmarkProfile")
    metadata = {
        "schemaVersion": 1,
        "name": "uvdoc-onnx",
        "license": "MIT AND Apache-2.0",
        "delivery": "bundled",
        "modelIdentity": uvdoc.model_identity(),
        "benchmarkProfile": profile.get("id") if isinstance(profile, dict) else None,
        "sourceManifestSha256": sha256_file(corpus / "manifest.json"),
        "outputs": outputs,
    }
    path = destination / "candidate.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
