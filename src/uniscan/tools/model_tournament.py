"""Reference-image tournament for outputs produced by arbitrary document models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from uniscan.io import imread_unicode
from uniscan.tools.document_metrics import (
    DOCUNET_MS_SSIM_WEIGHTS,
    aad_opencv_dis_proxy,
    docunet_ms_ssim,
    levenshtein_distance,
    tesseract_text,
    tesseract_version,
)


SUPPORTED_TASKS = frozenset({"geometry", "lighting", "restoration"})
SUPPORTED_BENCHMARK_PROFILES = frozenset(
    {"docunet-corrected", "docunet-corrected-common-128", "dir300"}
)
METRIC_NAMES = ("ssim", "edgeF1", "psnr")
DEFAULT_METRIC_WEIGHTS: dict[str, dict[str, float]] = {
    "geometry": {"ssim": 0.4, "edgeF1": 0.4, "psnr": 0.2},
    "lighting": {"ssim": 0.45, "edgeF1": 0.3, "psnr": 0.25},
    "restoration": {"ssim": 0.4, "edgeF1": 0.35, "psnr": 0.25},
}


@dataclass(frozen=True)
class TournamentCaseScore:
    """Metrics for one candidate output against one paired reference."""

    case_id: str
    category: str
    output_path: str
    output_sha256: str
    resized_to_reference: bool
    reference_size: tuple[int, int]
    candidate_original_size: tuple[int, int]
    aspect_ratio_log_error: float
    aspect_ratio_score: float
    alignment: str
    ssim: float
    edge_f1: float
    psnr_db: float
    psnr_score: float
    quality_score: float
    latency_ms: float | None
    docunet_ms_ssim: float | None
    aad_opencv_dis_proxy: float | None
    ocr_edit_distance: int | None
    ocr_cer: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "caseId": self.case_id,
            "category": self.category,
            "outputPath": self.output_path,
            "outputSha256": self.output_sha256,
            "resizedToReference": self.resized_to_reference,
            "referenceSize": list(self.reference_size),
            "candidateOriginalSize": list(self.candidate_original_size),
            "aspectRatioLogError": self.aspect_ratio_log_error,
            "aspectRatioScore": self.aspect_ratio_score,
            "alignment": self.alignment,
            "ssim": self.ssim,
            "edgeF1": self.edge_f1,
            "psnrDb": self.psnr_db,
            "psnrScore": self.psnr_score,
            "qualityScore": self.quality_score,
            "latencyMs": self.latency_ms,
            "docunetMsSsim": self.docunet_ms_ssim,
            "aadOpenCvDisProxy": self.aad_opencv_dis_proxy,
            "ocrEditDistance": self.ocr_edit_distance,
            "ocrCer": self.ocr_cer,
        }


@dataclass(frozen=True)
class CandidateTournamentResult:
    """Aggregate result for one model or processing pipeline."""

    name: str
    output_root: str
    license: str | None
    delivery: str
    model_identity: str | None
    eligible: bool
    error: str | None
    quality_score: float | None
    mean_latency_ms: float | None
    categories: dict[str, float]
    geometry_metrics: dict[str, float]
    output_set_sha256: str | None
    official_evaluation: dict[str, object] | None
    cases: tuple[TournamentCaseScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "outputRoot": self.output_root,
            "license": self.license,
            "delivery": self.delivery,
            "modelIdentity": self.model_identity,
            "eligible": self.eligible,
            "error": self.error,
            "qualityScore": self.quality_score,
            "meanLatencyMs": self.mean_latency_ms,
            "categories": self.categories,
            "geometryMetrics": self.geometry_metrics,
            "outputSetSha256": self.output_set_sha256,
            "officialEvaluation": self.official_evaluation,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class ModelTournamentReport:
    """Complete quality-first ranking and its reproducibility metadata."""

    corpus_version: str
    task: str
    manifest_sha256: str
    manifest_identity_sha256: str
    metric_weights: dict[str, float]
    benchmark_profile: dict[str, object] | None
    metric_implementations: dict[str, object]
    ocr_runtime: dict[str, object] | None
    winner: str | None
    ranking: tuple[str, ...]
    candidates: tuple[CandidateTournamentResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 3,
            "selectionPolicy": "quality-first-license-agnostic",
            "corpusVersion": self.corpus_version,
            "task": self.task,
            "manifestSha256": self.manifest_sha256,
            "manifestIdentitySha256": self.manifest_identity_sha256,
            "metricWeights": self.metric_weights,
            "benchmarkProfile": self.benchmark_profile,
            "metricImplementations": self.metric_implementations,
            "ocrRuntime": self.ocr_runtime,
            "winner": self.winner,
            "ranking": list(self.ranking),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_tournament_manifest_identity_sha256(manifest: Mapping[str, object]) -> str:
    """Hash manifest semantics while excluding local source-provenance paths."""

    def portable_provenance(value: object) -> object:
        if isinstance(value, dict):
            return {key: portable_provenance(item) for key, item in value.items() if key != "path"}
        if isinstance(value, list):
            return [portable_provenance(item) for item in value]
        return value

    payload = dict(manifest)
    if "sourceProvenance" in payload:
        payload["sourceProvenance"] = portable_provenance(payload["sourceProvenance"])
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_path(root: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"Tournament {field} must be a non-empty relative path.")
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"Tournament {field} must be relative: {relative}")
    resolved_root = root.resolve(strict=False)
    resolved = (resolved_root / raw).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Tournament {field} escapes its root: {relative}")
    return resolved


def _validated_metric_weights(payload: object, task: str) -> dict[str, float]:
    if payload is None:
        return dict(DEFAULT_METRIC_WEIGHTS[task])
    if not isinstance(payload, dict) or set(payload) != set(METRIC_NAMES):
        raise ValueError("metricWeights must contain exactly: " + ", ".join(METRIC_NAMES))
    weights: dict[str, float] = {}
    for name in METRIC_NAMES:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric weight {name} must be numeric.")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Metric weight {name} must be finite and non-negative.")
        weights[name] = weight
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Tournament metric weights must sum to 1.0.")
    return weights


def load_model_tournament_manifest(corpus_dir: Path) -> dict[str, object]:
    """Load and validate a paired-reference tournament manifest."""
    corpus = Path(corpus_dir)
    path = corpus / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model tournament manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError("Unsupported model tournament schema version.")
    corpus_version = payload.get("corpusVersion")
    task = payload.get("task")
    cases = payload.get("cases")
    if not isinstance(corpus_version, str) or not corpus_version.strip():
        raise ValueError("Tournament corpusVersion must be a non-empty string.")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Tournament task must be one of: {', '.join(sorted(SUPPORTED_TASKS))}.")
    _validated_metric_weights(payload.get("metricWeights"), str(task))
    benchmark_profile = payload.get("benchmarkProfile")
    if benchmark_profile is not None:
        if task != "geometry" or not isinstance(benchmark_profile, dict):
            raise ValueError("benchmarkProfile is supported only for geometry manifests.")
        profile_id = benchmark_profile.get("id")
        target_area = benchmark_profile.get("targetAreaPixels")
        ms_weights = benchmark_profile.get("msSsimWeights")
        if profile_id not in SUPPORTED_BENCHMARK_PROFILES:
            raise ValueError(
                "benchmarkProfile.id must be one of: "
                + ", ".join(sorted(SUPPORTED_BENCHMARK_PROFILES))
                + "."
            )
        if benchmark_profile.get("protocolVersion") != 1:
            raise ValueError("benchmarkProfile.protocolVersion must be 1.")
        if target_area != 598400:
            raise ValueError(
                "benchmarkProfile.targetAreaPixels must be the published value 598400."
            )
        if (
            not isinstance(ms_weights, list)
            or len(ms_weights) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in ms_weights
            )
            or any(
                not math.isclose(float(value), expected, abs_tol=1e-9)
                for value, expected in zip(ms_weights, DOCUNET_MS_SSIM_WEIGHTS)
            )
        ):
            raise ValueError(
                "benchmarkProfile.msSsimWeights must match the published five weights."
            )
    if not isinstance(cases, list) or not cases:
        raise ValueError("Tournament manifest must contain at least one case.")

    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Tournament case {index} must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Tournament case {index} has no non-empty id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate tournament case id: {case_id}")
        case_ids.add(case_id)
        input_path = _safe_relative_path(corpus, case.get("input"), field=f"case {case_id} input")
        reference_path = _safe_relative_path(
            corpus, case.get("reference"), field=f"case {case_id} reference"
        )
        if not input_path.is_file():
            raise ValueError(f"Tournament input is missing: {input_path}")
        if not reference_path.is_file():
            raise ValueError(f"Tournament reference is missing: {reference_path}")
        for field, file_path in (
            ("inputSha256", input_path),
            ("referenceSha256", reference_path),
        ):
            expected_sha256 = case.get(field)
            if expected_sha256 is None:
                continue
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
            ):
                raise ValueError(f"Tournament case {case_id} {field} must be a SHA-256 digest.")
            actual_sha256 = _sha256(file_path)
            if actual_sha256 != expected_sha256.lower():
                raise ValueError(
                    f"Tournament case {case_id} {field} mismatch: expected "
                    f"{expected_sha256.lower()}, got {actual_sha256}."
                )
        output = case.get("output", Path(str(case["input"])).name)
        _safe_relative_path(corpus, output, field=f"case {case_id} output")
        weight = case.get("weight", 1.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise ValueError(f"Tournament case {case_id} weight must be positive and finite.")
        category = case.get("category", "default")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Tournament case {case_id} category must be a non-empty string.")
        subsets = case.get("subsets", [])
        if (
            not isinstance(subsets, list)
            or any(not isinstance(value, str) or not value.strip() for value in subsets)
            or len(set(subsets)) != len(subsets)
        ):
            raise ValueError(f"Tournament case {case_id} subsets must be unique strings.")
    return payload


def parse_candidate_specs(specs: list[str] | tuple[str, ...]) -> dict[str, Path]:
    """Parse repeatable ``NAME=OUTPUT_DIR`` CLI values."""
    candidates: dict[str, Path] = {}
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"Invalid candidate {spec!r}; expected NAME=OUTPUT_DIR.")
        if name in candidates:
            raise ValueError(f"Duplicate tournament candidate name: {name}")
        candidates[name] = Path(raw_path)
    if not candidates:
        raise ValueError("At least one model tournament candidate is required.")
    return candidates


def _candidate_metadata(root: Path) -> dict[str, object]:
    path = root / "candidate.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read candidate metadata: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise ValueError(f"Unsupported candidate metadata schema: {path}")
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"Candidate outputs must be an object: {path}")
    return payload


def _output_set_sha256(scores: list[TournamentCaseScore]) -> str:
    digest = hashlib.sha256()
    for score in sorted(scores, key=lambda item: item.case_id):
        digest.update(score.case_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(score.output_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _official_evaluation(
    root: Path,
    *,
    manifest_sha256: str,
    manifest_identity_sha256: str,
    output_set_sha256: str,
    benchmark_profile: Mapping[str, object] | None,
) -> dict[str, object] | None:
    path = root / "official-metrics.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read official metric sidecar: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") not in (1, 2):
        raise ValueError(f"Unsupported official metric sidecar schema: {path}")
    expected_profile = benchmark_profile.get("id") if benchmark_profile else None
    if payload.get("benchmarkProfile") != expected_profile:
        raise ValueError(f"Official metric sidecar benchmarkProfile does not match: {path}")
    if payload["schemaVersion"] == 1:
        if payload.get("manifestSha256") != manifest_sha256:
            raise ValueError(f"Official metric sidecar manifest SHA-256 does not match: {path}")
    elif payload.get("manifestIdentitySha256") != manifest_identity_sha256:
        raise ValueError(f"Official metric sidecar manifest identity does not match: {path}")
    if payload.get("outputSetSha256") != output_set_sha256:
        raise ValueError(f"Official metric sidecar output-set SHA-256 does not match: {path}")
    metrics = payload.get("metrics")
    allowed = {"msSsim", "ld", "aad", "editDistance", "cer"}
    if not isinstance(metrics, dict) or not metrics or not set(metrics).issubset(allowed):
        raise ValueError(f"Official metric sidecar metrics are invalid: {path}")
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"Official metric {name} must be finite and non-negative: {path}")
    implementation = payload.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError(f"Official metric sidecar must identify its implementation: {path}")
    result = dict(payload)
    result["sidecarSha256"] = _sha256(path)
    return result


def _candidate_output(
    root: Path,
    metadata: Mapping[str, object],
    case: Mapping[str, object],
) -> tuple[Path, float | None]:
    case_id = str(case["id"])
    default_output = case.get("output", Path(str(case["input"])).name)
    outputs = metadata.get("outputs", {})
    record = outputs.get(case_id) if isinstance(outputs, dict) else None
    latency: float | None = None
    if isinstance(record, str):
        relative = record
    elif isinstance(record, dict):
        relative = record.get("path", default_output)
        raw_latency = record.get("latencyMs")
        if raw_latency is not None:
            if (
                isinstance(raw_latency, bool)
                or not isinstance(raw_latency, (int, float))
                or not math.isfinite(float(raw_latency))
                or float(raw_latency) < 0
            ):
                raise ValueError(
                    f"Candidate latency for {case_id} must be finite and non-negative."
                )
            latency = float(raw_latency)
    elif record is None:
        relative = default_output
    else:
        raise ValueError(f"Candidate output record for {case_id} must be a string or object.")
    return _safe_relative_path(root, relative, field=f"candidate output for {case_id}"), latency


def _luminance(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _align_candidate_to_reference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, bool, float, str]:
    """Fit a candidate to the reference canvas without changing its aspect ratio."""
    reference_height, reference_width = reference.shape[:2]
    candidate_height, candidate_width = candidate.shape[:2]
    reference_aspect = reference_width / float(reference_height)
    candidate_aspect = candidate_width / float(candidate_height)
    aspect_error = abs(math.log(candidate_aspect / reference_aspect))
    if candidate.shape[:2] == reference.shape[:2]:
        return candidate, False, aspect_error, "identity"

    scale = min(
        reference_width / float(candidate_width),
        reference_height / float(candidate_height),
    )
    fitted_width = min(reference_width, max(1, int(round(candidate_width * scale))))
    fitted_height = min(reference_height, max(1, int(round(candidate_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    fitted = cv2.resize(candidate, (fitted_width, fitted_height), interpolation=interpolation)

    border = np.concatenate(
        (
            fitted[0, :].reshape(-1, fitted.shape[2]),
            fitted[-1, :].reshape(-1, fitted.shape[2]),
            fitted[:, 0].reshape(-1, fitted.shape[2]),
            fitted[:, -1].reshape(-1, fitted.shape[2]),
        ),
        axis=0,
    )
    fill = np.median(border, axis=0).astype(fitted.dtype)
    aligned = np.empty((reference_height, reference_width, fitted.shape[2]), dtype=fitted.dtype)
    aligned[...] = fill
    offset_x = (reference_width - fitted_width) // 2
    offset_y = (reference_height - fitted_height) // 2
    aligned[offset_y : offset_y + fitted_height, offset_x : offset_x + fitted_width] = fitted
    return aligned, True, aspect_error, "fit-preserve-aspect"


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = _luminance(reference)
    y = _luminance(candidate)
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / np.maximum(
        denominator, np.finfo(np.float32).eps
    )
    return float(np.clip(np.mean(score), 0.0, 1.0))


def _edge_f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_edges = cv2.Canny((_luminance(reference) * 255).astype(np.uint8), 50, 150) > 0
    candidate_edges = cv2.Canny((_luminance(candidate) * 255).astype(np.uint8), 50, 150) > 0
    reference_count = int(np.count_nonzero(reference_edges))
    candidate_count = int(np.count_nonzero(candidate_edges))
    if reference_count == 0 and candidate_count == 0:
        return 1.0
    if reference_count == 0 or candidate_count == 0:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    reference_tolerance = cv2.dilate(reference_edges.astype(np.uint8), kernel) > 0
    candidate_tolerance = cv2.dilate(candidate_edges.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(candidate_edges & reference_tolerance)) / candidate_count
    recall = float(np.count_nonzero(reference_edges & candidate_tolerance)) / reference_count
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    difference = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(difference * difference))
    psnr_db = 100.0 if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return psnr_db, min(max(psnr_db, 0.0) / 50.0, 1.0)


def _score_candidate(
    name: str,
    root: Path,
    *,
    corpus: Path,
    cases: list[dict[str, object]],
    weights: Mapping[str, float],
    manifest_sha256: str,
    manifest_identity_sha256: str,
    benchmark_profile: Mapping[str, object] | None,
    tesseract_executable: str | Path | None,
    tesseract_language: str | None,
    ocr_subset: str | None,
    reference_ocr_cache: dict[Path, str],
) -> CandidateTournamentResult:
    resolved_root = root.resolve(strict=False)
    metadata: dict[str, object] = {}
    try:
        if not resolved_root.is_dir():
            raise ValueError(f"Candidate output directory is missing: {resolved_root}")
        metadata = _candidate_metadata(resolved_root)
        metadata_profile = metadata.get("benchmarkProfile")
        expected_profile = benchmark_profile.get("id") if benchmark_profile else None
        if metadata_profile is not None and metadata_profile != expected_profile:
            raise ValueError(
                f"Candidate {name} benchmark profile {metadata_profile!r} does not match "
                f"{expected_profile!r}."
            )
        scores: list[TournamentCaseScore] = []
        category_scores: dict[str, list[tuple[float, float]]] = {}
        for case in cases:
            case_id = str(case["id"])
            reference_path = _safe_relative_path(
                corpus, case["reference"], field=f"case {case_id} reference"
            )
            output_path, latency = _candidate_output(resolved_root, metadata, case)
            if not output_path.is_file():
                raise ValueError(f"Candidate {name} output is missing for {case_id}: {output_path}")
            reference = imread_unicode(reference_path)
            candidate = imread_unicode(output_path)
            if reference is None:
                raise ValueError(f"Cannot decode tournament reference: {reference_path}")
            if candidate is None:
                raise ValueError(f"Cannot decode candidate output: {output_path}")
            reference_size = (reference.shape[1], reference.shape[0])
            candidate_original_size = (candidate.shape[1], candidate.shape[0])
            candidate, resized, aspect_error, alignment = _align_candidate_to_reference(
                reference,
                candidate,
            )
            aspect_score = math.exp(-aspect_error)
            ssim = _ssim(reference, candidate)
            edge_f1 = _edge_f1(reference, candidate)
            psnr_db, psnr_score = _psnr(reference, candidate)
            profile_ms_ssim: float | None = None
            aad_proxy: float | None = None
            if benchmark_profile is not None:
                target_area = int(benchmark_profile["targetAreaPixels"])
                profile_ms_ssim = docunet_ms_ssim(reference, candidate, target_area=target_area)
                aad_proxy = aad_opencv_dis_proxy(reference, candidate, target_area=target_area)
            ocr_edit_distance: int | None = None
            ocr_cer: float | None = None
            if ocr_subset is not None and ocr_subset in case.get("subsets", []):
                if tesseract_executable is None:
                    raise ValueError("OCR subset selected without a Tesseract executable.")
                reference_text = reference_ocr_cache.get(reference_path)
                if reference_text is None:
                    reference_text = tesseract_text(
                        reference_path,
                        executable=tesseract_executable,
                        language=tesseract_language,
                    )
                    reference_ocr_cache[reference_path] = reference_text
                if not reference_text:
                    raise ValueError(f"Tesseract returned empty reference text: {reference_path}")
                candidate_text = tesseract_text(
                    output_path,
                    executable=tesseract_executable,
                    language=tesseract_language,
                )
                ocr_edit_distance = levenshtein_distance(reference_text, candidate_text)
                ocr_cer = ocr_edit_distance / len(reference_text)
            visual_quality = (
                weights["ssim"] * ssim + weights["edgeF1"] * edge_f1 + weights["psnr"] * psnr_score
            )
            quality = visual_quality * aspect_score
            case_weight = float(case.get("weight", 1.0))
            category = str(case.get("category", "default"))
            category_scores.setdefault(category, []).append((quality, case_weight))
            scores.append(
                TournamentCaseScore(
                    case_id=case_id,
                    category=category,
                    output_path=str(output_path),
                    output_sha256=_sha256(output_path),
                    resized_to_reference=resized,
                    reference_size=reference_size,
                    candidate_original_size=candidate_original_size,
                    aspect_ratio_log_error=aspect_error,
                    aspect_ratio_score=aspect_score,
                    alignment=alignment,
                    ssim=ssim,
                    edge_f1=edge_f1,
                    psnr_db=psnr_db,
                    psnr_score=psnr_score,
                    quality_score=quality,
                    latency_ms=latency,
                    docunet_ms_ssim=profile_ms_ssim,
                    aad_opencv_dis_proxy=aad_proxy,
                    ocr_edit_distance=ocr_edit_distance,
                    ocr_cer=ocr_cer,
                )
            )
        latencies = [score.latency_ms for score in scores if score.latency_ms is not None]
        categories = {
            category: sum(score * weight for score, weight in values)
            / sum(weight for _score, weight in values)
            for category, values in sorted(category_scores.items())
        }
        geometry_metrics: dict[str, float] = {}
        profile_scores = [
            score.docunet_ms_ssim for score in scores if score.docunet_ms_ssim is not None
        ]
        aad_scores = [
            score.aad_opencv_dis_proxy for score in scores if score.aad_opencv_dis_proxy is not None
        ]
        edit_distances = [
            score.ocr_edit_distance for score in scores if score.ocr_edit_distance is not None
        ]
        cer_scores = [score.ocr_cer for score in scores if score.ocr_cer is not None]
        if profile_scores:
            geometry_metrics["docunetMsSsim"] = sum(profile_scores) / len(profile_scores)
        if aad_scores:
            geometry_metrics["aadOpenCvDisProxy"] = sum(aad_scores) / len(aad_scores)
        if edit_distances:
            geometry_metrics["ocrEditDistance"] = sum(edit_distances) / len(edit_distances)
        if cer_scores:
            geometry_metrics["ocrCer"] = sum(cer_scores) / len(cer_scores)
        output_set_sha256 = _output_set_sha256(scores)
        official_evaluation = _official_evaluation(
            resolved_root,
            manifest_sha256=manifest_sha256,
            manifest_identity_sha256=manifest_identity_sha256,
            output_set_sha256=output_set_sha256,
            benchmark_profile=benchmark_profile,
        )
        return CandidateTournamentResult(
            name=name,
            output_root=str(resolved_root),
            license=str(metadata["license"]) if metadata.get("license") is not None else None,
            delivery=str(metadata.get("delivery", "external")),
            model_identity=(
                str(metadata["modelIdentity"])
                if metadata.get("modelIdentity") is not None
                else None
            ),
            eligible=True,
            error=None,
            quality_score=sum(categories.values()) / len(categories),
            mean_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
            categories=categories,
            geometry_metrics=geometry_metrics,
            output_set_sha256=output_set_sha256,
            official_evaluation=official_evaluation,
            cases=tuple(scores),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return CandidateTournamentResult(
            name=name,
            output_root=str(resolved_root),
            license=str(metadata["license"]) if metadata.get("license") is not None else None,
            delivery=str(metadata.get("delivery", "external")),
            model_identity=(
                str(metadata["modelIdentity"])
                if metadata.get("modelIdentity") is not None
                else None
            ),
            eligible=False,
            error=str(exc),
            quality_score=None,
            mean_latency_ms=None,
            categories={},
            geometry_metrics={},
            output_set_sha256=None,
            official_evaluation=None,
            cases=(),
        )


def run_model_tournament(
    *,
    corpus_dir: Path,
    output_path: Path,
    candidates: Mapping[str, Path],
    tesseract_executable: str | Path | None = None,
    tesseract_language: str | None = None,
    ocr_subset: str | None = None,
) -> ModelTournamentReport:
    """Score complete candidate output sets; licenses are recorded but never scored."""
    corpus = Path(corpus_dir).resolve(strict=False)
    manifest = load_model_tournament_manifest(corpus)
    if not candidates:
        raise ValueError("At least one model tournament candidate is required.")
    task = str(manifest["task"])
    weights = _validated_metric_weights(manifest.get("metricWeights"), task)
    cases = list(manifest["cases"])
    benchmark_profile = manifest.get("benchmarkProfile")
    if benchmark_profile is not None and not isinstance(benchmark_profile, dict):
        raise ValueError("benchmarkProfile must be an object.")
    available_subsets = sorted(
        {str(subset) for case in cases for subset in case.get("subsets", [])}
    )
    if (tesseract_executable is None) != (ocr_subset is None):
        raise ValueError("Tesseract executable and OCR subset must be provided together.")
    if ocr_subset is not None and ocr_subset not in available_subsets:
        raise ValueError(
            f"Unknown OCR subset {ocr_subset!r}; available: {', '.join(available_subsets)}"
        )
    ocr_runtime: dict[str, object] | None = None
    if tesseract_executable is not None:
        ocr_runtime = {
            "executable": str(tesseract_executable),
            "version": tesseract_version(tesseract_executable),
            "language": tesseract_language,
            "subset": ocr_subset,
            "driver": "direct-cli-v1",
        }
    manifest_sha256 = _sha256(corpus / "manifest.json")
    manifest_identity_sha256 = model_tournament_manifest_identity_sha256(manifest)
    reference_ocr_cache: dict[Path, str] = {}
    results = tuple(
        _score_candidate(
            name,
            Path(root),
            corpus=corpus,
            cases=cases,
            weights=weights,
            manifest_sha256=manifest_sha256,
            manifest_identity_sha256=manifest_identity_sha256,
            benchmark_profile=benchmark_profile,
            tesseract_executable=tesseract_executable,
            tesseract_language=tesseract_language,
            ocr_subset=ocr_subset,
            reference_ocr_cache=reference_ocr_cache,
        )
        for name, root in candidates.items()
    )
    eligible = sorted(
        (candidate for candidate in results if candidate.eligible),
        key=lambda candidate: (-float(candidate.quality_score or 0.0), candidate.name),
    )
    ranking = tuple(candidate.name for candidate in eligible)
    report = ModelTournamentReport(
        corpus_version=str(manifest["corpusVersion"]),
        task=task,
        manifest_sha256=manifest_sha256,
        manifest_identity_sha256=manifest_identity_sha256,
        metric_weights=weights,
        benchmark_profile=benchmark_profile,
        metric_implementations={
            "manifestIdentity": {
                "implementation": "canonical-json-without-provenance-paths-v1",
                "artifactHash": "manifestSha256",
                "portableHash": "manifestIdentitySha256",
            },
            "candidateAlignment": {
                "implementation": "fit-preserve-aspect-v1",
                "aspectError": "absolute-log-ratio",
                "aspectScore": "exp(-absolute-log-ratio)",
                "padding": "median-candidate-border",
            },
            "categoryAggregation": {
                "implementation": "equal-category-macro-average-v1",
                "caseWeights": "within-category",
            },
            **(
                {
                    "docunetMsSsim": {
                        "implementation": "uniscan-python-opencv-v1",
                        "publishedProtocol": True,
                        "officialMatlabComparable": False,
                        "reason": "MATLAB ssim/impyramid results vary by release.",
                    },
                    "aadOpenCvDisProxy": {
                        "implementation": "AAD equations with OpenCV DIS fast flow",
                        "officialMetric": False,
                        "reason": "Published AAD requires the official MATLAB SIFTflow implementation.",
                    },
                    "officialEvaluation": {
                        "sidecar": "official-metrics.json",
                        "integrity": "manifest and candidate output-set SHA-256 required",
                    },
                }
                if benchmark_profile is not None
                else {}
            ),
        },
        ocr_runtime=ocr_runtime,
        winner=ranking[0] if ranking else None,
        ranking=ranking,
        candidates=results,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def summarize_model_tournament(report: ModelTournamentReport) -> str:
    """Return a compact human-readable ranking without treating license as a gate."""
    lines = [
        f"Model tournament: {report.task}, corpus {report.corpus_version}",
        "Selection policy: quality-first; license metadata is informational.",
    ]
    if not report.ranking:
        lines.append("No candidate produced a complete, decodable output set.")
    else:
        for rank, name in enumerate(report.ranking, start=1):
            candidate = next(item for item in report.candidates if item.name == name)
            suffix = ""
            if candidate.geometry_metrics:
                rendered = ", ".join(
                    f"{metric}={value:.6f}" for metric, value in candidate.geometry_metrics.items()
                )
                suffix = f"; {rendered}"
            lines.append(f"{rank}. {name}: quality={candidate.quality_score:.6f}{suffix}")
    for candidate in report.candidates:
        if not candidate.eligible:
            lines.append(f"- {candidate.name}: excluded from ranking ({candidate.error})")
    return "\n".join(lines)
