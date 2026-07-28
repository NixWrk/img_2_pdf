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


SUPPORTED_TASKS = frozenset({"geometry", "lighting", "restoration"})
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
    ssim: float
    edge_f1: float
    psnr_db: float
    psnr_score: float
    quality_score: float
    latency_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "caseId": self.case_id,
            "category": self.category,
            "outputPath": self.output_path,
            "outputSha256": self.output_sha256,
            "resizedToReference": self.resized_to_reference,
            "ssim": self.ssim,
            "edgeF1": self.edge_f1,
            "psnrDb": self.psnr_db,
            "psnrScore": self.psnr_score,
            "qualityScore": self.quality_score,
            "latencyMs": self.latency_ms,
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
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class ModelTournamentReport:
    """Complete quality-first ranking and its reproducibility metadata."""

    corpus_version: str
    task: str
    manifest_sha256: str
    metric_weights: dict[str, float]
    winner: str | None
    ranking: tuple[str, ...]
    candidates: tuple[CandidateTournamentResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "selectionPolicy": "quality-first-license-agnostic",
            "corpusVersion": self.corpus_version,
            "task": self.task,
            "manifestSha256": self.manifest_sha256,
            "metricWeights": self.metric_weights,
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
) -> CandidateTournamentResult:
    resolved_root = root.resolve(strict=False)
    metadata: dict[str, object] = {}
    try:
        if not resolved_root.is_dir():
            raise ValueError(f"Candidate output directory is missing: {resolved_root}")
        metadata = _candidate_metadata(resolved_root)
        scores: list[TournamentCaseScore] = []
        weighted_quality = 0.0
        total_weight = 0.0
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
            resized = candidate.shape[:2] != reference.shape[:2]
            if resized:
                candidate = cv2.resize(
                    candidate,
                    (reference.shape[1], reference.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            ssim = _ssim(reference, candidate)
            edge_f1 = _edge_f1(reference, candidate)
            psnr_db, psnr_score = _psnr(reference, candidate)
            quality = (
                weights["ssim"] * ssim + weights["edgeF1"] * edge_f1 + weights["psnr"] * psnr_score
            )
            case_weight = float(case.get("weight", 1.0))
            category = str(case.get("category", "default"))
            weighted_quality += quality * case_weight
            total_weight += case_weight
            category_scores.setdefault(category, []).append((quality, case_weight))
            scores.append(
                TournamentCaseScore(
                    case_id=case_id,
                    category=category,
                    output_path=str(output_path),
                    output_sha256=_sha256(output_path),
                    resized_to_reference=resized,
                    ssim=ssim,
                    edge_f1=edge_f1,
                    psnr_db=psnr_db,
                    psnr_score=psnr_score,
                    quality_score=quality,
                    latency_ms=latency,
                )
            )
        latencies = [score.latency_ms for score in scores if score.latency_ms is not None]
        categories = {
            category: sum(score * weight for score, weight in values)
            / sum(weight for _score, weight in values)
            for category, values in sorted(category_scores.items())
        }
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
            quality_score=weighted_quality / total_weight,
            mean_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
            categories=categories,
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
            cases=(),
        )


def run_model_tournament(
    *,
    corpus_dir: Path,
    output_path: Path,
    candidates: Mapping[str, Path],
) -> ModelTournamentReport:
    """Score complete candidate output sets; licenses are recorded but never scored."""
    corpus = Path(corpus_dir).resolve(strict=False)
    manifest = load_model_tournament_manifest(corpus)
    if not candidates:
        raise ValueError("At least one model tournament candidate is required.")
    task = str(manifest["task"])
    weights = _validated_metric_weights(manifest.get("metricWeights"), task)
    cases = list(manifest["cases"])
    results = tuple(
        _score_candidate(
            name,
            Path(root),
            corpus=corpus,
            cases=cases,
            weights=weights,
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
        manifest_sha256=_sha256(corpus / "manifest.json"),
        metric_weights=weights,
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
            lines.append(f"{rank}. {name}: quality={candidate.quality_score:.6f}")
    for candidate in report.candidates:
        if not candidate.eligible:
            lines.append(f"- {candidate.name}: excluded from ranking ({candidate.error})")
    return "\n".join(lines)
