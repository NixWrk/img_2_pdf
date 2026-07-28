"""Import reproducible DocUNet and DIR300 geometry benchmark corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import cv2

from uniscan.io import imread_unicode, imwrite_unicode


DOCUNET_OCR_SETTING_1 = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 24, 27, 30, 31, 32, 36, 38, 40,
     41, 44, 45, 46, 47, 48, 50, 51, 52, 53}
)
DOCUNET_OCR_SETTING_2 = frozenset(
    {1, 9, 10, 12, 19, 20, 21, 22, 23, 24, 30, 31, 32, 34, 35, 36, 37, 38, 39,
     40, 44, 45, 46, 47, 49}
)
DIR300_OCR = frozenset(
    {
        5, 7, 8, 10, 12, 27, 28, 29, 31, 36, 53, 55, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 85, 94, 96, 103, 107, 108, 111,
        115, 126, 128, 129, 130, 133, 135, 139, 140, 148, 149, 151, 159, 160,
        161, 162, 163, 164, 165, 166, 167, 169, 170, 173, 174, 177, 201, 202,
        203, 205, 217, 218, 222, 223, 225, 227, 228, 237, 238, 239, 264, 265,
        266, 271, 273, 277, 278, 285, 286, 288, 291, 294, 295, 296, 298, 300,
    }
)


@dataclass(frozen=True)
class StandardGeometryProfile:
    """Filename and evaluation conventions for one published benchmark."""

    id: str
    case_ids: tuple[str, ...]
    input_templates: tuple[str, ...]
    reference_templates: tuple[str, ...]
    candidate_templates: tuple[str, ...]
    corrections: dict[str, str]
    subsets: dict[str, frozenset[str]]

    def document_id(self, case_id: str) -> str:
        return case_id.partition("_")[0] if self.id == "docunet-corrected" else case_id


def _docunet_case_ids() -> tuple[str, ...]:
    return tuple(f"{document}_{view}" for document in range(1, 66) for view in (1, 2))


def _docunet_subset(documents: Iterable[int]) -> frozenset[str]:
    return frozenset(f"{document}_{view}" for document in documents for view in (1, 2))


STANDARD_GEOMETRY_PROFILES: dict[str, StandardGeometryProfile] = {
    "docunet-corrected": StandardGeometryProfile(
        id="docunet-corrected",
        case_ids=_docunet_case_ids(),
        input_templates=("{case} copy.png", "{case}.png"),
        reference_templates=("{document}.png",),
        candidate_templates=(
            "{case}.png",
            "{case} copy_rec.png",
            "{case}_rec.png",
            "{case}_unwarp.png",
            "warped_{case} copy.png",
        ),
        corrections={"64_1": "rotate180", "64_2": "rotate180"},
        subsets={
            "docunet-ocr-setting-1": _docunet_subset(DOCUNET_OCR_SETTING_1),
            "docunet-ocr-setting-2": _docunet_subset(DOCUNET_OCR_SETTING_2),
        },
    ),
    "dir300": StandardGeometryProfile(
        id="dir300",
        case_ids=tuple(str(index) for index in range(1, 301)),
        input_templates=("{case}.png", "{case}.jpg", "{case}.JPG"),
        reference_templates=("{case}.png", "{case}.jpg", "{case}.JPG"),
        candidate_templates=("{case}.png", "{case}_rec.png", "{case}_unwarp.png"),
        corrections={},
        subsets={"dir300-ocr-90": frozenset(str(index) for index in DIR300_OCR)},
    ),
}

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash relative names and bytes of every file in a directory tree."""
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Geometry source is not a directory: {resolved}")
    files = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"Geometry source contains no files: {resolved}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _verified_tree(root: Path, expected_sha256: str | None) -> tuple[str, bool]:
    actual = sha256_tree(root)
    if expected_sha256 is None:
        return actual, False
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("Expected source SHA-256 must contain exactly 64 hexadecimal characters.")
    if actual != expected_sha256.lower():
        raise ValueError(
            f"Geometry source SHA-256 mismatch for {Path(root).resolve()}: "
            f"expected {expected_sha256.lower()}, got {actual}."
        )
    return actual, True


def _empty_destination(path: Path) -> Path:
    destination = Path(path).resolve(strict=False)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Geometry import destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _file_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in Path(root).resolve(strict=True).rglob("*"):
        if path.is_file():
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def _resolve_template(
    index: dict[str, list[Path]],
    templates: tuple[str, ...],
    *,
    case_id: str,
    document_id: str,
    source_label: str,
) -> Path:
    for template in templates:
        name = template.format(case=case_id, document=document_id)
        matches = index.get(name.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = ", ".join(str(path) for path in matches)
            raise ValueError(f"Ambiguous {source_label} file {name}: {rendered}")
    expected = ", ".join(
        template.format(case=case_id, document=document_id) for template in templates
    )
    raise ValueError(f"Missing {source_label} for {case_id}; tried: {expected}")


def _write_normalized_image(source: Path, destination: Path, transform: str | None = None) -> None:
    image = imread_unicode(source)
    if image is None:
        raise ValueError(f"Cannot decode geometry image: {source}")
    if transform == "rotate180":
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif transform is not None:
        raise ValueError(f"Unsupported geometry import transform: {transform}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(destination, image):
        raise ValueError(f"Cannot write normalized geometry image: {destination}")


def import_standard_geometry_corpus(
    *,
    profile_id: str,
    distorted_dir: Path,
    reference_dir: Path,
    destination_dir: Path,
    expected_distorted_sha256: str | None = None,
    expected_reference_sha256: str | None = None,
) -> Path:
    """Normalize a complete standard benchmark and write a hashed manifest."""
    try:
        profile = STANDARD_GEOMETRY_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown standard geometry profile: {profile_id}") from exc
    distorted_root = Path(distorted_dir).resolve(strict=True)
    reference_root = Path(reference_dir).resolve(strict=True)
    distorted_hash, distorted_verified = _verified_tree(
        distorted_root, expected_distorted_sha256
    )
    reference_hash, reference_verified = _verified_tree(
        reference_root, expected_reference_sha256
    )
    distorted_index = _file_index(distorted_root)
    reference_index = _file_index(reference_root)

    resolved_cases: list[tuple[str, str, Path, Path]] = []
    for case_id in profile.case_ids:
        document_id = profile.document_id(case_id)
        resolved_cases.append(
            (
                case_id,
                document_id,
                _resolve_template(
                    distorted_index,
                    profile.input_templates,
                    case_id=case_id,
                    document_id=document_id,
                    source_label="distorted input",
                ),
                _resolve_template(
                    reference_index,
                    profile.reference_templates,
                    case_id=case_id,
                    document_id=document_id,
                    source_label="reference",
                ),
            )
        )

    destination = _empty_destination(destination_dir)
    cases: list[dict[str, object]] = []
    written_references: dict[str, tuple[str, str]] = {}
    for case_id, document_id, source_input, source_reference in resolved_cases:
        input_relative = f"inputs/{case_id}.png"
        reference_record = written_references.get(document_id)
        if reference_record is None:
            reference_relative = f"references/{document_id}.png"
            _write_normalized_image(source_reference, destination / reference_relative)
            reference_record = (
                reference_relative,
                sha256_file(destination / reference_relative),
            )
            written_references[document_id] = reference_record
        reference_relative, reference_sha256 = reference_record
        transform = profile.corrections.get(case_id)
        _write_normalized_image(source_input, destination / input_relative, transform)
        subsets = sorted(name for name, members in profile.subsets.items() if case_id in members)
        cases.append(
            {
                "id": case_id,
                "category": profile.id,
                "input": input_relative,
                "inputSha256": sha256_file(destination / input_relative),
                "reference": reference_relative,
                "referenceSha256": reference_sha256,
                "output": f"{case_id}.png",
                "sourceInput": source_input.relative_to(distorted_root).as_posix(),
                "sourceReference": source_reference.relative_to(reference_root).as_posix(),
                "inputTransform": transform,
                "subsets": subsets,
            }
        )

    manifest = {
        "schemaVersion": 1,
        "corpusVersion": f"{profile.id}-v1",
        "task": "geometry",
        "metricWeights": {"ssim": 0.4, "edgeF1": 0.4, "psnr": 0.2},
        "benchmarkProfile": {
            "id": profile.id,
            "protocolVersion": 1,
            "targetAreaPixels": 598400,
            "msSsimWeights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            "knownCorrections": profile.corrections,
            "ocrSubsets": {name: len(members) for name, members in profile.subsets.items()},
        },
        "sourceProvenance": {
            "distorted": {
                "path": str(distorted_root),
                "treeSha256": distorted_hash,
                "expectedSha256Verified": distorted_verified,
            },
            "references": {
                "path": str(reference_root),
                "treeSha256": reference_hash,
                "expectedSha256Verified": reference_verified,
            },
        },
        "cases": cases,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def import_standard_geometry_candidate(
    *,
    profile_id: str,
    source_dir: Path,
    destination_dir: Path,
    name: str,
    license_name: str | None = None,
    delivery: str = "published-outputs",
    model_identity: str | None = None,
    expected_source_sha256: str | None = None,
    filename_templates: tuple[str, ...] | None = None,
) -> Path:
    """Normalize a complete published/model output directory for the tournament."""
    try:
        profile = STANDARD_GEOMETRY_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown standard geometry profile: {profile_id}") from exc
    if not name.strip():
        raise ValueError("Candidate name must not be empty.")
    source_root = Path(source_dir).resolve(strict=True)
    source_hash, source_verified = _verified_tree(source_root, expected_source_sha256)
    source_index = _file_index(source_root)
    templates = filename_templates or profile.candidate_templates
    resolved_outputs = {
        case_id: _resolve_template(
            source_index,
            templates,
            case_id=case_id,
            document_id=profile.document_id(case_id),
            source_label=f"candidate {name} output",
        )
        for case_id in profile.case_ids
    }
    destination = _empty_destination(destination_dir)
    outputs: dict[str, str] = {}
    for case_id, source in resolved_outputs.items():
        relative = f"{case_id}.png"
        _write_normalized_image(source, destination / relative)
        outputs[case_id] = relative

    metadata = {
        "schemaVersion": 1,
        "name": name,
        "license": license_name,
        "delivery": delivery,
        "modelIdentity": model_identity or f"sha256:{source_hash}",
        "benchmarkProfile": profile.id,
        "sourceProvenance": {
            "path": str(source_root),
            "treeSha256": source_hash,
            "expectedSha256Verified": source_verified,
        },
        "outputs": outputs,
    }
    metadata_path = destination / "candidate.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata_path
