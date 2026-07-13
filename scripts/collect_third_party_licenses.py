"""Fail-closed license gate for the Windows portable distribution."""

from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    distributions,
)
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tomllib

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DISTRIBUTION = canonicalize_name("uniscan")
FORBIDDEN_DISTRIBUTIONS = {"fitz", "onnxruntime", "pymupdf"}
LICENSE_MARKERS = ("license", "copying", "notice")
NOTICE_SUFFIXES = {"", ".htm", ".html", ".ijg", ".md", ".rst", ".txt"}
NATIVE_SUFFIXES = {".dll", ".dylib", ".exe", ".pyd", ".so"}

# Canonical SPDX identifiers reviewed for redistribution in the portable ZIP.
# GPL/AGPL/SSPL/BUSL are intentionally absent and rejected explicitly below.
ALLOWED_RUNTIME_LICENSE_IDS = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "MIT",
        "MIT-CMU",
        "MPL-2.0",
        "PSF-2.0",
        "Zlib",
    }
)
ALLOWED_BUILD_LICENSE_IDS = ALLOWED_RUNTIME_LICENSE_IDS | {
    "LicenseRef-PyInstaller-GPL-Exception",
    "LicenseRef-PyInstaller-Hooks-Build-Only",
}
DENIED_LICENSE_PREFIXES = ("AGPL-", "BUSL-", "GPL-", "SSPL-")

# These projects publish missing, non-SPDX, or contradictory metadata. Overrides
# are version-specific so an upstream release cannot silently inherit an older
# legal review. Adding a version here requires reviewing its shipped notices.
LICENSE_OVERRIDES = {
    "customtkinter": {"5.2.2": "MIT"},
    "img2pdf": {"0.6.3": "LGPL-3.0-or-later"},
    "opencv-python": {"4.13.0.92": "Apache-2.0"},
    "pyinstaller": {"6.21.0": "LicenseRef-PyInstaller-GPL-Exception"},
    "pyinstaller-hooks-contrib": {
        "2026.6": "LicenseRef-PyInstaller-Hooks-Build-Only",
    },
    "pypdfium2": {"5.11.0": "BSD-3-Clause AND Apache-2.0 AND CC-BY-4.0"},
    "tkinterdnd2": {"0.6.2": "MIT"},
}

RUNTIME_NOTICE_FILENAMES = {
    "python": "PYTHON-PSF-LICENSE.txt",
    "tcl": "TCL-LICENSE.txt",
    "tk": "TK-LICENSE.txt",
}
TCL_TK_COPYRIGHT_MARKER = "This software is copyrighted by the Regents of"
ROBOTO_ASSET_PREFIX = "customtkinter/assets/fonts/roboto/"
ROBOTO_ASSET_DESTINATIONS = frozenset(
    {
        f"{ROBOTO_ASSET_PREFIX}roboto-medium.ttf",
        f"{ROBOTO_ASSET_PREFIX}roboto-regular.ttf",
    }
)
CUSTOMTKINTER_SHAPES_ASSET_DESTINATION = "customtkinter/assets/fonts/customtkinter_shapes_font.otf"
ROBOTO_NOTICE_RELATIVE_PATH = Path("ASSETS/Roboto-Apache-2.0.txt")
ROBOTO_NOTICE_SOURCE = ROOT / "licenses" / "Roboto-Apache-2.0.txt"


@dataclass(frozen=True)
class FrozenEntry:
    destination: str
    source: Path
    kind: str
    owner: str | None = None


def _project_requirements() -> tuple[list[Requirement], list[Requirement]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = [Requirement(raw) for raw in project["dependencies"]]
    build = [Requirement("pyinstaller")]
    return runtime, build


def _active(requirement: Requirement) -> bool:
    return requirement.marker is None or requirement.marker.evaluate({"extra": ""})


def _resolve_closure(requirements: list[Requirement]) -> dict[str, object]:
    pending = deque(req for req in requirements if _active(req))
    resolved: dict[str, object] = {}
    while pending:
        requirement = pending.popleft()
        key = canonicalize_name(requirement.name)
        if key in resolved:
            continue
        try:
            dist = distribution(requirement.name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required distribution is not installed: {requirement.name}"
            ) from exc
        resolved[key] = dist
        for raw_child in dist.requires or ():
            child = Requirement(raw_child)
            if _active(child):
                pending.append(child)
    return resolved


def _resolved_distribution_scopes() -> dict[str, tuple[object, frozenset[str]]]:
    runtime_roots, build_roots = _project_requirements()
    runtime = _resolve_closure(runtime_roots)
    build = _resolve_closure(build_roots)
    records: dict[str, tuple[object, frozenset[str]]] = {}
    for key in sorted(runtime.keys() | build.keys()):
        scopes = set()
        if key in runtime:
            scopes.add("runtime")
        if key in build:
            scopes.add("build")
        records[key] = (runtime.get(key) or build[key], frozenset(scopes))
    return records


def _resolved_distributions() -> list[object]:
    """Compatibility wrapper used by local tooling and tests."""
    return [record[0] for record in _resolved_distribution_scopes().values()]


def _license_ids(expression: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", expression)
        if token not in {"AND", "OR", "WITH"}
    }


def validate_distribution_license(dist: object, *, scope: str) -> str:
    """Return a canonical expression or reject forbidden/unknown metadata."""
    name = str(dist.metadata["Name"])
    key = canonicalize_name(name)
    if key in FORBIDDEN_DISTRIBUTIONS:
        raise RuntimeError(f"Forbidden distribution in {scope} scope: {name}")

    version_overrides = LICENSE_OVERRIDES.get(key)
    if version_overrides is not None:
        raw = version_overrides.get(str(dist.version))
        if raw is None:
            reviewed = ", ".join(sorted(version_overrides))
            raise RuntimeError(
                f"Unreviewed license override version for {name} {dist.version}; "
                f"reviewed: {reviewed}"
            )
    else:
        raw = dist.metadata.get("License-Expression")
    if not raw:
        raw = dist.metadata.get("License")
    if not raw or not str(raw).strip():
        raise RuntimeError(f"Unknown license for {name} {dist.version}")
    try:
        expression = canonicalize_license_expression(str(raw).strip())
    except InvalidLicenseExpression as exc:
        raise RuntimeError(
            f"Unknown or ambiguous license for {name} {dist.version}: {raw!r}"
        ) from exc

    identifiers = _license_ids(expression)
    denied = sorted(
        identifier for identifier in identifiers if identifier.startswith(DENIED_LICENSE_PREFIXES)
    )
    if denied:
        raise RuntimeError(f"Forbidden license for {name} {dist.version}: {', '.join(denied)}")
    allowed = ALLOWED_BUILD_LICENSE_IDS if scope == "build" else ALLOWED_RUNTIME_LICENSE_IDS
    unknown = sorted(identifiers - allowed)
    if unknown:
        raise RuntimeError(
            f"License policy has no {scope} approval for {name} {dist.version}: "
            f"{', '.join(unknown)}"
        )
    return expression


def _is_license_file(path: PurePosixPath) -> bool:
    if path.suffix.lower() not in NOTICE_SUFFIXES:
        return False
    filename = path.name.lower()
    if any(marker in filename for marker in LICENSE_MARKERS):
        return True
    parts = tuple(part.lower() for part in path.parts)
    return any(part.endswith(".dist-info") for part in parts) and "licenses" in parts


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _installed_file_owners() -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = canonicalize_name(name)
        for entry in dist.files or ():
            located = Path(dist.locate_file(entry))
            owners.setdefault(_normalized_path(located), set()).add(key)
    return owners


def _read_toc_entries(paths: list[Path]) -> list[FrozenEntry]:
    entries: dict[tuple[str, str], FrozenEntry] = {}

    def visit(value: object) -> None:
        if isinstance(value, (list, tuple)):
            if (
                len(value) >= 3
                and isinstance(value[0], str)
                and isinstance(value[1], str)
                and isinstance(value[2], str)
            ):
                source = Path(value[1])
                if source.is_absolute():
                    destination = value[0].replace("\\", "/")
                    entries[(destination.lower(), _normalized_path(source))] = FrozenEntry(
                        destination=destination,
                        source=source,
                        kind=value[2],
                    )
            for child in value:
                visit(child)

    for path in paths:
        try:
            payload = ast.literal_eval(Path(path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError) as exc:
            raise RuntimeError(f"Cannot read PyInstaller TOC: {path}: {exc}") from exc
        visit(payload)
    return sorted(entries.values(), key=lambda item: (item.destination.lower(), str(item.source)))


def _assign_frozen_owners(entries: list[FrozenEntry]) -> list[FrozenEntry]:
    file_owners = _installed_file_owners()
    resolved: list[FrozenEntry] = []
    for entry in entries:
        owners = file_owners.get(_normalized_path(entry.source), set()) - {PROJECT_DISTRIBUTION}
        if len(owners) > 1:
            raise RuntimeError(
                f"Ambiguous frozen distribution owner for {entry.source}: {sorted(owners)}"
            )
        owner = next(iter(owners), None)
        resolved.append(
            FrozenEntry(
                destination=entry.destination,
                source=entry.source,
                kind=entry.kind,
                owner=owner,
            )
        )
    return resolved


def _audit_frozen_distributions(
    entries: list[FrozenEntry],
    scoped: dict[str, tuple[object, frozenset[str]]],
) -> set[str]:
    frozen = {entry.owner for entry in entries if entry.owner}
    unexpected = sorted(
        key for key in frozen if key not in scoped or "runtime" not in scoped[key][1]
    )
    if unexpected:
        raise RuntimeError(
            "Frozen payload contains undeclared/build-only distribution(s): "
            + ", ".join(unexpected)
        )
    return frozen


def _normalized_frozen_destination(destination: str) -> str:
    normalized = destination.replace("\\", "/").lower()
    return normalized.removeprefix("_internal/")


def _display_frozen_destination(destination: str) -> str:
    normalized = destination.replace("\\", "/")
    if normalized.lower().startswith("_internal/"):
        return normalized[len("_internal/") :]
    return normalized


def collect_frozen_asset_notices(output_dir: Path, entries: list[FrozenEntry]) -> dict[str, str]:
    """License reviewed non-code assets that distribution metadata cannot describe."""
    normalized_entries = {
        _normalized_frozen_destination(entry.destination): entry for entry in entries
    }
    roboto_candidates = {
        destination
        for destination in normalized_entries
        if destination.startswith(ROBOTO_ASSET_PREFIX)
    }
    unexpected = sorted(roboto_candidates - ROBOTO_ASSET_DESTINATIONS)
    if unexpected:
        raise RuntimeError(
            "Frozen payload contains unreviewed Roboto asset(s): " + ", ".join(unexpected)
        )
    present_roboto = roboto_candidates & ROBOTO_ASSET_DESTINATIONS
    if present_roboto and present_roboto != ROBOTO_ASSET_DESTINATIONS:
        missing = sorted(ROBOTO_ASSET_DESTINATIONS - present_roboto)
        raise RuntimeError("Frozen Roboto asset set is incomplete: " + ", ".join(missing))

    reviewed_destinations = set(present_roboto)
    if CUSTOMTKINTER_SHAPES_ASSET_DESTINATION in normalized_entries:
        reviewed_destinations.add(CUSTOMTKINTER_SHAPES_ASSET_DESTINATION)
    misowned = sorted(
        destination
        for destination in reviewed_destinations
        if normalized_entries[destination].owner != "customtkinter"
    )
    if misowned:
        raise RuntimeError(
            "Frozen CustomTkinter font asset has an unexpected owner: " + ", ".join(misowned)
        )

    labels: dict[str, str] = {}
    if present_roboto:
        if not ROBOTO_NOTICE_SOURCE.is_file():
            raise RuntimeError(f"Roboto Apache-2.0 license is missing: {ROBOTO_NOTICE_SOURCE}")
        destination = Path(output_dir) / ROBOTO_NOTICE_RELATIVE_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROBOTO_NOTICE_SOURCE, destination)
        for asset in sorted(present_roboto):
            display = _display_frozen_destination(normalized_entries[asset].destination)
            labels[display] = f"Roboto; Apache-2.0; {ROBOTO_NOTICE_RELATIVE_PATH.as_posix()}"
    if CUSTOMTKINTER_SHAPES_ASSET_DESTINATION in reviewed_destinations:
        entry = normalized_entries[CUSTOMTKINTER_SHAPES_ASSET_DESTINATION]
        labels[_display_frozen_destination(entry.destination)] = (
            "customtkinter; MIT; customtkinter distribution license"
        )
    return labels


def _runtime_license_texts(runtime_prefix: Path) -> dict[str, str]:
    prefix = Path(runtime_prefix)
    python_candidates = (prefix / "LICENSE.txt", prefix / "LICENSE")
    python_path = next((path for path in python_candidates if path.is_file()), None)
    if python_path is None:
        raise RuntimeError(f"Python PSF license not found under runtime prefix: {prefix}")
    python_text = python_path.read_text(encoding="utf-8")
    if "Python Software Foundation" not in python_text:
        raise RuntimeError(f"Python runtime license is not a PSF notice: {python_path}")

    def versioned_notice(component: str) -> Path | None:
        candidates = sorted((prefix / "tcl").glob(f"{component}*/license.terms"))
        return next((path for path in candidates if path.is_file()), None)

    tcl_path = versioned_notice("tcl")
    tk_path = versioned_notice("tk")
    sections = [
        TCL_TK_COPYRIGHT_MARKER + section
        for section in python_text.split(TCL_TK_COPYRIGHT_MARKER)[1:]
    ]
    tcl_text = tcl_path.read_text(encoding="utf-8") if tcl_path else ""
    tk_text = tk_path.read_text(encoding="utf-8") if tk_path else ""
    if not tcl_text and sections:
        tcl_text = sections[0].strip() + "\n"
    if not tk_text and len(sections) >= 2:
        tk_text = sections[1].strip() + "\n"
    for component, text in (("Tcl", tcl_text), ("Tk", tk_text)):
        if TCL_TK_COPYRIGHT_MARKER not in text:
            raise RuntimeError(f"{component} runtime license notice is missing under: {prefix}")
    return {"python": python_text, "tcl": tcl_text, "tk": tk_text}


def collect_runtime_notices(
    output_dir: Path, *, runtime_prefix: Path | None = None
) -> dict[str, Path]:
    """Copy PSF/Tcl/Tk notices from the interpreter used for freezing."""
    texts = _runtime_license_texts(runtime_prefix or Path(sys.base_prefix))
    runtime_dir = Path(output_dir) / "RUNTIME"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for component, filename in RUNTIME_NOTICE_FILENAMES.items():
        destination = runtime_dir / filename
        destination.write_text(texts[component], encoding="utf-8")
        outputs[component] = destination
    return outputs


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _native_runtime_owner(entry: FrozenEntry, runtime_prefix: Path) -> str | None:
    if entry.owner:
        return entry.owner
    destination = entry.destination.lower()
    source = entry.source
    if Path(destination).suffix.lower() not in NATIVE_SUFFIXES:
        return None
    if destination == "uniscan.exe":
        return "pyinstaller-bootloader"
    if _is_relative_to(source, runtime_prefix):
        return "python-runtime"
    windows_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    if _is_relative_to(source, windows_root):
        return "windows-runtime"
    return None


def _write_frozen_inventory(
    output_dir: Path,
    *,
    portable_root: Path,
    entries: list[FrozenEntry],
    frozen_distributions: set[str],
    frozen_asset_licenses: dict[str, str],
    runtime_prefix: Path,
) -> Path:
    entry_by_destination = {entry.destination.lower(): entry for entry in entries}
    native_lines: list[str] = []
    unclassified: list[str] = []
    for path in sorted(Path(portable_root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        relative = path.relative_to(portable_root).as_posix()
        relative_key = relative.lower()
        entry = entry_by_destination.get(relative_key)
        if entry is None and relative_key.startswith("_internal/"):
            entry = entry_by_destination.get(relative_key.removeprefix("_internal/"))
        owner = _native_runtime_owner(entry, runtime_prefix) if entry else None
        if owner is None:
            unclassified.append(relative)
        else:
            native_lines.append(f"  {relative} [{owner}]")
    if unclassified:
        raise RuntimeError(
            "Frozen payload contains unclassified native binaries: " + ", ".join(unclassified)
        )

    lines = ["UniScan frozen payload license inventory", "", "Frozen distributions:"]
    lines.extend(f"  {name}" for name in sorted(frozen_distributions))
    lines.extend(["", "Frozen licensed assets:"])
    lines.extend(
        f"  {destination} [{label}]" for destination, label in sorted(frozen_asset_licenses.items())
    )
    lines.extend(["", "Native runtime files:"])
    lines.extend(native_lines)
    destination = Path(output_dir) / "FROZEN_PAYLOAD.txt"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def collect_licenses(
    output_dir: Path,
    *,
    portable_root: Path | None = None,
    pyinstaller_tocs: list[Path] | None = None,
    runtime_prefix: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    runtime_prefix = Path(runtime_prefix or sys.base_prefix)
    scoped = _resolved_distribution_scopes()
    entries: list[FrozenEntry] = []
    frozen_distributions: set[str] | None = None
    frozen_asset_licenses: dict[str, str] = {}
    if pyinstaller_tocs:
        entries = _assign_frozen_owners(
            _read_toc_entries([Path(path) for path in pyinstaller_tocs])
        )
        frozen_distributions = _audit_frozen_distributions(entries, scoped)
        frozen_asset_licenses = collect_frozen_asset_notices(output_dir, entries)

    index_lines = [
        "UniScan third-party license inventory",
        "Policy: fail-closed canonical SPDX allowlist; GPL/AGPL/unknown licenses are rejected.",
        "",
    ]
    for key, (dist, scopes) in scoped.items():
        policy_scope = "runtime" if "runtime" in scopes else "build"
        expression = validate_distribution_license(dist, scope=policy_scope)
        name = str(dist.metadata["Name"])
        version = dist.version
        copied: list[str] = []
        for entry in dist.files or ():
            relative = PurePosixPath(str(entry).replace("\\", "/"))
            if not _is_license_file(relative):
                continue
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                continue
            destination = output_dir / key / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination.relative_to(output_dir).as_posix())
        if not copied:
            raise RuntimeError(f"No license/notice file found for {name} {version}")

        if frozen_distributions is None:
            payload_scope = ",".join(sorted(scopes))
        elif key in frozen_distributions:
            payload_scope = "frozen-runtime"
        elif "build" in scopes and "runtime" not in scopes:
            payload_scope = "build-only"
        else:
            payload_scope = "declared-runtime-not-frozen"
        index_lines.append(f"{name} {version} [{payload_scope}] - {expression}")
        index_lines.extend(f"  {path}" for path in copied)
        index_lines.append("")

    if frozen_asset_licenses:
        index_lines.append("Bundled frozen asset licenses:")
        index_lines.extend(
            f"  {destination} [{label}]"
            for destination, label in sorted(frozen_asset_licenses.items())
        )
        index_lines.append("")

    if portable_root is not None:
        if not pyinstaller_tocs:
            raise RuntimeError("Portable payload audit requires at least one PyInstaller TOC.")
        notices = collect_runtime_notices(output_dir, runtime_prefix=runtime_prefix)
        index_lines.append("Bundled native runtime notices:")
        for component, path in notices.items():
            index_lines.append(f"  {component}: {path.relative_to(output_dir).as_posix()}")
        index_lines.append("")
        inventory = _write_frozen_inventory(
            output_dir,
            portable_root=Path(portable_root),
            entries=entries,
            frozen_distributions=frozen_distributions or set(),
            frozen_asset_licenses=frozen_asset_licenses,
            runtime_prefix=runtime_prefix,
        )
        index_lines.append(f"Frozen payload inventory: {inventory.name}")
        index_lines.append("")

    (output_dir / "INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--portable-root", type=Path)
    parser.add_argument("--pyinstaller-toc", type=Path, action="append", default=[])
    parser.add_argument("--runtime-prefix", type=Path)
    args = parser.parse_args()
    collect_licenses(
        args.output_dir,
        portable_root=args.portable_root,
        pyinstaller_tocs=args.pyinstaller_toc,
        runtime_prefix=args.runtime_prefix,
    )
    print(f"Collected and validated third-party licenses in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
