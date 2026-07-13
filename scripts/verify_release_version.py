"""Fail closed when a release tag and source metadata do not describe one release."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"git {' '.join(args)} failed: {detail.strip()}") from exc
    return result.stdout.strip()


def _verify_annotated_tag(tag: str) -> None:
    reference = f"refs/tags/{tag}"
    object_type = _git_output("cat-file", "-t", reference)
    if object_type != "tag":
        raise RuntimeError(
            f"release tag {tag!r} must be annotated; git object type is {object_type!r}"
        )
    tagged_commit = _git_output("rev-parse", f"{reference}^{{commit}}")
    head_commit = _git_output("rev-parse", "HEAD")
    if tagged_commit != head_commit:
        raise RuntimeError(
            f"annotated release tag {tag!r} points to {tagged_commit}, "
            f"but the build is using HEAD {head_commit}"
        )


def source_version() -> str:
    module = ast.parse((ROOT / "src/uniscan/__init__.py").read_text(encoding="utf-8"))
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(statement.value)
                    if isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+", value):
                        return value
    raise RuntimeError("src/uniscan/__init__.py has no semantic __version__ assignment")


def verify_release_metadata(tag: str | None = None) -> str:
    version = source_version()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if "version" in project or "version" not in project.get("dynamic", []):
        raise RuntimeError("pyproject.toml must derive its version from uniscan.__version__")

    if tag is None:
        return version
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise RuntimeError(f"release tag {tag!r} does not match source version {expected_tag!r}")
    _verify_annotated_tag(tag)

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M):
        raise RuntimeError(f"CHANGELOG.md has no dated [{version}] release section")
    unreleased = re.search(r"^## \[Unreleased\]\s*(.*?)(?=^## \[)", changelog, re.M | re.S)
    if unreleased and re.search(r"^- ", unreleased.group(1), re.M):
        raise RuntimeError("CHANGELOG.md still contains unreleased entries; cut a release first")
    return version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Expected annotated tag, for example v1.2.3")
    args = parser.parse_args()
    version = verify_release_metadata(args.tag)
    print(f"Release metadata OK: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
