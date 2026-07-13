from __future__ import annotations

from pathlib import Path

import pytest

from scripts.collect_third_party_licenses import (
    CUSTOMTKINTER_SHAPES_ASSET_DESTINATION,
    FrozenEntry,
    collect_frozen_asset_notices,
    collect_licenses,
    collect_runtime_notices,
    validate_distribution_license,
)
from scripts.audit_portable_contents import (
    REQUIRED_PATHS,
    ROBOTO_ASSET_DESTINATIONS,
    ROBOTO_NOTICE_PATH,
    RUNTIME_NOTICE_MARKERS,
    audit_portable_contents,
)
from scripts.verify_release_version import source_version, verify_release_metadata


def test_project_uses_one_release_version_source() -> None:
    assert source_version() == "0.1.0"
    assert verify_release_metadata() == source_version()


def test_release_tag_must_match_source_version() -> None:
    with pytest.raises(RuntimeError, match="does not match"):
        verify_release_metadata("v9.9.9")


def test_release_tag_must_be_annotated(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify_release_version._git_output",
        lambda *_args: "commit",
    )

    with pytest.raises(RuntimeError, match="must be annotated"):
        verify_release_metadata("v0.1.0")


def test_release_tag_must_dereference_to_build_head(monkeypatch) -> None:
    responses = {
        ("cat-file", "-t", "refs/tags/v0.1.0"): "tag",
        ("rev-parse", "refs/tags/v0.1.0^{commit}"): "tagged-commit",
        ("rev-parse", "HEAD"): "different-head",
    }
    monkeypatch.setattr(
        "scripts.verify_release_version._git_output",
        lambda *args: responses[args],
    )

    with pytest.raises(RuntimeError, match="build is using HEAD"):
        verify_release_metadata("v0.1.0")


def test_license_inventory_contains_runtime_dependencies(tmp_path: Path) -> None:
    output = tmp_path / "licenses"
    collect_licenses(output)

    index = (output / "INDEX.txt").read_text(encoding="utf-8").lower()
    for package in ("img2pdf", "numpy", "opencv-python", "pypdfium2", "tkinterdnd2"):
        assert package in index
    assert "pymupdf" not in index
    assert "fail-closed canonical spdx allowlist" in index
    img2pdf_line = next(line for line in index.splitlines() if line.startswith("img2pdf "))
    assert img2pdf_line.endswith(" - lgpl-3.0-or-later")
    assert not (output / "RUNTIME").exists()


class _FakeDistribution:
    def __init__(self, name: str, license_expression: str | None) -> None:
        self.metadata = {"Name": name}
        if license_expression is not None:
            self.metadata["License-Expression"] = license_expression
        self.version = "1.0"


@pytest.mark.parametrize("expression", ["AGPL-3.0-only", "GPL-3.0-only"])
def test_license_policy_rejects_forbidden_copyleft(expression: str) -> None:
    with pytest.raises(RuntimeError, match="Forbidden license"):
        validate_distribution_license(
            _FakeDistribution("fake-forbidden", expression),
            scope="runtime",
        )


def test_license_policy_rejects_unknown_or_ambiguous_distribution() -> None:
    with pytest.raises(RuntimeError, match="Unknown license"):
        validate_distribution_license(
            _FakeDistribution("fake-unknown", None),
            scope="runtime",
        )
    with pytest.raises(RuntimeError, match="Unknown or ambiguous"):
        validate_distribution_license(
            _FakeDistribution("fake-ambiguous", "MIT or proprietary"),
            scope="runtime",
        )


def test_license_policy_rejects_forbidden_distribution_even_with_permissive_metadata() -> None:
    with pytest.raises(RuntimeError, match="Forbidden distribution"):
        validate_distribution_license(
            _FakeDistribution("PyMuPDF", "MIT"),
            scope="runtime",
        )


def test_license_override_rejects_unreviewed_distribution_version() -> None:
    with pytest.raises(RuntimeError, match="Unreviewed license override version"):
        validate_distribution_license(
            _FakeDistribution("customtkinter", "MIT"),
            scope="runtime",
        )


def test_release_workflow_treats_tag_name_as_data() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    assert '--tag "${{ github.ref_name }}"' not in workflow
    assert 'gh release create "${{ github.ref_name }}"' not in workflow
    assert workflow.count("RELEASE_TAG: ${{ github.ref_name }}") == 2
    assert '--tag "$env:RELEASE_TAG"' in workflow
    assert 'gh release create "$env:RELEASE_TAG"' in workflow


def test_runtime_notices_include_python_tcl_and_tk(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "LICENSE.txt").write_text(
        "Python Software Foundation License\n",
        encoding="utf-8",
    )
    tcl = runtime / "tcl" / "tcl8.6" / "license.terms"
    tk = runtime / "tcl" / "tk8.6" / "license.terms"
    tcl.parent.mkdir(parents=True)
    tk.parent.mkdir(parents=True)
    tcl.write_text("This software is copyrighted by the Regents of Tcl\n", encoding="utf-8")
    tk.write_text("This software is copyrighted by the Regents of Tk\n", encoding="utf-8")

    copied = collect_runtime_notices(tmp_path / "licenses", runtime_prefix=runtime)

    assert set(copied) == {"python", "tcl", "tk"}
    assert "Python Software Foundation" in copied["python"].read_text(encoding="utf-8")
    assert "Regents of Tcl" in copied["tcl"].read_text(encoding="utf-8")
    assert "Regents of Tk" in copied["tk"].read_text(encoding="utf-8")


def test_frozen_asset_notices_separate_roboto_from_customtkinter(tmp_path: Path) -> None:
    destinations = [*ROBOTO_ASSET_DESTINATIONS, CUSTOMTKINTER_SHAPES_ASSET_DESTINATION]
    entries = [
        FrozenEntry(destination, tmp_path / Path(destination).name, "DATA", "customtkinter")
        for destination in destinations
    ]

    labels = collect_frozen_asset_notices(tmp_path / "licenses", entries)
    normalized_labels = {destination.lower(): label for destination, label in labels.items()}

    for asset in ROBOTO_ASSET_DESTINATIONS:
        assert normalized_labels[asset].startswith("Roboto; Apache-2.0;")
    assert normalized_labels[CUSTOMTKINTER_SHAPES_ASSET_DESTINATION].startswith(
        "customtkinter; MIT;"
    )
    notice = tmp_path / "licenses/ASSETS/Roboto-Apache-2.0.txt"
    assert "Roboto font files" in notice.read_text(encoding="utf-8")


def test_frozen_asset_notice_rejects_incomplete_roboto_set(tmp_path: Path) -> None:
    asset = next(iter(ROBOTO_ASSET_DESTINATIONS))
    entry = FrozenEntry(asset, tmp_path / Path(asset).name, "DATA", "customtkinter")

    with pytest.raises(RuntimeError, match="Roboto asset set is incomplete"):
        collect_frozen_asset_notices(tmp_path / "licenses", [entry])


def test_portable_content_audit_rejects_model_weights(tmp_path: Path) -> None:
    root = tmp_path / "portable"
    for relative in REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RUNTIME_NOTICE_MARKERS.get(relative, "portable"), encoding="utf-8")
    tkdnd = root / "_internal/tkinterdnd2/tkdnd/win-x64/tkdnd.tcl"
    tkdnd.parent.mkdir(parents=True)
    tkdnd.write_text("package provide tkdnd 1", encoding="utf-8")

    audit_portable_contents(root)

    model = root / "unlicensed.ort"
    model.write_bytes(b"model")
    with pytest.raises(RuntimeError, match="forbidden"):
        audit_portable_contents(root)

    model.unlink()
    leaked_native = root / "_internal/mupdfcpp64.dll"
    leaked_native.parent.mkdir(parents=True, exist_ok=True)
    leaked_native.write_bytes(b"native")
    with pytest.raises(RuntimeError, match="forbidden"):
        audit_portable_contents(root)


def test_portable_content_audit_rejects_invalid_runtime_notice(tmp_path: Path) -> None:
    root = tmp_path / "portable"
    for relative in REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RUNTIME_NOTICE_MARKERS.get(relative, "portable"), encoding="utf-8")
    tkdnd = root / "_internal/tkinterdnd2/tkdnd/win-x64/tkdnd.tcl"
    tkdnd.parent.mkdir(parents=True)
    tkdnd.write_text("package provide tkdnd 1", encoding="utf-8")
    (root / "THIRD_PARTY_LICENSES/RUNTIME/TCL-LICENSE.txt").write_text(
        "not a license",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime notice is invalid"):
        audit_portable_contents(root)


def test_portable_content_audit_requires_roboto_notice_and_inventory(tmp_path: Path) -> None:
    root = tmp_path / "portable"
    for relative in REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RUNTIME_NOTICE_MARKERS.get(relative, "portable"), encoding="utf-8")
    tkdnd = root / "_internal/tkinterdnd2/tkdnd/win-x64/tkdnd.tcl"
    tkdnd.parent.mkdir(parents=True)
    tkdnd.write_text("package provide tkdnd 1", encoding="utf-8")
    for asset in ROBOTO_ASSET_DESTINATIONS:
        path = root / "_internal" / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"font")

    with pytest.raises(RuntimeError, match="missing their Apache-2.0 license notice"):
        audit_portable_contents(root)

    notice = root / ROBOTO_NOTICE_PATH
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text(
        "Roboto font files\nCopyright 2011 Google Inc.\nApache License\nVersion 2.0\n",
        encoding="utf-8",
    )
    inventory = root / "THIRD_PARTY_LICENSES/FROZEN_PAYLOAD.txt"
    inventory.write_text(
        "\n".join(sorted(ROBOTO_ASSET_DESTINATIONS)) + "\nRoboto; Apache-2.0\n",
        encoding="utf-8",
    )

    audit_portable_contents(root)
