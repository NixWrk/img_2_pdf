from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uniscan.cli import main
from uniscan.diagnostics import diagnostics_json, format_diagnostics, run_diagnostics


class FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def isOpened(self) -> bool:
        return True

    def read(self):
        return True, self.frame

    def release(self) -> None:
        return None

    def set(self, _prop_id: int, _value: float) -> bool:
        return True


def test_runtime_diagnostics_pass_without_camera() -> None:
    report = run_diagnostics()
    assert report.ok
    assert "UniScan diagnostics: OK" in format_diagnostics(report)
    payload = json.loads(diagnostics_json(report))
    assert payload["ok"] is True
    assert any(check["name"] == "optional:office-lens" for check in payload["checks"])
    assert any(check["name"] == "optional:office-lens-classifier" for check in payload["checks"])
    assert any(check["name"] == "model:uvdoc" for check in payload["checks"])
    assert any(check["name"] == "model:docshadow" for check in payload["checks"])
    assert any(check["name"] == "model:docscanner-l" for check in payload["checks"])


def test_office_lens_quad_model_does_not_require_classifier(tmp_path, monkeypatch) -> None:
    quad_model = tmp_path / "quad.ort"
    quad_model.write_bytes(b"licensed quad")
    classifier_model = tmp_path / "missing-classifier.ort"
    opened: list[Path] = []
    monkeypatch.setattr("uniscan.diagnostics.QUAD_MODEL", quad_model)
    monkeypatch.setattr("uniscan.diagnostics.CLASSIFIER_MODEL", classifier_model)
    monkeypatch.setattr("uniscan.diagnostics._open_onnx_session", opened.append)

    report = run_diagnostics()

    checks = {check.name: check for check in report.checks}
    assert report.ok
    assert checks["optional:office-lens"].ok
    assert "quad model loaded" in checks["optional:office-lens"].detail
    assert checks["optional:office-lens-classifier"].ok
    assert (
        "explicit document/photo/whiteboard modes remain available"
        in checks["optional:office-lens-classifier"].detail
    )
    assert opened == [quad_model]


def test_office_lens_configured_classifier_is_validated_separately(tmp_path, monkeypatch) -> None:
    quad_model = tmp_path / "quad.ort"
    classifier_model = tmp_path / "classifier.ort"
    quad_model.write_bytes(b"licensed quad")
    classifier_model.write_bytes(b"broken classifier")
    monkeypatch.setattr("uniscan.diagnostics.QUAD_MODEL", quad_model)
    monkeypatch.setattr("uniscan.diagnostics.CLASSIFIER_MODEL", classifier_model)

    def open_model(path: Path) -> None:
        if path == classifier_model:
            raise RuntimeError("incompatible classifier")

    monkeypatch.setattr("uniscan.diagnostics._open_onnx_session", open_model)

    report = run_diagnostics()
    checks = {check.name: check for check in report.checks}

    assert report.ok
    assert checks["optional:office-lens"].ok
    assert not checks["optional:office-lens-classifier"].ok
    assert not checks["optional:office-lens-classifier"].blocking
    assert "incompatible classifier" in checks["optional:office-lens-classifier"].detail
    assert "[warn] optional:office-lens-classifier" in format_diagnostics(report)


def test_orphan_office_lens_classifier_never_blocks_doctor(tmp_path, monkeypatch) -> None:
    missing_quad = tmp_path / "missing-quad.ort"
    classifier_model = tmp_path / "classifier.ort"
    classifier_model.write_bytes(b"broken orphan classifier")
    monkeypatch.setattr("uniscan.diagnostics.QUAD_MODEL", missing_quad)
    monkeypatch.setattr("uniscan.diagnostics.CLASSIFIER_MODEL", classifier_model)
    monkeypatch.setattr(
        "uniscan.diagnostics._open_onnx_session",
        lambda _path: (_ for _ in ()).throw(RuntimeError("orphan is incompatible")),
    )

    report = run_diagnostics()
    checks = {check.name: check for check in report.checks}

    assert report.ok
    assert checks["optional:office-lens"].ok
    assert not checks["optional:office-lens-classifier"].ok
    assert not checks["optional:office-lens-classifier"].blocking


def test_runtime_diagnostics_with_fake_camera() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    report = run_diagnostics(
        check_camera=True,
        camera_index=4,
        capture_factory=lambda _index, _api: FakeCapture(frame),
    )
    assert report.ok
    camera = next(check for check in report.checks if check.name == "camera")
    assert "20x10" in camera.detail


def test_runtime_diagnostics_can_probe_native_gui(monkeypatch) -> None:
    monkeypatch.setattr(
        "uniscan.diagnostics._gui_runtime_check",
        lambda: type("Check", (), {"name": "gui-runtime", "ok": True, "detail": "fake"})(),
    )
    report = run_diagnostics(check_gui_runtime=True)
    assert any(check.name == "gui-runtime" for check in report.checks)


def test_cli_doctor_json(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
