from __future__ import annotations

import json

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
