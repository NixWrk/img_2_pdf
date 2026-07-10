"""Runtime diagnostics shared by the CLI and Windows packaging smoke checks."""

from __future__ import annotations

import importlib
import json
import platform
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import onnxruntime as ort

from uniscan.io.camera_service import CaptureFactory, CameraService, _opencv_capture_factory
from uniscan.office_lens import CLASSIFIER_MODEL, QUAD_MODEL


RUNTIME_MODULES = (
    "cv2",
    "customtkinter",
    "fitz",
    "img2pdf",
    "numpy",
    "onnxruntime",
    "PIL",
    "tkinterdnd2",
)


@dataclass(slots=True, frozen=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True, frozen=True)
class DiagnosticReport:
    python: str
    platform: str
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "python": self.python,
            "platform": self.platform,
            "checks": [asdict(check) for check in self.checks],
        }


def _module_checks(module_names: Sequence[str]) -> list[DiagnosticCheck]:
    checks: list[DiagnosticCheck] = []
    for name in module_names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            checks.append(DiagnosticCheck(f"module:{name}", True, str(version)))
        except Exception as exc:
            checks.append(DiagnosticCheck(f"module:{name}", False, str(exc)))
    return checks


def _model_check(name: str, path: Path) -> DiagnosticCheck:
    if not path.is_file():
        return DiagnosticCheck(name, False, f"missing: {path}")
    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return DiagnosticCheck(name, False, f"cannot load {path}: {exc}")
    return DiagnosticCheck(name, True, f"{path} ({path.stat().st_size} bytes)")


def run_diagnostics(
    *,
    check_camera: bool = False,
    camera_index: int = 0,
    capture_factory: CaptureFactory = _opencv_capture_factory,
) -> DiagnosticReport:
    checks: list[DiagnosticCheck] = [
        DiagnosticCheck(
            "python-version",
            sys.version_info >= (3, 11),
            platform.python_version(),
        )
    ]
    checks.extend(_module_checks(RUNTIME_MODULES))
    checks.append(_model_check("model:quad", QUAD_MODEL))
    checks.append(_model_check("model:classifier", CLASSIFIER_MODEL))

    try:
        with tempfile.TemporaryDirectory(prefix="uniscan_doctor_") as directory:
            probe = Path(directory) / "write-test"
            probe.write_text("ok", encoding="utf-8")
        checks.append(DiagnosticCheck("temporary-storage", True, "writable"))
    except Exception as exc:
        checks.append(DiagnosticCheck("temporary-storage", False, str(exc)))

    if check_camera:
        camera = CameraService(index=camera_index, capture_factory=capture_factory)
        try:
            camera.open()
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("camera opened but returned no frame")
            detail = f"index {camera_index}, frame {frame.shape[1]}x{frame.shape[0]}"
            checks.append(DiagnosticCheck("camera", True, detail))
        except Exception as exc:
            checks.append(DiagnosticCheck("camera", False, str(exc)))
        finally:
            camera.release()

    return DiagnosticReport(
        python=platform.python_version(),
        platform=platform.platform(),
        checks=tuple(checks),
    )


def format_diagnostics(report: DiagnosticReport) -> str:
    lines = [f"UniScan diagnostics: {'OK' if report.ok else 'FAILED'}"]
    for check in report.checks:
        lines.append(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return "\n".join(lines)


def diagnostics_json(report: DiagnosticReport) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False)
