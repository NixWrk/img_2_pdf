"""Runtime diagnostics shared by the CLI and Windows packaging smoke checks."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from uniscan.io.camera_service import CaptureFactory, CameraService, _opencv_capture_factory
from uniscan.office_lens import CLASSIFIER_MODEL, QUAD_MODEL
from uniscan.model_assets import model_asset, verify_model_asset


RUNTIME_MODULES = (
    "cv2",
    "customtkinter",
    "img2pdf",
    "numpy",
    "onnxruntime",
    "PIL",
    "pypdfium2",
    "tkinterdnd2",
)


CAMERA_FPS_SAMPLE_SEC = 1.5


def _backend_name(api_preference: int | None) -> str | None:
    """Human-readable OpenCV backend name, when one is selected."""
    if api_preference is None:
        return None
    try:
        import cv2

        names = {cv2.CAP_MSMF: "MSMF", cv2.CAP_DSHOW: "DirectShow", cv2.CAP_V4L2: "V4L2"}
    except Exception:
        return None
    return names.get(api_preference, str(api_preference))


def _measure_stream_fps(camera: CameraService) -> float | None:
    """Briefly run the frame stream and report its measured rate."""
    try:
        camera.start_stream()
        deadline = time.monotonic() + CAMERA_FPS_SAMPLE_SEC
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return camera.measured_fps
    except Exception:
        return None
    finally:
        try:
            camera.stop_stream()
        except Exception:
            pass


@dataclass(slots=True, frozen=True)
class DiagnosticCheck:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


@dataclass(slots=True, frozen=True)
class DiagnosticReport:
    python: str
    platform: str
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok or not check.blocking for check in self.checks)

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


def _open_onnx_session(path: Path) -> None:
    """Load one configured ONNX model so doctor catches incompatible payloads."""
    import onnxruntime as ort

    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _optional_onnx_model_check(
    *,
    name: str,
    path: Path,
    missing_detail: str,
    loaded_label: str,
    blocking: bool = True,
) -> DiagnosticCheck:
    if not path.is_file():
        return DiagnosticCheck(name, True, missing_detail, blocking=blocking)
    try:
        _open_onnx_session(path)
    except Exception as exc:
        return DiagnosticCheck(name, False, str(exc), blocking=blocking)
    return DiagnosticCheck(
        name,
        True,
        f"{loaded_label} loaded from {path.parent}",
        blocking=blocking,
    )


def _optional_office_lens_checks() -> tuple[DiagnosticCheck, DiagnosticCheck]:
    """Report the required quad model separately from optional auto-classification."""
    return (
        _optional_onnx_model_check(
            name="optional:office-lens",
            path=QUAD_MODEL,
            missing_detail=(
                "disabled; install a licensed quad model and the 'office-lens' extra to enable it"
            ),
            loaded_label="licensed quad model",
        ),
        _optional_onnx_model_check(
            name="optional:office-lens-classifier",
            path=CLASSIFIER_MODEL,
            missing_detail=("disabled; explicit document/photo/whiteboard modes remain available"),
            loaded_label="licensed classifier model",
            blocking=False,
        ),
    )


def _optional_office_lens_check() -> DiagnosticCheck:
    """Return the historical primary Office Lens check for API compatibility."""
    return _optional_office_lens_checks()[0]


def _bundled_model_checks() -> tuple[DiagnosticCheck, DiagnosticCheck]:
    """Verify the exact on-disk identities used by the built-in model stages."""
    from uniscan.core import docshadow, uvdoc

    uvdoc_path = uvdoc.model_path()
    try:
        if os.environ.get(uvdoc.MODEL_ENV_VAR):
            _open_onnx_session(uvdoc_path)
            uvdoc_detail = f"custom model loaded from {uvdoc_path}"
        else:
            verify_model_asset("uvdoc_graph", uvdoc_path)
            verify_model_asset("uvdoc_data", uvdoc_path.with_name(f"{uvdoc_path.name}.data"))
            uvdoc_detail = f"pinned SHA-256 verified: {model_asset('uvdoc_graph').sha256}"
        uvdoc_check = DiagnosticCheck("model:uvdoc", True, uvdoc_detail)
    except Exception as exc:
        uvdoc_check = DiagnosticCheck("model:uvdoc", False, str(exc))

    docshadow_path = docshadow.model_path()
    if not docshadow_path.is_file():
        docshadow_check = DiagnosticCheck(
            "model:docshadow",
            True,
            "disabled in this source checkout; run scripts/download_model_assets.py",
            blocking=False,
        )
    else:
        try:
            if os.environ.get(docshadow.MODEL_ENV_VAR):
                _open_onnx_session(docshadow_path)
                docshadow_detail = f"custom model loaded from {docshadow_path}"
            else:
                verify_model_asset("docshadow_sd7k", docshadow_path)
                docshadow_detail = (
                    "pinned SHA-256 verified: " + model_asset("docshadow_sd7k").sha256
                )
            docshadow_check = DiagnosticCheck("model:docshadow", True, docshadow_detail)
        except Exception as exc:
            docshadow_check = DiagnosticCheck("model:docshadow", False, str(exc), blocking=False)
    return uvdoc_check, docshadow_check


def _gui_runtime_check() -> DiagnosticCheck:
    """Create a real TkDND root so frozen native payload failures are observable."""
    root = None
    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
        root.withdraw()
        root.update_idletasks()
        return DiagnosticCheck("gui-runtime", True, "Tk and native TkDND initialized")
    except Exception as exc:
        return DiagnosticCheck("gui-runtime", False, str(exc))
    finally:
        if root is not None:
            root.destroy()


def run_diagnostics(
    *,
    check_camera: bool = False,
    check_gui_runtime: bool = False,
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
    checks.extend(_bundled_model_checks())
    checks.extend(_optional_office_lens_checks())
    if check_gui_runtime:
        checks.append(_gui_runtime_check())

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
            # Sample the live stream: frame rate is what makes the preview
            # usable, and it is the first thing to check on a slow camera.
            fps = _measure_stream_fps(camera)
            if fps is not None:
                detail += f", {fps:.0f} fps"
            backend = _backend_name(camera.api_preference)
            if backend:
                detail += f", backend {backend}"
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
        status = "ok" if check.ok else ("warn" if not check.blocking else "FAIL")
        lines.append(f"[{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def diagnostics_json(report: DiagnosticReport) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False)
