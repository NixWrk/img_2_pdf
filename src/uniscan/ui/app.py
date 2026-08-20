"""Unified application shell."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageTk
from tkinter import filedialog, messagebox

from uniscan.export import (
    export_image_paths_as_files,
    export_image_paths_as_pdf,
)
from uniscan.core.geometry import warp_perspective_from_points
from uniscan.core.dewarp import (
    DEWARP_METHOD_AUTO,
    DEWARP_METHOD_DOCSCANNER,
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_PADDLEOCR_UVDOC,
    DEWARP_METHOD_TEXTLINE,
    DEWARP_METHOD_UVDOC,
    DewarpModel,
    estimate_textline_dewarp_model,
    interpolate_control_curve,
)
from uniscan.core.pipeline import PageResult, PipelineOptions, process_loaded_items
from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.core.lighting import (
    SHADOW_METHOD_AUTO,
    SHADOW_METHOD_CLASSICAL,
    SHADOW_METHOD_DOCSHADOW,
    SHADOW_METHOD_NONE,
)
from uniscan.core.orientation import ORIENTATION_METHOD_AUTO, ORIENTATION_METHOD_NONE
from uniscan.core.preprocess import (
    DESKEW_METHOD_MANUAL,
    DESKEW_METHOD_NONE,
    DESKEW_METHOD_HOUGH,
    DESKEW_METHOD_HYBRID,
    DESKEW_METHOD_MIN_AREA,
    LENS_MODE_VALUES,
    PREPROCESS_PRESETS,
    PreprocessSettings,
    infer_lens_mode,
    resolve_lens_mode_profile,
)
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    ScanAdapterError,
    scan_with_document_detector,
)
from uniscan.core.spread import split_spread_accurate
from uniscan.diagnostics import run_diagnostics
from uniscan.io import CameraService
from uniscan.io.camera_service import CameraMode, best_realtime_mode, list_camera_device_names
from uniscan.io.loaders import (
    IMG_EXTS,
    PDF_EXTS,
    imread_unicode,
    imwrite_unicode,
    iter_input_items,
    list_supported_in_folder,
)
from uniscan.session import (
    CROP_STATE_APPLIED,
    CROP_STATE_NONE,
    CROP_STATE_PROPOSED,
    CommittedPageProcessing,
    SessionInUseError,
    UnsafeSessionLockError,
    acquire_autosave_lock,
    create_persistent_session,
    default_autosave_path,
    load_or_create_session,
)
from uniscan.storage import ProcessingStageCache
from uniscan.ui.camera_health import camera_health_state
from uniscan.ui.import_sources import (
    clipboard_file_paths,
    clipboard_image_to_bgr,
    paths_from_tk_drop,
)
from uniscan.ui.live_detect import DEFAULT_LIVE_BACKEND, LIVE_BACKEND_CHOICES, LiveContourDetector
from uniscan.ui.overlays import draw_quad_overlay, scale_contour

# Poll faster than the camera delivers: a tick without a new frame costs
# almost nothing, while Tk's ~15 ms timer granularity on Windows would
# otherwise round a frame-rate-matched interval up and drop every third frame.
PREVIEW_WAIT_MS = 12
# Below this measured rate the preview feels like a slideshow and shots carry
# visible latency, so the status suggests a smaller capture size.
LOW_PREVIEW_FPS = 10.0
CAMERA_HEALTH_REFRESH_SEC = 1.0
FRESH_FRAME_TIMEOUT_SEC = 5.0
# Preview and capture share one resolution so a shot is the frame already on
# screen; see _default_camera_resolution for why 1080p is the default.
DEFAULT_CAPTURE_RESOLUTION = (1920, 1080)
REVIEW_PREVIEW_DEBOUNCE_MS = 120
REVIEW_RESIZE_DEBOUNCE_MS = 90
RESOLUTIONS = [
    "3264x2448",
    "3264x1836",
    "2592x1944",
    "2048x1536",
    "1920x1080",
    "1600x1200",
    "1280x720",
    "1024x768",
    "800x600",
    "640x480",
]

DEWARP_UI_METHODS = {
    "None": DEWARP_METHOD_NONE,
    "Automatic (validated)": DEWARP_METHOD_AUTO,
    "Page model (UVDoc)": DEWARP_METHOD_UVDOC,
    "Page model (DocScanner-L)": DEWARP_METHOD_DOCSCANNER,
    "Text lines (offline)": DEWARP_METHOD_TEXTLINE,
}
ORIENTATION_UI_METHODS = {
    "Off": ORIENTATION_METHOD_NONE,
    "Automatic (conservative)": ORIENTATION_METHOD_AUTO,
    "Rotate 90 degrees": "90",
    "Rotate 180 degrees": "180",
    "Rotate 270 degrees": "270",
}
DESKEW_UI_METHODS = {
    "Off": DESKEW_METHOD_NONE,
    "Hybrid (recommended)": DESKEW_METHOD_HYBRID,
    "Text lines / Hough": DESKEW_METHOD_HOUGH,
    "Foreground box": DESKEW_METHOD_MIN_AREA,
    "Manual angle": DESKEW_METHOD_MANUAL,
}
BINARIZATION_UI_METHODS = {
    "None": "none",
    "Otsu (global)": "otsu",
    "Sauvola (uneven light)": "sauvola",
    "Wolf (uneven light)": "wolf",
    "Fixed threshold": "fixed",
}
DESPECKLE_UI_STRENGTHS = {
    "None": "none",
    "Conservative": "conservative",
    "Normal": "normal",
    "Strong": "strong",
}
SHADOW_UI_METHODS = {
    "None": SHADOW_METHOD_NONE,
    "Automatic (validated)": SHADOW_METHOD_AUTO,
    "Model (DocShadow)": SHADOW_METHOD_DOCSHADOW,
    "Classical": SHADOW_METHOD_CLASSICAL,
}
PAGE_LAYOUT_UI_METHODS = {
    "Keep source page": "none",
    "A4": "a4",
    "Letter": "letter",
}


@dataclass(slots=True, frozen=True)
class _ExportPageSnapshot:
    name: str
    current_path: Path


@dataclass(slots=True, frozen=True)
class _StagedImportPage:
    name: str
    raw_path: Path
    contour: np.ndarray | None
    backend: str | None
    fallback_reason: str | None
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class _ApplyPageSnapshot:
    entry_id: str
    name: str
    source_path: Path
    previous_current_path: Path
    revision: int
    request: PageProcessingRequest
    previous_committed: CommittedPageProcessing | None


@dataclass(slots=True, frozen=True)
class _StagedAppliedPage:
    entry_id: str
    result_path: Path
    committed: CommittedPageProcessing
    cache_hits: tuple[str, ...]


def _split_spread_pair(
    raw: np.ndarray,
    warped: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Detect one warped gutter and replay its ratio on the raw source."""
    warped_halves = split_spread_accurate(warped, fallback="none")
    if len(warped_halves) != 2 or raw.shape[1] < 2:
        return None
    ratio = warped_halves[0].shape[1] / max(1, warped.shape[1])
    cut = max(1, min(raw.shape[1] - 1, int(round(raw.shape[1] * ratio))))
    raw_halves = [raw[:, :cut], raw[:, cut:]]
    return raw_halves, warped_halves


def _split_at_ratio(image: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    width = image.shape[1]
    cut = max(1, min(width - 1, int(round(width * float(ratio)))))
    return image[:, :cut], image[:, cut:]


def _compose_split_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Place two output pages side by side with a stable visible separator."""
    if left.ndim != right.ndim:
        if left.ndim == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if right.ndim == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    target_height = max(left.shape[0], right.shape[0])

    def fit_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        width = max(1, int(round(image.shape[1] * target_height / image.shape[0])))
        return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)

    left = fit_height(left)
    right = fit_height(right)
    gap_width = max(8, int(round((left.shape[1] + right.shape[1]) * 0.012)))
    gap_shape = (
        (target_height, gap_width) if left.ndim == 2 else (target_height, gap_width, left.shape[2])
    )
    gap = np.full(gap_shape, 48, dtype=left.dtype)
    return np.concatenate((left, gap, right), axis=1)


def _fit_image_to_box(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize only for presentation while preserving the full-resolution source."""
    height, width = image.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height))
    if abs(scale - 1.0) < 0.01:
        return image
    return cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )


def _image_to_tk_photo(image: np.ndarray) -> ImageTk.PhotoImage:
    rgb = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def _show_canvas_magnifier(
    canvas: tk.Canvas,
    source: np.ndarray,
    source_x: float,
    source_y: float,
    canvas_x: float,
    canvas_y: float,
    photo_refs: dict[str, object],
) -> None:
    """Draw a circular full-resolution lens centered on the active cursor."""
    canvas.delete("geometry-magnifier")
    lens_size = 160
    source_height, source_width = source.shape[:2]
    crop_size = max(24, int(round(min(source_width, source_height) * 0.06)))
    crop_size = min(crop_size, max(2, source_width), max(2, source_height))
    crop = cv2.getRectSubPix(
        source,
        (crop_size, crop_size),
        (
            float(np.clip(source_x, 0, source_width - 1)),
            float(np.clip(source_y, 0, source_height - 1)),
        ),
    )
    enlarged = cv2.resize(crop, (lens_size, lens_size), interpolation=cv2.INTER_CUBIC)
    if enlarged.ndim == 2:
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2RGB)
    elif enlarged.shape[2] == 4:
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
    coordinates = np.arange(lens_size, dtype=np.float32) - (lens_size - 1) / 2
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    radius = lens_size / 2 - 2
    alpha = np.clip((radius - np.hypot(grid_x, grid_y)) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))
    photo = ImageTk.PhotoImage(Image.fromarray(rgba))
    center_x = float(canvas_x)
    center_y = float(canvas_y)
    ring_radius = lens_size / 2 + 2
    canvas.create_oval(
        center_x - ring_radius,
        center_y - ring_radius,
        center_x + ring_radius,
        center_y + ring_radius,
        fill="#111111",
        outline="#ffffff",
        width=2,
        tags=("geometry-magnifier", "geometry-magnifier-ring"),
    )
    canvas.create_image(
        center_x,
        center_y,
        image=photo,
        anchor=tk.CENTER,
        tags="geometry-magnifier",
    )
    canvas.create_line(
        center_x - 14,
        center_y,
        center_x + 14,
        center_y,
        fill="#ff3355",
        width=2,
        tags="geometry-magnifier",
    )
    canvas.create_line(
        center_x,
        center_y - 14,
        center_x,
        center_y + 14,
        fill="#ff3355",
        width=2,
        tags="geometry-magnifier",
    )
    photo_refs["magnifier_photo"] = photo
    photo_refs["magnifier_center"] = (center_x, center_y)


def _hide_canvas_magnifier(canvas: tk.Canvas, photo_refs: dict[str, object]) -> None:
    canvas.delete("geometry-magnifier")
    photo_refs["magnifier_photo"] = None
    photo_refs["magnifier_center"] = None


def _perspective_source_image(entry, *, from_current_geometry: bool) -> np.ndarray:
    return entry.original_image if from_current_geometry else entry.raw_image


def _add_dewarp_control_point(
    points: list[tuple[float, float]],
    x_value: float,
    displacement: float,
) -> int | None:
    if len(points) >= 32:
        return None
    x_value = float(np.clip(x_value, 0.0, 1.0))
    displacement = float(np.clip(displacement, -0.24, 0.24))
    if any(abs(existing_x - x_value) < 0.005 for existing_x, _value in points):
        return None
    points.append((x_value, displacement))
    points.sort(key=lambda point: point[0])
    return next(index for index, point in enumerate(points) if point[0] == x_value)


def _move_dewarp_control_point(
    points: list[tuple[float, float]],
    index: int,
    x_value: float,
    displacement: float,
) -> None:
    margin = 0.005
    lower = points[index - 1][0] + margin if index > 0 else 0.0
    upper = points[index + 1][0] - margin if index + 1 < len(points) else 1.0
    points[index] = (
        float(np.clip(x_value, lower, upper)),
        float(np.clip(displacement, -0.24, 0.24)),
    )


def _remove_dewarp_control_point(points: list[tuple[float, float]], index: int) -> bool:
    if len(points) <= 3 or not 0 <= index < len(points):
        return False
    points.pop(index)
    return True


def _move_dewarp_guide_anchor(
    points: list[tuple[float, float]],
    anchor: float,
    delta: float,
    *,
    lower_anchor: float = 0.0,
    upper_anchor: float = 1.0,
) -> float:
    if not points:
        return float(np.clip(anchor + delta, lower_anchor, upper_anchor))
    minimum = min(value for _x, value in points)
    maximum = max(value for _x, value in points)
    lower = max(lower_anchor, -minimum)
    upper = min(upper_anchor, 1.0 - maximum)
    return float(np.clip(anchor + delta, lower, upper))


def _detection_summary(results: list[PageResult]) -> str:
    """Describe detector outcomes without calling fallback pages detected."""
    fallback = sum(result.fallback_reason is not None for result in results)
    return _detection_summary_counts(len(results), fallback)


def _entry_crop_state(entry) -> str:
    state = getattr(entry, "crop_state", None)
    if state in {CROP_STATE_NONE, CROP_STATE_PROPOSED, CROP_STATE_APPLIED}:
        return state
    # Compatibility for lightweight test doubles and pre-v5 in-memory entries.
    return (
        CROP_STATE_APPLIED
        if entry.detected_contour is not None or entry.detected_backend is not None
        else CROP_STATE_NONE
    )


def _entry_has_crop_proposal(entry) -> bool:
    """Return whether an entry has detected corners that are still unapplied."""
    return _entry_crop_state(entry) == CROP_STATE_PROPOSED and entry.detected_contour is not None


def _entry_needs_crop_review(entry) -> bool:
    """Return whether automatic geometry should be checked by an operator."""
    return (
        bool(getattr(entry, "needs_review", False)) or _entry_crop_state(entry) == CROP_STATE_NONE
    )


IMPORT_PREFERENCES_SCHEMA = 1
DEFAULT_IMPORT_PDF_DPI = 300


def _load_import_preferences(path: Path) -> tuple[int, bool]:
    """Load import preferences fail-soft; malformed data restores safe defaults."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return DEFAULT_IMPORT_PDF_DPI, False
    if not isinstance(payload, dict) or payload.get("schemaVersion") != IMPORT_PREFERENCES_SCHEMA:
        return DEFAULT_IMPORT_PDF_DPI, False
    dpi = payload.get("pdfDpi")
    split_spreads = payload.get("splitSpreads")
    if type(dpi) is not int or not 72 <= dpi <= 2400 or type(split_spreads) is not bool:
        return DEFAULT_IMPORT_PDF_DPI, False
    return dpi, split_spreads


def _save_import_preferences(path: Path, *, pdf_dpi: int, split_spreads: bool) -> bool:
    """Atomically persist validated import preferences without blocking the UI on failure."""
    if type(pdf_dpi) is not int or not 72 <= pdf_dpi <= 2400 or type(split_spreads) is not bool:
        return False
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_stage = tempfile.mkstemp(
            prefix=f".{path.name}.stage-",
            suffix=".json",
            dir=path.parent,
        )
    except OSError:
        return False
    stage = Path(raw_stage)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {
                    "schemaVersion": IMPORT_PREFERENCES_SCHEMA,
                    "pdfDpi": pdf_dpi,
                    "splitSpreads": split_spreads,
                },
                stream,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        return True
    except OSError:
        try:
            os.close(descriptor)
        except OSError:
            pass
        return False
    finally:
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass


def _detection_summary_counts(total: int, fallback: int) -> str:
    detected = total - fallback
    if total == 0:
        return "no pages were produced"
    if fallback == 0:
        return f"detected boundaries for all {detected} page(s)"
    if detected == 0:
        return f"kept {fallback} page(s) unchanged because no boundary was confident"
    return f"detected {detected} page(s); kept {fallback} fallback page(s) unchanged"


class UnifiedScanApp(ctk.CTk):
    """Main window for the unified scanner application."""

    def __init__(self) -> None:
        try:
            self._initialize()
        except BaseException:
            try:
                lock = object.__getattribute__(self, "_autosave_lock")
            except (AttributeError, TypeError):
                lock = None
            if lock is not None:
                lock.release()
            if "tk" in object.__getattribute__(self, "__dict__"):
                try:
                    self.destroy()
                except Exception:
                    pass
            raise

    def _initialize(self) -> None:
        super().__init__()
        self.title("UniScan")
        self.geometry("1280x800")
        self.minsize(1024, 680)

        self.autosave_path = default_autosave_path()
        self.import_preferences_path = self.autosave_path.parent / "import_preferences.json"
        try:
            self._autosave_lock = acquire_autosave_lock(self.autosave_path)
        except SessionInUseError:
            self.destroy()
            raise
        self._restore_error: str | None = None
        try:
            self.session, self._session_restored = load_or_create_session(self.autosave_path)
        except ValueError as exc:
            self._restore_error = str(exc)
            self.session = create_persistent_session(self.autosave_path.parent)
            self._session_restored = False
        self.camera: CameraService | None = None
        self.burst_camera: CameraService | None = None
        self._burst_capture_active = False
        self._camera_state_lock = threading.RLock()
        self._camera_opening = False
        self._camera_open_generation = 0
        self._preview_last_seq = 0
        self._camera_health_refreshed_at = 0.0
        self._effective_capture_resolution: tuple[int, int] | None = None
        self.camera_device_names: list[str] = []
        self.camera_device_indices: list[int] = []
        self.camera_modes: list[CameraMode] = []
        self._camera_modes_probing = False
        self._camera_modes_probed_index: int | None = None
        self._camera_resolution_chosen = False
        self.processing_cache = ProcessingStageCache(
            self.autosave_path.parent / "stage_cache",
            max_bytes=512 * 1024 * 1024,
            max_entries=256,
        )
        self.preview_job: str | None = None
        self.preview_photo: ctk.CTkImage | None = None
        self.page_preview_before_photo: ctk.CTkImage | None = None
        self.page_preview_after_photo: ctk.CTkImage | None = None
        self.page_preview_before_image: np.ndarray | None = None
        self.page_preview_after_image: np.ndarray | None = None
        self.pending_split_entry_id: str | None = None
        self.pending_split_ratio: float | None = None
        self.pending_split_revision: int | None = None
        self.review_preview_resize_job: str | None = None
        self.review_preview_render_size: tuple[int, int, int, int] | None = None
        self.review_preview_job: str | None = None
        self.review_preview_thread: threading.Thread | None = None
        self.review_preview_threads: list[threading.Thread] = []
        self.review_preview_cancel_event = threading.Event()
        self.review_preview_generation = 0
        self.review_processing_window: ctk.CTkFrame | None = None
        self.export_dialog_window: ctk.CTkToplevel | None = None
        self.inline_editor_host: ctk.CTkFrame | None = None
        self.inline_editor_close_callback = None
        self.corner_editor_window: ctk.CTkFrame | None = None
        self.corner_source_canvas: tk.Canvas | None = None
        self.corner_preview_canvas: tk.Canvas | None = None
        self.corner_meta_var: tk.StringVar | None = None
        self.corner_prev_button: ctk.CTkButton | None = None
        self.corner_apply_button: ctk.CTkButton | None = None
        self.corner_next_button: ctk.CTkButton | None = None
        self.corner_editor_state: dict[str, object] | None = None
        self.corner_resize_job: str | None = None
        self.split_editor_window: ctk.CTkFrame | None = None
        self.split_source_canvas: tk.Canvas | None = None
        self.split_preview_canvas: tk.Canvas | None = None
        self.split_editor_state: dict[str, object] | None = None
        self.split_resize_job: str | None = None
        self.dewarp_editor_window: ctk.CTkFrame | None = None
        self.dewarp_source_canvas: tk.Canvas | None = None
        self.dewarp_preview_canvas: tk.Canvas | None = None
        self.dewarp_apply_points_button: ctk.CTkButton | None = None
        self.dewarp_resize_job: str | None = None
        self.page_drag_state: dict[str, object] | None = None
        self.live_detector = LiveContourDetector(backend=DEFAULT_LIVE_BACKEND)

        initial_status = "Ready"
        if self._session_restored:
            initial_status = f"Restored {len(self.session)} autosaved page(s)."
            if self.session.restore_warnings:
                initial_status += f" Restore warnings: {len(self.session.restore_warnings)}."
        elif self._restore_error:
            initial_status = f"Autosave restore skipped: {self._restore_error}"
        self.status_var = tk.StringVar(value=initial_status)
        self.page_count_var = tk.StringVar(value="0 pages")
        self.camera_health_var = tk.StringVar(value="Camera: Closed")
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_shots_var = tk.IntVar(value=1)
        self.camera_delay_var = tk.DoubleVar(value=1.0)
        self.camera_resolution = self._default_camera_resolution()
        self.apply_changes_to_all_var = tk.BooleanVar(value=False)
        self.lightweight_preview_var = tk.BooleanVar(value=True)
        self.preview_mode_var = tk.StringVar(value="Processed")
        self.postprocess_var = tk.StringVar(value="None")
        self.lens_mode_var = tk.StringVar(value="Document")
        self.preprocess_preset_var = tk.StringVar(value="Document")
        self.preprocess_contrast_var = tk.DoubleVar(value=1.25)
        self.preprocess_brightness_var = tk.IntVar(value=10)
        self.preprocess_denoise_var = tk.IntVar(value=4)
        self.preprocess_threshold_var = tk.IntVar(value=170)
        self.shadow_method_var = tk.StringVar(value="Automatic (validated)")
        self.binarization_method_var = tk.StringVar(value="None")
        self.binarization_window_var = tk.IntVar(value=31)
        self.binarization_k_var = tk.DoubleVar(value=0.2)
        self._binarization_k_custom = False
        self.despeckle_strength_var = tk.StringVar(value="None")
        self.page_layout_var = tk.StringVar(value="Keep source page")
        self.page_margin_mm_var = tk.DoubleVar(value=10.0)
        self.page_align_x_var = tk.StringVar(value="center")
        self.page_align_y_var = tk.StringVar(value="center")
        self.lighting_summary_var = tk.StringVar(value="Lighting: not analyzed")
        self.stage_settings_var = tk.StringVar(value="Stage settings: document defaults")
        self.orientation_method_var = tk.StringVar(value="Automatic (conservative)")
        self.dewarp_method_var = tk.StringVar(value="Automatic (validated)")
        self.geometry_summary_var = tk.StringVar(value="Wave preview: pending")
        self.split_preview_var = tk.StringVar(value="Split: not previewed")
        self.deskew_method_var = tk.StringVar(value="Hybrid (recommended)")
        self.manual_deskew_angle_var = tk.DoubleVar(value=0.0)
        self.manual_deskew_summary_var = tk.StringVar(value="Manual deskew: 0.0 degrees")
        self._loading_page_recipe = False
        import_pdf_dpi, import_split_spreads = _load_import_preferences(
            self.import_preferences_path
        )
        self.import_pdf_dpi_var = tk.IntVar(value=import_pdf_dpi)
        self.import_two_page_mode_var = tk.BooleanVar(value=import_split_spreads)
        self.import_selected_files: list[str] = []
        self.live_edge_var = tk.BooleanVar(value=False)
        self.live_backend_var = tk.StringVar(value="opencv_quad")
        self.live_status_var = tk.StringVar(value="Detector: Idle")
        self.export_scope_var = tk.StringVar(value="All pages")
        self.export_pdf_path_var = tk.StringVar()
        self.export_dir_var = tk.StringVar()
        self.export_format_var = tk.StringVar(value="png")
        self.export_pdf_dpi_var = tk.IntVar(value=300)
        self.export_dialog_mode_var = tk.StringVar(value="PDF")
        self.export_dialog_scope_var = tk.StringVar(value="All pages")
        self.export_dialog_dpi_var = tk.IntVar(value=300)
        self.export_dialog_format_var = tk.StringVar(value="png")
        self.job_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.job_cancel_event = threading.Event()
        self.job_thread: threading.Thread | None = None
        self._closing = False
        self._close_wait_job: str | None = None
        self._close_deadline: float | None = None
        self.autosave_job: str | None = None
        self._last_autosave_signature: tuple[object, ...] | None = None
        self.tab_review_name = "Workspace"
        self.tab_camera_name = "Camera"

        self._build_ui()
        self._bind_shortcuts()
        self._enable_drag_drop()
        self.on_lens_mode_change(self.lens_mode_var.get())
        self._update_camera_health()
        self.status_var.set(initial_status)
        self.after(120, self._poll_job_queue)
        self.after(500, self._start_startup_diagnostics)
        self.autosave_job = self.after(2000, self._autosave_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self)
        container.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        header = ctk.CTkFrame(container, fg_color=("#dbdbdb", "#2b2b2b"))
        header.pack(fill=ctk.X, padx=12, pady=(10, 4))
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side=ctk.LEFT)
        ctk.CTkLabel(
            brand,
            text="UniScan",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="Capture, clean and export documents",
            text_color=("#60646c", "#a0a4ab"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            textvariable=self.camera_health_var,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(side=ctk.RIGHT, padx=(8, 0))
        ctk.CTkLabel(
            header,
            textvariable=self.page_count_var,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(side=ctk.RIGHT, padx=8)

        toolbar = ctk.CTkFrame(container)
        toolbar.pack(fill=ctk.X, padx=12, pady=(6, 4))
        self.toolbar_add_files_button = ctk.CTkButton(
            toolbar,
            text="+ Add files",
            width=110,
            command=self.quick_add_files,
        )
        self.toolbar_add_files_button.pack(side=ctk.LEFT, padx=(8, 4), pady=8)
        self.toolbar_add_folder_button = ctk.CTkButton(
            toolbar,
            text="Add folder",
            width=105,
            fg_color="transparent",
            border_width=1,
            command=self.quick_add_folder,
        )
        self.toolbar_add_folder_button.pack(side=ctk.LEFT, padx=4, pady=8)
        self.toolbar_paste_button = ctk.CTkButton(
            toolbar,
            text="Paste",
            width=80,
            fg_color="transparent",
            border_width=1,
            command=self.import_from_clipboard,
        )
        self.toolbar_paste_button.pack(side=ctk.LEFT, padx=4, pady=8)
        self.toolbar_camera_button = ctk.CTkButton(
            toolbar,
            text="Camera",
            width=90,
            fg_color="transparent",
            border_width=1,
            command=self.go_to_camera_tab,
        )
        self.toolbar_camera_button.pack(side=ctk.LEFT, padx=4, pady=8)
        self.toolbar_import_options_button = ctk.CTkButton(
            toolbar,
            text="Import options...",
            width=125,
            fg_color="transparent",
            border_width=1,
            command=self.open_import_options_dialog,
        )
        self.toolbar_import_options_button.pack(side=ctk.LEFT, padx=4, pady=8)
        self.toolbar_export_pdf_button = ctk.CTkButton(
            toolbar,
            text="Export PDF",
            width=120,
            fg_color="#2f855a",
            hover_color="#276749",
            command=self.quick_export_pdf,
        )
        self.toolbar_export_pdf_button.pack(side=ctk.RIGHT, padx=(4, 8), pady=8)
        self.toolbar_export_options_button = ctk.CTkButton(
            toolbar,
            text="Export options...",
            width=135,
            fg_color="transparent",
            border_width=1,
            command=self.open_export_dialog,
        )
        self.toolbar_export_options_button.pack(side=ctk.RIGHT, padx=4, pady=8)

        self.status_frame = ctk.CTkFrame(container)
        self.status_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=(0, 12))
        status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=10, pady=7)
        self.cancel_task_button = ctk.CTkButton(
            self.status_frame,
            text="Cancel task",
            width=90,
            height=26,
            fg_color="transparent",
            border_width=1,
            command=self.cancel_current_job,
            state=tk.DISABLED,
        )
        self.cancel_task_button.pack(side=ctk.RIGHT, padx=8, pady=5)

        self.tabs = ctk.CTkTabview(container, command=self._sync_camera_to_active_tab)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(4, 8))

        self.pages_tab = self.tabs.add(self.tab_review_name)
        self.camera_tab = self.tabs.add(self.tab_camera_name)

        self._build_pages_tab(self.pages_tab)
        self._build_capture_tab(self.camera_tab)
        self.tabs.set(self.tab_review_name)

    def go_to_camera_tab(self) -> None:
        """Switch to the in-window Camera tab; the preview starts by itself."""
        self.tabs.set(self.tab_camera_name)
        self._sync_camera_to_active_tab()

    def _sync_camera_to_active_tab(self) -> None:
        """Start the camera on the Camera tab, release it everywhere else.

        Called both by the tab bar (user click) and after programmatic
        ``tabs.set`` calls, which do not fire the tab command.
        """
        try:
            on_camera_tab = self.tabs.get() == self.tab_camera_name
        except tk.TclError:
            on_camera_tab = False
        if on_camera_tab:
            self.start_preview()
        elif (
            self.camera is not None
            or self.__dict__.get("_camera_opening", False)
            or self.preview_job is not None
        ):
            self.close_camera()

    def _build_capture_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkScrollableFrame(tab, width=340)
        controls.grid(row=0, column=0, sticky="ns", padx=(10, 8), pady=10)

        self.camera_health_label = ctk.CTkLabel(
            controls,
            textvariable=self.camera_health_var,
            text_color="#6c757d",
            anchor="w",
        )
        self.camera_health_label.pack(fill=ctk.X, padx=10, pady=(10, 6))

        self.capture_one_button = ctk.CTkButton(
            controls,
            text="Capture Page",
            height=40,
            fg_color="#2f855a",
            hover_color="#276749",
            command=self.capture_one,
        )
        self.capture_one_button.pack(fill=ctk.X, padx=10, pady=(0, 8))

        row_burst = ctk.CTkFrame(controls, fg_color="transparent")
        row_burst.pack(fill=ctk.X, padx=10, pady=(0, 2))
        shots_column = ctk.CTkFrame(row_burst, fg_color="transparent")
        shots_column.pack(side=ctk.LEFT)
        ctk.CTkLabel(shots_column, text=f"Shots (1-{CameraService.MAX_BURST_SHOTS})").pack(
            anchor="w"
        )
        self.shots_entry = ctk.CTkEntry(shots_column, textvariable=self.camera_shots_var, width=70)
        self.shots_entry.pack(anchor="w")
        delay_column = ctk.CTkFrame(row_burst, fg_color="transparent")
        delay_column.pack(side=ctk.LEFT, padx=(10, 0))
        ctk.CTkLabel(delay_column, text="Delay (sec)").pack(anchor="w")
        self.delay_entry = ctk.CTkEntry(delay_column, textvariable=self.camera_delay_var, width=70)
        self.delay_entry.pack(anchor="w")
        ctk.CTkButton(
            row_burst,
            text="Capture Burst",
            width=120,
            command=self.capture_burst,
        ).pack(side=ctk.LEFT, padx=(10, 0), pady=(16, 0))

        ctk.CTkButton(
            controls,
            text="Open Workspace",
            fg_color="transparent",
            border_width=1,
            command=self.go_to_review_tab,
        ).pack(fill=ctk.X, padx=10, pady=(8, 10))

        ctk.CTkLabel(
            controls,
            text="Captured pages land in the Workspace immediately.\n"
            "Processing stays non-destructive until applied.",
            justify="left",
            anchor="w",
        ).pack(fill=ctk.X, padx=10, pady=(0, 8))

        # Live edge detection is off by default: it currently proposes an
        # axis-aligned box rather than a true perspective quad, so drawing it
        # over the preview misleads more than it helps. Page boundaries are
        # still detected per page after capture, in the workspace.
        live_edge_box = ctk.CTkFrame(controls)
        live_edge_box.pack(fill=ctk.X, padx=10, pady=(8, 4))
        ctk.CTkLabel(live_edge_box, text="Live edge detection (experimental)").pack(
            anchor="w", padx=8, pady=(6, 2)
        )
        ctk.CTkCheckBox(
            live_edge_box,
            text="Show document boundaries",
            variable=self.live_edge_var,
            command=self._on_live_edge_toggle,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        row_backend = ctk.CTkFrame(live_edge_box, fg_color="transparent")
        row_backend.pack(fill=ctk.X, padx=8, pady=(0, 4))
        ctk.CTkLabel(row_backend, text="Backend").pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkOptionMenu(
            row_backend,
            values=list(LIVE_BACKEND_CHOICES),
            variable=self.live_backend_var,
            command=self._on_live_backend_change,
            width=140,
        ).pack(side=ctk.LEFT)
        ctk.CTkLabel(live_edge_box, textvariable=self.live_status_var, anchor="w").pack(
            fill=ctk.X, padx=8, pady=(0, 6)
        )

        settings_box = ctk.CTkFrame(controls)
        settings_box.pack(fill=ctk.X, padx=10, pady=(8, 10))
        ctk.CTkLabel(settings_box, text="Camera settings").pack(anchor="w", padx=8, pady=(6, 2))

        row_index = ctk.CTkFrame(settings_box, fg_color="transparent")
        row_index.pack(fill=ctk.X, padx=8, pady=(0, 4))
        ctk.CTkLabel(row_index, text="Device").pack(side=ctk.LEFT, padx=(0, 6))
        self._refresh_camera_device_names()
        self.camera_index_menu = ctk.CTkOptionMenu(
            row_index,
            values=self._device_menu_values(),
            command=self._on_camera_device_selected,
            width=220,
        )
        self.camera_index_menu.set(self._device_menu_selection())
        self.camera_index_menu.pack(side=ctk.LEFT)
        self.camera_identify_button = ctk.CTkButton(
            row_index,
            text="Find cameras",
            width=110,
            command=self._identify_cameras_async,
        )
        self.camera_identify_button.pack(side=ctk.LEFT, padx=(6, 0))

        row_resolution = ctk.CTkFrame(settings_box, fg_color="transparent")
        row_resolution.pack(fill=ctk.X, padx=8, pady=(0, 4))
        ctk.CTkLabel(row_resolution, text="Capture").pack(side=ctk.LEFT, padx=(0, 6))
        self.camera_resolution_menu = ctk.CTkOptionMenu(
            row_resolution,
            values=self._resolution_menu_values(),
            command=self._apply_resolution_string,
            width=190,
        )
        self.camera_resolution_menu.set(self._resolution_menu_selection())
        self.camera_resolution_menu.pack(side=ctk.LEFT)
        self.camera_detect_modes_button = ctk.CTkButton(
            settings_box,
            text="Detect capture modes",
            fg_color="transparent",
            border_width=1,
            command=self._detect_camera_modes_async,
        )
        self.camera_detect_modes_button.pack(fill=ctk.X, padx=8, pady=(0, 4))

        row_custom = ctk.CTkFrame(settings_box, fg_color="transparent")
        row_custom.pack(fill=ctk.X, padx=8, pady=(0, 4))
        self.camera_custom_resolution_var = tk.StringVar(
            value=f"{self.camera_resolution[0]}x{self.camera_resolution[1]}"
        )
        custom_entry = ctk.CTkEntry(
            row_custom, textvariable=self.camera_custom_resolution_var, width=120
        )
        custom_entry.pack(side=ctk.LEFT)
        ctk.CTkButton(
            row_custom,
            text="Set custom",
            width=110,
            command=lambda: self._apply_resolution_string(self.camera_custom_resolution_var.get()),
        ).pack(side=ctk.LEFT, padx=(6, 0))

        ctk.CTkButton(
            settings_box,
            text="Reconnect camera",
            fg_color="transparent",
            border_width=1,
            command=self.start_preview,
        ).pack(fill=ctk.X, padx=8, pady=(2, 8))

        preview_area = ctk.CTkFrame(tab)
        preview_area.grid(row=0, column=1, sticky="nsew", padx=(8, 10), pady=10)
        preview_area.grid_rowconfigure(0, weight=1)
        preview_area.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            preview_area,
            text="No camera frame",
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _build_pages_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=0, minsize=290)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_columnconfigure(2, weight=0, minsize=280)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(tab, width=290)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 6), pady=10)
        left.grid_propagate(False)
        self.workspace_page_list_frame = left

        list_header = ctk.CTkFrame(left, fg_color="transparent")
        list_header.pack(fill=ctk.X, padx=10, pady=(10, 6))
        ctk.CTkLabel(
            list_header,
            text="Pages",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side=ctk.LEFT)
        ctk.CTkLabel(
            list_header,
            textvariable=self.page_count_var,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(side=ctk.RIGHT)
        ctk.CTkLabel(
            left,
            text="⚠  automatic crop not found",
            anchor="w",
            text_color=("#8a5a00", "#d6a84b"),
            font=ctk.CTkFont(size=11),
        ).pack(fill=ctk.X, padx=10, pady=(0, 5))
        self.page_listbox = tk.Listbox(
            left,
            selectmode=tk.EXTENDED,
            width=30,
            height=8,
            bg="#202225",
            fg="#f2f2f2",
            selectbackground="#1f6aa5",
            selectforeground="#ffffff",
            activestyle="none",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#45484d",
            highlightcolor="#1f6aa5",
            font=("Segoe UI", 11),
            exportselection=False,
        )
        self.page_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.page_listbox.bind("<<ListboxSelect>>", self.on_page_select)
        self.page_listbox.bind("<ButtonPress-1>", self._on_page_drag_start, add="+")
        self.page_listbox.bind("<B1-Motion>", self._on_page_drag_motion, add="+")
        self.page_listbox.bind("<ButtonRelease-1>", self._on_page_drag_end, add="+")
        self.page_listbox.bind("<Button-3>", self._show_page_context_menu)

        page_actions = ctk.CTkFrame(left, fg_color="transparent")
        page_actions.pack(fill=ctk.X, padx=10, pady=(0, 4))
        self.move_pages_up_button = ctk.CTkButton(
            page_actions,
            text="Move up",
            width=82,
            command=self.move_selected_up,
        )
        self.move_pages_up_button.pack(side=ctk.LEFT, padx=(0, 4))
        self.move_pages_down_button = ctk.CTkButton(
            page_actions,
            text="Move down",
            width=86,
            command=self.move_selected_down,
        )
        self.move_pages_down_button.pack(side=ctk.LEFT, padx=(0, 4))
        self.delete_pages_button = ctk.CTkButton(
            page_actions,
            text="Delete",
            width=72,
            fg_color="#b42318",
            hover_color="#912018",
            command=self.delete_selected_pages,
        )
        self.delete_pages_button.pack(side=ctk.LEFT)

        self.page_context_menu = tk.Menu(self, tearoff=False)
        self.page_context_menu.add_command(label="Move up", command=self.move_selected_up)
        self.page_context_menu.add_command(label="Move down", command=self.move_selected_down)
        self.page_context_menu.add_separator()
        self.page_context_menu.add_command(label="Delete", command=self.delete_selected_pages)

        edit_actions = ctk.CTkFrame(left, fg_color="transparent")
        edit_actions.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(
            edit_actions,
            text="Rotate left",
            width=118,
            fg_color="transparent",
            border_width=1,
            command=self.rotate_selected_left,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkButton(
            edit_actions,
            text="Rotate right",
            width=118,
            fg_color="transparent",
            border_width=1,
            command=self.rotate_selected_right,
        ).pack(side=ctk.LEFT)

        selection_actions = ctk.CTkFrame(left, fg_color="transparent")
        selection_actions.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(
            selection_actions,
            text="Select all",
            width=118,
            fg_color="transparent",
            border_width=1,
            command=self.select_all_pages,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkButton(
            selection_actions,
            text="Clear",
            width=118,
            fg_color="transparent",
            border_width=1,
            command=self.clear_page_selection,
        ).pack(side=ctk.LEFT)

        preview = ctk.CTkFrame(tab)
        preview.grid(row=0, column=1, sticky="nsew", padx=6, pady=10)
        self.workspace_preview_frame = preview
        preview.grid_rowconfigure(1, weight=1)
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_columnconfigure(1, weight=1)

        preview_toolbar = ctk.CTkFrame(preview, fg_color="transparent")
        preview_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        ctk.CTkLabel(
            preview_toolbar,
            text="Preview",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(side=ctk.LEFT)
        self.preview_mode_selector = ctk.CTkSegmentedButton(
            preview_toolbar,
            values=["Processed", "Original", "Compare"],
            variable=self.preview_mode_var,
            command=self._on_preview_mode_change,
        )
        self.preview_mode_selector.pack(side=ctk.RIGHT)

        self.page_preview_before_frame = ctk.CTkFrame(preview)
        self.page_preview_before_frame.grid_rowconfigure(1, weight=1)
        self.page_preview_before_frame.grid_columnconfigure(0, weight=1)
        self.page_preview_before_title = ctk.CTkLabel(
            self.page_preview_before_frame, text="Original"
        )
        self.page_preview_before_title.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.page_preview_before_label = ctk.CTkLabel(
            self.page_preview_before_frame,
            text="No page selected",
        )
        self.page_preview_before_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.page_preview_after_frame = ctk.CTkFrame(preview)
        self.page_preview_after_frame.grid_rowconfigure(1, weight=1)
        self.page_preview_after_frame.grid_columnconfigure(0, weight=1)
        self.page_preview_after_title = ctk.CTkLabel(
            self.page_preview_after_frame, text="Processed preview"
        )
        self.page_preview_after_title.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.page_preview_after_label = ctk.CTkLabel(
            self.page_preview_after_frame,
            text="No page selected",
        )
        self.page_preview_after_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.page_preview_before_frame.bind(
            "<Configure>",
            self._on_review_preview_resize,
            add="+",
        )
        self.page_preview_after_frame.bind(
            "<Configure>",
            self._on_review_preview_resize,
            add="+",
        )
        self._layout_page_previews()

        processing = ctk.CTkScrollableFrame(tab, width=270, label_text="Processing")
        processing.grid(row=0, column=2, sticky="nsew", padx=(6, 10), pady=10)
        self.workspace_processing_frame = processing

        ctk.CTkLabel(
            processing,
            text="Geometry workflow",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(fill=ctk.X, padx=6, pady=(4, 6))
        ctk.CTkButton(
            processing,
            text="1  Correct spread perspective",
            command=self.open_manual_corners_editor,
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        split_actions = ctk.CTkFrame(processing, fg_color="transparent")
        split_actions.pack(fill=ctk.X, padx=6, pady=(0, 4))
        ctk.CTkButton(
            split_actions,
            text="2  Adjust split",
            width=118,
            command=self.open_split_editor,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        self.apply_split_button = ctk.CTkButton(
            split_actions,
            text="Create 2 pages",
            width=104,
            state=tk.DISABLED,
            command=self.apply_previewed_spread_split,
        )
        self.apply_split_button.pack(side=ctk.LEFT)
        ctk.CTkLabel(
            processing,
            textvariable=self.split_preview_var,
            anchor="w",
            text_color=("#60646c", "#a0a4ab"),
        ).pack(fill=ctk.X, padx=6, pady=(0, 5))
        ctk.CTkButton(
            processing,
            text="3  Page perspective",
            fg_color="transparent",
            border_width=1,
            command=self.open_current_geometry_corners_editor,
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        ctk.CTkButton(
            processing,
            text="4  Edit page waves",
            fg_color="transparent",
            border_width=1,
            command=self.open_dewarp_points_editor,
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        geometry_utilities = ctk.CTkFrame(processing, fg_color="transparent")
        geometry_utilities.pack(fill=ctk.X, padx=6, pady=(0, 4))
        ctk.CTkButton(
            geometry_utilities,
            text="Auto orient",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.auto_orient_selected,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkButton(
            geometry_utilities,
            text="Auto deskew",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.auto_deskew_selected,
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(
            processing,
            text="Detect page boundaries",
            fg_color="transparent",
            border_width=1,
            command=self.open_auto_crop_editor,
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        source_actions = ctk.CTkFrame(processing, fg_color="transparent")
        source_actions.pack(fill=ctk.X, padx=6, pady=(0, 10))
        ctk.CTkButton(
            source_actions,
            text="Replace...",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.replace_selected_page_from_file,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkButton(
            source_actions,
            text="Retake",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.retake_selected_page_from_camera,
        ).pack(side=ctk.LEFT)

        ctk.CTkLabel(processing, text="Document type", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(4, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(LENS_MODE_VALUES),
            variable=self.lens_mode_var,
            command=self.on_lens_mode_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))

        ctk.CTkLabel(processing, text="Output style", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            variable=self.postprocess_var,
            command=self._on_postprocess_mode_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))

        ctk.CTkLabel(processing, text="Cleanup preset", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(PREPROCESS_PRESETS.keys()),
            variable=self.preprocess_preset_var,
            command=self.on_preprocess_preset_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 10))

        ctk.CTkCheckBox(
            processing,
            text="Apply to all pages",
            variable=self.apply_changes_to_all_var,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        ctk.CTkCheckBox(
            processing,
            text="Fast preview (approximate)",
            variable=self.lightweight_preview_var,
            command=self.update_page_preview,
        ).pack(anchor="w", padx=6, pady=(0, 6))
        ctk.CTkLabel(
            processing,
            textvariable=self.stage_settings_var,
            anchor="w",
            justify="left",
            wraplength=230,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))
        ctk.CTkLabel(processing, text="Page orientation", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(ORIENTATION_UI_METHODS),
            variable=self.orientation_method_var,
            command=lambda _value: self.update_page_preview(),
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))
        ctk.CTkLabel(processing, text="Small-angle deskew", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(DESKEW_UI_METHODS),
            variable=self.deskew_method_var,
            command=lambda _value: self.update_page_preview(),
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        ctk.CTkSlider(
            processing,
            from_=-10.0,
            to=10.0,
            number_of_steps=200,
            variable=self.manual_deskew_angle_var,
            command=self._on_manual_deskew_angle_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 2))
        ctk.CTkLabel(
            processing,
            textvariable=self.manual_deskew_summary_var,
            anchor="w",
            text_color=("#60646c", "#a0a4ab"),
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))
        ctk.CTkLabel(processing, text="Remove page waves", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(DEWARP_UI_METHODS),
            variable=self.dewarp_method_var,
            command=self._on_dewarp_method_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 5))
        ctk.CTkLabel(
            processing,
            textvariable=self.geometry_summary_var,
            anchor="w",
            justify="left",
            wraplength=230,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(fill=ctk.X, padx=6, pady=(0, 10))

        ctk.CTkLabel(processing, text="Even page lighting", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(SHADOW_UI_METHODS),
            variable=self.shadow_method_var,
            command=lambda _value: self.update_page_preview(),
        ).pack(fill=ctk.X, padx=6, pady=(0, 10))

        ctk.CTkLabel(processing, text="Binarization", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(BINARIZATION_UI_METHODS),
            variable=self.binarization_method_var,
            command=self._on_binarization_method_change,
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))

        ctk.CTkLabel(processing, text="Despeckle", anchor="w").pack(fill=ctk.X, padx=6, pady=(0, 2))
        ctk.CTkOptionMenu(
            processing,
            values=list(DESPECKLE_UI_STRENGTHS),
            variable=self.despeckle_strength_var,
            command=lambda _value: self.update_page_preview(),
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))

        ctk.CTkLabel(processing, text="Standard page layout", anchor="w").pack(
            fill=ctk.X, padx=6, pady=(0, 2)
        )
        ctk.CTkOptionMenu(
            processing,
            values=list(PAGE_LAYOUT_UI_METHODS),
            variable=self.page_layout_var,
            command=lambda _value: self.update_page_preview(),
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))
        ctk.CTkButton(
            processing,
            text="Analyze lighting",
            fg_color="transparent",
            border_width=1,
            command=self.analyze_selected_page_lighting,
        ).pack(fill=ctk.X, padx=6, pady=(0, 4))
        ctk.CTkLabel(
            processing,
            textvariable=self.lighting_summary_var,
            anchor="w",
            justify="left",
            wraplength=230,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(fill=ctk.X, padx=6, pady=(0, 10))

        processing_actions = ctk.CTkFrame(processing, fg_color="transparent")
        processing_actions.pack(fill=ctk.X, padx=6, pady=(0, 6))
        ctk.CTkButton(
            processing_actions,
            text="Preview",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.update_page_preview,
        ).pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkButton(
            processing_actions,
            text="Advanced processing",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.open_review_processing_dialog,
        ).pack(side=ctk.LEFT)
        self.apply_processing_button = ctk.CTkButton(
            processing,
            text="Apply preview to pages",
            command=self.apply_review_changes,
        )
        self.apply_processing_button.pack(fill=ctk.X, padx=6, pady=(0, 14))

        self.inline_editor_host = ctk.CTkFrame(tab)
        self.inline_editor_host.grid_rowconfigure(0, weight=1)
        self.inline_editor_host.grid_columnconfigure(0, weight=1)
        self.refresh_page_list()

    def _bind_shortcuts(self) -> None:
        """Bind the common document actions without stealing text-entry shortcuts."""
        self.bind("<Control-o>", lambda _event: self._run_shortcut(self.quick_add_files))
        self.bind("<Control-Shift-O>", lambda _event: self._run_shortcut(self.quick_add_folder))
        self.bind("<Control-Shift-C>", lambda _event: self._run_shortcut(self.capture_one))
        self.bind("<Control-e>", lambda _event: self._run_shortcut(self.quick_export_pdf))
        self.bind("<F5>", lambda _event: self._run_shortcut(self.update_page_preview))

        self.page_listbox.bind(
            "<Delete>",
            lambda _event: self._run_shortcut(self.delete_selected_pages),
        )
        self.page_listbox.bind(
            "<Control-Left>",
            lambda _event: self._run_shortcut(self.rotate_selected_left),
        )
        self.page_listbox.bind(
            "<Control-Right>",
            lambda _event: self._run_shortcut(self.rotate_selected_right),
        )
        self.page_listbox.bind(
            "<Alt-Up>",
            lambda _event: self._run_shortcut(self.move_selected_up),
        )
        self.page_listbox.bind(
            "<Alt-Down>",
            lambda _event: self._run_shortcut(self.move_selected_down),
        )
        self.page_listbox.bind(
            "<Control-a>",
            lambda _event: self._run_shortcut(self.select_all_pages),
        )

    @staticmethod
    def _run_shortcut(command) -> str:
        command()
        return "break"

    def _on_close(self) -> None:
        if self._closing:
            return
        inline_close = self.inline_editor_close_callback
        if inline_close is not None:
            inline_close()
            if self.inline_editor_close_callback is inline_close:
                return
        self._closing = True
        self.job_cancel_event.set()
        self.stop_preview()
        self._cancel_active_burst()
        self._release_camera_handle()
        self._cancel_review_page_preview()
        if (
            self.review_processing_window is not None
            and self.review_processing_window.winfo_exists()
        ):
            self.review_processing_window.destroy()
            self.review_processing_window = None
        if self.export_dialog_window is not None and self.export_dialog_window.winfo_exists():
            self.export_dialog_window.destroy()
        self.export_dialog_window = None
        if self.autosave_job is not None:
            self.after_cancel(self.autosave_job)
            self.autosave_job = None
        if self.review_preview_resize_job is not None:
            self.after_cancel(self.review_preview_resize_job)
            self.review_preview_resize_job = None
        if self.corner_resize_job is not None:
            self.after_cancel(self.corner_resize_job)
            self.corner_resize_job = None
        if self.dewarp_resize_job is not None:
            self.after_cancel(self.dewarp_resize_job)
            self.dewarp_resize_job = None
        corner_window = self.corner_editor_window
        if corner_window is not None:
            try:
                if corner_window.winfo_exists():
                    corner_window.destroy()
            except tk.TclError:
                pass
            self.corner_editor_window = None
            self.corner_source_canvas = None
            self.corner_preview_canvas = None
            self.corner_meta_var = None
            self.corner_prev_button = None
            self.corner_next_button = None
        dewarp_window = self.dewarp_editor_window
        if dewarp_window is not None:
            try:
                if dewarp_window.winfo_exists():
                    dewarp_window.destroy()
            except tk.TclError:
                pass
            self.dewarp_editor_window = None
            self.dewarp_source_canvas = None
            self.dewarp_preview_canvas = None
        self._set_status("Closing: waiting for background work to stop...")
        self._close_deadline = time.monotonic() + 5.0
        self._finish_close_when_idle()

    def _finish_close_when_idle(self) -> None:
        workers = [self.job_thread, *self.review_preview_threads]
        alive = any(worker is not None and worker.is_alive() for worker in workers)
        if alive and (self._close_deadline is None or time.monotonic() < self._close_deadline):
            self._close_wait_job = self.after(25, self._finish_close_when_idle)
            return
        # Workers receive immutable arrays/files and never mutate CaptureSession.
        # After the bounded grace period it is therefore safe to finalize assets
        # even if an uncooperative third-party call has not returned yet.
        self._close_wait_job = None
        try:
            if self.session.has_recoverable_state:
                self.session.save_manifest(self.autosave_path)
                self.session.close(preserve=True)
            else:
                self.autosave_path.unlink(missing_ok=True)
                self.session.close(preserve=False)
        except Exception:
            self.session.store.close()
        finally:
            self._autosave_lock.release()
            self.destroy()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _autosave_tick(self) -> None:
        try:
            signature = tuple(
                (
                    entry.entry_id,
                    entry.revision,
                    entry.name,
                    entry.selected,
                    entry.detected_backend,
                )
                for entry in self.session.entries
            )
            if signature != self._last_autosave_signature:
                if self.session.has_recoverable_state:
                    self.session.save_manifest(self.autosave_path)
                else:
                    self.autosave_path.unlink(missing_ok=True)
                self._last_autosave_signature = signature
        except Exception as exc:
            self._set_status(f"Autosave failed: {exc}")
        finally:
            if self.winfo_exists():
                self.autosave_job = self.after(2000, self._autosave_tick)

    def _enable_drag_drop(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            TkinterDnD._require(self)
            targets = (
                self.page_listbox,
                getattr(self.page_preview_before_label, "_label", self.page_preview_before_label),
                getattr(self.page_preview_after_label, "_label", self.page_preview_after_label),
            )
            for target in targets:
                TkinterDnD.DnDWrapper.drop_target_register(target, DND_FILES)
                TkinterDnD.DnDWrapper.dnd_bind(target, "<<Drop>>", self._on_drop_files)
        except Exception as exc:
            self._drag_drop_error = str(exc)
        else:
            self._drag_drop_error = None

    def _camera_health_detail(self) -> str | None:
        camera = self.camera
        if camera is None:
            return None
        resolution = getattr(camera, "effective_resolution", None) or getattr(
            camera, "resolution", None
        )
        parts: list[str] = []
        if resolution:
            parts.append(f"{resolution[0]}x{resolution[1]}")
        fps = getattr(camera, "measured_fps", None)
        if fps:
            parts.append(f"{fps:.0f} fps")
        detail = " @ ".join(parts) if parts else None
        capture = self.__dict__.get("_effective_capture_resolution") or self.__dict__.get(
            "camera_resolution"
        )
        if detail and capture and tuple(capture) != tuple(resolution or ()):
            detail += f" - capture {capture[0]}x{capture[1]}"
        return detail

    def _update_camera_health(self, error_text: str | None = None) -> None:
        state = camera_health_state(
            is_open=(self.camera is not None or getattr(self, "burst_camera", None) is not None),
            is_previewing=self.preview_job is not None,
            is_opening=bool(self.__dict__.get("_camera_opening", False)),
            error_text=error_text,
            detail=self._camera_health_detail(),
        )
        self._camera_health_refreshed_at = time.monotonic()
        self.camera_health_var.set(state.label)
        label = getattr(self, "camera_health_label", None)
        if label is not None:
            try:
                if label.winfo_exists():
                    label.configure(text_color=state.color)
            except tk.TclError:
                self.camera_health_label = None

    def go_to_review_tab(self) -> None:
        self.tabs.set(self.tab_review_name)
        self._sync_camera_to_active_tab()
        self.lift()

    def open_import_options_dialog(self) -> None:
        """Edit import-only settings without starting a file operation."""
        window = ctk.CTkToplevel(self)
        window.title("Import options")
        window.geometry("420x245")
        window.resizable(False, False)
        window.transient(self)
        dpi_var = tk.IntVar(value=int(self.import_pdf_dpi_var.get()))
        split_var = tk.BooleanVar(value=bool(self.import_two_page_mode_var.get()))

        ctk.CTkLabel(
            window,
            text="Import options",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(fill=ctk.X, padx=16, pady=(16, 10))
        dpi_row = ctk.CTkFrame(window, fg_color="transparent")
        dpi_row.pack(fill=ctk.X, padx=16, pady=(0, 10))
        ctk.CTkLabel(dpi_row, text="PDF rasterization DPI").pack(side=ctk.LEFT)
        ctk.CTkEntry(dpi_row, textvariable=dpi_var, width=100).pack(side=ctk.RIGHT)
        ctk.CTkCheckBox(
            window,
            text="Split confidently detected book spreads during import",
            variable=split_var,
        ).pack(anchor="w", padx=16, pady=(0, 8))
        ctk.CTkLabel(
            window,
            text="Split is off by default. Imported pixels remain unchanged unless you enable it.",
            justify="left",
            wraplength=380,
            text_color=("#60646c", "#a0a4ab"),
        ).pack(anchor="w", padx=16)

        def apply_options() -> None:
            try:
                dpi = int(dpi_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Import Options", "PDF DPI must be a whole number.")
                return
            if not 72 <= dpi <= 2400:
                messagebox.showerror("Import Options", "PDF DPI must be between 72 and 2400.")
                return
            self.import_pdf_dpi_var.set(dpi)
            self.import_two_page_mode_var.set(bool(split_var.get()))
            saved = _save_import_preferences(
                self.import_preferences_path,
                pdf_dpi=dpi,
                split_spreads=bool(split_var.get()),
            )
            split_label = "enabled" if split_var.get() else "disabled"
            persistence = "" if saved else " Preferences could not be persisted."
            self._set_status(
                f"Import options saved: PDF {dpi} DPI, spread split {split_label}.{persistence}"
            )
            window.destroy()

        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(side=ctk.BOTTOM, fill=ctk.X, padx=16, pady=16)
        ctk.CTkButton(
            actions,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=window.destroy,
        ).pack(side=ctk.RIGHT)
        ctk.CTkButton(actions, text="Save", command=apply_options).pack(side=ctk.RIGHT, padx=6)
        window.grab_set()

    def quick_add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Add images or PDFs",
            filetypes=[
                ("Images and PDF", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp *.pdf"),
                ("All files", "*.*"),
            ],
            multiple=True,
        )
        if not files:
            return
        self.import_selected_files = self._normalize_selected_files(files)
        self._import_paths(paths=[Path(path) for path in self.import_selected_files])

    def quick_add_folder(self) -> None:
        folder_raw = filedialog.askdirectory(title="Add a folder of images or PDFs")
        if not folder_raw:
            return
        folder = Path(folder_raw)
        paths = list_supported_in_folder(folder)
        if not paths:
            messagebox.showinfo("Add Folder", "No supported images or PDFs were found.")
            return
        self._import_paths(paths=paths)

    def _sync_lens_mode_from_controls(self) -> None:
        inferred = infer_lens_mode(self.preprocess_preset_var.get(), self.postprocess_var.get())
        self.lens_mode_var.set(inferred)

    def _on_postprocess_mode_change(self, _value: str) -> None:
        self._sync_lens_mode_from_controls()
        self.update_page_preview()

    def _on_dewarp_method_change(self, _value: str) -> None:
        self.geometry_summary_var.set("Wave preview: pending")
        self.update_page_preview()

    def on_lens_mode_change(self, mode_name: str) -> None:
        profile = resolve_lens_mode_profile(mode_name)
        if profile is None:
            self._set_status("Lens mode set to Custom (manual controls).")
            self.update_page_preview()
            return

        self.preprocess_preset_var.set(profile.preset_name)
        self.postprocess_var.set(profile.postprocess_name)
        self._apply_preprocess_preset_values(profile.preset_name)
        self._sync_lens_mode_from_controls()
        self._set_status(f"Lens mode set to {mode_name}.")
        self.update_page_preview()

    def _apply_preprocess_preset_values(self, preset_name: str) -> bool:
        preset = PREPROCESS_PRESETS.get(preset_name)
        if preset is None:
            return False
        self.preprocess_contrast_var.set(float(preset.contrast))
        self.preprocess_brightness_var.set(int(preset.brightness))
        self.preprocess_denoise_var.set(int(preset.denoise))
        self.preprocess_threshold_var.set(int(preset.threshold))
        return True

    def on_preprocess_preset_change(self, preset_name: str) -> None:
        if not self._apply_preprocess_preset_values(preset_name):
            return

        # Keep preset selection aligned with the canonical lens profiles.  In
        # particular, Whiteboard must retain marker colour (`postprocess=None`).
        matching_profile = next(
            (
                profile
                for mode_name in LENS_MODE_VALUES
                if (profile := resolve_lens_mode_profile(mode_name)) is not None
                and profile.preset_name == preset_name
            ),
            None,
        )
        if matching_profile is not None:
            self.postprocess_var.set(matching_profile.postprocess_name)
        self._sync_lens_mode_from_controls()
        self.update_page_preview()

    def _set_job_display(
        self, *, stage: str | None = None, current: str | None = None, progress: int | None = None
    ) -> None:
        parts: list[str] = []
        if stage:
            parts.append(stage)
        if current:
            parts.append(current)
        if progress is not None:
            p = max(0, min(100, int(progress)))
            parts.append(f"{p}%")
        if parts:
            self._set_status(" | ".join(parts))

    def _start_background_job(self, name: str, worker, on_done, *, on_error=None) -> bool:
        if self.job_thread is not None:
            messagebox.showwarning("Busy", "Another background job is already running.")
            return False

        self.job_cancel_event.clear()
        self.cancel_task_button.configure(state=tk.NORMAL)
        self._set_job_display(stage=name, current="Starting...", progress=0)

        def emit(
            stage: str | None = None, current: str | None = None, progress: int | None = None
        ) -> None:
            self.job_queue.put(("progress", (stage, current, progress)))

        def run() -> None:
            try:
                result = worker(emit, self.job_cancel_event.is_set)
                self.job_queue.put(("done", (on_done, result, name)))
            except Exception as exc:
                self.job_queue.put(("error", (name, str(exc), on_error)))

        self.job_thread = threading.Thread(target=run, daemon=True)
        self.job_thread.start()
        return True

    def _poll_job_queue(self) -> None:
        try:
            for _ in range(6):
                kind, payload = self.job_queue.get_nowait()
                if self._closing:
                    continue
                if kind == "progress":
                    stage, current, progress = payload
                    self._set_job_display(stage=stage, current=current, progress=progress)
                elif kind == "done":
                    on_done, result, name = payload
                    self.job_thread = None
                    try:
                        on_done(result)
                    except Exception as exc:
                        self._set_status(f"{name} failed: {exc}")
                        messagebox.showerror(f"{name} Error", str(exc))
                    finally:
                        self.cancel_task_button.configure(state=tk.DISABLED)
                elif kind == "error":
                    self.cancel_task_button.configure(state=tk.DISABLED)
                    self.job_thread = None
                    name, text, on_error = payload
                    if on_error is not None:
                        on_error()
                    if "Cancelled by user." in text:
                        self._set_job_display(stage=f"{name}: cancelled", current=text, progress=0)
                        self._set_status(f"{name} cancelled")
                        if name == "Import":
                            self.refresh_page_list(keep_index=len(self.session) - 1)
                    else:
                        self._set_job_display(stage=f"{name}: error", current=text, progress=0)
                        self._set_status(f"{name} failed")
                        messagebox.showerror(f"{name} Error", text)
                elif kind == "diagnostics":
                    report = payload
                    if not report.ok:
                        failed = ", ".join(
                            check.name for check in report.checks if check.blocking and not check.ok
                        )
                        if failed:
                            self._set_status(
                                f"Startup diagnostics failed: {failed}. Run 'uniscan doctor'."
                            )
                elif kind == "review_preview":
                    generation, image, diagnostics, error = payload
                    self._handle_review_preview_result(generation, image, diagnostics, error)
        except queue.Empty:
            pass
        finally:
            if not self._closing and self.winfo_exists():
                self.after(40, self._poll_job_queue)

    def _start_startup_diagnostics(self) -> None:
        def run() -> None:
            self.job_queue.put(("diagnostics", run_diagnostics()))

        threading.Thread(target=run, daemon=True, name="uniscan-diagnostics").start()

    def cancel_current_job(self) -> None:
        if self.job_thread is None or not self.job_thread.is_alive():
            self._set_status("No running job.")
            return
        self.job_cancel_event.set()
        self._set_job_display(current="Cancellation requested...")
        self._set_status("Cancellation requested")

    @staticmethod
    def _default_camera_resolution() -> tuple[int, int]:
        """Default capture size.

        The camera streams and shoots at one resolution, so this is a quality
        vs. frame-rate choice. Webcams commonly sustain full frame rate up to
        1080p and collapse to a few frames per second above it, so 1080p is
        the highest size that keeps the preview live and shots instant. Larger
        sizes remain selectable in the camera settings.
        """
        return DEFAULT_CAPTURE_RESOLUTION

    def _max_camera_resolution(self) -> tuple[int, int]:
        """Return the configured resolution (legacy method name retained)."""
        return getattr(self, "camera_resolution", self._default_camera_resolution())

    # Capture modes -------------------------------------------------------

    @property
    def _camera_modes_path(self) -> Path:
        return self.autosave_path.parent / "camera_modes.json"

    def _load_camera_modes(self, index: int) -> list[CameraMode]:
        """Cached measurements for one device, or an empty list."""
        try:
            payload = json.loads(self._camera_modes_path.read_text(encoding="utf-8"))
            entries = payload[str(index)]
        except Exception:
            return []
        modes: list[CameraMode] = []
        for entry in entries:
            try:
                modes.append(
                    CameraMode(
                        requested=(int(entry["requested"][0]), int(entry["requested"][1])),
                        granted=(int(entry["granted"][0]), int(entry["granted"][1])),
                        fps=float(entry["fps"]),
                    )
                )
            except Exception:
                return []  # partial or stale cache: measure again
        return modes

    def _save_camera_modes(self, index: int, modes: list[CameraMode]) -> None:
        path = self._camera_modes_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}
        payload[str(index)] = [
            {"requested": list(mode.requested), "granted": list(mode.granted), "fps": mode.fps}
            for mode in modes
        ]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass  # a lost cache only costs one re-probe

    def _resolution_menu_values(self) -> list[str]:
        modes = self.__dict__.get("camera_modes") or []
        if modes:
            return [mode.label for mode in modes]
        return RESOLUTIONS

    def _resolution_menu_selection(self) -> str:
        """Menu text for the resolution currently in use."""
        current = self._max_camera_resolution()
        for mode in self.__dict__.get("camera_modes") or []:
            if tuple(mode.requested) == tuple(current) or tuple(mode.granted) == tuple(current):
                return mode.label
        return f"{current[0]}x{current[1]}"

    def _refresh_resolution_menu(self) -> None:
        # __dict__.get, not getattr: tkinter's Misc.__getattr__ recurses on
        # instances built without a Tk window.
        menu = self.__dict__.get("camera_resolution_menu")
        if menu is None:
            return
        try:
            if menu.winfo_exists():
                menu.configure(values=self._resolution_menu_values())
                menu.set(self._resolution_menu_selection())
        except tk.TclError:
            pass

    def _detect_camera_modes_async(self, *, on_done=None) -> None:
        """Measure the device's capture modes off the UI thread.

        The device is exclusive, so the live stream is released first; the
        preview restarts on the chosen mode once the measurements land.
        """
        if self.__dict__.get("_camera_modes_probing"):
            return
        if self._burst_is_active() or self.__dict__.get("job_thread") is not None:
            self._set_status("Camera is busy; detect capture modes when it is idle.")
            return
        index = int(self.camera_index_var.get())
        self._camera_modes_probing = True
        button = self.__dict__.get("camera_detect_modes_button")
        if button is not None:
            try:
                button.configure(state=tk.DISABLED, text="Detecting...")
            except tk.TclError:
                button = None
        was_previewing = self.preview_job is not None
        self.close_camera()
        self._set_status(f"Detecting {self._device_label(index)} capture modes...")
        found: list[list[CameraMode]] = []
        progress: list[tuple[int, int]] = []

        def work() -> None:
            try:
                found.append(
                    CameraService.probe_modes(
                        index=index,
                        # Tk variables are not thread-safe; the poll below
                        # reads this and updates the status bar.
                        on_progress=lambda done, total: progress.append((done, total)),
                    )
                )
            except Exception:
                found.append([])

        thread = threading.Thread(target=work, name="CameraModeProbe", daemon=True)
        thread.start()

        def poll() -> None:
            if thread.is_alive():
                if progress:
                    done, total = progress[-1]
                    self._set_status(
                        f"Detecting {self._device_label(index)} capture modes... ({done}/{total})"
                    )
                self.after(100, poll)
                return
            self._camera_modes_probing = False
            if button is not None:
                try:
                    if button.winfo_exists():
                        button.configure(state=tk.NORMAL, text="Detect capture modes")
                except tk.TclError:
                    pass
            modes = found[0] if found else []
            # Remember the attempt either way: a device that cannot be measured
            # (busy, unplugged) must not send the caller back here in a loop.
            self._camera_modes_probed_index = index
            if modes:
                self.camera_modes = modes
                self._save_camera_modes(index, modes)
                best = best_realtime_mode(modes)
                if best is not None:
                    self.camera_resolution = best.granted
                self._refresh_resolution_menu()
                self._set_status(f"Capture mode: {self._resolution_menu_selection()}")
            else:
                self._set_status("Could not detect capture modes; keeping current settings.")
            if on_done is not None:
                on_done()
            elif was_previewing:
                self.start_preview()

        self.after(100, poll)

    def _apply_cached_camera_modes(self, index: int) -> bool:
        """Adopt cached measurements for a device. True when any were found."""
        modes = self._load_camera_modes(index)
        if not modes:
            self.camera_modes = []
            return False
        self.camera_modes = modes
        best = best_realtime_mode(modes)
        if best is not None and not self.__dict__.get("_camera_resolution_chosen", False):
            self.camera_resolution = best.granted
        self._refresh_resolution_menu()
        return True

    def _camera_guard(self):
        lock = self.__dict__.get("_camera_state_lock")
        if lock is None:
            lock = threading.RLock()
            self._camera_state_lock = lock
        return lock

    def _burst_is_active(self) -> bool:
        with self._camera_guard():
            return bool(self.__dict__.get("_burst_capture_active", False))

    def _begin_burst_capture(self) -> None:
        with self._camera_guard():
            if self.__dict__.get("_burst_capture_active", False):
                raise RuntimeError("Burst capture is already in progress.")
            self._burst_capture_active = True

    def _end_burst_capture(self) -> None:
        with self._camera_guard():
            self._burst_capture_active = False
            camera = self.__dict__.get("burst_camera")
            self.burst_camera = None
        if camera is not None:
            camera.release()
        self._update_camera_health()

    def _cancel_active_burst(self) -> None:
        if not self._burst_is_active():
            return
        cancel_event = self.__dict__.get("job_cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        with self._camera_guard():
            camera = self.__dict__.get("burst_camera")
        if camera is not None:
            camera.release()

    def _ensure_camera(self) -> CameraService:
        """Blocking open/reuse of the shared camera with a running stream.

        Prefer :meth:`_open_camera_async` from UI callbacks; this stays for
        capture paths where the camera is normally already open.
        """
        if self._burst_is_active():
            raise RuntimeError("Camera is busy with burst capture.")
        return self._ensure_camera_for(
            int(self.camera_index_var.get()), self._max_camera_resolution()
        )

    def _ensure_camera_for(self, index: int, resolution: tuple[int, int]) -> CameraService:
        """Open (or refresh) the shared camera and start its frame stream."""
        camera = self.camera
        if camera is None or camera.index != index:
            replaced = camera
            camera = CameraService(index=index, resolution=resolution)
            camera.open()
            # Publish only after a successful open so a failure leaves the
            # previous handle (if any) usable.
            with self._camera_guard():
                if replaced is not None and self.camera is replaced:
                    replaced.release()
                self.camera = camera
        elif camera.resolution != resolution:
            camera.set_resolution(resolution)
        elif camera.stream_error is not None:
            camera.open()
        camera.start_stream()
        return camera

    def _open_camera_async(self, *, on_ready=None) -> None:
        """Open the camera off the UI thread; the Tk thread polls for the result."""
        if self._camera_opening or self.__dict__.get("_camera_modes_probing"):
            return
        if self._burst_is_active():
            self._set_status("Camera is busy with burst capture.")
            return
        index = int(self.camera_index_var.get())
        if (
            not self.__dict__.get("camera_modes")
            and self.__dict__.get("_camera_modes_probed_index") != index
            and not self._apply_cached_camera_modes(index)
        ):
            # First use of this device: measure what it can actually deliver,
            # then open on the best real-time mode. Cached from then on, and
            # attempted at most once per device so a failed probe still opens.
            self._detect_camera_modes_async(
                on_done=lambda: self._open_camera_async(on_ready=on_ready)
            )
            return
        # One resolution for preview and capture: a shot is then the very frame
        # the user is looking at, taken with no device reconfiguration.
        resolution = self._max_camera_resolution()
        generation = self._camera_open_generation
        self._camera_opening = True
        self._update_camera_health()
        self._show_preview_placeholder(f"Opening {self._device_label(index)}...")
        result: dict[str, object] = {}

        def work() -> None:
            try:
                result["camera"] = self._ensure_camera_for(index, resolution)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=work, name="CameraOpen", daemon=True)
        thread.start()

        def poll() -> None:
            if thread.is_alive():
                self.after(50, poll)
                return
            self._camera_opening = False
            camera = result.get("camera")
            if generation != self._camera_open_generation:
                # The camera was closed while opening; drop the fresh handle.
                if isinstance(camera, CameraService):
                    camera.release()
                    with self._camera_guard():
                        if self.camera is camera:
                            self.camera = None
                self._update_camera_health()
                return
            error = result.get("error")
            if error is not None:
                self._update_camera_health(error_text=str(error))
                self._show_preview_placeholder("No camera frame")
                self._set_status(f"Camera open failed: {error}")
                return
            self._update_camera_health()
            if on_ready is not None:
                on_ready()

        self.after(50, poll)

    def _show_preview_placeholder(self, text: str) -> None:
        """Update the preview label text while no frame is being shown."""
        if self.preview_photo is not None:
            return
        label = getattr(self, "preview_label", None)
        if label is None:
            return
        try:
            if label.winfo_exists():
                label.configure(text=text)
        except tk.TclError:
            pass

    def _release_camera_handle(self) -> None:
        with self._camera_guard():
            camera = self.camera
            self.camera = None
        if camera is not None:
            camera.release()

    def open_camera(self) -> None:
        """Open the camera and start the live preview without blocking the UI."""
        self.start_preview()

    def close_camera(self) -> None:
        # Invalidate any in-flight async open. (__dict__.get, not getattr:
        # tkinter's Misc.__getattr__ recurses on partially built instances.)
        self._camera_open_generation = self.__dict__.get("_camera_open_generation", 0) + 1
        self._camera_opening = False
        self.stop_preview()
        self._cancel_active_burst()
        self._release_camera_handle()
        self._update_camera_health()
        self._set_status("Camera closed")

    def start_preview(self) -> None:
        self._open_camera_async(on_ready=self._begin_preview)

    def _begin_preview(self) -> None:
        self._preview_last_seq = 0
        # The detector thread only runs while its overlay is switched on.
        if self.live_edge_var.get():
            self.live_detector.set_backend(self.live_backend_var.get())
            self.live_detector.start()
            self.live_status_var.set("Detector: Searching")
        else:
            self.live_status_var.set("Detector: Off")
        if self.preview_job is None:
            self._preview_loop()
        self._update_camera_health()
        self._set_status("Preview started")

    def _on_live_edge_toggle(self) -> None:
        """Start or stop the detector with its overlay checkbox."""
        if not self.live_edge_var.get():
            self.live_detector.stop()
            self.live_status_var.set("Detector: Off")
            return
        if self.preview_job is not None:
            self.live_detector.set_backend(self.live_backend_var.get())
            self.live_detector.start()
            self.live_status_var.set("Detector: Searching")

    def stop_preview(self) -> None:
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        self.live_detector.stop()
        self.live_status_var.set("Detector: Idle")
        self._update_camera_health()
        self._set_status("Preview stopped")

    def _preview_loop(self) -> None:
        started_at = time.perf_counter()
        camera = self.camera
        if camera is None:
            self.preview_job = None
            self._update_camera_health()
            return
        stream_error = getattr(camera, "stream_error", None)
        if stream_error:
            self.preview_job = None
            self.live_detector.stop()
            self._update_camera_health(error_text=stream_error)
            self._set_status(f"Camera preview stopped: {stream_error}")
            return
        info = camera.latest_frame_info()
        if info is not None and info.seq != self._preview_last_seq:
            # Frames arrive on the stream thread; the UI only renders here when
            # a new one is available, so ticks never block on camera I/O.
            self._preview_last_seq = info.seq
            if self.live_edge_var.get():
                self.live_detector.submit(info.frame)
            preview = self._preview_image_with_contour(info.frame)
            self._show_in_preview(preview)
        if time.monotonic() - self._camera_health_refreshed_at >= CAMERA_HEALTH_REFRESH_SEC:
            self._update_camera_health()
        # Subtract the render cost from the interval, otherwise every frame
        # pays "render + full wait" and the preview settles below camera rate.
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self.preview_job = self.after(max(1, int(PREVIEW_WAIT_MS - elapsed_ms)), self._preview_loop)

    def _on_live_backend_change(self, value: str) -> None:
        try:
            self.live_detector.set_backend(value)
            self.live_status_var.set(f"Detector: Switched to {value}")
        except ValueError as exc:
            messagebox.showerror("Live Detector", str(exc))

    def _preview_image_with_contour(self, frame: np.ndarray) -> np.ndarray:
        if not self.live_edge_var.get():
            self.live_status_var.set("Detector: Off")
            return frame
        contour, _age_ms = self.live_detector.latest()
        if contour is None:
            self.live_status_var.set(f"Detector: Searching ({self.live_detector.backend})")
            return frame
        self.live_status_var.set(f"Detector: Detected ({self.live_detector.backend})")
        return draw_quad_overlay(frame, contour)

    def _current_preprocess_settings(self) -> PreprocessSettings:
        preset_name = self.preprocess_preset_var.get()
        preset = PREPROCESS_PRESETS.get(preset_name, PREPROCESS_PRESETS["Custom"])
        apply_threshold = bool(
            preset.apply_threshold or self.postprocess_var.get() == "Black and White"
        )
        binarization_method = BINARIZATION_UI_METHODS.get(
            self.binarization_method_var.get(),
            "none",
        )
        return PreprocessSettings(
            contrast=float(self.preprocess_contrast_var.get()),
            brightness=int(self.preprocess_brightness_var.get()),
            denoise=int(self.preprocess_denoise_var.get()),
            threshold=int(self.preprocess_threshold_var.get()),
            apply_threshold=apply_threshold,
            # Lighting has one canonical stage after geometry correction.
            # The legacy cleanup flag stays off to prevent double correction.
            correct_illumination=False,
            binarization_method=binarization_method,
            binarization_window=int(self.binarization_window_var.get()),
            binarization_k=(
                float(self.binarization_k_var.get())
                if binarization_method in {"sauvola", "wolf"}
                and getattr(self, "_binarization_k_custom", False)
                else None
            ),
            despeckle_strength=DESPECKLE_UI_STRENGTHS.get(
                self.despeckle_strength_var.get(),
                "none",
            ),
        )

    def _on_binarization_method_change(self, value: str) -> None:
        method = BINARIZATION_UI_METHODS.get(value, "none")
        if not self._binarization_k_custom:
            # Match the controller/CLI defaults instead of applying Sauvola's
            # 0.2 coefficient to Wolf, whose documented default is 0.5.
            if method == "wolf":
                self.binarization_k_var.set(0.5)
            elif method == "sauvola":
                self.binarization_k_var.set(0.2)
        self.update_page_preview()

    def _on_binarization_k_change(self, _value: float) -> None:
        self._binarization_k_custom = True
        self.update_page_preview()

    def _on_manual_deskew_angle_change(self, value: float) -> None:
        angle = float(value)
        self.manual_deskew_summary_var.set(f"Manual deskew: {angle:+.1f} degrees")
        if self._loading_page_recipe:
            return
        self.deskew_method_var.set("Manual angle")
        self.update_page_preview()

    @staticmethod
    def _ui_name_for_method(mapping: dict[str, str], method: str, fallback: str) -> str:
        return next((name for name, value in mapping.items() if value == method), fallback)

    def _sync_controls_from_single_committed_page(self) -> None:
        """Load one page's durable recipe so editing one stage preserves all others."""
        selected = self.page_listbox.curselection()
        if len(selected) != 1 or not 0 <= selected[0] < len(self.session.entries):
            self.stage_settings_var.set("Stage settings: document defaults / mixed selection")
            return
        entry = self.session.entries[selected[0]]
        committed = entry.committed_processing
        if committed is None:
            self.stage_settings_var.set("Stage settings: document defaults")
            return
        request = committed.recipe.to_request()
        self._loading_page_recipe = True
        try:
            self.orientation_method_var.set(
                self._ui_name_for_method(
                    ORIENTATION_UI_METHODS,
                    request.orientation_method,
                    "Off",
                )
            )
            self.deskew_method_var.set(
                self._ui_name_for_method(
                    DESKEW_UI_METHODS,
                    request.deskew_method,
                    "Off",
                )
            )
            angle = float(request.deskew_angle_degrees or 0.0)
            self.manual_deskew_angle_var.set(angle)
            self.manual_deskew_summary_var.set(f"Manual deskew: {angle:+.1f} degrees")
            self.dewarp_method_var.set(
                self._ui_name_for_method(DEWARP_UI_METHODS, request.dewarp_method, "None")
            )
            self.shadow_method_var.set(
                self._ui_name_for_method(SHADOW_UI_METHODS, request.shadow_method, "None")
            )
            self.postprocess_var.set(request.postprocess_name)
            settings = request.preprocess_settings
            if settings is not None:
                self.preprocess_preset_var.set("Custom")
                self.preprocess_contrast_var.set(float(settings.contrast))
                self.preprocess_brightness_var.set(int(settings.brightness))
                self.preprocess_denoise_var.set(int(settings.denoise))
                self.preprocess_threshold_var.set(int(settings.threshold))
                self.binarization_method_var.set(
                    self._ui_name_for_method(
                        BINARIZATION_UI_METHODS,
                        settings.binarization_method,
                        "None",
                    )
                )
                self.binarization_window_var.set(int(settings.binarization_window))
                if settings.binarization_k is not None:
                    self.binarization_k_var.set(float(settings.binarization_k))
                self._binarization_k_custom = settings.binarization_k is not None
                self.despeckle_strength_var.set(
                    self._ui_name_for_method(
                        DESPECKLE_UI_STRENGTHS,
                        settings.despeckle_strength,
                        "None",
                    )
                )
            else:
                # Do not leak cleanup controls from the previously selected
                # page into a recipe whose cleanup stage was explicitly off.
                self.preprocess_preset_var.set("Custom")
                self.preprocess_contrast_var.set(1.0)
                self.preprocess_brightness_var.set(0)
                self.preprocess_denoise_var.set(0)
                self.preprocess_threshold_var.set(170)
                self.binarization_method_var.set("None")
                self.binarization_window_var.set(31)
                self.binarization_k_var.set(0.2)
                self._binarization_k_custom = False
                self.despeckle_strength_var.set("None")
            self.page_layout_var.set(
                self._ui_name_for_method(
                    PAGE_LAYOUT_UI_METHODS, request.page_layout, "Keep source page"
                )
            )
            self.page_margin_mm_var.set(float(request.page_margin_mm))
            self.page_align_x_var.set(request.horizontal_alignment)
            self.page_align_y_var.set(request.vertical_alignment)
            if request.page_layout != "none":
                self.export_pdf_dpi_var.set(int(request.page_dpi))
            self.lens_mode_var.set(
                infer_lens_mode(self.preprocess_preset_var.get(), self.postprocess_var.get())
            )
        finally:
            self._loading_page_recipe = False
        self.stage_settings_var.set(f"Stage settings: loaded from {entry.name}")

    def _apply_postprocess(self, image: np.ndarray) -> np.ndarray:
        """Compatibility helper routed through the canonical controller."""
        return process_document_page(
            image,
            PageProcessingRequest(
                postprocess_name=self.postprocess_var.get(),
                preprocess_settings=self._current_preprocess_settings(),
            ),
        ).image

    def _apply_page_layout(self, image: np.ndarray, *, preview: bool):
        """Compatibility helper routed through the canonical controller."""
        dpi = 100 if preview else max(72, int(self.export_pdf_dpi_var.get()))
        result = process_document_page(
            image,
            PageProcessingRequest(
                page_layout=PAGE_LAYOUT_UI_METHODS.get(self.page_layout_var.get(), "none"),
                page_dpi=dpi,
                page_margin_mm=float(self.page_margin_mm_var.get()),
                horizontal_alignment=self.page_align_x_var.get(),
                vertical_alignment=self.page_align_y_var.get(),
            ),
        )
        return result.image, result.diagnostics.layout

    def _entry_dewarp_model(self, entry) -> DewarpModel | None:
        if entry is None or entry.dewarp_control_points is None:
            return None
        return DewarpModel(
            method=DEWARP_METHOD_TEXTLINE,
            control_points=entry.dewarp_control_points,
            source="user",
            control_curves=entry.dewarp_control_curves,
        )

    def _apply_dewarp(self, image: np.ndarray, *, entry=None):
        """Compatibility helper routed through the canonical controller."""
        dewarp_method = DEWARP_UI_METHODS.get(
            self.dewarp_method_var.get(),
            DEWARP_METHOD_NONE,
        )
        request = PageProcessingRequest(
            dewarp_method=dewarp_method,
            dewarp_model=self._entry_dewarp_model(entry),
            dewarp_already_applied=self._entry_was_dewarped(entry, dewarp_method),
        )
        result = process_document_page(image, request)
        return result.image, result.diagnostics.dewarp

    def _processing_request(
        self,
        *,
        entry=None,
        preview: bool = False,
        lighting_diagnostics: bool = False,
    ) -> PageProcessingRequest:
        self._last_processing_cache_hits = ()
        dewarp_method = DEWARP_UI_METHODS.get(
            self.dewarp_method_var.get(),
            DEWARP_METHOD_NONE,
        )
        perspective_points = None
        if entry is not None and entry.committed_processing is not None:
            perspective_points = entry.committed_processing.recipe.perspective_points
        return PageProcessingRequest(
            orientation_method=ORIENTATION_UI_METHODS.get(
                self.orientation_method_var.get(), ORIENTATION_METHOD_NONE
            ),
            perspective_points=perspective_points,
            deskew_method=DESKEW_UI_METHODS.get(self.deskew_method_var.get(), DESKEW_METHOD_NONE),
            deskew_angle_degrees=(
                float(self.manual_deskew_angle_var.get())
                if DESKEW_UI_METHODS.get(self.deskew_method_var.get()) == DESKEW_METHOD_MANUAL
                else None
            ),
            dewarp_method=dewarp_method,
            dewarp_model=self._entry_dewarp_model(entry),
            dewarp_already_applied=self._entry_was_dewarped(entry, dewarp_method),
            shadow_method=SHADOW_UI_METHODS.get(self.shadow_method_var.get(), SHADOW_METHOD_NONE),
            postprocess_name=self.postprocess_var.get(),
            preprocess_settings=self._current_preprocess_settings(),
            page_layout=PAGE_LAYOUT_UI_METHODS.get(self.page_layout_var.get(), "none"),
            page_dpi=(100 if preview else max(72, int(self.export_pdf_dpi_var.get()))),
            page_margin_mm=float(self.page_margin_mm_var.get()),
            horizontal_alignment=self.page_align_x_var.get(),
            vertical_alignment=self.page_align_y_var.get(),
            lighting_diagnostics=lighting_diagnostics,
            stage_cache=self.processing_cache,
        )

    @staticmethod
    def _entry_was_dewarped(entry, requested_method: str) -> bool:
        # `none` means identity over the already-rectified original.  Marking it
        # as "already applied" is invalid in the controller and unnecessary.
        return bool(
            requested_method != DEWARP_METHOD_NONE
            and entry is not None
            and entry.detected_backend in {DETECTOR_BACKEND_UVDOC, DETECTOR_BACKEND_PADDLEOCR_UVDOC}
        )

    def _process_review_page(self, image: np.ndarray, *, entry=None, preview: bool):
        result = process_document_page(
            image,
            self._processing_request(entry=entry, preview=preview),
        )
        self._last_processing_cache_hits = result.diagnostics.cache_hits
        return result

    def clear_processing_cache(self) -> None:
        self.processing_cache.clear()
        self._last_processing_cache_hits = ()
        self._set_status("Processing stage cache cleared.")

    def _review_before_image(self, entry) -> np.ndarray:
        """Raw source with the detected contour drawn over it."""
        proposal = _entry_has_crop_proposal(entry)
        if self.lightweight_preview_var.get():
            raw_image = entry.preview_raw_image
        else:
            raw_image = entry.raw_image

        contour = entry.detected_contour
        if not proposal or contour is None:
            return raw_image

        # Contour is stored in the original raw coordinate space.
        # The preview may have been resized — scale the contour accordingly.
        full_raw = entry.raw_image if self.lightweight_preview_var.get() else raw_image
        scaled = scale_contour(
            contour,
            src_shape=full_raw.shape[:2],
            dst_shape=raw_image.shape[:2],
        )
        return draw_quad_overlay(raw_image, scaled)

    def _review_source_image(self, entry, *, fast_preview: bool) -> np.ndarray:
        """Return committed source pixels or an explicitly labelled crop proposal."""
        committed = entry.preview_original_image if fast_preview else entry.original_image
        if not _entry_has_crop_proposal(entry):
            recipe = (
                entry.committed_processing.recipe
                if entry.committed_processing is not None
                else None
            )
            if recipe is not None and recipe.perspective_points is not None:
                points = np.asarray(recipe.perspective_points, dtype=np.float32)
                if fast_preview:
                    points = scale_contour(
                        points,
                        src_shape=entry.original_image.shape[:2],
                        dst_shape=committed.shape[:2],
                    )
                try:
                    return warp_perspective_from_points(committed, points)
                except ValueError:
                    pass
            return committed

        raw = entry.preview_raw_image if fast_preview else entry.raw_image
        contour = np.asarray(entry.detected_contour, dtype=np.float32)
        if fast_preview:
            contour = scale_contour(
                contour,
                src_shape=entry.raw_image.shape[:2],
                dst_shape=raw.shape[:2],
            )
        try:
            proposed = warp_perspective_from_points(raw, contour)
        except ValueError:
            return committed
        return proposed if proposed is not None and proposed.size else committed

    def _review_after_image(self, entry, before_image: np.ndarray) -> np.ndarray:
        # A detected contour is an unapplied proposal.  Show its full-resolution
        # perspective result, but keep export/current pixels on the durable
        # original until the user presses Apply in the corner editor.
        del before_image
        source = (
            self._review_source_image(entry, fast_preview=False)
            if _entry_has_crop_proposal(entry)
            else entry.original_image
        )
        return self._process_review_page(
            source,
            entry=entry,
            preview=False,
        ).image

    def _show_in_preview(self, image: np.ndarray) -> None:
        photo = self._to_ctk_photo_for_label(image, self.preview_label)
        self.preview_label.configure(image=photo, text="")
        self.preview_photo = photo

    def _to_ctk_photo_for_label(self, image: np.ndarray, label: ctk.CTkLabel) -> ctk.CTkImage:
        max_width = max(200, label.winfo_width())
        max_height = max(120, label.winfo_height())
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Colour conversion happens after the resize: at display size it is far
        # cheaper than at full capture resolution.
        if resized.ndim == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))

    def _process_capture_frame(
        self,
        frame: np.ndarray,
        base_name: str,
    ) -> list[PageResult]:
        options = PipelineOptions(
            detect_document=True,
            detect_proposal_only=True,
            two_page_mode=False,
            postprocess_name="None",
        )
        return process_loaded_items([(base_name, frame)], options=options)

    def _ingest_page_results(self, results: list[PageResult]) -> None:
        for result in results:
            self.session.add_image_with_contour(
                name=result.name,
                raw_image=result.raw,
                warped_image=result.raw,
                contour=result.contour,
                backend=(
                    result.backend if result.detected and result.contour is not None else None
                ),
                crop_state=CROP_STATE_PROPOSED if result.contour is not None else CROP_STATE_NONE,
                needs_review=result.needs_review,
                review_reasons=result.review_reasons,
            )

    def _ingest_staged_import_pages(self, pages: list[_StagedImportPage]) -> None:
        """Publish a fully staged import to the session or roll it back logically."""
        added_entry_ids: list[str] = []
        try:
            for page in pages:
                raw = imread_unicode(page.raw_path)
                if raw is None:
                    raise RuntimeError(f"Cannot read staged imported page: {page.name}")
                entry = self.session.add_image_with_contour(
                    name=page.name,
                    raw_image=raw,
                    warped_image=raw,
                    contour=page.contour,
                    backend=page.backend if page.contour is not None else None,
                    crop_state=CROP_STATE_PROPOSED if page.contour is not None else CROP_STATE_NONE,
                    needs_review=page.needs_review,
                    review_reasons=page.review_reasons,
                )
                added_entry_ids.append(entry.entry_id)
        except Exception:
            for entry_id in reversed(added_entry_ids):
                self.session.remove_entry(entry_id)
            raise

    def _detect_single_page(self, frame: np.ndarray, *, name: str) -> PageResult:
        """Run detection on a single frame (no spread split) and return one PageResult."""
        options = PipelineOptions(
            detect_document=True,
            detect_proposal_only=True,
            two_page_mode=False,
            postprocess_name="None",
        )
        results = process_loaded_items([(name, frame)], options=options)
        if not results:
            raise RuntimeError("Document detection returned no pages.")
        return results[0]

    def _grab_still_frame(self, camera: CameraService) -> np.ndarray | None:
        """The shot: the frame already on screen when the button was pressed.

        Preview and capture share one resolution, so the newest streamed frame
        is the still. No device reconfiguration, no waiting - what the user saw
        is what gets captured.
        """
        latest_frame = getattr(camera, "latest_frame", None)
        if callable(latest_frame):
            frame = latest_frame()
            if frame is not None:
                return frame
        return camera.read_frame()

    def _raise_if_camera_opening(self) -> None:
        if self.__dict__.get("_camera_opening", False):
            raise RuntimeError("Camera is still opening; try again in a moment.")

    def _capture_workspace_still(self) -> np.ndarray:
        """Still for workspace actions such as page retake.

        Takes the live frame when the Camera tab keeps a stream running;
        otherwise opens the device once for a single shot.
        """
        self._raise_if_camera_opening()
        if self._burst_is_active():
            raise RuntimeError("Camera is busy with burst capture.")
        camera = self.camera
        if camera is not None:
            frame = self._grab_still_frame(camera)
        else:
            temp = CameraService(
                index=int(self.camera_index_var.get()),
                resolution=self._max_camera_resolution(),
            )
            try:
                frame = temp.capture_still(timeout_sec=FRESH_FRAME_TIMEOUT_SEC)
            finally:
                temp.release()
        if frame is None:
            raise RuntimeError("Could not capture an image from the camera.")
        return frame

    def capture_one(self) -> None:
        """Capture a single page; runs on the shared background capture path."""
        self._start_capture_job(shots=1, delay_sec=0.0)

    def capture_burst(self) -> None:
        try:
            shots = int(self.camera_shots_var.get())
            delay_sec = float(self.camera_delay_var.get())
            if not 1 <= shots <= CameraService.MAX_BURST_SHOTS:
                raise ValueError(
                    f"Burst shots must be between 1 and {CameraService.MAX_BURST_SHOTS}."
                )
            if delay_sec < 0:
                raise ValueError("Delay must be >= 0 seconds.")
        except Exception as exc:
            messagebox.showerror("Burst Error", str(exc))
            self._set_status("Burst capture failed")
            return
        self._start_capture_job(shots=shots, delay_sec=delay_sec)

    def _start_capture_job(self, *, shots: int, delay_sec: float) -> None:
        """Capture ``shots`` full-resolution pages in a background job.

        The open preview stream is reused when possible (the live view keeps
        running); detection and staging always happen off the UI thread.
        """
        single = shots == 1
        job_name = "Capture" if single else "Capture Burst"
        stage_name = "Capture" if single else "Burst capture"
        burst_reserved = False
        staging_dir = None
        try:
            if self.__dict__.get("_camera_opening", False):
                # No modal here: rapid shutter presses should not spawn popups.
                self._set_status("Camera is reconfiguring; try again in a moment.")
                return
            if self.__dict__.get("job_thread") is not None:
                self._set_status("Busy: the previous capture is still processing.")
                return
            index = int(self.camera_index_var.get())
            resolution = self._max_camera_resolution()
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")

            shared_camera = self._shared_burst_camera(index, resolution)
            # Grab the shutter frame here, on the UI thread, before any
            # background work starts: the stored page is then precisely the
            # frame that was on screen when the button was pressed, not one
            # taken a job-startup later.
            shutter_frame = None
            if shared_camera is not None:
                shutter_frame = self._grab_still_frame(shared_camera)

            self._begin_burst_capture()
            burst_reserved = True
            if shared_camera is None:
                # Legacy path: no live stream to reuse, so a dedicated device
                # handle is opened inside the worker.
                self.stop_preview()
                self._release_camera_handle()
            self._update_camera_health()
            staging_dir = tempfile.TemporaryDirectory(prefix="uniscan_gui_burst_")

            def worker(emit, is_cancelled):
                camera = shared_camera
                owns_camera = camera is None
                if owns_camera:
                    emit(
                        stage=stage_name, current=f"Opening {self._device_label(index)}", progress=0
                    )
                    camera = CameraService(index=index, resolution=resolution)
                frame_paths: list[tuple[int, Path]] = []
                try:
                    if owns_camera:
                        # Publish and open under one app-level critical section: a
                        # simultaneous window close can then always find and release
                        # the only burst handle after open() returns.
                        with self._camera_guard():
                            if is_cancelled():
                                raise RuntimeError("Cancelled by user.")
                            self.burst_camera = camera
                            camera.open()
                    # Preview and capture share one resolution, so the live
                    # stream supplies the shots directly: the first is the
                    # frame that was on screen at the shutter press.
                    frames = camera.iter_burst(
                        shots=shots,
                        delay_sec=delay_sec,
                        cancel_cb=is_cancelled,
                        first_frame=shutter_frame,
                        on_progress=lambda i, total: emit(
                            stage=stage_name,
                            current=f"Shot {i}/{total}",
                            progress=int((i / total) * 45),
                        ),
                    )
                    for frame_index, frame in frames:
                        # Report what the device actually delivered, not what
                        # was requested: drivers often grant a smaller size.
                        self._effective_capture_resolution = (frame.shape[1], frame.shape[0])
                        frame_path = Path(staging_dir.name) / f"{frame_index:06d}-raw.png"
                        if not imwrite_unicode(frame_path, frame):
                            raise RuntimeError(
                                f"Cannot stage captured frame {frame_index}/{shots}."
                            )
                        if is_cancelled():
                            raise RuntimeError("Cancelled by user.")
                        frame_paths.append((frame_index, frame_path))
                finally:
                    if owns_camera:
                        camera.release()
                        with self._camera_guard():
                            if self.burst_camera is camera:
                                self.burst_camera = None

                staged_pages: list[_StagedImportPage] = []
                fallback_pages = 0
                total_frames = len(frame_paths)
                for idx, frame_path in frame_paths:
                    if is_cancelled():
                        raise RuntimeError("Cancelled by user.")
                    frame = imread_unicode(frame_path)
                    if frame is None:
                        raise RuntimeError(f"Cannot read staged captured frame {idx}.")
                    current_results = self._process_capture_frame(
                        frame, base_name=f"{timestamp}_{idx:03d}"
                    )
                    for result in current_results:
                        staged_pages.append(
                            _StagedImportPage(
                                name=result.name,
                                raw_path=frame_path,
                                contour=result.contour,
                                backend=result.backend if result.detected else None,
                                fallback_reason=result.fallback_reason,
                                needs_review=result.needs_review,
                                review_reasons=result.review_reasons,
                            )
                        )
                        fallback_pages += result.fallback_reason is not None
                    emit(
                        stage="Processing pages",
                        current=f"Frame {idx}/{total_frames}",
                        progress=45 + int((idx / total_frames) * 55),
                    )
                return staged_pages, fallback_pages

            def finish_burst() -> None:
                nonlocal burst_reserved
                self._end_burst_capture()
                burst_reserved = False
                staging_dir.cleanup()

            def on_done(payload):
                try:
                    staged_pages, fallback_pages = payload
                    self._ingest_staged_import_pages(staged_pages)
                    self.refresh_page_list(keep_index=len(self.session) - 1)
                    # Stay on the Camera tab so the next shot starts
                    # immediately; Workspace is one click away.
                    summary = _detection_summary_counts(len(staged_pages), fallback_pages)
                    prefix = "Captured" if single else "Burst captured"
                    self._set_status(
                        f"{prefix} {len(staged_pages)} page(s): {summary}. "
                        f"Session pages: {len(self.session)}"
                    )
                finally:
                    finish_burst()

            if not self._start_background_job(job_name, worker, on_done, on_error=finish_burst):
                finish_burst()
        except Exception as exc:
            if burst_reserved:
                self._end_burst_capture()
            if staging_dir is not None:
                staging_dir.cleanup()
            messagebox.showerror(f"{job_name} Error", str(exc))
            self._set_status(f"{job_name} failed")

    def _shared_burst_camera(self, index: int, resolution: tuple[int, int]) -> CameraService | None:
        """Reuse the open streaming camera for capture so shots come straight
        off the live stream; None selects the legacy open-a-device path."""
        camera = self.camera
        if camera is None:
            return None
        if getattr(camera, "index", None) != index:
            return None
        current = getattr(camera, "resolution", None)
        if current is None or tuple(current) != tuple(resolution):
            return None
        is_streaming = getattr(camera, "is_streaming", None)
        if not callable(is_streaming) or not is_streaming():
            return None
        return camera

    def _set_camera_resolution(self, resolution: tuple[int, int]) -> None:
        """Apply a capture resolution and commit the preference only on success.

        Preview and capture share this resolution, so it is both the picture
        quality and the preview frame rate the camera has to sustain.
        """
        if self._burst_is_active():
            raise RuntimeError("Camera is busy with burst capture.")
        previous_resolution = self.camera_resolution
        if self.camera is None:
            candidate = CameraService(
                index=int(self.camera_index_var.get()),
                resolution=resolution,
            )
            try:
                candidate.open()
            except Exception:
                candidate.release()
                raise
            self._effective_capture_resolution = getattr(candidate, "effective_resolution", None)
            self.camera = candidate
        else:
            camera = self.camera
            try:
                camera.set_resolution(resolution)
                self._effective_capture_resolution = getattr(camera, "effective_resolution", None)
            except Exception:
                try:
                    camera.set_resolution(previous_resolution)
                except Exception:
                    camera.release()
                    self.camera = None
                raise
        self.camera_resolution = resolution

    def _refresh_camera_device_names(self) -> None:
        """Read the system's video capture device names (best effort)."""
        try:
            self.camera_device_names = list_camera_device_names()
        except Exception:
            self.camera_device_names = []

    def _device_label(self, index: int) -> str:
        """Menu text for a device: its system name, or the bare index."""
        names = self.__dict__.get("camera_device_names") or []
        if 0 <= index < len(names) and names[index]:
            return names[index]
        return f"Camera {index}"

    def _device_menu_values(self) -> list[str]:
        indices = self.__dict__.get("camera_device_indices")
        if not indices:
            names = self.__dict__.get("camera_device_names") or []
            # Before probing, offer the named devices the system reports; a
            # machine that reports none still gets a usable index list.
            indices = list(range(len(names))) if names else list(range(10))
        labels: list[str] = []
        for index in indices:
            label = self._device_label(index)
            # Menu entries must stay unique even when two devices share a name.
            if label in labels:
                label = f"{label} ({index})"
            labels.append(label)
        return labels

    def _device_menu_selection(self) -> str:
        current = int(self.camera_index_var.get())
        values = self._device_menu_values()
        indices = self.__dict__.get("camera_device_indices") or list(range(len(values)))
        for index, label in zip(indices, values):
            if index == current:
                return label
        return self._device_label(current)

    def _index_for_device_label(self, label: str) -> int | None:
        values = self._device_menu_values()
        indices = self.__dict__.get("camera_device_indices") or list(range(len(values)))
        for index, value in zip(indices, values):
            if value == label:
                return index
        return None

    def _refresh_device_menu(self) -> None:
        # __dict__.get, not getattr: tkinter's Misc.__getattr__ recurses on
        # instances built without a Tk window.
        menu = self.__dict__.get("camera_index_menu")
        if menu is None:
            return
        try:
            if menu.winfo_exists():
                menu.configure(values=self._device_menu_values())
                menu.set(self._device_menu_selection())
        except tk.TclError:
            pass

    def _on_camera_device_selected(self, label: str) -> None:
        """Device menu shows system names; resolve the label to its index."""
        index = self._index_for_device_label(label)
        if index is None:
            return
        self._on_camera_index_selected(str(index))

    def _on_camera_index_selected(self, index_str: str) -> None:
        """Inline device selector on the Camera tab."""
        try:
            index = int(index_str)
        except (TypeError, ValueError):
            return
        self.camera_index_var.set(index)
        # Modes belong to the device: drop them so the new one is measured or
        # its own cache is loaded on the next open.
        self.camera_modes = []
        self._camera_modes_probed_index = None
        self._camera_resolution_chosen = False
        self._refresh_resolution_menu()
        if self.camera is not None or self.__dict__.get("_camera_opening", False):
            # Re-open at the new index off the UI thread; the running preview
            # picks up the new stream automatically.
            self._open_camera_async()
        self._set_status(f"Camera: {self._device_label(index)}")

    def _identify_cameras_async(self) -> None:
        """Probe device indices off the UI thread and fill the device menu."""
        button = self.__dict__.get("camera_identify_button")
        if button is not None:
            try:
                button.configure(state=tk.DISABLED, text="Finding...")
            except tk.TclError:
                button = None
        found: list[list[int]] = []
        self._refresh_camera_device_names()
        # The system already lists its capture devices, and OpenCV indexes the
        # same enumeration: probe exactly that many instead of sweeping ten
        # indices and waiting on eight failures.
        max_indices = len(self.camera_device_names) or 10

        def probe() -> None:
            try:
                found.append(CameraService.get_available_device_indices(max_indices=max_indices))
            except Exception:
                found.append([])

        probe_thread = threading.Thread(target=probe, name="CameraProbe", daemon=True)
        probe_thread.start()

        def poll() -> None:
            if probe_thread.is_alive():
                self.after(50, poll)
                return
            if button is not None:
                try:
                    if button.winfo_exists():
                        button.configure(state=tk.NORMAL, text="Find cameras")
                except tk.TclError:
                    pass
            indices = found[0] if found else []
            self.camera_device_indices = indices
            if indices and int(self.camera_index_var.get()) not in indices:
                self._on_camera_index_selected(str(indices[0]))
            self._refresh_device_menu()
            if indices:
                names = ", ".join(self._device_label(index) for index in indices)
                self._set_status(f"Found cameras: {names}")
            else:
                self._set_status("No cameras found.")

        self.after(50, poll)

    def _apply_resolution_string(self, res_string: str) -> None:
        """Inline capture-resolution control on the Camera tab (async apply).

        Accepts both a plain ``<width>x<height>`` and a measured-mode label
        such as ``1920x1080 - 30 fps``.
        """
        match = re.match(r"^(\d+)x(\d+)", res_string.strip())
        if match is None:
            messagebox.showerror(
                "Resolution Error", "Resolution must be on the form <width>x<height>."
            )
            return
        resolution = (int(match.group(1)), int(match.group(2)))
        self._camera_resolution_chosen = True
        if self.__dict__.get("_camera_opening", False):
            self._set_status("Camera is still opening; try again in a moment.")
            return
        if self._burst_is_active():
            self._set_status("Camera is busy capturing; try again after it finishes.")
            return
        self._set_status(f"Applying capture resolution {resolution[0]}x{resolution[1]}...")
        errors: list[Exception] = []

        def apply() -> None:
            try:
                self._set_camera_resolution(resolution)
            except Exception as exc:
                errors.append(exc)

        apply_thread = threading.Thread(target=apply, name="CameraResolution", daemon=True)
        apply_thread.start()

        def poll() -> None:
            if apply_thread.is_alive():
                self.after(50, poll)
                return
            if errors:
                self._set_status(f"Capture resolution failed: {errors[0]}")
                messagebox.showerror("Resolution Error", str(errors[0]))
            else:
                granted = self.__dict__.get("_effective_capture_resolution")
                if granted and tuple(granted) != resolution:
                    self._set_status(
                        f"Capture resolution set to {resolution[0]}x{resolution[1]} "
                        f"(device grants {granted[0]}x{granted[1]})"
                    )
                else:
                    self._set_status(f"Capture resolution set to {resolution[0]}x{resolution[1]}")
            self._refresh_resolution_menu()
            self._update_camera_health()

        self.after(50, poll)

    @staticmethod
    def _expand_import_sources(sources: Iterable[Path]) -> list[Path]:
        supported: list[Path] = []
        seen: set[str] = set()
        for source in sources:
            source = Path(source)
            candidates = list_supported_in_folder(source) if source.is_dir() else [source]
            for candidate in candidates:
                if not candidate.is_file() or candidate.suffix.lower() not in (IMG_EXTS | PDF_EXTS):
                    continue
                key = str(candidate.resolve())
                if key not in seen:
                    seen.add(key)
                    supported.append(candidate)
        return supported

    def quick_export_pdf(self) -> None:
        if not self.session.entries:
            self._set_status("No pages available for export.")
            return
        self.export_pdf_path_var.set("")
        self.export_to_pdf()

    def open_export_dialog(self) -> None:
        if not self.session.entries:
            self._set_status("No pages available for export.")
            return
        if self.export_dialog_window is not None:
            try:
                if self.export_dialog_window.winfo_exists():
                    self.export_dialog_window.lift()
                    return
            except tk.TclError:
                pass

        window = ctk.CTkToplevel(self)
        self.export_dialog_window = window
        window.title("Export options")
        window.geometry("480x350")
        window.resizable(False, False)
        window.transient(self)

        mode_var = self.export_dialog_mode_var
        scope_var = self.export_dialog_scope_var
        dpi_var = self.export_dialog_dpi_var
        image_format_var = self.export_dialog_format_var
        mode_var.set("PDF")
        scope_var.set(self.export_scope_var.get())
        dpi_var.set(self.export_pdf_dpi_var.get())
        image_format_var.set(self.export_format_var.get())

        def close_dialog() -> None:
            self.export_dialog_window = None
            window.destroy()

        def custom_export() -> None:
            self.export_scope_var.set(scope_var.get())
            if mode_var.get() == "PDF":
                self.export_pdf_dpi_var.set(dpi_var.get())
                self.export_pdf_path_var.set("")
                close_dialog()
                self.export_to_pdf()
            else:
                self.export_format_var.set(image_format_var.get())
                self.export_dir_var.set("")
                close_dialog()
                self.export_to_files()

        ctk.CTkLabel(
            window,
            text="Export options",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(fill=ctk.X, padx=16, pady=(16, 8))
        mode_control = ctk.CTkSegmentedButton(window, values=["PDF", "Images"], variable=mode_var)
        mode_control.pack(fill=ctk.X, padx=16, pady=(0, 10))

        scope_row = ctk.CTkFrame(window, fg_color="transparent")
        scope_row.pack(fill=ctk.X, padx=16, pady=(0, 8))
        ctk.CTkLabel(scope_row, text="Pages").pack(side=ctk.LEFT)
        ctk.CTkOptionMenu(
            scope_row,
            values=["All pages", "Selected pages"],
            variable=scope_var,
            width=180,
        ).pack(side=ctk.RIGHT)

        option_host = ctk.CTkFrame(window, fg_color="transparent")
        option_host.pack(fill=ctk.X, padx=16, pady=(0, 10))
        pdf_options = ctk.CTkFrame(option_host, fg_color="transparent")
        ctk.CTkLabel(pdf_options, text="PDF DPI").pack(side=ctk.LEFT)
        ctk.CTkEntry(pdf_options, textvariable=dpi_var, width=100).pack(side=ctk.RIGHT)
        image_options = ctk.CTkFrame(option_host, fg_color="transparent")
        ctk.CTkLabel(image_options, text="Image format").pack(side=ctk.LEFT)
        ctk.CTkOptionMenu(
            image_options,
            values=["png", "jpg", "webp", "tif"],
            variable=image_format_var,
            width=140,
        ).pack(side=ctk.RIGHT)

        def update_mode(_value: str | None = None) -> None:
            pdf_options.pack_forget()
            image_options.pack_forget()
            options = pdf_options if mode_var.get() == "PDF" else image_options
            options.pack(fill=ctk.X)

        mode_control.configure(command=update_mode)
        update_mode()

        footer = ctk.CTkFrame(window, fg_color="transparent")
        footer.pack(side=ctk.BOTTOM, fill=ctk.X, padx=16, pady=16)
        ctk.CTkButton(
            footer,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=close_dialog,
        ).pack(side=ctk.RIGHT)
        self.export_custom_button = ctk.CTkButton(
            footer,
            text="Export with settings",
            command=custom_export,
        )
        self.export_custom_button.pack(side=ctk.RIGHT, padx=6)

        window.protocol("WM_DELETE_WINDOW", close_dialog)
        window.grab_set()

    def _normalize_selected_files(self, files: Iterable[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in files:
            key = str(Path(item))
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def _on_drop_files(self, event) -> str:
        paths = paths_from_tk_drop(str(event.data), self.tk.splitlist)
        supported = self._expand_import_sources(paths)
        if not supported:
            self._set_status("Drop contained no supported images or PDFs.")
            return "break"
        self.import_selected_files = self._normalize_selected_files(map(str, supported))
        self._import_paths(paths=[Path(path) for path in self.import_selected_files])
        return "break"

    def import_from_clipboard(self) -> None:
        try:
            payload = ImageGrab.grabclipboard()
            image = clipboard_image_to_bgr(payload)
            if image is not None:
                results = self._process_capture_frame(image, base_name="clipboard")
                self._ingest_page_results(results)
                self.refresh_page_list(keep_index=len(self.session) - 1)
                self.go_to_review_tab()
                self._set_status(f"Imported {len(results)} page(s) from clipboard image.")
                return

            paths = clipboard_file_paths(payload)
            supported = self._expand_import_sources(paths)
            if not supported:
                raise RuntimeError("Clipboard does not contain an image or supported files.")
            self._import_paths(paths=supported)
        except Exception as exc:
            messagebox.showerror("Clipboard Import Error", str(exc))
            self._set_status("Clipboard import failed")

    def _import_paths(self, *, paths: list[Path]) -> None:
        pdf_dpi = int(self.import_pdf_dpi_var.get())
        if pdf_dpi < 72:
            raise RuntimeError("PDF DPI must be >= 72.")
        two_page_mode = bool(self.import_two_page_mode_var.get())
        split_label = "enabled" if two_page_mode else "disabled"
        self._set_status(
            f"Starting import for {len(paths)} file(s): PDF {pdf_dpi} DPI, "
            f"spread split {split_label}."
        )
        staging_dir = tempfile.TemporaryDirectory(prefix="uniscan_gui_import_")

        def worker(emit, is_cancelled):
            emit(stage="Import", current=f"{len(paths)} input file(s)", progress=0)
            total_paths = len(paths)
            added_pages = 0
            fallback_pages = 0
            staged_pages: list[_StagedImportPage] = []
            options = PipelineOptions(
                detect_document=True,
                detect_proposal_only=not two_page_mode,
                two_page_mode=two_page_mode,
                postprocess_name="None",
            )

            for file_index, path in enumerate(paths, start=1):
                if is_cancelled():
                    raise RuntimeError("Cancelled by user.")

                emit(
                    stage="Import loading",
                    current=f"{file_index}/{total_paths}: {path.name}",
                    progress=int(((file_index - 1) / total_paths) * 30),
                )
                emit(
                    stage="Import detect",
                    current=f"{file_index}/{total_paths}: {path.name}",
                    progress=int(((file_index - 1) / total_paths) * 30) + 5,
                )
                loaded_items = iter_input_items(
                    [path],
                    pdf_dpi=pdf_dpi,
                    cancel_cb=is_cancelled,
                )
                for loaded_item in loaded_items:
                    if is_cancelled():
                        raise RuntimeError("Cancelled by user.")
                    results = process_loaded_items(
                        [loaded_item],
                        options=options,
                        cancel_cb=is_cancelled,
                    )
                    for result in results:
                        page_index = len(staged_pages) + 1
                        raw_path = Path(staging_dir.name) / f"{page_index:06d}-raw.png"
                        if not imwrite_unicode(raw_path, result.raw):
                            raise RuntimeError(f"Cannot stage imported page: {result.name}")
                        if is_cancelled():
                            raise RuntimeError("Cancelled by user.")
                        staged_pages.append(
                            _StagedImportPage(
                                name=result.name,
                                raw_path=raw_path,
                                contour=result.contour,
                                backend=result.backend if result.detected else None,
                                fallback_reason=result.fallback_reason,
                                needs_review=result.needs_review,
                                review_reasons=result.review_reasons,
                            )
                        )
                        added_pages += 1
                        fallback_pages += result.fallback_reason is not None

                emit(
                    stage="Import ingest",
                    current=f"{file_index}/{total_paths}: {path.name}",
                    progress=45 + int((file_index / total_paths) * 55),
                )

            return {
                "files": total_paths,
                "pages": added_pages,
                "fallbackPages": fallback_pages,
                "stagedPages": staged_pages,
            }

        def on_done(stats):
            try:
                files_count = int(stats["files"])
                pages_count = int(stats["pages"])
                fallback_pages = int(stats["fallbackPages"])
                self._ingest_staged_import_pages(stats["stagedPages"])
                self.refresh_page_list(keep_index=len(self.session) - 1)
                self.go_to_review_tab()
                summary = _detection_summary_counts(pages_count, fallback_pages)
                self._set_status(
                    f"Imported {files_count} file(s), added {pages_count} page(s): {summary}. "
                    f"PDF {pdf_dpi} DPI, spread split {split_label}. "
                    f"Session pages: {len(self.session)}"
                )
            finally:
                staging_dir.cleanup()

        if not self._start_background_job(
            "Import",
            worker,
            on_done,
            on_error=staging_dir.cleanup,
        ):
            staging_dir.cleanup()

    def refresh_page_list(
        self,
        keep_index: int | None = None,
        *,
        keep_entry_ids: Iterable[str] | None = None,
    ) -> None:
        selected_entry_ids = set(keep_entry_ids or ())
        self.page_listbox.delete(0, tk.END)
        for idx, entry in enumerate(self.session.entries, start=1):
            if _entry_has_crop_proposal(entry):
                tag = "  [crop proposal]"
            elif bool(getattr(entry, "needs_review", False)):
                tag = "  [Needs review]"
            elif _entry_needs_crop_review(entry):
                tag = "  ⚠"
            else:
                tag = ""
            self.page_listbox.insert(tk.END, f"{idx:03d}  {entry.name}{tag}")

        page_count = len(self.session.entries)
        self.page_count_var.set(f"{page_count} page" if page_count == 1 else f"{page_count} pages")
        export_state = tk.NORMAL if page_count else tk.DISABLED
        for button_name in (
            "toolbar_export_pdf_button",
            "toolbar_export_options_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.configure(state=export_state)

        if selected_entry_ids:
            for index, entry in enumerate(self.session.entries):
                if entry.entry_id in selected_entry_ids:
                    self.page_listbox.selection_set(index)
        elif keep_index is not None and len(self.session.entries) > 0:
            keep_index = max(0, min(keep_index, len(self.session.entries) - 1))
            self.page_listbox.selection_set(keep_index)
        self._sync_page_selection_to_session()
        self._update_page_action_states()
        self._sync_controls_from_single_committed_page()
        self.update_page_preview()

    def _sync_page_selection_to_session(self) -> None:
        selected = set(self.page_listbox.curselection())
        for idx, entry in enumerate(self.session.entries):
            entry.selected = idx in selected

    def on_page_select(self, _event=None) -> None:
        self._sync_page_selection_to_session()
        self._update_page_action_states()
        self._sync_controls_from_single_committed_page()
        self.update_page_preview()

    def _update_page_action_states(self) -> None:
        selected = set(self._selected_entry_indices())
        can_move_up = any(index > 0 and index - 1 not in selected for index in selected)
        can_move_down = any(
            index + 1 < len(self.session.entries) and index + 1 not in selected
            for index in selected
        )
        states = (
            ("move_pages_up_button", tk.NORMAL if can_move_up else tk.DISABLED),
            ("move_pages_down_button", tk.NORMAL if can_move_down else tk.DISABLED),
            ("delete_pages_button", tk.NORMAL if selected else tk.DISABLED),
        )
        for name, state in states:
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=state)
        split_button = getattr(self, "apply_split_button", None)
        if split_button is not None:
            split_ready = False
            if len(selected) == 1:
                selected_entry = self.session.entries[next(iter(selected))]
                split_ready = bool(
                    self.pending_split_entry_id == selected_entry.entry_id
                    and self.pending_split_ratio is not None
                    and self.pending_split_revision == selected_entry.revision
                )
            split_button.configure(state=tk.NORMAL if split_ready else tk.DISABLED)
        menu = getattr(self, "page_context_menu", None)
        if menu is not None:
            menu.entryconfigure(0, state=tk.NORMAL if can_move_up else tk.DISABLED)
            menu.entryconfigure(1, state=tk.NORMAL if can_move_down else tk.DISABLED)
            menu.entryconfigure(3, state=tk.NORMAL if selected else tk.DISABLED)

    def _page_index_at_y(self, y: int, *, clamp: bool = False) -> int | None:
        if self.page_listbox.size() == 0:
            return None
        if clamp:
            y = max(0, min(y, max(0, self.page_listbox.winfo_height() - 1)))
        index = int(self.page_listbox.nearest(y))
        bounds = self.page_listbox.bbox(index)
        if bounds is None:
            return None
        if not clamp and not bounds[1] <= y <= bounds[1] + bounds[3]:
            return None
        return index

    def _page_drop_position(self, y: int) -> tuple[int, bool] | None:
        target_index = self._page_index_at_y(y, clamp=True)
        if target_index is None:
            return None
        bounds = self.page_listbox.bbox(target_index)
        if bounds is None:
            return None
        if y <= bounds[1]:
            return target_index, False
        if y >= bounds[1] + bounds[3]:
            return target_index, True
        return target_index, y >= bounds[1] + bounds[3] / 2

    def _on_page_drag_start(self, event) -> None:
        index = self._page_index_at_y(event.y)
        selected_indexes = self._selected_entry_indices()
        if index is not None and index not in selected_indexes:
            selected_indexes = [index]
        self.page_drag_state = (
            None
            if index is None
            else {
                "index": index,
                "start_y": event.y,
                "entry_ids": tuple(
                    self.session.entries[selected_index].entry_id
                    for selected_index in selected_indexes
                ),
                "dragged": False,
                "moved": False,
            }
        )

    def _on_page_drag_motion(self, event) -> str | None:
        state = self.page_drag_state
        if state is None or abs(event.y - int(state["start_y"])) < 5:
            return None
        if not state["dragged"]:
            state["dragged"] = True
            self.page_listbox.configure(cursor="fleur")
        drop_position = self._page_drop_position(event.y)
        if drop_position is None:
            return "break"
        target_index, place_after = drop_position
        entry_ids = tuple(state["entry_ids"])
        target_entry_id = self.session.entries[target_index].entry_id
        if self.session.reorder_entries(
            entry_ids,
            target_entry_id,
            place_after=place_after,
        ):
            state["moved"] = True
            self.refresh_page_list(keep_entry_ids=entry_ids)
            direction = "after" if place_after else "before"
            self._set_status(f"Moving {len(entry_ids)} page(s) {direction} page {target_index + 1}")
        return "break"

    def _on_page_drag_end(self, event) -> str | None:
        state = self.page_drag_state
        self.page_drag_state = None
        self.page_listbox.configure(cursor="")
        if state is None or not state["dragged"]:
            return None
        entry_ids = tuple(state["entry_ids"])
        if state["moved"]:
            self._set_status(f"Moved {len(entry_ids)} page(s)")
        return "break"

    def _show_page_context_menu(self, event) -> str:
        index = self._page_index_at_y(event.y)
        if index is not None and index not in self.page_listbox.curselection():
            self.page_listbox.selection_clear(0, tk.END)
            self.page_listbox.selection_set(index)
            self.page_listbox.activate(index)
            self.on_page_select()
        self._update_page_action_states()
        try:
            self.page_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.page_context_menu.grab_release()
        return "break"

    def _on_preview_mode_change(self, _value: str | None = None) -> None:
        self._layout_page_previews()
        self.update_idletasks()
        self.update_page_preview()

    def _layout_page_previews(self) -> None:
        """Give a single preview the full center area and split it only for Compare."""
        mode = self.preview_mode_var.get()
        if mode not in {"Processed", "Original", "Compare"}:
            mode = "Processed"
            self.preview_mode_var.set(mode)

        self.page_preview_before_frame.grid_forget()
        self.page_preview_after_frame.grid_forget()
        if mode == "Compare":
            self.page_preview_before_frame.grid(
                row=1,
                column=0,
                sticky="nsew",
                padx=(8, 4),
                pady=8,
            )
            self.page_preview_after_frame.grid(
                row=1,
                column=1,
                sticky="nsew",
                padx=(4, 8),
                pady=8,
            )
        elif mode == "Original":
            self.page_preview_before_frame.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=8,
                pady=8,
            )
        else:
            self.page_preview_after_frame.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=8,
                pady=8,
            )

    def update_page_preview(self) -> None:
        self._cancel_review_page_preview()
        selected = self.page_listbox.curselection()
        if len(selected) != 1:
            self._clear_preview_label(self.page_preview_before_label)
            self._clear_preview_label(self.page_preview_after_label)
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            self.page_preview_before_image = None
            self.page_preview_after_image = None
            return

        index = selected[0]
        if index < 0 or index >= len(self.session.entries):
            self._clear_preview_label(self.page_preview_before_label)
            self._clear_preview_label(self.page_preview_after_label)
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            self.page_preview_before_image = None
            self.page_preview_after_image = None
            return

        entry = self.session.entries[index]
        mode = self.preview_mode_var.get()
        split_ratio = (
            self.pending_split_ratio
            if self.pending_split_entry_id == entry.entry_id
            and self.pending_split_revision == entry.revision
            else None
        )
        self.page_preview_after_title.configure(
            text=(
                "Processed preview · 2 output pages"
                if split_ratio is not None
                else "Crop proposal — not exported"
                if _entry_has_crop_proposal(entry)
                else "Processed preview"
            )
        )
        try:
            before = self._review_before_image(entry)
        except Exception as exc:
            message = f"Preview failed: {exc}"
            self._set_preview_message(self.page_preview_before_label, message)
            self._set_preview_message(self.page_preview_after_label, message)
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            self.page_preview_before_image = None
            self.page_preview_after_image = None
            self._set_status(message)
            return

        if mode in {"Original", "Compare"}:
            self.page_preview_before_image = before
            before_photo = self._to_ctk_photo_for_label(before, self.page_preview_before_label)
            self.page_preview_before_label.configure(image=before_photo, text="")
            self.page_preview_before_photo = before_photo
        else:
            self.page_preview_before_photo = None
            self.page_preview_before_image = None

        if mode in {"Processed", "Compare"}:
            self.page_preview_after_image = None
            fast_preview = bool(self.lightweight_preview_var.get())
            # Reading one committed generation on Tk is quick; all expensive
            # processing runs after debounce on a cancellable worker.
            try:
                source = self._review_source_image(entry, fast_preview=fast_preview).copy()
                request = self._processing_request(entry=entry, preview=fast_preview)
            except Exception as exc:
                message = f"Preview failed: {exc}"
                self._set_preview_message(self.page_preview_after_label, message)
                self.page_preview_after_photo = None
                self._set_status(message)
                return
            generation = self.review_preview_generation
            cancel_event = self.review_preview_cancel_event
            mode_label = "fast preview (approximate)" if fast_preview else "full-resolution"
            self._set_preview_message(
                self.page_preview_after_label,
                f"Preparing {mode_label}...",
            )
            self.page_preview_after_photo = None
            self.review_preview_job = self.after(
                REVIEW_PREVIEW_DEBOUNCE_MS,
                lambda: self._launch_review_page_preview(
                    generation,
                    source,
                    request,
                    cancel_event,
                    split_ratio,
                ),
            )
        else:
            self.page_preview_after_photo = None
            self.page_preview_after_image = None

    def _on_review_preview_resize(self, _event=None) -> None:
        """Resize cached preview pixels after layout settles, without reprocessing the page."""
        if self._closing:
            return
        if self.review_preview_resize_job is not None:
            self.after_cancel(self.review_preview_resize_job)
        self.review_preview_resize_job = self.after(
            REVIEW_RESIZE_DEBOUNCE_MS,
            self._render_cached_review_previews,
        )

    def _render_cached_review_previews(self) -> None:
        self.review_preview_resize_job = None
        if self._closing or not self.winfo_exists():
            return
        render_size = (
            self.page_preview_before_label.winfo_width(),
            self.page_preview_before_label.winfo_height(),
            self.page_preview_after_label.winfo_width(),
            self.page_preview_after_label.winfo_height(),
        )
        if render_size == self.review_preview_render_size:
            return
        self.review_preview_render_size = render_size
        mode = self.preview_mode_var.get()
        if mode in {"Original", "Compare"} and self.page_preview_before_image is not None:
            photo = self._to_ctk_photo_for_label(
                self.page_preview_before_image,
                self.page_preview_before_label,
            )
            self.page_preview_before_label.configure(image=photo, text="")
            self.page_preview_before_photo = photo
        if mode in {"Processed", "Compare"} and self.page_preview_after_image is not None:
            photo = self._to_ctk_photo_for_label(
                self.page_preview_after_image,
                self.page_preview_after_label,
            )
            self.page_preview_after_label.configure(image=photo, text="")
            self.page_preview_after_photo = photo

    def _cancel_review_page_preview(self) -> None:
        self.review_preview_generation = getattr(self, "review_preview_generation", 0) + 1
        event = getattr(self, "review_preview_cancel_event", None)
        if event is not None:
            event.set()
        self.review_preview_cancel_event = threading.Event()
        job = getattr(self, "review_preview_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except (tk.TclError, ValueError):
                pass
            self.review_preview_job = None

    def _launch_review_page_preview(
        self,
        generation: int,
        source: np.ndarray,
        request: PageProcessingRequest,
        cancel_event: threading.Event,
        split_ratio: float | None = None,
    ) -> None:
        self.review_preview_job = None
        if self._closing or generation != self.review_preview_generation:
            return
        active_thread = self.review_preview_thread
        if active_thread is not None and active_thread.is_alive():
            self.review_preview_job = self.after(
                50,
                lambda: self._launch_review_page_preview(
                    generation,
                    source,
                    request,
                    cancel_event,
                    split_ratio,
                ),
            )
            return

        def run() -> None:
            try:
                processing_request = replace(request, cancel_cb=cancel_event.is_set)
                if split_ratio is None:
                    result = process_document_page(source, processing_request)
                    result_image = result.image
                    diagnostics = result.diagnostics
                else:
                    left_source, right_source = _split_at_ratio(source, split_ratio)
                    left_result = process_document_page(left_source, processing_request)
                    right_result = process_document_page(right_source, processing_request)
                    result_image = _compose_split_preview(
                        left_result.image,
                        right_result.image,
                    )
                    diagnostics = right_result.diagnostics
            except Exception as exc:
                if cancel_event.is_set():
                    return
                self.job_queue.put(("review_preview", (generation, None, None, str(exc))))
            else:
                if not cancel_event.is_set():
                    self.job_queue.put(
                        ("review_preview", (generation, result_image, diagnostics, None))
                    )

        self.review_preview_thread = threading.Thread(
            target=run,
            daemon=True,
            name=f"uniscan-review-preview-{generation}",
        )
        self.review_preview_threads = [
            thread for thread in self.review_preview_threads if thread.is_alive()
        ]
        self.review_preview_threads.append(self.review_preview_thread)
        self.review_preview_thread.start()

    def _handle_review_preview_result(
        self,
        generation: int,
        image: np.ndarray | None,
        diagnostics,
        error: str | None,
    ) -> None:
        if self._closing or generation != self.review_preview_generation:
            return
        self.review_preview_threads = [
            thread for thread in self.review_preview_threads if thread.is_alive()
        ]
        if error is not None or image is None:
            message = f"Preview failed: {error or 'no image was produced'}"
            self._set_preview_message(self.page_preview_after_label, message)
            self.page_preview_after_photo = None
            self._set_status(message)
            return
        after_photo = self._to_ctk_photo_for_label(image, self.page_preview_after_label)
        self.page_preview_after_label.configure(image=after_photo, text="")
        self.page_preview_after_photo = after_photo
        self.page_preview_after_image = image
        if diagnostics is not None:
            dewarp = diagnostics.dewarp
            if dewarp.applied:
                method = dewarp.selected_method.replace("_", " ")
                self.geometry_summary_var.set(
                    f"Wave preview: {method}, {dewarp.line_count} lines, "
                    f"{dewarp.max_displacement_px:.1f}px"
                )
            elif dewarp.reason == "disabled":
                self.geometry_summary_var.set("Wave preview: off")
            else:
                reason = (dewarp.reason or "no confident model").replace("_", " ")
                self.geometry_summary_var.set(f"Wave preview unchanged: {reason}")

    @staticmethod
    def _set_preview_message(label: ctk.CTkLabel, message: str) -> None:
        label._label.configure(image="")
        label.configure(image=None, text=message)

    def _clear_preview_label(self, label: ctk.CTkLabel) -> None:
        # customtkinter 5.2 leaves the old Tcl image name behind for image=None.
        message = (
            "Add files, paste, or drop pages here"
            if len(self.session) == 0
            else "Select one page to preview"
        )
        self._set_preview_message(label, message)

    def _single_selected_index(self) -> int | None:
        selected = self.page_listbox.curselection()
        if len(selected) != 1:
            return None
        return selected[0]

    def _single_selected_entry(self):
        index = self._single_selected_index()
        if index is None:
            return None, None
        if index < 0 or index >= len(self.session.entries):
            return None, None
        return index, self.session.entries[index]

    def move_selected_up(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to move.")
            return
        entry_ids = tuple(self.session.entries[index].entry_id for index in indices)
        moved = self.session.move_many(entry_ids, -1)
        if moved:
            self.refresh_page_list(keep_entry_ids=entry_ids)
            self._set_status(f"Moved {len(indices)} page(s) up")
        else:
            self._update_page_action_states()

    def move_selected_down(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to move.")
            return
        entry_ids = tuple(self.session.entries[index].entry_id for index in indices)
        moved = self.session.move_many(entry_ids, 1)
        if moved:
            self.refresh_page_list(keep_entry_ids=entry_ids)
            self._set_status(f"Moved {len(indices)} page(s) down")
        else:
            self._update_page_action_states()

    def select_all_pages(self) -> None:
        self.page_listbox.selection_set(0, tk.END)
        self._sync_page_selection_to_session()
        self._update_page_action_states()
        self.update_page_preview()
        self._set_status("Selected all pages")

    def clear_page_selection(self) -> None:
        self.page_listbox.selection_clear(0, tk.END)
        self._sync_page_selection_to_session()
        self._update_page_action_states()
        self.update_page_preview()
        self._set_status("Selection cleared")

    def delete_selected_pages(self) -> None:
        indices = self._selected_entry_indices()
        self._sync_page_selection_to_session()
        removed = self.session.remove_selected()
        if removed <= 0:
            self._set_status("No selected pages to delete")
            self._update_page_action_states()
            return
        keep_index = (
            min(indices[0], len(self.session.entries) - 1) if self.session.entries else None
        )
        self.refresh_page_list(keep_index=keep_index)
        self._set_status(f"Deleted {removed} page(s). Session pages: {len(self.session)}")

    def replace_selected_page_from_file(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page to replace.")
            return

        path = filedialog.askopenfilename(
            title="Replace selected page from image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image_path = Path(path)
            image = imread_unicode(image_path)
            if image is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

            result = self._detect_single_page(image, name=image_path.name)
            previous_committed = entry.committed_processing
            ok = self.session.replace_entry_image(
                entry.entry_id,
                raw_image=result.raw,
                original_image=result.raw,
                current_image=result.raw,
                name=image_path.name,
                contour=result.contour,
                backend=(
                    result.backend if result.detected and result.contour is not None else None
                ),
                crop_state=CROP_STATE_PROPOSED if result.contour is not None else CROP_STATE_NONE,
                needs_review=result.needs_review,
                review_reasons=result.review_reasons,
            )
            if not ok:
                raise RuntimeError("Selected page was not found in session.")
            self._reprocess_after_geometry_change(entry, previous_committed)

            self.refresh_page_list(keep_index=index)
            self._set_status(f"Replaced page {index + 1} from {image_path.name}.")
        except Exception as exc:
            messagebox.showerror("Replace Page Error", str(exc))
            self._set_status("Replace page failed")

    def retake_selected_page_from_camera(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page to retake.")
            return

        try:
            frame = self._capture_workspace_still()

            item_name = datetime.now().strftime(r"retake_%Y%m%d_%H%M%S")
            result = self._detect_single_page(frame, name=item_name)
            previous_committed = entry.committed_processing
            ok = self.session.replace_entry_image(
                entry.entry_id,
                raw_image=result.raw,
                original_image=result.raw,
                current_image=result.raw,
                contour=result.contour,
                backend=(
                    result.backend if result.detected and result.contour is not None else None
                ),
                crop_state=CROP_STATE_PROPOSED if result.contour is not None else CROP_STATE_NONE,
                needs_review=result.needs_review,
                review_reasons=result.review_reasons,
            )
            if not ok:
                raise RuntimeError("Selected page was not found in session.")
            self._reprocess_after_geometry_change(entry, previous_committed)

            self.refresh_page_list(keep_index=index)
            self._set_status(f"Retook page {index + 1} from camera.")
        except Exception as exc:
            messagebox.showerror("Retake Page Error", str(exc))
            self._set_status("Retake page failed")

    def _default_corner_points(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return np.array(
            [
                [0.0, 0.0],
                [float(max(0, width - 1)), 0.0],
                [float(max(0, width - 1)), float(max(0, height - 1))],
                [0.0, float(max(0, height - 1))],
            ],
            dtype=np.float32,
        )

    def _detect_corner_points(self, image: np.ndarray) -> np.ndarray | None:
        try:
            scan_output = scan_with_document_detector(
                image,
                enabled=True,
                backends=DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
                proposal_only=True,
            )
        except ScanAdapterError:
            return None

        contour = scan_output.contour
        if contour is None:
            return None

        points = np.array(contour, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 4:
            return None
        return points[:4]

    def _render_after_geometry_change(
        self,
        image: np.ndarray,
        previous_committed,
        *,
        perspective_points,
    ):
        previous_request = (
            previous_committed.recipe.to_request()
            if previous_committed is not None
            else PageProcessingRequest()
        )
        request = replace(
            previous_request,
            perspective_points=tuple(
                (float(x), float(y))
                for x, y in np.asarray(perspective_points, dtype=np.float32).reshape(4, 2)
            ),
            dewarp_already_applied=False,
            stage_cache=self.processing_cache,
            source_fingerprint=None,
            cancel_cb=None,
        )
        result = process_document_page(image, request)
        committed = CommittedPageProcessing.from_result(
            request,
            result.diagnostics,
            result.image,
        )
        return result.image, committed

    def _apply_perspective_crops(self, proposals: list[tuple]) -> None:
        """Stage every crop, then commit the set with best-effort rollback."""
        if not proposals:
            return
        with tempfile.TemporaryDirectory(prefix="uniscan_crop_apply_") as raw_stage_dir:
            stage_dir = Path(raw_stage_dir)
            staged = []
            seen_ids: set[str] = set()
            for index, (entry, source_image, points, backend) in enumerate(proposals):
                if entry.entry_id in seen_ids:
                    raise RuntimeError(f"Duplicate page in crop apply: {entry.name}")
                seen_ids.add(entry.entry_id)
                if source_image is None or source_image.size == 0:
                    raise RuntimeError(f"Selected page is empty: {entry.name}")
                normalized = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                planned = warp_perspective_from_points(source_image, normalized)
                if planned is None or planned.size == 0:
                    raise RuntimeError("Perspective transform returned empty image.")
                current, committed = self._render_after_geometry_change(
                    source_image,
                    entry.committed_processing,
                    perspective_points=normalized,
                )
                paths = {
                    name: stage_dir / f"{index:06d}-{name}.png"
                    for name in ("old-original", "old-current", "new-original", "new-current")
                }
                images = {
                    "old-original": entry.original_image,
                    "old-current": entry.current_image,
                    "new-original": source_image,
                    "new-current": current,
                }
                for name, image in images.items():
                    if not imwrite_unicode(paths[name], image):
                        raise RuntimeError(f"Cannot stage crop transaction for {entry.name}.")
                staged.append(
                    {
                        "entry": entry,
                        "paths": paths,
                        "contour": normalized.copy(),
                        "backend": backend or "manual",
                        "committed": committed,
                        "snapshot_contour": (
                            entry.detected_contour.copy()
                            if entry.detected_contour is not None
                            else None
                        ),
                        "snapshot_backend": entry.detected_backend,
                        "snapshot_crop_state": entry.crop_state,
                        "snapshot_needs_review": entry.needs_review,
                        "snapshot_review_reasons": entry.review_reasons,
                        "snapshot_control_points": entry.dewarp_control_points,
                        "snapshot_control_curves": entry.dewarp_control_curves,
                        "snapshot_committed": entry.committed_processing,
                        "snapshot_revision": entry.revision,
                    }
                )

            attempted = []
            try:
                for item in staged:
                    entry = item["entry"]
                    attempted.append(item)
                    original = imread_unicode(item["paths"]["new-original"])
                    current = imread_unicode(item["paths"]["new-current"])
                    if original is None or current is None:
                        raise RuntimeError(f"Cannot read staged crop for {entry.name}.")
                    if not self.session.replace_entry_image(
                        entry.entry_id,
                        original_image=original,
                        current_image=current,
                        contour=item["contour"],
                        backend=item["backend"],
                        crop_state=CROP_STATE_APPLIED,
                    ):
                        raise RuntimeError(f"Page disappeared during crop apply: {entry.name}")
                    entry.committed_processing = item["committed"]
                    entry.dewarp_control_points = item["snapshot_control_points"]
                    entry.dewarp_control_curves = item["snapshot_control_curves"]
            except Exception as exc:
                rollback_errors = []
                for item in reversed(attempted):
                    entry = item["entry"]
                    try:
                        original = imread_unicode(item["paths"]["old-original"])
                        current = imread_unicode(item["paths"]["old-current"])
                        if original is None or current is None:
                            raise RuntimeError("staged rollback pixels are unreadable")
                        if not self.session.replace_entry_image(
                            entry.entry_id,
                            original_image=original,
                            current_image=current,
                            contour=item["snapshot_contour"],
                            backend=item["snapshot_backend"],
                            crop_state=item["snapshot_crop_state"],
                            needs_review=item["snapshot_needs_review"],
                            review_reasons=item["snapshot_review_reasons"],
                        ):
                            raise RuntimeError("page disappeared during rollback")
                        entry.dewarp_control_points = item["snapshot_control_points"]
                        entry.dewarp_control_curves = item["snapshot_control_curves"]
                        entry.committed_processing = item["snapshot_committed"]
                        entry.revision = item["snapshot_revision"]
                    except Exception as rollback_exc:
                        rollback_errors.append(f"{entry.name}: {rollback_exc}")
                if rollback_errors:
                    raise RuntimeError(
                        f"Crop apply failed ({exc}); rollback also failed: "
                        + "; ".join(rollback_errors)
                    ) from exc
                raise

    def _apply_perspective_crop(
        self,
        entry,
        *,
        source_image: np.ndarray,
        points: np.ndarray,
        backend: str | None,
    ) -> None:
        """Atomically commit one previewed perspective crop."""
        self._apply_perspective_crops([(entry, source_image, points, backend)])

    def _show_inline_geometry_editor(self) -> ctk.CTkFrame:
        if self.inline_editor_host is None:
            raise RuntimeError("Workspace editor host is unavailable.")
        for frame in (
            self.workspace_page_list_frame,
            self.workspace_preview_frame,
            self.workspace_processing_frame,
        ):
            frame.grid_remove()
        for child in self.inline_editor_host.winfo_children():
            child.destroy()
        self.inline_editor_host.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="nsew",
            padx=10,
            pady=10,
        )
        editor = ctk.CTkFrame(self.inline_editor_host, fg_color="transparent")
        editor.grid(row=0, column=0, sticky="nsew")
        return editor

    def _hide_inline_geometry_editor(self) -> None:
        if self.inline_editor_host is not None:
            self.inline_editor_host.grid_remove()
        self.workspace_page_list_frame.grid()
        self.workspace_preview_frame.grid()
        self.workspace_processing_frame.grid()
        self.update_page_preview()

    def _open_corner_editor_dialog(
        self,
        indices: list[int],
        *,
        auto_detect: bool,
        initial_entry_index: int | None = None,
        from_current_geometry: bool = False,
    ) -> None:
        if not indices:
            self._set_status("Select page(s) for corner editing.")
            return

        indices = [idx for idx in indices if 0 <= idx < len(self.session.entries)]
        entries = [self.session.entries[idx] for idx in indices]
        if not entries:
            self._set_status("No valid pages available for corner editing.")
            return

        initial_position = (
            indices.index(initial_entry_index)
            if initial_entry_index is not None and initial_entry_index in indices
            else 0
        )
        state = {"index": initial_position}
        points_by_entry: dict[str, np.ndarray] = {}
        backend_by_entry: dict[str, str | None] = {}
        dirty_entry_ids: set[str] = set()
        source_images_by_entry = {
            entry.entry_id: _perspective_source_image(
                entry,
                from_current_geometry=from_current_geometry,
            )
            for entry in entries
        }
        selected_entry_ids = tuple(
            entry.entry_id for entry in self.session.entries if entry.selected
        )

        if self.corner_editor_window is not None:
            try:
                if self.corner_editor_window.winfo_exists():
                    self.corner_editor_window.lift()
                    return
            except tk.TclError:
                pass
            self.corner_editor_window = None
        if self.inline_editor_close_callback is not None:
            self.inline_editor_close_callback()

        win = self._show_inline_geometry_editor()
        self.corner_editor_window = win

        header = ctk.CTkLabel(
            win,
            text="Page perspective" if from_current_geometry else "Spread perspective",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        header.pack(fill=ctk.X, padx=12, pady=(12, 6))

        meta_var = tk.StringVar(value="")
        self.corner_meta_var = meta_var
        meta_label = ctk.CTkLabel(win, textvariable=meta_var, anchor="w")
        meta_label.pack(fill=ctk.X, padx=12, pady=(0, 8))

        canvas_frame = ctk.CTkFrame(win)
        canvas_frame.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(0, 10))
        canvas_frame.grid_rowconfigure(1, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(canvas_frame, text="Source and corner handles").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(canvas_frame, text="Perspective preview").grid(
            row=0, column=1, sticky="w", padx=8, pady=(8, 4)
        )

        canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        corrected_canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        corrected_canvas.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 8))
        self.corner_source_canvas = canvas
        self.corner_preview_canvas = corrected_canvas

        labels = ["TL", "TR", "BR", "BL"]
        drag = {"idx": None}
        canvas_image_ref = {"photo": None, "corrected_photo": None}
        view_state = {
            "source_shape": None,
            "display_shape": None,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "display_image": None,
            "source_entry_id": None,
            "source_image": None,
            "points": self._default_corner_points(
                source_images_by_entry[entries[state["index"]].entry_id]
            ),
            "dirty_entry_ids": dirty_entry_ids,
        }
        self.corner_editor_state = view_state

        def _map_display_points_to_source(
            points: np.ndarray, source_shape: tuple[int, int], display_shape: tuple[int, int]
        ) -> np.ndarray:
            source_h, source_w = source_shape
            display_h, display_w = display_shape
            mapped = np.array(points, dtype=np.float32).copy()
            mapped[:, 0] *= source_w / max(1, display_w)
            mapped[:, 1] *= source_h / max(1, display_h)
            return mapped

        def _current_entry() -> tuple[int, object]:
            entry_index = indices[state["index"]]
            return entry_index, self.session.entries[entry_index]

        def _init_points_for(entry) -> np.ndarray:
            cached = points_by_entry.get(entry.entry_id)
            if cached is not None:
                return cached

            source_image = source_images_by_entry[entry.entry_id]
            # Prefer the already-detected contour from import / previous edit.
            existing = entry.detected_contour
            if existing is not None and not auto_detect and not from_current_geometry:
                backend_by_entry[entry.entry_id] = entry.detected_backend
                points = np.asarray(existing, dtype=np.float32).reshape(-1, 2).copy()
                points_by_entry[entry.entry_id] = points
                return points

            detected: np.ndarray | None = None
            if auto_detect:
                detected = self._detect_corner_points(source_image)
                if detected is not None:
                    backend_by_entry[entry.entry_id] = "automatic"
                    dirty_entry_ids.add(entry.entry_id)
                if detected is None and existing is not None and not from_current_geometry:
                    backend_by_entry[entry.entry_id] = entry.detected_backend
                    detected = np.asarray(existing, dtype=np.float32).reshape(-1, 2)

            if detected is not None:
                points = np.asarray(detected, dtype=np.float32).reshape(-1, 2).copy()
            else:
                points = self._default_corner_points(source_image)
            points_by_entry[entry.entry_id] = points
            return points

        def _redraw() -> None:
            canvas.delete("overlay")
            points = view_state["points"]
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            offset_x = float(view_state["offset_x"])
            offset_y = float(view_state["offset_y"])
            display_w = (
                int(view_state["display_shape"][1])
                if view_state["display_shape"] is not None
                else 1
            )
            display_h = (
                int(view_state["display_shape"][0])
                if view_state["display_shape"] is not None
                else 1
            )
            line_points = []
            for pt in points:
                x = float(pt[0]) / max(scale_x, 1e-6) + offset_x
                y = float(pt[1]) / max(scale_y, 1e-6) + offset_y
                line_points.extend([x, y])
            canvas.create_line(
                *line_points,
                line_points[0],
                line_points[1],
                fill="#00ff66",
                width=2,
                tags="overlay",
            )
            for idx_p, pt in enumerate(points):
                sx = float(pt[0]) / max(scale_x, 1e-6) + offset_x
                sy = float(pt[1]) / max(scale_y, 1e-6) + offset_y
                if (
                    offset_x <= sx <= offset_x + display_w
                    and offset_y <= sy <= offset_y + display_h
                ):
                    r = 7
                    canvas.create_oval(
                        sx - r, sy - r, sx + r, sy + r, fill="#ff3355", outline="", tags="overlay"
                    )
                    canvas.create_text(
                        sx + 14, sy - 10, text=labels[idx_p], fill="#ffffff", tags="overlay"
                    )

        def _render_corrected_preview() -> None:
            source_image = view_state["source_image"]
            if source_image is None:
                return
            try:
                view_state["last_corrected_source_shape"] = source_image.shape
                corrected = warp_perspective_from_points(
                    source_image,
                    np.asarray(view_state["points"], dtype=np.float32),
                )
            except ValueError:
                return
            available_w = max(200, corrected_canvas.winfo_width() - 16)
            available_h = max(200, corrected_canvas.winfo_height() - 16)
            display_corrected = _fit_image_to_box(corrected, available_w, available_h)
            photo = _image_to_tk_photo(display_corrected)
            preview_h, preview_w = display_corrected.shape[:2]
            offset_x = max(0, (corrected_canvas.winfo_width() - preview_w) // 2)
            offset_y = max(0, (corrected_canvas.winfo_height() - preview_h) // 2)
            corrected_canvas.delete("all")
            corrected_canvas.create_image(
                offset_x,
                offset_y,
                image=photo,
                anchor=tk.NW,
                tags="perspective-preview",
            )
            canvas_image_ref["corrected_photo"] = photo

        def _load_current_entry() -> None:
            entry_index, entry = _current_entry()
            source_image = (
                view_state["source_image"]
                if view_state["source_entry_id"] == entry.entry_id
                else source_images_by_entry[entry.entry_id]
            )
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")

            available_w = max(200, canvas.winfo_width() - 16)
            available_h = max(200, canvas.winfo_height() - 16)
            display_image = _fit_image_to_box(
                source_image,
                available_w,
                available_h,
            )
            display_h, display_w = display_image.shape[:2]
            source_h, source_w = source_image.shape[:2]
            view_h = max(1, int(display_h))
            view_w = max(1, int(display_w))
            rgb = (
                cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
                if len(display_image.shape) == 2
                else cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            )
            tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            offset_x = max(0, (canvas.winfo_width() - view_w) // 2)
            offset_y = max(0, (canvas.winfo_height() - view_h) // 2)
            canvas.delete("all")
            canvas.create_image(
                offset_x,
                offset_y,
                image=tk_img,
                anchor=tk.NW,
                tags="source-image",
            )
            canvas_image_ref["photo"] = tk_img

            points = _init_points_for(entry)
            view_state["points"] = points
            view_state["source_shape"] = (source_h, source_w)
            view_state["display_shape"] = (display_h, display_w)
            view_state["scale_x"] = source_w / max(1, display_w)
            view_state["scale_y"] = source_h / max(1, display_h)
            view_state["offset_x"] = float(offset_x)
            view_state["offset_y"] = float(offset_y)
            view_state["display_image"] = display_image
            view_state["source_entry_id"] = entry.entry_id
            view_state["source_image"] = source_image
            meta_var.set(f"{state['index'] + 1}/{len(entries)}  {entry.name}")
            if self.corner_prev_button is not None:
                self.corner_prev_button.configure(
                    state=tk.NORMAL if state["index"] > 0 else tk.DISABLED
                )
            if self.corner_next_button is not None:
                self.corner_next_button.configure(
                    state=tk.NORMAL if state["index"] < len(entries) - 1 else tk.DISABLED
                )
            _redraw()
            _render_corrected_preview()

        def _nearest_handle(px: float, py: float) -> int | None:
            points = view_state["points"]
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            offset_x = float(view_state["offset_x"])
            offset_y = float(view_state["offset_y"])
            best_i = None
            best_d2 = 14.0 * 14.0
            for idx_p, pt in enumerate(points):
                sx = float(pt[0]) / max(scale_x, 1e-6) + offset_x
                sy = float(pt[1]) / max(scale_y, 1e-6) + offset_y
                d2 = (sx - px) ** 2 + (sy - py) ** 2
                if d2 <= best_d2:
                    best_i = idx_p
                    best_d2 = d2
            return best_i

        def _on_down(event):
            drag["idx"] = _nearest_handle(event.x, event.y)
            idx_p = drag["idx"]
            if idx_p is not None:
                point = view_state["points"][idx_p]
                _show_canvas_magnifier(
                    canvas,
                    view_state["source_image"],
                    float(point[0]),
                    float(point[1]),
                    float(event.x),
                    float(event.y),
                    canvas_image_ref,
                )

        def _on_move(event):
            idx_p = drag["idx"]
            if idx_p is None:
                return
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            offset_x = float(view_state["offset_x"])
            offset_y = float(view_state["offset_y"])
            source_h, source_w = view_state["source_shape"]
            x = (float(event.x) - offset_x) * max(scale_x, 1e-6)
            y = (float(event.y) - offset_y) * max(scale_y, 1e-6)
            x = max(0.0, min(float(source_w - 1), x))
            y = max(0.0, min(float(source_h - 1), y))
            points = view_state["points"]
            points[idx_p][0] = x
            points[idx_p][1] = y
            _entry_index, entry = _current_entry()
            backend_by_entry[entry.entry_id] = "manual"
            dirty_entry_ids.add(entry.entry_id)
            _redraw()
            _show_canvas_magnifier(
                canvas,
                view_state["source_image"],
                x,
                y,
                float(event.x),
                float(event.y),
                canvas_image_ref,
            )

        def _on_up(_event):
            drag["idx"] = None
            _hide_canvas_magnifier(canvas, canvas_image_ref)
            _render_corrected_preview()

        def _reset():
            source_h, source_w = view_state["source_shape"]
            points = view_state["points"]
            points[:] = self._default_corner_points(
                np.zeros((source_h, source_w, 3), dtype=np.uint8)
            )
            _entry_index, entry = _current_entry()
            backend_by_entry[entry.entry_id] = "manual"
            dirty_entry_ids.add(entry.entry_id)
            _redraw()
            _render_corrected_preview()

        def _auto_detect_current():
            entry_index, entry = _current_entry()
            detected = self._detect_corner_points(source_images_by_entry[entry.entry_id])
            if detected is None:
                messagebox.showwarning(
                    "Auto Crop", f"Document boundaries were not detected for {entry.name}."
                )
                return
            points = np.asarray(detected, dtype=np.float32).reshape(-1, 2).copy()
            points_by_entry[entry.entry_id] = points
            backend_by_entry[entry.entry_id] = "automatic"
            view_state["points"] = points
            dirty_entry_ids.add(entry.entry_id)
            _redraw()
            _render_corrected_preview()

        def _apply_current() -> None:
            _entry_index, entry = _current_entry()
            try:
                points = view_state["points"]
                self._apply_perspective_crop(
                    entry,
                    source_image=source_images_by_entry[entry.entry_id],
                    points=points,
                    backend=backend_by_entry.get(entry.entry_id) or entry.detected_backend,
                )
                dirty_entry_ids.discard(entry.entry_id)
                self.refresh_page_list(keep_entry_ids=(entry.entry_id,))
                self._set_status(f"Applied perspective crop to {entry.name}.")
            except Exception as exc:
                messagebox.showerror("Auto Crop Error", str(exc))

        def _apply_all():
            try:
                proposals = [
                    (
                        entry,
                        source_images_by_entry[entry.entry_id],
                        _init_points_for(entry),
                        backend_by_entry.get(entry.entry_id) or entry.detected_backend,
                    )
                    for entry in entries
                ]
                self._apply_perspective_crops(proposals)
                for entry in entries:
                    dirty_entry_ids.discard(entry.entry_id)
                self.refresh_page_list(keep_entry_ids=selected_entry_ids)
                self._set_status(f"Applied crop to {len(entries)} page(s).")
            except Exception as exc:
                messagebox.showerror("Auto Crop Error", str(exc))

        def _prev_page():
            if state["index"] > 0:
                state["index"] -= 1
                _load_current_entry()

        def _next_page():
            if state["index"] < len(entries) - 1:
                state["index"] += 1
                _load_current_entry()

        def _resize_editor_preview() -> None:
            self.corner_resize_job = None
            if self.corner_editor_window is win and win.winfo_exists():
                _load_current_entry()

        def _schedule_editor_resize(_event=None) -> None:
            if self.corner_resize_job is not None:
                win.after_cancel(self.corner_resize_job)
            self.corner_resize_job = win.after(
                REVIEW_RESIZE_DEBOUNCE_MS,
                _resize_editor_preview,
            )

        canvas.bind("<Button-1>", _on_down)
        canvas.bind("<B1-Motion>", _on_move)
        canvas.bind("<ButtonRelease-1>", _on_up)
        canvas_frame.bind("<Configure>", _schedule_editor_resize, add="+")

        controls = ctk.CTkFrame(win)
        controls.pack(fill=ctk.X, padx=12, pady=(0, 12))
        self.corner_prev_button = ctk.CTkButton(
            controls,
            text="Prev",
            width=90,
            command=_prev_page,
        )
        self.corner_prev_button.pack(side=ctk.LEFT)
        self.corner_next_button = ctk.CTkButton(
            controls,
            text="Next",
            width=90,
            command=_next_page,
        )
        self.corner_next_button.pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(controls, text="Auto Detect", width=110, command=_auto_detect_current).pack(
            side=ctk.LEFT,
            padx=6,
        )
        self.corner_apply_button = ctk.CTkButton(
            controls,
            text="Apply current",
            width=120,
            command=_apply_current,
        )
        self.corner_apply_button.pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(controls, text="Reset", width=90, command=_reset).pack(side=ctk.LEFT, padx=6)
        if auto_detect:
            ctk.CTkButton(
                controls,
                text="Apply detected to all",
                width=150,
                command=_apply_all,
            ).pack(side=ctk.LEFT, padx=6)

        def _close_editor() -> None:
            if self.corner_resize_job is not None:
                win.after_cancel(self.corner_resize_job)
                self.corner_resize_job = None
            self.corner_editor_window = None
            self.corner_source_canvas = None
            self.corner_preview_canvas = None
            self.corner_meta_var = None
            self.corner_prev_button = None
            self.corner_apply_button = None
            self.corner_next_button = None
            self.corner_editor_state = None
            self.inline_editor_close_callback = None
            win.destroy()
            self.refresh_page_list(keep_entry_ids=selected_entry_ids)
            if dirty_entry_ids:
                self._set_status(
                    f"Closed corner editor without applying {len(dirty_entry_ids)} pending edit(s)."
                )
            self._hide_inline_geometry_editor()

        self.corner_close_button = ctk.CTkButton(
            controls,
            text="Done",
            width=90,
            command=_close_editor,
        )
        self.corner_close_button.pack(side=ctk.RIGHT)
        self.inline_editor_close_callback = _close_editor

        win.after_idle(_load_current_entry)

    def open_manual_corners_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page for manual corner edit.")
            return
        self._open_corner_editor_dialog(
            list(range(len(self.session.entries))),
            auto_detect=False,
            initial_entry_index=index,
        )

    def open_current_geometry_corners_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page for additional perspective correction.")
            return
        self._open_corner_editor_dialog(
            list(range(len(self.session.entries))),
            auto_detect=False,
            initial_entry_index=index,
            from_current_geometry=True,
        )

    def open_auto_crop_editor(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            indices = list(range(len(self.session.entries)))
        if not indices:
            self._set_status("No pages available for auto crop.")
            return
        self._open_corner_editor_dialog(indices, auto_detect=True)

    def _reprocess_entry_from_original(self, entry):
        request = self._processing_request(entry=entry, preview=False)
        return self._commit_processing_request(entry, request)

    def _commit_processing_request(self, entry, request: PageProcessingRequest):
        result = process_document_page(entry.original_image, request)
        committed = CommittedPageProcessing.from_result(
            request,
            result.diagnostics,
            result.image,
        )
        entry.current_image = result.image
        entry.committed_processing = committed
        self._last_processing_cache_hits = result.diagnostics.cache_hits
        return result.diagnostics.dewarp

    def _reprocess_after_geometry_change(
        self,
        entry,
        previous_committed,
        *,
        baked_stages: frozenset[str] = frozenset(),
    ) -> None:
        """Replay the durable recipe after an upstream edit.

        Only stages explicitly baked into ``original_image`` are disabled.
        Every downstream automatic or manual policy is otherwise preserved.
        """
        if previous_committed is None:
            return
        previous_request = previous_committed.recipe.to_request()
        request = replace(
            previous_request,
            orientation_method=(
                ORIENTATION_METHOD_NONE
                if "orientation" in baked_stages
                else previous_request.orientation_method
            ),
            deskew_method=(
                DESKEW_METHOD_NONE if "deskew" in baked_stages else previous_request.deskew_method
            ),
            deskew_angle_degrees=(
                None if "deskew" in baked_stages else previous_request.deskew_angle_degrees
            ),
            dewarp_already_applied=False,
            stage_cache=self.processing_cache,
            source_fingerprint=None,
            cancel_cb=None,
        )
        self._commit_processing_request(entry, request)

    def _reprocess_with_dewarp(self, entry, previous_committed, *, method: str):
        """Change only dewarp while preserving the page's committed appearance recipe."""
        request = (
            previous_committed.recipe.to_request()
            if previous_committed is not None
            else PageProcessingRequest()
        )
        request = replace(
            request,
            dewarp_method=method,
            dewarp_model=self._entry_dewarp_model(entry),
            dewarp_already_applied=False,
            stage_cache=self.processing_cache,
            source_fingerprint=None,
            cancel_cb=None,
        )
        return self._commit_processing_request(entry, request)

    def analyze_selected_page_lighting(self) -> None:
        _index, entry = self._single_selected_entry()
        if entry is None:
            self._set_status("Select exactly one page to analyze lighting.")
            return
        request = self._processing_request(
            entry=entry,
            preview=False,
            lighting_diagnostics=True,
        )
        lighting = process_document_page(entry.original_image, request).diagnostics.lighting
        if lighting is None:
            raise RuntimeError("Lighting diagnostics were not produced.")
        warnings = ", ".join(lighting.warnings) if lighting.warnings else "none"
        self.lighting_summary_var.set(
            f"Shadow {lighting.shadow_fraction:.1%} | glare {lighting.glare_fraction:.1%}\n"
            f"Clipped {lighting.clipped_pixel_fraction:.1%} | warnings: {warnings}"
        )
        self._set_status(
            f"Lighting analyzed: unevenness {lighting.unevenness:.2f}; warnings {warnings}."
        )

    def _selected_entry_indices(self) -> list[int]:
        indexes = list(self.page_listbox.curselection())
        valid = [idx for idx in indexes if 0 <= idx < len(self.session.entries)]
        return valid

    def rotate_selected_left(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to rotate.")
            return
        for idx in indices:
            entry = self.session.entries[idx]
            rotated = cv2.rotate(entry.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            previous_committed = entry.committed_processing
            entry.original_image = rotated
            self._reprocess_after_geometry_change(
                entry,
                previous_committed,
                baked_stages=frozenset({"orientation"}),
            )
        entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)
        self.refresh_page_list(keep_entry_ids=entry_ids)
        self._set_status(f"Rotated {len(indices)} page(s) left.")

    def rotate_selected_right(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to rotate.")
            return
        for idx in indices:
            entry = self.session.entries[idx]
            rotated = cv2.rotate(entry.original_image, cv2.ROTATE_90_CLOCKWISE)
            previous_committed = entry.committed_processing
            entry.original_image = rotated
            self._reprocess_after_geometry_change(
                entry,
                previous_committed,
                baked_stages=frozenset({"orientation"}),
            )
        entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)
        self.refresh_page_list(keep_entry_ids=entry_ids)
        self._set_status(f"Rotated {len(indices)} page(s) right.")

    def _clear_pending_split_preview(self) -> None:
        self.pending_split_entry_id = None
        self.pending_split_ratio = None
        self.pending_split_revision = None
        self.split_preview_var.set("Split: not previewed")
        button = getattr(self, "apply_split_button", None)
        if button is not None:
            button.configure(state=tk.DISABLED)
        title = getattr(self, "page_preview_after_title", None)
        if title is not None:
            title.configure(text="Processed preview")

    def _activate_split_preview(self, entry, ratio: float) -> None:
        ratio = float(np.clip(ratio, 0.05, 0.95))
        self.pending_split_entry_id = entry.entry_id
        self.pending_split_ratio = ratio
        self.pending_split_revision = entry.revision
        self.split_preview_var.set(f"Split: 2 pages at {ratio * 100:.1f}%")
        self.apply_split_button.configure(state=tk.NORMAL)
        self.preview_mode_var.set("Compare")
        self._layout_page_previews()
        self.update_page_preview()
        self._set_status("Split preview ready: original spread and two output pages.")

    def open_split_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one spread to adjust its split.")
            return
        if self.split_editor_window is not None:
            try:
                if self.split_editor_window.winfo_exists():
                    self.split_editor_window.lift()
                    return
            except tk.TclError:
                pass
            self.split_editor_window = None
        if self.inline_editor_close_callback is not None:
            self.inline_editor_close_callback()

        source = entry.original_image
        source_height, source_width = source.shape[:2]
        if source_width < 2:
            self._set_status("The selected image is too narrow to split.")
            return

        detected_pair = _split_spread_pair(entry.raw_image, source)
        detected_ratio = None
        if detected_pair is not None:
            detected_ratio = detected_pair[1][0].shape[1] / max(1, source_width)
        if (
            self.pending_split_entry_id == entry.entry_id
            and self.pending_split_revision == entry.revision
            and self.pending_split_ratio is not None
        ):
            initial_ratio = self.pending_split_ratio
            initial_message = "Loaded the current split preview."
        elif detected_ratio is not None:
            initial_ratio = detected_ratio
            initial_message = "Automatic gutter detected. Drag the line to correct it."
        else:
            initial_ratio = 0.5
            initial_message = "No confident gutter found. The line starts at the center."

        window = self._show_inline_geometry_editor()
        self.split_editor_window = window
        ctk.CTkLabel(
            window,
            text="Spread split",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(fill=ctk.X, padx=16, pady=(14, 2))

        panes = ctk.CTkFrame(window)
        panes.pack(fill=ctk.BOTH, expand=True, padx=16, pady=(0, 10))
        panes.grid_columnconfigure(0, weight=1)
        panes.grid_columnconfigure(1, weight=1)
        panes.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(panes, text="Source and split line").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(panes, text="Two output pages").grid(
            row=0, column=1, sticky="w", padx=8, pady=(8, 4)
        )
        source_canvas = tk.Canvas(panes, bg="black", highlightthickness=0)
        preview_canvas = tk.Canvas(panes, bg="black", highlightthickness=0)
        source_canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        preview_canvas.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 8))
        self.split_source_canvas = source_canvas
        self.split_preview_canvas = preview_canvas

        state: dict[str, object] = {
            "ratio": float(np.clip(initial_ratio, 0.05, 0.95)),
            "dragging": False,
            "display_width": source_width,
            "display_height": source_height,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "source_photo": None,
            "preview_photo": None,
            "magnifier_photo": None,
            "source_shape": source.shape,
        }
        self.split_editor_state = state
        status = tk.StringVar(value=initial_message)
        self.split_editor_status_var = status

        def draw_split_line() -> None:
            source_canvas.delete("split-overlay")
            x_pos = float(state["offset_x"]) + float(state["ratio"]) * max(
                1, int(state["display_width"]) - 1
            )
            top = float(state["offset_y"])
            bottom = top + int(state["display_height"])
            source_canvas.create_line(
                x_pos,
                top,
                x_pos,
                bottom,
                fill="#00ff66",
                width=3,
                tags="split-overlay",
            )
            center_y = (top + bottom) / 2
            source_canvas.create_oval(
                x_pos - 9,
                center_y - 9,
                x_pos + 9,
                center_y + 9,
                fill="#ff3355",
                outline="#ffffff",
                width=2,
                tags="split-overlay",
            )

        def render_output() -> None:
            left, right = _split_at_ratio(source, float(state["ratio"]))
            full_preview = _compose_split_preview(left, right)
            display_preview = _fit_image_to_box(
                full_preview,
                max(200, preview_canvas.winfo_width() - 16),
                max(200, preview_canvas.winfo_height() - 16),
            )
            state["preview_photo"] = _image_to_tk_photo(display_preview)
            height, width = display_preview.shape[:2]
            offset_x = max(0, (preview_canvas.winfo_width() - width) // 2)
            offset_y = max(0, (preview_canvas.winfo_height() - height) // 2)
            preview_canvas.delete("all")
            preview_canvas.create_image(
                offset_x,
                offset_y,
                image=state["preview_photo"],
                anchor=tk.NW,
                tags="split-preview",
            )

        def render_views() -> None:
            self.split_resize_job = None
            if self.split_editor_window is not window or not window.winfo_exists():
                return
            display_source = _fit_image_to_box(
                source,
                max(200, source_canvas.winfo_width() - 16),
                max(200, source_canvas.winfo_height() - 16),
            )
            display_height, display_width = display_source.shape[:2]
            offset_x = max(0, (source_canvas.winfo_width() - display_width) // 2)
            offset_y = max(0, (source_canvas.winfo_height() - display_height) // 2)
            state.update(
                {
                    "display_width": display_width,
                    "display_height": display_height,
                    "offset_x": float(offset_x),
                    "offset_y": float(offset_y),
                    "source_photo": _image_to_tk_photo(display_source),
                }
            )
            source_canvas.delete("all")
            source_canvas.create_image(
                offset_x,
                offset_y,
                image=state["source_photo"],
                anchor=tk.NW,
                tags="split-source",
            )
            draw_split_line()
            render_output()

        def ratio_at(x_pos: float) -> float:
            local_x = float(x_pos) - float(state["offset_x"])
            return float(np.clip(local_x / max(1, int(state["display_width"]) - 1), 0.05, 0.95))

        def show_magnifier(event) -> None:
            ratio = float(state["ratio"])
            local_y = float(event.y) - float(state["offset_y"])
            source_y = np.clip(
                local_y / max(1, int(state["display_height"]) - 1) * (source_height - 1),
                0,
                source_height - 1,
            )
            _show_canvas_magnifier(
                source_canvas,
                source,
                ratio * (source_width - 1),
                float(source_y),
                float(event.x),
                float(event.y),
                state,
            )

        def on_down(event) -> None:
            line_x = float(state["offset_x"]) + float(state["ratio"]) * max(
                1, int(state["display_width"]) - 1
            )
            if abs(float(event.x) - line_x) > 20:
                return
            state["dragging"] = True
            show_magnifier(event)

        def on_move(event) -> None:
            if not state["dragging"]:
                return
            state["ratio"] = ratio_at(event.x)
            draw_split_line()
            show_magnifier(event)
            status.set(f"Split position: {float(state['ratio']) * 100:.1f}%")

        def on_up(_event) -> None:
            if not state["dragging"]:
                return
            state["dragging"] = False
            _hide_canvas_magnifier(source_canvas, state)
            render_output()

        def use_automatic() -> None:
            if detected_ratio is None:
                status.set("No confident automatic gutter was found.")
                return
            state["ratio"] = float(detected_ratio)
            draw_split_line()
            render_output()
            status.set(f"Automatic split position: {detected_ratio * 100:.1f}%")

        def close_editor() -> None:
            if self.split_resize_job is not None:
                window.after_cancel(self.split_resize_job)
                self.split_resize_job = None
            self.split_editor_window = None
            self.split_source_canvas = None
            self.split_preview_canvas = None
            self.split_editor_state = None
            self.split_editor_status_var = None
            self.inline_editor_close_callback = None
            window.destroy()
            self._hide_inline_geometry_editor()

        def preview_split() -> None:
            ratio = float(state["ratio"])
            close_editor()
            self._activate_split_preview(entry, ratio)

        def schedule_resize(_event=None) -> None:
            if self.split_resize_job is not None:
                window.after_cancel(self.split_resize_job)
            self.split_resize_job = window.after(REVIEW_RESIZE_DEBOUNCE_MS, render_views)

        source_canvas.bind("<Button-1>", on_down)
        source_canvas.bind("<B1-Motion>", on_move)
        source_canvas.bind("<ButtonRelease-1>", on_up)
        panes.bind("<Configure>", schedule_resize, add="+")
        ctk.CTkLabel(window, textvariable=status, anchor="w").pack(fill=ctk.X, padx=16, pady=(0, 8))
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(fill=ctk.X, padx=16, pady=(0, 14))
        ctk.CTkButton(actions, text="Auto detect", command=use_automatic).pack(side=ctk.LEFT)
        self.split_editor_preview_button = ctk.CTkButton(
            actions,
            text="Preview split",
            command=preview_split,
        )
        self.split_editor_preview_button.pack(side=ctk.RIGHT)
        self.split_editor_close_button = ctk.CTkButton(
            actions,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=close_editor,
        )
        self.split_editor_close_button.pack(side=ctk.RIGHT, padx=8)
        self.inline_editor_close_callback = close_editor
        window.after_idle(render_views)

    def preview_selected_spread_split(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one spread to preview its split.")
            return
        split_pair = _split_spread_pair(entry.raw_image, entry.original_image)
        if split_pair is None:
            self._clear_pending_split_preview()
            self._set_status("No confident spread gutter was found; the page was not changed.")
            self.split_preview_var.set("Split: no confident gutter")
            return
        _raw_halves, warped_halves = split_pair
        ratio = warped_halves[0].shape[1] / max(1, entry.original_image.shape[1])
        self._activate_split_preview(entry, ratio)

    def _commit_entry_split(self, index: int, entry, ratio: float):
        previous_committed = entry.committed_processing
        left_raw, right_raw = _split_at_ratio(entry.raw_image, ratio)
        left_warped, right_warped = _split_at_ratio(entry.original_image, ratio)
        base_name = re.sub(r" \[[LR]\]$", "", entry.name)
        self.session.replace_entry_image(
            entry.entry_id,
            raw_image=left_raw,
            original_image=left_warped,
            current_image=left_warped,
            name=f"{base_name} [L]",
            contour=None,
            backend="intentional_split",
        )
        right_entry = self.session.add_image_with_contour(
            name=f"{base_name} [R]",
            raw_image=right_raw,
            warped_image=right_warped,
            contour=None,
            backend="intentional_split",
        )
        self._reprocess_after_geometry_change(entry, previous_committed)
        self._reprocess_after_geometry_change(right_entry, previous_committed)
        from_index = self.session.entries.index(right_entry)
        while from_index > index + 1:
            if not self.session.move(right_entry.entry_id, -1):
                break
            from_index -= 1
        return right_entry

    def apply_previewed_spread_split(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select the spread used for the split preview.")
            return
        if (
            self.pending_split_entry_id != entry.entry_id
            or self.pending_split_ratio is None
            or self.pending_split_revision != entry.revision
        ):
            self._clear_pending_split_preview()
            self._set_status("Split preview is stale; preview the spread again.")
            return
        try:
            right_entry = self._commit_entry_split(index, entry, self.pending_split_ratio)
            self._clear_pending_split_preview()
            self.preview_mode_var.set("Processed")
            self._layout_page_previews()
            self.refresh_page_list(keep_entry_ids=(entry.entry_id,))
            self._set_status(
                f"Created pages {entry.name} and {right_entry.name}; left page selected."
            )
        except Exception as exc:
            messagebox.showerror("Split Spread Error", str(exc))
            self._set_status("Split spread failed")

    def auto_deskew_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to deskew.")
            return

        angles: list[float] = []
        entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)
        method = DESKEW_UI_METHODS.get(
            self.deskew_method_var.get(),
            DESKEW_METHOD_HYBRID,
        )
        if method in {DESKEW_METHOD_NONE, DESKEW_METHOD_MANUAL}:
            method = DESKEW_METHOD_HYBRID
        for idx in indices:
            entry = self.session.entries[idx]
            previous_committed = entry.committed_processing
            request = (
                previous_committed.recipe.to_request()
                if previous_committed is not None
                else self._processing_request(entry=entry, preview=False)
            )
            request = replace(
                request,
                deskew_method=method,
                deskew_angle_degrees=None,
                stage_cache=self.processing_cache,
                source_fingerprint=None,
                cancel_cb=None,
            )
            self._commit_processing_request(entry, request)
            assert entry.committed_processing is not None
            angles.append(float(entry.committed_processing.diagnostics["deskew_angle_degrees"]))

        self.refresh_page_list(keep_entry_ids=entry_ids)
        mean_angle = sum(angles) / max(1, len(angles))
        self._set_status(f"Deskewed {len(indices)} page(s), avg angle {mean_angle:.1f} deg.")

    def auto_orient_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to orient.")
            return

        diagnostics = []
        entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)
        for idx in indices:
            entry = self.session.entries[idx]
            previous_committed = entry.committed_processing
            request = (
                previous_committed.recipe.to_request()
                if previous_committed is not None
                else self._processing_request(entry=entry, preview=False)
            )
            request = replace(
                request,
                orientation_method=ORIENTATION_METHOD_AUTO,
                stage_cache=self.processing_cache,
                source_fingerprint=None,
                cancel_cb=None,
            )
            result = process_document_page(entry.original_image, request)
            item = result.diagnostics.orientation
            committed = CommittedPageProcessing.from_result(
                request,
                result.diagnostics,
                result.image,
            )
            entry.current_image = result.image
            entry.committed_processing = committed
            diagnostics.append(item)

        self.refresh_page_list(keep_entry_ids=entry_ids)
        applied = sum(item.applied for item in diagnostics)
        uncertain = sum(item.reason not in {None, "already_upright"} for item in diagnostics)
        self._set_status(
            f"Auto-oriented {applied}/{len(indices)} page(s); "
            f"{uncertain} left unchanged as uncertain."
        )

    def open_dewarp_points_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page to adjust dewarp points.")
            return
        if self.dewarp_editor_window is not None:
            try:
                if self.dewarp_editor_window.winfo_exists():
                    self.dewarp_editor_window.lift()
                    return
            except tk.TclError:
                pass
            self.dewarp_editor_window = None
        if self.inline_editor_close_callback is not None:
            self.inline_editor_close_callback()

        authoritative_source = entry.original_image
        previous_committed = entry.committed_processing
        committed_dewarp_method = DEWARP_UI_METHODS.get(
            self.dewarp_method_var.get(),
            DEWARP_METHOD_TEXTLINE,
        )
        if previous_committed is not None:
            committed_dewarp_method = previous_committed.recipe.dewarp_method
            if committed_dewarp_method == DEWARP_METHOD_AUTO:
                prior_dewarp = previous_committed.diagnostics.get("dewarp", {})
                if isinstance(prior_dewarp, dict):
                    selected = prior_dewarp.get("selected_method")
                    if selected in {
                        DEWARP_METHOD_TEXTLINE,
                        DEWARP_METHOD_UVDOC,
                        DEWARP_METHOD_DOCSCANNER,
                        DEWARP_METHOD_PADDLEOCR_UVDOC,
                    }:
                        committed_dewarp_method = str(selected)
        if committed_dewarp_method == DEWARP_METHOD_NONE:
            committed_dewarp_method = DEWARP_METHOD_TEXTLINE
        editor_base_request = (
            previous_committed.recipe.to_request()
            if previous_committed is not None
            else self._processing_request(entry=entry, preview=False)
        )
        display_dewarp_method = (
            committed_dewarp_method
            if committed_dewarp_method
            in {
                DEWARP_METHOD_UVDOC,
                DEWARP_METHOD_DOCSCANNER,
                DEWARP_METHOD_PADDLEOCR_UVDOC,
            }
            else DEWARP_METHOD_NONE
        )
        model_input_request = replace(
            editor_base_request,
            dewarp_method=display_dewarp_method,
            dewarp_model=None,
            dewarp_already_applied=False,
            deskew_method=DESKEW_METHOD_NONE,
            deskew_angle_degrees=None,
            shadow_method=SHADOW_METHOD_NONE,
            postprocess_name="None",
            preprocess_settings=None,
            page_layout="none",
            lighting_diagnostics=False,
            stage_cache=self.processing_cache,
            source_fingerprint=None,
            cancel_cb=None,
        )
        source = process_document_page(authoritative_source, model_input_request).image
        source_height, source_width = source.shape[:2]

        if entry.dewarp_control_curves is not None:
            initial_curves = [
                {"anchor": anchor, "points": list(points)}
                for anchor, points in entry.dewarp_control_curves
            ]
            initial_message = "Loaded three saved page-correction curves."
        elif entry.dewarp_control_points is not None:
            initial_points = list(entry.dewarp_control_points)
            initial_curves = [
                {"anchor": anchor, "points": list(initial_points)} for anchor in (0.25, 0.5, 0.75)
            ]
            initial_message = "Expanded the saved legacy curve across three page regions."
        else:
            automatic_model, automatic_diagnostics = estimate_textline_dewarp_model(source)
            if automatic_model is not None:
                initial_points = list(automatic_model.control_points)
                initial_message = (
                    f"Automatic model: {automatic_diagnostics.line_count} supporting lines, "
                    f"{automatic_diagnostics.max_displacement_px:.1f}px max correction."
                )
            else:
                initial_points = [
                    (float(x), 0.0) for x in np.linspace(0.0, 1.0, 9, dtype=np.float32)
                ]
                initial_message = (
                    "Automatic model was not confident; adjust the neutral points if needed."
                )
            initial_curves = [
                {"anchor": anchor, "points": list(initial_points)} for anchor in (0.25, 0.5, 0.75)
            ]

        window = self._show_inline_geometry_editor()
        self.dewarp_editor_window = window

        ctk.CTkLabel(
            window,
            text="Page wave correction",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))

        curve_names = ("Top", "Middle", "Bottom")
        curve_colors = ("#28c7d9", "#36a3ff", "#e05aa8")

        panes = ctk.CTkFrame(window)
        panes.pack(fill=ctk.BOTH, expand=True, padx=16, pady=(0, 10))
        panes.grid_columnconfigure(0, weight=1)
        panes.grid_columnconfigure(1, weight=1)
        panes.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(panes, text="Source model").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(panes, text="Corrected preview").grid(
            row=0, column=1, sticky="w", padx=8, pady=(8, 4)
        )

        left_canvas = tk.Canvas(
            panes,
            bg="#202225",
            highlightthickness=1,
            highlightbackground="#45484d",
        )
        right_canvas = tk.Canvas(
            panes,
            bg="#202225",
            highlightthickness=1,
            highlightbackground="#45484d",
        )
        left_canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        right_canvas.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 8))
        self.dewarp_source_canvas = left_canvas
        self.dewarp_preview_canvas = right_canvas

        active_curve = min(1, len(initial_curves) - 1)
        state = {
            "curves": initial_curves,
            "active_curve": active_curve,
            "points": initial_curves[active_curve]["points"],
            "active": None,
            "selected": None,
            "drag_mode": None,
            "drag_start_y": 0.0,
            "drag_start_anchor": 0.5,
            "guide_anchor": initial_curves[active_curve]["anchor"],
            "add_mode": False,
            "display_width": source_width,
            "display_height": source_height,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "source_photo": None,
            "corrected_photo": None,
            "magnifier_photo": None,
            "source_shape": source.shape,
        }
        status = tk.StringVar(value=initial_message)
        self.dewarp_editor_state = state
        self.dewarp_status_var = status

        def selected_curve() -> dict[str, object]:
            return state["curves"][int(state["active_curve"])]

        def select_curve(curve_index: int) -> None:
            curve_index = int(np.clip(curve_index, 0, len(state["curves"]) - 1))
            state["active_curve"] = curve_index
            curve = selected_curve()
            state["points"] = curve["points"]
            state["guide_anchor"] = curve["anchor"]
            state["selected"] = None
            state["active"] = None
            draw_overlay()
            status.set(f"Editing the {curve_names[curve_index].lower()} page curve.")

        def current_model(*, source_name: str = "user") -> DewarpModel:
            curves = tuple(
                (float(curve["anchor"]), tuple(curve["points"])) for curve in state["curves"]
            )
            return DewarpModel(
                method=DEWARP_METHOD_TEXTLINE,
                control_points=curves[min(1, len(curves) - 1)][1],
                source=source_name,
                control_curves=curves,
            )

        def adjusted_request() -> PageProcessingRequest:
            return replace(
                editor_base_request,
                dewarp_method=committed_dewarp_method,
                dewarp_model=current_model(),
                dewarp_already_applied=False,
                stage_cache=self.processing_cache,
                source_fingerprint=None,
                cancel_cb=None,
            )

        def draw_overlay() -> None:
            left_canvas.delete("dewarp-overlay")
            display_width = int(state["display_width"])
            display_height = int(state["display_height"])
            offset_x = float(state["offset_x"])
            offset_y = float(state["offset_y"])
            guide_x = np.linspace(0.0, 1.0, 160, dtype=np.float32)
            for curve_index, curve in enumerate(state["curves"]):
                points = curve["points"]
                guide_y = interpolate_control_curve(points, guide_x)
                guide_anchor = float(curve["anchor"])
                coords: list[float] = []
                for x_value, displacement in zip(guide_x, guide_y):
                    coords.extend(
                        [
                            offset_x + float(x_value * (display_width - 1)),
                            offset_y + float((guide_anchor + displacement) * display_height),
                        ]
                    )
                left_canvas.create_line(
                    *coords,
                    fill=curve_colors[curve_index],
                    width=3 if curve_index == state["active_curve"] else 2,
                    tags="dewarp-overlay",
                )
                for point_index, (x_value, displacement) in enumerate(points):
                    x_pos = offset_x + x_value * (display_width - 1)
                    y_pos = offset_y + (guide_anchor + displacement) * display_height
                    active = curve_index == state["active_curve"]
                    selected = active and point_index == state["selected"]
                    radius = 8 if selected else (6 if active else 3)
                    left_canvas.create_oval(
                        x_pos - radius,
                        y_pos - radius,
                        x_pos + radius,
                        y_pos + radius,
                        fill="#ffb000" if selected else curve_colors[curve_index],
                        outline="#ffffff" if active else curve_colors[curve_index],
                        width=2 if selected else 1,
                        tags=("dewarp-overlay", f"curve-{curve_index}-point-{point_index}"),
                    )

        def render_corrected() -> None:
            state["last_corrected_source_shape"] = authoritative_source.shape
            corrected = process_document_page(authoritative_source, adjusted_request()).image
            state["last_corrected"] = corrected
            display_corrected = _fit_image_to_box(
                corrected,
                max(200, right_canvas.winfo_width() - 16),
                max(200, right_canvas.winfo_height() - 16),
            )
            state["corrected_photo"] = _image_to_tk_photo(display_corrected)
            right_canvas.delete("all")
            corrected_height, corrected_width = display_corrected.shape[:2]
            offset_x = max(
                0,
                (right_canvas.winfo_width() - corrected_width) // 2,
            )
            offset_y = max(
                0,
                (right_canvas.winfo_height() - corrected_height) // 2,
            )
            right_canvas.create_image(
                offset_x,
                offset_y,
                anchor=tk.NW,
                image=state["corrected_photo"],
                tags="dewarp-preview",
            )

        def render_views() -> None:
            self.dewarp_resize_job = None
            if self.dewarp_editor_window is not window or not window.winfo_exists():
                return
            available_width = max(
                200,
                min(left_canvas.winfo_width(), right_canvas.winfo_width()) - 16,
            )
            available_height = max(
                200,
                min(left_canvas.winfo_height(), right_canvas.winfo_height()) - 16,
            )
            scale = min(
                available_width / max(1, source_width),
                available_height / max(1, source_height),
            )
            display_width = max(1, int(round(source_width * scale)))
            display_height = max(1, int(round(source_height * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            display_source = cv2.resize(
                source,
                (display_width, display_height),
                interpolation=interpolation,
            )
            offset_x = max(0, (left_canvas.winfo_width() - display_width) // 2)
            offset_y = max(0, (left_canvas.winfo_height() - display_height) // 2)
            state.update(
                {
                    "display_width": display_width,
                    "display_height": display_height,
                    "offset_x": float(offset_x),
                    "offset_y": float(offset_y),
                    "source_photo": _image_to_tk_photo(display_source),
                }
            )
            left_canvas.delete("all")
            left_canvas.create_image(
                offset_x,
                offset_y,
                anchor=tk.NW,
                image=state["source_photo"],
                tags="dewarp-source",
            )
            draw_overlay()
            render_corrected()

        def schedule_resize(_event=None) -> None:
            if self.dewarp_resize_job is not None:
                window.after_cancel(self.dewarp_resize_job)
            self.dewarp_resize_job = window.after(
                REVIEW_RESIZE_DEBOUNCE_MS,
                render_views,
            )

        def nearest_point(x_pos: float, y_pos: float) -> tuple[int, int] | None:
            display_width = int(state["display_width"])
            display_height = int(state["display_height"])
            offset_x = float(state["offset_x"])
            offset_y = float(state["offset_y"])
            best_hit = None
            best_distance = 16.0
            for curve_index, curve in enumerate(state["curves"]):
                guide_anchor = float(curve["anchor"])
                for point_index, (x_value, displacement) in enumerate(curve["points"]):
                    px = offset_x + x_value * (display_width - 1)
                    py = offset_y + (guide_anchor + displacement) * display_height
                    distance = float(np.hypot(x_pos - px, y_pos - py))
                    if distance < best_distance:
                        best_distance = distance
                        best_hit = (curve_index, point_index)
            return best_hit

        def event_values(x_pos: float, y_pos: float) -> tuple[float, float]:
            local_x = float(x_pos) - float(state["offset_x"])
            local_y = float(y_pos) - float(state["offset_y"])
            x_value = local_x / max(1, int(state["display_width"]) - 1)
            displacement = local_y / max(1, int(state["display_height"])) - float(
                state["guide_anchor"]
            )
            return (
                float(np.clip(x_value, 0.0, 1.0)),
                float(np.clip(displacement, -0.24, 0.24)),
            )

        def nearest_curve(x_pos: float, y_pos: float) -> int | None:
            local_x = float(x_pos) - float(state["offset_x"])
            x_value = float(np.clip(local_x / max(1, int(state["display_width"]) - 1), 0.0, 1.0))
            best_curve = None
            best_distance = 12.0
            for curve_index, curve in enumerate(state["curves"]):
                curve_y = float(
                    interpolate_control_curve(
                        curve["points"],
                        np.asarray([x_value], dtype=np.float32),
                    )[0]
                )
                expected_y = float(state["offset_y"]) + (float(curve["anchor"]) + curve_y) * int(
                    state["display_height"]
                )
                distance = abs(float(y_pos) - expected_y)
                if distance < best_distance:
                    best_distance = distance
                    best_curve = curve_index
            return best_curve

        def show_magnifier(x_pos: float, y_pos: float) -> None:
            x_value, displacement = event_values(x_pos, y_pos)
            _show_canvas_magnifier(
                left_canvas,
                source,
                x_value * (source_width - 1),
                (float(state["guide_anchor"]) + displacement) * source_height,
                x_pos,
                y_pos,
                state,
            )

        def add_point_at(x_pos: float, y_pos: float) -> None:
            curve_hit = nearest_curve(x_pos, y_pos)
            if curve_hit is not None and curve_hit != state["active_curve"]:
                select_curve(curve_hit)
            x_value, displacement = event_values(x_pos, y_pos)
            added = _add_dewarp_control_point(state["points"], x_value, displacement)
            state["add_mode"] = False
            if added is None:
                status.set("Point was not added: choose a free position on the curve.")
                return
            state["selected"] = added
            draw_overlay()
            render_corrected()
            status.set(f"Added point {added + 1}; {len(state['points'])} points total.")

        def on_down(event) -> None:
            if state["add_mode"]:
                add_point_at(event.x, event.y)
                return
            point_hit = nearest_point(event.x, event.y)
            if point_hit is not None:
                curve_index, point_index = point_hit
                if curve_index != state["active_curve"]:
                    select_curve(curve_index)
                state["active"] = point_index
                state["selected"] = point_index
                state["drag_mode"] = "point"
            else:
                curve_hit = nearest_curve(event.x, event.y)
                if curve_hit is not None and curve_hit != state["active_curve"]:
                    select_curve(curve_hit)
                state["active"] = None
                state["selected"] = None
            if point_hit is None and curve_hit is not None:
                state["drag_mode"] = "line"
                state["drag_start_y"] = float(event.y)
                state["drag_start_anchor"] = float(state["guide_anchor"])
            elif point_hit is None:
                state["drag_mode"] = None
            draw_overlay()
            if state["drag_mode"] is not None:
                show_magnifier(event.x, event.y)

        def on_move(event) -> None:
            if state["drag_mode"] == "point" and state["active"] is not None:
                x_value, displacement = event_values(event.x, event.y)
                _move_dewarp_control_point(
                    state["points"],
                    int(state["active"]),
                    x_value,
                    displacement,
                )
            elif state["drag_mode"] == "line":
                delta = (float(event.y) - float(state["drag_start_y"])) / max(
                    1, int(state["display_height"])
                )
                curve_index = int(state["active_curve"])
                curves = state["curves"]
                lower_anchor = (
                    float(curves[curve_index - 1]["anchor"]) + 0.02 if curve_index > 0 else 0.0
                )
                upper_anchor = (
                    float(curves[curve_index + 1]["anchor"]) - 0.02
                    if curve_index + 1 < len(curves)
                    else 1.0
                )
                state["guide_anchor"] = _move_dewarp_guide_anchor(
                    state["points"],
                    float(state["drag_start_anchor"]),
                    delta,
                    lower_anchor=lower_anchor,
                    upper_anchor=upper_anchor,
                )
                selected_curve()["anchor"] = state["guide_anchor"]
            else:
                return
            draw_overlay()
            show_magnifier(event.x, event.y)

        def on_up(_event) -> None:
            if state["drag_mode"] is None:
                return
            completed_mode = state["drag_mode"]
            state["active"] = None
            state["drag_mode"] = None
            _hide_canvas_magnifier(left_canvas, state)
            if completed_mode == "point":
                render_corrected()
                status.set("User-adjusted preview. Apply to save these points for the page.")
            else:
                render_corrected()
                status.set("Curve region moved; vertical interpolation preview updated.")

        def begin_add_point() -> None:
            state["add_mode"] = True
            status.set("Click the source curve where the new point should be placed.")

        def remove_selected_point() -> None:
            selected = state["selected"]
            if selected is None:
                status.set("Select a point to remove.")
                return
            if not _remove_dewarp_control_point(state["points"], int(selected)):
                status.set("At least three wave points must remain.")
                return
            state["selected"] = None
            draw_overlay()
            render_corrected()
            status.set(f"Removed point; {len(state['points'])} points remain.")

        def remove_point_at(event) -> str:
            point_hit = nearest_point(event.x, event.y)
            if point_hit is not None:
                curve_index, point_index = point_hit
                if curve_index != state["active_curve"]:
                    select_curve(curve_index)
                state["selected"] = point_index
                remove_selected_point()
            return "break"

        def use_automatic() -> None:
            model, diagnostics = estimate_textline_dewarp_model(source)
            if model is None:
                status.set(f"Automatic model unavailable: {diagnostics.reason}.")
                return
            for curve in state["curves"]:
                curve["points"] = list(model.control_points)
            select_curve(int(state["active_curve"]))
            state["selected"] = None
            draw_overlay()
            render_corrected()
            status.set(f"Automatic model restored from {diagnostics.line_count} supporting lines.")

        def use_neutral() -> None:
            neutral = [(float(x), 0.0) for x in np.linspace(0.0, 1.0, 9, dtype=np.float32)]
            for curve in state["curves"]:
                curve["points"] = list(neutral)
            select_curve(int(state["active_curve"]))
            state["selected"] = None
            draw_overlay()
            render_corrected()
            status.set("Neutral correction model. The page will not be locally warped.")

        def apply_points() -> None:
            try:
                entry.set_dewarp_control_curves(
                    [(float(curve["anchor"]), curve["points"]) for curve in state["curves"]]
                )
                method_label = next(
                    (
                        label
                        for label, method in DEWARP_UI_METHODS.items()
                        if method == committed_dewarp_method
                    ),
                    self.dewarp_method_var.get(),
                )
                self.dewarp_method_var.set(method_label)
                diagnostics = self._commit_processing_request(entry, adjusted_request())
                self.refresh_page_list(keep_entry_ids=(entry.entry_id,))
                self._set_status(
                    f"Saved 3 dewarp curves; "
                    f"max correction {diagnostics.max_displacement_px:.1f}px."
                )
                self.geometry_summary_var.set(
                    f"Wave applied: 3 regional curves, {diagnostics.max_displacement_px:.1f}px"
                )
                close_editor()
            except Exception as exc:
                messagebox.showerror("Dewarp Control Points", str(exc))

        def close_editor() -> None:
            if self.dewarp_resize_job is not None:
                window.after_cancel(self.dewarp_resize_job)
                self.dewarp_resize_job = None
            self.dewarp_editor_window = None
            self.dewarp_source_canvas = None
            self.dewarp_preview_canvas = None
            self.dewarp_editor_state = None
            self.dewarp_status_var = None
            self.dewarp_apply_points_button = None
            self.inline_editor_close_callback = None
            window.destroy()
            self._hide_inline_geometry_editor()

        left_canvas.bind("<Button-1>", on_down)
        left_canvas.bind("<B1-Motion>", on_move)
        left_canvas.bind("<ButtonRelease-1>", on_up)
        left_canvas.bind("<Double-Button-1>", lambda event: add_point_at(event.x, event.y))
        left_canvas.bind("<Button-3>", remove_point_at)
        panes.bind("<Configure>", schedule_resize, add="+")

        ctk.CTkLabel(window, textvariable=status, anchor="w").pack(fill=ctk.X, padx=16, pady=(0, 8))
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(fill=ctk.X, padx=16, pady=(0, 14))
        ctk.CTkButton(actions, text="Use automatic", command=use_automatic).pack(side=ctk.LEFT)
        ctk.CTkButton(
            actions,
            text="Neutral curve",
            fg_color="transparent",
            border_width=1,
            command=use_neutral,
        ).pack(side=ctk.LEFT, padx=8)
        self.dewarp_add_point_button = ctk.CTkButton(
            actions,
            text="Add point",
            fg_color="transparent",
            border_width=1,
            command=begin_add_point,
        )
        self.dewarp_add_point_button.pack(side=ctk.LEFT)
        self.dewarp_remove_point_button = ctk.CTkButton(
            actions,
            text="Remove point",
            fg_color="transparent",
            border_width=1,
            command=remove_selected_point,
        )
        self.dewarp_remove_point_button.pack(side=ctk.LEFT, padx=8)
        self.dewarp_apply_points_button = ctk.CTkButton(
            actions,
            text="Apply points",
            command=apply_points,
        )
        self.dewarp_apply_points_button.pack(side=ctk.RIGHT)
        self.dewarp_close_button = ctk.CTkButton(
            actions,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=close_editor,
        )
        self.dewarp_close_button.pack(side=ctk.RIGHT, padx=8)
        self.inline_editor_close_callback = close_editor
        window.after_idle(render_views)

    def _on_review_processing_slider_change(self, _value: float) -> None:
        self.update_page_preview()

    def _snapshot_apply_pages(self, target_entries):
        snapshot_dir = tempfile.TemporaryDirectory(prefix="uniscan_gui_apply_")
        snapshots: list[_ApplyPageSnapshot] = []
        try:
            root = Path(snapshot_dir.name)
            for position, (_index, entry) in enumerate(target_entries, start=1):
                source_path = root / f"{position:06d}-source.png"
                current_path = root / f"{position:06d}-previous.png"
                entry.store.snapshot_image(entry.original_path, source_path)
                entry.store.snapshot_image(entry.current_path, current_path)
                snapshots.append(
                    _ApplyPageSnapshot(
                        entry_id=entry.entry_id,
                        name=entry.name,
                        source_path=source_path,
                        previous_current_path=current_path,
                        revision=entry.revision,
                        request=self._processing_request(entry=entry, preview=False),
                        previous_committed=entry.committed_processing,
                    )
                )
        except Exception:
            snapshot_dir.cleanup()
            raise
        return snapshot_dir, snapshots

    @staticmethod
    def _stage_apply_pages(snapshots, *, emit, is_cancelled):
        staged: list[_StagedAppliedPage] = []
        total = len(snapshots)
        for position, snapshot in enumerate(snapshots, start=1):
            if is_cancelled():
                raise RuntimeError("Cancelled by user.")
            emit(
                stage="Apply preview",
                current=f"{position}/{total}: {snapshot.name}",
                progress=int(((position - 1) / max(1, total)) * 100),
            )
            source = imread_unicode(snapshot.source_path)
            if source is None:
                raise RuntimeError(f"Cannot read processing snapshot: {snapshot.name}")
            request = replace(snapshot.request, cancel_cb=is_cancelled)
            result = process_document_page(source, request)
            committed = CommittedPageProcessing.from_result(
                request,
                result.diagnostics,
                result.image,
            )
            result_path = snapshot.source_path.with_name(f"{snapshot.source_path.stem}-result.png")
            if not imwrite_unicode(result_path, result.image):
                raise RuntimeError(f"Cannot stage processed page: {snapshot.name}")
            if is_cancelled():
                raise RuntimeError("Cancelled by user.")
            staged.append(
                _StagedAppliedPage(
                    entry_id=snapshot.entry_id,
                    result_path=result_path,
                    committed=committed,
                    cache_hits=result.diagnostics.cache_hits,
                )
            )
        return staged

    def _commit_staged_apply(self, snapshots, staged) -> int:
        entries_by_id = {entry.entry_id: entry for entry in self.session.entries}
        staged_by_id = {page.entry_id: page for page in staged}
        if len(staged_by_id) != len(snapshots):
            raise RuntimeError("Processed page set is incomplete; no pages were changed.")
        for snapshot in snapshots:
            entry = entries_by_id.get(snapshot.entry_id)
            if entry is None or entry.revision != snapshot.revision:
                raise RuntimeError(
                    f"Page changed while processing: {snapshot.name}. No pages were changed."
                )
            page = staged_by_id[snapshot.entry_id]
            if not page.result_path.is_file() or page.result_path.stat().st_size == 0:
                raise RuntimeError(f"Processed page is missing: {snapshot.name}")

        committed_snapshots: list[_ApplyPageSnapshot] = []
        try:
            for snapshot in snapshots:
                entry = entries_by_id[snapshot.entry_id]
                image = imread_unicode(staged_by_id[snapshot.entry_id].result_path)
                if image is None:
                    raise RuntimeError(f"Processed page is unreadable: {snapshot.name}")
                entry.current_image = image
                entry.committed_processing = staged_by_id[snapshot.entry_id].committed
                committed_snapshots.append(snapshot)
        except Exception as exc:
            rollback_error: Exception | None = None
            for snapshot in reversed(committed_snapshots):
                try:
                    entry = entries_by_id[snapshot.entry_id]
                    previous = imread_unicode(snapshot.previous_current_path)
                    if previous is None:
                        raise RuntimeError(f"Rollback snapshot is unreadable: {snapshot.name}")
                    entry.current_image = previous
                    entry.committed_processing = snapshot.previous_committed
                    entry.revision = snapshot.revision
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
            if rollback_error is not None:
                raise RuntimeError(
                    f"Apply failed and rollback was incomplete: {rollback_error}"
                ) from exc
            raise
        return sum(len(page.cache_hits) for page in staged)

    def open_review_processing_dialog(self) -> None:
        if (
            self.review_processing_window is not None
            and self.review_processing_window.winfo_exists()
        ):
            self.review_processing_window.lift()
            return
        if self.inline_editor_close_callback is not None:
            self.inline_editor_close_callback()

        window = self._show_inline_geometry_editor()
        self.review_processing_window = window

        ctk.CTkLabel(
            window,
            text="Advanced processing",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(
            anchor="w",
            padx=12,
            pady=(12, 8),
        )

        body = ctk.CTkFrame(window)
        body.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(0, 10))
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text="Contrast").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ctk.CTkSlider(
            body,
            from_=0.7,
            to=2.0,
            number_of_steps=26,
            variable=self.preprocess_contrast_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))

        ctk.CTkLabel(body, text="Brightness").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkSlider(
            body,
            from_=-80,
            to=80,
            number_of_steps=160,
            variable=self.preprocess_brightness_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="Denoise").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkSlider(
            body,
            from_=0,
            to=20,
            number_of_steps=20,
            variable=self.preprocess_denoise_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="B/W Threshold").grid(
            row=3, column=0, sticky="w", padx=8, pady=(4, 8)
        )
        ctk.CTkSlider(
            body,
            from_=80,
            to=240,
            number_of_steps=160,
            variable=self.preprocess_threshold_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=(4, 8))

        ctk.CTkLabel(body, text="Even page lighting").grid(
            row=4, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        ctk.CTkOptionMenu(
            body,
            values=list(SHADOW_UI_METHODS),
            variable=self.shadow_method_var,
            command=lambda _value: self.update_page_preview(),
        ).grid(row=4, column=1, sticky="ew", padx=8, pady=(0, 8))

        ctk.CTkLabel(body, text="Adaptive window").grid(row=5, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkSlider(
            body,
            from_=15,
            to=81,
            number_of_steps=33,
            variable=self.binarization_window_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=5, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="Sauvola/Wolf k").grid(row=6, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkSlider(
            body,
            from_=0.05,
            to=0.8,
            number_of_steps=15,
            variable=self.binarization_k_var,
            command=self._on_binarization_k_change,
        ).grid(row=6, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="Page margin (mm)").grid(
            row=7, column=0, sticky="w", padx=8, pady=4
        )
        ctk.CTkSlider(
            body,
            from_=0,
            to=30,
            number_of_steps=30,
            variable=self.page_margin_mm_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=7, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="Horizontal alignment").grid(
            row=8, column=0, sticky="w", padx=8, pady=4
        )
        ctk.CTkOptionMenu(
            body,
            values=["left", "center", "right"],
            variable=self.page_align_x_var,
            command=lambda _value: self.update_page_preview(),
        ).grid(row=8, column=1, sticky="ew", padx=8, pady=4)

        ctk.CTkLabel(body, text="Vertical alignment").grid(
            row=9, column=0, sticky="w", padx=8, pady=(4, 8)
        )
        ctk.CTkOptionMenu(
            body,
            values=["top", "center", "bottom"],
            variable=self.page_align_y_var,
            command=lambda _value: self.update_page_preview(),
        ).grid(row=9, column=1, sticky="ew", padx=8, pady=(4, 8))

        def _on_close() -> None:
            self.review_processing_window = None
            self.inline_editor_close_callback = None
            window.destroy()
            self._hide_inline_geometry_editor()

        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.pack(fill=ctk.X, padx=12, pady=(0, 12))
        ctk.CTkButton(
            actions,
            text="Use Preset Values",
            command=lambda: self.on_preprocess_preset_change(self.preprocess_preset_var.get()),
            width=140,
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(
            actions,
            text="Clear cache",
            command=self.clear_processing_cache,
            fg_color="transparent",
            border_width=1,
            width=100,
        ).pack(side=ctk.LEFT, padx=(8, 0))
        self.review_processing_close_button = ctk.CTkButton(
            actions,
            text="Close",
            command=_on_close,
            width=100,
        )
        self.review_processing_close_button.pack(side=ctk.LEFT, padx=8)
        self.inline_editor_close_callback = _on_close

    def apply_review_changes(self) -> None:
        indices = self._selected_entry_indices()
        if self.apply_changes_to_all_var.get():
            target_entries = list(enumerate(self.session.entries))
            if not target_entries:
                self._set_status("No pages available to process.")
                return
        else:
            if not indices:
                self._set_status("Select page(s) to apply processing.")
                return
            target_entries = [(idx, self.session.entries[idx]) for idx in indices]

        try:
            snapshot_dir, snapshots = self._snapshot_apply_pages(target_entries)
        except Exception as exc:
            messagebox.showerror("Postprocess Error", str(exc))
            return

        scope = "all pages" if self.apply_changes_to_all_var.get() else "selected pages"
        keep_index = target_entries[-1][0]
        selected_entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)

        def worker(emit, is_cancelled):
            return self._stage_apply_pages(
                snapshots,
                emit=emit,
                is_cancelled=is_cancelled,
            )

        def on_done(staged):
            try:
                cache_hit_count = self._commit_staged_apply(snapshots, staged)
                if selected_entry_ids:
                    self.refresh_page_list(keep_entry_ids=selected_entry_ids)
                else:
                    self.refresh_page_list(keep_index=keep_index)
                cache_note = f" Stage cache hits: {cache_hit_count}." if cache_hit_count else ""
                self._set_status(f"Reprocessed {len(target_entries)} {scope}.{cache_note}")
            finally:
                snapshot_dir.cleanup()

        if not self._start_background_job(
            "Apply preview",
            worker,
            on_done,
            on_error=snapshot_dir.cleanup,
        ):
            snapshot_dir.cleanup()

    def _entries_for_export(self):
        if self.export_scope_var.get() == "Selected pages":
            self._sync_page_selection_to_session()
            entries = self.session.selected_entries()
        else:
            entries = self.session.entries
        return entries

    def _snapshot_entries_for_export(self, entries):
        """Freeze each page's committed full-resolution result for export."""
        snapshot_dir = tempfile.TemporaryDirectory(prefix="uniscan_export_snapshot_")
        snapshots: list[_ExportPageSnapshot] = []
        try:
            root = Path(snapshot_dir.name)
            for index, entry in enumerate(entries, start=1):
                destination = root / f"{index:05d}.png"
                entry.store.snapshot_image(entry.current_path, destination)
                snapshots.append(
                    _ExportPageSnapshot(
                        name=str(entry.name),
                        current_path=destination,
                    )
                )
        except Exception:
            snapshot_dir.cleanup()
            raise
        return snapshot_dir, snapshots

    @staticmethod
    def _validate_pdf_layout_dpi(entries, export_dpi: int) -> None:
        mismatches: list[tuple[str, int]] = []
        for entry in entries:
            committed = entry.committed_processing
            if committed is None:
                continue
            recipe = committed.recipe
            if recipe.page_layout in {"a4", "letter"} and recipe.page_dpi != export_dpi:
                mismatches.append((entry.name, recipe.page_dpi))
        if not mismatches:
            return
        examples = ", ".join(f"{name} ({dpi} DPI)" for name, dpi in mismatches[:3])
        if len(mismatches) > 3:
            examples += f", and {len(mismatches) - 3} more"
        raise RuntimeError(
            f"PDF DPI {export_dpi} would change the physical A4/Letter size of committed "
            f"page(s): {examples}. Set PDF DPI to the committed value, or set the desired "
            "DPI and apply the preview to those pages again."
        )

    @staticmethod
    def _render_export_paths(
        snapshots: list[_ExportPageSnapshot],
        *,
        stage_dir: Path,
        emit,
        is_cancelled,
        job_name: str,
    ) -> list[Path]:
        """Validate and stage committed full-resolution pixels for export."""
        stage_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        total = len(snapshots)
        for index, snapshot in enumerate(snapshots, start=1):
            if is_cancelled():
                raise RuntimeError("Cancelled by user.")
            emit(
                stage=f"{job_name} staging",
                current=f"{index}/{total}: {snapshot.name}",
                progress=int(((index - 1) / max(1, total)) * 75),
            )
            path = stage_dir / f"{index:05d}.png"
            if imread_unicode(snapshot.current_path) is None:
                raise RuntimeError(f"Cannot read committed page: {snapshot.name}")
            try:
                shutil.copy2(snapshot.current_path, path)
            except OSError as exc:
                raise RuntimeError(f"Cannot stage committed page: {snapshot.name}") from exc
            paths.append(path)
        if is_cancelled():
            raise RuntimeError("Cancelled by user.")
        return paths

    def export_to_pdf(self) -> None:
        try:
            entries = self._entries_for_export()
            if not entries:
                raise RuntimeError("No pages available for export.")
            path_raw = self.export_pdf_path_var.get().strip()
            if not path_raw:
                chosen = filedialog.asksaveasfilename(
                    title="Save merged PDF as",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                )
                if not chosen:
                    return
                path_raw = chosen
                self.export_pdf_path_var.set(chosen)
            dpi = int(self.export_pdf_dpi_var.get())
            if dpi < 72:
                raise RuntimeError("PDF DPI must be >= 72.")
            entries = list(entries)
            self._validate_pdf_layout_dpi(entries, dpi)
            snapshot_dir, snapshots = self._snapshot_entries_for_export(entries)

            def worker(emit, is_cancelled):
                try:
                    with tempfile.TemporaryDirectory(prefix="uniscan_gui_export_") as raw_stage:
                        image_paths = self._render_export_paths(
                            snapshots,
                            stage_dir=Path(raw_stage),
                            emit=emit,
                            is_cancelled=is_cancelled,
                            job_name="Export PDF",
                        )
                        emit(
                            stage="Export PDF",
                            current=f"Writing {len(image_paths)} page(s)",
                            progress=80,
                        )
                        out_path = export_image_paths_as_pdf(
                            image_paths,
                            out_pdf=Path(path_raw),
                            dpi=dpi,
                            cancel_cb=is_cancelled,
                        )
                        emit(stage="Export PDF", current="Finalizing", progress=100)
                        return out_path
                finally:
                    snapshot_dir.cleanup()

            def on_done(out_path):
                self._set_status(f"Exported {len(entries)} page(s) to PDF: {out_path}")

            if not self._start_background_job("Export PDF", worker, on_done):
                snapshot_dir.cleanup()
        except Exception as exc:
            messagebox.showerror("Export PDF Error", str(exc))
            self._set_status("PDF export failed")

    def export_to_files(self) -> None:
        try:
            entries = self._entries_for_export()
            if not entries:
                raise RuntimeError("No pages available for export.")
            path_raw = self.export_dir_var.get().strip()
            if not path_raw:
                chosen = filedialog.askdirectory(title="Select output directory")
                if not chosen:
                    return
                path_raw = chosen
                self.export_dir_var.set(chosen)

            fmt = self.export_format_var.get()
            entries = list(entries)
            snapshot_dir, snapshots = self._snapshot_entries_for_export(entries)

            def worker(emit, is_cancelled):
                try:
                    with tempfile.TemporaryDirectory(prefix="uniscan_gui_export_") as raw_stage:
                        image_paths = self._render_export_paths(
                            snapshots,
                            stage_dir=Path(raw_stage),
                            emit=emit,
                            is_cancelled=is_cancelled,
                            job_name="Export files",
                        )
                        emit(
                            stage="Export files",
                            current=f"Writing {len(image_paths)} page(s)",
                            progress=80,
                        )
                        out_paths = export_image_paths_as_files(
                            image_paths,
                            output_dir=Path(path_raw),
                            ext=fmt,
                            base_name="page",
                            cancel_cb=is_cancelled,
                        )
                        emit(stage="Export files", current="Finalizing", progress=100)
                        return out_paths
                finally:
                    snapshot_dir.cleanup()

            def on_done(out_paths):
                self._set_status(f"Exported {len(out_paths)} file(s) to: {Path(path_raw)}")

            if not self._start_background_job("Export files", worker, on_done):
                snapshot_dir.cleanup()
        except Exception as exc:
            messagebox.showerror("Export Files Error", str(exc))
            self._set_status("Files export failed")


def run_app() -> int:
    try:
        app = UnifiedScanApp()
    except (SessionInUseError, UnsafeSessionLockError, OSError) as exc:
        print(f"UniScan startup failed: {exc}", file=sys.stderr)
        return 2
    app.mainloop()
    return 0
