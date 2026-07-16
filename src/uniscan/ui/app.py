"""Unified application shell."""

from __future__ import annotations

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
    DEWARP_METHOD_NONE,
    DEWARP_METHOD_TEXTLINE,
    DewarpModel,
    estimate_textline_dewarp_model,
)
from uniscan.core.pipeline import PageResult, PipelineOptions, process_loaded_items
from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.core.orientation import ORIENTATION_METHOD_AUTO, ORIENTATION_METHOD_NONE
from uniscan.core.preprocess import (
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
from uniscan.io.loaders import (
    IMG_EXTS,
    PDF_EXTS,
    imread_unicode,
    imwrite_unicode,
    iter_input_items,
    list_supported_in_folder,
)
from uniscan.session import (
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
from uniscan.ui.overlays import draw_quad_overlay

PREVIEW_WAIT_MS = 66
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
    "Text lines (offline)": DEWARP_METHOD_TEXTLINE,
}
DESKEW_UI_METHODS = {
    "Hybrid (recommended)": DESKEW_METHOD_HYBRID,
    "Text lines / Hough": DESKEW_METHOD_HOUGH,
    "Foreground box": DESKEW_METHOD_MIN_AREA,
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
    warped_path: Path
    contour: np.ndarray | None
    backend: str | None
    fallback_reason: str | None


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
    photo_refs: dict[str, object],
) -> None:
    """Draw a full-resolution crop opposite the active drag position."""
    canvas.delete("geometry-magnifier")
    canvas_width = max(1, canvas.winfo_width())
    canvas_height = max(1, canvas.winfo_height())
    lens_size = max(96, min(180, canvas_width // 3, canvas_height // 3))
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
    photo = _image_to_tk_photo(enlarged)
    margin = 12
    left = margin if canvas_x > canvas_width / 2 else canvas_width - lens_size - margin
    left = max(margin, left)
    top = margin
    canvas.create_rectangle(
        left - 3,
        top - 3,
        left + lens_size + 3,
        top + lens_size + 3,
        fill="#111111",
        outline="#ffffff",
        width=2,
        tags="geometry-magnifier",
    )
    canvas.create_image(
        left,
        top,
        image=photo,
        anchor=tk.NW,
        tags="geometry-magnifier",
    )
    center_x = left + lens_size / 2
    center_y = top + lens_size / 2
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


def _hide_canvas_magnifier(canvas: tk.Canvas, photo_refs: dict[str, object]) -> None:
    canvas.delete("geometry-magnifier")
    photo_refs["magnifier_photo"] = None


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
) -> float:
    if not points:
        return float(np.clip(anchor + delta, 0.0, 1.0))
    minimum = min(value for _x, value in points)
    maximum = max(value for _x, value in points)
    return float(np.clip(anchor + delta, -minimum, 1.0 - maximum))


def _detection_summary(results: list[PageResult]) -> str:
    """Describe detector outcomes without calling fallback pages detected."""
    fallback = sum(result.fallback_reason is not None for result in results)
    return _detection_summary_counts(len(results), fallback)


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
        self.camera_window: ctk.CTkToplevel | None = None
        self.inline_editor_host: ctk.CTkFrame | None = None
        self.inline_editor_close_callback = None
        self.corner_editor_window: ctk.CTkFrame | None = None
        self.corner_source_canvas: tk.Canvas | None = None
        self.corner_preview_canvas: tk.Canvas | None = None
        self.corner_meta_var: tk.StringVar | None = None
        self.corner_prev_button: ctk.CTkButton | None = None
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
        self.preprocess_illumination_var = tk.BooleanVar(value=False)
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
        self.dewarp_method_var = tk.StringVar(value="Automatic (validated)")
        self.geometry_summary_var = tk.StringVar(value="Wave preview: pending")
        self.split_preview_var = tk.StringVar(value="Split: not previewed")
        self.deskew_method_var = tk.StringVar(value="Hybrid (recommended)")
        self.import_pdf_dpi_var = tk.IntVar(value=300)
        self.import_two_page_mode_var = tk.BooleanVar(value=False)
        self.import_selected_files: list[str] = []
        self.live_edge_var = tk.BooleanVar(value=True)
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
            command=self.open_camera_window,
        )
        self.toolbar_camera_button.pack(side=ctk.LEFT, padx=4, pady=8)
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

        self.tabs = ctk.CTkTabview(container)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(4, 8))

        self.pages_tab = self.tabs.add(self.tab_review_name)

        self._build_pages_tab(self.pages_tab)
        self.tabs.set(self.tab_review_name)

    def open_camera_window(self) -> None:
        if self.camera_window is not None:
            try:
                if self.camera_window.winfo_exists():
                    self.camera_window.lift()
                    return
            except tk.TclError:
                pass

        window = ctk.CTkToplevel(self)
        self.camera_window = window
        window.title("Camera")
        window.geometry("1100x760")
        window.minsize(900, 620)
        window.transient(self)
        body = ctk.CTkFrame(window)
        body.pack(fill=ctk.BOTH, expand=True)
        self._build_capture_tab(body)
        self._update_camera_health()

        def close_window() -> None:
            self.stop_preview()
            self.camera_window = None
            self.camera_health_label = None
            self.preview_label = None
            window.destroy()

        self.camera_close_callback = close_window
        window.protocol("WM_DELETE_WINDOW", close_window)
        window.lift()

    def _build_capture_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkScrollableFrame(tab, width=340)
        controls.grid(row=0, column=0, sticky="ns", padx=(10, 8), pady=10)

        ctk.CTkLabel(controls, text="Camera index").pack(anchor="w", padx=10, pady=(10, 4))
        self.camera_index_entry = ctk.CTkEntry(
            controls, textvariable=self.camera_index_var, width=120
        )
        self.camera_index_entry.pack(anchor="w", padx=10)

        row_open = ctk.CTkFrame(controls, fg_color="transparent")
        row_open.pack(fill=ctk.X, padx=10, pady=(8, 2))
        ctk.CTkButton(row_open, text="Open", width=90, command=self.open_camera).pack(side=ctk.LEFT)
        ctk.CTkButton(row_open, text="Close", width=90, command=self.close_camera).pack(
            side=ctk.LEFT, padx=6
        )
        self.camera_health_label = ctk.CTkLabel(
            controls,
            textvariable=self.camera_health_var,
            text_color="#6c757d",
            anchor="w",
        )
        self.camera_health_label.pack(fill=ctk.X, padx=10, pady=(2, 6))

        ctk.CTkButton(controls, text="Configure Camera", command=self.configure_camera_event).pack(
            fill=ctk.X,
            padx=10,
            pady=(6, 8),
        )

        ctk.CTkLabel(
            controls,
            text="Capture adds pages to the workspace.\nProcessing stays non-destructive until applied.",
            justify="left",
            anchor="w",
        ).pack(fill=ctk.X, padx=10, pady=(0, 8))

        ctk.CTkLabel(controls, text="Burst shots").pack(anchor="w", padx=10, pady=(4, 2))
        self.shots_entry = ctk.CTkEntry(controls, textvariable=self.camera_shots_var, width=120)
        self.shots_entry.pack(anchor="w", padx=10)

        ctk.CTkLabel(controls, text="Delay (sec)").pack(anchor="w", padx=10, pady=(4, 2))
        self.delay_entry = ctk.CTkEntry(controls, textvariable=self.camera_delay_var, width=120)
        self.delay_entry.pack(anchor="w", padx=10, pady=(0, 8))

        row_preview = ctk.CTkFrame(controls, fg_color="transparent")
        row_preview.pack(fill=ctk.X, padx=10, pady=(6, 4))
        ctk.CTkButton(
            row_preview, text="Start Preview", width=120, command=self.start_preview
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(row_preview, text="Stop", width=70, command=self.stop_preview).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_capture = ctk.CTkFrame(controls, fg_color="transparent")
        row_capture.pack(fill=ctk.X, padx=10, pady=(4, 10))
        ctk.CTkButton(row_capture, text="Capture One", width=120, command=self.capture_one).pack(
            side=ctk.LEFT
        )
        ctk.CTkButton(
            row_capture, text="Capture Burst", width=120, command=self.capture_burst
        ).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(row_capture, text="Workspace", width=90, command=self.go_to_review_tab).pack(
            side=ctk.LEFT
        )

        live_edge_box = ctk.CTkFrame(controls)
        live_edge_box.pack(fill=ctk.X, padx=10, pady=(8, 4))
        ctk.CTkLabel(live_edge_box, text="Live edge detection").pack(
            anchor="w", padx=8, pady=(6, 2)
        )
        ctk.CTkCheckBox(
            live_edge_box,
            text="Show document boundaries",
            variable=self.live_edge_var,
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

        ctk.CTkLabel(
            controls,
            text="Captured pages are immediately available in Workspace.",
            justify="left",
            wraplength=250,
        ).pack(anchor="w", padx=10, pady=(0, 8))

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
        ctk.CTkButton(
            processing,
            text="Auto remove waves",
            fg_color="transparent",
            border_width=1,
            command=self.remove_waves_selected,
        ).pack(fill=ctk.X, padx=6, pady=(0, 10))
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
        ctk.CTkCheckBox(
            processing,
            text="Correct uneven lighting",
            variable=self.preprocess_illumination_var,
            command=self.update_page_preview,
        ).pack(anchor="w", padx=6, pady=(0, 10))

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
        self.stop_preview()
        self.job_cancel_event.set()
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
        if self.camera_window is not None and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        self.camera_window = None
        if self.camera is not None:
            self.camera.release()
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

    def _update_camera_health(self, error_text: str | None = None) -> None:
        state = camera_health_state(
            is_open=self.camera is not None,
            is_previewing=self.preview_job is not None,
            error_text=error_text,
        )
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
        if self.camera_window is not None:
            callback = getattr(self, "camera_close_callback", None)
            if callback is not None:
                callback()
        self.lift()

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
        best = RESOLUTIONS[0]
        match = re.match(r"^(\d+)x(\d+)$", best.strip())
        if match is None:
            return (3264, 2448)
        return (int(match.group(1)), int(match.group(2)))

    def _max_camera_resolution(self) -> tuple[int, int]:
        """Return the configured resolution (legacy method name retained)."""
        return getattr(self, "camera_resolution", self._default_camera_resolution())

    def _ensure_camera(self) -> CameraService:
        index = int(self.camera_index_var.get())
        resolution = self._max_camera_resolution()
        if self.camera is None:
            self.camera = CameraService(index=index, resolution=resolution)
            self.camera.open()
        elif self.camera.index != index:
            self.camera.release()
            self.camera = CameraService(index=index, resolution=resolution)
            self.camera.open()
        elif self.camera.resolution != resolution:
            self.camera.set_resolution(resolution)
        elif self.camera.read_frame() is None:
            self.camera.open()
        return self.camera

    def open_camera(self) -> None:
        try:
            self._ensure_camera()
            self._update_camera_health()
            self._set_status(f"Camera opened (index {self.camera_index_var.get()})")
        except Exception as exc:
            self._update_camera_health(error_text=str(exc))
            messagebox.showerror("Camera Error", str(exc))
            self._set_status("Camera open failed")

    def close_camera(self) -> None:
        self.stop_preview()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self._update_camera_health()
        self._set_status("Camera closed")

    def start_preview(self) -> None:
        try:
            self._ensure_camera()
        except Exception as exc:
            self._update_camera_health(error_text=str(exc))
            messagebox.showerror("Camera Error", str(exc))
            return
        self.live_detector.set_backend(self.live_backend_var.get())
        self.live_detector.start()
        if self.preview_job is None:
            self._preview_loop()
        self._update_camera_health()
        self._set_status("Preview started")
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
        if self.camera is None:
            self.preview_job = None
            self._update_camera_health()
            return
        frame = self.camera.read_frame()
        if frame is not None:
            if self.live_edge_var.get():
                self.live_detector.submit(frame)
            preview = self._preview_image_with_contour(frame)
            self._show_in_preview(preview)
        self.preview_job = self.after(PREVIEW_WAIT_MS, self._preview_loop)

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
            correct_illumination=bool(self.preprocess_illumination_var.get()),
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
        return PageProcessingRequest(
            dewarp_method=dewarp_method,
            dewarp_model=self._entry_dewarp_model(entry),
            dewarp_already_applied=self._entry_was_dewarped(entry, dewarp_method),
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
        if self.lightweight_preview_var.get():
            raw_image = entry.preview_raw_image
        else:
            raw_image = entry.raw_image

        contour = entry.detected_contour
        if contour is None:
            return raw_image

        # Contour is stored in the original raw coordinate space.
        # The preview may have been resized — scale the contour accordingly.
        full_raw = entry.raw_image if self.lightweight_preview_var.get() else raw_image
        from uniscan.ui.overlays import scale_contour as _scale_contour

        scaled = _scale_contour(
            contour,
            src_shape=full_raw.shape[:2],
            dst_shape=raw_image.shape[:2],
        )
        return draw_quad_overlay(raw_image, scaled)

    def _review_after_image(self, entry, before_image: np.ndarray) -> np.ndarray:
        # Process the durable full-resolution source with the exact request used
        # by export.  The UI downsizes only the finished pixels for display.
        del before_image
        return self._process_review_page(
            entry.original_image,
            entry=entry,
            preview=False,
        ).image

    def _show_in_preview(self, image: np.ndarray) -> None:
        photo = self._to_ctk_photo_for_label(image, self.preview_label)
        self.preview_label.configure(image=photo, text="")
        self.preview_photo = photo

    def _to_ctk_photo_for_label(self, image: np.ndarray, label: ctk.CTkLabel) -> ctk.CTkImage:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_width = max(200, label.winfo_width())
        max_height = max(120, label.winfo_height())
        h, w = rgb.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(resized)
        return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))

    def _process_capture_frame(
        self,
        frame: np.ndarray,
        base_name: str,
        *,
        two_page_mode: bool,
    ) -> list[PageResult]:
        options = PipelineOptions(
            detect_document=True,
            two_page_mode=bool(two_page_mode),
            postprocess_name="None",
        )
        return process_loaded_items([(base_name, frame)], options=options)

    def _ingest_page_results(self, results: list[PageResult]) -> None:
        for result in results:
            self.session.add_image_with_contour(
                name=result.name,
                raw_image=result.raw,
                warped_image=result.warped,
                contour=result.contour,
                backend=result.backend,
            )

    def _ingest_staged_import_pages(self, pages: list[_StagedImportPage]) -> None:
        """Publish a fully staged import to the session or roll it back logically."""
        added_entry_ids: list[str] = []
        try:
            for page in pages:
                raw = imread_unicode(page.raw_path)
                warped = imread_unicode(page.warped_path)
                if raw is None or warped is None:
                    raise RuntimeError(f"Cannot read staged imported page: {page.name}")
                entry = self.session.add_image_with_contour(
                    name=page.name,
                    raw_image=raw,
                    warped_image=warped,
                    contour=page.contour,
                    backend=page.backend,
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
            two_page_mode=False,
            postprocess_name="None",
        )
        results = process_loaded_items([(name, frame)], options=options)
        if not results:
            raise RuntimeError("Document detection returned no pages.")
        return results[0]

    def capture_one(self) -> None:
        try:
            camera = self._ensure_camera()
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("Could not capture an image from the camera.")
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
            results = self._process_capture_frame(
                frame,
                base_name=timestamp,
                two_page_mode=bool(self.import_two_page_mode_var.get()),
            )
            self._ingest_page_results(results)
            self.refresh_page_list(keep_index=len(self.session) - 1)
            self.go_to_review_tab()
            self._set_status(
                f"Captured {len(results)} page(s): {_detection_summary(results)}. "
                f"Session pages: {len(self.session)}"
            )
        except Exception as exc:
            messagebox.showerror("Capture Error", str(exc))
            self._set_status("Capture failed")

    def capture_burst(self) -> None:
        try:
            shots = int(self.camera_shots_var.get())
            delay_sec = float(self.camera_delay_var.get())
            index = int(self.camera_index_var.get())
            two_page_mode = bool(self.import_two_page_mode_var.get())
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")

            self.stop_preview()

            def worker(emit, is_cancelled):
                emit(stage="Burst capture", current=f"Opening camera {index}", progress=0)
                camera = CameraService(index=index, resolution=self._max_camera_resolution())
                camera.open()
                try:
                    frames = camera.capture_burst(
                        shots=shots,
                        delay_sec=delay_sec,
                        cancel_cb=is_cancelled,
                        on_progress=lambda i, total: emit(
                            stage="Burst capture",
                            current=f"Shot {i}/{total}",
                            progress=int((i / total) * 45),
                        ),
                    )
                finally:
                    camera.release()

                results: list[PageResult] = []
                total_frames = len(frames)
                for idx, frame in enumerate(frames, start=1):
                    if is_cancelled():
                        raise RuntimeError("Cancelled by user.")
                    current_results = self._process_capture_frame(
                        frame,
                        base_name=f"{timestamp}_{idx:03d}",
                        two_page_mode=two_page_mode,
                    )
                    results.extend(current_results)
                    emit(
                        stage="Processing burst",
                        current=f"Frame {idx}/{total_frames}",
                        progress=45 + int((idx / total_frames) * 55),
                    )
                return results

            def on_done(results):
                self._ingest_page_results(results)
                self.refresh_page_list(keep_index=len(self.session) - 1)
                self.go_to_review_tab()
                self._set_status(
                    f"Burst captured {len(results)} page(s): {_detection_summary(results)}. "
                    f"Session pages: {len(self.session)}"
                )

            self._start_background_job("Capture Burst", worker, on_done)
        except Exception as exc:
            messagebox.showerror("Burst Error", str(exc))
            self._set_status("Burst capture failed")

    def _set_camera_resolution(self, resolution: tuple[int, int]) -> None:
        """Apply a camera resolution and commit the preference only on success."""
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
            self.camera = candidate
        else:
            camera = self.camera
            try:
                camera.set_resolution(resolution)
            except Exception:
                try:
                    camera.set_resolution(previous_resolution)
                except Exception:
                    camera.release()
                    self.camera = None
                raise
        self.camera_resolution = resolution

    def configure_camera_event(self) -> None:
        def _set_camera_index(index_str: str) -> None:
            self.camera_index_var.set(int(index_str))
            if self.camera is not None:
                self.camera.set_index(int(index_str))
            self._set_status(f"Camera index set to {index_str}")

        def _identify_cameras() -> None:
            indices = CameraService.get_available_device_indices(max_indices=10)
            values = [str(i) for i in indices] if indices else [str(i) for i in range(10)]
            index_menu.configure(values=values)
            if values:
                index_menu.set(values[0])
                _set_camera_index(values[0])

        def _set_resolution(res_string: str) -> None:
            match = re.match(r"^(\d+)x(\d+)$", res_string.strip())
            if match is None:
                messagebox.showerror(
                    "Resolution Error", "Resolution must be on the form <width>x<height>."
                )
                return
            resolution = (int(match.group(1)), int(match.group(2)))
            try:
                self._set_camera_resolution(resolution)
            except Exception as exc:
                messagebox.showerror("Resolution Error", str(exc))
                self._set_status(f"Camera resolution failed: {exc}")
                return
            self._set_status(f"Camera resolution set to {resolution[0]}x{resolution[1]}")

        window = ctk.CTkToplevel(self)
        window.title("Camera Configuration")
        window.resizable(width=False, height=False)

        ctk.CTkLabel(window, text="Camera index").pack(anchor="w", padx=12, pady=(16, 4))
        index_values = [str(i) for i in range(10)]
        index_var = tk.StringVar(value=str(self.camera_index_var.get()))
        index_menu = ctk.CTkOptionMenu(
            window,
            values=index_values,
            variable=index_var,
            command=_set_camera_index,
        )
        index_menu.pack(fill=ctk.X, padx=12, pady=(0, 8))

        ctk.CTkButton(window, text="Identify cameras", command=_identify_cameras).pack(
            fill=ctk.X,
            padx=12,
            pady=(0, 12),
        )

        ctk.CTkLabel(window, text="Preset resolution").pack(anchor="w", padx=12, pady=(0, 4))
        preset_var = tk.StringVar(value=RESOLUTIONS[-1])
        preset_var.set(f"{self.camera_resolution[0]}x{self.camera_resolution[1]}")
        preset_menu = ctk.CTkOptionMenu(
            window,
            values=RESOLUTIONS,
            variable=preset_var,
            command=_set_resolution,
        )
        preset_menu.pack(fill=ctk.X, padx=12, pady=(0, 8))

        ctk.CTkLabel(window, text="Custom resolution").pack(anchor="w", padx=12, pady=(0, 4))
        custom_var = tk.StringVar(value=RESOLUTIONS[-1])
        custom_var.set(f"{self.camera_resolution[0]}x{self.camera_resolution[1]}")
        custom_entry = ctk.CTkEntry(window, textvariable=custom_var)
        custom_entry.pack(fill=ctk.X, padx=12, pady=(0, 8))
        ctk.CTkButton(
            window,
            text="Set custom resolution",
            command=lambda: _set_resolution(custom_var.get()),
        ).pack(fill=ctk.X, padx=12, pady=(0, 14))

        window.attributes("-topmost", True)
        window.grab_set()
        window.attributes("-topmost", False)

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
        self.export_scope_var.set("All pages")
        self.export_pdf_dpi_var.set(300)
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
                results = self._process_capture_frame(
                    image,
                    base_name="clipboard",
                    two_page_mode=bool(self.import_two_page_mode_var.get()),
                )
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
        self._set_status(f"Starting import for {len(paths)} file(s)...")
        staging_dir = tempfile.TemporaryDirectory(prefix="uniscan_gui_import_")

        def worker(emit, is_cancelled):
            emit(stage="Import", current=f"{len(paths)} input file(s)", progress=0)
            total_paths = len(paths)
            added_pages = 0
            fallback_pages = 0
            staged_pages: list[_StagedImportPage] = []
            options = PipelineOptions(
                detect_document=True,
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
                        warped_path = Path(staging_dir.name) / f"{page_index:06d}-warped.png"
                        if not imwrite_unicode(raw_path, result.raw) or not imwrite_unicode(
                            warped_path, result.warped
                        ):
                            raise RuntimeError(f"Cannot stage imported page: {result.name}")
                        if is_cancelled():
                            raise RuntimeError("Cancelled by user.")
                        staged_pages.append(
                            _StagedImportPage(
                                name=result.name,
                                raw_path=raw_path,
                                warped_path=warped_path,
                                contour=result.contour,
                                backend=result.backend,
                                fallback_reason=result.fallback_reason,
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
            tag = "" if entry.detected_contour is not None else "  ⚠"
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
        self.update_page_preview()

    def _sync_page_selection_to_session(self) -> None:
        selected = set(self.page_listbox.curselection())
        for idx, entry in enumerate(self.session.entries):
            entry.selected = idx in selected

    def on_page_select(self, _event=None) -> None:
        self._sync_page_selection_to_session()
        self._update_page_action_states()
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
            text="Processed preview · 2 output pages"
            if split_ratio is not None
            else "Processed preview"
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
                source = (
                    entry.preview_original_image.copy()
                    if fast_preview
                    else entry.original_image.copy()
                )
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
                original_image=result.warped,
                current_image=result.current,
                name=image_path.name,
                contour=result.contour,
                backend=result.backend,
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
            camera = self._ensure_camera()
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("Could not capture an image from the camera.")

            item_name = datetime.now().strftime(r"retake_%Y%m%d_%H%M%S")
            result = self._detect_single_page(frame, name=item_name)
            previous_committed = entry.committed_processing
            ok = self.session.replace_entry_image(
                entry.entry_id,
                raw_image=result.raw,
                original_image=result.warped,
                current_image=result.current,
                contour=result.contour,
                backend=result.backend,
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
                points = np.asarray(existing, dtype=np.float32).reshape(-1, 2).copy()
                points_by_entry[entry.entry_id] = points
                return points

            detected: np.ndarray | None = None
            if auto_detect:
                detected = self._detect_corner_points(source_image)
                if detected is not None:
                    dirty_entry_ids.add(entry.entry_id)
                if detected is None and existing is not None and not from_current_geometry:
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
            dirty_entry_ids.add(entry.entry_id)
            _redraw()
            _show_canvas_magnifier(
                canvas,
                view_state["source_image"],
                x,
                y,
                float(event.x),
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
            view_state["points"] = points
            dirty_entry_ids.add(entry.entry_id)
            _redraw()
            _render_corrected_preview()

        def _apply_entry(entry_index: int, entry, points: np.ndarray) -> None:
            source_image = source_images_by_entry[entry.entry_id]
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")
            warped = warp_perspective_from_points(source_image, points.astype(np.float32))
            if warped is None or warped.size == 0:
                raise RuntimeError("Perspective transform returned empty image.")
            previous_committed = entry.committed_processing
            entry.original_image = warped
            entry.detected_contour = points.astype(np.float32).reshape(-1, 2).copy()
            entry.detected_backend = "manual"
            self._reprocess_after_geometry_change(entry, previous_committed)

        def _save_current_if_dirty() -> bool:
            entry_index, entry = _current_entry()
            if entry.entry_id not in dirty_entry_ids:
                return True
            try:
                points = view_state["points"]
                _apply_entry(entry_index, entry, points)
                dirty_entry_ids.discard(entry.entry_id)
                self._set_status(f"Saved perspective points for {entry.name}.")
                return True
            except Exception as exc:
                messagebox.showerror("Auto Crop Error", str(exc))
                return False

        def _apply_all():
            try:
                for idx_offset, entry in enumerate(entries):
                    points = _init_points_for(entry)
                    _apply_entry(indices[idx_offset], entry, points)
                    dirty_entry_ids.discard(entry.entry_id)
                self.refresh_page_list(keep_entry_ids=selected_entry_ids)
                self._set_status(f"Applied crop to {len(entries)} page(s).")
            except Exception as exc:
                messagebox.showerror("Auto Crop Error", str(exc))

        def _prev_page():
            if state["index"] > 0 and _save_current_if_dirty():
                state["index"] -= 1
                _load_current_entry()

        def _next_page():
            if state["index"] < len(entries) - 1 and _save_current_if_dirty():
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
        ctk.CTkButton(controls, text="Reset", width=90, command=_reset).pack(side=ctk.LEFT, padx=6)
        if auto_detect:
            ctk.CTkButton(
                controls,
                text="Apply detected to all",
                width=150,
                command=_apply_all,
            ).pack(side=ctk.LEFT, padx=6)

        def _close_editor() -> None:
            if not _save_current_if_dirty():
                return
            if self.corner_resize_job is not None:
                win.after_cancel(self.corner_resize_job)
                self.corner_resize_job = None
            self.corner_editor_window = None
            self.corner_source_canvas = None
            self.corner_preview_canvas = None
            self.corner_meta_var = None
            self.corner_prev_button = None
            self.corner_next_button = None
            self.corner_editor_state = None
            self.inline_editor_close_callback = None
            win.destroy()
            self.refresh_page_list(keep_entry_ids=selected_entry_ids)
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

    def _reprocess_after_geometry_change(self, entry, previous_committed) -> None:
        """Replay durable appearance settings without adopting pending GUI controls."""
        if previous_committed is None:
            return
        previous_request = previous_committed.recipe.to_request()
        request = replace(
            previous_request,
            orientation_method=ORIENTATION_METHOD_NONE,
            deskew_method=DESKEW_METHOD_NONE,
            dewarp_method=DEWARP_METHOD_NONE,
            dewarp_model=None,
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
            self._reprocess_after_geometry_change(entry, previous_committed)
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
            self._reprocess_after_geometry_change(entry, previous_committed)
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
            return float(
                np.clip(local_x / max(1, int(state["display_width"]) - 1), 0.05, 0.95)
            )

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
        ctk.CTkLabel(window, textvariable=status, anchor="w").pack(
            fill=ctk.X, padx=16, pady=(0, 8)
        )
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
            backend=None,
        )
        right_entry = self.session.add_image_with_contour(
            name=f"{base_name} [R]",
            raw_image=right_raw,
            warped_image=right_warped,
            contour=None,
            backend=None,
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
        for idx in indices:
            entry = self.session.entries[idx]
            previous_committed = entry.committed_processing
            stage_result = process_document_page(
                entry.original_image,
                PageProcessingRequest(deskew_method=method),
            )
            entry.original_image = stage_result.image
            self._reprocess_after_geometry_change(entry, previous_committed)
            angles.append(stage_result.diagnostics.deskew_angle_degrees)

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
            stage_result = process_document_page(
                entry.original_image,
                PageProcessingRequest(orientation_method=ORIENTATION_METHOD_AUTO),
            )
            oriented = stage_result.image
            item = stage_result.diagnostics.orientation
            if item.applied:
                entry.original_image = oriented
                self._reprocess_after_geometry_change(entry, previous_committed)
            diagnostics.append(item)

        self.refresh_page_list(keep_entry_ids=entry_ids)
        applied = sum(item.applied for item in diagnostics)
        uncertain = sum(item.reason not in {None, "already_upright"} for item in diagnostics)
        self._set_status(
            f"Auto-oriented {applied}/{len(indices)} page(s); "
            f"{uncertain} left unchanged as uncertain."
        )

    def remove_waves_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to remove waves.")
            return

        self.dewarp_method_var.set("Automatic (validated)")
        diagnostics = []
        entry_ids = tuple(self.session.entries[idx].entry_id for idx in indices)
        for idx in indices:
            entry = self.session.entries[idx]
            previous_committed = entry.committed_processing
            entry.clear_dewarp_control_points()
            diagnostics.append(
                self._reprocess_with_dewarp(
                    entry,
                    previous_committed,
                    method=DEWARP_METHOD_AUTO,
                )
            )
        self.refresh_page_list(keep_entry_ids=entry_ids)
        applied = sum(item.applied for item in diagnostics)
        max_displacement = max((item.max_displacement_px for item in diagnostics), default=0.0)
        self._set_status(
            f"Removed page waves on {applied}/{len(indices)} page(s); "
            f"max correction {max_displacement:.1f}px."
        )
        self.geometry_summary_var.set(
            f"Wave applied: {applied}/{len(indices)} pages, {max_displacement:.1f}px max"
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

        source = entry.original_image
        source_height, source_width = source.shape[:2]

        if entry.dewarp_control_points is not None:
            initial_points = list(entry.dewarp_control_points)
            initial_message = "Loaded saved page correction."
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

        window = self._show_inline_geometry_editor()
        self.dewarp_editor_window = window

        ctk.CTkLabel(
            window,
            text="Page wave correction",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))

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

        state = {
            "points": initial_points,
            "active": None,
            "selected": None,
            "drag_mode": None,
            "drag_start_y": 0.0,
            "drag_start_anchor": 0.5,
            "guide_anchor": 0.5,
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

        def current_model(*, source_name: str = "user") -> DewarpModel:
            return DewarpModel(
                method=DEWARP_METHOD_TEXTLINE,
                control_points=tuple(state["points"]),
                source=source_name,
            )

        def draw_overlay() -> None:
            left_canvas.delete("dewarp-overlay")
            display_width = int(state["display_width"])
            display_height = int(state["display_height"])
            offset_x = float(state["offset_x"])
            offset_y = float(state["offset_y"])
            point_x = np.asarray([point[0] for point in state["points"]], dtype=np.float32)
            point_y = np.asarray([point[1] for point in state["points"]], dtype=np.float32)
            guide_x = np.linspace(0.0, 1.0, 160, dtype=np.float32)
            guide_y = np.interp(guide_x, point_x, point_y)
            guide_anchor = float(state["guide_anchor"])
            for guide_offset in (-0.25, 0.0, 0.25):
                coords: list[float] = []
                for x_value, displacement in zip(guide_x, guide_y):
                    coords.extend(
                        [
                            offset_x + float(x_value * (display_width - 1)),
                            offset_y
                            + float(
                                (guide_anchor + guide_offset + displacement) * display_height
                            ),
                        ]
                    )
                left_canvas.create_line(
                    *coords,
                    fill="#36a3ff",
                    width=3 if guide_offset == 0.0 else 1,
                    tags="dewarp-overlay",
                )
            for point_index, (x_value, displacement) in enumerate(state["points"]):
                x_pos = offset_x + x_value * (display_width - 1)
                y_pos = offset_y + (guide_anchor + displacement) * display_height
                selected = point_index == state["selected"]
                radius = 8 if selected else 6
                left_canvas.create_oval(
                    x_pos - radius,
                    y_pos - radius,
                    x_pos + radius,
                    y_pos + radius,
                    fill="#ffb000" if selected else "#1f6aa5",
                    outline="#ffffff",
                    width=2 if selected else 1,
                    tags=("dewarp-overlay", f"point-{point_index}"),
                )

        def render_corrected() -> None:
            state["last_corrected_source_shape"] = source.shape
            corrected = process_document_page(
                source,
                PageProcessingRequest(
                    dewarp_method=DEWARP_METHOD_TEXTLINE,
                    dewarp_model=current_model(),
                ),
            ).image
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

        def nearest_point(x_pos: float, y_pos: float) -> int | None:
            display_width = int(state["display_width"])
            display_height = int(state["display_height"])
            offset_x = float(state["offset_x"])
            offset_y = float(state["offset_y"])
            guide_anchor = float(state["guide_anchor"])
            best_index = None
            best_distance = 16.0
            for point_index, (x_value, displacement) in enumerate(state["points"]):
                px = offset_x + x_value * (display_width - 1)
                py = offset_y + (guide_anchor + displacement) * display_height
                distance = float(np.hypot(x_pos - px, y_pos - py))
                if distance < best_distance:
                    best_distance = distance
                    best_index = point_index
            return best_index

        def event_values(x_pos: float, y_pos: float) -> tuple[float, float]:
            local_x = float(x_pos) - float(state["offset_x"])
            local_y = float(y_pos) - float(state["offset_y"])
            x_value = local_x / max(1, int(state["display_width"]) - 1)
            displacement = (
                local_y / max(1, int(state["display_height"]))
                - float(state["guide_anchor"])
            )
            return (
                float(np.clip(x_value, 0.0, 1.0)),
                float(np.clip(displacement, -0.24, 0.24)),
            )

        def near_center_curve(x_pos: float, y_pos: float) -> bool:
            x_value, _displacement = event_values(x_pos, y_pos)
            point_x = np.asarray([point[0] for point in state["points"]], dtype=np.float32)
            point_y = np.asarray([point[1] for point in state["points"]], dtype=np.float32)
            curve_y = float(np.interp(x_value, point_x, point_y))
            expected_y = float(state["offset_y"]) + (
                float(state["guide_anchor"]) + curve_y
            ) * int(state["display_height"])
            return abs(float(y_pos) - expected_y) <= 12.0

        def show_magnifier(x_pos: float, y_pos: float) -> None:
            x_value, displacement = event_values(x_pos, y_pos)
            _show_canvas_magnifier(
                left_canvas,
                source,
                x_value * (source_width - 1),
                (float(state["guide_anchor"]) + displacement) * source_height,
                x_pos,
                state,
            )

        def add_point_at(x_pos: float, y_pos: float) -> None:
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
            active = nearest_point(event.x, event.y)
            state["active"] = active
            state["selected"] = active
            if active is not None:
                state["drag_mode"] = "point"
            elif near_center_curve(event.x, event.y):
                state["drag_mode"] = "line"
                state["drag_start_y"] = float(event.y)
                state["drag_start_anchor"] = float(state["guide_anchor"])
            else:
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
                state["guide_anchor"] = _move_dewarp_guide_anchor(
                    state["points"],
                    float(state["drag_start_anchor"]),
                    delta,
                )
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
                status.set(
                    "Working line moved for easier tracing; page correction is unchanged."
                )

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
            selected = nearest_point(event.x, event.y)
            if selected is not None:
                state["selected"] = selected
                remove_selected_point()
            return "break"

        def use_automatic() -> None:
            model, diagnostics = estimate_textline_dewarp_model(source)
            if model is None:
                status.set(f"Automatic model unavailable: {diagnostics.reason}.")
                return
            state["points"] = list(model.control_points)
            state["selected"] = None
            draw_overlay()
            render_corrected()
            status.set(f"Automatic model restored from {diagnostics.line_count} supporting lines.")

        def use_neutral() -> None:
            state["points"] = [(float(x), 0.0) for x in np.linspace(0.0, 1.0, 9, dtype=np.float32)]
            state["selected"] = None
            draw_overlay()
            render_corrected()
            status.set("Neutral correction model. The page will not be locally warped.")

        def apply_points() -> None:
            try:
                previous_committed = entry.committed_processing
                entry.set_dewarp_control_points(state["points"])
                self.dewarp_method_var.set("Text lines (offline)")
                diagnostics = self._reprocess_with_dewarp(
                    entry,
                    previous_committed,
                    method=DEWARP_METHOD_TEXTLINE,
                )
                self.refresh_page_list(keep_entry_ids=(entry.entry_id,))
                self._set_status(
                    f"Saved {len(entry.dewarp_control_points or ())} dewarp points; "
                    f"max correction {diagnostics.max_displacement_px:.1f}px."
                )
                self.geometry_summary_var.set(
                    f"Wave applied: user curve, {len(entry.dewarp_control_points or ())} points, "
                    f"{diagnostics.max_displacement_px:.1f}px"
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
        ctk.CTkButton(actions, text="Apply points", command=apply_points).pack(side=ctk.RIGHT)
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

        ctk.CTkCheckBox(
            body,
            text="Correct uneven lighting (experimental)",
            variable=self.preprocess_illumination_var,
            command=self.update_page_preview,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

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
