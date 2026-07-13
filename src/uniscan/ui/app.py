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
from uniscan.core.orientation import ORIENTATION_METHOD_AUTO
from uniscan.core.preprocess import (
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

PREVIEW_WAIT_MS = 25
REVIEW_PREVIEW_DEBOUNCE_MS = 120
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
        self.review_preview_job: str | None = None
        self.review_preview_thread: threading.Thread | None = None
        self.review_preview_threads: list[threading.Thread] = []
        self.review_preview_cancel_event = threading.Event()
        self.review_preview_generation = 0
        self.review_processing_window: ctk.CTkToplevel | None = None
        self.corner_editor_window: ctk.CTkToplevel | None = None
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
        self.dewarp_method_var = tk.StringVar(value="None")
        self.deskew_method_var = tk.StringVar(value="Hybrid (recommended)")
        self.import_folder_var = tk.StringVar()
        self.import_files_var = tk.StringVar()
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
        self.job_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.job_cancel_event = threading.Event()
        self.job_thread: threading.Thread | None = None
        self._closing = False
        self._close_wait_job: str | None = None
        self._close_deadline: float | None = None
        self.autosave_job: str | None = None
        self.tab_review_name = "Workspace"
        self.tab_scan_name = "Camera"
        self.tab_import_name = "Import options"
        self.tab_export_name = "Export options"

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
        ctk.CTkButton(
            toolbar,
            text="+ Add files",
            width=110,
            command=self.quick_add_files,
        ).pack(side=ctk.LEFT, padx=(8, 4), pady=8)
        ctk.CTkButton(
            toolbar,
            text="Add folder",
            width=105,
            fg_color="transparent",
            border_width=1,
            command=self.quick_add_folder,
        ).pack(side=ctk.LEFT, padx=4, pady=8)
        ctk.CTkButton(
            toolbar,
            text="Paste",
            width=80,
            fg_color="transparent",
            border_width=1,
            command=self.import_from_clipboard,
        ).pack(side=ctk.LEFT, padx=4, pady=8)
        ctk.CTkButton(
            toolbar,
            text="Camera",
            width=90,
            fg_color="transparent",
            border_width=1,
            command=lambda: self.tabs.set(self.tab_scan_name),
        ).pack(side=ctk.LEFT, padx=4, pady=8)
        self.toolbar_export_button = ctk.CTkButton(
            toolbar,
            text="Export PDF...",
            width=120,
            fg_color="#2f855a",
            hover_color="#276749",
            command=self.export_to_pdf,
        )
        self.toolbar_export_button.pack(side=ctk.RIGHT, padx=8, pady=8)

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
        self.capture_tab = self.tabs.add(self.tab_scan_name)
        self.import_tab = self.tabs.add(self.tab_import_name)
        self.export_tab = self.tabs.add(self.tab_export_name)

        self._build_pages_tab(self.pages_tab)
        self._build_capture_tab(self.capture_tab)
        self._build_import_tab(self.import_tab)
        self._build_export_tab(self.export_tab)
        self.tabs.set(self.tab_review_name)

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

        page_actions = ctk.CTkFrame(left, fg_color="transparent")
        page_actions.pack(fill=ctk.X, padx=10, pady=(0, 4))
        for text, command in (
            ("Up", self.move_selected_up),
            ("Down", self.move_selected_down),
            ("Delete", self.delete_selected_pages),
        ):
            ctk.CTkButton(page_actions, text=text, width=78, command=command).pack(
                side=ctk.LEFT, padx=(0, 4)
            )

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

        ctk.CTkButton(
            left,
            text="More page tools...",
            fg_color="transparent",
            border_width=1,
            command=self.open_page_tools_dialog,
        ).pack(fill=ctk.X, padx=10, pady=(2, 10))
        ctk.CTkLabel(
            left,
            text="Del delete  •  Ctrl+←/→ rotate\nAlt+↑/↓ move  •  Ctrl+A select all",
            justify="left",
            text_color=("#60646c", "#a0a4ab"),
            font=ctk.CTkFont(size=11),
        ).pack(fill=ctk.X, padx=10, pady=(0, 10))

        preview = ctk.CTkFrame(tab)
        preview.grid(row=0, column=1, sticky="nsew", padx=6, pady=10)
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
        ctk.CTkLabel(self.page_preview_before_frame, text="Original").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        self.page_preview_before_label = ctk.CTkLabel(
            self.page_preview_before_frame,
            text="No page selected",
        )
        self.page_preview_before_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.page_preview_after_frame = ctk.CTkFrame(preview)
        self.page_preview_after_frame.grid_rowconfigure(1, weight=1)
        self.page_preview_after_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.page_preview_after_frame, text="Processed preview").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        self.page_preview_after_label = ctk.CTkLabel(
            self.page_preview_after_frame,
            text="No page selected",
        )
        self.page_preview_after_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._layout_page_previews()

        processing = ctk.CTkScrollableFrame(tab, width=270, label_text="Processing")
        processing.grid(row=0, column=2, sticky="nsew", padx=(6, 10), pady=10)

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
            text="Advanced...",
            width=108,
            fg_color="transparent",
            border_width=1,
            command=self.open_review_processing_dialog,
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(
            processing,
            text="Apply processing",
            command=self.apply_review_changes,
        ).pack(fill=ctk.X, padx=6, pady=(0, 14))

        ctk.CTkLabel(
            processing,
            text="Export",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(fill=ctk.X, padx=6, pady=(2, 6))
        ctk.CTkOptionMenu(
            processing,
            values=["All pages", "Selected pages"],
            variable=self.export_scope_var,
        ).pack(fill=ctk.X, padx=6, pady=(0, 8))
        export_dpi = ctk.CTkFrame(processing, fg_color="transparent")
        export_dpi.pack(fill=ctk.X, padx=6, pady=(0, 8))
        ctk.CTkLabel(export_dpi, text="PDF DPI").pack(side=ctk.LEFT)
        ctk.CTkEntry(export_dpi, textvariable=self.export_pdf_dpi_var, width=80).pack(
            side=ctk.RIGHT
        )
        self.workspace_export_pdf_button = ctk.CTkButton(
            processing,
            text="Export PDF...",
            fg_color="#2f855a",
            hover_color="#276749",
            command=self.export_to_pdf,
        )
        self.workspace_export_pdf_button.pack(fill=ctk.X, padx=6, pady=(0, 6))
        self.workspace_export_files_button = ctk.CTkButton(
            processing,
            text="Export images...",
            fg_color="transparent",
            border_width=1,
            command=self.export_to_files,
        )
        self.workspace_export_files_button.pack(fill=ctk.X, padx=6, pady=(0, 10))
        self.refresh_page_list()

    def _bind_shortcuts(self) -> None:
        """Bind the common document actions without stealing text-entry shortcuts."""
        self.bind("<Control-o>", lambda _event: self._run_shortcut(self.quick_add_files))
        self.bind("<Control-Shift-O>", lambda _event: self._run_shortcut(self.quick_add_folder))
        self.bind("<Control-Shift-C>", lambda _event: self._run_shortcut(self.capture_one))
        self.bind("<Control-e>", lambda _event: self._run_shortcut(self.export_to_pdf))
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
        if self.camera is not None:
            self.camera.release()
        if self.autosave_job is not None:
            self.after_cancel(self.autosave_job)
            self.autosave_job = None
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
            if self.session.has_recoverable_state:
                self.session.save_manifest(self.autosave_path)
            else:
                self.autosave_path.unlink(missing_ok=True)
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
                getattr(self.import_files_entry, "_entry", self.import_files_entry),
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
        if hasattr(self, "camera_health_label"):
            self.camera_health_label.configure(text_color=state.color)

    def go_to_review_tab(self) -> None:
        self.tabs.set(self.tab_review_name)

    def go_to_export_tab(self) -> None:
        self.tabs.set(self.tab_export_name)

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
        self.import_files_var.set("\n".join(self.import_selected_files))
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
        self.import_folder_var.set(str(folder))
        self._import_paths(paths=paths)

    def open_page_tools_dialog(self) -> None:
        window = ctk.CTkToplevel(self)
        window.title("Page tools")
        window.geometry("440x510")
        window.resizable(False, False)
        window.transient(self)

        ctk.CTkLabel(
            window,
            text="Page tools",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 2))
        ctk.CTkLabel(
            window,
            text="These actions use the selection in Workspace.",
            text_color=("#60646c", "#a0a4ab"),
        ).pack(anchor="w", padx=16, pady=(0, 12))

        body = ctk.CTkFrame(window)
        body.pack(fill=ctk.BOTH, expand=True, padx=16, pady=(0, 12))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        def add(row: int, column: int, text: str, command) -> None:
            ctk.CTkButton(
                body,
                text=text,
                fg_color="transparent",
                border_width=1,
                command=command,
            ).grid(row=row, column=column, sticky="ew", padx=6, pady=6)

        ctk.CTkLabel(body, text="Deskew estimator", anchor="w").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        ctk.CTkOptionMenu(
            body,
            values=list(DESKEW_UI_METHODS),
            variable=self.deskew_method_var,
        ).grid(row=0, column=1, sticky="ew", padx=6, pady=6)

        add(1, 0, "Manual corners", self.open_manual_corners_editor)
        add(1, 1, "Auto crop", self.open_auto_crop_editor)
        add(2, 0, "Auto orient", self.auto_orient_selected)
        add(2, 1, "Auto deskew", self.auto_deskew_selected)
        add(3, 0, "Replace from file", self.replace_selected_page_from_file)
        add(3, 1, "Retake with camera", self.retake_selected_page_from_camera)
        add(4, 0, "Split book spread", self.split_selected_as_spread)
        add(4, 1, "Auto remove waves", self.remove_waves_selected)
        add(5, 0, "Adjust dewarp points", self.open_dewarp_points_editor)
        add(5, 1, "Refresh pages", self.refresh_page_list)
        add(6, 1, "Close", window.destroy)

        window.grab_set()

    def _sync_lens_mode_from_controls(self) -> None:
        inferred = infer_lens_mode(self.preprocess_preset_var.get(), self.postprocess_var.get())
        self.lens_mode_var.set(inferred)

    def _on_postprocess_mode_change(self, _value: str) -> None:
        self._sync_lens_mode_from_controls()
        self.update_page_preview()

    def _on_dewarp_method_change(self, _value: str) -> None:
        self.update_page_preview()

    def on_lens_mode_change(self, mode_name: str) -> None:
        profile = resolve_lens_mode_profile(mode_name)
        if profile is None:
            self._set_status("Lens mode set to Custom (manual controls).")
            self.update_page_preview()
            return

        self.preprocess_preset_var.set(profile.preset_name)
        self.postprocess_var.set(profile.postprocess_name)
        self.on_preprocess_preset_change(profile.preset_name)
        self._set_status(f"Lens mode set to {mode_name}.")
        self.update_page_preview()

    def on_preprocess_preset_change(self, preset_name: str) -> None:
        preset = PREPROCESS_PRESETS.get(preset_name)
        if preset is None:
            return
        self.preprocess_contrast_var.set(float(preset.contrast))
        self.preprocess_brightness_var.set(int(preset.brightness))
        self.preprocess_denoise_var.set(int(preset.denoise))
        self.preprocess_threshold_var.set(int(preset.threshold))

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
        if self.job_thread is not None and self.job_thread.is_alive():
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
                    try:
                        on_done(result)
                    except Exception as exc:
                        self._set_status(f"{name} failed: {exc}")
                        messagebox.showerror(f"{name} Error", str(exc))
                    finally:
                        self.cancel_task_button.configure(state=tk.DISABLED)
                elif kind == "error":
                    self.cancel_task_button.configure(state=tk.DISABLED)
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
                        report = replace(
                            report,
                            checks=tuple(check for check in report.checks if check.blocking),
                        )
                        failed = ", ".join(check.name for check in report.checks if not check.ok)
                        self._set_status(
                            f"Startup diagnostics failed: {failed}. Run 'uniscan doctor'."
                        )
                elif kind == "review_preview":
                    generation, image, error = payload
                    self._handle_review_preview_result(generation, image, error)
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

    def _build_import_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)

        row_folder = ctk.CTkFrame(tab)
        row_folder.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        row_folder.grid_columnconfigure(0, weight=1)
        self.import_folder_entry = ctk.CTkEntry(row_folder, textvariable=self.import_folder_var)
        self.import_folder_entry.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=10)
        ctk.CTkButton(
            row_folder, text="Folder...", width=100, command=self.choose_import_folder
        ).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkButton(
            row_folder, text="Import Folder", width=130, command=self.import_from_folder
        ).grid(
            row=0,
            column=2,
            padx=(0, 10),
            pady=10,
        )

        row_files = ctk.CTkFrame(tab)
        row_files.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        row_files.grid_columnconfigure(0, weight=1)
        self.import_files_entry = ctk.CTkEntry(row_files, textvariable=self.import_files_var)
        self.import_files_entry.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=10)
        ctk.CTkButton(
            row_files, text="Files... (multi)", width=110, command=self.choose_import_files
        ).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkButton(
            row_files, text="Import Files", width=130, command=self.import_from_files
        ).grid(
            row=0,
            column=2,
            padx=(0, 10),
            pady=10,
        )

        row_options = ctk.CTkFrame(tab)
        row_options.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        ctk.CTkLabel(row_options, text="PDF render DPI").pack(side=ctk.LEFT, padx=(10, 8), pady=10)
        ctk.CTkEntry(row_options, textvariable=self.import_pdf_dpi_var, width=90).pack(
            side=ctk.LEFT,
            padx=(0, 12),
            pady=10,
        )
        ctk.CTkCheckBox(
            row_options,
            text="Two-page spread mode (split at detected gutter)",
            variable=self.import_two_page_mode_var,
        ).pack(side=ctk.LEFT, padx=(0, 12), pady=10)
        ctk.CTkLabel(
            row_options,
            text="Boundary detection runs on every page during import.",
            anchor="w",
        ).pack(side=ctk.LEFT, padx=(0, 10), pady=10)

        row_actions = ctk.CTkFrame(tab)
        row_actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkButton(
            row_actions, text="Import from Listed Paths", command=self.import_from_files
        ).pack(
            side=ctk.LEFT,
            padx=10,
            pady=10,
        )
        ctk.CTkButton(row_actions, text="Workspace", command=self.go_to_review_tab).pack(
            side=ctk.LEFT, padx=0, pady=10
        )
        ctk.CTkButton(
            row_actions,
            text="Paste Clipboard",
            command=self.import_from_clipboard,
        ).pack(side=ctk.LEFT, padx=8, pady=10)

    def _build_export_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=1)

        row_scope = ctk.CTkFrame(tab)
        row_scope.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        ctk.CTkLabel(row_scope, text="Export scope").pack(side=ctk.LEFT, padx=(10, 8), pady=10)
        ctk.CTkOptionMenu(
            row_scope,
            values=["All pages", "Selected pages"],
            variable=self.export_scope_var,
        ).pack(side=ctk.LEFT, padx=(0, 12), pady=10)

        ctk.CTkLabel(row_scope, text="PDF DPI").pack(side=ctk.LEFT, padx=(0, 8), pady=10)
        ctk.CTkEntry(row_scope, textvariable=self.export_pdf_dpi_var, width=90).pack(
            side=ctk.LEFT,
            padx=(0, 10),
            pady=10,
        )

        row_pdf = ctk.CTkFrame(tab)
        row_pdf.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        row_pdf.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row_pdf, textvariable=self.export_pdf_path_var).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(10, 8),
            pady=10,
        )
        ctk.CTkButton(
            row_pdf, text="Save PDF...", width=120, command=self.choose_export_pdf_path
        ).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkButton(row_pdf, text="Export PDF", width=120, command=self.export_to_pdf).grid(
            row=0,
            column=2,
            padx=(0, 10),
            pady=10,
        )

        row_files = ctk.CTkFrame(tab)
        row_files.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        row_files.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row_files, textvariable=self.export_dir_var).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(10, 8),
            pady=10,
        )
        ctk.CTkButton(
            row_files, text="Dir...", width=80, command=self.choose_export_directory
        ).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkOptionMenu(
            row_files,
            values=["png", "jpg", "jpeg", "webp", "tif"],
            variable=self.export_format_var,
            width=100,
        ).grid(row=0, column=2, padx=(0, 6), pady=10)
        ctk.CTkButton(
            row_files,
            text="Export Files",
            width=120,
            command=self.export_to_files,
        ).grid(row=0, column=3, padx=(0, 10), pady=10)

        row_note = ctk.CTkFrame(tab)
        row_note.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkLabel(
            row_note,
            text="This build exports processed images and merged PDF only (no OCR stage).",
            anchor="w",
        ).pack(fill=ctk.X, padx=10, pady=8)

    def _parse_import_files_text(self, raw_text: str) -> list[str]:
        parts = [
            part.strip().strip('"') for part in re.split(r"[;\n\r]+", raw_text) if part.strip()
        ]
        return parts

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

    def choose_import_folder(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self.import_folder_var.set(path)

    def choose_import_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select image/PDF files",
            filetypes=[
                (
                    "Image and PDF",
                    "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp *.pdf",
                ),
                ("All files", "*.*"),
            ],
            multiple=True,
        )
        if files:
            self.import_selected_files = self._normalize_selected_files(files)
            self.import_files_var.set("\n".join(self.import_selected_files))

    def _on_drop_files(self, event) -> str:
        paths = paths_from_tk_drop(str(event.data), self.tk.splitlist)
        supported: list[Path] = []
        for path in paths:
            if path.is_dir():
                supported.extend(list_supported_in_folder(path))
            elif path.is_file() and path.suffix.lower() in (IMG_EXTS | PDF_EXTS):
                supported.append(path)
        if not supported:
            self._set_status("Drop contained no supported images or PDFs.")
            return "break"
        self.import_selected_files = self._normalize_selected_files(map(str, supported))
        self.import_files_var.set("\n".join(self.import_selected_files))
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
            supported = [
                path
                for path in paths
                if path.is_file() and path.suffix.lower() in (IMG_EXTS | PDF_EXTS)
            ]
            if not supported:
                raise RuntimeError("Clipboard does not contain an image or supported files.")
            self._import_paths(paths=supported)
        except Exception as exc:
            messagebox.showerror("Clipboard Import Error", str(exc))
            self._set_status("Clipboard import failed")

    def import_from_folder(self) -> None:
        try:
            folder = Path(self.import_folder_var.get().strip())
            paths = list_supported_in_folder(folder)
            if not paths:
                raise RuntimeError("No supported image/PDF files found in selected folder.")
            self._import_paths(paths=paths)
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            self._set_status("Folder import failed")

    def import_from_files(self) -> None:
        try:
            text_paths = self._parse_import_files_text(self.import_files_var.get())
            raw = text_paths if text_paths else list(self.import_selected_files)
            if not raw:
                raise RuntimeError("No files selected.")
            paths = [Path(item) for item in raw]
            missing = [path for path in paths if not path.exists() or not path.is_file()]
            if missing:
                raise RuntimeError(
                    "Some selected files do not exist:\n" + "\n".join(map(str, missing))
                )
            unsupported = [
                path for path in paths if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS)
            ]
            if unsupported:
                raise RuntimeError("Unsupported file type(s):\n" + "\n".join(map(str, unsupported)))
            self._import_paths(paths=paths)
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            self._set_status("File import failed")

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

    def refresh_page_list(self, keep_index: int | None = None) -> None:
        self.page_listbox.delete(0, tk.END)
        for idx, entry in enumerate(self.session.entries, start=1):
            tag = "" if entry.detected_contour is not None else "  ⚠"
            self.page_listbox.insert(tk.END, f"{idx:03d}  {entry.name}{tag}")

        page_count = len(self.session.entries)
        self.page_count_var.set(f"{page_count} page" if page_count == 1 else f"{page_count} pages")
        export_state = tk.NORMAL if page_count else tk.DISABLED
        for button_name in (
            "toolbar_export_button",
            "workspace_export_pdf_button",
            "workspace_export_files_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.configure(state=export_state)

        if keep_index is not None and len(self.session.entries) > 0:
            keep_index = max(0, min(keep_index, len(self.session.entries) - 1))
            self.page_listbox.selection_set(keep_index)
        self._sync_page_selection_to_session()
        self.update_page_preview()

    def _sync_page_selection_to_session(self) -> None:
        selected = set(self.page_listbox.curselection())
        for idx, entry in enumerate(self.session.entries):
            entry.selected = idx in selected

    def on_page_select(self, _event=None) -> None:
        self._sync_page_selection_to_session()
        self.update_page_preview()

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
            return

        index = selected[0]
        if index < 0 or index >= len(self.session.entries):
            self._clear_preview_label(self.page_preview_before_label)
            self._clear_preview_label(self.page_preview_after_label)
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            return

        entry = self.session.entries[index]
        mode = self.preview_mode_var.get()
        try:
            before = self._review_before_image(entry)
        except Exception as exc:
            message = f"Preview failed: {exc}"
            self._set_preview_message(self.page_preview_before_label, message)
            self._set_preview_message(self.page_preview_after_label, message)
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            self._set_status(message)
            return

        if mode in {"Original", "Compare"}:
            before_photo = self._to_ctk_photo_for_label(before, self.page_preview_before_label)
            self.page_preview_before_label.configure(image=before_photo, text="")
            self.page_preview_before_photo = before_photo
        else:
            self.page_preview_before_photo = None

        if mode in {"Processed", "Compare"}:
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
                ),
            )
        else:
            self.page_preview_after_photo = None

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
    ) -> None:
        self.review_preview_job = None
        if self._closing or generation != self.review_preview_generation:
            return

        def run() -> None:
            try:
                result = process_document_page(
                    source,
                    replace(request, cancel_cb=cancel_event.is_set),
                )
            except Exception as exc:
                if cancel_event.is_set():
                    return
                self.job_queue.put(("review_preview", (generation, None, str(exc))))
            else:
                if not cancel_event.is_set():
                    self.job_queue.put(("review_preview", (generation, result.image, None)))

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
        index = self._single_selected_index()
        if index is None:
            self._set_status("Select exactly one page to move.")
            return
        if index == 0:
            return
        entry_id = self.session.entries[index].entry_id
        moved = self.session.move(entry_id, -1)
        if moved:
            self.refresh_page_list(keep_index=index - 1)
            self._set_status("Moved page up")

    def move_selected_down(self) -> None:
        index = self._single_selected_index()
        if index is None:
            self._set_status("Select exactly one page to move.")
            return
        if index >= len(self.session.entries) - 1:
            return
        entry_id = self.session.entries[index].entry_id
        moved = self.session.move(entry_id, 1)
        if moved:
            self.refresh_page_list(keep_index=index + 1)
            self._set_status("Moved page down")

    def select_all_pages(self) -> None:
        self.page_listbox.selection_set(0, tk.END)
        self._sync_page_selection_to_session()
        self.update_page_preview()
        self._set_status("Selected all pages")

    def clear_page_selection(self) -> None:
        self.page_listbox.selection_clear(0, tk.END)
        self._sync_page_selection_to_session()
        self.update_page_preview()
        self._set_status("Selection cleared")

    def delete_selected_pages(self) -> None:
        self._sync_page_selection_to_session()
        removed = self.session.remove_selected()
        if removed <= 0:
            self._set_status("No selected pages to delete")
            return
        self.refresh_page_list()
        self._set_status(f"Deleted {removed} page(s). Session pages: {len(self.session)}")

    def replace_selected_page_from_file(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page to replace.")
            return

        path = filedialog.askopenfilename(
            title="Replace selected page from image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.webp;*.bmp"),
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

    def _open_corner_editor_dialog(self, indices: list[int], *, auto_detect: bool) -> None:
        if not indices:
            self._set_status("Select page(s) for corner editing.")
            return

        entries = [
            self.session.entries[idx] for idx in indices if 0 <= idx < len(self.session.entries)
        ]
        if not entries:
            self._set_status("No valid pages available for corner editing.")
            return

        state = {"index": 0}
        points_by_entry: dict[str, np.ndarray] = {}

        win = ctk.CTkToplevel(self)
        win.title("Auto Crop" if auto_detect else "Manual Corners")
        win.geometry("1120x860")
        win.minsize(760, 580)

        header = ctk.CTkLabel(
            win,
            text="Browse pages, adjust corners on the raw source, then apply to the current page or all loaded pages.",
            anchor="w",
        )
        header.pack(fill=ctk.X, padx=12, pady=(12, 6))

        meta_var = tk.StringVar(value="")
        meta_label = ctk.CTkLabel(win, textvariable=meta_var, anchor="w")
        meta_label.pack(fill=ctk.X, padx=12, pady=(0, 8))

        canvas_frame = ctk.CTkFrame(win)
        canvas_frame.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(0, 10))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        labels = ["TL", "TR", "BR", "BL"]
        drag = {"idx": None}
        canvas_image_ref = {"photo": None}
        view_state = {
            "source_shape": None,
            "display_shape": None,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "points": self._default_corner_points(
                entries[0].preview_raw_image
                if self.lightweight_preview_var.get()
                else entries[0].raw_image
            ),
        }

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

        def _display_image_for(entry) -> np.ndarray:
            return (
                entry.preview_raw_image if self.lightweight_preview_var.get() else entry.raw_image
            )

        def _init_points_for(entry) -> np.ndarray:
            cached = points_by_entry.get(entry.entry_id)
            if cached is not None:
                return cached

            # Prefer the already-detected contour from import / previous edit.
            existing = entry.detected_contour
            if existing is not None and not auto_detect:
                points = np.asarray(existing, dtype=np.float32).reshape(-1, 2).copy()
                points_by_entry[entry.entry_id] = points
                return points

            detected: np.ndarray | None = None
            if auto_detect:
                detected = self._detect_corner_points(entry.raw_image)
                if detected is None and existing is not None:
                    detected = np.asarray(existing, dtype=np.float32).reshape(-1, 2)

            if detected is not None:
                points = np.asarray(detected, dtype=np.float32).reshape(-1, 2).copy()
            else:
                # Default points already live in source coordinates of the raw image.
                points = self._default_corner_points(entry.raw_image)
            points_by_entry[entry.entry_id] = points
            return points

        def _redraw() -> None:
            canvas.delete("overlay")
            points = view_state["points"]
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
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
                x = float(pt[0]) / max(scale_x, 1e-6)
                y = float(pt[1]) / max(scale_y, 1e-6)
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
                sx = float(pt[0]) / max(scale_x, 1e-6)
                sy = float(pt[1]) / max(scale_y, 1e-6)
                if 0 <= sx <= display_w and 0 <= sy <= display_h:
                    r = 7
                    canvas.create_oval(
                        sx - r, sy - r, sx + r, sy + r, fill="#ff3355", outline="", tags="overlay"
                    )
                    canvas.create_text(
                        sx + 14, sy - 10, text=labels[idx_p], fill="#ffffff", tags="overlay"
                    )

        def _load_current_entry() -> None:
            entry_index, entry = _current_entry()
            source_image = entry.raw_image
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")

            display_image = _display_image_for(entry)
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
            canvas.configure(width=view_w, height=view_h)
            canvas.delete("all")
            canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
            canvas_image_ref["photo"] = tk_img

            points = _init_points_for(entry)
            view_state["points"] = points
            view_state["source_shape"] = (source_h, source_w)
            view_state["display_shape"] = (display_h, display_w)
            view_state["scale_x"] = source_w / max(1, display_w)
            view_state["scale_y"] = source_h / max(1, display_h)
            meta_var.set(f"{state['index'] + 1}/{len(entries)}  {entry.name}")
            _redraw()

        def _nearest_handle(px: float, py: float) -> int | None:
            points = view_state["points"]
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            best_i = None
            best_d2 = 14.0 * 14.0
            for idx_p, pt in enumerate(points):
                sx = float(pt[0]) / max(scale_x, 1e-6)
                sy = float(pt[1]) / max(scale_y, 1e-6)
                d2 = (sx - px) ** 2 + (sy - py) ** 2
                if d2 <= best_d2:
                    best_i = idx_p
                    best_d2 = d2
            return best_i

        def _on_down(event):
            drag["idx"] = _nearest_handle(event.x, event.y)

        def _on_move(event):
            idx_p = drag["idx"]
            if idx_p is None:
                return
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            source_h, source_w = view_state["source_shape"]
            x = float(event.x) * max(scale_x, 1e-6)
            y = float(event.y) * max(scale_y, 1e-6)
            x = max(0.0, min(float(source_w - 1), x))
            y = max(0.0, min(float(source_h - 1), y))
            points = view_state["points"]
            points[idx_p][0] = x
            points[idx_p][1] = y
            _redraw()

        def _on_up(_event):
            drag["idx"] = None

        def _reset():
            source_h, source_w = view_state["source_shape"]
            points = view_state["points"]
            points[:] = self._default_corner_points(
                np.zeros((source_h, source_w, 3), dtype=np.uint8)
            )
            _redraw()

        def _auto_detect_current():
            entry_index, entry = _current_entry()
            detected = self._detect_corner_points(entry.raw_image)
            if detected is None:
                messagebox.showwarning(
                    "Auto Crop", f"Document boundaries were not detected for {entry.name}."
                )
                return
            points = np.asarray(detected, dtype=np.float32).reshape(-1, 2).copy()
            points_by_entry[entry.entry_id] = points
            view_state["points"] = points
            _redraw()

        def _apply_entry(entry_index: int, entry, points: np.ndarray) -> None:
            source_image = entry.raw_image
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")
            warped = warp_perspective_from_points(source_image, points.astype(np.float32))
            if warped is None or warped.size == 0:
                raise RuntimeError("Perspective transform returned empty image.")
            entry.original_image = warped
            entry.detected_contour = points.astype(np.float32).reshape(-1, 2).copy()
            entry.detected_backend = "manual"
            self._reprocess_entry_from_original(entry)

        def _apply_current():
            try:
                entry_index, entry = _current_entry()
                points = view_state["points"]
                _apply_entry(entry_index, entry, points)
                self.refresh_page_list(keep_index=entry_index)
                self._set_status(f"Applied crop to {entry.name}.")
            except Exception as exc:
                messagebox.showerror("Auto Crop Error", str(exc))

        def _apply_all():
            try:
                for idx_offset, entry in enumerate(entries):
                    points = points_by_entry.get(entry.entry_id)
                    if points is None:
                        detected = self._detect_corner_points(entry.raw_image)
                        if detected is not None:
                            points = np.asarray(detected, dtype=np.float32).reshape(-1, 2).copy()
                        elif entry.detected_contour is not None:
                            points = (
                                np.asarray(entry.detected_contour, dtype=np.float32)
                                .reshape(-1, 2)
                                .copy()
                            )
                        else:
                            points = self._default_corner_points(entry.raw_image)
                    _apply_entry(indices[idx_offset], entry, points)
                self.refresh_page_list(keep_index=indices[min(state["index"], len(indices) - 1)])
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

        canvas.bind("<Button-1>", _on_down)
        canvas.bind("<B1-Motion>", _on_move)
        canvas.bind("<ButtonRelease-1>", _on_up)

        controls = ctk.CTkFrame(win)
        controls.pack(fill=ctk.X, padx=12, pady=(0, 12))
        ctk.CTkButton(controls, text="Prev", width=90, command=_prev_page).pack(side=ctk.LEFT)
        ctk.CTkButton(controls, text="Next", width=90, command=_next_page).pack(
            side=ctk.LEFT, padx=6
        )
        ctk.CTkButton(controls, text="Auto Detect", width=110, command=_auto_detect_current).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(controls, text="Reset", width=90, command=_reset).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(controls, text="Apply Current", width=120, command=_apply_current).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(controls, text="Apply All", width=100, command=_apply_all).pack(
            side=ctk.LEFT, padx=6
        )
        ctk.CTkButton(
            controls,
            text="Close",
            width=90,
            command=win.destroy,
        ).pack(side=ctk.RIGHT)

        _load_current_entry()
        win.attributes("-topmost", True)
        win.lift()
        win.attributes("-topmost", False)

    def open_manual_corners_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page for manual corner edit.")
            return
        self._open_corner_editor_dialog([index], auto_detect=False)

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
            entry.original_image = rotated
            self._reprocess_entry_from_original(entry)
        self.refresh_page_list(keep_index=indices[-1])
        self._set_status(f"Rotated {len(indices)} page(s) left.")

    def rotate_selected_right(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to rotate.")
            return
        for idx in indices:
            entry = self.session.entries[idx]
            rotated = cv2.rotate(entry.original_image, cv2.ROTATE_90_CLOCKWISE)
            entry.original_image = rotated
            self._reprocess_entry_from_original(entry)
        self.refresh_page_list(keep_index=indices[-1])
        self._set_status(f"Rotated {len(indices)} page(s) right.")

    def split_selected_as_spread(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to split as spread.")
            return

        try:
            split_count = 0
            untouched_count = 0
            # Process from the end backwards so insertion indices stay stable.
            for idx in sorted(indices, reverse=True):
                entry = self.session.entries[idx]
                raw = entry.raw_image
                warped = entry.original_image
                split_pair = _split_spread_pair(raw, warped)
                if split_pair is None:
                    untouched_count += 1
                    continue
                raw_halves, warped_halves = split_pair

                # Replace existing entry with the left half, append the right half after it.
                left_raw, right_raw = raw_halves
                left_warped, right_warped = warped_halves
                self.session.replace_entry_image(
                    entry.entry_id,
                    raw_image=left_raw,
                    original_image=left_warped,
                    current_image=left_warped,
                    name=f"{entry.name} [L]",
                    contour=None,
                    backend=None,
                )
                right_entry = self.session.add_image_with_contour(
                    name=f"{entry.name} [R]",
                    raw_image=right_raw,
                    warped_image=right_warped,
                    contour=None,
                    backend=None,
                )
                self._reprocess_entry_from_original(entry)
                self._reprocess_entry_from_original(right_entry)
                # New entry is at the end of the list; move it to right after the original.
                from_index = self.session.entries.index(right_entry)
                target_index = idx + 1
                while from_index > target_index:
                    if not self.session.move(right_entry.entry_id, -1):
                        break
                    from_index -= 1
                split_count += 1

            self.refresh_page_list(keep_index=indices[0])
            msg = f"Split {split_count} page(s) at the detected gutter."
            if untouched_count:
                msg += (
                    f" {untouched_count} page(s) had no confident gutter and were left untouched."
                )
            self._set_status(msg)
        except Exception as exc:
            messagebox.showerror("Split Spread Error", str(exc))
            self._set_status("Split spread failed")

    def auto_deskew_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to deskew.")
            return

        angles: list[float] = []
        method = DESKEW_UI_METHODS.get(
            self.deskew_method_var.get(),
            DESKEW_METHOD_HYBRID,
        )
        for idx in indices:
            entry = self.session.entries[idx]
            stage_result = process_document_page(
                entry.original_image,
                PageProcessingRequest(deskew_method=method),
            )
            entry.original_image = stage_result.image
            self._reprocess_entry_from_original(entry)
            angles.append(stage_result.diagnostics.deskew_angle_degrees)

        self.refresh_page_list(keep_index=indices[-1])
        mean_angle = sum(angles) / max(1, len(angles))
        self._set_status(f"Deskewed {len(indices)} page(s), avg angle {mean_angle:.1f} deg.")

    def auto_orient_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to orient.")
            return

        diagnostics = []
        for idx in indices:
            entry = self.session.entries[idx]
            stage_result = process_document_page(
                entry.original_image,
                PageProcessingRequest(orientation_method=ORIENTATION_METHOD_AUTO),
            )
            oriented = stage_result.image
            item = stage_result.diagnostics.orientation
            if item.applied:
                entry.original_image = oriented
                self._reprocess_entry_from_original(entry)
            diagnostics.append(item)

        self.refresh_page_list(keep_index=indices[-1])
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
        for idx in indices:
            entry = self.session.entries[idx]
            entry.clear_dewarp_control_points()
            diagnostics.append(self._reprocess_entry_from_original(entry))
        self.refresh_page_list(keep_index=indices[-1])
        applied = sum(item.applied for item in diagnostics)
        max_displacement = max((item.max_displacement_px for item in diagnostics), default=0.0)
        self._set_status(
            f"Removed page waves on {applied}/{len(indices)} page(s); "
            f"max correction {max_displacement:.1f}px."
        )

    def open_dewarp_points_editor(self) -> None:
        index, entry = self._single_selected_entry()
        if entry is None or index is None:
            self._set_status("Select exactly one page to adjust dewarp points.")
            return

        source = entry.original_image
        source_height, source_width = source.shape[:2]
        scale = min(450 / max(1, source_width), 520 / max(1, source_height), 1.0)
        display_width = max(1, int(round(source_width * scale)))
        display_height = max(1, int(round(source_height * scale)))
        display_source = cv2.resize(
            source,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

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

        window = ctk.CTkToplevel(self)
        window.title("Adjust dewarp control points")
        window.geometry(f"{max(980, display_width * 2 + 80)}x{max(700, display_height + 150)}")
        window.minsize(960, 680)
        window.transient(self)

        ctk.CTkLabel(
            window,
            text="Automatic dewarp with user correction",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            window,
            text=(
                "Drag the blue points vertically. The guides show the same correction at "
                "three page heights; the right pane is the resulting preview."
            ),
            text_color=("#60646c", "#a0a4ab"),
        ).pack(anchor="w", padx=16, pady=(0, 10))

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
            width=display_width,
            height=display_height,
            bg="#202225",
            highlightthickness=1,
            highlightbackground="#45484d",
        )
        right_canvas = tk.Canvas(
            panes,
            width=display_width,
            height=display_height,
            bg="#202225",
            highlightthickness=1,
            highlightbackground="#45484d",
        )
        left_canvas.grid(row=1, column=0, padx=8, pady=(0, 8))
        right_canvas.grid(row=1, column=1, padx=8, pady=(0, 8))

        def to_photo(image: np.ndarray) -> ImageTk.PhotoImage:
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return ImageTk.PhotoImage(Image.fromarray(rgb))

        state = {
            "points": initial_points,
            "active": None,
            "source_photo": to_photo(display_source),
            "corrected_photo": None,
        }
        left_canvas.create_image(0, 0, anchor=tk.NW, image=state["source_photo"])
        status = tk.StringVar(value=initial_message)

        def current_model(*, source_name: str = "user") -> DewarpModel:
            return DewarpModel(
                method=DEWARP_METHOD_TEXTLINE,
                control_points=tuple(state["points"]),
                source=source_name,
            )

        def draw_overlay() -> None:
            left_canvas.delete("dewarp-overlay")
            point_x = np.asarray([point[0] for point in state["points"]], dtype=np.float32)
            point_y = np.asarray([point[1] for point in state["points"]], dtype=np.float32)
            guide_x = np.linspace(0.0, 1.0, 160, dtype=np.float32)
            guide_y = np.interp(guide_x, point_x, point_y)
            for anchor in (0.25, 0.5, 0.75):
                coords: list[float] = []
                for x_value, displacement in zip(guide_x, guide_y):
                    coords.extend(
                        [
                            float(x_value * (display_width - 1)),
                            float((anchor + displacement) * display_height),
                        ]
                    )
                left_canvas.create_line(
                    *coords,
                    fill="#36a3ff",
                    width=2 if anchor == 0.5 else 1,
                    tags="dewarp-overlay",
                )
            for point_index, (x_value, displacement) in enumerate(state["points"]):
                x_pos = x_value * (display_width - 1)
                y_pos = (0.5 + displacement) * display_height
                radius = 6
                left_canvas.create_oval(
                    x_pos - radius,
                    y_pos - radius,
                    x_pos + radius,
                    y_pos + radius,
                    fill="#1f6aa5",
                    outline="#ffffff",
                    width=1,
                    tags=("dewarp-overlay", f"point-{point_index}"),
                )

        def render_corrected() -> None:
            corrected = process_document_page(
                display_source,
                PageProcessingRequest(
                    dewarp_method=DEWARP_METHOD_TEXTLINE,
                    dewarp_model=current_model(),
                ),
            ).image
            state["corrected_photo"] = to_photo(corrected)
            right_canvas.delete("all")
            right_canvas.create_image(
                0,
                0,
                anchor=tk.NW,
                image=state["corrected_photo"],
            )

        def nearest_point(x_pos: float, y_pos: float) -> int | None:
            best_index = None
            best_distance = 16.0
            for point_index, (x_value, displacement) in enumerate(state["points"]):
                px = x_value * (display_width - 1)
                py = (0.5 + displacement) * display_height
                distance = float(np.hypot(x_pos - px, y_pos - py))
                if distance < best_distance:
                    best_distance = distance
                    best_index = point_index
            return best_index

        def on_down(event) -> None:
            state["active"] = nearest_point(event.x, event.y)

        def on_move(event) -> None:
            active = state["active"]
            if active is None:
                return
            x_value, _old_displacement = state["points"][active]
            displacement = float(np.clip((event.y / display_height) - 0.5, -0.2, 0.2))
            state["points"][active] = (x_value, displacement)
            draw_overlay()

        def on_up(_event) -> None:
            if state["active"] is None:
                return
            state["active"] = None
            render_corrected()
            status.set("User-adjusted preview. Apply to save these points for the page.")

        def use_automatic() -> None:
            model, diagnostics = estimate_textline_dewarp_model(source)
            if model is None:
                status.set(f"Automatic model unavailable: {diagnostics.reason}.")
                return
            state["points"] = list(model.control_points)
            draw_overlay()
            render_corrected()
            status.set(f"Automatic model restored from {diagnostics.line_count} supporting lines.")

        def use_neutral() -> None:
            state["points"] = [(float(x), 0.0) for x in np.linspace(0.0, 1.0, 9, dtype=np.float32)]
            draw_overlay()
            render_corrected()
            status.set("Neutral correction model. The page will not be locally warped.")

        def apply_points() -> None:
            try:
                entry.set_dewarp_control_points(state["points"])
                self.dewarp_method_var.set("Text lines (offline)")
                diagnostics = self._reprocess_entry_from_original(entry)
                self.refresh_page_list(keep_index=index)
                self._set_status(
                    f"Saved {len(entry.dewarp_control_points or ())} dewarp points; "
                    f"max correction {diagnostics.max_displacement_px:.1f}px."
                )
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Dewarp Control Points", str(exc))

        left_canvas.bind("<Button-1>", on_down)
        left_canvas.bind("<B1-Motion>", on_move)
        left_canvas.bind("<ButtonRelease-1>", on_up)
        draw_overlay()
        render_corrected()

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
        ctk.CTkButton(actions, text="Apply points", command=apply_points).pack(side=ctk.RIGHT)
        ctk.CTkButton(
            actions,
            text="Cancel",
            fg_color="transparent",
            border_width=1,
            command=window.destroy,
        ).pack(side=ctk.RIGHT, padx=8)
        window.grab_set()

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
                stage="Apply processing",
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
            self.review_processing_window.focus()
            return

        window = ctk.CTkToplevel(self)
        window.title("Review Processing - Advanced")
        window.resizable(width=False, height=False)
        self.review_processing_window = window

        ctk.CTkLabel(window, text="Tune processing settings for Review preview/apply.").pack(
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
            window.destroy()

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
        ctk.CTkButton(actions, text="Close", command=_on_close, width=100).pack(
            side=ctk.LEFT, padx=8
        )

        window.protocol("WM_DELETE_WINDOW", _on_close)
        window.attributes("-topmost", True)
        window.grab_set()
        window.attributes("-topmost", False)

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

        def worker(emit, is_cancelled):
            return self._stage_apply_pages(
                snapshots,
                emit=emit,
                is_cancelled=is_cancelled,
            )

        def on_done(staged):
            try:
                cache_hit_count = self._commit_staged_apply(snapshots, staged)
                self.refresh_page_list(keep_index=keep_index)
                cache_note = f" Stage cache hits: {cache_hit_count}." if cache_hit_count else ""
                self._set_status(f"Reprocessed {len(target_entries)} {scope}.{cache_note}")
            finally:
                snapshot_dir.cleanup()

        if not self._start_background_job(
            "Apply processing",
            worker,
            on_done,
            on_error=snapshot_dir.cleanup,
        ):
            snapshot_dir.cleanup()

    def choose_export_pdf_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save merged PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.export_pdf_path_var.set(path)

    def choose_export_directory(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.export_dir_var.set(path)

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
            "DPI and Apply processing to those pages again."
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
