"""Unified application shell."""

from __future__ import annotations

import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from uniscan.export import (
    export_image_paths_as_files,
    export_image_paths_as_pdf,
)
from uniscan.core.geometry import warp_perspective_from_points
from uniscan.core.preprocess import (
    LENS_MODE_VALUES,
    PREPROCESS_PRESETS,
    PreprocessSettings,
    apply_enhancements,
    deskew_document,
    infer_lens_mode,
    resolve_lens_mode_profile,
)
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.scanner_adapter import (
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    ScanAdapterError,
    scan_with_document_detector,
)
from uniscan.io import CameraService
from uniscan.io.loaders import IMG_EXTS, PDF_EXTS, imread_unicode, list_supported_in_folder, load_input_items
from uniscan.session import CaptureSession
from uniscan.ui.camera_health import camera_health_state

PREVIEW_WAIT_MS = 25
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


class UnifiedScanApp(ctk.CTk):
    """Main window for the unified scanner application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("UniScan")
        self.geometry("1280x800")
        self.minsize(1024, 680)

        self.session = CaptureSession()
        self.camera: CameraService | None = None
        self.preview_job: str | None = None
        self.preview_photo: ctk.CTkImage | None = None
        self.page_preview_before_photo: ctk.CTkImage | None = None
        self.page_preview_after_photo: ctk.CTkImage | None = None
        self.review_processing_window: ctk.CTkToplevel | None = None
        self.corner_editor_window: ctk.CTkToplevel | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.camera_health_var = tk.StringVar(value="Camera: Closed")
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_shots_var = tk.IntVar(value=1)
        self.camera_delay_var = tk.DoubleVar(value=1.0)
        self.apply_changes_to_all_var = tk.BooleanVar(value=False)
        self.lightweight_preview_var = tk.BooleanVar(value=True)
        self.postprocess_var = tk.StringVar(value="None")
        self.lens_mode_var = tk.StringVar(value="Document")
        self.preprocess_preset_var = tk.StringVar(value="Document")
        self.preprocess_contrast_var = tk.DoubleVar(value=1.25)
        self.preprocess_brightness_var = tk.IntVar(value=10)
        self.preprocess_denoise_var = tk.IntVar(value=4)
        self.preprocess_threshold_var = tk.IntVar(value=170)
        self.import_folder_var = tk.StringVar()
        self.import_files_var = tk.StringVar()
        self.import_pdf_dpi_var = tk.IntVar(value=300)
        self.import_selected_files: list[str] = []
        self.export_scope_var = tk.StringVar(value="All pages")
        self.export_pdf_path_var = tk.StringVar()
        self.export_dir_var = tk.StringVar()
        self.export_format_var = tk.StringVar(value="png")
        self.export_pdf_dpi_var = tk.IntVar(value=300)
        self.job_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.job_cancel_event = threading.Event()
        self.job_thread: threading.Thread | None = None
        self.tab_import_name = "1. Import"
        self.tab_scan_name = "2. Scan"
        self.tab_review_name = "3. Review"
        self.tab_export_name = "4. Export"

        self._build_ui()
        self.on_lens_mode_change(self.lens_mode_var.get())
        self._update_camera_health()
        self.after(120, self._poll_job_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self)
        container.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        title = ctk.CTkLabel(
            container,
            text="UniScan - Unified Document Processing",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(anchor="w", padx=12, pady=(12, 8))

        flow_tip = ctk.CTkLabel(
            container,
            text="Flow: 1) Import files (main) or scan camera  2) Review: all processing/editing  3) Export PDF or images",
            anchor="w",
        )
        flow_tip.pack(fill=ctk.X, padx=12, pady=(0, 6))

        self.tabs = ctk.CTkTabview(container)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        self.import_tab = self.tabs.add(self.tab_import_name)
        self.capture_tab = self.tabs.add(self.tab_scan_name)
        self.pages_tab = self.tabs.add(self.tab_review_name)
        self.export_tab = self.tabs.add(self.tab_export_name)

        self._build_import_tab(self.import_tab)
        self._build_capture_tab(self.capture_tab)
        self._build_pages_tab(self.pages_tab)
        self._build_export_tab(self.export_tab)
        self.tabs.set(self.tab_import_name)

        status_frame = ctk.CTkFrame(container)
        status_frame.pack(fill=ctk.X, padx=12, pady=(0, 12))
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=ctk.X, padx=10, pady=8)

    def _build_capture_tab(self, tab: ctk.CTkFrame) -> None:
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, sticky="ns", padx=(10, 8), pady=10)

        ctk.CTkLabel(controls, text="Camera index").pack(anchor="w", padx=10, pady=(10, 4))
        self.camera_index_entry = ctk.CTkEntry(controls, textvariable=self.camera_index_var, width=120)
        self.camera_index_entry.pack(anchor="w", padx=10)

        row_open = ctk.CTkFrame(controls, fg_color="transparent")
        row_open.pack(fill=ctk.X, padx=10, pady=(8, 2))
        ctk.CTkButton(row_open, text="Open", width=90, command=self.open_camera).pack(side=ctk.LEFT)
        ctk.CTkButton(row_open, text="Close", width=90, command=self.close_camera).pack(side=ctk.LEFT, padx=6)
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
            text="Scan only acquires raw images.\nAll processing is in Review tab.",
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
        ctk.CTkButton(row_preview, text="Start Preview", width=120, command=self.start_preview).pack(
            side=ctk.LEFT
        )
        ctk.CTkButton(row_preview, text="Stop", width=70, command=self.stop_preview).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_capture = ctk.CTkFrame(controls, fg_color="transparent")
        row_capture.pack(fill=ctk.X, padx=10, pady=(4, 10))
        ctk.CTkButton(row_capture, text="Capture One", width=120, command=self.capture_one).pack(side=ctk.LEFT)
        ctk.CTkButton(row_capture, text="Capture Burst", width=120, command=self.capture_burst).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(row_capture, text="Review", width=80, command=self.go_to_review_tab).pack(side=ctk.LEFT)

        ctk.CTkLabel(
            controls,
            text="Tip: capture first, then open Review to process pages.",
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
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(tab)
        left.grid(row=0, column=0, sticky="ns", padx=(12, 8), pady=12)

        ctk.CTkLabel(left, text="Session Pages").pack(anchor="w", padx=10, pady=(10, 6))
        self.page_listbox = tk.Listbox(left, selectmode=tk.EXTENDED, width=46, height=24)
        self.page_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.page_listbox.bind("<<ListboxSelect>>", self.on_page_select)

        row_a = ctk.CTkFrame(left, fg_color="transparent")
        row_a.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_a, text="Move Up", width=110, command=self.move_selected_up).pack(side=ctk.LEFT)
        ctk.CTkButton(row_a, text="Move Down", width=110, command=self.move_selected_down).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_b = ctk.CTkFrame(left, fg_color="transparent")
        row_b.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_b, text="Select All", width=110, command=self.select_all_pages).pack(side=ctk.LEFT)
        ctk.CTkButton(row_b, text="Clear Sel", width=110, command=self.clear_page_selection).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_c = ctk.CTkFrame(left, fg_color="transparent")
        row_c.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_c, text="Delete Sel", width=110, command=self.delete_selected_pages).pack(side=ctk.LEFT)
        ctk.CTkButton(row_c, text="Refresh", width=110, command=self.refresh_page_list).pack(side=ctk.LEFT, padx=6)

        row_d = ctk.CTkFrame(left, fg_color="transparent")
        row_d.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(
            row_d,
            text="Manual Corners",
            width=110,
            command=self.open_manual_corners_editor,
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(row_d, text="Auto Crop...", width=110, command=self.open_auto_crop_editor).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(row_d, text="Replace Sel...", width=110, command=self.replace_selected_page_from_file).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_e = ctk.CTkFrame(left, fg_color="transparent")
        row_e.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_e, text="Rotate Left", width=110, command=self.rotate_selected_left).pack(side=ctk.LEFT)
        ctk.CTkButton(row_e, text="Rotate Right", width=110, command=self.rotate_selected_right).pack(
            side=ctk.LEFT,
            padx=6,
        )

        row_f = ctk.CTkFrame(left, fg_color="transparent")
        row_f.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_f, text="Auto Deskew Sel", width=226, command=self.auto_deskew_selected).pack(
            side=ctk.LEFT
        )

        row_g = ctk.CTkFrame(left, fg_color="transparent")
        row_g.pack(fill=ctk.X, padx=10, pady=(0, 4))
        ctk.CTkButton(row_g, text="Retake Cam", width=226, command=self.retake_selected_page_from_camera).pack(
            side=ctk.LEFT
        )

        processing = ctk.CTkFrame(left)
        processing.pack(fill=ctk.X, padx=10, pady=(6, 8))
        ctk.CTkLabel(processing, text="Review Processing").pack(anchor="w", padx=8, pady=(8, 6))

        ctk.CTkCheckBox(
            processing,
            text="Apply all changes to all files",
            variable=self.apply_changes_to_all_var,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ctk.CTkCheckBox(
            processing,
            text="Use lightweight previews (Full HD)",
            variable=self.lightweight_preview_var,
        ).pack(anchor="w", padx=8, pady=(0, 8))

        row_modes = ctk.CTkFrame(processing, fg_color="transparent")
        row_modes.pack(fill=ctk.X, padx=8, pady=(0, 6))
        ctk.CTkLabel(row_modes, text="Lens").pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkOptionMenu(
            row_modes,
            values=list(LENS_MODE_VALUES),
            variable=self.lens_mode_var,
            command=self.on_lens_mode_change,
            width=90,
        ).pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkLabel(row_modes, text="Post").pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkOptionMenu(
            row_modes,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            variable=self.postprocess_var,
            command=self._on_postprocess_mode_change,
            width=105,
        ).pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkLabel(row_modes, text="Preset").pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkOptionMenu(
            row_modes,
            values=list(PREPROCESS_PRESETS.keys()),
            variable=self.preprocess_preset_var,
            command=self.on_preprocess_preset_change,
            width=105,
        ).pack(side=ctk.LEFT)

        row_process_a = ctk.CTkFrame(processing, fg_color="transparent")
        row_process_a.pack(fill=ctk.X, padx=8, pady=(0, 6))
        ctk.CTkButton(
            row_process_a,
            text="Preview Selected",
            width=102,
            command=self.update_page_preview,
        ).pack(side=ctk.LEFT)
        ctk.CTkButton(
            row_process_a,
            text="Advanced...",
            width=102,
            command=self.open_review_processing_dialog,
        ).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(
            processing,
            text="Apply Changes",
            command=self.apply_review_changes,
        ).pack(fill=ctk.X, padx=8, pady=(0, 8))

        ctk.CTkButton(
            left,
            text="Go to Export",
            command=self.go_to_export_tab,
        ).pack(fill=ctk.X, padx=10, pady=(0, 10))

        right = ctk.CTkFrame(tab)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 12), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        before_frame = ctk.CTkFrame(right)
        before_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        before_frame.grid_rowconfigure(1, weight=1)
        before_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(before_frame, text="Before (Original)").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.page_preview_before_label = ctk.CTkLabel(before_frame, text="No page selected")
        self.page_preview_before_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        after_frame = ctk.CTkFrame(right)
        after_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        after_frame.grid_rowconfigure(1, weight=1)
        after_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(after_frame, text="After (Review Settings Preview)").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        self.page_preview_after_label = ctk.CTkLabel(after_frame, text="No page selected")
        self.page_preview_after_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.refresh_page_list()

    def _on_close(self) -> None:
        self.stop_preview()
        self.job_cancel_event.set()
        if self.review_processing_window is not None and self.review_processing_window.winfo_exists():
            self.review_processing_window.destroy()
            self.review_processing_window = None
        if self.camera is not None:
            self.camera.release()
        self.session.close()
        self.destroy()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

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

    def _sync_lens_mode_from_controls(self) -> None:
        inferred = infer_lens_mode(self.preprocess_preset_var.get(), self.postprocess_var.get())
        self.lens_mode_var.set(inferred)

    def _on_postprocess_mode_change(self, _value: str) -> None:
        self._sync_lens_mode_from_controls()
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

        # Preset may guide main postprocess mode for clearer UX.
        if preset_name == "B/W High Contrast":
            self.postprocess_var.set("Black and White")
        elif preset_name in {"Document", "Whiteboard"} and self.postprocess_var.get() == "None":
            self.postprocess_var.set("Grayscale")
        elif preset_name == "Photo" and self.postprocess_var.get() == "Black and White":
            self.postprocess_var.set("None")
        self._sync_lens_mode_from_controls()
        self.update_page_preview()

    def _set_job_display(self, *, stage: str | None = None, current: str | None = None, progress: int | None = None) -> None:
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

    def _start_background_job(self, name: str, worker, on_done) -> bool:
        if self.job_thread is not None and self.job_thread.is_alive():
            messagebox.showwarning("Busy", "Another background job is already running.")
            return False

        self.job_cancel_event.clear()
        self._set_job_display(stage=name, current="Starting...", progress=0)

        def emit(stage: str | None = None, current: str | None = None, progress: int | None = None) -> None:
            self.job_queue.put(("progress", (stage, current, progress)))

        def run() -> None:
            try:
                result = worker(emit, self.job_cancel_event.is_set)
                self.job_queue.put(("done", (on_done, result, name)))
            except Exception as exc:
                self.job_queue.put(("error", (name, str(exc))))

        self.job_thread = threading.Thread(target=run, daemon=True)
        self.job_thread.start()
        return True

    def _poll_job_queue(self) -> None:
        try:
            for _ in range(6):
                kind, payload = self.job_queue.get_nowait()
                if kind == "progress":
                    stage, current, progress = payload
                    self._set_job_display(stage=stage, current=current, progress=progress)
                elif kind == "import_chunk":
                    items = payload
                    if items:
                        self.session.add_images(items)
                elif kind == "done":
                    on_done, result, name = payload
                    try:
                        on_done(result)
                    finally:
                        self._set_job_display(stage=f"{name}: done", current="Completed", progress=100)
                elif kind == "error":
                    name, text = payload
                    if "Cancelled by user." in text:
                        self._set_job_display(stage=f"{name}: cancelled", current=text, progress=0)
                        self._set_status(f"{name} cancelled")
                        if name == "Import":
                            self.refresh_page_list(keep_index=len(self.session) - 1)
                    else:
                        self._set_job_display(stage=f"{name}: error", current=text, progress=0)
                        self._set_status(f"{name} failed")
                        messagebox.showerror(f"{name} Error", text)
        except queue.Empty:
            pass
        finally:
            self.after(40, self._poll_job_queue)

    def cancel_current_job(self) -> None:
        if self.job_thread is None or not self.job_thread.is_alive():
            self._set_status("No running job.")
            return
        self.job_cancel_event.set()
        self._set_job_display(current="Cancellation requested...")
        self._set_status("Cancellation requested")

    def _max_camera_resolution(self) -> tuple[int, int]:
        best = RESOLUTIONS[0]
        match = re.match(r"^(\d+)x(\d+)$", best.strip())
        if match is None:
            return (3264, 2448)
        return (int(match.group(1)), int(match.group(2)))

    def _ensure_camera(self) -> CameraService:
        index = int(self.camera_index_var.get())
        resolution = self._max_camera_resolution()
        if self.camera is None:
            self.camera = CameraService(index=index, resolution=resolution)
            self.camera.open()
        elif self.camera.index != index:
            self.camera.set_index(index)
            self.camera.set_resolution(resolution)
        elif self.camera.read_frame() is None:
            self.camera.open()
            self.camera.set_resolution(resolution)
        else:
            self.camera.set_resolution(resolution)
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
        if self.preview_job is None:
            self._preview_loop()
        self._update_camera_health()
        self._set_status("Preview started")

    def stop_preview(self) -> None:
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        self._update_camera_health()
        self._set_status("Preview stopped")

    def _preview_loop(self) -> None:
        if self.camera is None:
            self.preview_job = None
            self._update_camera_health()
            return
        frame = self.camera.read_frame()
        if frame is not None:
            preview = self._preview_image_with_contour(frame)
            self._show_in_preview(preview)
        self.preview_job = self.after(PREVIEW_WAIT_MS, self._preview_loop)

    def _preview_image_with_contour(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def _current_preprocess_settings(self) -> PreprocessSettings:
        preset_name = self.preprocess_preset_var.get()
        preset = PREPROCESS_PRESETS.get(preset_name, PREPROCESS_PRESETS["Custom"])
        apply_threshold = bool(preset.apply_threshold or self.postprocess_var.get() == "Black and White")
        return PreprocessSettings(
            contrast=float(self.preprocess_contrast_var.get()),
            brightness=int(self.preprocess_brightness_var.get()),
            denoise=int(self.preprocess_denoise_var.get()),
            threshold=int(self.preprocess_threshold_var.get()),
            apply_threshold=apply_threshold,
        )

    def _apply_postprocess(self, image: np.ndarray) -> np.ndarray:
        mode = self.postprocess_var.get()
        fn = POSTPROCESSING_OPTIONS.get(mode, POSTPROCESSING_OPTIONS["None"])
        out = fn(image)
        return apply_enhancements(out, self._current_preprocess_settings())

    def _review_before_image(self, entry) -> np.ndarray:
        return entry.preview_original_image if self.lightweight_preview_var.get() else entry.original_image

    def _review_after_image(self, entry, before_image: np.ndarray) -> np.ndarray:
        if self.lightweight_preview_var.get():
            return self._apply_postprocess(before_image)
        return self._apply_postprocess(entry.original_image)

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
    ) -> list[tuple[str, np.ndarray]]:
        return [(base_name, frame)]

    def capture_one(self) -> None:
        try:
            camera = self._ensure_camera()
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("Could not capture an image from the camera.")
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
            items = self._process_capture_frame(frame, base_name=timestamp)
            self.session.add_images(items)
            self.refresh_page_list(keep_index=len(self.session) - 1)
            self.go_to_review_tab()
            self._set_status(f"Captured {len(items)} raw page(s). Session pages: {len(self.session)}")
        except Exception as exc:
            messagebox.showerror("Capture Error", str(exc))
            self._set_status("Capture failed")

    def capture_burst(self) -> None:
        try:
            shots = int(self.camera_shots_var.get())
            delay_sec = float(self.camera_delay_var.get())
            index = int(self.camera_index_var.get())
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

                items: list[tuple[str, np.ndarray]] = []
                total_frames = len(frames)
                for idx, frame in enumerate(frames, start=1):
                    if is_cancelled():
                        raise RuntimeError("Cancelled by user.")
                    current_items = self._process_capture_frame(
                        frame,
                        base_name=f"{timestamp}_{idx:03d}",
                    )
                    items.extend(current_items)
                    emit(
                        stage="Processing burst",
                        current=f"Frame {idx}/{total_frames}",
                        progress=45 + int((idx / total_frames) * 55),
                    )
                return items

            def on_done(items):
                self.session.add_images(items)
                self.refresh_page_list(keep_index=len(self.session) - 1)
                self.go_to_review_tab()
                self._set_status(f"Burst captured {len(items)} raw page(s). Session pages: {len(self.session)}")

            self._start_background_job("Capture Burst", worker, on_done)
        except Exception as exc:
            messagebox.showerror("Burst Error", str(exc))
            self._set_status("Burst capture failed")

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
                messagebox.showerror("Resolution Error", "Resolution must be on the form <width>x<height>.")
                return
            resolution = (int(match.group(1)), int(match.group(2)))
            if self.camera is None:
                self.camera = CameraService(index=int(self.camera_index_var.get()), resolution=resolution)
                self.camera.open()
            else:
                self.camera.set_resolution(resolution)
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
        preset_menu = ctk.CTkOptionMenu(
            window,
            values=RESOLUTIONS,
            variable=preset_var,
            command=_set_resolution,
        )
        preset_menu.pack(fill=ctk.X, padx=12, pady=(0, 8))

        ctk.CTkLabel(window, text="Custom resolution").pack(anchor="w", padx=12, pady=(0, 4))
        custom_var = tk.StringVar(value=RESOLUTIONS[-1])
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
        ctk.CTkButton(row_folder, text="Folder...", width=100, command=self.choose_import_folder).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkButton(row_folder, text="Import Folder", width=130, command=self.import_from_folder).grid(
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
        ctk.CTkButton(row_files, text="Files... (multi)", width=110, command=self.choose_import_files).grid(
            row=0,
            column=1,
            padx=(0, 6),
            pady=10,
        )
        ctk.CTkButton(row_files, text="Import Files", width=130, command=self.import_from_files).grid(
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
        ctk.CTkLabel(
            row_options,
            text="Import loads raw pages only. Open Review for all processing.",
            anchor="w",
        ).pack(side=ctk.LEFT, padx=(0, 10), pady=10)

        row_actions = ctk.CTkFrame(tab)
        row_actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkButton(row_actions, text="Import from Listed Paths", command=self.import_from_files).pack(
            side=ctk.LEFT,
            padx=10,
            pady=10,
        )
        ctk.CTkButton(row_actions, text="Review", command=self.go_to_review_tab).pack(side=ctk.LEFT, padx=0, pady=10)

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
        ctk.CTkButton(row_pdf, text="Save PDF...", width=120, command=self.choose_export_pdf_path).grid(
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
        ctk.CTkButton(row_files, text="Dir...", width=80, command=self.choose_export_directory).grid(
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
        parts = [part.strip().strip('"') for part in re.split(r"[;\n\r]+", raw_text) if part.strip()]
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
                raise RuntimeError("Some selected files do not exist:\n" + "\n".join(map(str, missing)))
            unsupported = [path for path in paths if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS)]
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
        self._set_status(f"Starting import for {len(paths)} file(s)...")

        def worker(emit, is_cancelled):
            emit(stage="Import", current=f"{len(paths)} input file(s)", progress=0)
            total_paths = len(paths)
            added_pages = 0

            for file_index, path in enumerate(paths, start=1):
                if is_cancelled():
                    raise RuntimeError("Cancelled by user.")

                emit(
                    stage="Import loading",
                    current=f"{file_index}/{total_paths}: {path.name}",
                    progress=int(((file_index - 1) / total_paths) * 45),
                )
                loaded = load_input_items(
                    [path],
                    pdf_dpi=pdf_dpi,
                    cancel_cb=is_cancelled,
                )
                items_chunk: list[tuple[str, np.ndarray]] = []
                chunk_size = 4
                for name, page in loaded:
                    items_chunk.append((name, page))
                    added_pages += 1
                    if len(items_chunk) >= chunk_size:
                        self.job_queue.put(("import_chunk", items_chunk))
                        items_chunk = []

                if items_chunk:
                    self.job_queue.put(("import_chunk", items_chunk))

                emit(
                    stage="Import ingest",
                    current=f"{file_index}/{total_paths}: {path.name}",
                    progress=45 + int((file_index / total_paths) * 55),
                )

            return {"files": total_paths, "pages": added_pages}

        def on_done(stats):
            files_count = int(stats["files"])
            pages_count = int(stats["pages"])
            self.refresh_page_list(keep_index=len(self.session) - 1)
            self.go_to_review_tab()
            self._set_status(
                f"Imported {files_count} file(s), added {pages_count} raw page(s). Session pages: {len(self.session)}"
            )

        self._start_background_job("Import", worker, on_done)

    def refresh_page_list(self, keep_index: int | None = None) -> None:
        self.page_listbox.delete(0, tk.END)
        for idx, entry in enumerate(self.session.entries, start=1):
            self.page_listbox.insert(tk.END, f"{idx:04d}  {entry.name}")

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

    def update_page_preview(self) -> None:
        selected = self.page_listbox.curselection()
        if len(selected) != 1:
            self.page_preview_before_label.configure(image=None, text="Select one page to preview")
            self.page_preview_after_label.configure(image=None, text="Select one page to preview")
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            return

        index = selected[0]
        if index < 0 or index >= len(self.session.entries):
            self.page_preview_before_label.configure(image=None, text="Select one page to preview")
            self.page_preview_after_label.configure(image=None, text="Select one page to preview")
            self.page_preview_before_photo = None
            self.page_preview_after_photo = None
            return

        entry = self.session.entries[index]
        before = self._review_before_image(entry)
        try:
            after = self._review_after_image(entry, before)
        except Exception:
            after = entry.preview_current_image if self.lightweight_preview_var.get() else entry.current_image

        before_photo = self._to_ctk_photo_for_label(before, self.page_preview_before_label)
        after_photo = self._to_ctk_photo_for_label(after, self.page_preview_after_label)

        self.page_preview_before_label.configure(image=before_photo, text="")
        self.page_preview_after_label.configure(image=after_photo, text="")
        self.page_preview_before_photo = before_photo
        self.page_preview_after_photo = after_photo

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

            ok = self.session.replace_entry_image(
                entry.entry_id,
                original_image=image,
                current_image=image,
                name=image_path.name,
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
            items = self._process_capture_frame(frame, base_name=item_name)
            _, image = items[0]
            ok = self.session.replace_entry_image(
                entry.entry_id,
                original_image=image,
                current_image=image,
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

        entries = [self.session.entries[idx] for idx in indices if 0 <= idx < len(self.session.entries)]
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
            text="Browse pages, adjust corners, then apply changes to the current page or all loaded pages.",
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
                entries[0].preview_original_image if self.lightweight_preview_var.get() else entries[0].original_image
            ),
        }

        def _map_display_points_to_source(points: np.ndarray, source_shape: tuple[int, int], display_shape: tuple[int, int]) -> np.ndarray:
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
            return entry.preview_original_image if self.lightweight_preview_var.get() else entry.original_image

        def _init_points_for(entry) -> np.ndarray:
            cached = points_by_entry.get(entry.entry_id)
            if cached is not None:
                return cached

            display_image = _display_image_for(entry)
            detected: np.ndarray | None = None
            if auto_detect:
                detected = self._detect_corner_points(display_image)
            points = detected if detected is not None else self._default_corner_points(display_image)
            source_shape = entry.original_image.shape[:2]
            display_shape = display_image.shape[:2]
            points = _map_display_points_to_source(points, source_shape, display_shape)
            points_by_entry[entry.entry_id] = points
            return points

        def _redraw() -> None:
            canvas.delete("overlay")
            points = view_state["points"]
            scale_x = float(view_state["scale_x"])
            scale_y = float(view_state["scale_y"])
            display_w = int(view_state["display_shape"][1]) if view_state["display_shape"] is not None else 1
            display_h = int(view_state["display_shape"][0]) if view_state["display_shape"] is not None else 1
            line_points = []
            for pt in points:
                x = float(pt[0]) / max(scale_x, 1e-6)
                y = float(pt[1]) / max(scale_y, 1e-6)
                line_points.extend([x, y])
            canvas.create_line(*line_points, line_points[0], line_points[1], fill="#00ff66", width=2, tags="overlay")
            for idx_p, pt in enumerate(points):
                sx = float(pt[0]) / max(scale_x, 1e-6)
                sy = float(pt[1]) / max(scale_y, 1e-6)
                if 0 <= sx <= display_w and 0 <= sy <= display_h:
                    r = 7
                    canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="#ff3355", outline="", tags="overlay")
                    canvas.create_text(sx + 14, sy - 10, text=labels[idx_p], fill="#ffffff", tags="overlay")

        def _load_current_entry() -> None:
            entry_index, entry = _current_entry()
            source_image = entry.original_image
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")

            display_image = _display_image_for(entry)
            display_h, display_w = display_image.shape[:2]
            source_h, source_w = source_image.shape[:2]
            view_h = max(1, int(display_h))
            view_w = max(1, int(display_w))
            rgb = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB) if len(display_image.shape) == 2 else cv2.cvtColor(
                display_image, cv2.COLOR_BGR2RGB
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
            points[:] = self._default_corner_points(np.zeros((source_h, source_w, 3), dtype=np.uint8))
            _redraw()

        def _auto_detect_current():
            entry_index, entry = _current_entry()
            display_image = _display_image_for(entry)
            detected = self._detect_corner_points(display_image)
            if detected is None:
                messagebox.showwarning("Auto Crop", f"Document boundaries were not detected for {entry.name}.")
                return
            mapped = _map_display_points_to_source(detected, entry.original_image.shape[:2], display_image.shape[:2])
            points_by_entry[entry.entry_id] = mapped
            view_state["points"] = mapped
            _redraw()

        def _apply_entry(entry_index: int, entry, points: np.ndarray) -> None:
            source_image = entry.original_image
            if source_image is None or source_image.size == 0:
                raise RuntimeError(f"Selected page is empty: {entry.name}")
            warped = warp_perspective_from_points(source_image, points.astype(np.float32))
            if warped is None or warped.size == 0:
                raise RuntimeError("Perspective transform returned empty image.")
            entry.original_image = warped
            entry.current_image = self._apply_postprocess(warped)

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
                        current_display = _display_image_for(entry.original_image)
                        detected = self._detect_corner_points(current_display)
                        points = detected if detected is not None else self._default_corner_points(current_display)
                        points = _map_display_points_to_source(
                            points,
                            entry.original_image.shape[:2],
                            current_display.shape[:2],
                        )
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
        ctk.CTkButton(controls, text="Next", width=90, command=_next_page).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(controls, text="Auto Detect", width=110, command=_auto_detect_current).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(controls, text="Reset", width=90, command=_reset).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(controls, text="Apply Current", width=120, command=_apply_current).pack(
            side=ctk.LEFT,
            padx=6,
        )
        ctk.CTkButton(controls, text="Apply All", width=100, command=_apply_all).pack(side=ctk.LEFT, padx=6)
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

    def _reprocess_entry_from_original(self, entry) -> None:
        postprocess_fn = POSTPROCESSING_OPTIONS.get(self.postprocess_var.get(), POSTPROCESSING_OPTIONS["None"])
        settings = self._current_preprocess_settings()
        base = postprocess_fn(entry.original_image)
        entry.current_image = apply_enhancements(base, settings)

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

    def auto_deskew_selected(self) -> None:
        indices = self._selected_entry_indices()
        if not indices:
            self._set_status("Select page(s) to deskew.")
            return

        angles: list[float] = []
        for idx in indices:
            entry = self.session.entries[idx]
            deskewed, angle = deskew_document(entry.original_image)
            entry.original_image = deskewed
            self._reprocess_entry_from_original(entry)
            angles.append(angle)

        self.refresh_page_list(keep_index=indices[-1])
        mean_angle = sum(angles) / max(1, len(angles))
        self._set_status(f"Deskewed {len(indices)} page(s), avg angle {mean_angle:.1f} deg.")

    def _on_review_processing_slider_change(self, _value: float) -> None:
        self.update_page_preview()

    def open_review_processing_dialog(self) -> None:
        if self.review_processing_window is not None and self.review_processing_window.winfo_exists():
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

        ctk.CTkLabel(body, text="B/W Threshold").grid(row=3, column=0, sticky="w", padx=8, pady=(4, 8))
        ctk.CTkSlider(
            body,
            from_=80,
            to=240,
            number_of_steps=160,
            variable=self.preprocess_threshold_var,
            command=self._on_review_processing_slider_change,
        ).grid(row=3, column=1, sticky="ew", padx=8, pady=(4, 8))

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
        ctk.CTkButton(actions, text="Close", command=_on_close, width=100).pack(side=ctk.LEFT, padx=8)

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
            for _idx, entry in target_entries:
                self._reprocess_entry_from_original(entry)
            self.refresh_page_list(keep_index=target_entries[-1][0])
            scope = "all pages" if self.apply_changes_to_all_var.get() else "selected pages"
            self._set_status(f"Reprocessed {len(target_entries)} {scope}.")
        except Exception as exc:
            messagebox.showerror("Postprocess Error", str(exc))

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
            image_paths = [entry.current_path for entry in entries]

            def worker(emit, _is_cancelled):
                emit(stage="Export PDF", current=f"Writing {len(image_paths)} page(s)", progress=10)
                out_path = export_image_paths_as_pdf(image_paths, out_pdf=Path(path_raw), dpi=dpi)
                emit(stage="Export PDF", current="Finalizing", progress=100)
                return out_path

            def on_done(out_path):
                self._set_status(f"Exported {len(image_paths)} page(s) to PDF: {out_path}")

            self._start_background_job("Export PDF", worker, on_done)
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
            image_paths = [entry.current_path for entry in entries]

            def worker(emit, _is_cancelled):
                emit(stage="Export files", current=f"Writing {len(image_paths)} page(s)", progress=10)
                out_paths = export_image_paths_as_files(
                    image_paths,
                    output_dir=Path(path_raw),
                    ext=fmt,
                    base_name="page",
                )
                emit(stage="Export files", current="Finalizing", progress=100)
                return out_paths

            def on_done(out_paths):
                self._set_status(f"Exported {len(out_paths)} file(s) to: {Path(path_raw)}")

            self._start_background_job("Export files", worker, on_done)
        except Exception as exc:
            messagebox.showerror("Export Files Error", str(exc))
            self._set_status("Files export failed")


def run_app() -> int:
    app = UnifiedScanApp()
    app.mainloop()
    return 0

