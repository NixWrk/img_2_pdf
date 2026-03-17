"""Unified application shell."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import filedialog, messagebox

from uniscan.core.pipeline import PipelineOptions, process_loaded_items, split_spread
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.scanner_adapter import ScanAdapterError, scan_with_document_detector
from uniscan.io import CameraService
from uniscan.io.loaders import IMG_EXTS, PDF_EXTS, list_supported_in_folder, load_input_items
from uniscan.session import CaptureSession

PREVIEW_WAIT_MS = 80
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

        self.project_root = Path(__file__).resolve().parents[3]
        candidate_scanner_root = self.project_root / "camscan_suhren"
        self.scanner_root = candidate_scanner_root if candidate_scanner_root.exists() else None

        self.session = CaptureSession()
        self.camera: CameraService | None = None
        self.preview_job: str | None = None
        self.preview_photo: ctk.CTkImage | None = None

        self.status_var = tk.StringVar(value="Ready")
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_shots_var = tk.IntVar(value=1)
        self.camera_delay_var = tk.DoubleVar(value=1.0)
        self.detect_document_var = tk.BooleanVar(value=True)
        self.two_page_mode_var = tk.BooleanVar(value=False)
        self.free_capture_var = tk.BooleanVar(value=False)
        self.postprocess_var = tk.StringVar(value="None")
        self.import_folder_var = tk.StringVar()
        self.import_files_var = tk.StringVar()
        self.import_pdf_dpi_var = tk.IntVar(value=300)

        self._build_ui()
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

        self.tabs = ctk.CTkTabview(container)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        self.capture_tab = self.tabs.add("Capture")
        self.import_tab = self.tabs.add("Import")
        self.pages_tab = self.tabs.add("Pages")
        self.export_tab = self.tabs.add("Export")
        self.jobs_tab = self.tabs.add("Jobs")

        self._build_capture_tab(self.capture_tab)
        self._build_import_tab(self.import_tab)
        for tab, name in ((self.pages_tab, "Pages"), (self.export_tab, "Export"), (self.jobs_tab, "Jobs")):
            body = ctk.CTkLabel(
                tab,
                text=f"{name} module: implementation in progress",
                anchor="w",
            )
            body.pack(fill=ctk.X, padx=16, pady=16)

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

        ctk.CTkButton(controls, text="Configure Camera", command=self.configure_camera_event).pack(
            fill=ctk.X,
            padx=10,
            pady=(6, 8),
        )

        ctk.CTkCheckBox(
            controls,
            text="Detect document contour",
            variable=self.detect_document_var,
        ).pack(anchor="w", padx=10, pady=(2, 2))
        ctk.CTkCheckBox(
            controls,
            text="Two-page split",
            variable=self.two_page_mode_var,
        ).pack(anchor="w", padx=10, pady=(2, 2))
        ctk.CTkCheckBox(
            controls,
            text="Free capture mode",
            variable=self.free_capture_var,
        ).pack(anchor="w", padx=10, pady=(2, 8))

        ctk.CTkLabel(controls, text="Postprocess").pack(anchor="w", padx=10, pady=(4, 2))
        self.postprocess_menu = ctk.CTkOptionMenu(
            controls,
            values=list(POSTPROCESSING_OPTIONS.keys()),
            variable=self.postprocess_var,
        )
        self.postprocess_menu.pack(anchor="w", padx=10, pady=(0, 8))

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

    def _on_close(self) -> None:
        self.stop_preview()
        if self.camera is not None:
            self.camera.release()
        self.destroy()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _ensure_camera(self) -> CameraService:
        index = int(self.camera_index_var.get())
        if self.camera is None:
            self.camera = CameraService(index=index)
            self.camera.open()
        elif self.camera.index != index:
            self.camera.set_index(index)
        elif self.camera.read_frame() is None:
            self.camera.open()
        return self.camera

    def open_camera(self) -> None:
        try:
            self._ensure_camera()
            self._set_status(f"Camera opened (index {self.camera_index_var.get()})")
        except Exception as exc:
            messagebox.showerror("Camera Error", str(exc))
            self._set_status("Camera open failed")

    def close_camera(self) -> None:
        self.stop_preview()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self._set_status("Camera closed")

    def start_preview(self) -> None:
        try:
            self._ensure_camera()
        except Exception as exc:
            messagebox.showerror("Camera Error", str(exc))
            return
        if self.preview_job is None:
            self._preview_loop()
        self._set_status("Preview started")

    def stop_preview(self) -> None:
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        self._set_status("Preview stopped")

    def _preview_loop(self) -> None:
        if self.camera is None:
            self.preview_job = None
            return
        frame = self.camera.read_frame()
        if frame is not None:
            preview = self._preview_image_with_contour(frame)
            self._show_in_preview(preview)
        self.preview_job = self.after(PREVIEW_WAIT_MS, self._preview_loop)

    def _preview_image_with_contour(self, frame: np.ndarray) -> np.ndarray:
        image = self._apply_postprocess(frame)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.detect_document_var.get() and not self.free_capture_var.get():
            try:
                scan_output = scan_with_document_detector(
                    frame,
                    enabled=True,
                    scanner_root=self.scanner_root,
                )
            except ScanAdapterError:
                return image
            contour = scan_output.contour
            if contour is not None:
                image = self._draw_contour(image, contour)
        return image

    def _draw_contour(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        out = image.copy()
        points = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        return out

    def _apply_postprocess(self, image: np.ndarray) -> np.ndarray:
        mode = self.postprocess_var.get()
        fn = POSTPROCESSING_OPTIONS.get(mode, POSTPROCESSING_OPTIONS["None"])
        return fn(image)

    def _show_in_preview(self, image: np.ndarray) -> None:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_width = max(200, self.preview_label.winfo_width())
        max_height = max(120, self.preview_label.winfo_height())
        h, w = rgb.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(resized)
        self.preview_photo = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))
        self.preview_label.configure(image=self.preview_photo, text="")

    def _process_capture_frame(self, frame: np.ndarray, base_name: str) -> list[tuple[str, np.ndarray]]:
        use_detector = self.detect_document_var.get() and not self.free_capture_var.get()
        working = frame
        if use_detector:
            scan_output = scan_with_document_detector(
                frame,
                enabled=True,
                scanner_root=self.scanner_root,
            )
            if scan_output.warped is None:
                raise RuntimeError(
                    "Could not extract the document image from camera. Enable 'Free capture mode' to keep full frame."
                )
            working = scan_output.warped

        processed = self._apply_postprocess(working)
        pages = split_spread(processed) if self.two_page_mode_var.get() else [processed]
        if len(pages) == 1:
            return [(base_name, pages[0])]
        return [(f"{base_name}_{idx}", page) for idx, page in enumerate(pages, start=1)]

    def capture_one(self) -> None:
        try:
            camera = self._ensure_camera()
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("Could not capture an image from the camera.")
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S_%f")
            items = self._process_capture_frame(frame, base_name=timestamp)
            self.session.add_images(items)
            self._set_status(f"Captured {len(items)} page(s). Session pages: {len(self.session)}")
        except Exception as exc:
            messagebox.showerror("Capture Error", str(exc))
            self._set_status("Capture failed")

    def capture_burst(self) -> None:
        try:
            camera = self._ensure_camera()
            shots = int(self.camera_shots_var.get())
            delay_sec = float(self.camera_delay_var.get())
            frames = camera.capture_burst(shots=shots, delay_sec=delay_sec)
            added_pages = 0
            timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
            for idx, frame in enumerate(frames, start=1):
                items = self._process_capture_frame(frame, base_name=f"{timestamp}_{idx:03d}")
                self.session.add_images(items)
                added_pages += len(items)
            self._set_status(f"Burst captured {added_pages} page(s). Session pages: {len(self.session)}")
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
        ctk.CTkButton(row_files, text="Files...", width=100, command=self.choose_import_files).grid(
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
            text="Uses current processing options from Capture tab (detect/split/postprocess)",
            anchor="w",
        ).pack(side=ctk.LEFT, padx=(0, 10), pady=10)

        row_actions = ctk.CTkFrame(tab)
        row_actions.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkButton(row_actions, text="Import from Session Paths", command=self.import_from_files).pack(
            side=ctk.LEFT,
            padx=10,
            pady=10,
        )

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
                    "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.webp;*.bmp;*.pdf",
                ),
                ("All files", "*.*"),
            ],
        )
        if files:
            self.import_files_var.set(";".join(files))

    def import_from_folder(self) -> None:
        try:
            folder = Path(self.import_folder_var.get().strip())
            paths = list_supported_in_folder(folder)
            if not paths:
                raise RuntimeError("No supported image/PDF files found in selected folder.")
            self._import_paths(paths=paths, source_label=folder.name or "folder")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            self._set_status("Folder import failed")

    def import_from_files(self) -> None:
        try:
            raw = [part.strip().strip('"') for part in self.import_files_var.get().split(";") if part.strip()]
            if not raw:
                raise RuntimeError("No files selected.")
            paths = [Path(item) for item in raw]
            missing = [path for path in paths if not path.exists() or not path.is_file()]
            if missing:
                raise RuntimeError("Some selected files do not exist:\n" + "\n".join(map(str, missing)))
            unsupported = [path for path in paths if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS)]
            if unsupported:
                raise RuntimeError("Unsupported file type(s):\n" + "\n".join(map(str, unsupported)))
            paths.sort(key=lambda p: p.name.lower())
            self._import_paths(paths=paths, source_label="files")
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            self._set_status("File import failed")

    def _import_paths(self, *, paths: list[Path], source_label: str) -> None:
        pdf_dpi = int(self.import_pdf_dpi_var.get())
        if pdf_dpi < 72:
            raise RuntimeError("PDF DPI must be >= 72.")
        self._set_status(f"Loading {len(paths)} input file(s)...")

        loaded = load_input_items(
            paths,
            pdf_dpi=pdf_dpi,
            on_progress=lambda i, total, name: self._set_status(f"Loading {i}/{total}: {name}"),
        )
        pipeline_options = PipelineOptions(
            detect_document=bool(self.detect_document_var.get()),
            two_page_mode=bool(self.two_page_mode_var.get()),
            postprocess_name=self.postprocess_var.get(),
        )
        pages = process_loaded_items(
            loaded,
            options=pipeline_options,
            scanner_root=self.scanner_root,
        )
        items = [(f"{source_label}_{idx:05d}", page) for idx, page in enumerate(pages, start=1)]
        self.session.add_images(items)
        self._set_status(
            f"Imported {len(paths)} file(s), added {len(items)} page(s). Session pages: {len(self.session)}"
        )


def run_app() -> int:
    app = UnifiedScanApp()
    app.mainloop()
    return 0
