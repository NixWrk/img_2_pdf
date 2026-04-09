#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import img2pdf
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PDF_EXTS = {".pdf"}
ASCII_TMP_ROOT = Path(os.environ.get("SystemDrive", "C:")) / "_uniscan_tmp"
PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "camscan_suhren"

PROFILE_SETTINGS = {
    "Fast": {"dpi": 220},
    "Balanced": {"dpi": 300},
    "Best quality": {"dpi": 400},
}


if THIRD_PARTY_ROOT.exists():
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

SCAN_IMPORT_ERROR = None
SCANNER_MODULE = None
POSTPROCESS_MODULE = None
try:
    from camscan import scanner as _scanner  # type: ignore
    from camscan import postprocessing as _postprocessing  # type: ignore

    SCANNER_MODULE = _scanner
    POSTPROCESS_MODULE = _postprocessing
except Exception as e:  # pragma: no cover - handled in GUI runtime
    SCAN_IMPORT_ERROR = e


@contextlib.contextmanager
def ascii_tempdir(prefix: str = "camscan_hybrid_"):
    ASCII_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=str(ASCII_TMP_ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray):
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def split_spread(image: np.ndarray):
    h, w = image.shape[:2]
    if w < 2:
        return [image]
    mid = w // 2
    left = image[:, :mid]
    right = image[:, mid:]
    if left.size == 0 or right.size == 0:
        return [image]
    return [left, right]


def build_pdf_from_images(image_paths: list[Path], out_pdf: Path, dpi: int):
    with out_pdf.open("wb") as f:
        try:
            payload = img2pdf.convert([str(p) for p in image_paths], dpi=dpi)
        except TypeError:
            layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
            payload = img2pdf.convert([str(p) for p in image_paths], layout_fun=layout)
        f.write(payload)


def render_pdf_pages(pdf_path: Path, dpi: int):
    try:
        import fitz  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF import requires PyMuPDF. Install with: pip install pymupdf"
        ) from e

    pages = []
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append((f"{pdf_path.stem}_p{i:04d}.png", arr))
    finally:
        doc.close()

    return pages


class App(tk.Tk):
    POLL_MS = 120

    def __init__(self):
        super().__init__()
        self.title("CamScan Hybrid Tool (camera + files/folder -> PDF)")
        self.geometry("980x620")
        self.minsize(900, 580)

        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        self.mode = tk.StringVar(value="folder")
        self.input_folder = tk.StringVar()
        self.input_files = tk.StringVar()
        self.out_pdf = tk.StringVar()

        self.profile = tk.StringVar(value="Balanced")
        self.detect_document = tk.BooleanVar(value=True)
        self.two_page_mode = tk.BooleanVar(value=False)
        self.postprocess = tk.StringVar(value="None")

        self.camera_index = tk.IntVar(value=0)
        self.camera_shots = tk.IntVar(value=1)
        self.camera_delay = tk.DoubleVar(value=1.0)

        self.stage = tk.StringVar(value="Idle")
        self.current = tk.StringVar(value="-")
        self.percent = tk.StringVar(value="0%")

        self._build_ui()
        self._apply_mode_ui()
        self.after(self.POLL_MS, self._poll_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        mode_box = ttk.LabelFrame(root, text="Source")
        mode_box.pack(fill="x", **pad)
        row_mode = ttk.Frame(mode_box)
        row_mode.pack(fill="x", padx=10, pady=8)
        ttk.Radiobutton(
            row_mode,
            text="Import folder",
            variable=self.mode,
            value="folder",
            command=self._apply_mode_ui,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            row_mode,
            text="Import files",
            variable=self.mode,
            value="files",
            command=self._apply_mode_ui,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            row_mode,
            text="Camera capture",
            variable=self.mode,
            value="camera",
            command=self._apply_mode_ui,
        ).pack(side="left", padx=8)

        row_folder = ttk.Frame(root)
        row_folder.pack(fill="x", **pad)
        ttk.Label(row_folder, text="Folder:").pack(side="left")
        self.folder_entry = ttk.Entry(row_folder, textvariable=self.input_folder)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.folder_btn = ttk.Button(row_folder, text="Choose...", command=self.choose_folder)
        self.folder_btn.pack(side="left")

        row_files = ttk.Frame(root)
        row_files.pack(fill="x", **pad)
        ttk.Label(row_files, text="Files:").pack(side="left")
        self.files_entry = ttk.Entry(row_files, textvariable=self.input_files)
        self.files_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.files_btn = ttk.Button(row_files, text="Choose...", command=self.choose_files)
        self.files_btn.pack(side="left")

        row_out = ttk.Frame(root)
        row_out.pack(fill="x", **pad)
        ttk.Label(row_out, text="Output PDF:").pack(side="left")
        ttk.Entry(row_out, textvariable=self.out_pdf).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row_out, text="Save as...", command=self.choose_out_pdf).pack(side="left")

        settings = ttk.LabelFrame(root, text="Processing")
        settings.pack(fill="x", **pad)

        row1 = ttk.Frame(settings)
        row1.pack(fill="x", padx=10, pady=6)
        ttk.Label(row1, text="Profile:").pack(side="left")
        ttk.Combobox(
            row1,
            values=list(PROFILE_SETTINGS.keys()),
            textvariable=self.profile,
            width=14,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Checkbutton(row1, text="Detect document contour", variable=self.detect_document).pack(side="left", padx=10)
        ttk.Checkbutton(row1, text="Two-page split", variable=self.two_page_mode).pack(side="left", padx=10)

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row2, text="Postprocess:").pack(side="left")
        ttk.Combobox(
            row2,
            values=["None", "Sharpen", "Grayscale", "Black and White"],
            textvariable=self.postprocess,
            width=18,
            state="readonly",
        ).pack(side="left", padx=8)

        cam = ttk.LabelFrame(root, text="Camera options")
        cam.pack(fill="x", **pad)
        cam_row = ttk.Frame(cam)
        cam_row.pack(fill="x", padx=10, pady=6)
        ttk.Label(cam_row, text="Camera index:").pack(side="left")
        self.camera_index_spin = ttk.Spinbox(cam_row, from_=0, to=9, width=6, textvariable=self.camera_index)
        self.camera_index_spin.pack(side="left", padx=8)
        ttk.Label(cam_row, text="Shots:").pack(side="left")
        self.camera_shots_spin = ttk.Spinbox(cam_row, from_=1, to=200, width=8, textvariable=self.camera_shots)
        self.camera_shots_spin.pack(side="left", padx=8)
        ttk.Label(cam_row, text="Delay between shots (sec):").pack(side="left")
        self.camera_delay_spin = ttk.Spinbox(
            cam_row, from_=0.0, to=30.0, increment=0.5, width=8, textvariable=self.camera_delay
        )
        self.camera_delay_spin.pack(side="left", padx=8)

        prog = ttk.LabelFrame(root, text="Progress")
        prog.pack(fill="x", **pad)
        p1 = ttk.Frame(prog)
        p1.pack(fill="x", padx=10, pady=6)
        ttk.Label(p1, text="Stage:").pack(side="left")
        ttk.Label(p1, textvariable=self.stage).pack(side="left", padx=8)
        p2 = ttk.Frame(prog)
        p2.pack(fill="x", padx=10, pady=6)
        ttk.Label(p2, text="Current:").pack(side="left")
        ttk.Label(p2, textvariable=self.current).pack(side="left", padx=8)
        p3 = ttk.Frame(prog)
        p3.pack(fill="x", padx=10, pady=6)
        self.bar = ttk.Progressbar(p3, orient="horizontal", mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        ttk.Label(p3, textvariable=self.percent, width=8, anchor="e").pack(side="left", padx=8)

        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

    def _apply_mode_ui(self):
        mode = self.mode.get()
        folder_state = "normal" if mode == "folder" else "disabled"
        files_state = "normal" if mode == "files" else "disabled"
        cam_state = "normal" if mode == "camera" else "disabled"

        self.folder_entry.configure(state=folder_state)
        self.folder_btn.configure(state=folder_state)
        self.files_entry.configure(state=files_state)
        self.files_btn.configure(state=files_state)

        self.camera_index_spin.configure(state=cam_state)
        self.camera_shots_spin.configure(state=cam_state)
        self.camera_delay_spin.configure(state=cam_state)

    def choose_folder(self):
        d = filedialog.askdirectory(title="Select input folder")
        if d:
            self.input_folder.set(d)

    def choose_files(self):
        files = filedialog.askopenfilenames(
            title="Select image/PDF files",
            filetypes=[
                ("Image and PDF", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.webp;*.bmp;*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if files:
            self.input_files.set(";".join(files))

    def choose_out_pdf(self):
        f = filedialog.asksaveasfilename(
            title="Save PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if f:
            self.out_pdf.set(f)

    def set_busy(self, busy: bool):
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    def _emit(self, kind: str, payload=None):
        self.q.put((kind, payload))

    def _validate(self):
        if SCAN_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Cannot import third-party processing modules from camscan_suhren.\n"
                f"Details: {SCAN_IMPORT_ERROR}"
            )

        out = Path(self.out_pdf.get().strip())
        if not out.name:
            raise RuntimeError("Please choose output PDF path.")
        if out.suffix.lower() != ".pdf":
            out = out.with_suffix(".pdf")
            self.out_pdf.set(str(out))
        out.parent.mkdir(parents=True, exist_ok=True)

        mode = self.mode.get()
        if mode == "folder":
            d = Path(self.input_folder.get().strip())
            if not d.exists() or not d.is_dir():
                raise RuntimeError("Please choose a valid input folder.")
            items = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (IMG_EXTS | PDF_EXTS)]
            items.sort(key=lambda p: natural_key(p.name))
            if not items:
                raise RuntimeError("No supported files found in folder.")
            source = {"mode": "folder", "items": items}
        elif mode == "files":
            raw = [x.strip().strip('"') for x in self.input_files.get().split(";") if x.strip()]
            if not raw:
                raise RuntimeError("Please choose input files.")
            paths = [Path(p) for p in raw]
            bad = [p for p in paths if (not p.exists()) or (not p.is_file())]
            if bad:
                raise RuntimeError("Some selected files do not exist:\n" + "\n".join(map(str, bad)))
            ext_bad = [p for p in paths if p.suffix.lower() not in (IMG_EXTS | PDF_EXTS)]
            if ext_bad:
                raise RuntimeError("Unsupported file type(s):\n" + "\n".join(map(str, ext_bad)))
            paths.sort(key=lambda p: natural_key(p.name))
            source = {"mode": "files", "items": paths}
        else:
            shots = int(self.camera_shots.get())
            delay = float(self.camera_delay.get())
            if shots < 1:
                raise RuntimeError("Shots must be >= 1.")
            if delay < 0:
                raise RuntimeError("Delay must be >= 0.")
            source = {
                "mode": "camera",
                "camera_index": int(self.camera_index.get()),
                "shots": shots,
                "delay": delay,
            }

        profile_name = self.profile.get()
        if profile_name not in PROFILE_SETTINGS:
            raise RuntimeError("Invalid profile selected.")

        return {
            "source": source,
            "out_pdf": out,
            "profile": profile_name,
            "detect_document": bool(self.detect_document.get()),
            "two_page_mode": bool(self.two_page_mode.get()),
            "postprocess_name": self.postprocess.get(),
        }

    def start(self):
        try:
            job = self._validate()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.stop_flag.clear()
        self.set_busy(True)
        self.stage.set("Starting...")
        self.current.set("-")
        self.percent.set("0%")
        self.bar.stop()
        self.bar.configure(mode="determinate")
        self.bar["value"] = 0

        self.worker = threading.Thread(target=self._worker, args=(job,), daemon=True)
        self.worker.start()

    def cancel(self):
        self.stop_flag.set()
        self.stage.set("Cancelling...")

    def _camera_api(self):
        if os.name == "nt":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def _load_inputs(self, source: dict, pdf_dpi: int):
        items = []
        mode = source["mode"]

        if mode in {"folder", "files"}:
            paths = source["items"]
            total = len(paths)
            self._emit("stage", ("Loading inputs...", "determinate"))
            for idx, path in enumerate(paths, start=1):
                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")
                self._emit("current", path.name)
                ext = path.suffix.lower()
                if ext in IMG_EXTS:
                    img = imread_unicode(path)
                    if img is None:
                        raise RuntimeError(f"Cannot read image: {path}")
                    items.append((path.name, img))
                elif ext in PDF_EXTS:
                    items.extend(render_pdf_pages(path, dpi=pdf_dpi))
                else:
                    raise RuntimeError(f"Unsupported input: {path}")
                self._emit("progress", int((idx / total) * 100))
            return items

        cam_idx = source["camera_index"]
        shots = source["shots"]
        delay = source["delay"]

        self._emit("stage", ("Capturing camera images...", "determinate"))
        self._emit("current", f"camera index {cam_idx}")

        cap = cv2.VideoCapture(cam_idx, self._camera_api())
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_idx}.")

        try:
            for i in range(1, shots + 1):
                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")

                for _ in range(4):
                    cap.read()
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to capture frame {i}/{shots}.")
                items.append((f"camera_{i:03d}.jpg", frame))
                self._emit("current", f"shot {i}/{shots}")
                self._emit("progress", int((i / shots) * 100))

                if i < shots and delay > 0:
                    wait_total = int(max(1, delay / 0.1))
                    for _ in range(wait_total):
                        if self.stop_flag.is_set():
                            raise RuntimeError("Cancelled by user.")
                        time.sleep(0.1)
        finally:
            cap.release()

        return items

    def _process_images(
        self,
        loaded_items: list[tuple[str, np.ndarray]],
        detect_document: bool,
        two_page_mode: bool,
        postprocess_name: str,
        out_dir: Path,
    ):
        if POSTPROCESS_MODULE is None or SCANNER_MODULE is None:
            raise RuntimeError("Third-party processing modules are not available.")

        postprocess_map = {
            "None": POSTPROCESS_MODULE.dummy,
            "Sharpen": POSTPROCESS_MODULE.sharpen,
            "Grayscale": POSTPROCESS_MODULE.grayscale,
            "Black and White": POSTPROCESS_MODULE.black_and_white,
        }
        if postprocess_name not in postprocess_map:
            raise RuntimeError(f"Unsupported postprocess mode: {postprocess_name}")
        post_fn = postprocess_map[postprocess_name]

        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        counter = 1
        total = len(loaded_items)

        self._emit("stage", ("Processing pages...", "determinate"))
        for idx, (name, image) in enumerate(loaded_items, start=1):
            if self.stop_flag.is_set():
                raise RuntimeError("Cancelled by user.")

            self._emit("current", name)
            working = image

            if detect_document:
                scan_result = SCANNER_MODULE.main(working)
                if scan_result is not None and getattr(scan_result, "warped", None) is not None:
                    working = scan_result.warped

            processed = post_fn(working)
            pages = split_spread(processed) if two_page_mode else [processed]

            for page in pages:
                out_path = out_dir / f"{counter:05d}.png"
                if not imwrite_unicode(out_path, page):
                    raise RuntimeError(f"Failed writing page: {out_path}")
                output_paths.append(out_path)
                counter += 1

            self._emit("progress", int((idx / total) * 100))

        return output_paths

    def _worker(self, job: dict):
        try:
            profile = PROFILE_SETTINGS[job["profile"]]
            dpi = int(profile["dpi"])
            out_pdf: Path = job["out_pdf"]

            with ascii_tempdir() as tmp:
                loaded = self._load_inputs(job["source"], pdf_dpi=dpi)
                if not loaded:
                    raise RuntimeError("No input pages loaded.")

                prepared_dir = tmp / "prepared_pages"
                pages = self._process_images(
                    loaded_items=loaded,
                    detect_document=bool(job["detect_document"]),
                    two_page_mode=bool(job["two_page_mode"]),
                    postprocess_name=job["postprocess_name"],
                    out_dir=prepared_dir,
                )

                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")

                self._emit("stage", ("Building PDF...", "indeterminate"))
                self._emit("current", f"Writing {len(pages)} page(s)")
                build_pdf_from_images(pages, out_pdf=out_pdf, dpi=dpi)

            self._emit("done", str(out_pdf))
        except Exception as e:
            self._emit("error", str(e))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "stage":
                    text, mode = payload
                    self.stage.set(text)
                    if mode == "indeterminate":
                        self.bar.configure(mode="indeterminate")
                        self.bar.start(10)
                        self.percent.set("...")
                    else:
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                elif kind == "current":
                    self.current.set(payload)
                elif kind == "progress":
                    if str(self.bar.cget("mode")) == "indeterminate":
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                    v = max(0, min(100, int(payload)))
                    self.bar["value"] = v
                    self.percent.set(f"{v}%")
                elif kind == "done":
                    self.bar.stop()
                    self.bar.configure(mode="determinate")
                    self.bar["value"] = 100
                    self.percent.set("100%")
                    self.stage.set("Done")
                    self.current.set(payload)
                    self.set_busy(False)
                    messagebox.showinfo("Done", f"Saved:\n{payload}")
                elif kind == "error":
                    self.bar.stop()
                    self.stage.set("Error")
                    self.set_busy(False)
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()
