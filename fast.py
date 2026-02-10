#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import types
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pypdf import PdfReader, PdfWriter  # still used in PDF mode only

# Practical fix (Pillow): allow loading truncated JPEGs (best-effort)
try:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

ASCII_TMP_ROOT = Path(os.environ.get("SystemDrive", "C:")) / "_ocrmypdf_tmp"


# -------------------------
# Utilities / context mgrs
# -------------------------

@contextlib.contextmanager
def ascii_tempdir(prefix: str = "ocr_"):
    """Temporary dir guaranteed to have ASCII path (e.g. C:\\_ocrmypdf_tmp\\ocr_xxx)."""
    ASCII_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=str(ASCII_TMP_ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def temp_env(**updates):
    """Temporarily set environment variables."""
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def prepend_to_path(dir_path: Path | str | None):
    """Temporarily prepend directory to PATH."""
    if not dir_path:
        yield
        return
    p = str(dir_path)
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = p + os.pathsep + old
    try:
        yield
    finally:
        os.environ["PATH"] = old


def safe_ascii_name(i: int, ext: str) -> str:
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext if ext else ""
    return f"img_{i:05d}{ext}"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_importable(module_name: str, install_hint: str):
    try:
        __import__(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Не могу импортировать пакет '{module_name}'. {install_hint}\n\nDetails: {e}"
        ) from e


def fast_copy_or_link(src: Path, dst: Path) -> None:
    """
    Try hardlink (fast, no extra disk usage) then fallback to copy2.
    Hardlink works only within same volume and if allowed by FS/permissions.
    """
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def which_any(names: list[str]) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def build_images_pdf(img_paths: list[Path], out_pdf: Path) -> None:
    # Requires img2pdf
    import img2pdf
    with out_pdf.open("wb") as f:
        f.write(img2pdf.convert([str(p) for p in img_paths]))


def find_bad_image(paths: list[Path]) -> str | None:
    """
    Practical fix: identify a problematic image by attempting to load via Pillow.
    Returns "filename: error" or None if cannot test / all ok.
    """
    try:
        from PIL import Image
    except Exception:
        return None

    for p in paths:
        try:
            with Image.open(p) as im:
                im.load()  # load/decode; stricter than verify() for many JPEG issues
        except Exception as e:
            return f"{p.name}: {e}"
    return None


# -------------------------
# OCRmyPDF Tk progress plugin (left as-is)
# -------------------------

def build_tk_progress_plugin():
    """
    Creates in-memory OCRmyPDF plugin module that replaces progress bar with Tk callback.
    Returns (module_name, module_object).
    """
    plugin_name = "_ocrmypdf_tk_progress_plugin"
    if plugin_name in sys.modules:
        return plugin_name, sys.modules[plugin_name]

    m = types.ModuleType(plugin_name)
    sys.modules[plugin_name] = m

    code = r"""
from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import ProgressBar

_callback = None

def set_callback(cb):
    global _callback
    _callback = cb

class TkProgressBar(ProgressBar):
    def __init__(self, *, total=None, desc=None, unit=None, disable=False, **kwargs):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.current = 0.0

    def __enter__(self):
        if (not self.disable) and _callback:
            _callback(kind="enter", desc=self.desc, total=self.total, unit=self.unit, current=self.current)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (not self.disable) and _callback:
            _callback(kind="exit", desc=self.desc, total=self.total, unit=self.unit, current=self.current, error=exc_val)
        return False

    def update(self, n=1, *, completed=None):
        if completed is not None:
            self.current = float(completed)
        else:
            self.current += float(n)

        if (not self.disable) and _callback:
            _callback(kind="update", desc=self.desc, total=self.total, unit=self.unit, current=self.current)

@hookimpl
def get_progressbar_class():
    return TkProgressBar
"""
    exec(code, m.__dict__)
    return plugin_name, m


# -------------------------
# GUI App
# -------------------------

class App(tk.Tk):
    UI_TICK_MS = 80  # polling interval for event queue
    PROGRESS_THROTTLE_S = 0.06  # throttle OCR progress updates

    def __init__(self):
        super().__init__()
        self.title("OCRmyPDF GUI (PDF / Images → searchable PDF)")
        self.geometry("900x560")
        self.minsize(800, 500)

        # Mode
        self.mode = tk.StringVar(value="images")  # images | pdf

        # Inputs
        self.in_dir = tk.StringVar()
        self.in_pdf = tk.StringVar()
        self.out_pdf = tk.StringVar()

        # Options
        self.lang = tk.StringVar(value="rus+eng")
        self.image_dpi = tk.IntVar(value=300)
        self.skip_text = tk.BooleanVar(value=True)

        # Optional tesseract.exe (if user wants to force PATH override)
        self.tess_exe_user = tk.StringVar(value="")

        # Progress UI
        self.stage = tk.StringVar(value="Idle")
        self.current = tk.StringVar(value="-")
        self.percent = tk.StringVar(value="0%")
        self._progress_mode = "determinate"

        self._stop = threading.Event()
        self._worker = None

        # Thread-safe UI event queue
        self._events: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._polling = False

        # For throttling progress callback
        self._last_progress_post = 0.0

        # OCRmyPDF Tk progress plugin
        self._ocr_plugin_name, self._ocr_plugin_mod = build_tk_progress_plugin()

        self._build_ui()
        self._apply_mode_ui()

    # ---------- UI ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        mode_box = ttk.LabelFrame(root, text="Mode")
        mode_box.pack(fill="x", **pad)
        mrow = ttk.Frame(mode_box)
        mrow.pack(fill="x", padx=10, pady=8)
        ttk.Radiobutton(
            mrow, text="Images folder → PDF (OCR)",
            variable=self.mode, value="images",
            command=self._apply_mode_ui
        ).pack(side="left", padx=6)
        ttk.Radiobutton(
            mrow, text="PDF file → PDF (OCR)",
            variable=self.mode, value="pdf",
            command=self._apply_mode_ui
        ).pack(side="left", padx=6)

        # Images folder
        r1 = ttk.Frame(root)
        r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Images folder:").pack(side="left")
        self.in_dir_entry = ttk.Entry(r1, textvariable=self.in_dir)
        self.in_dir_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_dir_btn = ttk.Button(r1, text="Choose…", command=self.choose_in_dir)
        self.in_dir_btn.pack(side="left")

        # Input PDF
        r1b = ttk.Frame(root)
        r1b.pack(fill="x", **pad)
        ttk.Label(r1b, text="Input PDF:").pack(side="left")
        self.in_pdf_entry = ttk.Entry(r1b, textvariable=self.in_pdf)
        self.in_pdf_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_pdf_btn = ttk.Button(r1b, text="Choose…", command=self.choose_in_pdf)
        self.in_pdf_btn.pack(side="left")

        # Output
        r2 = ttk.Frame(root)
        r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Save PDF as:").pack(side="left")
        ttk.Entry(r2, textvariable=self.out_pdf).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r2, text="Save as…", command=self.choose_out).pack(side="left")

        # Tesseract override
        r3 = ttk.Frame(root)
        r3.pack(fill="x", **pad)
        ttk.Label(r3, text="tesseract.exe (optional override):").pack(side="left")
        ttk.Entry(r3, textvariable=self.tess_exe_user).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r3, text="Browse…", command=self.choose_tesseract).pack(side="left")

        # Options
        opts = ttk.LabelFrame(root, text="Options")
        opts.pack(fill="x", **pad)

        o1 = ttk.Frame(opts)
        o1.pack(fill="x", padx=10, pady=6)
        ttk.Label(o1, text="Language(s):").pack(side="left")
        ttk.Entry(o1, textvariable=self.lang, width=14).pack(side="left", padx=8)
        ttk.Label(o1, text="пример: rus+eng").pack(side="left")

        o2 = ttk.Frame(opts)
        o2.pack(fill="x", padx=10, pady=6)
        ttk.Label(o2, text="Image DPI (for images / missing DPI):").pack(side="left")
        self.dpi_spin = ttk.Spinbox(o2, from_=100, to=600, increment=50, textvariable=self.image_dpi, width=8)
        self.dpi_spin.pack(side="left", padx=8)
        ttk.Label(o2, text="(обычно 300)").pack(side="left")

        o3 = ttk.Frame(opts)
        o3.pack(fill="x", padx=10, pady=6)
        self.skip_text_chk = ttk.Checkbutton(
            o3, variable=self.skip_text,
            text="PDF mode: skip pages that already have text (faster)"
        )
        self.skip_text_chk.pack(side="left")

        # Progress
        prog = ttk.LabelFrame(root, text="Progress")
        prog.pack(fill="x", **pad)

        p1 = ttk.Frame(prog)
        p1.pack(fill="x", padx=10, pady=6)
        ttk.Label(p1, text="Stage:").pack(side="left")
        ttk.Label(p1, textvariable=self.stage).pack(side="left", padx=6)

        p2 = ttk.Frame(prog)
        p2.pack(fill="x", padx=10, pady=6)
        ttk.Label(p2, text="Current:").pack(side="left")
        ttk.Label(p2, textvariable=self.current).pack(side="left", padx=6)

        p3 = ttk.Frame(prog)
        p3.pack(fill="x", padx=10, pady=6)
        self.bar = ttk.Progressbar(p3, orient="horizontal", mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        ttk.Label(p3, textvariable=self.percent, width=8, anchor="e").pack(side="left", padx=8)

        # Buttons
        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

    def _apply_mode_ui(self):
        is_images = self.mode.get() == "images"
        is_pdf = self.mode.get() == "pdf"

        self.in_dir_entry.configure(state="normal" if is_images else "disabled")
        self.in_dir_btn.configure(state="normal" if is_images else "disabled")

        self.in_pdf_entry.configure(state="normal" if is_pdf else "disabled")
        self.in_pdf_btn.configure(state="normal" if is_pdf else "disabled")

        self.skip_text_chk.configure(state="normal" if is_pdf else "disabled")

    def choose_in_dir(self):
        d = filedialog.askdirectory(title="Select folder with images")
        if d:
            self.in_dir.set(d)

    def choose_in_pdf(self):
        files = filedialog.askopenfilenames(
            title="Select input PDF(s)",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if files:
            self.in_pdf.set(";".join(files))

    def choose_out(self):
        # if in PDF mode and multiple files selected -> choose folder
        mode = self.mode.get()
        pdfs = [p.strip() for p in self.in_pdf.get().split(";") if p.strip()]
        is_batch_pdf = (mode == "pdf" and len(pdfs) > 1)

        if is_batch_pdf:
            d = filedialog.askdirectory(title="Select output folder for OCR PDFs")
            if d:
                self.out_pdf.set(d)
            return

        f = filedialog.asksaveasfilename(
            title="Save PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if f:
            self.out_pdf.set(f)

    def choose_tesseract(self):
        f = filedialog.askopenfilename(
            title="Select tesseract.exe",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if f:
            self.tess_exe_user.set(f)

    def set_busy(self, busy: bool):
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    def _progress_set_determinate(self, maximum=100):
        if self._progress_mode != "determinate":
            self.bar.stop()
        self.bar.configure(mode="determinate", maximum=maximum)
        self._progress_mode = "determinate"

    def _progress_set_indeterminate(self):
        if self._progress_mode != "indeterminate":
            self.bar.configure(mode="indeterminate")
            self.bar.start(12)
            self._progress_mode = "indeterminate"

    def _progress_reset(self):
        self._progress_set_determinate(100)
        self.bar.configure(value=0)
        self.percent.set("0%")

    # ---------- UI event queue / polling ----------

    def _post(self, kind: str, **payload):
        """Worker thread -> UI thread event."""
        self._events.put((kind, payload))

    def _start_polling(self):
        if self._polling:
            return
        self._polling = True
        self.after(self.UI_TICK_MS, self._poll_once)

    def _poll_once(self):
        latest = {}
        try:
            while True:
                k, p = self._events.get_nowait()
                latest[k] = p
        except queue.Empty:
            pass

        if "ui_progress" in latest:
            p = latest["ui_progress"]
            self._apply_progress(**p)
        if "ui_state" in latest:
            p = latest["ui_state"]
            self.stage.set(p.get("stage", self.stage.get()))
            self.current.set(p.get("current", self.current.get()))
        if "ui_busy" in latest:
            p = latest["ui_busy"]
            self.set_busy(bool(p.get("busy", False)))
        if "ui_done" in latest:
            p = latest["ui_done"]
            self.stage.set("Done")
            self.current.set(str(p["out_pdf"]))
            self._progress_set_determinate(100)
            self.bar.configure(value=100)
            self.percent.set("100%")
            self.set_busy(False)
            messagebox.showinfo("Done", f"Saved:\n{p['out_pdf']}")
        if "ui_error" in latest:
            p = latest["ui_error"]
            self.stage.set("Error")
            self.set_busy(False)
            self._progress_set_determinate(100)
            messagebox.showerror("Error", p["message"])

        if self._polling:
            self.after(self.UI_TICK_MS, self._poll_once)

    def _apply_progress(
        self, *, mode: str, stage: str, current: str, pct: str | None,
        maximum: float | None = None, value: float | None = None
    ):
        self.stage.set(stage)
        self.current.set(current)

        if mode == "indeterminate":
            self._progress_set_indeterminate()
            self.percent.set(pct or "…")
            return

        if maximum is not None:
            self._progress_set_determinate(maximum=maximum)
        else:
            self._progress_set_determinate(maximum=self.bar["maximum"])

        if value is not None:
            self.bar.configure(value=value)

        if pct is not None:
            self.percent.set(pct)

    # ---------- Cancel ----------

    def cancel(self):
        self._stop.set()
        self.stage.set("Cancelling…")

    # ---------- OCRmyPDF progress callback ----------

    def _ocrmypdf_progress_cb(self, *, kind: str, desc, total, unit, current, error=None):
        if self._stop.is_set():
            raise RuntimeError("Cancelled by user.")

        now = time.monotonic()
        if kind not in ("enter", "exit") and (now - self._last_progress_post) < self.PROGRESS_THROTTLE_S:
            return
        self._last_progress_post = now

        desc = desc or "Working…"
        unit = unit or ""
        cur = float(current or 0.0)

        def fmt_units():
            if total is None:
                return f"{cur:.0f} {unit}".strip()
            try:
                return f"{cur:.0f}/{float(total):.0f} {unit}".strip()
            except Exception:
                return f"{cur:.0f} {unit}".strip()

        if kind in ("enter", "update"):
            if total is None:
                self._post(
                    "ui_progress",
                    mode="indeterminate",
                    stage=desc,
                    current=fmt_units(),
                    pct="…"
                )
            else:
                try:
                    tot = float(total)
                except Exception:
                    tot = None

                if tot and tot > 0:
                    pct = int((cur / tot) * 100)
                    self._post(
                        "ui_progress",
                        mode="determinate",
                        stage=desc,
                        current=fmt_units(),
                        maximum=tot,
                        value=cur,
                        pct=f"{pct}%"
                    )
                else:
                    self._post(
                        "ui_progress",
                        mode="indeterminate",
                        stage=desc,
                        current=fmt_units(),
                        pct="…"
                    )

    # ---------- Dependency warnings (non-blocking) ----------

    def _warn_if_missing_tools(self, mode: str):
        warnings = []

        tesseract = shutil.which("tesseract")
        gs = which_any(["gswin64c", "gswin32c", "gs"])
        qpdf = shutil.which("qpdf")

        if not tesseract:
            warnings.append("Tesseract не найден в PATH (если упадёт — укажи tesseract.exe).")
        if not gs:
            warnings.append("Ghostscript (gs/gswin64c) не найден в PATH (если упадёт — установи Ghostscript или добавь в PATH).")
        if not qpdf:
            warnings.append("qpdf не найден в PATH (некоторые сценарии OCRmyPDF требуют qpdf).")

        if mode == "images":
            try:
                import img2pdf  # noqa: F401
            except Exception:
                warnings.append("Пакет img2pdf не установлен (нужен для режима Images folder → PDF). Установи: pip install img2pdf")

        if warnings:
            messagebox.showwarning(
                "Предупреждение о зависимостях",
                "Возможные проблемы:\n\n- " + "\n- ".join(warnings)
            )

    # ---------- Start / Worker ----------

    def start(self):
        mode = self.mode.get()

        # Determine if batch PDF selected (needed for minimal fix #0)
        pdf_list: list[Path] = []
        if mode == "pdf":
            raw = self.in_pdf.get().strip()
            pdf_list = [Path(p.strip().strip('"')) for p in raw.split(";") if p.strip()]
        is_batch_pdf = (mode == "pdf" and len(pdf_list) > 1)

        out_raw = self.out_pdf.get().strip()
        if not out_raw:
            messagebox.showerror("Error", "Укажи путь для сохранения.")
            return

        out_path = Path(out_raw)

        # Minimal fix #0: only force .pdf for non-batch jobs
        if not is_batch_pdf:
            if not out_path.name:
                messagebox.showerror("Error", "Укажи путь для сохранения PDF.")
                return
            if out_path.suffix.lower() != ".pdf":
                out_path = out_path.with_suffix(".pdf")
                self.out_pdf.set(str(out_path))

        # Validate inputs
        if mode == "images":
            in_dir = Path(self.in_dir.get().strip())
            if not in_dir.exists() or not in_dir.is_dir():
                messagebox.showerror("Error", "Выбери корректную папку с изображениями.")
                return

            imgs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            imgs.sort(key=lambda p: natural_key(p.name))
            if not imgs:
                messagebox.showerror("Error", "В папке не найдено изображений.")
                return

            job = ("images", imgs, out_path)

        else:
            if not pdf_list:
                messagebox.showerror("Error", "Выбери корректный входной PDF.")
                return

            bad = [p for p in pdf_list if (not p.exists()) or p.suffix.lower() != ".pdf"]
            if bad:
                messagebox.showerror(
                    "Error",
                    "Некоторые выбранные файлы не существуют или не PDF:\n" + "\n".join(map(str, bad))
                )
                return

            if len(pdf_list) == 1:
                job = ("pdf", pdf_list[0], out_path)
            else:
                job = ("pdf_batch", pdf_list, out_path)

        # Import checks (hard)
        try:
            ensure_importable("ocrmypdf", "Установи: pip install ocrmypdf (и системные зависимости).")
            if mode == "images":
                ensure_importable("img2pdf", "Установи: pip install img2pdf")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._warn_if_missing_tools(mode)

        # Launch
        self._stop.clear()
        self.stage.set("Starting…")
        self.current.set("-")
        self._progress_reset()
        self.set_busy(True)

        self._start_polling()

        self._worker = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._worker.start()

    def _run(self, job):
        wd = None
        try:
            import ocrmypdf

            out_path: Path = job[2]

            # Safe method #2: create correct target dir depending on whether out_path is file or folder hint
            if str(out_path).lower().endswith(".pdf"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path.mkdir(parents=True, exist_ok=True)

            lang = (self.lang.get().strip() or "rus+eng")
            dpi = int(self.image_dpi.get() or 300)

            tess = self.tess_exe_user.get().strip()
            tess_dir = None
            if tess:
                p = Path(tess)
                if p.exists():
                    tess_dir = p.parent

            with ascii_tempdir(prefix="ocrjob_") as wd:
                with temp_env(TMP=str(wd), TEMP=str(wd)):
                    with prepend_to_path(tess_dir):
                        self._post("ui_state", stage="Preparing…", current=str(wd))

                        # Attach progress callback
                        self._ocr_plugin_mod.set_callback(self._ocrmypdf_progress_cb)

                        def resolve_batch_out(in_file: Path, out_hint: Path) -> Path:
                            out_str = str(out_hint)
                            looks_like_dir = (out_hint.exists() and out_hint.is_dir()) or (not out_str.lower().endswith(".pdf"))
                            out_dir = (out_hint if looks_like_dir else in_file.parent)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            return out_dir / f"{in_file.stem}_ocr.pdf"

                        if job[0] == "pdf":
                            in_pdf: Path = job[1]

                            in_local = wd / "input.pdf"
                            out_local = wd / "output.pdf"
                            shutil.copy2(in_pdf, in_local)

                            try:
                                ocrmypdf.ocr(
                                    str(in_local),
                                    str(out_local),
                                    language=[lang],
                                    skip_text=bool(self.skip_text.get()),
                                    progress_bar=True,
                                    plugins=[self._ocr_plugin_name],
                                )
                            except Exception as e:
                                raise self._decorate_ocr_exception(e) from e

                            shutil.copy2(out_local, out_path)
                            self._post("ui_done", out_pdf=str(out_path))
                            return

                        if job[0] == "pdf_batch":
                            pdfs: list[Path] = job[1]
                            out_hint: Path = job[2]

                            t0_all = time.monotonic()
                            print(f"[batch] start: {len(pdfs)} file(s)")

                            for idx, src_pdf in enumerate(pdfs, start=1):
                                if self._stop.is_set():
                                    raise RuntimeError("Cancelled by user.")

                                dst_pdf = resolve_batch_out(src_pdf, out_hint)
                                self._post("ui_state", stage=f"Batch OCR ({idx}/{len(pdfs)})", current=src_pdf.name)

                                in_local = wd / f"input_{idx:04d}.pdf"
                                out_local = wd / f"output_{idx:04d}.pdf"
                                shutil.copy2(src_pdf, in_local)

                                t0 = time.monotonic()
                                try:
                                    ocrmypdf.ocr(
                                        str(in_local),
                                        str(out_local),
                                        language=[lang],
                                        skip_text=bool(self.skip_text.get()),
                                        progress_bar=True,
                                        plugins=[self._ocr_plugin_name],
                                    )
                                except Exception as e:
                                    dt = time.monotonic() - t0
                                    print(f"[batch] ERROR {idx}/{len(pdfs)} {src_pdf} ({dt:.1f}s): {e}")
                                    raise self._decorate_ocr_exception(e) from e

                                shutil.copy2(out_local, dst_pdf)

                                dt = time.monotonic() - t0
                                print(f"[batch] done {idx}/{len(pdfs)} -> {dst_pdf} ({dt:.1f}s)")

                            dt_all = time.monotonic() - t0_all
                            print(f"[batch] all done: {len(pdfs)} file(s) in {dt_all:.1f}s")

                            self._post("ui_done", out_pdf=str(out_hint))
                            return

                        # images mode: copy/link images -> build one PDF -> one OCR pass
                        imgs: list[Path] = job[1]
                        img_dir = wd / "imgs"
                        img_dir.mkdir(parents=True, exist_ok=True)

                        total = len(imgs)
                        self._post("ui_state", stage="Staging images…", current=f"0/{total}")

                        for i, src in enumerate(imgs, start=1):
                            if self._stop.is_set():
                                raise RuntimeError("Cancelled by user.")
                            dst = img_dir / safe_ascii_name(i, src.suffix)
                            fast_copy_or_link(src, dst)

                            pct = int(i / total * 10)
                            self._post(
                                "ui_progress",
                                mode="determinate",
                                stage="Staging images…",
                                current=f"{i}/{total}",
                                maximum=100.0,
                                value=float(pct),
                                pct=f"{pct}%"
                            )

                        if self._stop.is_set():
                            raise RuntimeError("Cancelled by user.")

                        self._post("ui_state", stage="Building PDF…", current="img2pdf")
                        in_pdf = wd / "input.pdf"
                        local_imgs = sorted(img_dir.iterdir(), key=lambda p: natural_key(p.name))

                        # Practical fix #1: better error message + identify offending image
                        try:
                            build_images_pdf(local_imgs, in_pdf)
                        except Exception as e:
                            bad = find_bad_image(local_imgs)
                            if bad:
                                raise RuntimeError(
                                    "Не удалось собрать PDF из изображений (img2pdf/Pillow).\n"
                                    f"Проблемный файл:\n{bad}"
                                ) from e
                            raise

                        out_local = wd / "output.pdf"
                        try:
                            ocrmypdf.ocr(
                                str(in_pdf),
                                str(out_local),
                                language=[lang],
                                image_dpi=dpi,
                                progress_bar=True,
                                plugins=[self._ocr_plugin_name],
                            )
                        except Exception as e:
                            raise self._decorate_ocr_exception(e) from e

                        shutil.copy2(out_local, out_path)
                        self._post("ui_done", out_pdf=str(out_path))

        except Exception as e:
            log_path = None
            try:
                if wd is not None:
                    log_path = Path(wd) / "error_traceback.txt"
                    log_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
            except Exception:
                log_path = None

            msg = str(e)
            if log_path:
                msg += f"\n\nTraceback saved to:\n{log_path}"

            self._post("ui_error", message=msg)

        finally:
            self._post("ui_busy", busy=False)

    def _decorate_ocr_exception(self, e: Exception) -> Exception:
        name = type(e).__name__
        msg = str(e) or name

        if "PriorOcrFoundError" in name or "page already has text" in msg.lower():
             return RuntimeError(
            "PDF уже содержит текст (векторный или скрытый OCR-слой), поэтому OCRmyPDF прерывает обработку.\n\n"
            "Что можно сделать:\n"
            "• Включить опцию 'skip pages that already have text' (skip_text) — пропустить такие страницы.\n"
            "• Или запускать с --redo-ocr (перераспознать существующий OCR-слой).\n"
            "• Или с --force-ocr (растрировать всё и распознать заново).\n"
        )


        if "MissingDependencyError" in name:
            extra = []
            mlow = msg.lower()
            if "tesseract" in mlow:
                extra.append("Похоже, не найден Tesseract. Установи Tesseract или укажи tesseract.exe (Browse…) чтобы добавить его папку в PATH.")
            if "ghostscript" in mlow or "gs" in mlow:
                extra.append("Похоже, не найден Ghostscript (gs/gswin64c). Установи Ghostscript и/или добавь в PATH.")
            if "qpdf" in mlow:
                extra.append("Похоже, не найден qpdf. Установи qpdf и/или добавь в PATH.")

            if extra:
                msg = msg + "\n\n" + "\n".join(f"- {x}" for x in extra)
            else:
                msg = msg + "\n\nНе хватает внешней утилиты. Проверь установку Tesseract/Ghostscript/qpdf и PATH."

            return RuntimeError(msg)

        if isinstance(e, RuntimeError) and "Cancelled by user" in msg:
            return RuntimeError("Cancelled by user.")

        return e


if __name__ == "__main__":
    App().mainloop()
