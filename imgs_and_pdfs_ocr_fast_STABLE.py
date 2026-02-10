
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import types
import sys
import traceback

from pypdf import PdfReader, PdfWriter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

import contextlib
import uuid

ASCII_TMP_ROOT = Path(os.environ.get("SystemDrive", "C:")) / "_ocrmypdf_tmp"

@contextlib.contextmanager
def ascii_tempdir(prefix: str = "ocr_"):
    """
    Временная папка гарантированно в ASCII-пути (например C:\\_ocrmypdf_tmp\\ocr_xxx).
    """
    ASCII_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # mkdtemp делает реально уникальную папку
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=str(ASCII_TMP_ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

def safe_ascii_name(i: int, ext: str) -> str:
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext if ext else ""
    return f"img_{i:05d}{ext}"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_tesseract_in_path() -> str | None:
    exe = shutil.which("tesseract")
    return exe if exe else None


def ensure_ocrmypdf_importable():
    try:
        import ocrmypdf  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Не могу импортировать пакет 'ocrmypdf'. Установи его (pip install ocrmypdf) "
            "и зависимости (Tesseract, Ghostscript, ...).\n\n"
            f"Details: {e}"
        )


def build_tk_progress_plugin():
    """
    Создаёт in-memory модуль-плагин OCRmyPDF, который заменяет progress bar на наш (Tk).
    Возвращает (module_name, module_object).
    """
    plugin_name = "_ocrmypdf_tk_progress_plugin"
    if plugin_name in sys.modules:
        return plugin_name, sys.modules[plugin_name]

    m = types.ModuleType(plugin_name)
    sys.modules[plugin_name] = m

    # Ленивая и безопасная инициализация: модуль будет импортироваться и в воркерах,
    # но сам прогресс-бар OCRmyPDF держит в главном процессе. :contentReference[oaicite:2]{index=2}
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


class App(tk.Tk):
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

        # Remember user's chosen tesseract.exe (only if not in PATH)
        self.tess_exe_user = tk.StringVar(value="")

        # Progress UI
        self.stage = tk.StringVar(value="Idle")
        self.current = tk.StringVar(value="-")
        self.percent = tk.StringVar(value="0%")
        self._progress_mode = "determinate"

        self._stop = threading.Event()
        self._worker = None

        # OCRmyPDF Tk progress plugin
        self._ocr_plugin_name, self._ocr_plugin_mod = build_tk_progress_plugin()

        self._build_ui()
        self._apply_mode_ui()

    # ---------- UI ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # Mode selector
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

        # Tesseract (optional prompt)
        r3 = ttk.Frame(root)
        r3.pack(fill="x", **pad)
        ttk.Label(r3, text="tesseract.exe (only if not in PATH):").pack(side="left")
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
        self.skip_text_chk = ttk.Checkbutton(o3, variable=self.skip_text, text="PDF mode: skip pages that already have text (faster)")
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
        f = filedialog.askopenfilename(
            title="Select input PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if f:
            self.in_pdf.set(f)

    def choose_out(self):
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

    def _ui(self, fn):
        self.after(0, fn)

    def _progress_set_determinate(self, maximum=100):
        def _do():
            if self._progress_mode != "determinate":
                self.bar.stop()
            self.bar.configure(mode="determinate", maximum=maximum)
            self._progress_mode = "determinate"
        self._ui(_do)

    def _progress_set_indeterminate(self):
        def _do():
            if self._progress_mode != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(12)
                self._progress_mode = "indeterminate"
        self._ui(_do)

    def _progress_reset(self):
        self._progress_set_determinate(100)
        self._ui(lambda: (self.bar.configure(value=0), self.percent.set("0%")))

    def _progress_update_pct(self, pct: int):
        pct = max(0, min(100, int(pct)))
        self._ui(lambda v=pct: (self.bar.configure(value=v), self.percent.set(f"{v}%")))

    def cancel(self):
        self._stop.set()
        self.stage.set("Cancelling…")

    # ---------- OCRmyPDF progress callback (PDF mode) ----------

    def _ocrmypdf_progress_cb(self, *, kind: str, desc, total, unit, current, error=None):
        # allow "Cancel" to abort OCRmyPDF run
        if self._stop.is_set():
            raise RuntimeError("Cancelled by user.")

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
                self._progress_set_indeterminate()
                self._ui(lambda d=desc, s=fmt_units(): (self.stage.set(d), self.current.set(s), self.percent.set("…")))
            else:
                try:
                    tot = float(total)
                except Exception:
                    tot = None

                if tot and tot > 0:
                    self._progress_set_determinate(maximum=tot)
                    self._ui(lambda d=desc, s=fmt_units(), v=cur, tot=tot: (
                        self.stage.set(d),
                        self.current.set(s),
                        self.bar.configure(value=v),
                        self.percent.set(f"{int((v/tot)*100)}%")
                    ))
                else:
                    self._progress_set_indeterminate()
                    self._ui(lambda d=desc, s=fmt_units(): (self.stage.set(d), self.current.set(s), self.percent.set("…")))

        elif kind == "exit":
            # на выходе конкретного этапа просто оставим то, что есть
            pass

    # ---------- Dependency checks ----------

    def _ensure_tesseract_available(self):
        if find_tesseract_in_path():
            return

        tess = self.tess_exe_user.get().strip()
        if not tess:
            messagebox.showinfo(
                "Tesseract not found",
                "Tesseract не найден в PATH.\nПожалуйста, выбери tesseract.exe (кнопка Browse…)."
            )
            raise RuntimeError("Tesseract not found in PATH; user must select tesseract.exe.")

        p = Path(tess)
        if not p.exists() or p.suffix.lower() != ".exe":
            raise RuntimeError("Указан некорректный путь к tesseract.exe.")

        os.environ["PATH"] = str(p.parent) + os.pathsep + os.environ.get("PATH", "")

    # ---------- Start / Worker ----------

    def start(self):
        out_pdf = Path(self.out_pdf.get().strip())
        if not out_pdf.name:
            messagebox.showerror("Error", "Укажи путь для сохранения PDF.")
            return
        if out_pdf.suffix.lower() != ".pdf":
            out_pdf = out_pdf.with_suffix(".pdf")
            self.out_pdf.set(str(out_pdf))

        try:
            ensure_ocrmypdf_importable()
            self._ensure_tesseract_available()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        mode = self.mode.get()
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

            job = ("images", imgs, out_pdf)

        else:
            in_pdf = Path(self.in_pdf.get().strip())
            if not in_pdf.exists() or in_pdf.suffix.lower() != ".pdf":
                messagebox.showerror("Error", "Выбери корректный входной PDF.")
                return

            job = ("pdf", in_pdf, out_pdf)

        self._stop.clear()
        self.stage.set("Starting…")
        self.current.set("-")
        self._progress_reset()
        self.set_busy(True)

        self._worker = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._worker.start()

    def _run(self, job):
        import traceback
        wd = None

        try:
            import ocrmypdf

            out_pdf: Path = job[2]
            out_pdf.parent.mkdir(parents=True, exist_ok=True)

            lang = (self.lang.get().strip() or "rus+eng")
            dpi = int(self.image_dpi.get() or 300)

            # ВСЁ делаем внутри ASCII-temp
            with ascii_tempdir(prefix="ocrjob_") as wd:
                # Форсим TEMP/TMP в ASCII-папку, чтобы Ghostscript/Tesseract не лезли в кириллицу
                old_tmp = (os.environ.get("TMP"), os.environ.get("TEMP"))
                os.environ["TMP"] = str(wd)
                os.environ["TEMP"] = str(wd)

                try:
                    self._ui(lambda: (self.stage.set("Preparing…"), self.current.set(str(wd))))

                    if job[0] == "pdf":
                        in_pdf: Path = job[1]

                        # Копируем входной PDF в ASCII-путь
                        in_local = wd / "input.pdf"
                        out_local = wd / "output.pdf"
                        shutil.copy2(in_pdf, in_local)

                        # Подключаем Tk progress plugin
                        self._ocr_plugin_mod.set_callback(self._ocrmypdf_progress_cb)

                        ocrmypdf.ocr(
                            str(in_local),
                            str(out_local),
                            language=[lang],
                            skip_text=bool(self.skip_text.get()),
                            progress_bar=True,
                            plugins=[self._ocr_plugin_name],
                        )

                        # Копируем результат туда, куда указал пользователь
                        shutil.copy2(out_local, out_pdf)

                        self._progress_set_determinate(100)
                        self._progress_update_pct(100)

                    else:
                        imgs: list[Path] = job[1]

                        # Копируем все картинки в ASCII-путь с ASCII-именами
                        img_dir = wd / "imgs"
                        img_dir.mkdir(parents=True, exist_ok=True)

                        total = len(imgs)
                        self._progress_set_determinate(100)
                        self._ui(lambda: (self.stage.set("Copying images…"), self.current.set(f"0/{total}")))

                        local_imgs: list[Path] = []
                        for i, src in enumerate(imgs, start=1):
                            if self._stop.is_set():
                                raise RuntimeError("Cancelled by user.")

                            dst = img_dir / safe_ascii_name(i, src.suffix)
                            shutil.copy2(src, dst)
                            local_imgs.append(dst)

                            pct = int(i / total * 10)  # 0..10% на копирование
                            self._progress_update_pct(pct)

                        # OCR каждой картинки -> PDF в ASCII-temp
                        pages_dir = wd / "pages"
                        pages_dir.mkdir(parents=True, exist_ok=True)

                        page_pdfs: list[Path] = []
                        self._ui(lambda: (self.stage.set("OCR images…"), self.current.set(f"0/{total}")))

                        for i, img in enumerate(local_imgs, start=1):
                            if self._stop.is_set():
                                raise RuntimeError("Cancelled by user.")

                            self._ui(lambda i=i, t=total, n=img.name: (
                                self.stage.set(f"OCR image… ({i}/{t})"),
                                self.current.set(n)
                            ))

                            page_out = pages_dir / f"page_{i:05d}.pdf"
                            ocrmypdf.ocr(
                                str(img),
                                str(page_out),
                                language=[lang],
                                image_dpi=dpi,
                                output_type="pdf",
                                progress_bar=False,
                            )
                            page_pdfs.append(page_out)

                            pct = 10 + int(i / total * 70)  # 10..80% на OCR
                            self._progress_update_pct(pct)

                        if self._stop.is_set():
                            raise RuntimeError("Cancelled by user.")

                        # Merge pages (80..100%) — тоже внутри ASCII-temp
                        self._ui(lambda: (self.stage.set("Merging…"), self.current.set("Combining pages…")))
                        writer = PdfWriter()
                        mtotal = len(page_pdfs)

                        for i, p in enumerate(page_pdfs, start=1):
                            if self._stop.is_set():
                                raise RuntimeError("Cancelled by user.")
                            reader = PdfReader(str(p))
                            for page in reader.pages:
                                writer.add_page(page)

                            pct = 80 + int(i / mtotal * 20)
                            self._progress_update_pct(pct)

                        merged_local = wd / "merged.pdf"
                        with open(merged_local, "wb") as f:
                            writer.write(f)

                        shutil.copy2(merged_local, out_pdf)
                        self._progress_update_pct(100)

                    self._ui(lambda: (
                        self.stage.set("Done"),
                        self.current.set(str(out_pdf)),
                        self.set_busy(False),
                        messagebox.showinfo("Done", f"Saved:\n{out_pdf}")
                    ))

                finally:
                    # Вернуть TEMP/TMP обратно
                    if old_tmp[0] is None:
                        os.environ.pop("TMP", None)
                    else:
                        os.environ["TMP"] = old_tmp[0]

                    if old_tmp[1] is None:
                        os.environ.pop("TEMP", None)
                    else:
                        os.environ["TEMP"] = old_tmp[1]

        except Exception as e:
            # Пишем полный traceback в лог (в ASCII temp, если был создан)
            log_path = None
            try:
                if wd is not None:
                    log_path = Path(wd) / "error_traceback.txt"
                    log_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
            except Exception:
                log_path = None

            err = f"{e}"
            if log_path:
                err += f"\n\nTraceback saved to:\n{log_path}"

            self._ui(lambda err=err: (
                self.stage.set("Error"),
                self.set_busy(False),
                self._progress_set_determinate(100),
                messagebox.showerror("Error", err)
            ))



if __name__ == "__main__":
    App().mainloop()