#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pypdf import PdfReader, PdfWriter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def resolve_exe_from_env_or_path(env_name: str, exe_name: str):
    env = os.environ.get(env_name)
    if env:
        p = Path(env)
        if p.exists():
            return str(p)
    w = shutil.which(exe_name)
    if w and Path(w).exists():
        return w
    return None


def resolve_tesseract_exe():
    w = resolve_exe_from_env_or_path("TESSERACT_EXE", "tesseract")
    if w:
        return w
    candidates = [
        r"J:\PC\AI\Tesseract\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def resolve_poppler_tool(exe_name: str, poppler_bin: str | None):
    """
    Ищет poppler tool (pdftoppm/pdfunite) сначала в poppler_bin,
    затем по env, затем в PATH.
    """
    if poppler_bin:
        p = Path(poppler_bin) / f"{exe_name}.exe"
        if p.exists():
            return str(p)

    env_specific = resolve_exe_from_env_or_path(f"{exe_name.upper()}_EXE", exe_name)
    if env_specific:
        return env_specific

    env_bin = os.environ.get("POPPLER_BIN")
    if env_bin:
        p = Path(env_bin) / f"{exe_name}.exe"
        if p.exists():
            return str(p)

    w = shutil.which(exe_name)
    if w and Path(w).exists():
        return w

    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCR: Photos/PDF → searchable PDF (Tesseract)")
        self.geometry("860x500")
        self.minsize(760, 450)

        # Mode: "images" or "pdf"
        self.mode = tk.StringVar(value="images")

        # Inputs
        self.in_dir = tk.StringVar()
        self.in_pdf = tk.StringVar()
        self.out_pdf = tk.StringVar()

        # OCR options
        self.lang = tk.StringVar(value="rus+eng")
        self.dpi = tk.IntVar(value=300)  # for PDF->images rendering

        # Tools
        self.tess_exe = tk.StringVar(value=resolve_tesseract_exe() or "")
        self.poppler_bin = tk.StringVar(value="")  # optional for images-mode; required for pdf-mode unless in PATH

        # UI progress
        self.stage = tk.StringVar(value="Idle")
        self.current = tk.StringVar(value="-")
        self.percent = tk.StringVar(value="0%")

        self._stop = threading.Event()
        self._worker = None
        self._progress_mode = "determinate"  # or "indeterminate"

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

        # Input folder (mode images)
        r1 = ttk.Frame(root)
        r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Images folder:").pack(side="left")
        self.in_dir_entry = ttk.Entry(r1, textvariable=self.in_dir)
        self.in_dir_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_dir_btn = ttk.Button(r1, text="Choose…", command=self.choose_in_dir)
        self.in_dir_btn.pack(side="left")

        # Input PDF (mode pdf)
        r1b = ttk.Frame(root)
        r1b.pack(fill="x", **pad)
        ttk.Label(r1b, text="Input PDF:").pack(side="left")
        self.in_pdf_entry = ttk.Entry(r1b, textvariable=self.in_pdf)
        self.in_pdf_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_pdf_btn = ttk.Button(r1b, text="Choose…", command=self.choose_in_pdf)
        self.in_pdf_btn.pack(side="left")

        # Output file
        r2 = ttk.Frame(root)
        r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Save PDF as:").pack(side="left")
        ttk.Entry(r2, textvariable=self.out_pdf).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r2, text="Save as…", command=self.choose_out).pack(side="left")

        # Tesseract exe
        r3 = ttk.Frame(root)
        r3.pack(fill="x", **pad)
        ttk.Label(r3, text="tesseract.exe:").pack(side="left")
        ttk.Entry(r3, textvariable=self.tess_exe).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r3, text="Browse…", command=self.choose_tesseract).pack(side="left")

        # Poppler bin
        r4 = ttk.Frame(root)
        r4.pack(fill="x", **pad)
        ttk.Label(r4, text="Poppler bin:").pack(side="left")
        ttk.Entry(r4, textvariable=self.poppler_bin).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r4, text="Browse…", command=self.choose_poppler_bin).pack(side="left")

        # Options
        opts = ttk.LabelFrame(root, text="Options")
        opts.pack(fill="x", **pad)
        o1 = ttk.Frame(opts)
        o1.pack(fill="x", padx=10, pady=6)
        ttk.Label(o1, text="Tesseract lang:").pack(side="left")
        ttk.Entry(o1, textvariable=self.lang, width=14).pack(side="left", padx=8)
        ttk.Label(o1, text="пример: rus+eng").pack(side="left")

        o2 = ttk.Frame(opts)
        o2.pack(fill="x", padx=10, pady=6)
        ttk.Label(o2, text="PDF render DPI (for PDF→PDF):").pack(side="left")
        self.dpi_spin = ttk.Spinbox(o2, from_=100, to=600, increment=50, textvariable=self.dpi, width=8)
        self.dpi_spin.pack(side="left", padx=8)
        ttk.Label(o2, text="(обычно 300)").pack(side="left")

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
        ttk.Label(p3, textvariable=self.percent, width=6, anchor="e").pack(side="left", padx=8)

        # Buttons
        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

    def _apply_mode_ui(self):
        mode = self.mode.get()
        is_images = mode == "images"
        is_pdf = mode == "pdf"

        state_images = "normal" if is_images else "disabled"
        state_pdf = "normal" if is_pdf else "disabled"

        self.in_dir_entry.configure(state=state_images)
        self.in_dir_btn.configure(state=state_images)

        self.in_pdf_entry.configure(state=state_pdf)
        self.in_pdf_btn.configure(state=state_pdf)

        self.dpi_spin.configure(state="normal" if is_pdf else "disabled")

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
            self.tess_exe.set(f)

    def choose_poppler_bin(self):
        d = filedialog.askdirectory(title="Select Poppler bin folder (where pdfunite.exe / pdftoppm.exe live)")
        if d:
            self.poppler_bin.set(d)

    def set_busy(self, busy: bool):
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    # ---------- Progress helpers ----------

    def _progress_set_determinate(self):
        def _do():
            if self._progress_mode != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate", maximum=100, value=0)
                self._progress_mode = "determinate"
        self._ui(_do)

    def _progress_set_indeterminate(self):
        def _do():
            if self._progress_mode != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(12)
                self._progress_mode = "indeterminate"
        self._ui(_do)

    def _progress_update(self, done: int, total: int):
        # determinate stage progress
        pct = int(done / max(total, 1) * 100)
        self._ui(lambda v=pct: (self.bar.configure(value=v), self.percent.set(f"{v}%")))

    def _progress_reset(self):
        self._progress_set_determinate()
        self._ui(lambda: (self.bar.configure(value=0), self.percent.set("0%")))

    # ---------- Start / Cancel ----------

    def start(self):
        mode = self.mode.get()
        out_pdf = Path(self.out_pdf.get().strip())
        tess = Path(self.tess_exe.get().strip())

        if not out_pdf.name:
            messagebox.showerror("Error", "Укажи путь для сохранения PDF.")
            return
        if not out_pdf.name.lower().endswith(".pdf"):
            out_pdf = out_pdf.with_suffix(".pdf")
            self.out_pdf.set(str(out_pdf))

        if not tess.exists():
            messagebox.showerror("Error", "Не найден tesseract.exe. Укажи путь кнопкой Browse…")
            return

        # quick check tesseract
        try:
            r = subprocess.run([str(tess), "--version"], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr or r.stdout)
        except Exception as e:
            messagebox.showerror("Error", f"Tesseract не запускается:\n{e}")
            return

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

            job = ("images", imgs, out_pdf, str(tess))

        else:
            in_pdf = Path(self.in_pdf.get().strip())
            if not in_pdf.exists() or in_pdf.suffix.lower() != ".pdf":
                messagebox.showerror("Error", "Выбери корректный входной PDF.")
                return

            poppler_bin = self.poppler_bin.get().strip() or None
            pdftoppm = resolve_poppler_tool("pdftoppm", poppler_bin)
            if not pdftoppm:
                messagebox.showerror(
                    "Error",
                    "Для режима PDF→PDF нужен Poppler (pdftoppm.exe).\n"
                    "Укажи Poppler bin или добавь Poppler в PATH."
                )
                return

            job = ("pdf", in_pdf, out_pdf, str(tess))

        self._stop.clear()
        self.stage.set("Starting…")
        self.current.set("-")
        self._progress_reset()
        self.set_busy(True)

        self._worker = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._worker.start()

    def cancel(self):
        self._stop.set()
        self.stage.set("Cancelling…")

    # ---------- Core pipeline helpers ----------

    def _ocr_images_to_page_pdfs(self, imgs: list[Path], tmp_img_dir: Path, tmp_pdf_dir: Path, tess_exe: str, lang: str):
        self._progress_set_determinate()
        page_pdfs = []
        total = len(imgs)

        for i, src in enumerate(imgs, start=1):
            if self._stop.is_set():
                raise RuntimeError("Cancelled by user.")

            self._ui(lambda i=i, t=total, n=src.name: (
                self.stage.set(f"OCR… ({i}/{t})"),
                self.current.set(n)
            ))

            # копируем в ASCII-имя
            safe_img = tmp_img_dir / f"{i:05d}{src.suffix.lower()}"
            shutil.copyfile(src, safe_img)

            out_base = tmp_pdf_dir / f"page_{i:05d}"
            cmd = [tess_exe, str(safe_img), str(out_base), "-l", lang, "pdf"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise RuntimeError(f"Tesseract failed on {src.name}\n{err}")

            page_pdf = out_base.with_suffix(".pdf")
            page_pdfs.append(page_pdf)

            self._progress_update(i, total)

        return page_pdfs

    def _pdf_page_count(self, in_pdf: Path) -> int:
        # быстрое получение числа страниц
        reader = PdfReader(str(in_pdf))
        return len(reader.pages)

    def _render_pdf_to_pngs_postpage(self, in_pdf: Path, imgs_dir: Path, pdftoppm: str, dpi: int):
        """
        Рендерим PDF по одной странице, чтобы иметь прогресс.
        Итоговые файлы делаем вида page-00001.png, page-00002.png ...
        """
        self._progress_set_determinate()
        total = self._pdf_page_count(in_pdf)
        if total <= 0:
            raise RuntimeError("PDF без страниц?")

        out_pngs: list[Path] = []

        for i in range(1, total + 1):
            if self._stop.is_set():
                raise RuntimeError("Cancelled by user.")

            self._ui(lambda i=i, t=total: (
                self.stage.set(f"Rendering PDF… ({i}/{t})"),
                self.current.set(f"page {i}")
            ))

            # base prefix (pdftoppm добавит "-1.png" при single-page)
            base = imgs_dir / f"page_{i:05d}"
            cmd = [pdftoppm, "-png", "-r", str(dpi), "-f", str(i), "-l", str(i), str(in_pdf), str(base)]
            rr = subprocess.run(cmd, capture_output=True, text=True)
            if rr.returncode != 0:
                raise RuntimeError(f"pdftoppm failed on page {i}:\n{(rr.stderr or rr.stdout).strip()}")

            produced = imgs_dir / f"page_{i:05d}-1.png"
            if not produced.exists():
                # на всякий случай, если другой нейминг
                candidates = sorted(imgs_dir.glob(f"page_{i:05d}-*.png"))
                if not candidates:
                    raise RuntimeError(f"pdftoppm не создал PNG для страницы {i}.")
                produced = candidates[0]

            final_png = imgs_dir / f"page-{i:05d}.png"
            if final_png.exists():
                final_png.unlink()
            produced.rename(final_png)

            out_pngs.append(final_png)
            self._progress_update(i, total)

        return out_pngs

    def _merge_page_pdfs(self, page_pdfs: list[Path], out_pdf: Path):
        poppler_bin = self.poppler_bin.get().strip() or None
        pdfunite = resolve_poppler_tool("pdfunite", poppler_bin)

        if pdfunite:
            # реального прогресса нет -> indeterminate
            self._ui(lambda: (self.stage.set("Merging… (pdfunite)"), self.current.set("Running pdfunite…")))
            self._progress_set_indeterminate()
            self._ui(lambda: self.percent.set("…"))

            cmd = [pdfunite, *map(str, page_pdfs), str(out_pdf)]
            r = subprocess.run(cmd, capture_output=True, text=True)

            # вернуть determinate
            self._progress_set_determinate()

            if r.returncode != 0:
                raise RuntimeError(f"pdfunite failed:\n{(r.stderr or r.stdout).strip()}")

            # считаем merge завершённым
            self._ui(lambda: (self.bar.configure(value=100), self.percent.set("100%")))
            return

        # Python merge -> можем показывать прогресс
        self._progress_set_determinate()
        total = len(page_pdfs)
        writer = PdfWriter()

        for i, p in enumerate(page_pdfs, start=1):
            if self._stop.is_set():
                raise RuntimeError("Cancelled by user.")

            self._ui(lambda i=i, t=total: (
                self.stage.set(f"Merging… ({i}/{t})"),
                self.current.set(f"adding page {i}")
            ))

            reader = PdfReader(str(p))
            for page in reader.pages:
                writer.add_page(page)

            self._progress_update(i, total)

        with open(out_pdf, "wb") as f:
            writer.write(f)

    # ---------- Worker ----------

    def _run(self, job):
        try:
            if job[0] == "images":
                _, imgs, out_pdf, tess_exe = job
                out_pdf.parent.mkdir(parents=True, exist_ok=True)

                tmp_root = Path(tempfile.mkdtemp(prefix="tess_job_", dir=str(out_pdf.parent)))
                tmp_img_dir = tmp_root / "imgs"
                tmp_pdf_dir = tmp_root / "pages"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                tmp_pdf_dir.mkdir(parents=True, exist_ok=True)

                try:
                    lang = (self.lang.get().strip() or "rus+eng")
                    self._progress_reset()

                    page_pdfs = self._ocr_images_to_page_pdfs(imgs, tmp_img_dir, tmp_pdf_dir, tess_exe, lang)

                    if self._stop.is_set():
                        raise RuntimeError("Cancelled by user.")

                    # merge stage (progress handled inside)
                    self._merge_page_pdfs(page_pdfs, out_pdf)

                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)

            else:
                _, in_pdf, out_pdf, tess_exe = job
                out_pdf.parent.mkdir(parents=True, exist_ok=True)

                tmp_root = Path(tempfile.mkdtemp(prefix="tess_job_", dir=str(out_pdf.parent)))
                imgs_dir = tmp_root / "rendered"
                pages_dir = tmp_root / "pages"
                imgs_dir.mkdir(parents=True, exist_ok=True)
                pages_dir.mkdir(parents=True, exist_ok=True)

                try:
                    poppler_bin = self.poppler_bin.get().strip() or None
                    pdftoppm = resolve_poppler_tool("pdftoppm", poppler_bin)
                    if not pdftoppm:
                        raise RuntimeError("pdftoppm not found (Poppler required for PDF mode).")

                    dpi = int(self.dpi.get() or 300)

                    # 1) Render stage with progress
                    self._progress_reset()
                    pngs = self._render_pdf_to_pngs_postpage(in_pdf, imgs_dir, pdftoppm, dpi)

                    # 2) OCR stage with progress
                    lang = (self.lang.get().strip() or "rus+eng")
                    self._progress_reset()
                    total = len(pngs)
                    page_pdfs: list[Path] = []

                    for i, img in enumerate(pngs, start=1):
                        if self._stop.is_set():
                            raise RuntimeError("Cancelled by user.")

                        self._ui(lambda i=i, t=total: (
                            self.stage.set(f"OCR… ({i}/{t})"),
                            self.current.set(f"page {i}")
                        ))

                        out_base = pages_dir / f"page_{i:05d}"
                        cmd = [tess_exe, str(img), str(out_base), "-l", lang, "pdf"]
                        tr = subprocess.run(cmd, capture_output=True, text=True)
                        if tr.returncode != 0:
                            raise RuntimeError(
                                f"Tesseract failed on page {i}:\n{(tr.stderr or tr.stdout).strip()}"
                            )

                        page_pdfs.append(out_base.with_suffix(".pdf"))
                        self._progress_update(i, total)

                    if self._stop.is_set():
                        raise RuntimeError("Cancelled by user.")

                    # 3) Merge stage with progress
                    self._merge_page_pdfs(page_pdfs, out_pdf)

                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)

            # Done
            self._ui(lambda: (
                self.stage.set("Done"),
                self.current.set(str(job[2] if job[0] == "images" else job[2])),
                self.bar.configure(value=100),
                self.percent.set("100%"),
                self.set_busy(False),
                messagebox.showinfo("Done", f"Saved:\n{job[2] if job[0] == 'images' else job[2]}")
            ))

        except Exception as e:
            err = str(e)
            self._ui(lambda err=err: (
                self.stage.set("Error"),
                self.set_busy(False),
                self._progress_set_determinate(),
                messagebox.showerror("Error", err)
            ))

    def _ui(self, fn):
        self.after(0, fn)


if __name__ == "__main__":
    App().mainloop()
