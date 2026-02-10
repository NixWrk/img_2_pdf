#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import img2pdf


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def imread_unicode(path: Path):
    # Работает с кириллицей в пути на Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path: Path, image: np.ndarray):
    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth < 10 or maxHeight < 10:
        return image

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_page_quad(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else image_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 40, 160)
    edged = cv2.dilate(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    best = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            best = approx.reshape(4, 2)
            break

    if best is None:
        return None

    if scale != 1.0:
        best = (best / scale).astype(np.float32)

    return best


def deskew_and_crop(image_bgr: np.ndarray, crop_pad: int = 12) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(thr)
    if coords is None:
        return image_bgr

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    (h, w) = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thr2 = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    thr2 = cv2.morphologyEx(thr2, cv2.MORPH_OPEN, kernel, iterations=1)

    coords2 = cv2.findNonZero(thr2)
    if coords2 is None:
        return rotated

    x, y, ww, hh = cv2.boundingRect(coords2)
    x = max(0, x - crop_pad)
    y = max(0, y - crop_pad)
    ww = min(rotated.shape[1] - x, ww + 2 * crop_pad)
    hh = min(rotated.shape[0] - y, hh + 2 * crop_pad)

    return rotated[y:y + hh, x:x + ww]


def split_spread(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    mid = w // 2
    return image_bgr[:, :mid], image_bgr[:, mid:]


def preprocess_one(path: Path, do_perspective: bool, do_deskew_crop: bool, split: bool):
    img = imread_unicode(path)

    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if do_perspective:
        quad = find_page_quad(img)
        if quad is not None:
            img = four_point_transform(img, quad)

    pages = split_spread(img) if split else (img,)
    out = []
    for p in pages:
        if do_deskew_crop:
            p = deskew_and_crop(p)
        out.append(p)
    return out


def build_pdf_from_images(image_paths, out_pdf: Path, dpi: int):
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert([str(p) for p in image_paths], dpi=dpi))


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Photos → PDF (auto preprocess + optional OCR)")
        self.geometry("720x360")
        self.minsize(680, 340)

        self.q = queue.Queue()
        self.worker_thread = None
        self.stop_flag = threading.Event()

        # Vars
        self.in_dir = tk.StringVar()
        self.out_pdf = tk.StringVar()
        self.use_ocr = tk.BooleanVar(value=False)
        self.lang = tk.StringVar(value="rus+eng")
        self.optimize = tk.IntVar(value=1)
        self.dpi = tk.IntVar(value=300)
        self.split_spreads = tk.BooleanVar(value=False)
        self.perspective = tk.BooleanVar(value=True)
        self.deskew_crop = tk.BooleanVar(value=True)

        self.stage = tk.StringVar(value="Idle")
        self.current_file = tk.StringVar(value="-")
        self.percent_text = tk.StringVar(value="0%")

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        # Input folder
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Images folder:").pack(side="left")
        ttk.Entry(row1, textvariable=self.in_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row1, text="Choose…", command=self.choose_in_dir).pack(side="left")

        # Output file
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Save PDF as:").pack(side="left")
        ttk.Entry(row2, textvariable=self.out_pdf).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(row2, text="Save as…", command=self.choose_out_pdf).pack(side="left")

        # Options
        opts = ttk.LabelFrame(frm, text="Options")
        opts.pack(fill="x", **pad)

        line1 = ttk.Frame(opts)
        line1.pack(fill="x", padx=10, pady=6)

        ttk.Checkbutton(line1, text="Auto perspective", variable=self.perspective).pack(side="left")
        ttk.Checkbutton(line1, text="Auto deskew + crop", variable=self.deskew_crop).pack(side="left", padx=12)
        ttk.Checkbutton(line1, text="Split spreads (left/right)", variable=self.split_spreads).pack(side="left")

        line2 = ttk.Frame(opts)
        line2.pack(fill="x", padx=10, pady=6)

        ttk.Label(line2, text="DPI:").pack(side="left")
        ttk.Spinbox(line2, from_=72, to=600, textvariable=self.dpi, width=6).pack(side="left", padx=6)

        ttk.Checkbutton(line2, text="Run OCR (ocrmypdf)", variable=self.use_ocr).pack(side="left", padx=14)

        ttk.Label(line2, text="Lang:").pack(side="left")
        ttk.Entry(line2, textvariable=self.lang, width=10).pack(side="left", padx=6)

        ttk.Label(line2, text="Optimize:").pack(side="left")
        ttk.Combobox(line2, values=[0, 1, 2, 3], textvariable=self.optimize, width=4, state="readonly").pack(
            side="left", padx=6
        )

        # Progress
        prog = ttk.LabelFrame(frm, text="Progress")
        prog.pack(fill="x", **pad)

        p1 = ttk.Frame(prog)
        p1.pack(fill="x", padx=10, pady=6)
        ttk.Label(p1, text="Stage:").pack(side="left")
        ttk.Label(p1, textvariable=self.stage).pack(side="left", padx=6)

        p2 = ttk.Frame(prog)
        p2.pack(fill="x", padx=10, pady=6)
        ttk.Label(p2, text="Current:").pack(side="left")
        ttk.Label(p2, textvariable=self.current_file).pack(side="left", padx=6)

        p3 = ttk.Frame(prog)
        p3.pack(fill="x", padx=10, pady=6)
        self.bar = ttk.Progressbar(p3, orient="horizontal", mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        ttk.Label(p3, textvariable=self.percent_text, width=6, anchor="e").pack(side="left", padx=8)

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", **pad)

        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")

        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

        ttk.Label(btns, text="Tip: progress bar shows percent only (no X/N).").pack(side="right")

    def choose_in_dir(self):
        d = filedialog.askdirectory(title="Select folder with images")
        if d:
            self.in_dir.set(d)

    def choose_out_pdf(self):
        f = filedialog.asksaveasfilename(
            title="Save PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if f:
            self.out_pdf.set(f)

    def set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.start_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    def start(self):
        in_dir = Path(self.in_dir.get().strip())
        out_pdf = Path(self.out_pdf.get().strip())

        if not in_dir.exists() or not in_dir.is_dir():
            messagebox.showerror("Error", "Please choose a valid images folder.")
            return
        if not out_pdf.parent.exists():
            try:
                out_pdf.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder:\n{e}")
                return

        if self.use_ocr.get() and not which("ocrmypdf"):
            messagebox.showerror(
                "Error",
                "OCR is enabled, but 'ocrmypdf' is not found in PATH.\n"
                "Install it or disable OCR.",
            )
            return

        self.stop_flag.clear()
        self.stage.set("Starting…")
        self.current_file.set("-")
        self.percent_text.set("0%")
        self.bar.configure(mode="determinate")
        self.bar["value"] = 0

        self.set_busy(True)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def cancel(self):
        self.stop_flag.set()
        self.stage.set("Cancelling…")

    def _emit(self, kind, payload=None):
        self.q.put((kind, payload))

    def _worker(self):
        in_dir = Path(self.in_dir.get().strip())
        out_pdf = Path(self.out_pdf.get().strip())

        do_ocr = self.use_ocr.get()
        lang = self.lang.get().strip() or "rus+eng"
        optimize = int(self.optimize.get())
        dpi = int(self.dpi.get())

        do_perspective = bool(self.perspective.get())
        do_deskew_crop = bool(self.deskew_crop.get())
        split = bool(self.split_spreads.get())

        images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        images.sort(key=lambda p: natural_key(p.name))

        if not images:
            self._emit("error", "No images found in the selected folder.")
            return

        tmp_root = tempfile.mkdtemp(prefix="photos2pdf_")
        tmp_img_dir = Path(tmp_root) / "preprocessed"
        tmp_img_dir.mkdir(parents=True, exist_ok=True)
        tmp_raw_pdf = Path(tmp_root) / "raw.pdf"

        try:
            total = len(images)

            # Stage 1: preprocess (determinate)
            self._emit("stage", ("Preprocessing (auto perspective/deskew/crop)…", "determinate"))
            out_image_paths = []
            counter = 1

            for idx, img_path in enumerate(images, start=1):
                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")

                self._emit("current", img_path.name)

                pages = preprocess_one(
                    img_path,
                    do_perspective=do_perspective,
                    do_deskew_crop=do_deskew_crop,
                    split=split,
                )

                # save as PNG (lossless)
                for page in pages:
                    out_name = f"{counter:05d}.png"
                    out_path = tmp_img_dir / out_name
                    ok = imwrite_unicode(out_path, page)

                    if not ok:
                        raise RuntimeError(f"Failed to write: {out_path}")
                    out_image_paths.append(out_path)
                    counter += 1

                # progress as percent only
                percent = int((idx / total) * 100)
                self._emit("progress", percent)

            # Stage 2: build PDF (indeterminate)
            self._emit("stage", ("Building PDF…", "indeterminate"))
            self._emit("current", "Packing images into PDF")
            build_pdf_from_images(out_image_paths, tmp_raw_pdf, dpi=dpi)

            # Stage 3: OCR (optional, indeterminate)
            if do_ocr:
                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")
                self._emit("stage", ("Running OCR (ocrmypdf)…", "indeterminate"))
                self._emit("current", "This may take a while")

                cmd = ["ocrmypdf", str(tmp_raw_pdf), str(out_pdf), "-l", lang, f"-O", str(optimize),
                       "--deskew", "--rotate-pages"]
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    raise RuntimeError("ocrmypdf failed. Check ocrmypdf + dependencies.")
            else:
                shutil.copyfile(tmp_raw_pdf, out_pdf)

            self._emit("done", str(out_pdf))

        except Exception as e:
            self._emit("error", str(e))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

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
                        self.percent_text.set("…")
                    else:
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                        # keep current value
                elif kind == "current":
                    self.current_file.set(payload)
                elif kind == "progress":
                    # determinate only
                    if str(self.bar.cget("mode")) == "indeterminate":
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                    v = max(0, min(100, int(payload)))
                    self.bar["value"] = v
                    self.percent_text.set(f"{v}%")
                elif kind == "done":
                    self.bar.stop()
                    self.bar.configure(mode="determinate")
                    self.bar["value"] = 100
                    self.percent_text.set("100%")
                    self.stage.set("Done")
                    self.current_file.set(payload)
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
            self.after(100, self._poll_queue)


if __name__ == "__main__":
    # On Windows, OpenCV sometimes needs this to behave well with multiprocessing,
    # but we're using threads, so it's mostly fine.
    app = App()
    app.mainloop()
