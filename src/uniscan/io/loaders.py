"""File and PDF loading helpers."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

# PIL's default decompression-bomb guard is 178 956 970 px.
# We cap renders slightly below that so downstream tools can open
# rendered pages safely.
_MAX_RENDER_PIXELS: int = 150_000_000


def _safe_render_dpi(page_rect, requested_dpi: int, max_pixels: int = _MAX_RENDER_PIXELS) -> int:
    """Return the highest integer DPI ≤ *requested_dpi* that keeps the
    rendered page within *max_pixels* total pixels.

    This prevents PIL ``DecompressionBombError`` in downstream tools
    that open rendered PNG pages.
    """
    w_pt: float = page_rect.width   # page width in PDF points (1 pt = 1/72 in)
    h_pt: float = page_rect.height
    if w_pt <= 0 or h_pt <= 0:
        return requested_dpi
    w_px = w_pt / 72.0 * requested_dpi
    h_px = h_pt / 72.0 * requested_dpi
    if w_px * h_px <= max_pixels:
        return requested_dpi
    scale = math.sqrt(max_pixels / (w_px * h_px))
    safe_dpi = max(1, int(requested_dpi * scale))
    warnings.warn(
        f"PDF page ({w_pt:.0f}×{h_pt:.0f} pt) at {requested_dpi} DPI would produce "
        f"{w_px * h_px / 1_000_000:.0f} Mpx — capping to {safe_dpi} DPI "
        f"({w_pt / 72 * safe_dpi:.0f}×{h_pt / 72 * safe_dpi:.0f} px) "
        f"to stay below PIL decompression-bomb limit.",
        stacklevel=4,
    )
    return safe_dpi

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
PDF_EXTS = {".pdf"}

LoadedItem = tuple[str, np.ndarray]
ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def natural_key(value: str) -> list[int | str]:
    """Natural sorting helper for file names."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def list_supported_in_folder(folder: Path) -> list[Path]:
    """List supported image and PDF files in a folder, naturally sorted."""
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid input folder: {folder}")
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (IMG_EXTS | PDF_EXTS)]
    paths.sort(key=lambda p: natural_key(p.name))
    return paths


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read image path using unicode-safe bytes decode path."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Write image path using unicode-safe bytes encode path."""
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def render_pdf_pages(pdf_path: Path, dpi: int) -> list[LoadedItem]:
    """Render PDF pages to BGR images."""
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    pages: list[LoadedItem] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index, page in enumerate(doc, start=1):
            safe_dpi = _safe_render_dpi(page.rect, dpi)
            pix = page.get_pixmap(dpi=safe_dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append((f"{pdf_path.name} [p{page_index:04d}]", arr))
    finally:
        doc.close()

    return pages


def render_pdf_page_indices(pdf_path: Path, page_indices: Iterable[int], dpi: int) -> list[LoadedItem]:
    """Render selected PDF pages to BGR images without materializing the full document."""
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    pages: list[LoadedItem] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= doc.page_count:
                raise IndexError(f"PDF page index out of range: {page_index}")
            page = doc[page_index]
            safe_dpi = _safe_render_dpi(page.rect, dpi)
            pix = page.get_pixmap(dpi=safe_dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append((f"{pdf_path.name} [p{page_index + 1:04d}]", arr))
    finally:
        doc.close()

    return pages


def load_input_items(
    paths: Iterable[Path],
    *,
    pdf_dpi: int,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> list[LoadedItem]:
    """
    Load a mixed list of image/PDF paths into in-memory BGR items.

    Progress callback receives `(current_index, total_count, name)`.
    """
    input_paths = list(paths)
    total = len(input_paths)
    items: list[LoadedItem] = []

    for index, path in enumerate(input_paths, start=1):
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Cancelled by user.")

        ext = path.suffix.lower()
        if ext in IMG_EXTS:
            image = imread_unicode(path)
            if image is None:
                raise RuntimeError(f"Cannot read image: {path}")
            items.append((path.name, image))
        elif ext in PDF_EXTS:
            items.extend(render_pdf_pages(path, dpi=pdf_dpi))
        else:
            raise RuntimeError(f"Unsupported input: {path}")

        if on_progress is not None:
            on_progress(index, total, path.name)

    return items
