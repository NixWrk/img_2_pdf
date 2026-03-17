"""File and PDF loading helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

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
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append((f"{pdf_path.stem}_p{page_index:04d}.png", arr))
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
