"""File and PDF loading helpers."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# PIL's default decompression-bomb guard is 178 956 970 px.
# We cap renders slightly below that so downstream tools can open
# rendered pages safely.
_MAX_RENDER_PIXELS: int = 150_000_000


def _safe_render_dpi(page_size, requested_dpi: int, max_pixels: int = _MAX_RENDER_PIXELS) -> int:
    """Calculate the highest safe DPI without silently changing the request."""
    if hasattr(page_size, "width") and hasattr(page_size, "height"):
        w_pt = float(page_size.width)
        h_pt = float(page_size.height)
    else:
        w_pt, h_pt = (float(value) for value in page_size)
    if w_pt <= 0 or h_pt <= 0:
        return requested_dpi
    w_px = w_pt / 72.0 * requested_dpi
    h_px = h_pt / 72.0 * requested_dpi
    if w_px * h_px <= max_pixels:
        return requested_dpi
    scale = math.sqrt(max_pixels / (w_px * h_px))
    return max(1, int(requested_dpi * scale))


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
    paths = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (IMG_EXTS | PDF_EXTS)
    ]
    paths.sort(key=lambda p: natural_key(p.name))
    return paths


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert decoded samples to uint8 without saturating 16-bit sources."""
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    if values.size == 0:
        return values.astype(np.uint8)
    if values.dtype == np.bool_:
        return values.astype(np.uint8) * 255

    finite = values.astype(np.float64)
    if np.issubdtype(values.dtype, np.floating):
        finite = np.nan_to_num(finite, nan=0.0, posinf=255.0, neginf=0.0)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if minimum >= 0.0 and maximum <= 1.0:
            finite *= 255.0
        elif not (minimum >= 0.0 and maximum <= 255.0):
            if maximum <= minimum:
                finite.fill(0.0)
            else:
                finite = (finite - minimum) * (255.0 / (maximum - minimum))
    else:
        minimum = float(finite.min())
        maximum = float(finite.max())
        if minimum >= 0.0 and maximum <= 255.0:
            pass
        elif minimum >= 0.0 and maximum <= 65535.0:
            finite *= 255.0 / 65535.0
        elif maximum <= minimum:
            finite.fill(0.0)
        else:
            finite = (finite - minimum) * (255.0 / (maximum - minimum))
    return np.clip(np.rint(finite), 0, 255).astype(np.uint8)


def _pil_frame_to_bgr(frame: Image.Image) -> np.ndarray:
    bands = frame.getbands()
    has_alpha = "A" in bands or (frame.mode == "P" and "transparency" in frame.info)
    if has_alpha:
        rgba = frame.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = np.asarray(Image.alpha_composite(white, rgba).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raw = np.asarray(frame)
    if raw.ndim == 2:
        return cv2.cvtColor(_scale_to_uint8(raw), cv2.COLOR_GRAY2BGR)
    rgb = np.asarray(frame if frame.mode == "RGB" else frame.convert("RGB"))
    return cv2.cvtColor(_scale_to_uint8(rgb), cv2.COLOR_RGB2BGR)


def _opencv_image_to_bgr(image: np.ndarray) -> np.ndarray:
    normalized = _scale_to_uint8(image)
    if normalized.ndim == 2:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    if normalized.ndim == 3 and normalized.shape[2] == 4:
        alpha = normalized[:, :, 3:4].astype(np.float32) / 255.0
        composited = normalized[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.clip(np.rint(composited), 0, 255).astype(np.uint8)
    if normalized.ndim == 3 and normalized.shape[2] == 3:
        return normalized
    raise ValueError(f"Unsupported decoded image shape: {normalized.shape}")


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image as BGR and apply its EXIF orientation when present."""
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            return _pil_frame_to_bgr(oriented)
    except (OSError, UnidentifiedImageError, ValueError):
        # OpenCV remains a useful fallback for formats/plugins Pillow cannot decode.
        data = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return None if decoded is None else _opencv_image_to_bgr(decoded)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Atomically write an image using a unicode-safe encoded byte buffer."""
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            output.write(buf.tobytes())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        return True
    except OSError:
        return False
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _render_pdf_page(page, *, pdf_path: Path, page_index: int, dpi: int) -> LoadedItem:
    """Render one zero-based PDFium page index to a named BGR image."""
    safe_dpi = _safe_render_dpi(page.get_size(), dpi)
    if safe_dpi != dpi:
        raise RuntimeError(
            f"PDF page {page_index + 1} from {pdf_path.name} exceeds the safe pixel limit "
            f"at {dpi} DPI (maximum safe render: {safe_dpi} DPI). Lower --pdf-dpi; "
            "the page was not rendered because silently lowering DPI would change its "
            "physical output size."
        )
    bitmap = page.render(
        scale=safe_dpi / 72.0,
        rev_byteorder=True,
        maybe_alpha=False,
        fill_color=(255, 255, 255, 255),
    )
    try:
        arr = np.array(bitmap.to_numpy(), copy=True)
    finally:
        bitmap.close()
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return f"{pdf_path.name} [p{page_index + 1:04d}]", arr


def iter_pdf_pages(
    pdf_path: Path,
    dpi: int,
    *,
    cancel_cb: CancelCb | None = None,
) -> Iterator[LoadedItem]:
    """Yield PDF pages one at a time without materializing the full document."""
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "PDF import requires pypdfium2. Install with: pip install pypdfium2"
        ) from exc

    doc = pdfium.PdfDocument(pdf_path)
    try:
        for page_index in range(len(doc)):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Cancelled by user.")
            page = doc[page_index]
            try:
                yield _render_pdf_page(page, pdf_path=pdf_path, page_index=page_index, dpi=dpi)
            finally:
                page.close()
    finally:
        doc.close()


def render_pdf_pages(pdf_path: Path, dpi: int) -> list[LoadedItem]:
    """Render all PDF pages to BGR images for backwards-compatible callers."""
    return list(iter_pdf_pages(pdf_path, dpi))


def render_pdf_page_indices(
    pdf_path: Path, page_indices: Iterable[int], dpi: int
) -> list[LoadedItem]:
    """Render selected PDF pages to BGR images without materializing the full document."""
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        raise RuntimeError(
            "PDF import requires pypdfium2. Install with: pip install pypdfium2"
        ) from exc

    pages: list[LoadedItem] = []
    doc = pdfium.PdfDocument(pdf_path)
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"PDF page index out of range: {page_index}")
            page = doc[page_index]
            try:
                pages.append(
                    _render_pdf_page(
                        page,
                        pdf_path=pdf_path,
                        page_index=page_index,
                        dpi=dpi,
                    )
                )
            finally:
                page.close()
    finally:
        doc.close()

    return pages


def _iter_image_frames(
    path: Path,
    *,
    cancel_cb: CancelCb | None = None,
) -> Iterator[LoadedItem]:
    """Yield every frame from an image container (notably multi-page TIFF)."""
    try:
        source = Image.open(path)
    except (OSError, UnidentifiedImageError):
        image = imread_unicode(path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        yield path.name, image
        return

    with source:
        frame_count = int(getattr(source, "n_frames", 1))
        for frame_index in range(frame_count):
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Cancelled by user.")
            source.seek(frame_index)
            frame = ImageOps.exif_transpose(source.copy())
            name = path.name if frame_count == 1 else f"{path.name} [p{frame_index + 1:04d}]"
            yield name, _pil_frame_to_bgr(frame)


def iter_input_items(
    paths: Iterable[Path],
    *,
    pdf_dpi: int,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> Iterator[LoadedItem]:
    """
    Yield a mixed list of image/PDF paths as BGR items.

    Progress callback receives `(current_index, total_count, name)`.
    """
    input_paths = list(paths)
    total = len(input_paths)
    for index, path in enumerate(input_paths, start=1):
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Cancelled by user.")

        ext = path.suffix.lower()
        if ext in IMG_EXTS:
            yield from _iter_image_frames(path, cancel_cb=cancel_cb)
        elif ext in PDF_EXTS:
            yield from iter_pdf_pages(path, dpi=pdf_dpi, cancel_cb=cancel_cb)
        else:
            raise RuntimeError(f"Unsupported input: {path}")

        if on_progress is not None:
            on_progress(index, total, path.name)


def load_input_items(
    paths: Iterable[Path],
    *,
    pdf_dpi: int,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> list[LoadedItem]:
    """Load mixed inputs into memory; prefer ``iter_input_items`` for large batches."""
    return list(
        iter_input_items(
            paths,
            pdf_dpi=pdf_dpi,
            on_progress=on_progress,
            cancel_cb=cancel_cb,
        )
    )
