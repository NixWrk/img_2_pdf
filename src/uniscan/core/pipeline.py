"""Processing pipeline for document pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import img2pdf
import numpy as np

from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.scanner_adapter import scan_with_document_detector
from uniscan.io.loaders import imwrite_unicode

LoadedItem = tuple[str, np.ndarray]
ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


@dataclass(slots=True)
class PipelineOptions:
    detect_document: bool = True
    two_page_mode: bool = False
    postprocess_name: str = "None"


def split_spread(image: np.ndarray) -> list[np.ndarray]:
    """Split a two-page spread into left and right pages."""
    _, width = image.shape[:2]
    if width < 2:
        return [image]
    midpoint = width // 2
    left = image[:, :midpoint]
    right = image[:, midpoint:]
    if left.size == 0 or right.size == 0:
        return [image]
    return [left, right]


def process_loaded_items(
    loaded_items: list[LoadedItem],
    *,
    options: PipelineOptions,
    scanner_root: Path | None = None,
    on_progress: ProgressCb | None = None,
    cancel_cb: CancelCb | None = None,
) -> list[np.ndarray]:
    """Process loaded input items and return page images in order."""
    if options.postprocess_name not in POSTPROCESSING_OPTIONS:
        raise ValueError(f"Unsupported postprocess mode: {options.postprocess_name}")

    postprocess_fn = POSTPROCESSING_OPTIONS[options.postprocess_name]
    pages: list[np.ndarray] = []

    total = len(loaded_items)
    for index, (name, image) in enumerate(loaded_items, start=1):
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Cancelled by user.")

        scan_output = scan_with_document_detector(
            image,
            enabled=options.detect_document,
            scanner_root=scanner_root,
        )
        working = scan_output.warped if scan_output.warped is not None else image
        processed = postprocess_fn(working)
        split_pages = split_spread(processed) if options.two_page_mode else [processed]
        pages.extend(split_pages)
        if on_progress is not None:
            on_progress(index, total, name)

    return pages


def write_pages_to_dir(
    pages: list[np.ndarray],
    out_dir: Path,
    *,
    start_index: int = 1,
) -> list[Path]:
    """Persist pages as sequential PNG files and return path list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    index = start_index
    for page in pages:
        out_path = out_dir / f"{index:05d}.png"
        if not imwrite_unicode(out_path, page):
            raise RuntimeError(f"Failed writing page: {out_path}")
        output_paths.append(out_path)
        index += 1

    return output_paths


def build_pdf_from_images(image_paths: list[Path], out_pdf: Path, dpi: int) -> None:
    """Build merged PDF from image path list."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as file:
        try:
            payload = img2pdf.convert([str(p) for p in image_paths], dpi=dpi)
        except TypeError:
            layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
            payload = img2pdf.convert([str(p) for p in image_paths], layout_fun=layout)
        file.write(payload)
