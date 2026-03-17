"""Export page images to PDF and separate image files."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from uniscan.core.pipeline import build_pdf_from_images
from uniscan.io.loaders import imwrite_unicode
from uniscan.ocr import image_paths_to_searchable_pdf


def export_pages_as_pdf(
    pages: Sequence[np.ndarray],
    *,
    out_pdf: Path,
    dpi: int = 300,
) -> Path:
    """Export pages to one merged PDF."""
    if len(pages) == 0:
        raise ValueError("No pages to export.")

    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="uniscan_pdf_") as tmp:
        tmp_dir = Path(tmp)
        image_paths: list[Path] = []
        for idx, page in enumerate(pages, start=1):
            page_path = tmp_dir / f"{idx:05d}.png"
            if not imwrite_unicode(page_path, page):
                raise RuntimeError(f"Failed to write temporary page image: {page_path}")
            image_paths.append(page_path)
        build_pdf_from_images(image_paths, out_pdf=out_pdf, dpi=int(dpi))
    return out_pdf


def export_pages_as_files(
    pages: Sequence[np.ndarray],
    *,
    output_dir: Path,
    ext: str = "png",
    base_name: str = "page",
) -> list[Path]:
    """Export pages as separate image files."""
    if len(pages) == 0:
        raise ValueError("No pages to export.")

    ext = ext.lower().lstrip(".")
    if not ext:
        ext = "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for idx, page in enumerate(pages, start=1):
        path = output_dir / f"{base_name}_{idx:05d}.{ext}"
        if not imwrite_unicode(path, page):
            raise RuntimeError(f"Failed to write page image: {path}")
        output_paths.append(path)
    return output_paths


def export_image_paths_as_pdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    dpi: int = 300,
) -> Path:
    """Export image file paths to merged PDF without loading all images in memory."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to export.")
    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf_from_images([Path(p) for p in image_paths], out_pdf=out_pdf, dpi=int(dpi))
    return out_pdf


def export_image_paths_as_searchable_pdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str = "eng",
    engine_name: str = "pytesseract",
) -> Path:
    """Export image paths to searchable PDF via OCR."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to export.")
    return image_paths_to_searchable_pdf(
        [Path(p) for p in image_paths],
        out_pdf=out_pdf,
        lang=lang,
        engine_name=engine_name,
    )


def export_image_paths_as_files(
    image_paths: Sequence[Path],
    *,
    output_dir: Path,
    ext: str = "png",
    base_name: str = "page",
) -> list[Path]:
    """Export images from existing file paths to target format incrementally."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to export.")
    ext = ext.lower().lstrip(".")
    if not ext:
        ext = "png"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    for idx, src in enumerate(image_paths, start=1):
        dst = output_dir / f"{base_name}_{idx:05d}.{ext}"
        src_path = Path(src)
        if src_path.suffix.lower().lstrip(".") == ext:
            shutil.copy2(src_path, dst)
            out_paths.append(dst)
            continue

        data = np.fromfile(str(src_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read source image: {src_path}")
        if not imwrite_unicode(dst, image):
            raise RuntimeError(f"Failed to write page image: {dst}")
        out_paths.append(dst)
    return out_paths
