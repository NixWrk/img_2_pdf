"""OCR engine helpers and dependency checks."""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence


@dataclass(slots=True, frozen=True)
class OcrDependencyStatus:
    pytesseract_available: bool
    pypdf_available: bool
    tesseract_available: bool

    @property
    def ready(self) -> bool:
        return self.pytesseract_available and self.pypdf_available and self.tesseract_available

    @property
    def missing(self) -> list[str]:
        missing: list[str] = []
        if not self.pytesseract_available:
            missing.append("pytesseract")
        if not self.pypdf_available:
            missing.append("pypdf")
        if not self.tesseract_available:
            missing.append("tesseract")
        return missing


def detect_ocr_dependencies(
    *,
    import_module=importlib.import_module,
    which_fn=shutil.which,
) -> OcrDependencyStatus:
    try:
        import_module("pytesseract")
        pytesseract_available = True
    except Exception:
        pytesseract_available = False

    try:
        import_module("pypdf")
        pypdf_available = True
    except Exception:
        pypdf_available = False

    tesseract_available = bool(which_fn("tesseract") or which_fn("tesseract.exe"))
    return OcrDependencyStatus(
        pytesseract_available=pytesseract_available,
        pypdf_available=pypdf_available,
        tesseract_available=tesseract_available,
    )


def _ensure_ocr_ready(status: OcrDependencyStatus) -> None:
    if status.ready:
        return
    missing = ", ".join(status.missing) if status.missing else "unknown"
    raise RuntimeError(f"OCR dependencies are not ready: missing {missing}")


def image_paths_to_searchable_pdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str = "eng",
    dependency_status: OcrDependencyStatus | None = None,
    import_module=importlib.import_module,
) -> Path:
    """Build searchable PDF via pytesseract and merge pages via pypdf."""
    if len(image_paths) == 0:
        raise ValueError("No image paths to OCR.")

    status = dependency_status or detect_ocr_dependencies(import_module=import_module)
    _ensure_ocr_ready(status)

    pytesseract = import_module("pytesseract")
    pypdf = import_module("pypdf")

    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    merger = pypdf.PdfMerger()
    streams: list[BytesIO] = []
    try:
        for src_path in image_paths:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(str(src_path), extension="pdf", lang=lang)
            stream = BytesIO(pdf_bytes)
            streams.append(stream)
            merger.append(stream)
        with out_pdf.open("wb") as fh:
            merger.write(fh)
    finally:
        merger.close()
        for stream in streams:
            stream.close()

    return out_pdf
