"""OCR engine helpers and dependency checks."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

from uniscan.core.pipeline import build_pdf_from_images

OCR_ENGINE_PYTESSERACT = "pytesseract"
OCR_ENGINE_OCRMYPDF = "ocrmypdf"
OCR_ENGINE_PADDLEOCR = "paddleocr"
OCR_ENGINE_PYMUPDF = "pymupdf"
OCR_ENGINE_SURYA = "surya"
OCR_ENGINE_MINERU = "mineru"

OCR_ENGINE_VALUES: tuple[str, ...] = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_MINERU,
)

OCR_ENGINE_LABELS: dict[str, str] = {
    OCR_ENGINE_PYTESSERACT: "pytesseract",
    OCR_ENGINE_OCRMYPDF: "OCRmyPDF",
    OCR_ENGINE_PADDLEOCR: "PaddleOCR",
    OCR_ENGINE_PYMUPDF: "PyMuPDF OCR",
    OCR_ENGINE_SURYA: "Surya",
    OCR_ENGINE_MINERU: "MinerU",
}

SEARCHABLE_PDF_ENGINES: tuple[str, ...] = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PYMUPDF,
)


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


@dataclass(slots=True, frozen=True)
class OcrEngineStatus:
    engine_name: str
    ready: bool
    missing: list[str]
    searchable_pdf: bool

    @property
    def label(self) -> str:
        return OCR_ENGINE_LABELS.get(self.engine_name, self.engine_name)


def _has_module(name: str, import_module) -> bool:
    try:
        import_module(name)
        return True
    except Exception:
        return False


def _has_command(name: str, which_fn) -> bool:
    return bool(which_fn(name) or which_fn(f"{name}.exe"))


def detect_ocr_dependencies(
    *,
    import_module=importlib.import_module,
    which_fn=shutil.which,
) -> OcrDependencyStatus:
    """Backward-compatible checker for pytesseract workflow."""
    return OcrDependencyStatus(
        pytesseract_available=_has_module("pytesseract", import_module),
        pypdf_available=_has_module("pypdf", import_module),
        tesseract_available=_has_command("tesseract", which_fn),
    )


def detect_ocr_engine_status(
    engine_name: str,
    *,
    import_module=importlib.import_module,
    which_fn=shutil.which,
) -> OcrEngineStatus:
    engine = engine_name.strip().lower()
    if engine not in OCR_ENGINE_VALUES:
        raise ValueError(f"Unsupported OCR engine: {engine_name}")

    if engine == OCR_ENGINE_PYTESSERACT:
        deps = detect_ocr_dependencies(import_module=import_module, which_fn=which_fn)
        return OcrEngineStatus(
            engine_name=engine,
            ready=deps.ready,
            missing=deps.missing,
            searchable_pdf=True,
        )

    if engine == OCR_ENGINE_OCRMYPDF:
        missing: list[str] = []
        if not _has_command("ocrmypdf", which_fn):
            missing.append("ocrmypdf")
        if not _has_module("img2pdf", import_module):
            missing.append("img2pdf")
        return OcrEngineStatus(
            engine_name=engine,
            ready=not missing,
            missing=missing,
            searchable_pdf=True,
        )

    if engine == OCR_ENGINE_PYMUPDF:
        missing = []
        if not _has_module("fitz", import_module):
            missing.append("pymupdf(fitz)")
        if not _has_module("pypdf", import_module):
            missing.append("pypdf")
        if not _has_command("tesseract", which_fn):
            missing.append("tesseract")
        return OcrEngineStatus(
            engine_name=engine,
            ready=not missing,
            missing=missing,
            searchable_pdf=True,
        )

    if engine == OCR_ENGINE_PADDLEOCR:
        missing = [] if _has_module("paddleocr", import_module) else ["paddleocr"]
        return OcrEngineStatus(
            engine_name=engine,
            ready=not missing,
            missing=missing,
            searchable_pdf=False,
        )

    if engine == OCR_ENGINE_SURYA:
        has_surya = _has_module("surya", import_module)
        has_marker = _has_module("marker", import_module)
        missing = [] if (has_surya or has_marker) else ["surya/marker"]
        return OcrEngineStatus(
            engine_name=engine,
            ready=not missing,
            missing=missing,
            searchable_pdf=False,
        )

    has_mineru = _has_module("mineru", import_module) or _has_module("magic_pdf", import_module)
    missing = [] if has_mineru else ["mineru(magic_pdf)"]
    return OcrEngineStatus(
        engine_name=engine,
        ready=not missing,
        missing=missing,
        searchable_pdf=False,
    )


def _ensure_engine_ready(status: OcrEngineStatus) -> None:
    if status.ready:
        return
    missing = ", ".join(status.missing) if status.missing else "unknown"
    raise RuntimeError(f"OCR engine '{status.label}' is not ready: missing {missing}")


def _image_paths_to_searchable_pdf_pytesseract(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str,
    import_module,
) -> Path:
    pytesseract = import_module("pytesseract")
    pypdf = import_module("pypdf")

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


def _image_paths_to_searchable_pdf_ocrmypdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str,
    which_fn,
    run_cmd,
    build_pdf_fn,
) -> Path:
    ocrmypdf_cmd = which_fn("ocrmypdf") or which_fn("ocrmypdf.exe")
    if not ocrmypdf_cmd:
        raise RuntimeError("OCRmyPDF command was not found in PATH.")

    with tempfile.TemporaryDirectory(prefix="uniscan_ocrmypdf_") as tmp:
        tmp_pdf = Path(tmp) / "input.pdf"
        build_pdf_fn([Path(p) for p in image_paths], out_pdf=tmp_pdf, dpi=300)
        proc = run_cmd(
            [
                str(ocrmypdf_cmd),
                "--force-ocr",
                "--skip-text",
                "--optimize",
                "0",
                "--language",
                lang,
                str(tmp_pdf),
                str(out_pdf),
            ],
            capture_output=True,
            text=True,
        )
        if int(getattr(proc, "returncode", 1)) != 0:
            stderr = (getattr(proc, "stderr", "") or "").strip()
            stdout = (getattr(proc, "stdout", "") or "").strip()
            details = stderr or stdout or "unknown OCRmyPDF error"
            raise RuntimeError(f"OCRmyPDF failed: {details}")
    return out_pdf


def _image_paths_to_searchable_pdf_pymupdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str,
    import_module,
) -> Path:
    fitz = import_module("fitz")
    pypdf = import_module("pypdf")

    merger = pypdf.PdfMerger()
    streams: list[BytesIO] = []
    try:
        for src_path in image_paths:
            pix = fitz.Pixmap(str(src_path))
            if not hasattr(pix, "pdfocr_tobytes"):
                raise RuntimeError("Current PyMuPDF build has no OCR support (missing Pixmap.pdfocr_tobytes).")
            try:
                page_pdf = pix.pdfocr_tobytes(language=lang)
            except TypeError:
                page_pdf = pix.pdfocr_tobytes()
            stream = BytesIO(page_pdf)
            streams.append(stream)
            merger.append(stream)
        with out_pdf.open("wb") as fh:
            merger.write(fh)
    finally:
        merger.close()
        for stream in streams:
            stream.close()
    return out_pdf


def image_paths_to_searchable_pdf(
    image_paths: Sequence[Path],
    *,
    out_pdf: Path,
    lang: str = "eng",
    engine_name: str = OCR_ENGINE_PYTESSERACT,
    dependency_status: OcrDependencyStatus | None = None,
    engine_status: OcrEngineStatus | None = None,
    import_module=importlib.import_module,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
    build_pdf_fn=build_pdf_from_images,
) -> Path:
    """
    Build searchable PDF using selected OCR engine.

    Implemented searchable-PDF engines:
    - pytesseract
    - ocrmypdf
    - pymupdf
    """
    if len(image_paths) == 0:
        raise ValueError("No image paths to OCR.")

    engine = engine_name.strip().lower()
    if engine not in OCR_ENGINE_VALUES:
        raise ValueError(f"Unsupported OCR engine: {engine_name}")

    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if engine == OCR_ENGINE_PYTESSERACT:
        if dependency_status is None:
            status = detect_ocr_dependencies(import_module=import_module, which_fn=which_fn)
            if not status.ready:
                missing = ", ".join(status.missing) if status.missing else "unknown"
                raise RuntimeError(f"OCR dependencies are not ready: missing {missing}")
        else:
            if not dependency_status.ready:
                missing = ", ".join(dependency_status.missing) if dependency_status.missing else "unknown"
                raise RuntimeError(f"OCR dependencies are not ready: missing {missing}")
        return _image_paths_to_searchable_pdf_pytesseract(
            image_paths,
            out_pdf=out_pdf,
            lang=lang,
            import_module=import_module,
        )

    status = engine_status or detect_ocr_engine_status(
        engine,
        import_module=import_module,
        which_fn=which_fn,
    )
    _ensure_engine_ready(status)

    if engine == OCR_ENGINE_OCRMYPDF:
        return _image_paths_to_searchable_pdf_ocrmypdf(
            image_paths,
            out_pdf=out_pdf,
            lang=lang,
            which_fn=which_fn,
            run_cmd=run_cmd,
            build_pdf_fn=build_pdf_fn,
        )

    if engine == OCR_ENGINE_PYMUPDF:
        return _image_paths_to_searchable_pdf_pymupdf(
            image_paths,
            out_pdf=out_pdf,
            lang=lang,
            import_module=import_module,
        )

    label = OCR_ENGINE_LABELS.get(engine, engine)
    raise NotImplementedError(
        f"Engine '{label}' is detected but searchable PDF export is not wired yet in this build."
    )
