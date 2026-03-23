"""OCR utilities."""

from .canonical import (
    CanonicalOcrResult,
    run_ocr_canonical_package,
    summarize_ocr_canonical_package,
)
from .benchmark import (
    OcrBenchmarkResult,
    resolve_pdf_page_indices,
    run_ocr_benchmark,
    sample_pdf_page_indices,
    summarize_ocr_benchmark,
)
from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_VALUES,
    OcrDependencyStatus,
    OcrEngineStatus,
    detect_ocr_dependencies,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)

__all__ = [
    "OCR_ENGINE_LABELS",
    "OCR_ENGINE_CHANDRA",
    "OCR_ENGINE_MINERU",
    "OCR_ENGINE_OCRMYPDF",
    "OCR_ENGINE_PADDLEOCR",
    "OCR_ENGINE_PYMUPDF",
    "OCR_ENGINE_PYTESSERACT",
    "OCR_ENGINE_SURYA",
    "OCR_ENGINE_VALUES",
    "CanonicalOcrResult",
    "OcrDependencyStatus",
    "OcrEngineStatus",
    "OcrBenchmarkResult",
    "detect_ocr_dependencies",
    "detect_ocr_engine_status",
    "image_paths_to_searchable_pdf",
    "run_ocr_canonical_package",
    "run_ocr_benchmark",
    "resolve_pdf_page_indices",
    "sample_pdf_page_indices",
    "summarize_ocr_canonical_package",
    "summarize_ocr_benchmark",
]
