"""OCR utilities."""

from .engine import (
    OCR_ENGINE_LABELS,
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
    "OCR_ENGINE_MINERU",
    "OCR_ENGINE_OCRMYPDF",
    "OCR_ENGINE_PADDLEOCR",
    "OCR_ENGINE_PYMUPDF",
    "OCR_ENGINE_PYTESSERACT",
    "OCR_ENGINE_SURYA",
    "OCR_ENGINE_VALUES",
    "OcrDependencyStatus",
    "OcrEngineStatus",
    "detect_ocr_dependencies",
    "detect_ocr_engine_status",
    "image_paths_to_searchable_pdf",
]
