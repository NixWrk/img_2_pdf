"""OCR utilities."""

from .engine import OcrDependencyStatus, detect_ocr_dependencies, image_paths_to_searchable_pdf

__all__ = ["OcrDependencyStatus", "detect_ocr_dependencies", "image_paths_to_searchable_pdf"]
