"""Surya OCR engine plugin for OCRmyPDF (scaffold)."""

from ocrmypdf import hookimpl


@hookimpl
def add_options(parser):
    return None


@hookimpl
def check_options(options):
    return None


@hookimpl
def get_ocr_engine():
    raise NotImplementedError(
        "Surya plugin scaffold only. Implementation is added in subsequent commits."
    )
