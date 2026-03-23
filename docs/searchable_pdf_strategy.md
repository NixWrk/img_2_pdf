# Searchable PDF Strategy Per OCR Engine

This file describes how each OCR engine in UniScan can produce searchable PDFs.

## Current Engine Matrix

1. `pytesseract`
2. `ocrmypdf`
3. `pymupdf`
4. `paddleocr`
5. `surya`
6. `mineru`
7. `chandra`

## Built-in Searchable PDF Paths (already wired in code)

1. `pytesseract`
   - Method: `pytesseract.image_to_pdf_or_hocr(..., extension="pdf")` per page + merge.
   - In code: `src/uniscan/ocr/engine.py` (`_image_paths_to_searchable_pdf_pytesseract`).
2. `ocrmypdf`
   - Method: build temporary input PDF from images, then run `ocrmypdf --force-ocr`.
   - In code: `src/uniscan/ocr/engine.py` (`_image_paths_to_searchable_pdf_ocrmypdf`).
3. `pymupdf`
   - Method: `fitz.Pixmap.pdfocr_tobytes()` per page + merge.
   - In code: `src/uniscan/ocr/engine.py` (`_image_paths_to_searchable_pdf_pymupdf`).

## Plugin-based Searchable PDF Paths (OCRmyPDF bridge)

UniScan now supports a generic bridge for non-native engines:

1. Build temporary PDF from page images.
2. Run `ocrmypdf --plugin <module> --force-ocr --language <lang>`.
3. Output searchable PDF.

In code: `src/uniscan/ocr/engine.py` (`_image_paths_to_searchable_pdf_ocrmypdf_plugin`).

### Engine-by-engine status

1. `paddleocr`
   - Ready solution found: yes.
   - Plugin module candidates:
     - `ocrmypdf_paddleocr`
   - Local repos in this workspace:
     - `OCRmypdf_plugins/OCRmyPDF-PaddleOCR-main`
     - `OCRmypdf_plugins/ocrmypdf-paddleocr-master`
2. `surya`
   - Ready plugin package: not confirmed in this repo set.
   - Available local bridge script:
     - `OCRmypdf_plugins/Ocrmypdf+surya/ocrmypdf_with_surya.py`
   - Plugin module candidates in UniScan:
     - `ocrmypdf_surya`, `ocrmypdf_with_surya`
3. `mineru`
   - Ready plugin package: not confirmed.
   - Plugin module candidates in UniScan:
     - `ocrmypdf_mineru`, `ocrmypdf_magic_pdf`
4. `chandra`
   - Ready plugin package: not confirmed.
   - Plugin module candidate in UniScan:
     - `ocrmypdf_chandra`

## Overriding plugin module names

You can force plugin module names per engine with environment variables:

1. `UNISCAN_OCRMYPDF_PLUGIN_PADDLEOCR`
2. `UNISCAN_OCRMYPDF_PLUGIN_SURYA`
3. `UNISCAN_OCRMYPDF_PLUGIN_MINERU`
4. `UNISCAN_OCRMYPDF_PLUGIN_CHANDRA`

Examples:

```powershell
$env:UNISCAN_OCRMYPDF_PLUGIN_SURYA = "my_surya_plugin"
$env:UNISCAN_OCRMYPDF_PLUGIN_CHANDRA = "ocrmypdf_chandra_custom,ocrmypdf_chandra"
```

## Notes on "ready solutions"

Based on local plugin repositories currently present in this workspace:

1. Confirmed: PaddleOCR via OCRmyPDF plugin.
2. Confirmed local bridge script: Surya + OCRmyPDF internals (`Ocrmypdf+surya`).
3. Not confirmed as ready package in local set: MinerU plugin, Chandra plugin.

This means:

1. You can already run searchable PDF via `paddleocr` if `ocrmypdf_paddleocr` is installed.
2. For `surya`, `mineru`, `chandra`, UniScan has plugin hooks and module detection, but actual plugin package must be installed/provided.
