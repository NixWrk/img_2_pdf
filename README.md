# img_2_pdf

This repository now contains a practical document-processing toolkit focused on one main goal:

1. Accept camera captures or imported files/folders.
2. Run document preprocessing and cleanup.
3. Export a clean merged PDF.
4. Add optional OCR later as a controlled extension.

## Current Main App

Run:

```powershell
python camscan_hybrid_tool.py
```

`camscan_hybrid_tool.py` is the current unified variant for your workflow.

## New Unified App (In Progress)

A new package-based unified application is being built under `src/uniscan`.

Run (from repository root):

```powershell
set PYTHONPATH=src
python -m uniscan.cli
```

One-script launcher (recommended on Windows):

```powershell
.\run_uniscan.cmd
```

Or after installation:

```powershell
uniscan
```

Quick workflow (Office Lens style):

1. Open tab `1. Import` (main mode) and load files/folder, or use `2. Scan` for camera capture.
2. Camera opens at max configured preset resolution; preview is optimized for low latency.
3. App switches to `3. Review`: reorder, rotate, deskew, manual corners, before/after check.
4. Postprocess/preprocess is applied after capture/import and can be reapplied in `Review`.
5. Open `4. Export`, choose OCR engine if needed, then save merged PDF or image files.

Current implemented modules in this new app:

1. `Capture`: live preview, single capture, burst capture, camera configuration
2. `Import`: folder/files/PDF import into one session
3. `Pages`: page list management (preview, reorder, select/delete)
4. `Export`: merged PDF and separate image export
5. `Jobs`: background job progress and cancellation

Implementation notes:

1. Session pages are disk-backed (`uniscan` cache) with lazy reads to reduce RAM usage on large batches.
2. `Pages` review now shows `Before/After` preview for preprocessing visibility.
3. Capture and import keep originals first; preprocessing is applied after ingest, not in live preview.
4. Export tab supports OCR engine selection with dependency status checks.
5. `Import` and `Scan` tabs now have independent processing settings/profiles.
5. Searchable PDF is currently wired for `pytesseract`, `OCRmyPDF`, and `PyMuPDF OCR`.
6. `PaddleOCR`, `Surya`, and `MinerU` are available as selectable OCR backends with readiness checks (searchable-PDF wiring pending).

## What The App Does

`camscan_hybrid_tool.py` supports three source modes:

1. `Import folder`
2. `Import files`
3. `Camera capture`

Processing features:

1. Document detection and perspective extraction using third-party logic from `camscan_suhren`:
   `camscan.scanner.main`
2. Postprocessing effects from `camscan_suhren`:
   `None`, `Sharpen`, `Grayscale`, `Black and White`
3. Optional two-page split (`left/right`) for book-like captures.
4. Merged PDF export from all source modes.
5. Quality profiles (`Fast`, `Balanced`, `Best quality`) for practical output control.

Note:

1. OCR is intentionally left as the next stage and is not active in `camscan_hybrid_tool.py` yet.

## Processing Pipeline

For `folder/files` mode:

1. Load input images (and PDF pages if PDF files are provided and `pymupdf` is installed).
2. Optionally detect and extract document contour.
3. Apply selected postprocessing function.
4. Optionally split each page into left/right halves.
5. Convert processed pages into one merged PDF.

For `camera` mode:

1. Capture N shots from selected camera index.
2. Wait configured delay between shots.
3. Apply the same processing pipeline as above.
4. Export merged PDF.

## Setup

Recommended Python:

1. Python `3.11+`

Install dependencies:

```powershell
pip install opencv-python numpy pillow img2pdf pymupdf
```

Optional OCR dependencies in the new app:

```powershell
pip install pytesseract pypdf ocrmypdf paddleocr pymupdf
```

Also install CLI/system tools where needed:

1. Tesseract OCR engine in `PATH` for `pytesseract` and `PyMuPDF OCR` mode.
2. `ocrmypdf` command in `PATH` for `OCRmyPDF` mode.

Experimental engine packages:

1. `Surya` (or `marker` package path that bundles Surya OCR).
2. `MinerU` (`mineru` or `magic_pdf` package).

If you plan to use legacy scripts with OCR, install additionally:

```powershell
pip install ocrmypdf pypdf
```

External OCR tools for legacy OCR scripts:

1. Tesseract OCR
2. Ghostscript
3. qpdf
4. Poppler (`pdftoppm`, `pdfunite`) for `only_tesseract.py` in PDF mode

## How To Run

Main app:

```powershell
python camscan_hybrid_tool.py
```

Alternative app (images/file + optional OCR already integrated):

```powershell
python unified_pdf_tool.py
```

Legacy apps (kept for reference/fallback):

```powershell
python fast.py
python img_2_pdf.py
python only_tesseract.py
python "prepare pdf to tesseract.py"
```

## Script Map

| File | Role |
|---|---|
| `camscan_hybrid_tool.py` | Main hybrid app (`camera + files/folder`) using third-party processing logic from `camscan_suhren` |
| `unified_pdf_tool.py` | Unified app for folder/file workflows with optional OCR path |
| `fast.py` | OCR-focused GUI with batch PDF support |
| `img_2_pdf.py` | Photo-to-PDF app with OpenCV preprocessing and optional OCR |
| `only_tesseract.py` | OCR pipeline using direct `tesseract.exe` calls |
| `imgs_and_pdfs_ocr_fast_STABLE.py` | Stable previous OCR GUI version |
| `prepare pdf to tesseract.py` | PDF conditioning helper before OCR |
| `camscan_suhren/` | Third-party camera scanner project used as source of preprocessing logic |

## Known Limitations

1. OCR in `camscan_hybrid_tool.py` is not enabled yet (planned next).
2. Camera mode is shot-based capture (not a full continuous preview UI).
3. PDF import in hybrid mode requires `pymupdf`.

## Troubleshooting

1. Error about missing `camscan` modules:
   Ensure folder `camscan_suhren` exists directly inside repo root.
2. Cannot open camera:
   Check camera index and close other apps using webcam.
3. PDF import error:
   Install `pymupdf` (`pip install pymupdf`).

## Short Roadmap

1. Add optional OCR to `camscan_hybrid_tool.py` with toggle and language setting.
2. Add stronger camera UX (preview/retake/selection before export).
3. Add job queue for large folder batches.
4. Add tests for hybrid pipeline stages.
