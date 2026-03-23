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
2. `Import` and `Scan` are acquisition-only: they load/capture raw pages into session.
3. App switches to `3. Review`: reorder, rotate, deskew, auto crop, manual corners, and side-by-side `Before/After` preview.
4. All processing controls are in `Review`: quick dropdowns (`Lens`, `Post`, `Preset`), `Advanced...` popup sliders, and an `Apply all changes to all files` scope checkbox.
5. Review uses lightweight previews by default (`Full HD`); uncheck it to work directly with full-resolution previews.
6. `Auto Crop...` opens a page browser with auto-detect and manual corner editing for one page or all pages.
7. Open `4. Export`, choose OCR engine if needed, then save merged PDF or image files.

Current implemented modules in this new app:

1. `Capture`: live preview, single capture, burst capture, camera configuration
2. `Import`: folder/files (multi-select)/PDF import into one session
3. `Pages`: page list management (preview, reorder, select/delete)
4. `Export`: merged PDF and separate image export

Implementation notes:

1. Session pages are disk-backed (`uniscan` cache) with lazy reads to reduce RAM usage on large batches.
2. `Pages` review now shows `Before/After` preview for preprocessing visibility.
3. Capture/import keep originals first; processing is only applied from `Review`.
4. Export tab supports OCR engine selection with dependency status checks.
5. `Import` supports multi-file selection and background loading.
6. Import order is preserved end-to-end: folder order, document page order, and mixed import order are kept as selected.
7. Searchable PDF is currently wired for `pytesseract`, `OCRmyPDF`, and `PyMuPDF OCR`.
8. `PaddleOCR`, `Surya`, and `MinerU` are available as selectable OCR backends with readiness checks (searchable-PDF wiring pending).

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

Optional packages for full all-engine OCR benchmark (`surya`/`mineru` paths):

```powershell
pip install surya-ocr mineru ftfy dill omegaconf doclayout-yolo ultralytics
```

Known working stack for one-environment all-engine benchmark on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall `
  "paddleocr==3.4.0" "paddlex==3.4.2" "paddlepaddle==3.1.1" `
  "transformers==4.57.1" "tokenizers==0.22.1" "huggingface-hub==0.34.4" `
  "ftfy==6.3.1" "dill==0.4.1" "omegaconf==2.3.0" `
  "langchain==0.2.17" "langchain-community==0.2.19" "langchain-core==0.2.43"
```

Also install CLI/system tools where needed:

1. Tesseract OCR engine in `PATH` for `pytesseract` and `PyMuPDF OCR` mode.
2. `ocrmypdf` command in `PATH` for `OCRmyPDF` mode.

Experimental engine packages:

1. `Surya` (or `marker` package path that bundles Surya OCR).
2. `MinerU` (`mineru` or `magic_pdf` package).

Benchmark note:

1. `benchmark-ocr --sample-size N` now means total sampled pages (evenly distributed from first to last), not `N` pages per window.

Environment strategy and conflict notes:

1. See `docs/ocr_env_strategy.md` for known dependency conflicts and the one-venv pin set.
2. For latest-release comparisons, run isolated per-engine environments with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -OutputRoot ".\artifacts\ocr_latest_matrix" `
  -Pages "3,9" `
  -SampleSize 1 `
  -Dpi 160 `
  -Recreate
```

Equal-condition comparison package for all OCR engines (canonical per-page text + normalized searchable PDFs):

```powershell
.\.venv\Scripts\python.exe -m uniscan benchmark-ocr-canonical `
  --pdf "J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  --output ".\outputs\ocr_canonical_full_run" `
  --pages 3,9 `
  --sample-size 10 `
  --dpi 200 `
  --strict
```

`--pages` sets exact 1-based pages and overrides `--sample-size` for both `benchmark-ocr` and `benchmark-ocr-canonical`.

Build a readable comparison bundle (report + copied results + per-engine extracted text) in `.\outputs`:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_ocr_results.py `
  --input-root .\artifacts\ocr_latest_matrix_full_run_final `
  --output-root .\outputs
```

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
| `scripts/benchmark_ocr_matrix.ps1` | Creates isolated venv per OCR engine, runs benchmark, writes matrix summary |
| `uniscan benchmark-ocr-canonical` | Runs all engines on equal sampled pages, writes canonical per-page text and normalized searchable PDFs |
| `scripts/compare_ocr_results.py` | Copies one benchmark run to `outputs`, extracts readable text per engine, writes markdown comparison report |
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
