# img_2_pdf

`img_2_pdf` now contains only the pre-OCR stage:

1. Import/capture pages.
2. Page cleanup (deskew, crop, perspective, enhancement).
3. Build image outputs and merged PDF.

OCR benchmarking, OCR plugins, and searchable-PDF assembly were moved out into dedicated OCR repositories.

## Run

```powershell
python camscan_hybrid_tool.py
```

Package entrypoint:

```powershell
set PYTHONPATH=src
python -m uniscan.cli
```

## Scope of This Repository

- Image/PDF ingestion.
- Document geometry and preprocessing.
- Export to image files and plain merged PDF.
- Crop benchmark tooling.

## Not in This Repository Anymore

- OCR engine matrix benchmark scripts.
- OCRmyPDF plugin bundles.
- OCR comparison reports and OCR-specific execution docs.
- OCR-specific test suites.

## Main Modules

- `src/uniscan/core` - preprocessing and geometry pipeline.
- `src/uniscan/io` - loaders and camera input.
- `src/uniscan/session` - page/session state.
- `src/uniscan/export` - export to files and PDF.
- `src/uniscan/tools/crop_benchmark.py` - crop backend benchmark.

## Quick Setup

```powershell
pip install -e ".[dev]"
```

Minimal runtime dependencies are defined in `pyproject.toml`.
