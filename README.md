# UniScan (`img_2_pdf`)

`uniscan` is a pre-OCR document preparation pipeline. It imports images or PDF pages,
detects and rectifies document boundaries, applies cleanup presets, and exports processed
images and a plain merged PDF.

OCR, searchable-PDF assembly, and OCR engine benchmarking are intentionally out of scope.

## Quick start on Windows

The launcher creates `.venv`, installs the package and starts the GUI:

```powershell
.\run_uniscan.cmd
```

For a manual development setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uniscan
```

`python camscan_hybrid_tool.py` remains as a compatibility alias for the package GUI. It no
longer contains a separate legacy implementation.

![UniScan document workspace](docs/images/workspace-v1.png)

The GUI follows a single-document workspace with persistent add, camera, processing, and export
actions. Design rationale and the next UX iterations are in
[`docs/gui_ux_research.md`](docs/gui_ux_research.md).

## Headless image/PDF-to-PDF pipeline

Process one or more files or folders into a merged PDF:

```powershell
.\.venv\Scripts\python.exe -m uniscan convert `
  --input path\to\images another-file.pdf `
  --output out\document.pdf `
  --images-dir out\pages `
  --mode document
```

Folder contents are processed in natural filename order (`page2` before `page10`). Supported
inputs are JPG, JPEG, PNG, TIFF, WEBP, BMP, and PDF. The same command is available through the
launcher because it forwards CLI arguments:

```powershell
.\run_uniscan.cmd convert --input path\to\images --output out\document.pdf
```

Useful options:

- `--mode {none,document,whiteboard,photo,b/w}` selects the cleanup profile.
- `--backend {auto,office_lens_onnx,cv_hybrid,paddleocr_uvdoc}` fixes detector policy.
- `--strict-detect` fails atomically instead of keeping a page with no detected boundary.
- `--report PATH` selects the per-page JSON run report path.
- `--no-detect` disables boundary detection and perspective correction.
- `--two-page` splits book spreads using gutter detection with midpoint fallback.
- `--illumination-correction` opts into experimental shadow/highlight normalization.
- `--pdf-dpi 300` controls PDF rendering and export DPI.
- `--images-dir DIR --image-format png` also writes processed page images.

The default detector cascade is bundled Office Lens ONNX, optional PaddleOCR UVDoc, then the
OpenCV hybrid fallback. PDF input is streamed one page at a time. PDF, image-directory, and JSON
report outputs are staged and published together, so cancellation/failure preserves prior output.

## Runtime diagnostics

Check Python/runtime modules, both ONNX models, temporary storage, and optionally a camera:

```powershell
.\.venv\Scripts\python.exe -m uniscan doctor
.\.venv\Scripts\python.exe -m uniscan doctor --camera --camera-index 0 --json
```

The GUI restores unfinished sessions from `%LOCALAPPDATA%\UniScan`, supports file drag-and-drop
and clipboard images/files, and runs non-camera diagnostics shortly after startup.

## Office Lens adapter

Run the bundled Office Lens ONNX/OpenCV adapter on a single image and save its diagnostics:

```powershell
.\.venv\Scripts\python.exe -m uniscan.office_lens.cli `
  path\to\document.jpg --mode auto --out office_lens_out
```

Runtime flow:

```text
image -> ONNX classifier -> ONNX quad mask -> OpenCV quad refinement
      -> perspective warp -> cleanup -> PDF-ready image
```

## Crop backend benchmark

```powershell
.\.venv\Scripts\python.exe -m uniscan benchmark-crop `
  --input path\to\images --output out
```

The benchmark defaults to `office_lens_onnx` and writes a PDF for each requested backend. It is
a comparison tool, not the production conversion command.

Ground-truth quality regression (crop success, corner error, latency, and fallback rate):

```powershell
.\.venv\Scripts\python.exe -m uniscan benchmark-quality `
  --input benchmarks\corpus_v1 `
  --output quality-report.json `
  --baseline benchmarks\corpus_v1\baseline.json
```

## Portable Windows artifact

```powershell
.\scripts\build_windows.ps1
```

This produces a versioned x64 ZIP and SHA-256 file under `artifacts`. The frozen executable runs
both the GUI and CLI without development Python. See
[`docs/windows_release.md`](docs/windows_release.md) and the
[`manual smoke checklist`](docs/manual_smoke_checklist.md).

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest --cov=uniscan --cov-fail-under=60 -q
.\.venv\Scripts\python.exe -m coverage report --omit=src/uniscan/ui/app.py --fail-under=80
```

## Main modules

- `src/uniscan/core` — geometry, detector cascade, spread split, and preprocessing primitives.
- `src/uniscan/io` — image/PDF loaders and camera input.
- `src/uniscan/office_lens` — bundled Android-free Office Lens ONNX/OpenCV adapter.
- `src/uniscan/session` and `src/uniscan/storage` — disk-backed GUI session state.
- `src/uniscan/export` — image and merged-PDF export.
- `src/uniscan/tools/batch_pipeline.py` — supported headless production pipeline.
- `src/uniscan/tools/crop_benchmark.py` — crop-backend comparison tool.
- `src/uniscan/tools/quality_benchmark.py` — ground-truth quality regression tool.
- `src/uniscan/ui` — desktop Import → Scan → Review → Export workflow.

The prioritized development roadmap is in [`docs/next_steps.md`](docs/next_steps.md).
