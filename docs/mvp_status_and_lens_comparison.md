# Current MVP Status and Office Lens Comparison

## Purpose

`uniscan` is a Windows-first, pre-OCR document scanner. It acquires pages from camera, image
files, folders, or PDFs; detects and rectifies documents; supports review and cleanup; and exports
processed images or a plain merged PDF.

## Supported entry points

- `python -m uniscan` or `python -m uniscan.cli` launches the GUI.
- `python -m uniscan convert ...` runs the production headless input-to-PDF pipeline.
- `python -m uniscan benchmark-crop ...` compares crop backends.
- `python -m uniscan.office_lens.cli ...` runs Office Lens diagnostics for one image.
- `python camscan_hybrid_tool.py` is a compatibility alias for the package GUI.

## Current architecture

| Module | Responsibility |
|---|---|
| `core/pipeline.py` | Detect, warp, split, and postprocess loaded pages |
| `core/scanner_adapter.py` | Office Lens / UVDoc / OpenCV / legacy backend cascade |
| `core/preprocess.py` | Lens modes, cleanup presets, and deskew |
| `core/spread.py` | Gutter-based two-page split with midpoint fallback |
| `io/loaders.py` | Natural-order image loading and DPI-safe PDF rendering |
| `tools/batch_pipeline.py` | Headless files/folders/PDF → processed images + merged PDF |
| `session` + `storage` | Ordered, disk-backed GUI pages |
| `ui/app.py` | Import → Scan → Review → Export desktop flow |
| `office_lens` | Bundled ONNX classifier and quadrilateral detector |

The default document detector cascade is:

1. `office_lens_onnx` (bundled and offline),
2. `paddleocr_uvdoc` (optional),
3. `cv_hybrid` (OpenCV fallback).

Additional individually selectable backends remain available to the benchmark.

## Implemented capabilities

- Camera preview/capture and mixed image/PDF import.
- Live OpenCV boundary overlay.
- Office Lens ONNX document classification and quad detection.
- Perspective correction and manual four-corner editing.
- Accurate gutter-based two-page split.
- Rotate, deskew, Document/Whiteboard/Photo/B&W modes, and tuning controls.
- Ordered disk-backed sessions with page replace, delete, and reorder.
- Merged PDF and PNG/JPEG/WEBP/TIFF export.
- Headless `convert` flow for automation.
- Deterministic unit/integration coverage, including an end-to-end CLI conversion test.

## Known limitations

- No auto-capture based on frame stability.
- No glare/shadow removal or text-based orientation detection.
- No drag-and-drop, clipboard import, or session restore.
- No Office/OneNote/cloud export.
- OCR is intentionally handled in separate repositories.
- The GUI remains Windows-first; the headless pipeline is the portable automation surface.

## Office Lens comparison

| Capability | UniScan status |
|---|---|
| Document/Whiteboard/Photo/B&W modes | Implemented; no Business Card mode |
| Boundary detection and perspective warp | Implemented with bundled ONNX + fallbacks |
| Live boundary feedback | Implemented with OpenCV worker/smoothing |
| Multi-page review/reorder/replace | Implemented |
| Accurate book-spread split | Implemented |
| PDF and image export | Implemented |
| Headless folder/PDF-to-PDF conversion | Implemented |
| Auto-capture, glare removal, annotations | Not implemented |
| OCR and Office formats | Out of scope |

This file describes the current code. Future work is tracked separately in
`docs/unification_plan.md`.
