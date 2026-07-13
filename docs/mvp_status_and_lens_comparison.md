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
| `core/scanner_adapter.py` | OpenCV boundary detection plus optional BYOM backends |
| `core/preprocess.py` | Lens modes, cleanup presets, and deskew |
| `core/spread.py` | Gutter-based two-page split with midpoint fallback |
| `io/loaders.py` | Natural-order image loading and DPI-safe PDF rendering |
| `tools/batch_pipeline.py` | Headless files/folders/PDF → processed images + merged PDF |
| `session` + `storage` | Ordered, disk-backed GUI pages |
| `ui/app.py` | Workspace/Camera/Import options/Export options desktop flow |
| `office_lens` | Optional BYOM ONNX classifier and quadrilateral adapter |

The production boundary-detector policy is:

1. `cv_hybrid` (offline OpenCV default),
2. `office_lens_onnx` only when explicitly selected and externally licensed models are configured.

PaddleOCR UVDoc is an optional dewarp stage, not a boundary detector. Additional OpenCV backends
remain individually selectable for diagnostics and benchmarks.

## Implemented capabilities

- Camera preview/capture and mixed image/PDF import.
- Live OpenCV boundary overlay.
- Optional Office Lens ONNX document classification and quad detection with user-supplied models.
- Perspective correction and manual four-corner editing.
- Accurate gutter-based two-page split.
- Rotate, deskew, Document/Whiteboard/Photo/B&W modes, and tuning controls.
- Ordered disk-backed sessions with page replace, delete, and reorder.
- Merged PDF and PNG/JPEG/WEBP/TIFF export.
- Headless `convert` flow for automation.
- Drag-and-drop, clipboard import, and recoverable session restore.
- Deterministic unit/integration coverage, including an end-to-end CLI conversion test.

## Known limitations

- No auto-capture based on frame stability.
- Lighting diagnostics and opt-in correction are available; robust glare removal remains limited.
- Orientation is geometry-based because OCR/text recognition is intentionally outside the project.
- No Office/OneNote/cloud export.
- OCR is intentionally handled in separate repositories.
- The GUI remains Windows-first; the headless pipeline is the portable automation surface.

## Office Lens comparison

| Capability | UniScan status |
|---|---|
| Document/Whiteboard/Photo/B&W modes | Implemented; no Business Card mode |
| Boundary detection and perspective warp | Implemented with OpenCV default and optional BYOM ONNX |
| Live boundary feedback | Implemented with OpenCV worker/smoothing |
| Multi-page review/reorder/replace | Implemented |
| Accurate book-spread split | Implemented |
| PDF and image export | Implemented |
| Headless folder/PDF-to-PDF conversion | Implemented |
| Auto-capture, glare removal, annotations | Not implemented |
| OCR and Office formats | Out of scope |

This file describes the current code. Future work and acceptance criteria are tracked in
[`roadmap.md`](roadmap.md); `unification_plan.md` is a historical snapshot only.
