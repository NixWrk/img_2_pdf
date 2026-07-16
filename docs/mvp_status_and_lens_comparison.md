# Current MVP status and Microsoft Lens feature boundary

## Purpose

UniScan is a local, Windows-first document preparation tool. It acquires pages from a camera,
images, folders, or PDFs; corrects page geometry; applies optional cleanup; lets the user review
and commit the result; and exports processed images or a plain merged PDF.

UniScan is a pre-OCR tool. It does not recognize text, create searchable PDFs, preserve the text
or vector objects from an imported PDF, or send documents to a cloud service.

## Supported entry points

- `python -m uniscan` or `python -m uniscan.cli` opens the GUI.
- `python -m uniscan convert ...` runs the production headless conversion pipeline.
- `python -m uniscan doctor ...` checks the required runtime, storage, optional models, and,
  when requested, the GUI runtime or a camera.
- `python -m uniscan benchmark-crop ...` compares boundary detectors.
- `python -m uniscan benchmark-quality ...` checks crop quality against a committed baseline.
- `python -m uniscan benchmark-geometry ...` checks orientation, deskew, and dewarp behavior.
- `python -m uniscan.office_lens.cli ...` is the standalone BYOM Office Lens runner.
- `python camscan_hybrid_tool.py` remains a compatibility alias for the package GUI.

## Current architecture

| Module | Responsibility |
|---|---|
| `io/loaders.py` | Natural-order raster, multi-frame TIFF, and streamed PDF loading with per-page pixel limits |
| `core/pipeline.py` | Boundary detection, perspective correction, and spread splitting |
| `core/processing.py` | Shared orientation, deskew, dewarp, cleanup, lighting, and layout controller |
| `core/scanner_adapter.py` | OpenCV boundary policies plus the explicitly selected BYOM backend |
| `export/exporters.py` | Atomic PDF and owned-image publication with cross-process UniScan locks |
| `tools/batch_pipeline.py` | Headless conversion, JSON reporting, and journaled multi-output publication |
| `session` and `storage` | Disk-backed pages, autosave recipes, committed-result fingerprints, and stage cache |
| `ui/app.py` | Workspace, camera, import, processing, page review, and export workflow |
| `office_lens` | Optional BYOM ONNX quad detector and mode classifier |

The production `auto` boundary policy uses the offline OpenCV hybrid detector. Office Lens ONNX
is used only when `office_lens_onnx` is selected explicitly and compatible, lawfully obtained
weights are configured. The other OpenCV policies remain selectable for diagnosis and benchmark
runs.

PaddleOCR UVDoc is a separate dewarp backend, not a boundary detector. It is not included in the
standard dependency set or portable build. Explicit UVDoc use, or an explicitly enabled UVDoc
fallback, may let PaddleOCR initialize or download its model cache.

## Implemented capabilities

- Camera preview and capture, file/folder import, drag-and-drop, and clipboard import.
- JPG/JPEG, PNG, BMP, WEBP, multi-frame TIFF, and streamed PDF input.
- A fail-closed 150,000,000-pixel limit for each input image, TIFF frame, or rendered PDF page.
- Independent input-PDF rendering DPI and output/layout DPI controls.
- Live OpenCV boundary feedback, perspective correction, and manual four-corner editing.
- Conservative gutter-based two-page splitting; uncertain frames remain whole.
- Conservative non-OCR orientation, selectable deskew, validated dewarp, and editable dewarp
  control points.
- Document, Whiteboard, Photo, and B/W cleanup; selectable binarization and safe despeckling;
  optional lighting diagnostics and correction.
- Optional A4 or Letter layout with reproducible margins and alignment.
- Approximate fast preview and full-resolution background Apply. Export uses each page's last
  committed pixels; changing controls without Apply does not silently change the output.
- Ordered, disk-backed sessions with replace, delete, reorder, crash recovery, and a single-writer
  autosave lock.
- Versioned committed processing metadata. A restored result whose pixel fingerprint does not
  match its metadata is treated as stale instead of being trusted.
- Plain merged-PDF and PNG/JPEG/WEBP/TIFF export.
- Atomic direct export and journaled batch publication. Cooperating UniScan processes serialize
  writes to the same targets, and image export preserves files it does not own.
- Structured schema-versioned JSON reports and deterministic crop/geometry regression tools.

## Important limits

- The output PDF is image based: imported PDF text, vectors, bookmarks, and searchability are not
  preserved.
- Auto-capture based on frame stability is not implemented.
- Lighting analysis and opt-in normalization cannot reconstruct detail already clipped by glare.
- Geometry estimators deliberately leave sparse, ambiguous, or already-correct pages unchanged.
- The offline text-line dewarper needs enough usable line evidence; more difficult pages may need
  manual control-point adjustment or the optional UVDoc runtime.
- Office Lens model weights are not distributed. The quad model is required for that backend; its
  classifier is required only when the Office Lens runner is asked to choose a mode automatically.
- Output locks coordinate UniScan processes only. Unrelated applications do not honor them.
- The packaged desktop application is Windows x64. Source-based CLI use on other platforms still
  depends on the platform support of its native dependencies.

## Microsoft Lens feature boundary

This table records UniScan's product boundary; it is not a claim of feature or quality parity
with Microsoft Lens.

| Capability | UniScan status |
|---|---|
| Document/Whiteboard/Photo/B&W cleanup | Implemented; no Business Card workflow |
| Boundary detection and perspective warp | OpenCV default; optional BYOM ONNX quad backend |
| Automatic Office Lens mode classification | Optional BYOM classifier, used only for `mode=auto` |
| Live boundary feedback | Implemented with an OpenCV worker and smoothing |
| Multi-page review, reorder, replace, and recovery | Implemented |
| Book-spread split | Implemented with detected-gutter and midpoint paths |
| Plain PDF and processed-image export | Implemented |
| Headless folder/PDF-to-PDF conversion | Implemented |
| Auto-capture and annotations | Not implemented |
| Robust glare-detail recovery | Not implemented |
| OCR, searchable PDFs, Office formats, and cloud sync | Out of scope |

Future work and acceptance criteria are tracked in [`roadmap.md`](roadmap.md).
`unification_plan.md` is retained as implementation history, not as current behavior.
