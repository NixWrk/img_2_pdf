# MVP Status and Office Lens Comparison

## Purpose

`uniscan` (repository `img_2_pdf`) is a PC-first desktop document scanner. It covers everything **before OCR**:

1. Acquire pages from camera, image files, folders, or PDFs.
2. Detect document boundaries and rectify perspective / curl (UVDoc by default + 5 OpenCV fallbacks).
3. Review and edit: manual corners, rotate, deskew, lens-mode presets (Document / Whiteboard / Photo / B&W), per-channel preprocessing.
4. Export a merged PDF or individual images.

OCR engines, OCRmyPDF plugins, and OCR benchmark assets were extracted into separate repositories. This repo deliberately stops at "clean image + merged PDF".

## Architecture

### Entry points

- `python -m uniscan.cli` — package entry. With no arguments launches the GUI; with `benchmark-crop` runs the crop-backend comparison CLI.
- `python camscan_hybrid_tool.py` — legacy single-file launcher backed by the vendored `camscan_suhren`.

### Stack

| Layer | Tech |
|---|---|
| Runtime | Python 3.11+, Windows-first (DSHOW camera backend) |
| Image I/O | OpenCV, Pillow, NumPy |
| PDF I/O | PyMuPDF (read), img2pdf (write) |
| GUI | customtkinter, tkinter |
| Optional ML | PaddleOCR `TextImageUnwarping` (UVDoc) |

### Modules

| Module | Responsibility |
|---|---|
| `core/pipeline.py` | Orchestrates load → detect → warp → postprocess → split |
| `core/scanner_adapter.py` | 7 detector backends with caskaded fallback (UVDoc + 5 OpenCV + camscan) |
| `core/preprocess.py` | Lens-mode profiles, presets, deskew, contrast/brightness/denoise |
| `core/postprocess.py` | None / Sharpen / Grayscale / B&W adaptive threshold |
| `core/geometry.py` | Sort 4 corners + perspective warp |
| `io/loaders.py` | Image/PDF loading with DPI-safe PDF render (avoids PIL decompression-bomb) |
| `io/camera_service.py` | OpenCV VideoCapture wrapper with burst capture |
| `session/capture_session.py` | Ordered per-page session |
| `storage/page_store.py` | Disk-backed page assets (original + current + previews + thumbnail) |
| `ui/app.py` | CustomTkinter GUI: Import / Scan / Review / Export tabs |
| `export/exporters.py` | Streaming PDF + multi-format image export |
| `tools/crop_benchmark.py` | CLI tool to compare crop backends on a folder |

### Detector backends

Defined in `core/scanner_adapter.py`. Default active set is `(paddleocr_uvdoc,)` — UVDoc handles both perspective and page curl. CV-only backends are kept as fallbacks and as candidates for live-preview detection:

| Backend | Method | When to use |
|---|---|---|
| `paddleocr_uvdoc` | PaddleOCR `TextImageUnwarping` | Default; handles curl + perspective |
| `uvdoc` | Raw UVDoc model (no PaddleOCR wrapper) | Same model, different entry |
| `opencv_quad` | Contours + 4-point approximation | Live preview, fast |
| `opencv_hough` | HoughLines → quad intersection | Pages with strong straight edges |
| `opencv_minrect` | minAreaRect on convex hull | Rotated rectangles |
| `cv_hybrid` | Best-score among quad/hough/minrect | Highest CV-only quality |
| `camscan` | Vendored `camscan_suhren` scanner | Legacy/compat |

## Capabilities

### Input

- Camera (Windows DSHOW): live preview, single capture, burst with delay
- Image files: jpg, jpeg, png, tif, tiff, webp, bmp
- Folder with natural sort
- PDF (PyMuPDF), with automatic DPI downscale when render would exceed 150 Mpx

### Processing

- Auto document detection on import and capture (UVDoc by default)
- Manual 4-corner editing with drag handles, optional auto-detect inside the editor
- Auto-deskew (minAreaRect)
- Rotate ±90°
- Lens modes (Document / Whiteboard / Photo / B&W) + 5 preprocess presets
- Sliders: contrast, brightness, denoise, B&W threshold
- Postprocess: None / Sharpen / Grayscale / B&W adaptive threshold
- Two-page spread split (currently naive 50/50; accurate gutter detection arriving in this MVP)
- Replace page from file, retake page from camera
- Apply-to-selected / apply-to-all, undo via raw original

### Storage and session

- `PageStore` writes each page to a per-session temp directory: `original.png` + `current.png` + two preview JPGs + thumbnail JPG. Cleaned on close.
- `CaptureSession` keeps page order, supports move/select/remove/replace.
- Background jobs with progress and cancellation for long imports / exports.

### Export

- Merged PDF (img2pdf) with fixed DPI
- Individual images (png/jpg/jpeg/webp/tif)
- Scope: all pages or selected pages
- Streaming: image paths are passed to img2pdf without loading all pages into RAM

### Tooling

- `uniscan benchmark-crop --input <dir> --output <dir>` compares crop backends on one folder, writes one PDF per backend.
- 12 deterministic test modules in `tests/`.

## Comparison with Office Lens

| Office Lens capability | `uniscan` | Notes |
|---|---|---|
| Lens modes (Document / Whiteboard / Photo / Business Card) | Partial | 3 of 4 — no Business Card |
| Document boundary detection | Yes (stronger) | UVDoc + 5 CV fallbacks; OL has classical CV only |
| Live edge detection in viewfinder | **No → arriving in this MVP** | Currently a stub; preview shows raw frame only |
| Auto-capture on stability | No | Not in MVP scope |
| Perspective correction | Yes | + page-curl correction via UVDoc (OL cannot) |
| Manual corner edit | Yes | Drag handles + in-editor auto-detect |
| Multi-page session | Yes | |
| Reorder / delete / replace pages | Yes | + retake from camera |
| Rotate | Yes | |
| Auto-deskew | Yes | minAreaRect-based |
| Color filters | Partial | None / Gray / B&W / Sharpen |
| Glare removal | No | Stage D in `unification_plan.md` |
| Shadow removal | No | Stage D |
| Two-page spread split (books) | **Naive → accurate in this MVP** | Switching to gutter detection |
| Export PDF / JPG / PNG | Yes | + WEBP, TIF |
| Export Word / PPT / OneNote | No | Out of scope |
| OCR / text recognition | No | Extracted to a separate repo |
| Business card → contact | No | |
| QR / barcode | No | |
| Read Aloud / Immersive Reader | No | |
| Drag-and-drop import | No | Backlog in Stage E |
| Clipboard import | No | Backlog |
| Autosave / session restore | No | Backlog |
| Naming templates | No | Backlog |
| Cloud sync | No | Not relevant on desktop |
| Annotations / signature | No | |
| Watermark | No | |
| Windows installer (exe/msi) | No | Backlog |
| CI workflow | No | Backlog |
| Crop-backend batch benchmark | Yes (extra) | Office Lens does not offer this |

### Where `uniscan` already beats Office Lens

- UVDoc rectification for curved book pages — OL cannot fix page curl.
- Cascade of CV detectors with fallback when ML refuses.
- Headless `benchmark-crop` for comparing backends on a real folder.
- Streaming export keeps RAM flat on large batches.
- DPI-safe PDF rendering with PIL decompression-bomb guard.

### Where `uniscan` lags Office Lens

- Mobile UX is not available (desktop-only).
- No live edge feedback in the viewfinder (closing this gap in the MVP).
- No glare or shadow removal.
- No export to Office formats.
- No auto-capture on frame stability.
- No business-card / QR / OCR-driven features (last one is intentional).

## MVP scope being delivered

This MVP closes the two biggest functional gaps relative to Office Lens:

1. **Live edge detection** — green quad drawn over the camera viewfinder, updated 5–10 times per second on a worker thread, with EMA smoothing and TTL.
2. **Accurate two-page spread split** — gutter detection via vertical darkness + edge + content profiles, validated by left/right balance and edge continuity. Falls back to midpoint when confidence is low.
3. **Boundary detection becomes a mandatory stage for every source**. Pipeline now preserves the raw image alongside the warped result and the detected contour. Review shows raw + green overlay as `before`, warped + postprocess as `after`. Manual corners now edit the raw image directly.

## Roadmap after this MVP

From `docs/unification_plan.md` Stages D and E:

- Glare reduction, shadow removal
- Auto orientation (text-based)
- Drag-and-drop and clipboard import
- Autosave / session restore
- Naming templates
- CI workflow (lint + pytest on push)
- Windows packaging (PyInstaller / briefcase)
- User guide

Optional later improvements:

- Auto-capture on stable, sharp frame
- Headless batch CLI for "folder in → one PDF out"
- High-volume memory regression suite
