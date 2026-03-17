# UniScan Office Lens PC Plan

## Product Goal

Deliver a PC-first Office Lens analogue:

1. Fast capture flow: `Scan -> Review -> Export`
2. Clear preprocessing UX: visible `Before/After`, presets, manual corner edit
3. Memory-safe handling of large batches (disk-backed pages, streaming pipeline)
4. Searchable PDF with OCR

## Current Status (already implemented)

1. Unified package and launcher
2. Camera capture (single + burst), import, page review, export
3. Background jobs with progress/cancel hook
4. Core tests for session/pipeline/export

## Progress Snapshot (2026-03-17)

1. Memory-safe session pipeline is active: disk-backed pages, lazy reads, stream import/export.
2. Review quality tools are active: before/after preview, manual corners, rotate, auto-deskew.
3. Guided flow is active: step-based tabs and auto-jump to `Review` after capture/import.
4. Office Lens style modes are active: `Document`, `Whiteboard`, `Photo`, `B/W`.
5. In-place page replacement/retake is active for selected page (`Replace Sel...` and `Retake Cam`).
6. OCR core is active: dependency detection and optional searchable PDF export path.
7. Camera health indicator is active (`Closed/Open/Previewing/Error`) in scan controls.

## Implementation Stages

### Stage A: Memory & Storage

Status: Mostly complete, memory regression benchmarking still open.

1. [x] `feat(storage): add cache workspace manager for page assets`
2. [x] `refactor(session): store page originals/processed images on disk with lazy loading`
3. [x] `refactor(import): convert import pipeline to streaming to avoid full-memory batches`
4. [x] `refactor(export): stream export from disk-backed pages where possible`
5. [ ] `test(memory): add high-volume memory regression tests`

### Stage B: Office Lens Flow UX

Status: In progress.

1. [x] `feat(ui-flow): switch to guided flow Scan -> Review -> Export`
2. [x] `feat(ui-scan): simplify capture controls and add camera health state`
3. [x] `feat(ui-review): faster filmstrip with thumbnails from disk cache`
4. [x] `feat(ui-review): retake/replace page in-place`
5. [x] `docs(ui): in-app guidance and quick tips`

### Stage C: Preprocessing Clarity

Status: Complete for current scope.

1. [x] `feat(preprocess-ui): side-by-side before/after panel`
2. [x] `feat(preprocess-presets): Document, Whiteboard, Photo, B/W modes`
3. [x] `feat(preprocess-controls): expose threshold/contrast/denoise sliders`
4. [x] `feat(corners): manual 4-point corner correction`
5. [x] `test(preprocess): deterministic tests for filter and transform chain`

### Stage D: Scan Quality

1. `feat(cv): robust contour fallback and glare reduction`
2. `feat(cv): deskew/orientation correction`
3. `feat(cv): improved text enhancement profile`
4. `feat(cv): smart two-page split center detection`
5. `perf(cv): optimize preview + processing latency`

### Stage E: OCR/Searchable PDF

Status: In progress.

1. [x] `feat(ocr-core): OCR engine abstraction and dependency checks`
2. [x] `feat(ocr-ui): OCR language/profile controls`
3. [~] `feat(ocr-export): searchable PDF text layer` (`pytesseract`, `OCRmyPDF`, `PyMuPDF OCR` wired; others pending)
4. [~] `feat(ocr-engines): add engine selection for OCRmyPDF/PaddleOCR/PyMuPDF/Surya/MinerU`
5. [ ] `test(ocr): integration tests on fixtures`

### Stage F: Production Readiness

1. `feat(recovery): autosave/restore sessions`
2. `feat(io): drag-and-drop + clipboard import`
3. `feat(export): output naming templates and profile presets`
4. `ci: add root lint/test workflow`
5. `docs: user guide + troubleshooting`

## Next Iteration Backlog

### Priority 1: OCR Completion

1. `feat(ocr-engines): wire searchable-pdf for paddleocr`
2. `feat(ocr-engines): wire searchable-pdf for surya`
3. `feat(ocr-engines): wire searchable-pdf for mineru`
4. `test(ocr): add real fixture integration tests (rus/eng, mixed layouts)`
5. `refactor(ocr): add unified OCR profile presets (fast/balanced/quality)`

### Priority 2: Stability and Performance

1. `test(memory): add high-volume memory regression suite`
2. `perf(ui-review): virtualized filmstrip + lazy thumbnail decode`
3. `perf(import): large-batch import stress optimization`
4. `perf(export): chunked OCR/export progress + cancellation hardening`

### Priority 3: UX and Product Readiness

1. `feat(session): autosave + restore unfinished sessions`
2. `feat(io): drag-and-drop import`
3. `feat(io): clipboard image import`
4. `feat(ui-flow): first-run onboarding + contextual tips`
5. `feat(export): naming templates and export profiles`
6. `feat(export): one-click presets (Document PDF, Archive PDF, OCR PDF)`

### Priority 4: Delivery and Docs

1. `ci: add full pipeline (lint + tests + smoke UI checks)`
2. `docs: user guide + troubleshooting matrix by OCR engine`
3. `build: create Windows release bundle (one-click launcher + dependency check)`
