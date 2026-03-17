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

## Implementation Stages

### Stage A: Memory & Storage (In Progress)

1. `feat(storage): add cache workspace manager for page assets`
2. `refactor(session): store page originals/processed images on disk with lazy loading`
3. `refactor(import): convert import pipeline to streaming to avoid full-memory batches`
4. `refactor(export): stream export from disk-backed pages where possible`
5. `test(memory): add high-volume memory regression tests`

### Stage B: Office Lens Flow UX

1. `feat(ui-flow): switch to guided flow Scan -> Review -> Export`
2. `feat(ui-scan): simplify capture controls and add camera health state`
3. `feat(ui-review): faster filmstrip with thumbnails from disk cache`
4. `feat(ui-review): retake/replace page in-place`
5. `docs(ui): in-app guidance and quick tips`

### Stage C: Preprocessing Clarity

1. `feat(preprocess-ui): side-by-side before/after panel`
2. `feat(preprocess-presets): Document, Whiteboard, Photo, B/W modes`
3. `feat(preprocess-controls): expose threshold/contrast/denoise sliders`
4. `feat(corners): manual 4-point corner correction`
5. `test(preprocess): deterministic tests for filter and transform chain`

### Stage D: Scan Quality

1. `feat(cv): robust contour fallback and glare reduction`
2. `feat(cv): deskew/orientation correction`
3. `feat(cv): improved text enhancement profile`
4. `feat(cv): smart two-page split center detection`
5. `perf(cv): optimize preview + processing latency`

### Stage E: OCR/Searchable PDF

1. `feat(ocr-core): OCR engine abstraction and dependency checks`
2. `feat(ocr-ui): OCR language/profile controls`
3. `feat(ocr-export): searchable PDF text layer`
4. `test(ocr): integration tests on fixtures`

### Stage F: Production Readiness

1. `feat(recovery): autosave/restore sessions`
2. `feat(io): drag-and-drop + clipboard import`
3. `feat(export): output naming templates and profile presets`
4. `ci: add root lint/test workflow`
5. `docs: user guide + troubleshooting`
