# UniScan PC Plan (Pre-OCR Layer)

## Product Goal

Deliver a PC-first Office Lens analogue for document preparation:

1. Fast flow: `Import/Scan -> Review -> Export`.
2. Clear preprocessing UX: visible `Before/After`, presets, manual corner edit.
3. Memory-safe handling of large batches (disk-backed pages, streaming pipeline).
4. Stable merged PDF and image export for downstream OCR systems.

## Current Status

1. Unified package and launcher are active.
2. Camera capture, import, page review, and export are active.
3. Background jobs with progress/cancel are active.
4. Core tests for session/pipeline/export are active.

## Implementation Stages

### Stage A: Memory & Storage

1. [x] Cache workspace manager for page assets.
2. [x] Disk-backed session entries with lazy loading.
3. [x] Streaming import to avoid full-memory batches.
4. [x] Streaming export from disk-backed pages.
5. [ ] High-volume memory regression tests.

### Stage B: Guided UX

1. [x] Guided flow: `Import -> Review -> Export`.
2. [x] Simplified scan controls and camera health state.
3. [x] Fast review filmstrip with cached thumbnails.
4. [x] Replace/retake page in-place.
5. [x] In-app guidance.

### Stage C: Preprocessing Quality

1. [x] Side-by-side before/after preview.
2. [x] Presets: `Document`, `Whiteboard`, `Photo`, `B/W`.
3. [x] Threshold/contrast/denoise controls.
4. [x] Manual 4-point corner correction.
5. [x] Deterministic preprocessing tests.

### Stage D: Scan Quality

1. [ ] Robust contour fallback and glare reduction.
2. [ ] Deskew/orientation correction tuning.
3. [ ] Text enhancement profile tuning.
4. [ ] Better two-page split center detection.
5. [ ] Preview and processing latency optimization.

### Stage E: Production Readiness

1. [ ] Autosave/restore sessions.
2. [ ] Drag-and-drop and clipboard import.
3. [ ] Output naming templates and export presets.
4. [ ] CI workflow for lint/test.
5. [ ] User guide and troubleshooting docs.

## Next Iteration Backlog

1. Add memory regression suite for large batches.
2. Optimize filmstrip virtualization and lazy thumbnail decode.
3. Improve import stress performance for large folders.
4. Add autosave + restore unfinished sessions.
5. Add drag-and-drop import.
6. Add export naming templates.
7. Add release checklist and Windows packaging notes.
