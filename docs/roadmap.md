# UniScan implementation roadmap

This is the single source of truth for planned engineering work after version 0.1.0. Older plan
documents describe how the current baseline was reached; new work is tracked here.

## Product boundary

UniScan is a local, pre-OCR document preparation application for personal use:

```text
acquire/import -> find page -> orient -> deskew -> dewarp -> clean -> review -> PDF/images
```

Text recognition, searchable-PDF assembly, cloud processing, license-inventory automation, and
Authenticode signing are not current product requirements. A geometry or orientation model may
classify an image, but it must not recognize or store text. Manual control points remain available
as a correction layer over automatic dewarping.

Every processing stage must:

- work locally and preserve the source page;
- expose a deterministic `none`/disabled path;
- fail safely by keeping the previous valid image;
- report the selected method, confidence/reason, latency, and fallback;
- produce the same full-resolution result from GUI and CLI settings;
- have synthetic regression coverage before becoming the default.

GPL ScanTailor Advanced code is a design reference only and is not copied into this MIT project.
New code and model weights must have explicit compatible terms.

## Current baseline

The repository already has camera/image/PDF input, Office Lens ONNX and OpenCV page detection,
perspective correction, selectable deskew, offline text-line dewarp, optional PaddleOCR UVDoc,
editable persisted dewarp points, enhancement presets, split-page support, session recovery,
atomic PDF/image export, structured CLI reports, CI, a portable Windows build, and a deterministic
crop benchmark.

## Execution status

- [x] Phase 1 — automatic geometry and its regression corpus.
- [ ] Phase 2 — ScanTailor-style content boxes, page layout, and cleanup.
- [ ] Phase 3 — one processing controller and stage cache.
- [ ] Phase 4 — exception-focused review GUI.
- [ ] Phase 5 — acquisition and performance.
- [ ] Phase 6 — session lifecycle and reliability.
- [ ] Phase 7 — real-document validation and personal release.

## Phase 1 — automatic geometry

Goal: make the geometry path automatic and measurable before adding more cleanup controls.

1. Add independent 0/90/180/270 orientation correction without OCR.
   - Start with a dependency-free layout classifier with conservative confidence and no-op.
   - Keep manual 90-degree rotation and EXIF orientation handling.
   - Leave an adapter boundary for an optional licensed image-orientation model.
2. Add `dewarp=auto`.
   - Estimate the offline text-line model first.
   - Validate the candidate against the unmodified page.
   - Use UVDoc only when explicitly installed/enabled; never download a model unexpectedly.
   - Reject a candidate that worsens geometry or creates excessive blank/clipped borders.
3. Record geometry evidence per page:
   - orientation angle/confidence/reason;
   - skew method/angle/confidence;
   - dewarp requested/selected method and fallback reason;
   - curvature before/after, blank-border change, output-size/aspect change, and latency.
4. Add a versioned synthetic geometry corpus containing upright, sideways, upside-down, skewed,
   curved, straight, sparse, photo, and failure cases.

Exit criteria: automatic mode improves the curved/rotated corpus, leaves straight and ambiguous
pages unchanged, never depends on OCR, and produces deterministic diagnostics in GUI and CLI.

Planned commits:

1. `feat: add conservative non-OCR page orientation`
2. `feat: select and validate automatic dewarp backends`
3. `test: add geometry quality benchmark corpus`

## Phase 2 — ScanTailor-style document output

Goal: improve the document itself after geometry is correct.

1. Detect a content box independently from the physical page box.
2. Add document-wide page layout:
   - target page size or keep original;
   - consistent margins;
   - horizontal/vertical alignment;
   - optional guide values shared by selected pages.
3. Add comparable cleanup algorithms:
   - Otsu global threshold;
   - Sauvola and Wolf adaptive threshold;
   - grayscale/color preservation;
   - conservative despeckle with strength levels;
   - shadow normalization and glare-area detection as separate operations.

Implementation note: Otsu/Sauvola/Wolf, conservative isolated-speck removal, and non-destructive
shadow/glare/clipped-pixel diagnostics are implemented in core, batch report, CLI, and GUI.
4. Add picture/fill protection masks only if real examples show that automatic cleanup damages
   photographs, stamps, diagrams, or handwriting. Do not add OCR-driven zones.
5. Extend preview comparison so each cleanup method can be evaluated at 100% scale.

Exit criteria: mixed page sizes can be exported with consistent layout; binarization and
despeckle improve the cleanup corpus without removing protected details.

Planned commits:

1. `feat: add content boxes and consistent page layout`
2. `feat: add adaptive document binarization`
3. `feat: add measured despeckle and lighting diagnostics`

## Phase 3 — one processing controller

Goal: remove processing-policy duplication and make previews trustworthy.

Implementation status: the shared request/result controller and canonical stage ordering are
implemented for batch CLI and GUI preview/apply. Persistent per-page overrides and the bounded
stage cache remain next.

1. Introduce a GUI-independent page-processing request/result model.
2. Route GUI preview, GUI export, and headless conversion through the same ordered stages and
   diagnostics schema.
3. Persist every per-page override: boundary, orientation, deskew, dewarp model, cleanup, content
   box, layout, and output mode.
4. Add stage fingerprints and a bounded disk cache. Editing a late cleanup stage must not rerun
   page detection or dewarp; changing page geometry must invalidate downstream stages.
5. Add bounded worker-pool processing with deterministic output order and cancellation.

Exit criteria: identical inputs/settings produce equivalent pixels in preview, GUI export, and
CLI; repeated previews reuse valid stages; cancellation leaves no partial published output.

Planned commits:

1. `refactor: unify page processing requests and diagnostics`
2. `feat: add dependency-aware stage cache`
3. `perf: process independent pages with bounded workers`

## Phase 4 — fast review GUI

Goal: optimize the common workflow instead of exposing the pipeline as a wall of controls.

1. Add one primary `Auto optimize` action using the automatic geometry and cleanup policy.
2. Turn thumbnails into page cards with compact issue badges: boundary fallback, uncertain
   orientation, unusual skew, dewarp rejected/low confidence, clipping, blank page, and slow page.
3. Add filters for `All`, `Needs review`, and issue type, plus apply-to-selected/document actions.
4. Keep the main workspace focused on page list, large preview, processing mode, and export.
   Move rare tuning into a page inspector without removing manual corners or dewarp points.
5. Add keyboard navigation, before/after hold-to-compare, zoom/pan preservation, undo/redo, and
   clear progress/cancellation states.
6. Add controller tests and a short Windows interaction checklist; widget smoke tests cover only
   rendering and wiring.

Exit criteria: a normal document can be imported, auto-optimized, reviewed by exception, and
exported without opening advanced controls.

Planned commits:

1. `feat: add auto optimize and page quality flags`
2. `feat: add review filters and batch page actions`
3. `refactor: simplify the document workspace inspector`

## Phase 5 — acquisition and performance

Goal: keep large jobs and cameras responsive.

1. Stream GUI PDF import through the same page iterator used by CLI.
2. Add 100/500-page memory and cancellation regression tests.
3. Virtualize thumbnail decode/rendering and prefetch only nearby pages.
4. Measure stage timings and optimize only demonstrated bottlenecks.
5. Negotiate camera resolution, frame rate, focus, and exposure capabilities and surface readable
   diagnostics; keep unsupported controls hidden.

Exit criteria: memory is bounded for a 500-page job, the interface remains responsive, and camera
failures identify the unavailable capability rather than an opaque backend property.

## Phase 6 — session lifecycle and reliability

Goal: make long personal projects recoverable and predictable.

1. Add explicit New/Open/Save As/Discard recovery actions.
2. Version manifest migrations and quarantine corrupt sessions.
3. Add visible bounded retention for abandoned session directories and stage caches.
4. Test recovery across released manifest versions and failure during every processing stage.
5. Add output naming templates and reusable export presets.

Exit criteria: interrupted work is recoverable, storage cannot grow silently without a bound, and
old sessions either migrate or fail with an actionable message.

## Phase 7 — validation and personal release

Goal: prove the result works on the user's actual documents and Windows machine.

1. Add consented real examples for books, loose paper waves, roller waves, receipts, handwriting,
   diagrams, photographs, sparse pages, and difficult lighting. Keep them private if redistribution
   rights are unavailable; retain the synthetic corpus in CI.
2. Track per-category geometry/cleanup quality, latency, fallback, clipping, and peak memory.
3. Run the manual camera/import/review/export checklist on a clean Windows 11 machine.
4. Verify the portable ZIP, bundled ONNX models, session recovery, and uninstall/removal path.
5. Document known failure modes and which manual correction is appropriate.

Exit criteria: the portable build completes the real personal workflow and benchmark regressions
are visible before a release is accepted.

## Deferred unless the product scope changes

- OCR, searchable PDF, language packs, and text-based orientation.
- Cloud APIs and hosted processing.
- Public installer, Authenticode signing, SBOM, dependency-license inventory, vulnerability
  attestations, and release provenance.
- ScanTailor project import/export compatibility.
- Manual mesh construction; editable automatic-model control points remain supported.

These items can be reconsidered if UniScan becomes a distributed product or another user needs
the artifacts. They must not block the personal-use processing and GUI roadmap above.
