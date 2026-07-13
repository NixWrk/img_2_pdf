# Changelog

All notable changes to UniScan are documented here. The project follows Semantic Versioning.

## [Unreleased]

### Added

- Separate input-render and output-layout PDF DPI controls, plus fail-closed pixel limits for
  raster images, every TIFF/PDF page, standard-page layout, restored GUI pages, and stage-cache
  entries.
- Versioned per-page processing recipes, diagnostics, and current-pixel fingerprints in GUI
  autosave state, with schema migration and stale crash-window metadata rejection.
- Single-writer GUI session locking and shared cross-process locks for direct and batch output
  publication.
- Conservative local 0/90/180/270 page orientation based on layout and baseline evidence, with no
  OCR dependency and confidence-based no-op diagnostics.
- Validated automatic dewarp selection with curvature/artifact metrics and an explicitly enabled
  optional UVDoc fallback.
- Versioned generated geometry corpus and CLI regression benchmark for orientation, deskew,
  dewarp safety, and latency.
- Independent content-box detection and optional A4/Letter page layout with consistent margins
  and alignment.
- Dependency-free Otsu, Sauvola, and Wolf binarization; isolated-speck removal that protects nearby
  punctuation; and separate shadow, glare, and clipped-pixel diagnostics.
- Batch CLI and JSON reporting for binarization parameters, despeckle safety counts, and optional
  lighting evidence.
- GUI controls and live preview for document binarization, despeckle, A4/Letter layout,
  margins/alignment, plus on-demand lighting analysis for the selected page.
- One GUI-independent page-processing controller now owns the canonical orientation, deskew,
  dewarp, cleanup, lighting, and layout order for both batch conversion and GUI preview/apply.
- Binary A4/Letter layout now preserves strict black/white pixels during resizing.
- Atomic bounded lossless stage cache with pixel/options/upstream fingerprints, dependency-aware
  invalidation, corrupt-entry fallback, GUI persistence/clear action, optional CLI persistence, and
  hit/miss/write/eviction diagnostics.
- Fail-closed release metadata/x64 checks, dependency-license compatibility and frozen-payload
  audit with runtime notices, native GUI smoke, and draft-only tagged releases.

- Independent boundary, deskew, and local-dewarp stages with per-page JSON diagnostics.
- Offline text-line dewarp for curved or wavy pages, with confidence-based no-op fallback.
- Persisted per-page dewarp control points with side-by-side corrected preview.
- Selectable hybrid, Hough-line, and foreground-box deskew estimators.
- Direct CLI access to the OpenCV quad, Hough, and minimum-rectangle boundary backends.

### Changed

- GUI PDF import now streams pages through disk staging, while full-resolution Apply processing
  runs in a cancellable worker and commits page generations transactionally with stale-revision
  and rollback checks.
- Runtime diagnostics now distinguish the required Office Lens quad model from the optional
  classifier used only by uto cleanup mode.
- GUI processing now exposes page-wave removal, and Page tools allows choosing the deskew
  estimator before applying automatic rotation correction.
- Dewarp remains automatic-first while allowing correction of the generated model through control
  points; OCR remains out of scope.
- Switched PDF rendering from AGPL/commercial PyMuPDF to permissively licensed `pypdfium2`.
- Made `cv_hybrid` the redistributable default boundary detector; Office Lens remains an optional
  bring-your-own-model adapter and UVDoc is only a dewarp stage.
- Portable builds now carry a dedicated end-user README and only Windows x64 TkDND payloads.

### Fixed

- Preserved camera configuration after failed resolution changes, reused one detected gutter for
  raw/warped spread pairs, and made boundary/fallback status describe the result actually kept.
- Added cancellation checkpoints around native processing, cache, rendering, staging, and
  publication; excluded unidentified UVDoc generations from persistent downstream cache hits.
- Prevented concurrent UniScan writers from losing PDF, report, or neighbouring image updates;
  rejected link-like and multiply linked lock/target paths before mutation and made recovery use
  canonical output identities.
- Restored compatibility with the minimum supported pypdfium2 API, bounded exact PDF render
  allocations, and reported each spread page's own duration instead of cumulative elapsed time.
- Prevented report/output collisions from overwriting inputs or replacing directories with files.
- Corrected PDF DPI/page dimensions and made direct PDF/image exports atomic, cancellable, and free
  of stale pages.
- Fixed symmetric 45-degree quad ordering, full-frame OpenCV false positives, deskew clipping,
  negative-brightness inversion, disjoint Office Lens candidate selection, and redundant inference.
- Preserved 16-bit tonal range, composited alpha on white, imported every TIFF frame, and made image
  persistence atomic; oversized PDF renders now fail instead of changing physical page size.
- Kept Whiteboard color, removed double B/W thresholding, aligned Wolf GUI/CLI defaults, and added
  complete deskew/reproduction diagnostics to batch reports.
- Unified full-resolution GUI preview/apply processing, exported durable per-page applied pixels,
  honored cancellation, cleared stale replacement metadata, and made partial session recovery
  transactional without pruning quarantined source pages.
- Replaced wildcard backup cleanup with exact journal-owned, cross-process-locked recovery for
  image directories, so similarly named user files and folders are never treated as UniScan debris.
- Passed release tags to PowerShell as data, kept dependency-only license audits portable across
  CI operating systems, and pinned every metadata override to a reviewed dependency version.
- Invalidated persistent stage-cache entries produced by the previous processing algorithms.

### Removed

- Removed extracted Office Lens weights/metadata from the current tree and every built artifact;
  public mirrors still require the separately documented Git-history purge.

## [0.1.0] - 2026-07-10

### Added

- Streaming mixed image/PDF conversion with atomic PDF, image-set, and JSON report publishing.
- Explicit detector policy, strict detection mode, cancellation, and per-page fallback reporting.
- Crash-safe GUI session autosave/restore, file drag-and-drop, and clipboard import.
- EXIF orientation handling and opt-in uneven-lighting correction.
- Versioned synthetic crop-quality corpus with Office Lens baseline metrics.
- Runtime diagnostics, cross-platform CI, wheel verification, and portable Windows packaging.

### Changed

- Raised total automated coverage above 60% and non-GUI coverage above 80%.
- Normalized formatting and repository line-ending policy.
- Replaced the numbered GUI wizard as the default flow with a persistent document workspace,
  quick add/camera/export actions, reachable processing controls, and task cancellation.

### Removed

- Unused tracked NAPS2 Windows executable.
