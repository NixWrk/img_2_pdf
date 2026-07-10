# Changelog

All notable changes to UniScan are documented here. The project follows Semantic Versioning.

## [Unreleased]

### Added

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

- Independent boundary, deskew, and local-dewarp stages with per-page JSON diagnostics.
- Offline text-line dewarp for curved or wavy pages, with confidence-based no-op fallback.
- Persisted per-page dewarp control points with side-by-side corrected preview.
- Selectable hybrid, Hough-line, and foreground-box deskew estimators.
- Direct CLI access to the OpenCV quad, Hough, and minimum-rectangle boundary backends.

### Changed

- GUI processing now exposes page-wave removal, and Page tools allows choosing the deskew
  estimator before applying automatic rotation correction.
- Dewarp remains automatic-first while allowing correction of the generated model through control
  points; OCR remains out of scope.

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
