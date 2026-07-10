# Changelog

All notable changes to UniScan are documented here. The project follows Semantic Versioning.

## [Unreleased]

### Added

- Independent boundary, deskew, and local-dewarp stages with per-page JSON diagnostics.
- Offline text-line dewarp for curved or wavy pages, with confidence-based no-op fallback.
- Selectable hybrid, Hough-line, and foreground-box deskew estimators.
- Direct CLI access to the OpenCV quad, Hough, and minimum-rectangle boundary backends.

### Changed

- GUI processing now exposes page-wave removal, and Page tools allows choosing the deskew
  estimator before applying automatic rotation correction.
- Dewarp scope is explicitly automatic-only: no manual mesh or per-page curve editor is planned.

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
