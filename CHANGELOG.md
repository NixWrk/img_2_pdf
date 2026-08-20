# Changelog

All notable changes to UniScan are documented here. The project follows Semantic Versioning.

## [Unreleased]

### Added

- Model-tournament schema v2 preserves candidate aspect ratio during metric alignment, records the
  original/reference dimensions and aspect error, and macro-averages equally across categories.
- Model-tournament schema v3 separates the exact manifest-file SHA-256 from a portable canonical
  identity that excludes local provenance paths; new official-metric sidecars bind to that identity.
- Automatic dewarp now rejects a perspective regression larger than the same bounded noise allowance
  used for curvature safety, even when another geometry signal improves.

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
- Built-in Otsu, Sauvola, and Wolf binarization; isolated-speck removal that protects nearby
  punctuation; and separate shadow, glare, and clipped-pixel diagnostics.
- Batch CLI and JSON reporting for binarization parameters, despeckle safety counts, and optional
  lighting evidence.
- GUI controls and live preview for document binarization, despeckle, A4/Letter layout,
  margins/alignment, plus on-demand lighting analysis for the selected page.
- One GUI-independent page-processing controller now owns the canonical orientation, dewarp,
  deskew, lighting, cleanup, and layout order for both batch conversion and GUI preview/apply.
- Binary A4/Letter layout now preserves strict black/white pixels during resizing.
- Atomic bounded lossless stage cache with pixel/options/upstream fingerprints, dependency-aware
  invalidation, corrupt-entry fallback, GUI persistence/clear action, optional CLI persistence, and
  hit/miss/write/eviction diagnostics.
- Fail-closed release metadata/x64 checks, complete dependency-licence inventory and frozen-payload
  audit with runtime notices, native GUI smoke, and draft-only tagged releases.
- Quality-first, licence-agnostic model tournament for paired geometry/lighting/restoration outputs,
  including explicit metric/case weights, category scores, model/output identities and a shared
  research-candidate registry.
- Per-page Processing controls load the committed recipe before edits, preserving satisfactory
  orientation, deskew, dewarp, lighting, cleanup and layout stages when one stage is overridden;
  orientation and exact manual-angle deskew are now first-class policies.
- Manual deskew is serialized in autosave recipes and available to batch/CLI through
  `--deskew manual --deskew-angle N`.

- Independent boundary, deskew, and local-dewarp stages with per-page JSON diagnostics.
- Offline text-line dewarp for curved or wavy pages, with confidence-based no-op fallback.
- Persisted per-page dewarp control points with side-by-side corrected preview.
- Selectable hybrid, Hough-line, and foreground-box deskew estimators.
- Direct CLI access to the OpenCV quad, Hough, and minimum-rectangle boundary backends.

### Added

- Bundled UVDoc page rectifier (`--dewarp uvdoc`, GUI **Page model (UVDoc)**): a grid-based model
  that flattens the whole page surface, including perspective, and returns the page without its
  photographed background. It runs on the CPU through ONNX Runtime in roughly 150 ms per page, and
  its weights ship with UniScan under MIT/Apache-2.0 terms.
- Automatic dewarp now builds both the text-line and UVDoc candidates and prefers the page model
  when it measures better on projective convergence or on curvature, so pure page waves still take
  the cheap text-line path while photographed pages take the page model. A user-adjusted wave model
  refines the selected backend, and `--no-auto-dewarp-page-model` restores text-line-only behaviour.

- Bundled DocShadow shadow remover and a validated lighting stage that runs after geometry
  correction (`--shadow auto|classical|docshadow`). Automatic mode only acts on a page with
  measurable shadow, prefers the model, falls back to the classical OpenCV normalization, and
  keeps a candidate only when the page ends up more evenly lit with its ink and contrast intact.
- `docs/model_evaluation.md` records which models from the survey can and cannot be bundled, with
  the licence and delivery reasons, so rejected ones are not re-evaluated later.

### Changed

- Perspective crop, accepted dewarp grid, regional user curves and deskew now compose into one
  backward map and sample authoritative pixels once; batch diagnostics expose the actual geometry
  resample count.
- Dewarp now precedes deskew. Processing recipe schema v4 migrates legacy stage order with an
  explicit reason, and batch reports preserve the executed stage order.
- Editing dewarp curves preserves and refines the committed UVDoc/DocScanner-L/automatic backend;
  editor preview and Apply now execute the same durable request.
- The matched single-pass raw-half rectify-first spike was rejected: boundaries fell from 8/8 to
  4/8, while geometry, sharpness and OCR all regressed.

- DocShadow is fetched from its immutable upstream v1.0.0 release asset during Windows builds,
  published atomically only after pinned size/SHA-256 verification, and removed from Git history.
- Model content identities now participate in persistent stage-cache keys; bundled assets are
  verified before session creation, and GUI/batch reports retain shadow-stage diagnostics.
- UVDoc automatic acceptance now measures projective line convergence as well as curvature, and
  the legacy illumination option is a single-stage alias instead of a second correction pass.
- Automatic dewarp now rejects lost text-line/edge evidence and meaningful curvature regressions;
  automatic lighting rejects large new glare or clipped regions even when unevenness improves.

- Camera frame acquisition now runs on a background stream: the preview never blocks the UI on
  device reads, every shutter press waits for a guaranteed-fresh frame instead of stale
  driver-buffer frames or blind warm-up reads, and captures reuse the already-open camera with
  the live preview kept running.
- Windows camera capture now prefers Media Foundation with hardware transforms disabled, falling
  back to DirectShow and then the platform default. This turns a 28-second device open into a
  fraction of a second and triples the streamed frame rate; the DirectShow path additionally
  requests MJPG after the frame size and never sets the frame rate afterwards, both of which
  silently forced uncompressed low-rate video.
- Preview and capture now share one resolution, so pressing the shutter stores exactly the frame
  that was on screen at that moment: no device reconfiguration, no warm-up, no shutter lag.
- On first use of a camera UniScan measures each capture mode's real frame rate, caches the
  result per device, and starts on the largest mode that still runs in real time. The capture
  menu lists every mode with its measured rate and marks the slow ones, so choosing maximum
  resolution over responsiveness stays possible and its cost is visible.
- The camera picker now lists devices by their system name instead of a bare index, reads those
  names from the Windows registry with no added dependency, and probes only as many indices as
  the system reports devices, so discovery finishes in a fraction of a second.
- Live edge detection is off by default and its worker no longer runs while hidden: the current
  detector proposes an axis-aligned box rather than a true perspective quad, so the preview no
  longer draws boundaries. Per-page boundary detection after capture is unchanged.
- The camera health label reports the streamed resolution, measured fps, and the capture size the
  device actually grants; the camera opens asynchronously with an "Opening..." state, and
  `doctor --camera` now also reports measured frame rate and the selected backend.
- The camera moved from a separate window into a main-window tab that starts its preview on
  entry and releases the device on exit; device, capture-resolution, and burst settings now live
  inline on that tab (the Camera Configuration dialog is gone), and single capture runs on the
  same cancellable background job as burst, so detection no longer freezes the UI.
- Captures keep you on the Camera tab for shot-after-shot scanning; camera enumeration and
  resolution changes moved off the UI thread.
- GUI PDF import now streams pages through disk staging, while full-resolution Apply processing
  runs in a cancellable worker and commits page generations transactionally with stale-revision
  and rollback checks.
- Runtime diagnostics now distinguish the required Office Lens quad model from the optional
  classifier used only by `auto` cleanup mode.
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
  rejected link-like targets and multiply linked lock paths before mutation, and made recovery use
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
