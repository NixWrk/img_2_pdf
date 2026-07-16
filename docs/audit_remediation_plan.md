# UniScan audit and remediation plan

This document is the implementation source of truth for the repository audit performed against
`main` at `8e99f71`. Work is delivered as small, reviewable commits with tests. An item is complete
only when its acceptance criteria pass; changing documentation alone does not close an item.

## Definition of done

Every closed item must have:

- a regression test that fails before the fix where practical;
- bounded memory behaviour or an explicit pre-allocation guard;
- deterministic diagnostics and an actionable failure reason;
- cancellation/rollback coverage for state-changing or long-running work;
- equivalent GUI and CLI semantics where the feature is shared;
- updated user and architecture documentation;
- a dedicated commit or a narrowly coherent commit group.

## P0: correctness and data safety

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| COR-001 | Perspective output loses the last row and column because pixel-centre distance is used as pixel count. The Office Lens adapter duplicates the defect. | Define one inclusive pixel geometry convention and use it in both warp implementations. | An identity quad over `WxH` produces exactly `WxH`; tests cover axis-aligned, rotated, fractional, and one-pixel extents. |
| COR-002 | Perspective points can request an unbounded allocation. | Validate coordinates, aspect ratio, dimensions, and output pixels before OpenCV allocation. | Adversarial finite points fail with an actionable error without allocating the requested output. |
| COR-003 | Unsigned 10/12-bit samples are scaled as full-range 16-bit, making 12-bit pages nearly black. | Preserve source bit-depth metadata and implement explicit integer normalization rules. | `0..4095` maps to `0..255`; full 16-bit remains correct; constant and signed inputs are covered. |
| COR-004 | Multi-curve dewarp allows a non-monotonic `map_y`, folding and repeating rows. | Validate anchor separation and the vertical Jacobian; reject or safely constrain folding models. | Every accepted map has a positive minimum vertical derivative; adversarial curves are rejected before remap. |
| COR-005 | Layout applies a low-confidence content box; one speck can be enlarged to the printable area. | Gate crop/scale by confidence, coverage, component evidence, and blank-page policy. | Sparse/blank/speck pages remain full page; diagnostics explain the no-op. |
| COR-006 | Office Lens min-max luminance normalization converts a constant white page to black. | Add a flat-range guard and stable percentile/target-luminance normalization. | Constant black/gray/white pages remain stable and finite. |
| COR-007 | Spread analysis can produce arrays of different lengths when the smoothing kernel exceeds the search band. | Bound the kernel or use a shape-preserving implementation. | Every accepted minimum input size returns a result/None without exception. |
| COR-008 | Spread edge confidence is relative only to its own maximum, allowing weak noise to look continuous. | Add an absolute robust edge-strength term and calibrate confidence. | Flat/noisy negatives stay below the split threshold; real synthetic gutters remain detected. |
| COR-009 | A split ratio measured after homography is replayed linearly on raw pixels; embedded content boxes use the same invalid mapping. | Persist the homography and map split polygons back, or promote rectified halves to authoritative sources. | Raw/original/current halves refer to the same physical content under strong perspective. |
| COR-010 | Deskew hybrid unconditionally selects weak min-area fallback and applies angles above 0.05 degrees regardless of confidence. | Gate fallback using confidence, angular coherence, agreement, and page category. | Sparse graphics, frames, and tables remain unchanged unless evidence is sufficient. |
| COR-011 | Deskew/dewarp border replication can smear document edges into new content. | Use a measured/white document background and report added borders. | Rotation/dewarp tests show no replicated text or black edge streaks. |
| COR-012 | Split, replace/retake, multi-page crop, rotate, orient, deskew, and wave edits can partially commit. | Introduce a reusable session mutation transaction with staged assets and rollback. | Failure injection at each commit step leaves pixels, recipes, revisions, order, and manifest unchanged. |
| COR-013 | Final autosave failure is suppressed and the application closes anyway. | Keep the session locked and window open; offer retry, recovery export, or explicit discard. | Injected manifest failure cannot silently lose the newest state. |
| COR-014 | Processing recipe restore validates only a subset of types and ranges. | Use the same strict request validator for GUI, CLI, cache, and manifest restore; reject NaN/Infinity. | Corrupt-but-schema-shaped recipes are quarantined with field-specific reasons. |

## P1: resolution, memory, and performance

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| MEM-001 | The 150 MP pixel limit ignores channels, dtype, and per-stage working-set multipliers; dewarp can exceed 2 GB. | Add a memory budget estimator and stage-specific guards while retaining source resolution. | A job is rejected before OOM with required/available estimates; normal full-resolution jobs are not downscaled. |
| MEM-002 | Orientation rotates the full-resolution image four times before shrinking it for analysis. | Build one grayscale proxy, then rotate/analyse only the proxy; apply the chosen rotation once to full resolution. | Decisions match the baseline corpus; full-resolution memory traffic drops measurably. |
| MEM-003 | Dewarp separately repeats foreground/line/quality analysis before estimation and validation; benchmark repeats it again. | Create and reuse a dewarp analysis object and diagnostics. | One analysis pass per input generation; before/after metrics remain equivalent. |
| MEM-004 | Full dewarp creates dense float32 X, Y, and displacement maps. | Use compact map construction, row/stripe remap, or bounded tiles without seams. | Peak memory is bounded for the configured full-resolution limit. |
| MEM-005 | Connected-components stages allocate full-resolution int32 labels. | Analyse proxies where decisions permit; tile or guard full-resolution masks. | Large pages do not create an unbudgeted labels allocation. |
| MEM-006 | Clipboard conversion materializes RGB and BGR copies up to 150 MP. | Budget conversion and avoid redundant colour copies. | Clipboard imports obey the same memory policy as files/PDF. |
| MEM-007 | GUI import writes raw/warped PNG, reads them, then PageStore encodes them again. | Let PageStore adopt/link staged assets and metadata. | Each imported generation is encoded at most once. |
| MEM-008 | New pages commonly encode identical raw/original/current and three identical previews. | Add content-addressing/hard-link reuse and lazy derived assets. | Identical generations consume one full-resolution blob plus required references. |
| MEM-009 | Session restore fully decodes and re-encodes every valid page. | Validate assets first and rebuild only missing/corrupt derivatives. | Clean restore performs no full-page writes. |
| MEM-010 | Lightweight contour preview decodes full raw only to read its dimensions. | Persist image dimensions or inspect the header. | Lightweight preview never decodes full raw for shape alone. |
| MEM-011 | Crop editor eagerly retains full raw images for all selected pages and renders full-resolution warps during interaction. | Use bounded lazy loading and proxy previews; render full resolution only during staged commit. | Memory is independent of selected page count within a small cache bound. |
| MEM-012 | Batch image-directory transaction clones the existing directory, then the exporter clones the staged copy again. | Establish one transaction owner and one stage. | Existing output neighbours are copied no more than once. |
| MEM-013 | Export clones an entire directory, including unrelated deep trees, to preserve neighbours. | Publish only owned files or use a managed subdirectory/generation manifest. | Cost scales with UniScan-owned outputs, not unrelated neighbour volume. |
| MEM-014 | Crop benchmark materializes every PDF page instead of streaming. | Use the production iterator. | Benchmark memory remains bounded on a large multi-page PDF. |
| MEM-015 | CV detector repeats contour extraction and fallback candidate construction. | Compute shared maps/contours once and score all candidate types. | Equivalent quality with reduced stage timing. |
| MEM-016 | Office Lens bright detector and quad scoring repeatedly allocate full-resolution HSV/masks. | Use a shared bounded proxy and cache candidate-independent data. | Full-resolution output geometry is retained while analysis cost is bounded. |
| MEM-017 | Persistent PNG cache may cost more than cheap stages and hashes can copy non-contiguous full images. | Measure stage value, cache selectively, and stream/hash without a full contiguous copy. | Cache provides a demonstrated wall-time benefit for enabled stages. |

## P1: persistence and concurrency

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| REL-001 | Stage cache publishes PNG and JSON with separate replaces and only a thread lock. | Add per-key interprocess locking or a single atomically published entry container. | Concurrent writers/readers cannot observe mixed generations. |
| REL-002 | Cache `utime` failure marks a valid hit corrupt; cleanup unlink can escape and fail processing. | Separate best-effort LRU touch/cleanup from validity and suppress optional-cache I/O failures. | Read-only/permission-denied cache behaves as a miss without failing the page. |
| REL-003 | Orphan JSON is not pruned because pruning enumerates PNG only. | Reconcile and remove both orphan types and stale temp files. | Startup/prune leaves no unaccounted cache files. |
| REL-004 | Live detector stop/start reuses one event after a timed-out join, allowing two workers and stale results. | Use per-generation stop events and tokens; publish only current-generation results. | A deliberately blocked old detector cannot overwrite a new backend result. |
| REL-005 | Live detector inbox stores a caller-owned array reference. | Copy or formally transfer ownership. | Caller mutation after submit cannot change the analysed frame. |
| REL-006 | Camera read occurs in the Tk event loop. | Add a dedicated capture worker and latest-frame queue. | A blocked camera backend does not block window interaction or cancellation. |
| REL-007 | Camera index probing opens up to ten devices synchronously and then changes selection. | Probe in background and require explicit selection. | Identify remains responsive and never switches devices implicitly. |
| REL-008 | Camera property `set` results and actual width/height/FPS are ignored. | Negotiate, read back, and expose requested versus actual capabilities. | UI never claims an unsupported resolution was applied. |
| REL-009 | Burst timing floors most non-100 ms delays. | Wait to monotonic deadlines with cancellable slices. | Measured intervals meet the configured delay within documented tolerance. |
| REL-010 | Windows camera backend is fixed to DirectShow without fallback. | Probe configured, MSMF, and default backends with diagnostics. | A backend failure reports attempts and can fall back safely. |
| REL-011 | Optional camscan inserts a path at `sys.path[0]` permanently. | Use scoped import loading and restore global import state. | Optional loading cannot shadow later unrelated imports. |
| REL-012 | UVDoc cache uses environment `setdefault`, so a later explicit cache directory may be ignored. | Make cache selection explicit and validate model identity/path. | Requested cache path is either honoured or rejected clearly. |

## P1: workflow and GUI

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| GUI-001 | `ui/app.py` is a 5,339-line orchestration monolith. | Extract document, import, camera, geometry, appearance, job, and export controllers plus focused widgets. | Main app coordinates services but contains no processing policy or transaction implementation. |
| GUI-002 | Geometry and appearance controls are mixed in one processing recipe/dialog. | Introduce separate `GeometryRecipe` and `AppearanceRecipe`, tabs/sections, diagnostics, and apply scopes. | A user can change tone without invalidating geometry and can reset either domain independently. |
| GUI-003 | Several page tools, clipboard processing, capture, and editor renders remain synchronous. | Route long operations through the staged cancellable job framework. | No supported page operation performs unbounded work on Tk thread. |
| GUI-004 | Preview/committed/export state is not always obvious. | Add explicit badges for preview-only changes, committed generation, fallback, low confidence, and export source. | The user can always tell what export will contain. |
| GUI-005 | Import progress is file-based, so one large PDF appears stuck. | Emit page/render/detect/ingest progress with indeterminate mode when total pages is unknown. | Progress changes throughout a large PDF import and cancellation latency is bounded. |
| GUI-006 | Manual corner coordinate scaling uses width ratios rather than endpoint-consistent pixel mapping. | Centralize source/display transforms using the inclusive pixel convention. | Display corners round-trip to source within tolerance at edges and arbitrary zoom. |
| GUI-007 | Apply-all corner workflow can apply fallback full-frame points to pages not meaningfully reviewed. | Require per-page valid state/confidence or an explicit confirmation policy. | Unvisited/undetected pages are not silently changed. |
| GUI-008 | Optional cache initialization can prevent GUI startup. | Make cache creation best-effort and surface a warning. | GUI starts with cache disabled when its directory is unavailable. |
| GUI-009 | Portable GUI opens a console window. | Ship distinct windowed GUI and console CLI entry points sharing runtime files. | Normal GUI launch has no console; doctor/convert retain console output. |
| GUI-010 | Source launcher detects only missing imports, not stale dependency metadata. | Check project/dependency generation or install from a locked environment. | Dependency constraint changes cannot be silently ignored. |

## P1: automatic document geometry

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| AUTO-001 | Boundary detection repeats work and its confidence is not presented consistently. | Produce one ranked candidate set with calibrated confidence/reasons and manual fallback. | Detection metrics include miss, wrong boundary, fallback, and category. |
| AUTO-002 | Spread classification relies primarily on aspect/gutter heuristics. | Add a dedicated offline spread classifier using symmetry, two content regions, gutter evidence, and page-boundary structure. | Single landscape pages are not split; real spreads are detected with calibrated no-op fallback. |
| AUTO-003 | Automatic wave estimation repeats analysis and currently models limited text-line evidence. | Reuse analysis, evaluate multiple vertical regions, reject folding/clipping, and expose editable curves. | Curved pages improve; straight/photo/sparse pages remain unchanged. |
| AUTO-004 | Split rectification may retry Hough and contour work redundantly. | Share candidate evidence across whole-page and half-page decisions. | Automation latency is measured without duplicate detection passes. |
| AUTO-005 | BYOM ONNX input/output shapes and label contracts are assumed. | Validate model signature during doctor/initialization. | Incompatible models fail before page processing with expected/actual shapes. |

## P2: quality engineering and release

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| QA-001 | Current branch fails `ruff format --check` for six tracked files. | Format and keep CI clean. | CI format check passes. |
| QA-002 | Ruff uses its minimal default selection; there is no static type checker. | Enable a staged stronger lint set and pyright/mypy for core boundaries. | New policy passes without blanket ignores. |
| QA-003 | Tests using direct OpenCV paths fail under Cyrillic Windows temp paths; long staging paths also fail. | Use Unicode-safe I/O and shorten/validate internal paths. | Full suite passes with Cyrillic and deliberately long state/temp roots. |
| QA-004 | Total coverage is about 65%; the GUI monolith is about 31%. | Test extracted controllers and failure/rollback paths; keep widget tests focused. | Critical transaction/automation paths have explicit coverage and non-GUI gate rises. |
| QA-005 | Crop corpus has only five synthetic cases; geometry has one scored deskew case. | Add licensed/private real cases plus adversarial synthetic cases by category. | Category metrics and minimum sample counts are enforced. |
| QA-006 | Quality benchmark treats `detected=True` without a contour as crop success and excludes misses from mean corner error. | Report conditional and penalized error, IoU/mapping evidence, and misses. | A detector cannot pass geometry quality without comparable evidence. |
| QA-007 | Geometry benchmark duplicates dewarp quality work and writes reports non-atomically. | Consume pipeline diagnostics and publish atomically. | Benchmark timing reflects production stages and interrupted writes preserve the old report. |
| QA-008 | Latency baselines have no warm-up/repetitions and ceilings allow large regressions. | Separate cold/warm timing, repeat, track peak memory, and add trend thresholds. | Regressions are stable enough for CI and meaningful on fixed runners. |
| QA-009 | Release dependencies are not locked; the same tag can resolve different artifacts. | Add release constraints/lock and controlled update automation. | Rebuilding with the same lock uses identical dependency versions. |
| QA-010 | Image-path export copies same-extension files without decoding/validation. | Validate trusted/untrusted source contracts and image integrity. | Corrupt/non-image input cannot be published as a page by the public API. |
| QA-011 | Public PageStore methods accept arbitrary entry IDs that can form paths outside the page directory. | Validate UUID-like IDs at every public boundary. | Traversal and separator-containing IDs are rejected. |
| QA-012 | Broad exception handlers sometimes erase error type/context. | Narrow expected exceptions and preserve causes in diagnostics. | User errors remain actionable; programming errors are not silently converted to fallback. |

## Delivery order and commit policy

1. Record this plan and owner requirements.
2. Restore green formatting/tests and add the six minimal mathematical regressions.
3. Fix correctness items COR-001 through COR-008 in small commits.
4. Introduce page/session transactions and close/autosave safety.
5. Implement resolution and memory-budget policy without silent downscaling.
6. Add opt-in large-PDF export reduction with a visible GUI checkbox and report fields.
7. Separate geometry and appearance recipes/controllers/cache/UI.
8. Improve boundary, dewarp, and spread automation with calibrated diagnostics.
9. Extract GUI controllers and remove blocking Tk work.
10. Optimize storage, cache, import, export, and detector work from measured timings.
11. Harden camera/live detection.
12. Expand real/adversarial/large-job benchmarks, lock releases, and complete documentation.

Commits must not include unrelated working-tree content such as the pre-existing untracked
`example/` directory. Each implementation commit includes its tests and updates the status below.

## Progress log

| Date | Commit | Items | Result |
|---|---|---|---|
| 2026-07-16 | pending | Plan creation | Audit and owner requirements recorded. |
