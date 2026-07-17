# UniScan audit and remediation plan

This document is the implementation source of truth for the repository audit initially performed
against `main` at `8e99f71` and reconciled with the code at `d654a1b`. The intervening commits
`ab29504` and `d654a1b` changed documentation and added a candidate corpus file, not production
code, so unresolved code findings remain open. It records both the engineering findings and the
product requirements supplied by the project owner. Work is delivered as small, reviewable commits
with tests. An item is complete only when its acceptance criteria pass; changing documentation
alone does not close an item.

## Product invariants from the owner

1. Preserve the maximum available image resolution and source pixels for as long as possible.
   Analysis may use explicitly derived proxies, but authoritative page pixels must not be silently
   downscaled or destructively processed.
2. Do not apply document enhancement merely because a page was imported. Geometry and appearance
   changes are previewed and explicitly committed; export uses the selected committed generation.
3. Reduce data volume only for genuinely large PDF jobs, only when the user enables the export
   checkbox, and with the chosen policy recorded in the export report. The default is lossless,
   full-resolution export.
4. The GUI must be understandable, reproducible, and intuitive. Controls must state whether they
   affect geometry, appearance, or export; previews must identify unapplied changes; long work must
   be cancellable and must not freeze Tk.
5. Page geometry (boundary, perspective, orientation, deskew, dewarp, spread split, layout) and
   image appearance (colour, tone, illumination, denoise, binarization, despeckle) are separate
   models, controllers, recipes, UI sections, diagnostics, and cache namespaces.
6. Automate the expensive human steps: page boundary detection, page-wave/dewarp estimation, and
   book-spread classification. Automation must expose confidence/reason, preserve a no-op fallback,
   and retain precise manual correction.
7. Never report a failed operation after partially committing it. Multi-page and geometry mutations
   are staged, validated, atomically committed, and rolled back on failure.
8. Do not trade correctness for speed. Full-resolution output is produced from authoritative pixels;
   proxies are permitted only for analysis and interactive preview.

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
| COR-001 | Perspective output loses the last row and column because pixel-centre distance is used as pixel count. The Office Lens adapter duplicates the defect. | Define one inclusive pixel geometry convention and use it in both warp implementations. | Identity quads over `WxH` are byte-exact and produce exactly `WxH` in both the core and Office Lens paths; tests cover axis-aligned, rotated, fractional, and singleton-rejected extents. |
| COR-002 | Perspective points can request an unbounded allocation. | Validate coordinates, aspect ratio, dimensions, and output pixels before OpenCV allocation. | Non-finite/out-of-bounds points, source/output dimensions above 32766, pre-round aspect overflow, singleton/subpixel outputs, and outputs above 150 MP fail with actionable errors before allocation; the max-safe case is accepted. |
| COR-003 | Unsigned 10/12-bit samples are scaled as full-range 16-bit, making 12-bit pages nearly black. | Preserve source bit-depth metadata and implement explicit integer normalization rules. | `0..4095` maps to `0..255`; full 16-bit remains correct; constant and signed inputs are covered. |
| COR-004 | Multi-curve dewarp allows a non-monotonic `map_y`, folding and repeating rows. | Validate anchor separation and the vertical Jacobian; reject or safely constrain folding models. | Every accepted map has a positive minimum vertical derivative; adversarial curves are rejected before remap. |
| COR-005 | Blank pages already remain full page, but a single connected component that passes the current thresholds can be enlarged to the printable area. | Gate crop/scale by confidence, coverage, and multi-signal component evidence; preserve the existing blank-page no-op. | Single-speck/component and sparse negatives remain full page; valid content still lays out; diagnostics explain every no-op. |
| COR-006 | Office Lens min-max luminance normalization converts a constant white page to black. | Add a flat-range guard and stable percentile/target-luminance normalization. | Constant black/gray/white pages remain stable and finite; near-flat, outlier, affine, and nonlinear ramp fixtures stay bounded, smooth, and `uint8` without false full-frame edges or full-frame float buffers. |
| COR-007 | Spread analysis can produce arrays of different lengths when the smoothing kernel exceeds the search band. | Bound the kernel or use a shape-preserving implementation. | Every accepted minimum input size returns a result/None without exception, and oversized smoothing kernels preserve shape and match the centered zero-padded Gaussian reference within tolerance. |
| COR-008 | Spread edge confidence is relative only to its own maximum, allowing weak noise to look continuous. | Add an absolute robust edge-strength term and calibrate confidence. | Flat/noisy negatives stay below the split threshold; real synthetic gutters remain detected. |
| COR-009 | A split ratio measured after homography is replayed linearly on raw pixels; embedded content boxes use the same invalid mapping. | Persist the homography and map split polygons back, or promote rectified halves to authoritative sources. | Raw/original/current halves refer to the same physical content under strong perspective. |
| COR-010 | Deskew hybrid unconditionally selects weak min-area fallback and applies angles above 0.05 degrees regardless of confidence. | Gate fallback using confidence, angular coherence, agreement, and page category. | Sparse graphics, frames, and tables remain unchanged unless evidence is sufficient. |
| COR-011 | Deskew/dewarp border replication can smear document edges into new content. | Use a measured/white document background and report added borders. | Rotation/dewarp tests show no replicated text or black edge streaks. |
| COR-012 | Split, replace/retake, multi-page crop, rotate, orient, deskew, and wave edits can partially commit. | Introduce a reusable session mutation transaction with staged assets and rollback. | Failure injection at each commit step leaves pixels, recipes, revisions, order, and manifest unchanged. |
| COR-013 | Final autosave failure is suppressed and the application closes anyway. | Keep the session locked and window open; offer retry, recovery export, or explicit discard. | Injected manifest failure cannot silently lose the newest state. |
| COR-014 | Processing recipe restore validates only a subset of types and ranges. | Use the same strict request validator for GUI, CLI, cache, and manifest restore; reject NaN/Infinity. | Corrupt-but-schema-shaped recipes are quarantined with field-specific reasons. |
| COR-015 | GUI import immediately stores automatically warped pixels as `original/current`; export can therefore contain unreviewed geometry and “Original” no longer means the imported source. | Store authoritative imported pixels unchanged, keep detected geometry as an unapplied proposal, and commit it only through Apply. | Import followed by export without Apply preserves source pixels/geometry; preview and report identify the proposal; reject/cancel is a no-op. |
| COR-016 | Minimal CLI/API conversion implicitly selects the `document` profile, boundary detection, and JPEG quality 80, contradicting the no-op/lossless default. | Make the minimal invocation geometry/appearance no-op and lossless; require explicit flags/profile for destructive processing or lossy output. | A no-options CLI/API regression preserves decoded pixels and dimensions and reports no-op/lossless policy; explicit legacy-style options remain available and reported. |
| COR-017 | Batch recovery validates/discovers new inputs before reading the crash journal, and journal recovery requires callers to repeat exact report/image targets; recovery may be impossible after source removal or changed arguments. | Discover and validate an owned journal first, recover from its recorded inputs/targets, and reject only genuine target conflicts before mutation. | With inputs disconnected and CLI targets omitted/changed, a surviving journal can safely finish or roll back using recorded destinations; conflicting ownership fails without mutation. |
| COR-018 | Immediate automatic dewarp clears saved manual curves before estimation succeeds or the user accepts its result. | Make automatic dewarp preview-only and retain the committed manual model until an atomic Apply. | Failed, rejected, or cancelled automation leaves curves, pixels, revision, and manifest byte-for-byte unchanged; Apply replaces them transactionally. |
| COR-019 | The manifest separates active and quarantined pages, so restoring a quarantined page appends it instead of restoring its document position. | Persist a stable order key/original index across quarantine and restore. | Quarantining and restoring the first, middle, and last pages preserves page and export order across restart. |
| COR-020 | Raster loading drops source DPI, ICC profile, and related colour metadata before storage/export. | Carry physical/colour metadata with authoritative pixels, apply an explicit colour-management policy, and report any fallback/conversion. | A 600-DPI raster retains physical size; tagged RGB/CMYK fixtures preserve colour within stated tolerance; missing/unsupported metadata produces an explicit fallback diagnostic. |

## P1: resolution, memory, and performance

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| MEM-001 | The 150 MP pixel limit ignores channels, dtype, and per-stage working-set multipliers; dewarp can exceed 2 GB. | Add a memory budget estimator and stage-specific guards while retaining source resolution. | A job is rejected before OOM with required/available estimates; normal full-resolution jobs are not downscaled. |
| MEM-002 | Orientation materializes three full-resolution rotations before shrinking them for analysis. | Build one grayscale proxy, then rotate/analyse only the proxy; apply the chosen rotation once to full resolution. | Decisions match the baseline corpus; full-resolution memory traffic drops measurably. |
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
| MEM-017 | Cache economics are unmeasured: PNG encode/decode and hashing may cost more than cheap stages, and hashing can copy non-contiguous full images. | Benchmark each candidate stage before enabling persistence; cache selectively and stream/hash without a full contiguous copy. | Every enabled stage demonstrates a repeatable wall-time benefit and bounded peak memory on the benchmark matrix; stages without benefit are not cached. |
| MEM-018 | Low-variance and frame-validation paths create full-resolution grayscale/mask work after detection, making detector peak memory scale with the authoritative image. | Reuse bounded analysis proxies and map only final geometry back to full resolution. | Detector peak memory is bounded by the configured proxy plus documented fixed overhead at the 150 MP input limit; corpus decisions remain within tolerance. |
| MEM-019 | Grayscale preview expands a one-channel full-resolution image to three channels before resizing. | Resize the single-channel proxy first and colourize only the bounded preview. | Preview output is equivalent within tolerance and peak temporary memory is bounded by the preview size rather than three full-resolution channels. |
| MEM-020 | Burst capture retains all full-resolution frames and then all `PageResult` objects, so memory is `O(shots)` with no practical guard and a late failure loses the burst. | Set an explicit shot/byte budget and ingest/process incrementally with transactional publication and cancellation. | Excess requests are rejected before capture; peak memory stays within the budget; failure at any shot leaves either the old session or the fully committed burst. |
| MEM-021 | GUI export snapshots, fully decodes, copies to a second temporary store, and then stages again. | Pass immutable snapshot assets directly to a single export transaction and stream/decode only when required by the encoder. | Export has one transaction owner/stage, output is byte/visual equivalent, and peak copies/decodes are demonstrated by instrumentation. |
| MEM-022 | The default GUI crop-proposal path asks built-in contour detectors to construct a full-resolution perspective warp, then discards that array because source/export pixels remain unchanged until Apply. | Add an explicit proposal-only detector mode for built-in contour backends and defer the full-resolution warp to staged Apply; keep opt-in spread splitting on its existing rectification path. | Default import/capture/live proposal detection returns the source pixels plus contour/backend without calling the perspective-warp helper; raw/current remain identical until Apply; invalid proposal-only API combinations fail explicitly. |

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
| REL-013 | A cache entry can pass the generic JSON/schema check and then fail stage-specific decoding on every run without being evicted, creating a persistent poison hit. | Validate at the stage decoder boundary and atomically quarantine/remove entries that fail semantic decoding; optional-cache errors remain misses. | A semantically corrupt fixture causes one miss/recompute, then a valid hit; it never fails repeated processing or survives as the same poison entry. |
| REL-014 | PageStore still trusts lexical paths and follows filesystem indirection: public path-taking methods can receive arbitrary sources/destinations, and symlink/junction/reparse components or check/use swaps under session/pages/entry/stage/backup can redirect I/O, recovery, pruning, or destructive cleanup outside owned storage. | Replace internal path APIs with entry-ID/asset capabilities; make external snapshot destinations an explicit bounded capability; reject reparse components and use no-follow/handle-verified operations, or enforce an exclusively owned storage root with directory-identity checks before destructive actions. | Absolute/traversal/forged asset paths fail closed; Windows-junction and POSIX-symlink fixtures at root/session/pages/entry/stage/backup plus injected TOCTOU swaps cannot read, write, link/copy, replace, or delete an external sentinel; the sentinel remains byte-identical, while normal restore/add/replace/prune/close and explicitly authorized temporary snapshots still pass. |

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
| GUI-011 | PDF import DPI and two-page mode exist as internal variables but have no user-facing controls. | Expose, validate, persist, and echo both settings before import starts. | A user can change both values without code/config edits; the import summary/report records the effective values. |
| GUI-012 | Quick Export resets scope to all pages and DPI to 300 instead of respecting the current export/layout state. | Use one validated export-options state for quick and full export; never silently overwrite scope, DPI, or layout. | Quick and full export use identical visible options; selected-page scope and A4/Letter physical dimensions survive quick export. |
| GUI-013 | Closing the camera window stops preview but can retain the device while burst opens a second handle. | Give one capture service ownership of the handle, serialize preview/burst, and release deterministically on close/error. | An exclusive-device mock observes at most one handle; close releases it; preview→burst→preview and cancellation work without restart. |

## P1: automatic document geometry

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| AUTO-001 | Boundary detection repeats work and its confidence is not presented consistently. | Produce one ranked candidate set with calibrated confidence/reasons and manual fallback. | Detection metrics include miss, wrong boundary, fallback, and category. |
| AUTO-002 | The current spread heuristic already combines aspect, gutter, two content regions, and edge continuity, but its false-positive rate is not calibrated on a representative landscape-negative corpus. | Establish category metrics and a hard-negative corpus first; tune existing evidence and add a dedicated classifier only if measurements justify it. | Single landscape pages, panoramas, tables, and title pages remain unsplit; real spreads meet per-category recall with calibrated confidence/no-op thresholds. |
| AUTO-003 | Automatic wave estimation repeats analysis and currently models limited text-line evidence. | Reuse analysis, evaluate multiple vertical regions, reject folding/clipping, and expose editable curves. | Curved pages improve; straight/photo/sparse pages remain unchanged. |
| AUTO-004 | Whole-page and post-split half-page detection may repeat Hough/contour preparation, but the duplicated cost has not yet been isolated. | Instrument stage counters/timings first, then share candidate-independent evidence only where measurements prove equivalent reuse. | The benchmark reports whole/half preparation counts and timings; any optimization preserves geometry metrics and removes only demonstrated duplicate work. |
| AUTO-005 | BYOM ONNX input/output shapes and label contracts are assumed. | Validate model signature during doctor/initialization. | Incompatible models fail before page processing with expected/actual shapes. |
| AUTO-006 | Post-split half-page detection silently adds `opencv_hough` even when the configured detector policy selects only Office Lens, and the effective fallback chain is absent from the report. | Honour the selected detector chain exactly or expose an explicit fallback policy; record the effective backend/reason per half. | With fallback disabled no unselected backend runs; with fallback enabled the UI/CLI and report name every attempted backend and selection reason. |

## P2: quality engineering and release

| ID | Finding | Required remediation | Acceptance criteria |
|---|---|---|---|
| QA-001 | Current branch fails `ruff format --check` for six tracked files. | Format and keep CI clean. | CI format check passes. |
| QA-002 | Ruff uses its minimal default selection; there is no static type checker. | Enable a staged stronger lint set and pyright/mypy for core boundaries. | New policy passes without blanket ignores. |
| QA-003 | Tests using direct OpenCV paths fail under Cyrillic Windows temp paths; long staging paths also fail. | Use Unicode-safe I/O and shorten/validate internal paths. | Full suite passes with Cyrillic and deliberately long state/temp roots. |
| QA-004 | Fresh baseline at `d654a1b`: 370 passed, 4 skipped; total coverage 77%, `ui/app.py` 65%, and non-GUI coverage 83%. | Test extracted controllers and failure/rollback paths; keep widget tests focused and prevent baseline regression. | Critical transaction/automation paths have explicit branch coverage; total, app, and non-GUI gates do not fall below 77%, 65%, and 83% respectively. |
| QA-005 | Crop corpus has only five synthetic cases; geometry has one scored deskew case. | Add licensed/private real cases plus adversarial synthetic cases by category. | Category metrics and minimum sample counts are enforced. |
| QA-006 | Quality benchmark treats `detected=True` without a contour as crop success and excludes misses from mean corner error. | Report conditional and penalized error, IoU/mapping evidence, and misses. | A detector cannot pass geometry quality without comparable evidence. |
| QA-007 | Geometry benchmark duplicates dewarp quality work and writes reports non-atomically. | Consume pipeline diagnostics and publish atomically. | Benchmark timing reflects production stages and interrupted writes preserve the old report. |
| QA-008 | Latency baselines have no warm-up/repetitions and ceilings allow large regressions. | Separate cold/warm timing, repeat, track peak memory, and add trend thresholds. | Regressions are stable enough for CI and meaningful on fixed runners. |
| QA-009 | Release dependencies are not locked; the same tag can resolve different artifacts. | Add release constraints/lock and controlled update automation. | Rebuilding with the same lock uses identical dependency versions. |
| QA-010 | Image-path export copies same-extension files without decoding/validation. | Validate trusted/untrusted source contracts and image integrity. | Corrupt/non-image input cannot be published as a page by the public API. |
| QA-011 | Public PageStore methods accept arbitrary entry IDs that can form paths outside the page directory. | Validate UUID-like IDs at every public boundary. | Traversal and separator-containing IDs are rejected. |
| QA-012 | Broad exception handlers sometimes erase error type/context. | Narrow expected exceptions and preserve causes in diagnostics. | User errors remain actionable; programming errors are not silently converted to fallback. |
| QA-013 | The lossless-default regression used `PdfImage.get_px_size()`, which is absent from the declared minimum `pypdfium2` 4.30.0 API. | Verify embedded pixels through bitmap extraction available across the supported range and exercise the dependency floor. | The exact-pixel regression passes on `pypdfium2` 4.30.0 and the current development environment. |
| QA-014 | Explicit PageStore session IDs accept invalid names before session-directory creation, allowing traversal-shaped or malformed session roots. | Validate explicit session IDs before any directory is created; keep `None` as the only auto-generation path. | Only exact 32-character lowercase hex session IDs are accepted; empty, uppercase, hyphenated, separator-containing, and traversal IDs fail before `mkdir` and leave the store root untouched. |

## Delivery order and commit policy

1. Record and reconcile this plan and owner requirements; documentation commits close no code item.
2. Restore green formatting/tests and add named baseline regressions for COR-001, COR-002,
   COR-004, COR-006, COR-007, and GUI-006; add each remaining regression before its fix.
3. Fix local correctness items COR-001 through COR-008 in small commits.
4. Fix transformed geometry, border, and validation semantics: COR-009 through COR-011 and COR-014.
5. Make import/default processing non-destructive and metadata-aware: COR-015, COR-016, COR-020.
6. Introduce page/session/batch transactions and recovery: COR-012, COR-013, COR-017 through
   COR-019, including quarantine order and dewarp preview/Apply.
7. Implement resolution and memory-budget policy without silent downscaling; close measured
   MEM items before structural optimization.
8. Add opt-in large-PDF export reduction with a visible GUI checkbox and report fields.
9. Separate geometry and appearance recipes/controllers/cache/UI without changing semantics.
10. Improve boundary, dewarp, and spread automation with calibrated diagnostics and explicit
    detector policy, including AUTO-001 through AUTO-006.
11. Extract only useful GUI controllers, expose missing settings, and remove blocking Tk work.
12. Optimize storage, cache, import, export, and detector work from measured timings, including
    REL-001 through REL-003, REL-013, and REL-014.
13. Harden camera/live detection, including GUI-013 and REL-004 through REL-012.
14. Expand real/adversarial/large-job benchmarks, lock releases, and complete documentation.

Commits must not include unrelated working-tree content. The tracked `example/` PDF is a candidate
corpus asset, not an incidental example: it may remain distributable only with recorded provenance,
licence/redistribution status, expected use, and a benchmark/test manifest entry. Otherwise it must
move to a documented private corpus and be removed from distributable history. Each implementation
commit includes its tests; status is updated in the same commit or a dedicated follow-up status
commit. Corpus/documentation commits do not mark
engineering findings complete.

## Progress log

| Date | Commit | Items | Result |
|---|---|---|---|
| 2026-07-16 | `ab29504` | Plan creation | Initial audit and owner requirements recorded; no code item closed. |
| 2026-07-16 | `d654a1b` | Plan/candidate corpus update | Documentation and one tracked PDF added; provenance/licence and manifest linkage remain open; no code item closed. |
| 2026-07-16 | `676e66b` | Plan reconciliation | Reconciled the audit against the `d654a1b` code baseline, corrected stale/inaccurate claims, and added missing findings and acceptance tests; no code item closed by documentation alone. |
| 2026-07-16 | `06af2de` | COR-016, COR-017; AUTO-006 partial | Implemented opt-in processing/lossless defaults and journal-first recovery from recorded targets. COR-017 is complete. COR-016 implementation is complete and its minimum-version test portability is completed by `7612627`. Unselected split backends no longer run, but per-half attempted-backend/reason reporting remains open. |
| 2026-07-16 | `a5ec349` | MEM-018 partial; MEM-019, QA-001, QA-010; REL-013 partial | Bounded two detector checks, resized grayscale before colour expansion, validated same-format export sources, restored formatting, and repaired semantic cache hits. MEM-019, QA-001, and QA-010 are complete; MEM-018 still needs 150 MP peak-memory/corpus evidence; durable cross-process poison quarantine remains open under REL-013. |
| 2026-07-16 | `569309e` | COR-019, GUI-011, GUI-012; COR-012, COR-015, COR-018, GUI-004, GUI-013, MEM-020 partial | Preserved quarantine order and persisted/validated import options; Quick Export now preserves visible scope/DPI. Crop proposals have explicit persisted state and best-effort multi-page rollback; automatic dewarp is preview/Apply; camera/burst ownership is safer and burst is streamed with a 20-shot cap. The partial items still lack their full crash, Reject/report, blocked-read, or byte-budget acceptance evidence. |
| 2026-07-16 | `7612627` | COR-016, QA-013 | Removed the post-minimum PDFium API dependency from the exact-pixel regression; the test passes on `pypdfium2` 4.30.0 and the current environment. COR-016 and QA-013 are complete. |
| 2026-07-16 | `237f83f` | REL-002; REL-013 partial | Made LRU touch and temporary cleanup fail-soft, prevented a cleanup-locked rejected key from being reused in-process, and repaired it on a later successful write. REL-002 is complete; durable interprocess rejection remains part of REL-013/REL-001. |
| 2026-07-16 | `5318d8f` | MEM-022 | Added proposal-only detection to import, capture, manual corner detection, and live detection; built-in contour backends no longer create a discarded full-resolution warp. MEM-022 is complete. |
| 2026-07-17 | `589ccd3` | QA-011 | Validated every public entry-ID boundary before path resolution; traversal and separator forms are rejected. QA-011 is complete. |
| 2026-07-17 | `3c3cb70` | QA-014 | Validated explicit session IDs before `mkdir`; invalid input creates no directory. QA-014 is complete. |
| 2026-07-17 | `b4b7792` | COR-001, COR-002, COR-006, COR-007 | Unified inclusive warp geometry and Office Lens parity, bounded pre-allocation/proposal validation, stable fixed-point luminance normalization, and shape-preserving spread smoothing. The 122 focused acceptance tests pass; all four items are complete. |
| 2026-07-17 | `ac91527` | QA-009 partial | Kept the release license gate compatible with the supported dependency floor and current Pillow metadata by approving HPND and the `pypdfium2` 4.30.0 reviewed override. QA-009 remains open because dependencies are still not locked and controlled update automation is not implemented. |

`Complete` above means the item-specific acceptance criteria has direct automated evidence. `Partial`
means the implementation reduced the risk but the finding remains open. All findings not named as
complete remain open.

### Remaining acceptance gaps in this tranche

- REL-014 is newly identified and remains open: the QA-011/QA-014 lexical validators do not stop
  filesystem indirection, reparse points, or check/use swaps around PageStore-owned paths.
- AUTO-006 still needs attempted backend and selection/rejection reason per split half.
- MEM-018 still needs measured 150 MP peak-memory and corpus-decision evidence; MEM-020 still
  needs an explicit byte budget and peak/failure measurements.
- REL-013 still needs a durable cross-process rejection/quarantine mechanism coordinated with
  REL-001 interprocess publication.
- QA-009 still needs a release constraints/lock file and controlled dependency-update workflow;
  the license metadata follow-up only keeps the current supported dependency floor testable.
- COR-012, COR-015, COR-018, GUI-004, and GUI-013 remain partial: manifest-inclusive crash
  atomicity, explicit Reject/export reporting, byte-for-byte cancel/failure evidence, and a blocked
  camera-read cancellation test are not yet complete.
