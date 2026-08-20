# Geometry stage order and operator-control audit — 2026-08-15

This audit answers three questions raised against the `geometry-uvdoc` branch at `13febcb`: whether
the geometry stages run in the right order, whether an operator can correct every automatic
decision, and why the real-book pilot in
[the 2026-07-30 record](geometry_real_example_2026-07-30.md) produced such a small quality gain.

It is a design audit, not a defect report against the tests: the geometry suites
(`test_dewarp`, `test_uvdoc`, `test_docscanner`, `test_pipeline`, `test_boundary_review`,
`test_processing`) pass, and the documented 21-page/21-detection result reproduces on this code.
The findings below are about what the pipeline does correctly according to its own contract while
still losing image quality.

## Reproducing the measurements

```bash
python scripts/probe_stage_order.py --pdf "example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf" --page 3 --dpi 216 --tesseract <tesseract.exe> --work-dir <scratch>
```

The input is the tracked example PDF: 1,659,192 bytes, SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`. Tesseract used `rus+eng`;
its word/character/confidence probe is the same one the 2026-07-30 record used and remains a
proxy, not a reference-based score.

Sharpness is the variance of the Laplacian. It is a proxy for retained high-frequency detail and
is comparable only between images of the same content.

**Rendering trap.** PDFium returns BGRA and `uniscan.io.loaders._render_pdf_page` asks for
`rev_byteorder=True`. A probe that renders without it swaps red and blue, which changes the
grayscale projection every geometry metric and OCR pass is built on. On this aged-paper book it
changed boundary decisions enough to turn one spread into a single page. Any future probe must
render through the production loader settings; `scripts/probe_stage_order.py` does.

## Finding 1 — deskew runs before dewarp, and pays for it

`process_document_page()` orders the stages `orientation → deskew → dewarp → lighting → cleanup →
layout` (`src/uniscan/core/processing.py`). The small-angle estimate is therefore taken on a page
that is still curved, and the rotation — expanded and filled with `BORDER_REPLICATE` — is what the
page model receives.

The same operations on the same detected half of source page 3, differing only in order:

| Variant | Curvature | Perspective | Sharpness | OCR chars | OCR confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| left half, no dewarp | 0.453 | 0.1918 | 107.7 | 236 | 18.44 |
| left half, deskew → UVDoc (shipped) | **1.391** | 0.0566 | 78.1 | 289 | 22.68 |
| left half, UVDoc → deskew | **0.345** | 0.0829 | 86.1 | 191 | 25.16 |
| right half, no dewarp | 0.325 | 0.0236 | 272.5 | 767 | 56.54 |
| right half, deskew → UVDoc (shipped) | 0.301 | 0.0324 | 208.5 | 726 | 59.72 |
| right half, UVDoc → deskew | 0.307 | 0.0216 | 219.7 | 631 | 57.95 |

On the difficult half the order alone changes measured curvature by a factor of four. The correct
order is rectify first, then remove the residual small angle: a skew angle measured before
rectification describes a page that no longer exists after it.

The left half is also the page the boundary reviewer flags as `large_dark_border_region`; its OCR
numbers are dominated by table and binding background left inside the crop, not by dewarp. No page
model repairs a bad boundary, which is exactly why the `13febcb` review flag is the right control
for it.

## Finding 2 — the chain resamples three to four times

Each geometric pass is a separate interpolation of the authoritative pixels. Measured on the same
image with the geometry held at identity:

| Passes | Sharpness | Loss |
| --- | ---: | ---: |
| source | 158.9 | — |
| 1 rotation | 133.8 | 16% |
| 2 rotations (net zero) | 126.7 | 20% |
| 3 rotations | 123.4 | 22% |
| identity remap on the exact grid | 158.9 | 0% |
| remap with a half-pixel offset | 112.2 | 29% |

Production runs the boundary homography, then a second homography for a split half when
`rectify_split_pages` is enabled, then the deskew rotation, then the dewarp remap. A manual
four-corner crop adds another, because it bakes its warp into the stored source image. Three to
four interpolations remove roughly a quarter of the high-frequency detail before cleanup begins.

The exact-grid row matters: `uvdoc._upsample_channel` already avoids OpenCV's half-pixel and
1/32-px quantization for exactly this reason, and it works. The care simply stops at the stage
boundary — the passes are individually careful and collectively lossy.

Perspective and rotation compose exactly into one homography, and a sampling grid can be evaluated
through that homography, so the whole chain can produce one backward map and one interpolation.
For a pre-OCR tool this is worth more than any of the geometry gains measured here.

## Finding 3 — the page model is asked to work outside its distribution

UVDoc is trained on photographs of a page with its background and visible edges. In this pipeline
it runs after boundary detection, perspective correction and spread splitting, so it receives a
page that has already been cropped and rectified, and it re-frames it again.

Validated automatic dewarp on the eight halves of the four difficult spreads:

| Page | Applied | Selected | Curvature before → after | Reason |
| --- | --- | --- | ---: | --- |
| p3 [L] | no | none | 1.664 → 1.664 | `uvdoc_rejected:edge_content_lost`; `textline_rejected:geometry_not_improved` |
| p3 [R] | yes | textline | 0.347 → 0.167 | `uvdoc_rejected:edge_content_lost` |
| p5 [L] | yes | textline | 0.525 → 0.294 | `uvdoc_not_flatter` |
| p5 [R] | yes | textline | 0.580 → 0.304 | `uvdoc_not_flatter` |
| p7 [L] | yes | **uvdoc** | 0.879 → 0.677 | `textline_rejected:geometry_not_improved` |
| p7 [R] | no | none | 0.379 → 0.379 | `uvdoc_rejected:edge_content_lost`; `curvature_below_threshold` |
| p11 [L] | yes | textline | 0.669 → 0.432 | `uvdoc_rejected:edge_content_lost` |
| p11 [R] | yes | textline | 0.532 → 0.294 | `uvdoc_rejected:geometry_not_improved` |

The safety gate works: the bundled page model is rejected on seven of eight halves, four times
because it cut away edge content. But that also means the bundled model earns its inference on
roughly one page in eight while every page pays for the attempt, and it matches the 3-of-21
acceptance rate in the 2026-07-30 full run. The model is being used as a perspective corrector
downstream of the perspective corrector.

Rectifying before the crop is not a drop-in reorder: running UVDoc on the whole raw frame first
made the gutter detector return one page instead of two. If the model moves upstream, spread
splitting has to happen on the raw frame first and the model must then see each half with its own
background. The `v3_uvdoc_before_crop` row in the probe output is recorded for completeness but is
not comparable per half for that reason.

## Finding 4 — operator control is missing exactly where the models decide

| Stage | Automatic | Manual correction | Status |
| --- | --- | --- | --- |
| Boundary / perspective | detector policy | four draggable corners with live preview | present; the result is baked into the stored source |
| Spread split | gutter detector | draggable gutter with preview | present |
| Orientation | conservative 0/90/270 | explicit 90/180/270 | present |
| Deskew | hybrid / Hough / min-area | exact angle | present; GUI slider is ±10° while the controller accepts ±20° |
| Dewarp | auto / UVDoc / DocScanner-L | three editable curves | **replaces the model instead of refining it** |
| Lighting | auto / DocShadow / classical | none | **method choice only** |
| Cleanup, layout | thresholds and profiles | parameters | present |
| Batch CLI | global flags | none | **per-page manual work cannot be replayed headlessly** |

Dewarp is the important one. **Apply points** sets the method to `Text lines (offline)` and
reprocesses with `DEWARP_METHOD_TEXTLINE` (`src/uniscan/ui/app.py`), so a UVDoc or DocScanner-L
result is discarded the moment an operator touches a curve. In automatic mode the mere presence of
a user model short-circuits the page-model branch entirely (`_automatic_dewarp` in
`src/uniscan/core/dewarp.py`). The editor's own preview always renders the text-line chain against
`entry.original_image`, so the combination cannot even be inspected.

The core already supports the right behaviour: explicit `--dewarp uvdoc` with a user model applies
the curves *on top of* the rectified page and records `uvdoc_with_user_adjustment`. That path is
reachable in the GUI only by saving curves and then manually switching the dropdown back to the
page model, which is documented nowhere.

For book scans the missing lighting control matters too: the binding shadow is the dominant defect
and there is no strength, no spine-band protection, and no local override — only accept or reject
what the automatic stage decided.

## Finding 5 — documentation overstates two behaviours

- "Automatic mode builds two candidates and keeps the flatter one" describes an `OR`: the page
  model is selected when it is better on perspective **or** on curvature, not when it is flatter
  overall. Two independent noisy criteria, either of which can trigger a swap.
- The acceptance gate rejects a curvature regression but has no rule against a perspective
  regression, although a perspective improvement is allowed to justify accepting a candidate. It
  did not fire on this corpus, but the asymmetry is unguarded.

Related: the gate measures curvature with `measure_dewarp_quality`, which derives text lines from a
downscaled copy — the same family of evidence the text-line candidate optimizes. Selector and
candidate share a signal, so agreement between them is not independent confirmation.

## Finding 6 — why the pilot gain was small

The 0.653 → 0.611 CER movement in the 2026-07-30 record is not evidence that page rectification
works. On this corpus:

- on a clean page the page model is rejected, and when forced it costs a quarter of the sharpness
  and 5% of recognized characters;
- on the worst page the defect is the boundary, not the surface;
- every page pays three to four interpolations regardless of which candidate wins;
- the reference is a noisy embedded OCR layer, so the metric cannot certify geometry at all.

The pilot's own conclusion — treat validated auto as a candidate selector rather than a quality
certificate, and get a paired corpus — stands. This audit adds that the quality lost to the stage
chain is currently larger than the quality gained by the models it is chaining.

## Remediation plan

Work is delivered as small commits with tests, following the definition of done in
[`audit_remediation_plan.md`](audit_remediation_plan.md). An item closes only when its acceptance
criteria pass.

### P0 — image quality and stage correctness

| ID | Finding | Required remediation | Acceptance criteria |
| --- | --- | --- | --- |
| G1 | 3–4 interpolations of authoritative pixels | Compose boundary homography, manual corner crop, deskew rotation and the dewarp grid into one backward map applied once. Keep per-stage diagnostics and cache keys unchanged in meaning. | A page with crop, deskew and dewarp all active interpolates source pixels exactly once; sharpness of the composed result is within 3% of a single-pass reference and at least 15% above the current chained result on the tracked example; `benchmark-geometry` shows no accuracy regression. |
| G2 | Deskew measured and applied before rectification | Reorder the controller to `orientation → dewarp → deskew → lighting → cleanup → layout`; bump `PROCESSING_ALGORITHM_VERSION` and the recipe schema; migrate stored recipes. | Controller and batch report use the new order; committed recipes from the previous version still replay or migrate with an explicit reason; on the tracked example the left half of page 3 no longer ends with curvature above its pre-dewarp value. |
| G3 | Editing curves replaces the selected model | Curves refine the committed method: keep the selected page model, apply the user curve on top, make the editor preview run the same chain as Apply, and stop short-circuiting the page-model branch when a user model exists. | Applying curves to a UVDoc page keeps `selected_method=uvdoc` with `uvdoc_with_user_adjustment`; the editor preview and Apply produce equivalent pixels; a regression test round-trips model-plus-curves through the recipe and autosave. |

### P1 — model placement and safety

| ID | Finding | Required remediation | Acceptance criteria |
| --- | --- | --- | --- |
| G4 | Page model runs on already-cropped pages | Spike rectify-first: split the spread on the raw frame, run the page model per half with its background, compare against the shipped order on the four difficult spreads with `scripts/probe_stage_order.py`. Adopt or reject with numbers. | A dated record with per-half curvature, sharpness and OCR probe for both orders; if adopted, spread splitting moves ahead of rectification and the example still produces 21 pages with 21 detections. |
| G5 | Gate cannot reject a perspective regression | Add a bounded perspective-regression rejection symmetric with the curvature rule; document the `OR` selection explicitly or replace it with one combined score. | A synthetic case whose perspective worsens is rejected; `document_geometry.md` and `CHANGELOG.md` describe the actual selection rule. |
| G6 | No manual lighting control | Add operator control over the lighting stage: strength, and protection for a spine band or a selected region. | Shadow correction strength and a protected region are settable in GUI and CLI, serialized in the recipe, and recorded in diagnostics; the automatic path is unchanged when they are absent. |
| G7 | Manual work cannot be replayed headlessly | Let batch conversion consume per-page recipes (corners, split, angle, curves, method overrides) from a session or a recipe file. | A GUI session with manual corrections reproduces byte-equivalent pages through the CLI; the report records which pages used a per-page recipe. |

### P2 — parity and evidence

| ID | Finding | Required remediation | Acceptance criteria |
| --- | --- | --- | --- |
| G8 | GUI deskew slider is ±10° against a ±20° controller | Align the control range with the controller bound. | The GUI reaches every angle the controller and CLI accept. |
| G9 | Selector and candidate share one noisy signal | Acceptance evidence must be independent of the optimized signal: paired flat references, stratified by boundary, curvature, glare, hard shadow, tables and sparse pages. | The next model decision cites a paired corpus score, not `measure_dewarp_quality` alone. Tracked as the next gate in the 2026-07-30 record. |

### Sequencing

G1 first: it is independent of every model decision, cannot regress geometry accuracy, and
recovers quality on every page including pages where no model runs. G2 and G3 follow, because both
change the recipe schema and are best migrated once. G4 only makes sense after G1 and G2, since a
model comparison run through a four-pass chain measures the chain as much as the model.

## Closure — 2026-08-20

| ID | Status | Evidence |
| --- | --- | --- |
| G1 | closed | Perspective, accepted dewarp/user grids and deskew compose into one backward map; the regression test reports one authoritative resample and the tracked page-7 probe is 16.6% sharper than the chained reference. See [`geometry_composition_2026-08-20.md`](geometry_composition_2026-08-20.md). |
| G2 | closed | Controller order is `orientation → dewarp → deskew → lighting → cleanup → layout`; algorithm v5, recipe schema v4 and batch-report schema v6 expose the change and migration reason. |
| G3 | closed | Automatic selection no longer short-circuits on a user model; GUI preview and Apply share one request and model-plus-curves round-trip through recipe/autosave state. |
| G4 | closed — rejected | The corrected matched-half, single-pass spike retained 8/8 boundaries in the shipped order versus 4/8 raw-first. Full measurements and the decision are in [`geometry_rectify_first_spike_2026-08-20.md`](geometry_rectify_first_spike_2026-08-20.md). |
