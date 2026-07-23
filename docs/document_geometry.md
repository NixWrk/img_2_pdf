# Document geometry pipeline

UniScan treats four geometrically different problems as independent stages:

```text
source page → boundary detection / perspective → orientation → deskew → local dewarp → cleanup → export
```

This separation matters. A four-corner perspective transform can make the page rectangular but
cannot straighten curved text lines. Deskew rotates the whole page but also cannot remove local
paper waves.

After boundary detection, GUI preview/apply and headless conversion call the same
`PageProcessingRequest → process_document_page() → PageProcessingResult` controller. It owns stage
order, cancellation checkpoints, typed diagnostics, and per-stage latency. Acquisition and page
boundary detection remain outside it because they can produce zero, one, or two pages.

## Boundary detection

The production CLI accepts these policies through `--backend`:

- `auto`: offline OpenCV hybrid detector;
- `office_lens_onnx`: optional BYOM model mask plus OpenCV quad refinement;
- `cv_hybrid`: compares contour, Hough-line, and minimum-rectangle candidates;
- `opencv_quad`, `opencv_hough`, `opencv_minrect`: explicit classical baselines;

PaddleOCR UVDoc is deliberately absent from boundary policies: it is a holistic rectifier and
belongs to the independent dewarp stage. Keeping the classical baselines selectable makes quality
and latency comparisons reproducible.

## Orientation

Orientation correction is disabled by default (`--orientation none`). When explicitly selected,
`--orientation auto` performs conservative 0/90/270-degree correction without OCR. It compares
horizontal line-layout evidence and glyph baseline asymmetry. Sparse, graphical, or ambiguous
pages remain unchanged and the reason is recorded. EXIF orientation is still applied while loading,
and manual 90-degree rotation remains available in Page tools.

With `--two-page`, a forced `--orientation 90|180|270` is applied before spread classification so
a sideways portrait page is evaluated in its actual portrait geometry. The oriented source frame
must be landscape before a detector crop can be considered a spread. A spread is split only when
the central-gutter detector is confident; no midpoint fallback is used for uncertain frames. A
small boundary result that changes a portrait source into a landscape strip is rejected so an
internal table or diagram cannot become a destructive page crop. A large landscape crop is still
eligible for gutter detection because imported portrait PDF pages can contain photographed spreads.
If boundary detection leaves the PDF canvas intact, a horizontal content box covering at least 30%
of it is also eligible. Automatic 180-degree correction is intentionally disabled because page
direction cannot be established reliably without OCR; use explicit `--orientation 180` when known.
After splitting, each half is perspective-corrected independently. The candidate quadrilateral
must cover at least 60% of the half, span at least 80% of its width, reach safe top/bottom bands,
and have plausible page proportions; otherwise the uncut half is retained. This second pass
corrects planar camera perspective, not cylindrical paper curl.

## Deskew

`--deskew` controls small-angle page rotation:

- `hybrid`: weighted Hough estimate when enough long text/edge lines agree, otherwise the
  foreground `minAreaRect` baseline;
- `hough`: line-only estimate, useful for text-heavy pages;
- `min_area`: previous whole-foreground implementation;
- `none`: no rotation.

The GUI exposes the same estimators in Page tools. Right-angle orientation remains an independent
action because it uses layout evidence rather than a small-angle line estimate.

## Removing page waves

Page dewarping is disabled by default in the CLI (`--dewarp none`). The GUI starts with
**Automatic (validated)** so photographed pages immediately show a conservative corrected preview;
the preview must still be applied before export. `--dewarp auto` measures curvature, blank
borders, edge ink, aspect ratio, text-line curvature, and projective convergence before and after
correction, and rejects a candidate that does not improve measurable geometry or introduces
artifacts.

Automatic mode builds two candidates and keeps the flatter one. The built-in text-line method
models vertical displacement only, so it wins on pure page waves. The bundled UVDoc page model
(`--dewarp uvdoc`) predicts a sampling grid for the whole page, which also removes perspective and
returns the page without its photographed background, so it wins on real photographs. A
user-adjusted wave model outranks both and skips the extra inference. `--no-auto-dewarp-page-model`
keeps automatic mode on text lines alone.

Because UVDoc reframes the page, blank-background framing checks are relaxed; either reduced
curvature or reduced line convergence must prove that geometry improved. When neither the source
nor the result exposes at least three text lines there is nothing
to verify, so the page is left unchanged: photographs, near-blank scans and sparse pages are
protected from a forced warp, while a page that is unreadable until it is flattened is accepted
once rectification reveals its lines.

The model is bundled (MIT model, Apache-2.0 ONNX export; see `src/uniscan/models/README.md`) and
runs on the CPU through ONNX Runtime in roughly 150 ms per page. `UNISCAN_UVDOC_MODEL` points at a
different UVDoc ONNX file. Explicit `--dewarp paddleocr_uvdoc`, or automatic mode with
`--auto-dewarp-uvdoc`, instead uses the optional heavyweight PaddleOCR unwarping runtime and may
let it initialize or download its own model cache; UniScan does not bundle those weights.

## Content box and page layout

The physical page boundary and visible content are separate. `--page-layout a4|letter` detects the
content box after cleanup, crops unused source margins, and fits it onto a standard output page.
`--page-margin-mm`, `--align-x`, and `--align-y` provide document-wide reproducible margins and
alignment. The default `none` path is a zero-cost identity operation.

## DPI roles and allocation limits

Batch conversion separates input rasterization from output geometry:

- `--input-pdf-dpi` controls how input PDF pages are rasterized and therefore their captured detail
  and memory cost;
- `--output-pdf-dpi` controls A4/Letter pixel dimensions and the physical DPI written during PDF
  export;
- the legacy `--pdf-dpi` supplies either role when its role-specific option is omitted.

Both DPI values default to 300 and must be at least 72. `--max-input-pixels` defaults to
150,000,000 pixels and is enforced separately for every raster image, TIFF frame, and rendered PDF
page. PDF allocation is checked using PDFium's rounded-up render dimensions; an oversized page
fails before rendering instead of being silently downsampled. A4/Letter layout also refuses an
output canvas above 150,000,000 pixels before allocating it.

The batch JSON report uses schema version 4. It records `inputPdfDpi`, `outputPdfDpi`,
`pdfJpegQuality`, and `maxInputPixels`; `pdfDpi` remains as a compatibility alias for
`outputPdfDpi`. The report also records the requested orientation policy globally and the applied
angle, confidence, and reason for each page, plus the spread decision and confidence.

## Document cleanup and lighting evidence

The `document` cleanup profile preserves colour. Monochrome conversion is explicit through
`--mode grayscale`, while `--mode b/w` additionally binarizes. The cleanup stage exposes global
`fixed`/`otsu` thresholding and local `sauvola`/`wolf`
binarization. Sauvola and Wolf use local mean and standard deviation, so they tolerate smooth page
shadows better than one document-wide threshold. `--despeckle` removes only tiny connected
components that have no nearby ink; punctuation-like dots close to a text body are counted as
protected and retained.

`--lighting-diagnostics` is non-destructive. It records smooth-shadow fraction, anomalous clipped
highlight fraction, all clipped pixels, illumination range/unevenness, and warning codes. A glare
warning means detail may be absent in the source; illumination correction must not claim to recover
such clipped detail.

`--shadow auto|docshadow|classical` is the one lighting-correction stage and always runs after
geometry. Automatic mode uses DocShadow only when the page measurably needs correction, validates
the result, and falls back to the classical method. The legacy `--illumination-correction` option
is an alias for `--shadow classical`; it no longer enables a second cleanup correction. Per-page
diagnostics record requested/selected methods, applied/no-op state, before/after evidence, latency,
and rejection reason.

`--dewarp textline` is the built-in offline mode: it requires no optional model weights or
third-party inference runtime. It:

1. builds a local foreground mask;
2. connects characters into candidate text lines;
3. robustly fits line baselines and removes their linear component;
4. aggregates at least three agreeing curvature estimates;
5. smooths and bounds one automatic normalized vertical-displacement curve;
6. remaps the original pixels only when curvature exceeds a safe threshold.

When the page has too few usable lines or appears already straight, the operation is a no-op and
the reason is stored in the run report. This is intentional: forcing a warp on a photograph or a
nearly blank page is worse than leaving it unchanged.

`--dewarp paddleocr_uvdoc` uses the optional PaddleOCR `TextImageUnwarping` runtime as a separate
post-crop stage. It can handle deformation beyond the three manually editable regional curves used
by the offline method, but requires the heavyweight optional runtime and a model cache that PaddleOCR may
populate on first use. The PaddleOCR package and downloaded model artifacts must be assessed under
their own applicable terms; UniScan neither distributes those weights nor records a stable model
binary identity in a processing recipe.

The GUI offers **None**, **Automatic (validated)**, **Page model (UVDoc)**, and the explicit offline
text-line method. Workspace Processing exposes
**Page perspective** and **Edit page waves** directly. The inline perspective editor shows
draggable corner handles beside a live rectified result. Changed corners are saved when navigating
with Prev/Next or leaving with Done; Reset and Auto Detect remain explicit replacement actions.
The inline wave editor shows the source and corrected result with independent **Top**, **Middle**,
and **Bottom** curves. Each curve has its own vertical anchor and editable points, so uneven
curvature can be traced in three page regions; clicking a line or any of its points activates it
without a separate selector. The remap interpolates smoothly between regions. Along each line,
shape-preserving cubic Hermite interpolation derives local slopes from 3-4 neighboring points and
continues its edge segments cubically, bounded to the normal displacement limit.
Normalized anchors and point values are saved with the page and immediately reprocessed at full
resolution when **Apply points** is
pressed. Corner, split-line,
and wave-curve drags display a circular cursor-centered magnifier from the full-resolution
source. Perspective and wave previews are also calculated from that source, then resized only for
the screen. **Adjust split** opens the same inline layout: drag the gutter line, choose **Preview
split** to inspect two processed pages in Compare, and choose **Create 2 pages** only after the
result is correct. For automatic
settings, **Apply preview to pages** commits the
full-resolution result, and export publishes that committed generation without replaying different
global settings.

## Automation and source policy

Once dewarping is enabled, its workflow is automatic-first. A user may adjust the model's control
points when confidence or the preview is unsatisfactory, but does not need to construct the model
from scratch. A new
automatic backend is accepted only when:

- its implementation is available in a public repository;
- the code and model weights have explicit, compatible terms;
- inference works locally without a mandatory hosted service;
- it can fail without damaging the page and fall back to another backend;
- it can be compared on the same geometry corpus and timing report.

[PaddleOCR UVDoc](https://github.com/PaddlePaddle/PaddleOCR) is the currently integrated optional
model candidate and provides document unwarping in its upstream project. This does not make it the
default, and package licensing does not by itself establish the provenance or redistribution terms
of downloaded weights. [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) is an
MIT-licensed automatic comparison candidate, but requires modernization and separate verification
of the downloaded weights. [DocTr](https://github.com/fh2019ustc/DocTr) is not a merge candidate:
its current custom license is non-commercial and share-alike despite the repository being public.

## Diagnostics and examples

The schema-4 JSON run report records requested and selected methods, orientation decision, deskew
angle, whether dewarp was applied, supporting text lines, maximum displacement, curvature
and perspective evidence before/after, blank-border and edge-ink ratios, aspect change, dewarp
latency, lighting correction evidence, and any no-op or rejection reason.

Synthetic regression tests cover curved and straight pages. Real examples should be added before
tuning thresholds: camera photos near a book spine, rippled loose paper, roller-scanner waves, and
pages with few or no text lines behave differently. Difficult cases use automatic backend
selection and confidence-based fallback first, with persisted control points as the correction
layer when the selected model is close but not exact.

The generated `benchmarks/geometry_v1` corpus locks the automatic behavior for four right-angle
orientations, small-angle skew, curved/straight lines, sparse graphics, a synthetic photo, and a
blank page. `uniscan benchmark-geometry` compares accuracy and p95 latency with its committed
baseline.

OCR is explicitly out of scope. The existing geometry-based orientation handles automatic
0/90/270-degree candidates plus explicit 180-degree rotation; a future dedicated image-classifier
backend may improve ambiguous cases,
but it must not introduce text recognition or searchable-document assembly into this pipeline.

No ScanTailor GPL source was copied. The implementation uses an independent OpenCV/NumPy design;
ScanTailor served as a feature and workflow reference.
