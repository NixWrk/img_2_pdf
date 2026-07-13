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
`--orientation auto` performs conservative 0/90/180/270-degree correction without OCR. It compares
horizontal line-layout evidence and glyph baseline asymmetry. Sparse, graphical, or ambiguous
pages remain unchanged and the reason is recorded. EXIF orientation is still applied while loading,
and manual 90-degree rotation remains available in Page tools.

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

Page dewarping is disabled by default (`--dewarp none`, and **None** in the GUI). When explicitly
selected, `--dewarp auto` is the conservative automatic policy. It measures curvature, blank
borders, edge ink, and aspect ratio before and after correction, and rejects a candidate that does
not improve the measurable geometry or introduces artifacts. It tries the built-in text-line
method first. Without `--auto-dewarp-uvdoc`, automatic mode never invokes UVDoc. Explicit
`--dewarp paddleocr_uvdoc`, or automatic mode with that fallback enabled, may let PaddleOCR
initialize or download its model cache; UniScan does not bundle those weights.

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

The batch JSON report uses schema version 3. It records `inputPdfDpi`, `outputPdfDpi`, and
`maxInputPixels`; `pdfDpi` remains as a compatibility alias for `outputPdfDpi`. The report also
records the requested orientation policy globally and the applied angle, confidence, and reason for
each page.

## Document cleanup and lighting evidence

The cleanup stage exposes global `fixed`/`otsu` thresholding and local `sauvola`/`wolf`
binarization. Sauvola and Wolf use local mean and standard deviation, so they tolerate smooth page
shadows better than one document-wide threshold. `--despeckle` removes only tiny connected
components that have no nearby ink; punctuation-like dots close to a text body are counted as
protected and retained.

`--lighting-diagnostics` is non-destructive. It records smooth-shadow fraction, anomalous clipped
highlight fraction, all clipped pixels, illumination range/unevenness, and warning codes. A glare
warning means detail may be absent in the source; illumination correction must not claim to recover
such clipped detail.

`--dewarp textline` is the built-in offline mode: it requires no optional model weights or
third-party inference runtime. It:

1. builds a local foreground mask;
2. connects characters into candidate text lines;
3. robustly fits line baselines and removes their linear component;
4. aggregates at least three agreeing curvature estimates;
5. smooths and bounds one normalized vertical-displacement curve;
6. remaps the original pixels only when curvature exceeds a safe threshold.

When the page has too few usable lines or appears already straight, the operation is a no-op and
the reason is stored in the run report. This is intentional: forcing a warp on a photograph or a
nearly blank page is worse than leaving it unchanged.

`--dewarp paddleocr_uvdoc` uses the optional PaddleOCR `TextImageUnwarping` runtime as a separate
post-crop stage. It can handle deformation beyond the single common vertical curve used by the
offline method, but requires the heavyweight optional runtime and a model cache that PaddleOCR may
populate on first use. The PaddleOCR package and downloaded model artifacts must be assessed under
their own applicable terms; UniScan neither distributes those weights nor records a stable model
binary identity in a processing recipe.

The GUI offers **None**, validated automatic text-line correction, and the explicit offline
text-line method; it does not enable the UVDoc fallback. Use Workspace → Processing → Remove page
waves or Page tools → Auto remove waves. If the automatic curve needs correction, Page tools →
Adjust dewarp points opens the source model and a live corrected preview. The points are normalized
and saved with the page; `Apply processing` commits the full-resolution result, and export publishes
that committed generation without replaying different global settings.

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

The schema-3 JSON run report records requested and selected methods, orientation decision, deskew
angle, whether dewarp was applied, supporting text lines, maximum displacement, curvature
before/after, blank-border and edge-ink ratios, aspect change, dewarp latency, and any no-op or
rejection reason.

Synthetic regression tests cover curved and straight pages. Real examples should be added before
tuning thresholds: camera photos near a book spine, rippled loose paper, roller-scanner waves, and
pages with few or no text lines behave differently. Difficult cases use automatic backend
selection and confidence-based fallback first, with persisted control points as the correction
layer when the selected model is close but not exact.

The generated `benchmarks/geometry_v1` corpus locks the automatic behavior for four right-angle
orientations, small-angle skew, curved/straight lines, sparse graphics, a synthetic photo, and a
blank page. `uniscan benchmark-geometry` compares accuracy and p95 latency with its committed
baseline.

OCR is explicitly out of scope. The existing geometry-based orientation handles all four
right-angle candidates; a future dedicated image-classifier backend may improve ambiguous cases,
but it must not introduce text recognition or searchable-document assembly into this pipeline.

No ScanTailor GPL source was copied. The implementation uses an independent OpenCV/NumPy design;
ScanTailor served as a feature and workflow reference.
