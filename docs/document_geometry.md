# Document geometry pipeline

UniScan treats three geometrically different problems as independent stages:

```text
source page → boundary detection / perspective → orientation → deskew → local dewarp → cleanup → export
```

This separation matters. A four-corner perspective transform can make the page rectangular but
cannot straighten curved text lines. Deskew rotates the whole page but also cannot remove local
paper waves.

## Boundary detection

The production CLI accepts these policies through `--backend`:

- `auto`: bundled Office Lens ONNX, optional UVDoc, then the OpenCV hybrid fallback;
- `office_lens_onnx`: model mask plus OpenCV quad refinement;
- `cv_hybrid`: compares contour, Hough-line, and minimum-rectangle candidates;
- `opencv_quad`, `opencv_hough`, `opencv_minrect`: explicit classical baselines;
- `paddleocr_uvdoc`: optional holistic rectifier.

Keeping the classical baselines selectable makes quality and latency comparisons reproducible.

## Orientation

`--orientation auto` performs conservative 0/90/180/270 correction without OCR. It compares
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

`--dewarp auto` is the safe default candidate for automatic processing. It measures curvature,
blank borders, edge ink, and aspect ratio before and after correction, and rejects a candidate that
does not improve the measurable geometry or introduces artifacts. It tries the offline text-line
model first. UVDoc fallback requires the explicit `--auto-dewarp-uvdoc` flag so preview or batch
processing never downloads or initializes a heavyweight model unexpectedly.

`--dewarp textline` is the dependency-free mode. It:

1. builds a local foreground mask;
2. connects characters into candidate text lines;
3. robustly fits line baselines and removes their linear component;
4. aggregates at least three agreeing curvature estimates;
5. smooths and bounds the displacement field;
6. remaps the original pixels only when curvature exceeds a safe threshold.

When the page has too few usable lines or appears already straight, the operation is a no-op and
the reason is stored in the run report. This is intentional: forcing a warp on a photograph or a
nearly blank page is worse than leaving it unchanged.

`--dewarp paddleocr_uvdoc` uses the existing optional PaddleOCR `TextImageUnwarping` runtime as a
separate post-crop stage. It can handle deformation beyond the single common vertical curve used
by the offline method, but requires the large optional runtime and model cache.

The GUI currently offers the offline method so preview remains local and predictable. Use
Workspace → Processing → Remove page waves or Page tools → Auto remove waves. If the automatic
curve needs correction, Page tools → Adjust dewarp points opens the source model and a live
corrected preview. The points are normalized, saved with the page, and replayed identically at
preview and export resolution.

## Automation and source policy

Page dewarping is automatic-first. A user may adjust the model's control points when confidence or
the preview is unsatisfactory, but does not need to construct the model from scratch. A new
automatic backend is accepted only when:

- its implementation is available in a public repository;
- the code and model weights have explicit, compatible terms;
- inference works locally without a mandatory hosted service;
- it can fail without damaging the page and fall back to another backend;
- it can be compared on the same geometry corpus and timing report.

The preferred model backend is [PaddleOCR UVDoc](https://github.com/PaddlePaddle/PaddleOCR)
because its upstream project uses Apache-2.0 and already provides document unwarping and
orientation modules. [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) is an MIT-licensed
automatic comparison candidate, but requires modernization and separate verification of the
downloaded weights. [DocTr](https://github.com/fh2019ustc/DocTr) is not a merge candidate: its
current custom license is non-commercial and share-alike despite the repository being public.

## Diagnostics and examples

The JSON run report records requested and selected methods, deskew angle, whether dewarp was
applied, supporting text lines, maximum displacement, curvature before/after, blank-border and
edge-ink ratios, aspect change, dewarp latency, and any no-op or rejection reason.

Synthetic regression tests cover curved and straight pages. Real examples should be added before
tuning thresholds: camera photos near a book spine, rippled loose paper, roller-scanner waves, and
pages with few or no text lines behave differently. Difficult cases use automatic backend
selection and confidence-based fallback first, with persisted control points as the correction
layer when the selected model is close but not exact.

OCR is explicitly out of scope. Future 90°/180° orientation may use a dedicated image classifier,
but it must not introduce text recognition or searchable-document assembly into this pipeline.

No ScanTailor GPL source was copied. The implementation uses an independent OpenCV/NumPy design;
ScanTailor served as a feature and workflow reference.
