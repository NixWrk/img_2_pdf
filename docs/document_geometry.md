# Document geometry pipeline

UniScan treats three geometrically different problems as independent stages:

```text
source page → boundary detection / perspective → deskew → local dewarp → cleanup → export
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

## Deskew

`--deskew` controls small-angle page rotation:

- `hybrid`: weighted Hough estimate when enough long text/edge lines agree, otherwise the
  foreground `minAreaRect` baseline;
- `hough`: line-only estimate, useful for text-heavy pages;
- `min_area`: previous whole-foreground implementation;
- `none`: no rotation.

The GUI exposes the same estimators in Page tools. Automatic orientation by 90° or 180° remains
separate because it cannot be inferred reliably without text/layout semantics; manual 90° rotate
actions remain available.

## Removing page waves

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
Workspace → Processing → Remove page waves, or Page tools → Remove page waves.

## Diagnostics and examples

The JSON run report records the selected methods, deskew angle, whether dewarp was applied, the
number of supporting text lines, maximum displacement, and any no-op reason.

Synthetic regression tests cover curved and straight pages. Real examples should be added before
tuning thresholds: camera photos near a book spine, rippled loose paper, roller-scanner waves, and
pages with few or no text lines behave differently. The next advanced step is a manual editable
mesh for cases where neither text-line correction nor UVDoc produces an acceptable result.

No ScanTailor GPL source was copied. The implementation uses an independent OpenCV/NumPy design;
ScanTailor served as a feature and workflow reference.
