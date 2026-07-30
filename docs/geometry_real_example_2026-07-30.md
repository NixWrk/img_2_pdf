# Real-book geometry pilot and methodology audit — 2026-07-30

This pilot uses the tracked UniScan example
`example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf` as an unpaired real-camera gate. It is
useful for exposing spread, boundary, lighting and model failures, but it is not a substitute for a
paired flat reference or a diverse camera corpus.

## Exact input and challenge

The 1,659,192-byte PDF has SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`. It contains 12 portrait A4
canvases and an existing Russian OCR text layer. Three source pages are single pages and nine are
book spreads, so the expected production result is 21 pages.

The photographs combine white PDF letterbox bands, dark table/background, an off-centre gutter,
perspective and different left/right curvature, book-cover edges, spine shadows, aged paper,
Cyrillic text and multi-column tables. The embedded OCR layer is only a noisy text-preservation
reference; it is not a pixel reference.

## Model comparison on four difficult spreads

Source PDF pages 3, 5, 7 and 11 were rendered at 216 DPI. Boundary detection and spread splitting
first produced eight page halves. Those same halves were processed with no dewarp, explicit bundled
UVDoc, explicit DocScanner-L and the then-current validated auto policy. Tesseract
`5.5.0.20241111` used `rus+eng`; comparison case-folded and retained alphanumeric characters.

| Geometry mode | OCR CER vs embedded layer (lower) | Recognized characters | Mean curvature after | Mean dewarp time/page |
| --- | ---: | ---: | ---: | ---: |
| none | 0.653129 | 3,811 | not measured | 0 ms |
| explicit UVDoc | **0.611153** | **4,221** | 0.386 px | 391 ms |
| automatic validated | 0.616002 | 4,196 | **0.334 px** | 553 ms |
| explicit DocScanner-L | 0.669041 | 3,723 | 0.588 px | 989 ms |

The official seeded DvD GPU runner was also applied to outputs 3, 5, 7 and 8 after the same split
and perspective stage. It loaded every checkpoint tensor and ran in `2.34–2.73` seconds per page
with 1,266,132,480 bytes peak CUDA allocation. A direct Tesseract TSV probe gave UVDoc 431 words,
2,256 alphanumeric characters and 58.37 character-weighted confidence; DocScanner-L scored
402/2,042/56.02 and DvD 364/1,898/51.53.

This unpaired subset does not overturn DvD's paired DIR300 lead. It says only that the lead did not
transfer to this book/table distribution and did not justify DvD's deployment cost here.

## Boundary fallback fix

The first 300-DPI run produced 21 outputs but only 20 accepted detections. Source page 8 reproduced
the failure at 300 DPI but not 216 DPI. `cv_hybrid` returned its first quad before the outer
page-completeness gate could continue to its Hough/min-rectangle candidates. The corrected policy
continues through trusted candidates after rejecting a small destructive quad, without weakening
the existing page-area and page-band thresholds. It now returns 21/21 detector decisions.

That number is a control-flow check, not a quality score. In particular, the accepted Hough boundary
on page 8 left still contains table and binding background and needs a manual four-corner override.

## Methodology audit of the first “full-auto” report

The report previously described as full automatic was not a full pipeline run. Its report records
`orientationMethod=none`, `deskewMethod=none`, `shadowMethod=none`, `lensMode=none` and
`pageLayout=none`; only detection/split and dewarp were automatic. Therefore:

- `21/21 detected` meant that a detector returned a permitted crop, not that 21 page boundaries
  were visually correct;
- the mean-curvature reduction described only accepted dewarp candidates;
- output aspect ratios ranged roughly from 0.72 to 0.92 and were not validated against a page-shape
  prior;
- several UVDoc candidates removed 45–97% of measured edge ink;
- page 14 changed measured curvature from 0.000 to 0.617 px, while page 17 changed it from 1.149 to
  1.481 px and was accepted because perspective alone improved;
- OCR CER covered only four spreads at 216 DPI and used a noisy embedded OCR layer, so it could rank
  those candidates but could not certify geometric correctness.

The acceptance gate now rejects large loss of text-line evidence, large edge-content loss or gain,
and meaningful curvature regressions. A perspective improvement may tolerate only a small bounded
curvature trade-off. Explicit model modes remain available for operator review.

## Corrected complete automatic run

The corrected 300-DPI run enabled boundary detection, spread split, conservative orientation,
hybrid deskew, validated auto dewarp, validated auto shadow removal, document cleanup and lighting
diagnostics. A4/Letter placement remained off deliberately so it could not conceal bad source
geometry.

It produced 21 pages with 21 accepted detector decisions and no detector fallback. Orientation
correctly made no 90-degree changes; 20 pages received a non-zero small-angle deskew. Auto dewarp
accepted 13 candidates (10 text-line, 3 UVDoc) and left 8 unchanged. On accepted candidates, mean
measured curvature changed from 1.399 to 0.782 px. Auto lighting accepted DocShadow on 13 pages and
left 8 unchanged.

The run also exposed a second methodology bug. On source page 8 left, the classical lighting
fallback reduced unevenness but created 42.8% glare and 33.7% clipped pixels, so the old gate accepted
a visibly damaged page. Auto lighting now rejects a large *new* glare or clipping region in addition
to checking unevenness, ink and contrast. The repeated page-8 run rejected that candidate as
`excessive_glare` and preserved 0.57% clipping instead.

Final full-run artifacts:

- report SHA-256: `8ae1d232d45a38ad6bdaeb47e1b9d973bb71839eb5242ab74421ee421135de54`;
- PDF SHA-256: `4fd0fecf55f43bdda128eea82646e56091ec8e5c0fcb5c3b8a867ea04effe3ef`;
- output aspect-ratio range: `0.723571–0.928520`.

## Decision and next gate

Keep validated auto as a conservative candidate selector, not as a quality certificate. Keep UVDoc
and DocScanner-L as explicit operator choices; keep DvD isolated until a broader real-camera corpus
shows a repeatable benefit. A production decision now requires three separate outcomes:

1. stage execution health (model loaded, output returned, timing/memory recorded);
2. per-stage acceptance evidence (no destructive geometry, clipping or content loss);
3. visual or paired-reference acceptance of the final page.

The next corpus work should be paired DIR300 plus real UniScan captures, stratified by boundary,
curvature, glare, hard shadow, tables and sparse pages. The page-8 failure must remain as an
unpaired regression case for manual-boundary workflow and lighting safety.
