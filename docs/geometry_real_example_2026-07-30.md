# Real-book geometry pilot — 2026-07-30

This pilot evaluates the tracked UniScan example
`example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf` as an unpaired real-camera gate. It is
not a substitute for a broader camera corpus, but it exposes spread, detector and model failures
that are absent from DocUNet and DIR300.

## Exact input and challenge

The 1,659,192-byte PDF has SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`. It contains 12 portrait A4
canvases and an existing Russian OCR text layer. Three source pages are single pages and nine are
book spreads, so the expected production result is 21 pages.

The photographs combine:

- a spread embedded inside a portrait PDF canvas with large white letterbox bands;
- perspective, an off-centre gutter and different left/right page curvature;
- dark table/background, book-cover edges and shadows near the spine;
- aged low-contrast paper, dense Cyrillic text and multi-column tables.

The embedded OCR layer is used only as a noisy text-preservation reference. It is not a flat-image
pixel reference.

## Model comparison on four difficult spreads

Source PDF pages 3, 5, 7 and 11 were rendered at 216 DPI. Production boundary detection and spread
splitting first produced eight page halves. The same halves were then processed with no dewarp,
explicit bundled UVDoc, explicit DocScanner-L and the validated automatic policy. Tesseract
`5.5.0.20241111` used `rus+eng`; OCR comparison case-folded and retained only alphanumeric
characters.

| Geometry mode | OCR CER vs embedded layer (lower) | Recognized characters | Mean curvature after | Mean dewarp time/page |
| --- | ---: | ---: | ---: | ---: |
| none | 0.653129 | 3,811 | not measured | 0 ms |
| explicit UVDoc | **0.611153** | **4,221** | 0.386 px | 391 ms |
| automatic validated | 0.616002 | 4,196 | **0.334 px** | 553 ms |
| explicit DocScanner-L | 0.669041 | 3,723 | 0.588 px | 989 ms |

UVDoc reduced mean measured curvature from `0.504` to `0.386` pixels. DocScanner-L increased it to
`0.588` pixels and visibly pulled dark spine/border pixels into text on several halves; on source
page 5 its OCR CER reached `0.877153`. The automatic policy accepted six of eight corrections,
mixed UVDoc and text-line candidates, and left two pages unchanged when neither improved geometry.

The official seeded DvD GPU runner was also applied to outputs 3, 5, 7 and 8 after the same split
and perspective stage. It loaded every checkpoint tensor and ran in `2.34–2.73` seconds per page
with 1,266,132,480 bytes peak CUDA allocation. On those four outputs, a direct Tesseract TSV probe
found:

| Candidate | Words | Alphanumeric characters | Character-weighted confidence |
| --- | ---: | ---: | ---: |
| UVDoc | **431** | **2,256** | **58.37** |
| DocScanner-L | 402 | 2,042 | 56.02 |
| DvD | 364 | 1,898 | 51.53 |

This is an unpaired subset, so the table does not overturn DvD's paired DIR300 win. It does show
that its cost and benchmark lead do not automatically transfer to this book/table distribution.
Visual review likewise found DvD more stable than DocScanner-L on the severe spine case, but no
better than UVDoc overall and still prone to aggressive edge cropping.

## Scale-dependent spread detector bug

Before the fix, the complete default-300-DPI run produced all 21 pages but detected only 20. The
right half of source page 8 remained skewed with the dark table because a small landscape quad was
correctly rejected as destructive. The same page passed `2/2` at 216 DPI but reproduced `1/2` at
300 DPI. At 300 DPI the explicit Hough detector found a usable whole-spread candidate.

The cause was control flow rather than a threshold: `cv_hybrid` returned its first quad before the
outer page-completeness gate evaluated it, so rejection could not continue to the hybrid's Hough
or min-rectangle candidates. The pipeline now:

1. retries the original portrait-canvas spread with Hough after rejecting a small hybrid landscape
   crop, accepting it only under the existing 30% area rule;
2. continues through trusted Hough/min-rectangle variants when a split-page hybrid contour fails
   the stricter 60% area, 80% width and page-band checks.

No safety threshold was weakened. The final full run produced `21/21` detected pages, zero
fallbacks, 18 spread outputs and 16 accepted dewarps. Mean curvature changed from `0.935` to `0.546`
pixels. Its report SHA-256 is
`6b0fb9826f0111d3a2b5fd955d0afd9a9319f6663d6f77be58c4efdcafb68188`; the output PDF SHA-256 is
`3ea9f6775e0828d8f50b53d1a6f1e1b96b704e1eef0cda2b5ace828b211fd7cf`.

## Decision

Keep validated `auto` as the safe production policy and explicit UVDoc as the operator choice for
this document class. DocScanner-L remains useful as an explicit candidate but must not become the
automatic default from DIR300 evidence alone. Keep DvD isolated until a larger real-camera corpus
shows a repeatable advantage that justifies its runtime, memory and deployment cost.
