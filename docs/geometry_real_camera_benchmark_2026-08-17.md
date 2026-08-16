# Balanced real-camera geometry gate — 2026-08-17

This run closes the real-camera gate left open by the earlier DocUNet and DIR300 experiments.
Licences were recorded but were not used as quality weights. The automatic-default decision also
considers the measured safety gate, OCR preservation, runtime and whether the model has a verified
production path.

## Corpus

The local evaluation corpus contains 25 real camera images, balanced at five cases per category:
`perspective-only`, `wave-only`, `mixed`, `flat-no-op`, and `book-spine`. The first four categories
use 20 paired DocUNet camera/flat-scan cases. The five book-spine cases are split page halves from
`example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf`, 1,659,192 bytes, SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`. They have an embedded OCR
reference but no invented flat pixel reference, so book-spine contributes to OCR and safety, not
the paired visual score.

The paired visual score uses tournament schema v3. Candidate images are fitted to the reference
without changing aspect ratio, padded to the reference canvas, and the SSIM/edge-F1/PSNR composite
is multiplied by `exp(-abs(log(candidate_aspect/reference_aspect)))`. Cases are averaged within
each category and the four category means are macro-averaged, so no category wins by having more
examples.

OCR uses Tesseract `5.5.0.20241111`, `eng` on DocUNet and `rus+eng` on the book. Text is normalized
with Unicode NFKC, case-folding, alphanumeric filtering and collapsed whitespace before CER. The
book result is therefore a difficult OCR-layer cross-check, not pixel ground truth. The production
perspective-regression gate is applied to every raw model output:

`after <= before + max(0.02, before * 0.15)`

## Exact model runs

- bundled UVDoc used its SHA-pinned ONNX grid and the production CPU adapter;
- DocScanner-L used source commit `54f6063a61a52e4ce4012832e943d1871a9c3c66`, the pinned
  checkpoints and the verified 34,100,351-byte opset-17 graph with SHA-256
  `9fdebcb4067afb09d66b6637f3fd1036ba7952bbf0656778d44c2bd1c2c067f4`;
- DvD used source commit `db87dd2f9f7a7ab1dc3f3c2d86df0fd60460160b`, checkpoint
  `model1852000.pt`, seed 1992, three diffusion steps and two hypotheses on an RTX 6000 Ada.

The DocScanner export loaded all 798 segmentation and 62 rectification tensors. PyTorch versus
ONNX had mean absolute grid error `3.34e-8`, maximum `3.58e-7`, and passed `rtol=atol=1e-4`.

## Balanced real-camera result

| Candidate | Paired visual, macro (higher) | Mean adapter latency | OCR CER, category macro (lower) | Perspective gate pass |
| --- | ---: | ---: | ---: | ---: |
| UVDoc | 0.312886 | **107.77 ms CPU** | **0.385274** | **60%** |
| DocScanner-L | 0.326602 | 708.39 ms CPU | 0.396196 | 56% |
| DvD | **0.329418** | 310 ms GPU sampling timer | 0.419262 | 56% |

The DvD timer is the official runner's sampling mean and excludes cold model loading; it is not a
CPU adapter latency. DvD leads the paired image composite by 0.002815 over DocScanner-L and by
0.016531 over UVDoc. UVDoc leads OCR and the regression gate.

| Category | UVDoc visual | DocScanner-L visual | DvD visual |
| --- | ---: | ---: | ---: |
| perspective-only | 0.307016 | 0.320577 | **0.328235** |
| wave-only | 0.324847 | 0.335234 | **0.342005** |
| mixed | 0.297507 | **0.313293** | 0.304301 |
| flat-no-op | 0.322175 | 0.337306 | **0.343130** |

| Category | UVDoc CER | DocScanner-L CER | DvD CER | UVDoc / DocScanner / DvD gate pass |
| --- | ---: | ---: | ---: | ---: |
| perspective-only | 0.356891 | **0.295550** | 0.354742 | 60% / 60% / 60% |
| wave-only | **0.452264** | 0.545835 | 0.558751 | 80% / 60% / 60% |
| mixed | **0.174616** | 0.182672 | 0.229033 | 60% / 60% / 60% |
| flat-no-op | 0.279087 | **0.261491** | 0.262488 | 20% / 20% / 20% |
| book-spine | **0.663512** | 0.695428 | 0.691294 | 80% / 80% / 80% |

The low raw pass rate, especially 20% on flat/no-op, is evidence against forcing any neural warp
without validation. It does not mean the production validated path emits all those regressions: a
failed candidate is rejected.

## Independent DIR300 schema-v3 cross-check

The reconstructed source contains the exact official 300 distorted and 300 reference files,
4,203,131,406 bytes total. A public mirror was accepted only after restoring the original
case-sensitive reference name `248.PNG` and reproducing both pinned official tree hashes:

- distorted: `7a9106de4f7f6d245a7f13a7aee750c0f4a5a07a0d0960a696ebbcf51651d125`;
- references: `2535c3cc885bc0262d7c13732c936e34a1aa8e5679be039d81f001967c56798d`.

The portable manifest identity is
`9f974e8a41d8afd8a43f2cde98802d476dd16a62d14d60e7cda6f78eb3b25e53`. OCR uses the fixed
`dir300-ocr-90` subset. AAD remains the documented OpenCV DIS proxy, not published MATLAB
SIFTflow AAD.

| Candidate | Schema-v3 quality | Local MS-SSIM | AAD proxy (lower) | OCR CER (lower) | Mean adapter latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| DvD published outputs | **0.404715** | **0.574241** | **0.290356** | **0.188893** | n/a |
| DocScanner-L ONNX | 0.399995 | 0.558707 | 0.349106 | 0.213355 | 2,937.98 ms CPU |
| bundled UVDoc ONNX | 0.368276 | 0.450093 | 0.447727 | 0.257201 | **286.60 ms CPU** |

All candidates preserve the input canvas, so their per-case aspect errors are identical: mean
`0.085621`, maximum `0.282422`. The penalty removes the old stretch-to-reference inflation without
artificially changing the ordering. DvD and UVDoc reproduce their historical output-set SHA-256
identities exactly:

- DvD: `dc3dab72a29b777d12927df7e6032f9f3d45cacb63ad22810d4c7eb60e2b6a45`;
- UVDoc: `87d242c1abe2ba90ac21244bf8d8490163f0b2964ff24bd260bd539f9540f46a`.

DocScanner-L's new pinned-runtime output-set identity is
`bb2e1c1b89e24b611633d1f00ebe23d7c4db8bbc8bcff0b81d4c89c606ac3d3e`. The byte hash differs
from the historical environment, but a current-project-runtime rerun of case 1 differs from the
pinned benchmark runtime in only `4.65e-7` mean pixel values, maximum 1, with 99.9999535% of
channels exact. OCR aggregates reproduce the historical run exactly.

## Decision

Keep bundled UVDoc as the automatic neural default, with the perspective-regression gate enabled.
Keep DocScanner-L explicit. Keep DvD as an isolated GPU/reference-quality candidate.

This is not a claim that UVDoc has the highest paired pixel score: it does not. The decision is
that the small real-camera visual gain from switching (0.0137 for DocScanner-L or 0.0165 for DvD)
does not compensate for worse real-camera OCR, a lower raw gate-pass rate, and materially heavier
runtime/deployment. DIR300 still establishes DvD as the benchmark quality leader, but the balanced
camera corpus does not support making it the unattended application default.

Relevant tests: `tests/test_model_tournament.py` and `tests/test_standard_geometry.py`; 18 tests
passed in the targeted run.
