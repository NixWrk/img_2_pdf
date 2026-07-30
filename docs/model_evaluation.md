# Quality-first model evaluation

UniScan ranks document models by measured output quality, not by licence family. GPL, AGPL,
non-commercial, research-only, proprietary and unknown/custom terms do not remove a candidate from
the benchmark. Licence metadata is recorded in the report so that the chosen model can be delivered
and used in a way that satisfies its actual terms.

This policy separates three decisions that used to be conflated:

1. **Quality:** one paired corpus, one set of metrics, no licence term in the score.
2. **Runtime:** CPU/GPU memory, latency and package size are reported separately and never change the
   quality winner.
3. **Delivery:** `bundled`, verified `runtime-download`, or `external`/BYOM. Distribution still
   carries the applicable notices and obligations; an unidentified licence remains an inventory
   error, not a quality failure.

The source review below was refreshed on 2026-07-28 from official repositories and papers. Published
paper tables are useful for prioritising experiments, but they are not treated as UniScan results.
Only `uniscan benchmark-models` can select a winner for our capture distribution.
The same shortlist is stored machine-readably in `benchmarks/model_candidates.json`.

## Geometry tournament

| Candidate | Reproducible status | Tournament decision |
| --- | --- | --- |
| **UVDoc ONNX** | Integrated CPU baseline; graph/data hashes pinned | Run on every geometry corpus |
| **PaddleOCR UVDoc** | Optional upstream runtime/cache | Run as an independent candidate, not alias it to bundled UVDoc |
| **DocScanner-L** | [Official code and pretrained model](https://github.com/fh2019ustc/DocScanner); exact ONNX adapter integrated | **Explicit production backend**; DIR300 runner-up and strongest exact ONNX candidate; awaiting real-camera gate |
| **DvD** | [Official AGPL code and four pretrained files](https://github.com/hanquansanren/DvD); official GPU runner and DIR300 outputs reproduced | **DIR300 quality leader**; 2.61 s/sample and 1.27 GB peak CUDA allocated; exact ONNX blocked |
| **DocTr++** | [Official code and pretrained model](https://github.com/fh2019ustc/DocTr-Plus); designed for incomplete/in-the-wild boundaries | **Priority 2 external candidate** |
| **DocGeoNet** | [Official code, weights and DIR300 protocol](https://github.com/fh2019ustc/DocGeoNet) | Priority 2 reference baseline |
| **DocRes** | [Official MIT code and weights](https://github.com/ZZZHANG-jx/DocRes); supports dewarping plus restoration tasks | Priority 2 joint-model candidate |
| DewarpNet, RDGR/DocProj, D2Dewarp | Older but runnable/reference implementations | Add when their published outputs or weights are available locally |

The newest claimed methods are tracked but cannot enter a reproducible inference tournament yet.
[ForCenNet](https://github.com/caipeng328/ForCenNet) still lists result/evaluation release as TODO;
[AADD](https://github.com/chaoyunwang/AADD) says full code and models are coming later; and
[ArbDR](https://github.com/chaoyunwang/ArbDR) says its code is still being organised. Their paper
claims are not substituted for executable outputs.

## Lighting and restoration tournament

| Candidate | Reproducible status | Tournament decision |
| --- | --- | --- |
| **DocShadow / FSENet** | Integrated CPU baseline; immutable v1.0.0 ONNX asset, size and SHA-256 pinned | Run on every lighting corpus |
| **DocRes** | One generalist checkpoint covers deshadowing, appearance, deblurring, binarisation, dewarping and end-to-end restoration | **Priority 1 external candidate** |
| **ShaDocNet** | [Official archived MIT code and release weights](https://github.com/CXH-Research/ShadocNet); separate ~699 MB detector and ~1.12 GB remover assets | **Priority 1 external candidate**; pin locally computed SHA-256 because upstream supplies none |
| **DocTr / IllTr** | [Official code and pretrained models](https://github.com/fh2019ustc/DocTr) for geometry plus illumination | Priority 2 joint-pipeline candidate |
| **DocNLC** | [Official code and OneDrive model zoo](https://github.com/RylonW/DocNLC) for shadow/noise/blur/watermark/background degradations; no repository licence file found | Priority 2 external candidate; unknown terms do not affect score |
| **UDoc-GAN** | [Official code and pretrained outputs/models](https://github.com/harrytea/UDoc-GAN) for unpaired illumination correction | Priority 2 external candidate |
| **Classical OpenCV normalisation** | Built in, deterministic, no weights | Mandatory non-neural baseline |

DocRes is no longer deferred because of PyTorch size or delivery inconvenience. It can run in an
isolated external environment and submit output images to the same tournament. If it wins, runtime
engineering follows the quality decision: export/optimise it if faithful, ship a separate runtime,
or keep it external. We do not replace a better result with a worse one merely because the latter is
easier to bundle.

[MMDIR (CVPR 2026)](https://github.com/xiaomore/MMDIR) is the strongest new mixed-degradation
lead found in this review: its paper reports competitive/superior perceptual results across blur,
shadow, watermark and seal removal. The official repository currently publishes MixedDoc and the
authors' predicted outputs, but no inference code or weights, so it is tracked rather than called a
runnable candidate. Its CC BY-NC-ND terms are not the blocker; missing executable inference is.

## Benchmark contract

The tournament consumes precomputed output directories, so models with mutually incompatible
frameworks can be compared without contaminating the production environment. The paired manifest
sets the case weights and the explicit SSIM/edge-F1/PSNR weights. Candidate metadata records model
identity, licence, delivery and per-case latency. The report hashes the manifest and every submitted
output.

Current image metrics are a fast, framework-independent first gate. Standard DocUNet/DIR300 imports
now add a protocol-compatible OpenCV MS-SSIM, an explicitly non-official DIS-flow AAD proxy, named
OCR subsets with a recorded Tesseract version, and a hash-bound sidecar hook for official MATLAB
SIFTflow LD/AAD. A final geometry decision must use that official sidecar rather than relabelling the
proxy. Lighting selection must include paired real shadows, colour fidelity, clipped-detail checks
and OCR CER. Human review remains required for hallucinated/missing glyphs, ruling, stamps and
photographs that aggregate metrics can hide.

See [the model tournament guide](model_tournament.md) for the manifest and command. The first serious
run should contain:

- corrected DocUNet and DIR300 geometry pairs;
- released AnyPhotoDoc6300 subsets for difficult real photographs;
- paired SD7K/RealDAE lighting cases;
- consented UniScan camera captures split by flat sheet, book spine, crease, glare, hard shadow,
  coloured paper, tables and sparse content.

The first three-way corrected DocUNet run is now recorded in
[the 2026-07-28 geometry evidence](geometry_benchmark_2026-07-28.md): DvD wins geometric fidelity,
while DocScanner-L wins OCR. The [full DIR300 run](geometry_runtime_spike_2026-07-29.md) also ranks
DvD first on local composite, MS-SSIM, AAD proxy and OCR CER, and accepts its hash-bound official
MATLAB metrics. DocScanner-L is second and remains the practical exact ONNX production candidate.
The first [real-book pilot and methodology audit](geometry_real_example_2026-07-30.md) gives UVDoc
the best OCR result on its small dewarp subset, while DocScanner-L over-corrects several
spine/table pages and DvD does not justify its cost there. The full-pipeline audit also proves that
validated auto is a conservative candidate selector, not a quality certificate. More diverse
paired and UniScan camera cases are required before changing the automatic default.
Lighting remains open until classical, DocShadow, DocRes and ShaDocNet have outputs for identical
paired cases. MMDIR joins as soon as its authors publish runnable inference or outputs for those
cases.

## Current production assets

`src/uniscan/models/manifest.json` remains the authority for filenames, sources, exact sizes,
SHA-256 values and licence metadata of production assets. UVDoc stays in the wheel. DocShadow is
downloaded from the immutable upstream v1.0.0 release asset into a temporary file, accepted only
after its exact size and SHA-256 match, and then included in the portable build. The portable audit
hashes all frozen model assets again and requires their notices and inventory entries.

Both current models use ONNX Runtime's CPU provider. Automatic acceptance remains evidence-gated so
that a model can lose to the unchanged input on an individual page even when it wins overall.
Explicit model modes remain available for operator review, and actual model-content identities are
part of persistent cache keys.

## Bleed-through

Bleed-through is a separate paired-capture problem: show-through glyphs can have the same structure
as wanted text, so a shadow model is not a reliable remover. Front/back registration should be its
own tournament task, with OCR and preservation metrics; no geometry or shadow winner is implicitly
declared suitable for it.
