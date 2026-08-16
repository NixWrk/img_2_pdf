# Model tournament

`uniscan benchmark-models` compares already-generated images from any number of models against the
same paired references. Inference can happen in the standard UniScan environment, a separate
PyTorch/CUDA environment, a container or a remote workstation. No candidate licence family is used
as a filter or score input.

## Corpus manifest

Create `manifest.json` in a corpus directory:

```json
{
  "schemaVersion": 1,
  "corpusVersion": "geometry-real-v1",
  "task": "geometry",
  "metricWeights": {"ssim": 0.4, "edgeF1": 0.4, "psnr": 0.2},
  "cases": [
    {
      "id": "book-spine-001",
      "category": "book-spine",
      "input": "inputs/book-spine-001.png",
      "reference": "references/book-spine-001.png",
      "output": "book-spine-001.png",
      "weight": 1.0
    }
  ]
}
```

Supported tasks are `geometry`, `lighting` and `restoration`. Metric weights are explicit and must
sum to 1. Candidate images with a different size are fitted to the paired-reference canvas with a
uniform scale and median-border padding; their aspect ratio is never changed by the evaluator.
Original/reference sizes, absolute log aspect-ratio error, `exp(-error)` aspect score and the
alignment method are recorded in the schema-v2 report. The visual composite is multiplied by the
aspect score, so even blank pages cannot hide a shape error.

The current quality score is a weighted combination of luminance SSIM, one-pixel-tolerant edge F1,
and PSNR mapped to `[0, 1]` at 50 dB. Case weights produce a weighted mean inside each category;
the candidate quality score is the equal-weight macro-average of those category means. A large easy
category therefore cannot dominate merely by containing more cases. These generic metrics are not
a replacement for dataset-specific LD/AAD and OCR CER in a final publication-quality decision.

## Standard geometry profiles

Do not hand-build DocUNet or DIR300 manifests. The importer knows the published pair counts,
filenames, target area, OCR subsets and the corrected DocUNet convention:

```powershell
# Hash the downloaded archives before extraction and retain those hashes with the experiment.
Get-FileHash .\downloads\crop.zip -Algorithm SHA256
Get-FileHash .\downloads\scan.zip -Algorithm SHA256

# These expected values hash relative names + file bytes after extraction.
.\.venv\Scripts\python.exe scripts\import_standard_geometry.py corpus `
  --profile docunet-corrected `
  --distorted .\downloads\docunet-crop `
  --references .\downloads\docunet-scan `
  --expected-distorted-sha256 <PINNED_TREE_SHA256> `
  --expected-reference-sha256 <PINNED_TREE_SHA256> `
  --output benchmarks\docunet-corrected-v1
```

`docunet-corrected` requires all 130 images and 65 flatbed references. It rotates distorted inputs
`64_1` and `64_2` by 180 degrees and records that transform per case. It exposes both published OCR
subsets: `docunet-ocr-setting-1` (60 images) and `docunet-ocr-setting-2` (50 images). `dir300`
requires all 300 pairs and exposes `dir300-ocr-90`.

The separate `docunet-corrected-common-128` profile covers documents 1--64. This is the complete
intersection with the official DvD output archive, which does not contain `65_1` or `65_2`. Use the
130-case profile for models with full coverage and the 128-case profile only when DvD is included.

If the upstream host does not publish a digest, compute the archive SHA-256 once in a controlled
download, store it in the experiment inventory, and require that exact value on every subsequent
download. The importer additionally records the extracted tree SHA-256. A run with no expected tree
hash is marked `expectedSha256Verified: false`; it is useful for discovering a digest, not as a
trusted final import. The normalized manifest also pins every input and reference SHA-256, and the
tournament verifies those files again before scoring.

Published output folders often use incompatible suffixes. Normalize each complete output set before
comparison:

```powershell
.\.venv\Scripts\python.exe scripts\import_standard_geometry.py candidate `
  --profile docunet-corrected `
  --source .\downloads\DocScanner-L_DocUNet_rec `
  --expected-source-sha256 <PINNED_TREE_SHA256> `
  --name docscanner-l `
  --license LicenseRef-DocScanner-NonCommercial-ShareAlike `
  --output out\docscanner-l
```

The built-in templates cover the upstream DocScanner, UVDoc-style and common `_rec`/`_unwarp`
names. Use repeatable `--template`, with `{case}` and `{document}`, when an upstream archive differs.
Import fails on a missing or ambiguous case rather than benchmarking a partial result.

Generate the pinned CPU baseline on those exact corrected inputs:

```powershell
.\.venv\Scripts\python.exe scripts\run_bundled_uvdoc_candidate.py `
  --corpus benchmarks\docunet-corrected-v1 `
  --output out\uvdoc-onnx
```

Its `candidate.json` records the verified graph/data identity, manifest SHA-256 and per-case latency.

### Geometry metric identities

A standard-profile run adds the following clearly separated evidence:

- `docunetMsSsim`: the published target-area, five-level and weight protocol reproduced with
  Python/OpenCV. It is useful for local ranking, but is not labelled MATLAB-identical because
  `ssim` and `impyramid` vary by MATLAB release. The upstream weights sum to `1.0001`; this is kept
  exactly.
- `aadOpenCvDisProxy`: the published AAD equations evaluated with OpenCV DIS optical flow. It is a
  diagnostic proxy only. Official AAD and LD require the official MATLAB SIFTflow implementation.
- `officialEvaluation`: an optional, hash-bound result from the external official evaluator.
- `ocrEditDistance` and `ocrCer`: calculated only when Tesseract and a named subset are explicit.

OCR is therefore reproducible rather than ambient:

```powershell
.\.venv\Scripts\python.exe -m uniscan benchmark-models `
  --input benchmarks\docunet-corrected-v1 `
  --candidate uvdoc=out\uvdoc `
  --candidate docscanner-l=out\docscanner-l `
  --output out\docunet-report.json `
  --tesseract tesseract `
  --ocr-subset docunet-ocr-setting-1
```

The report records the resolved Tesseract version, language, subset and direct-CLI driver. Do not
compare its CER numerically with a paper that used a different engine release without labelling the
version difference.

### Official MATLAB LD/AAD hook

Run the original DocUNet evaluation package and official SIFTflow implementation on the exact
normalized output directory. Put the aggregate result in `official-metrics.json` beside
`candidate.json`:

```json
{
  "schemaVersion": 1,
  "benchmarkProfile": "docunet-corrected",
  "manifestSha256": "<from the first tournament report>",
  "outputSetSha256": "<from that candidate in the first report>",
  "implementation": {
    "matlab": "R2019a",
    "flow": "official SIFTflow",
    "source": "DocUNet evaluation package"
  },
  "metrics": {
    "msSsim": 0.5178,
    "ld": 7.45,
    "aad": 0.121,
    "editDistance": 390.43,
    "cer": 0.1486
  }
}
```

Rerun the tournament to attach it. UniScan rejects the sidecar if its benchmark profile, manifest
SHA-256 or candidate output-set SHA-256 differs. The numbers above illustrate the schema; never copy
paper-table values into a sidecar and present them as a local evaluation.

## Candidate output metadata

Each output directory may contain `candidate.json`:

```json
{
  "schemaVersion": 1,
  "license": "AGPL-3.0-only",
  "delivery": "external",
  "modelIdentity": "sha256:0123456789abcdef...",
  "outputs": {
    "book-spine-001": {
      "path": "book-spine-001.png",
      "latencyMs": 842.5
    }
  }
}
```

`license`, `delivery`, `modelIdentity` and latency are metadata only. Missing licence metadata does
not disqualify an experiment. Missing or undecodable case output does, because incomplete candidates
cannot be compared on the same distribution. If `outputs` is omitted, the manifest's `output` path
is used for each candidate.

## Run

```powershell
.\.venv\Scripts\python.exe -m uniscan benchmark-models `
  --input benchmarks\geometry-real-v1 `
  --candidate uvdoc=out\uvdoc `
  --candidate docscanner-l=out\docscanner-l `
  --candidate dvd=out\dvd `
  --output out\geometry-tournament.json
```

The JSON report records `selectionPolicy: quality-first-license-agnostic`, the manifest SHA-256,
metric weights, all output SHA-256 values, category scores, failures, full ranking and winner.
Ranking is descending quality score; licence, delivery, latency and package size do not break ties.

For a final model decision, archive the downloaded-asset hashes, extracted-tree hashes, manifest,
candidate metadata, raw outputs, local report and official sidecars together. Geometry selection uses
MS-SSIM (higher), LD/AAD/CER (lower) and a blinded visual review; the generic `qualityScore` winner is
the first gate, not by itself the final geometry winner.
