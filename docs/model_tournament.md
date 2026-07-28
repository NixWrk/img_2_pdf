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
sum to 1. Candidate images with a different size are resized to the paired reference and this is
recorded in the report. Inputs, references and candidate paths cannot escape their declared roots.

The current quality score is a weighted combination of luminance SSIM, one-pixel-tolerant edge F1,
and PSNR mapped to `[0, 1]` at 50 dB. Case weights and per-category aggregates prevent a large easy
category from silently dominating the result. These generic metrics are not a replacement for
dataset-specific LD/AAD and OCR CER in a final publication-quality decision.

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

For a final model decision, archive the manifest, reference-set identity, candidate metadata, raw
outputs and report together. Then run the task-specific official metric scripts and a blinded visual
review on exactly those outputs.
