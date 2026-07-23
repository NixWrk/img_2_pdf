# Model evaluation for geometry, lighting and bleed-through

UniScan redistributes a portable Windows build, so every bundled model has to clear the same
fail-closed licence gate as the Python dependencies (`scripts/collect_third_party_licenses.py`):
permissive SPDX only, GPL/AGPL and "non-commercial" terms rejected. The target machine has no
discrete GPU, so CUDA-only inference is equally disqualifying. This file records what was checked
against those two constraints, so a rejected model is not re-evaluated from scratch later.

Verified 2026-07-23 by reading each project's licence file directly, not its paper or summaries.

## Geometric rectification

| Model | Licence | Weights | Verdict |
| --- | --- | --- | --- |
| **UVDoc** | MIT (model), Apache-2.0 (ONNX export) | ONNX, 30 MB, bundled | **Integrated.** `--dewarp uvdoc`; exact graph/data hashes are pinned |
| **DewarpNet** | MIT | PyTorch checkpoints only | Possible, but needs PyTorch and is a 2019 model UVDoc supersedes |
| **page_dewarp** | MIT | none needed | Redundant: the built-in text-line dewarp is the same cubic sheet idea |
| **D2Dewarp** | CC BY-NC-ND 4.0 | — | **Rejected.** Non-commercial *and* no-derivatives; CUDA-only |
| **DvD** | AGPL-3.0, builds on DocGeoNet weights | — | **Rejected.** AGPL is barred for the same reason as PyMuPDF |
| **DocScanner** | Custom: "Any Commercial Use ... is strictly prohibited" | — | **Rejected.** Non-commercial |
| **DocGeoNet** | Custom: same non-commercial terms as DocScanner | — | **Rejected.** Non-commercial |
| **RDGR / DocProj** | MIT | PyTorch | Needs visible text *and* borders; heavy install for a refinement stage |
| DocTr++, ForCenNet, AADD, BookNet | — | not released or not reproducible | Out of scope, as the source report already notes |

## Lighting and shadows

| Model | Licence | Weights | Verdict |
| --- | --- | --- | --- |
| **DocShadow (FSENet)** | MIT (model), MIT (ONNX export) | SD7K ONNX, 120 MB | **Integrated.** Immutable v1.0.0 release asset, size and SHA-256 pinned |
| **DocRes** | MIT | PyTorch `.pkl` on OneDrive; no ONNX export published | **Deferred.** See below |
| **DocTr / IllTr** | non-commercial (same author as DocScanner) | — | **Rejected.** Non-commercial |
| **Classical OpenCV normalization** | n/a | none | Already implemented as `correct_illumination` |

### Why DocRes is deferred despite an MIT licence

DocRes is the source report's primary lighting recommendation and its licence is fine. It is
blocked on delivery, not terms:

- no ONNX export is published, and its weights are PyTorch pickles hosted on OneDrive, which is
  not scriptable as a reproducible download;
- running it would add PyTorch to a portable build that is currently around 100 MB. Torch alone is
  an order of magnitude larger, which defeats the point of the portable ZIP;
- it needs its DTSPrompt prior tensors alongside the image, so it is not a drop-in ONNX session.

Reconsider if an ONNX export with recorded provenance appears, or if the project ever accepts a
PyTorch runtime for a separate, non-portable install profile.

## Delivery and runtime decision

`src/uniscan/models/manifest.json` is the authority for filenames, sizes, SHA-256 values, sources,
and licenses. UVDoc stays in the wheel because its two files total about 30 MB. DocShadow exceeds
GitHub's normal Git-object limit, so the Windows build downloads the immutable upstream v1.0.0
release asset, verifies it into a temporary file, and atomically publishes it only after the exact
size and SHA-256 match. The portable audit then hashes all three frozen assets again and requires
their copied license notices and inventory entries.

Both models use ONNX Runtime's CPU provider. UVDoc automatic acceptance is evidence-gated: reduced
text curvature or projective line convergence must be measurable, otherwise the source is kept.
DocShadow automatic mode similarly requires improved lighting evidence without unacceptable ink or
contrast loss and falls back to the classical method. Explicit model modes remain available for
operator-reviewed pages. Actual model content identities, including environment overrides, are
part of persistent processing-cache keys.

## Bleed-through

The source report's own conclusion holds: bleed-through cannot be removed reliably by a shadow
model, because the show-through text has the same structure as the wanted text. The dependable
route is registering the front and back scans of the same sheet, which is a capture-workflow
feature rather than a model, and no model here is a substitute for it. DSR-GAN, DSRDiff and DE-GAN
were not evaluated further: the report already flags their reproducibility as unconfirmed, and
the OCR-facing binarization they feed is explicitly out of scope for UniScan.
