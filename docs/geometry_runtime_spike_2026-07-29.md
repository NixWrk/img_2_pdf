# DocScanner-L and DvD runtime spike — 2026-07-29

This experiment closes the runtime and exportability gate opened by the 2026-07-28 geometry
tournament. Candidate licences are recorded but were not used as a quality score or exclusion
criterion.

## Exact inputs

The official source revisions were DocScanner
`54f6063a61a52e4ce4012832e943d1871a9c3c66` and DvD
`db87dd2f9f7a7ab1dc3f3c2d86df0fd60460160b`. Google Drive did not publish checkpoint digests, so
the first controlled downloads are pinned here as trust-on-first-use identities:

| Model | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| DocScanner-L | `DocScanner-L.pth` | 29,328,510 | `1d907965aa5d8e99ea8d0891fb66d13bc4f23838547bac6f568d01d480ff8c8a` |
| DocScanner-L | `seg.pth` | 4,715,923 | `cb79fdec55a5ed435dc74d8112aa9285d8213bae475022f711c709744fb19dd4` |
| DvD | `line_model2.pth` | 69,145,090 | `3890f1758e0b11ba33e0765a9dc708dbee825e47bc495c6708d0614a50e50e3e` |
| DvD | `model1852000.pt` | 607,381,968 | `0d7d50da058213f1dee2a498ea66bb4dcbfb0122c462ade1a307328632e0133d` |
| DvD | `seg.pth` | 4,715,923 | `cb79fdec55a5ed435dc74d8112aa9285d8213bae475022f711c709744fb19dd4` |
| DvD | `seg_model.pth` | 4,795,962 | `b1716f4dec7ff865fa20837ab1cf7bdb7f7a24ff7b9820fbc9480036794b6da7` |

The isolated Windows runtime used PyTorch `2.1.1+cu118`, torchvision `0.16.1+cu118` and a GTX
1070 (compute capability 6.1). The downloaded official wheel hashes were
`d99be44487d3ed0f7e6ef5d6689a37fb4a2f2821a9e7b59e7e04002a876a667a` for PyTorch and
`bfd85e941d286c09c49df972db156dbab565371be4add105f290dbc6e8029b20` for torchvision.
DocScanner's custom licence permits attributed share-alike non-commercial use and requires separate
permission for commercial use. DvD is AGPL-3.0.

## Official PyTorch output and runtime

All 798 DocScanner segmentation tensors and all 62 rectifier tensors loaded; none were silently
ignored. DvD's U²-Net, line segmentation, second segmentation and diffusion states also loaded with
no missing or unexpected tensors.

| Measurement on DocUNet `1_1` | DocScanner-L | DvD |
| --- | ---: | ---: |
| Model inference, GTX 1070 | 98.55 ms mean over 10 | 2,614 ms full sample / 1,720 ms in 3 DDIM steps |
| Model load | 456 ms | 3,702 ms |
| Peak CUDA allocated | 233,535,488 B | 1,266,132,480 B |
| Peak CUDA reserved | 285,212,672 B | 1,518,338,048 B |
| Process RSS after inference | about 811 MiB | 1,600,671,744 B |
| CPU model inference | 711.07 ms mean | not a practical target for this GPU-first spike |

DocScanner reproduced its upstream sample with MAE `0.054/255`, maximum error 1 and 94.58% exact
pixel channels. On `1_1`, it reproduced the published candidate with MAE `0.02134/255`, maximum 1
and 97.87% exact channels.

DvD required two minimal source repairs before its official runner could complete on Windows:

1. restore the commented-out `for block in self.blocks` loop in the iterative DiT path (otherwise
   the published revision raises `UnboundLocalError`);
2. use `os.path.basename` instead of splitting a Windows path on `/` when saving the output.

The unused evaluation-time VGG download and large registry-only imports were isolated with
interface-compatible shims; checkpoint-bearing inference math was not replaced. With seed 1992,
two independent processes produced the same output SHA-256
`2f067d3bcb362de011a0e26de2df5a7f0b13c38f44f5845d7fd550250e956323`. Compared with the published
DvD `1_1` PNG, MAE was `0.7295/255`, maximum 34 and 72.84% of channels were exact. Seed 0 produced
a different file. The published archive does not record its random seed, so bit-for-bit equality to
that PNG cannot be demanded even though a pinned seed is locally reproducible.

## ONNX result

DocScanner-L exports cleanly as one opset-17 graph containing the U²-Net mask and all 12 recurrent
rectifier iterations. The 1,556-node, 34,100,351-byte graph has SHA-256
`9fdebcb4067afb09d66b6637f3fd1036ba7952bbf0656778d44c2bd1c2c067f4`. Against the official
PyTorch grid, ONNX Runtime's mean absolute error was `3.78e-8`, maximum error `2.98e-7`, and the
result passed `rtol=atol=1e-4`. Export took 8.79 seconds and one ORT CPU grid inference took 439.6
ms. [The reproducer](../scripts/export_docscanner_onnx.py) verifies the source commit, both checkpoint
hashes, the ONNX graph and numerical equality.

Exact full DvD export is currently blocked, not merely unattempted. Its selected inference path uses
tensor-dependent Python branches (`t[0] > 600`, `600 > t[0] > 300`), converts tensor values into a
Python index list, and mutates features at those indices. The three DDIM steps deliberately enter
different branches and create stochastic hypotheses. Ordinary tracing freezes the branch observed
at export time, so it cannot be an exact dynamic replacement. A defensible DvD ONNX graph requires
first refactoring that control flow into tensor/scriptable operations, then validating all three
timestep regimes and the seeded sampler numerically.

## DIR300 full-corpus gate

The controlled copy of the [official DocGeoNet DIR300 corpus](https://github.com/fh2019ustc/DocGeoNet)
contained 300 distorted and 300 reference PNGs: 600 files and 4,203,131,406 bytes. The distorted
tree SHA-256 was `7a9106de4f7f6d245a7f13a7aee750c0f4a5a07a0d0960a696ebbcf51651d125`;
the reference tree SHA-256 was
`2535c3cc885bc0262d7c13732c936e34a1aa8e5679be039d81f001967c56798d`. The fail-closed UniScan
import produced 300 cases and manifest SHA-256
`bbef9e5a5c31351a55b822111bacecb4f265e1bc0579eef5b4ac0ddc3222eb34`. Both expected source-tree
hashes were verified.

Every candidate supplied all 300 outputs. OCR used Tesseract `5.5.0.20241111` on the fixed
`dir300-ocr-90` subset. The quality score is UniScan's generic SSIM/edge-F1/PSNR composite;
`AAD proxy` uses OpenCV DIS flow and is explicitly not the published MATLAB SIFTflow AAD.

| Candidate | Quality | Local MS-SSIM | AAD proxy (lower) | OCR edit distance (lower) | OCR CER (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DvD published outputs | **0.474952** | **0.681496** | **0.262190** | **544.33** | **0.188893** |
| DocScanner-L ONNX | 0.451643 | 0.623311 | 0.291831 | 657.09 | 0.213355 |
| bundled UVDoc ONNX | 0.418912 | 0.522091 | 0.318603 | 626.58 | 0.257201 |

The hash-bound official DvD sidecar was accepted and records MATLAB MS-SSIM `0.67985454`, LD
`5.0164855` and AAD `0.17313691`. Its source archive SHA-256 is
`2423d3d974bb04c83c668dae1a5a0310baa846ac50c205baf1861765388e7dc0`, the upstream metric-record
SHA-256 is `11e48136ae335d404a73bd4350fce3bdd7283bc1e2f4ce8f1ca21eacdd1cb1dd`, and the validated sidecar
SHA-256 is `bc83a3b022d5270f65aefe089949c514530293023b3903ba9fd9ffc2fc6f0c36`.

The candidate output-set identities were:

- DvD: `dc3dab72a29b777d12927df7e6032f9f3d45cacb63ad22810d4c7eb60e2b6a45`;
- DocScanner-L: `f68d03a8d7e143f7ca44ec557ff76ecd8459d88d48a7f8cd3c51bc0febdcb21f`;
- UVDoc: `87d242c1abe2ba90ac21244bf8d8490163f0b2964ff24bd260bd539f9540f46a`.

The DocScanner-L/UVDoc report is 435,111 bytes with SHA-256
`3014b77393c9c3d2a1ff2b3f992d059e78ef1a48c6bfb6666a73626eb2c6021a`. Its first pass correctly
rejected a locally created DvD sidecar whose derived output-set hash used the wrong aggregation.
After correcting only that derived hash, the 216,943-byte DvD report passed the same integrity gate
and has SHA-256 `32a758c3cdaee16371ad8d5f582818f51e7eb283f174249667786ef74b95ae2a`.
Concurrent output generation timings were deliberately excluded from candidate metadata; the
isolated runtime measurements above remain the speed and memory comparison.

## UniScan decision

DocScanner-L is now a working explicit UniScan backend. Its external graph is size- and SHA-pinned,
has an optional non-blocking diagnostic, participates in processing-cache identity, and is exposed
as `--dewarp docscanner_l` and **Page model (DocScanner-L)**. A real 1521×1137 input passed through
the production adapter with MAE `0.01871/255`, maximum 1 and 98.13% exact channels against the
official PyTorch PNG. Warm adapter core time was about 919 ms on CPU; the sampler processes 256-row
chunks to avoid materialising four page-sized float neighbourhoods. A cold end-to-end CLI run,
including before/after geometry diagnostics, took 2.136 seconds and reduced its measured line
curvature from 2.801 to 0.351 pixels.

DIR300 establishes DvD as the quality leader on that corpus and DocScanner-L as the strongest exact
ONNX production candidate. Keep DocScanner-L explicit and bundled UVDoc as the automatic default
until real UniScan camera captures are scored. DvD remains an isolated GPU reference: it is
materially slower, uses about five times DocScanner's peak allocated GPU memory, needs upstream
repairs, and has no exact ONNX path yet. No real UniScan capture corpus is present in this
repository beyond the first [real-book pilot](geometry_real_example_2026-07-30.md). That example
favours validated auto/UVDoc over DocScanner-L and DvD, but a broader camera corpus is still needed
before changing the automatic default.

## Asset delivery

The exact graph and a conventional `.sha256` sidecar are staged outside Git. The manifest has no
invented URL: attach those files to a deliberate UniScan release, then install with
`scripts/download_model_assets.py --asset docscanner_l_grid --url <RELEASE_ASSET_URL>`. The
downloader streams to a `.part` file and refuses to publish it unless both the manifest byte length
and SHA-256 match. Until that release exists, local evaluation uses `UNISCAN_DOCSCANNER_MODEL` and
the same strict verification path.
