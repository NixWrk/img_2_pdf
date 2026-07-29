# Geometry model benchmark — 2026-07-28

This run applies the repository's quality-first, license-agnostic selection policy. It compares
published DvD and corrected DocScanner-L outputs with UniScan's bundled UVDoc ONNX graph on the
same corrected DocUNet cases. Raw images and multi-gigabyte candidate outputs are intentionally
kept outside Git; the hashes below bind the result to their exact contents.

The runtime/export gate proposed at the end of this record was completed on 2026-07-29; see the
[DocScanner-L and DvD runtime spike](geometry_runtime_spike_2026-07-29.md).

## Source provenance

| Asset | Cases/files | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Official DocUNet `crop.zip` | 130 | 294,741,912 | `30e385f4a8c7bb47ec93cb61c73e40d91ed390fbf86b22cf1c5e4e8898df4e81` |
| Official DocUNet `scan.zip` | 65 | 436,458,210 | `28b97e256c657a9a2d6320c54517cb378af13f5f3ca27bde0f915c6da02e2bd2` |
| Official DvD `benchmark_results.zip` | 300 DIR300 + 128 DocUNet outputs | 3,157,883,927 | `2423d3d974bb04c83c668dae1a5a0310baa846ac50c205baf1861765388e7dc0` |

The DocUNet server presented an expired TLS certificate on the run date. The initial archives were
therefore acquired with `curl --insecure` as trust-on-first-use, then pinned by the SHA-256 values
above. Both ZIP CRC checks passed. Future runs must verify these digests before extraction and do
not need to trust the broken transport certificate.

Post-extraction/source tree hashes (relative names and file bytes) were:

- DocUNet crop: `2aa1b0e1d45ad5ef822f2ae40c95dc7f8e85eb4597573774b448eee055f87c83`
- DocUNet scan: `12956c0060cd46c4ae5d6c1f9550f35d95b4e5877a2640f430a81f06113dc2fe`
- corrected DocScanner-L published outputs: `22aeb57593c4f37e9787f0184df3682540991101f8cd974e94b67c1c42d8dfb5`
- DvD DocUNet published outputs: `9638d0e20be9a4d55ca26d221252d20a5894cb292d224b22680f8d1b4ccc5713`

The DvD archive contains documents 1--64 only. It is missing `65_1` and `65_2`, so the three-way
comparison uses the explicit `docunet-corrected-common-128` profile. The canonical
`docunet-corrected` profile remains a complete 130-case run. Both profiles apply the known 180°
correction to `64_1` and `64_2` during corpus import.

## Results

The common-128 manifest SHA-256 is
`791bea59950181e12c267c4af05bbb14d53ca7f5ee708ce88cad84cf8f1fcee0`.
Higher MS-SSIM is better; lower AAD is better. `aadOpenCvDisProxy` applies the published AAD
equations to OpenCV DIS flow and must not be presented as official SIFTflow AAD. The MS-SSIM
implementation follows the published five-level protocol but uses OpenCV rather than MATLAB.

| Candidate | UniScan composite | MS-SSIM | AAD OpenCV-DIS proxy | Output-set SHA-256 |
| --- | ---: | ---: | ---: | --- |
| DvD | **0.390395** | **0.558478** | **0.292877** | `2c2d54d51411cad0f0f322e000f5b9c467de9d2f2906aefb88d30416e1b3c83c` |
| DocScanner-L | 0.380053 | 0.529286 | 0.312846 | `08a035fde6665d77e24062079ba206a0f0bdd094cb99d15b8de28c33b25a620d` |
| bundled UVDoc | 0.362330 | 0.453210 | 0.316252 | `af9bd378eadbc17c3e53a7027fde34d8b74c26e5e6634db8552dc052fd916903` |

DvD beats DocScanner-L on per-case MS-SSIM in 84 of 128 cases. The official MATLAB/SIFTflow record
shipped inside the exact DvD archive reports MS-SSIM `0.54870318` and LD `6.6188977`; its
hash-bound sidecar was accepted by the benchmark runner. The small difference from local MS-SSIM
is expected because MATLAB `ssim`/`impyramid` and the OpenCV reproduction are not numerically
identical.

On all 130 available cases, DocScanner-L remains ahead of bundled UVDoc:

| Candidate | UniScan composite | MS-SSIM | AAD OpenCV-DIS proxy |
| --- | ---: | ---: | ---: |
| DocScanner-L | **0.381019** | **0.530251** | **0.309988** |
| bundled UVDoc | 0.363337 | 0.455425 | 0.313015 |

The full-130 manifest SHA-256 is
`7a94bec58650ccfb5fdf65dadbc8264c6588f7203d2f32259e73c744f424648f`. Bundled UVDoc averaged
`136.03 ms` per CPU inference in this run; its model identity was
`uvdoc:sha256:044b406398e8accf2b8c896043f22d318221e443bb095c55d184e97f3eb003f7:242890:sha256:3a58a04944e59578200d7b0ed02ccb3ade934e9d88efa21454fbb8b53fa40f02:31588352`.

## OCR setting 1

The published 60-image setting-1 subset was evaluated directly with
`tesseract v5.5.0.20241111`, its default language, and no hidden preprocessing. Lower is better.

| Candidate | Mean edit distance | Mean CER |
| --- | ---: | ---: |
| DocScanner-L | **369.05** | **0.143900** |
| bundled UVDoc | 405.17 | 0.148940 |
| DvD | 471.22 | 0.169030 |

The OCR result prevents a defensible claim that one model wins every objective: DvD is the clear
geometry leader, while DocScanner-L preserves recognisable text best on this subset. The generic
UniScan composite is useful for regression ordering but is not an official DocUNet metric and does
not include OCR.

## Decision and next experiment

Do not replace bundled UVDoc yet. Prioritise a runnable DocScanner-L integration spike because it
substantially improves geometry and wins OCR, while keeping DvD as the geometry-quality ceiling and
a GPU/high-quality candidate. DvD's official inference stack is a GPU-first PyTorch diffusion
pipeline with four checkpoints totalling 686,038,943 bytes, including a 607,381,968-byte diffusion
checkpoint; CPU latency and exportability are not yet established.

The next gate should therefore be:

1. download all four DvD checkpoints and the DocScanner-L checkpoint with recorded size and SHA-256;
2. reproduce one image through each official runner before adapting any code;
3. measure cold start, CPU/GPU latency, peak memory and output equality against published images;
4. attempt isolated ONNX export only after the official PyTorch outputs match;
5. repeat on DIR300 (and then AnyPhotoDoc6300) before changing UniScan's automatic default;
6. add cached or OCR-only tournament execution so OCR runs do not recompute dense flow metrics.
