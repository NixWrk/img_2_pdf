# Geometry model benchmark schema-v2 rerun — 2026-08-16

This rerun evaluates commit `1eba9ce5fef39c0a25007359381494c08538525e` after the model
tournament stopped stretching candidate images to the reference aspect ratio. Schema v2 fits each
candidate without changing its aspect ratio, pads with the candidate border median, records an
explicit absolute-log-ratio aspect score, and macro-averages categories equally.

## Recovered inputs

The raw files remain outside Git. Every recovered source matched the inventory recorded in the
[2026-07-28 benchmark](geometry_benchmark_2026-07-28.md):

| Asset | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| DocUNet `crop.zip` | 294,741,912 | `30e385f4a8c7bb47ec93cb61c73e40d91ed390fbf86b22cf1c5e4e8898df4e81` |
| DocUNet `scan.zip` | 436,458,210 | `28b97e256c657a9a2d6320c54517cb378af13f5f3ca27bde0f915c6da02e2bd2` |
| DvD `benchmark_results.zip` | 3,157,883,927 | `2423d3d974bb04c83c668dae1a5a0310baa846ac50c205baf1861765388e7dc0` |
| DocScanner-L corrected output tree | 130 files | `22aeb57593c4f37e9787f0184df3682540991101f8cd974e94b67c1c42d8dfb5` |

The extracted DocUNet crop, scan and DvD DocUNet trees also matched their recorded hashes
`2aa1b0e1...f87c83`, `12956c00...dc2fe` and `9638d0e2...c5713` respectively. Candidate output-set
hashes for the common-128 run were unchanged from the old report:

- DvD: `2c2d54d51411cad0f0f322e000f5b9c467de9d2f2906aefb88d30416e1b3c83c`;
- DocScanner-L: `08a035fde6665d77e24062079ba206a0f0bdd094cb99d15b8de28c33b25a620d`;
- bundled UVDoc: `af9bd378eadbc17c3e53a7027fde34d8b74c26e5e6634db8552dc052fd916903`.

Thus the score changes below come from the metric/alignment correction, not different model
outputs.

## Corrected common-128 result

OCR used the same 60-image setting-1 subset and `tesseract v5.5.0.20241111`. Higher quality and
MS-SSIM are better; lower AAD proxy, OCR edit distance and CER are better.

| Candidate | Old stretched quality | Schema-v2 quality | Delta | MS-SSIM | AAD proxy | OCR ED | OCR CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DvD | 0.390395 | **0.334565** | -0.055830 | **0.468686** | **0.405111** | 471.22 | 0.169030 |
| DocScanner-L | 0.380053 | 0.334144 | -0.045909 | 0.456998 | 0.430443 | **369.05** | **0.143900** |
| bundled UVDoc | 0.362330 | 0.317465 | -0.044865 | 0.377250 | 0.447936 | 405.17 | 0.148940 |

DvD remains first on the image composite, but its lead over DocScanner-L collapses from `0.010342`
to `0.000421`. This is effectively a tie at the precision justified by these proxy metrics, while
DocScanner-L remains the clear OCR winner.

Across the 128 cases, mean aspect score was `0.907852` for DvD and `0.915885` for DocScanner-L and
UVDoc. DvD had 82 cases with more than 5% aspect mismatch and 47 with more than 10%; the other two
candidates had 81 and 46 respectively. The largest new evidence is DvD case `64_1`: reference size
`2550x3300`, output size `766x2639`, absolute log-ratio error `0.979144`, aspect score `0.375632`.
Case `64_2` similarly scores only `0.410087`. The previous unconditional resize concealed both
failures.

The 428,162-byte schema-v2 report has SHA-256
`12cbf2c28f94e53af10732e7ac1668a05b0352690985e2f16b966c16a7730839`.

## Full-130 cross-check

The two candidates with complete coverage preserve the same ordering:

| Candidate | Old stretched quality | Schema-v2 quality | Delta | MS-SSIM | AAD proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| DocScanner-L | 0.381019 | **0.334437** | -0.046582 | **0.458724** | **0.434936** |
| bundled UVDoc | 0.363337 | 0.317602 | -0.045735 | 0.379279 | 0.451522 |

The 286,110-byte full-130 report has SHA-256
`06a73df99042f4ae765a3045c38e4e1b9732deeea1c4be18e3348c3ded6c14b1`.

## Category and gate interpretation

DocUNet is deliberately a single-category profile (`docunet-corrected-common-128`), so its category
macro-average equals its only category score. This corpus cannot demonstrate a numerical category
balancing change. The multi-category unit test instead proves that duplicating cases in one category
does not let it dominate the final score; case weights apply only inside that category.

The perspective-regression gate is independent of the paired model tournament. Its regression tests
prove that a candidate is rejected when measured perspective worsens beyond the bounded tolerance,
while small measurement noise remains accepted. The focused benchmark/gate suite passed 34 tests;
the project suite passed 713 tests with 4 skipped and the locally broken Tk preview test deselected.

## Reproducibility defect found during the rerun

The importer writes absolute source paths into `sourceProvenance`, and the tournament hashes the
entire manifest. The same verified source bytes therefore produced manifest SHA-256
`6c643e1ff66cedb8624d20bc08c18e0a70aa36b3dd3b4574f9219c49629f474a` instead of the historical
`791bea59...fcee0`. This does not alter local image metrics, but it prevents a hash-bound official
sidecar from being reused after moving the source directories. A follow-up should make provenance
location-independent before treating manifest hashes as portable experiment identities.

## Decision

Do not use the old stretched quality values for model selection. Schema v2 removes a material bias
and shows that DvD and DocScanner-L are tied on the local image composite, with DocScanner-L still
better for OCR and far easier to deploy. Keep bundled UVDoc as the conservative automatic default
until the separate real-camera gate is broad enough to support a change.
