# Rectify-first placement spike — 2026-08-20

## Decision

**Reject `raw split → page model → boundary crop` for the production pipeline.** After both
branches were reduced to one composed geometry sample, the model-first branch made boundary
detection unreliable and substantially worsened sharpness, OCR and measured page geometry.
Production remains `boundary → spread split → UVDoc → deskew`.

This closes G4 from
[`geometry_stage_order_audit_2026-08-15.md`](geometry_stage_order_audit_2026-08-15.md). Because the
spike was rejected, the conditional 21-page/21-detection adoption gate does not apply and no
production split order changed.

## Reproduction

Input: `example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf`, 1,659,192 bytes, SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`; source pages 3, 5, 7 and
11 at 216 DPI. OCR is Tesseract `rus+eng`, PSM 1.

```powershell
$env:PYTHONPATH='src'
python scripts\probe_stage_order.py `
  --pdf "example\Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf" `
  --dpi 216 `
  --tesseract <tesseract.exe> `
  --work-dir <scratch> `
  --output <report.json> `
  --probe placement
```

The corrected probe first obtains matched raw-frame halves at the production gutter ratio. In the
spike branch UVDoc receives each raw half with its own background; its grid, the subsequent
boundary homography and deskew rotation are composed, then the raw pixels are sampled once. This
replaces the old, invalid whole-spread-before-split comparison and avoids penalizing the spike with
an extra interpolation chain.

## Aggregate result

| Order | Boundaries | Curvature | Perspective | Sharpness | OCR chars | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boundary/split → UVDoc → deskew | **8/8** | **0.515** | **0.0130** | **300.0** | **4,138** | **51.62** |
| raw split → UVDoc → boundary → deskew | 4/8 | 1.578 | 0.0808 | 197.5 | 3,443 | 39.48 |

The raw-first branch lost four boundaries, mean sharpness fell by 34.2%, mean curvature rose by
206%, the perspective proxy rose more than sixfold, recognized characters fell by 16.8%, and mean
OCR confidence fell 12.14 points.

## Per-half result

| Half | Order | Boundary | Curvature | Perspective | Sharpness | OCR chars | Confidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| p3 L | shipped | yes | **0.360** | **0.0823** | **129.0** | **276** | **19.18** |
| p3 L | raw-first | no | 0.812 | 0.1014 | 64.6 | 169 | 14.68 |
| p3 R | shipped | yes | **0.313** | **0.0220** | **339.1** | **767** | 54.77 |
| p3 R | raw-first | yes | 2.214 | 0.0350 | 316.8 | 715 | **55.33** |
| p5 L | shipped | yes | **0.755** | 0.0000 | **147.0** | **311** | **59.92** |
| p5 L | raw-first | no | 2.065 | 0.0000 | 55.8 | 0 | 0.00 |
| p5 R | shipped | yes | **0.427** | 0.0000 | **109.0** | 422 | **50.40** |
| p5 R | raw-first | no | 1.317 | 0.0000 | 70.5 | **612** | 45.94 |
| p7 L | shipped | yes | **0.834** | 0.0000 | **343.4** | 229 | **52.71** |
| p7 L | raw-first | yes | 2.104 | 0.0000 | 251.0 | **304** | 48.66 |
| p7 R | shipped | yes | 0.555 | 0.0000 | **621.0** | 690 | **69.21** |
| p7 R | raw-first | yes | **0.506** | 0.0000 | 438.8 | **835** | 64.73 |
| p11 L | shipped | yes | **0.557** | **0.0000** | **325.3** | **782** | **57.26** |
| p11 L | raw-first | no | 3.218 | 0.4045 | 136.4 | 245 | 43.92 |
| p11 R | shipped | yes | **0.317** | **0.0000** | **385.9** | **661** | **49.54** |
| p11 R | raw-first | yes | 0.392 | 0.1050 | 246.1 | 563 | 42.61 |

## Interpretation

The page model does not preserve a sufficiently stable outer boundary when it sees these raw
halves. The failure is not merely detector tuning: on p3 R, p5 R and p11 L the model-first output
also has severe residual curvature, and p11 L gains a large perspective score. A model trained or
explicitly constrained to preserve the photographed page boundary could justify a new spike;
moving the current UVDoc model upstream cannot.
