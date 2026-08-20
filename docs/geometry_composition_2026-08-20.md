# Single-pass geometry record — 2026-08-20

This record closes the numerical gate for G1 in
[`geometry_stage_order_audit_2026-08-15.md`](geometry_stage_order_audit_2026-08-15.md).

## Contract

Perspective boundary, split-page rectification, a manual four-corner adjustment, UVDoc's backward
grid and the residual deskew rotation are represented as output-to-source maps. Map composition
may interpolate coordinates, but only the final renderer reads authoritative source pixels.
`PageProcessingDiagnostics.geometry_resample_count` and batch-report
`geometryResampleCount` expose that count.

The unit regression constructs crop + user dewarp + deskew independently, verifies the controller
reports one sample, and compares its pixels byte-for-byte with the direct single-pass reference.
It also requires at least 15% more Laplacian variance than the old chained render.

## Tracked example

Input: `example/Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf`, SHA-256
`75511d2cddaba5773666fefd66454bf7d1aa9ce6469b9c14d6d62f5bbff2b343`, rendered at 216 DPI with
the production PDFium byte order. The deterministic probe uses the left half of source page 7 and
the same final geometry for both renderers.

```powershell
$env:PYTHONPATH='src'
python scripts\probe_stage_order.py `
  --pdf "example\Neudachi_v_mokrokollodionnom_protsesse [OCR].pdf" `
  --page 7 --dpi 216 --work-dir <scratch> --probe composition
```

| Renderer | Source-pixel samples | Sharpness | Curvature |
| --- | ---: | ---: | ---: |
| historical chained stages | 5 | 347.7 | 0.374 |
| composed single pass | **1** | **405.3** | **0.348** |

The composed result is 16.6% sharper than the chained result and is the single-pass reference
itself (0% difference, within the 3% limit). The same probe on page 3 improved 136.8 → 154.8
(13.2%); the gain varies with local texture and subpixel phase, while the one-sample invariant does
not.
