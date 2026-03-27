# OCR Benchmark Run: 2026-03-27 (`ocr_olmocr_only_final`)

## Configuration

- Script: `scripts/benchmark_ocr_matrix.ps1`
- PDF: `E:\Knowledge_Base\Elvis-V\...\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf`
- Output root: `artifacts/ocr_olmocr_only_final`
- Pages: sampled (`-SampleSize 1`)
- DPI: `160`
- Engine: `olmocr`
- Backend: `docker` (`chatdoc/ocrflux:latest`) via `UNISCAN_OLMOCR_BACKEND=docker`
- Venv: `.venv_latest_olmocr`

## Run Command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_ocr_matrix.ps1 -PdfPath "E:\Knowledge_Base\Elvis-V\...\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf" -OutputRoot ".\artifacts\ocr_olmocr_only_final" -Dpi 160 -Engines "olmocr"
```

## Result

| Engine | Status | Elapsed (s) | Text chars | Artifact |
|---|---|---:|---:|---|
| olmocr | ok | 155.350 | 723 | `artifacts/ocr_olmocr_only_final/olmocr/Imaging Edge Mobile_paddleocr_uvdoc_olmocr.txt` |

## Saved Files

- `artifacts/ocr_olmocr_only_final/summary.csv`
- `artifacts/ocr_olmocr_only_final/summary.json`
- `artifacts/ocr_olmocr_only_final/olmocr/run.log`
- `artifacts/ocr_olmocr_only_final/olmocr/Imaging Edge Mobile_paddleocr_uvdoc_ocr_benchmark.json`

## Notes

1. Tesseract-based paths (`pytesseract`, `ocrmypdf`, `pymupdf` in OCR mode) are excluded from target stack due quality mismatch on required language content.
2. `olmocr` is retained as a fast and high-quality PDF-to-markdown/text path.
3. Next integration task: use `olmocr` text output as source for searchable-PDF assembly path.
