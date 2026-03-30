# Searchable PDF Recheck From Existing TXT (2026-03-30)

Purpose:

1. Verify final implementation state for `chandra`, `surya`, `olmocr`.
2. Reuse existing OCR text artifacts only (no OCR model reruns).

Command:

```powershell
$env:PYTHONPATH='d:\Git_Code\img_2_pdf\src'
python -m uniscan build-searchable-from-artifacts `
  --compare-dir "d:\Git_Code\img_2_pdf\artifacts\ocr_obs_gost_oldbook_20260327_165121\_compare_txt" `
  --pdf-root "d:\Git_Code\PDF\PDFs" `
  --output "d:\Git_Code\img_2_pdf\artifacts\searchable_pdf_from_txt_20260330_recheck" `
  --engines chandra surya olmocr `
  --strict
```

Result:

1. `ok`: 6
2. `error`: 0

Per-model time (seconds):

1. `gost_bad_scan_quality` (`ГОСТ с плохим качеством скана.pdf`):
   - `chandra`: `0.16`
   - `olmocr`: `0.14`
   - `surya`: `0.16`
2. `oldbook_partial_handwriting` (`Старая книга с частично рукописным текстом.pdf`):
   - `chandra`: `0.15`
   - `olmocr`: `0.24`
   - `surya`: `0.22`

Artifacts:

1. `artifacts/searchable_pdf_from_txt_20260330_recheck/*/*_searchable.pdf`
2. `artifacts/searchable_pdf_from_txt_20260330_recheck/artifact_searchable_summary.json`
3. `artifacts/searchable_pdf_from_txt_20260330_recheck/artifact_searchable_summary.csv`

## Coordinate-Line Rebuild (2026-03-30)

Update:

1. Rebuilt the same 6 PDFs with per-line geometric placement (coordinate-like mode).
2. No OCR reruns were performed; same TXT inputs were reused.

Per-model time (seconds):

1. `gost_bad_scan_quality`:
   - `chandra`: `3.73`
   - `olmocr`: `3.78`
   - `surya`: `2.44`
2. `oldbook_partial_handwriting`:
   - `chandra`: `19.14`
   - `olmocr`: `31.79`
   - `surya`: `9.74`
