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

## Page-Split/Placement Fix Rebuild (2026-03-30)

Update:

1. Added weighted page split when source TXT has no explicit page markers.
2. Added robust line-box assignment to reduce text concentration in top page area.
3. Reused existing TXT artifacts only (no OCR reruns).

Command:

```powershell
$env:PYTHONPATH='d:\Git_Code\img_2_pdf\src'
python -m uniscan build-searchable-from-artifacts `
  --compare-dir "d:\Git_Code\img_2_pdf\artifacts\ocr_obs_gost_oldbook_20260327_165121\_compare_txt" `
  --pdf-root "d:\Git_Code\PDF\PDFs" `
  --output "d:\Git_Code\img_2_pdf\artifacts\searchable_pdf_from_txt_20260330_recheck_v3" `
  --engines chandra surya olmocr `
  --strict
```

Result:

1. `ok`: 6
2. `error`: 0
3. Pages with non-empty extractable text layer:
   - `ГОСТ` (`37 pages`): `chandra 37/37`, `olmocr 37/37`, `surya 37/37`
   - `Старая книга` (`33 pages`): `chandra 33/33`, `olmocr 33/33`, `surya 33/33`
