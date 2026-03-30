# Searchable PDF From Existing TXT Artifacts (2026-03-30)

## Goal

Build searchable PDFs for:

1. `chandra`
2. `olmocr`
3. `surya`

without rerunning OCR models, using existing latest `.txt` artifacts.

## Command

```powershell
d:\Git_Code\img_2_pdf\.venv_latest_olmocr\Scripts\python.exe -m uniscan build-searchable-from-artifacts `
  --compare-dir "D:\Git_Code\img_2_pdf\artifacts\ocr_obs_gost_oldbook_20260327_165121\_compare_txt" `
  --pdf-root "D:\Git_Code\PDF\PDFs" `
  --output ".\artifacts\searchable_pdf_from_txt_20260330" `
  --engines chandra surya olmocr `
  --strict
```

## Result

All 6 combinations completed with `status=ok`:

1. `ГОСТ с плохим качеством скана` x `chandra`
2. `ГОСТ с плохим качеством скана` x `olmocr`
3. `ГОСТ с плохим качеством скана` x `surya`
4. `Старая книга с частично рукописным текстом` x `chandra`
5. `Старая книга с частично рукописным текстом` x `olmocr`
6. `Старая книга с частично рукописным текстом` x `surya`

## Output Files

Summary:

1. `artifacts/searchable_pdf_from_txt_20260330/artifact_searchable_summary.json`
2. `artifacts/searchable_pdf_from_txt_20260330/artifact_searchable_summary.csv`

Generated PDFs:

1. `artifacts/searchable_pdf_from_txt_20260330/ГОСТ с плохим качеством скана/ГОСТ с плохим качеством скана__chandra_searchable.pdf`
2. `artifacts/searchable_pdf_from_txt_20260330/ГОСТ с плохим качеством скана/ГОСТ с плохим качеством скана__olmocr_searchable.pdf`
3. `artifacts/searchable_pdf_from_txt_20260330/ГОСТ с плохим качеством скана/ГОСТ с плохим качеством скана__surya_searchable.pdf`
4. `artifacts/searchable_pdf_from_txt_20260330/Старая книга с частично рукописным текстом/Старая книга с частично рукописным текстом__chandra_searchable.pdf`
5. `artifacts/searchable_pdf_from_txt_20260330/Старая книга с частично рукописным текстом/Старая книга с частично рукописным текстом__olmocr_searchable.pdf`
6. `artifacts/searchable_pdf_from_txt_20260330/Старая книга с частично рукописным текстом/Старая книга с частично рукописным текстом__surya_searchable.pdf`
