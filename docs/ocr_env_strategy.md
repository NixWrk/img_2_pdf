# OCR Environment Strategy

## Why Conflicts Happened In One venv

When all OCR engines are installed into one environment, they pull incompatible transitive dependencies. The most important collisions we observed:

1. `transformers` + `huggingface-hub`
`surya-ocr`/`mineru` paths require a `transformers`-compatible `huggingface-hub` (`<1.0`). Installing newer stacks can upgrade `huggingface-hub` too far and break imports.

2. `paddleocr` runtime (`PIR` / `oneDNN`)
`paddleocr` with some `paddlepaddle` versions can fail at runtime with:
`ConvertPirAttribute2RuntimeAttribute ... onednn_instruction`.

3. `langchain` API split
Some `paddle` ecosystem paths still rely on older `langchain` API locations (for example `langchain.docstore`). Newer `langchain 1.x` can break those imports.

4. `surya-ocr` strict pins
`surya-ocr` expects older `Pillow` and `pypdfium2` ranges than some other stacks install by default.

5. `mineru` runtime modules
`mineru` execution can fail if `ftfy`, `dill`, or `omegaconf` are missing.

## Single venv: Reproducible Conflict-Resolved Stack

Use this only when you intentionally want one environment for all engines:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall `
  "paddleocr==3.4.0" "paddlex==3.4.2" "paddlepaddle==3.1.1" `
  "transformers==4.57.1" "tokenizers==0.22.1" "huggingface-hub==0.34.4" `
  "ftfy==6.3.1" "dill==0.4.1" "omegaconf==2.3.0" `
  "langchain==0.2.17" "langchain-community==0.2.19" "langchain-core==0.2.43"
```

Also ensure system tools are in PATH when needed:

1. `tesseract` for `pytesseract`/`pymupdf` OCR mode.
2. `ocrmypdf` command for `ocrmypdf` engine mode.

## Recommended Benchmark Method: Separate venv Per Engine

For objective "latest release" comparison, use isolated environments per engine. This avoids cross-engine dependency pollution and gives cleaner performance numbers.

The repo includes an automation script:

`scripts/benchmark_ocr_matrix.ps1`

It will:

1. Create one venv per engine.
2. Install the project editable package in each venv.
3. Install latest engine-specific dependencies in each venv.
4. Run `uniscan benchmark-ocr` per engine.
5. Save per-engine logs, package versions, JSON report artifacts.
6. Build matrix summaries (`summary.json` and `summary.csv`).

From `2026-03-30` page-aware extraction artifacts are also written automatically
for text engines (`chandra`, `surya`, `olmocr`, etc.):

1. `<engine>/page_XXXX.txt` - one OCR text file per source page.
2. `<engine>/pages.json` - page index metadata.
3. `<engine>/all_pages.txt` and root `<pdf_stem>_<engine>.txt` - markerized
   aggregate text with `[SOURCE PAGE N]`.

## One-Command Run (Matrix Mode)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_ocr_matrix.ps1 `
  -PdfPath "J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf" `
  -OutputRoot ".\artifacts\ocr_latest_matrix" `
  -SampleSize 1 `
  -Dpi 160 `
  -Recreate
```

Optional quick subset run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_ocr_matrix.ps1 `
  -Engines paddleocr,surya `
  -SampleSize 1 `
  -Recreate
```

## Build compare_txt from benchmark output

After matrix run for one document:

```powershell
python -m uniscan prepare-compare-txt `
  --benchmark-root ".\artifacts\ocr_latest_matrix\My Document" `
  --output ".\artifacts\ocr_latest_matrix\My Document\_compare_txt" `
  --engines chandra surya olmocr `
  --strict
```

Then build searchable PDF from markerized artifacts:

```powershell
python -m uniscan build-searchable-from-artifacts `
  --compare-dir ".\artifacts\ocr_latest_matrix\My Document\_compare_txt" `
  --pdf-root "O:\OBS_TEST\PDF2OBS\PDFS" `
  --output ".\artifacts\searchable_from_compare_txt" `
  --engines chandra surya olmocr `
  --strict
```
