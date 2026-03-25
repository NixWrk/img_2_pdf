# ocrmypdf-surya

Surya OCR plugin for OCRmyPDF.

## Compatibility

The current stable stack is pinned to:

1. `ocrmypdf>=16,<17`
2. `surya-ocr>=0.17,<0.18`
3. `pillow<11`
4. `pypdfium2==4.30.0`

`surya-ocr 0.17.x` is not compatible with OCRmyPDF 17.x because of
`pypdfium2` requirements.

## Local development install

```powershell
.\.venv_latest_surya\Scripts\python.exe -m pip install -e .\OCRmypdf_plugins\ocrmypdf-surya-main
```

Use with OCRmyPDF bridge mode:

```powershell
ocrmypdf --plugin ocrmypdf_surya --force-ocr --language rus input.pdf output.pdf
```
