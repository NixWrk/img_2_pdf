# UniScan portable for Windows x64

This directory contains a self-contained UniScan build. It imports camera captures, images, and
PDF pages; corrects and cleans them locally; and exports processed images or a plain, image-based
merged PDF. It does not perform OCR, create searchable PDFs, or preserve the text and vector
objects from an imported PDF.

## Start UniScan

1. Extract the complete ZIP into a user-writable directory and keep its contents together.
2. Run `uniscan.exe doctor`. Required checks must report `ok`; unavailable optional model features
   may be reported as warnings.
3. Run `uniscan.exe` to open the GUI.
4. For headless conversion, run:

   ```powershell
   .\uniscan.exe convert --input INPUT --output OUTPUT.pdf
   ```

Use `.\uniscan.exe --help` and `.\uniscan.exe convert --help` for the complete option list. A
typical conversion with separate input and output DPI settings is:

```powershell
.\uniscan.exe convert `
  --input scans another-document.pdf `
  --output result.pdf `
  --images-dir processed-pages `
  --input-pdf-dpi 300 `
  --output-pdf-dpi 300
```

`--input-pdf-dpi` controls rasterization of imported PDF pages. `--output-pdf-dpi` controls
physical page layout and PDF export. The older `--pdf-dpi` option remains a compatibility default
for each role that does not have a specific override.

Each raster image, TIFF frame, or rendered PDF page is limited to 150,000,000 pixels by default.
Oversized input fails instead of being silently downscaled; headless runs can choose a different
positive limit with `--max-input-pixels`.

## GUI sessions and processing

Processing is local. Recoverable GUI sessions and the processing cache are stored under
`%LOCALAPPDATA%\UniScan`. Only one UniScan process can own the shared autosave session at a time;
a second instance exits with a clear startup error instead of writing the same state concurrently.

Fast preview is approximate. `Apply processing` computes and commits the full-resolution result
in the background. Export uses the last committed pixels for each page, so changing processing
controls without applying them does not silently alter an export. If A4 or Letter layout was
applied at one DPI, reapply it before exporting at another DPI.

## Optional model backends

The default document detector is the built-in OpenCV hybrid policy. The portable package does not
include Office Lens model weights, ONNX Runtime, PaddleOCR, or UVDoc weights.

The Office Lens adapter is a bring-your-own-model source-installation feature for compatible
weights that the user is legally permitted to use. The required quad model and optional automatic
mode classifier are documented in the source repository's `docs/office_lens_onnx.md`; they cannot
be enabled by copying unverified model files into this portable package.

UVDoc is also a source-installation feature. When explicitly selected, PaddleOCR may initialize or
download its own model cache. The bundled offline text-line dewarper does not require that optional
runtime.

## Output safety

UniScan stages output before publication so cancellation or failure preserves the previous valid
result. Image-directory export tracks only files it owns and preserves unrelated neighbours.
Concurrent UniScan processes serialize publication to the same targets. These locks do not
coordinate edits made by Explorer, an image editor, or another application.

## Package contents

- `LICENSE.txt` — UniScan's MIT license.
- `THIRD_PARTY_LICENSES\INDEX.txt` — dependency and frozen-runtime inventory with copied notices.
- `CHANGELOG.md` — version history.
- `docs\windows_release.md` — artifact and signing status.
- `docs\manual_smoke_checklist.md` — manual validation procedure.
- `docs\document_geometry.md` — processing stages, options, and geometry diagnostics.

## Remove UniScan

Close UniScan and delete the extracted directory. To remove recoverable pages, settings, and the
processing cache as well, delete `%LOCALAPPDATA%\UniScan`. UniScan installs no service, registry
autorun, scheduled task, or system-wide Python package.
