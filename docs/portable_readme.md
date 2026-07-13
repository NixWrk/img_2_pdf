# UniScan portable for Windows x64

This directory is a self-contained UniScan build. It prepares camera captures, images, and PDF
pages for downstream OCR, then exports processed images or a plain merged PDF. It does not perform
OCR or create searchable PDFs.

## Run

1. Keep the complete extracted directory together.
2. Run `uniscan.exe doctor` and confirm the required checks are `ok`.
3. Run `uniscan.exe` to open the GUI.
4. For headless conversion, run:

   ```powershell
   .\uniscan.exe convert --input INPUT --output OUTPUT.pdf
   ```

Use `.\uniscan.exe --help` and `.\uniscan.exe convert --help` for all options. Processing is local;
unfinished GUI sessions are stored under `%LOCALAPPDATA%\UniScan`.

The default document detector is the bundled-code OpenCV hybrid backend. Optional Office Lens
weights are not distributed; users with separately licensed weights may configure the source
installation described in `docs\document_geometry.md`.

## Package contents and removal

- `LICENSE.txt` — UniScan's MIT license.
- `THIRD_PARTY_LICENSES\INDEX.txt` — validated dependency/frozen-payload inventory and copied
  license/runtime notice files.
- `CHANGELOG.md` — version history.
- `docs\windows_release.md` — artifact and signing status.
- `docs\manual_smoke_checklist.md` — validation procedure.

To uninstall, close UniScan and delete this directory. Optionally delete
`%LOCALAPPDATA%\UniScan` to remove recoverable sessions. UniScan installs no service, autorun, or
system-wide package.
