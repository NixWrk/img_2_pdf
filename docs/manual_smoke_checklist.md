# Manual Windows smoke checklist

Record Windows version, artifact SHA-256, camera model, and operator in the release issue.

## Portable package

- [ ] Extract the ZIP into a new directory on a clean Windows 11 x64 machine.
- [ ] Run `uniscan.exe doctor` and confirm all non-camera checks are `ok`.
- [ ] Run `uniscan.exe doctor --camera --camera-index 0` with the target camera connected.
- [ ] Start `uniscan.exe`; verify Workspace, Camera, Import options, and Export options render.

## Camera and recovery

- [ ] Discover/open the real camera and verify the live preview updates.
- [ ] Capture one page and a delayed multi-shot burst; cancel one burst midway.
- [ ] Close UniScan with pages present, reopen it, and verify order/selection/images restore.
- [ ] Close the camera and application; confirm no camera lock remains in another app.

## Import, review, and export

- [ ] Import images by picker, drag-and-drop, and clipboard.
- [ ] Import a multi-page PDF and cancel a second large import midway.
- [ ] Review auto-crop, manual corners, page reorder/delete, and lighting correction.
- [ ] Export PNG/JPEG pages and a merged PDF; verify order and page count.
- [ ] Run `uniscan.exe convert --input INPUT --output OUTPUT.pdf --strict-detect`.
- [ ] Force a failed conversion over an existing output and verify the old output is preserved.

## Removal

- [ ] Delete the extracted `uniscan` directory.
- [ ] Optionally delete `%LOCALAPPDATA%\UniScan` to remove autosaved sessions.
- [ ] Confirm UniScan added no service, scheduled task, registry autorun, or system-wide package.
