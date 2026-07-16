# Manual Windows smoke checklist

Record Windows version, artifact SHA-256, camera model, and operator in the release issue.

## Portable package

- [ ] Extract the ZIP into a new directory on a clean Windows 11 x64 machine.
- [ ] Run `uniscan.exe doctor` and confirm all non-camera checks are `ok`.
- [ ] Run `uniscan.exe doctor --camera --camera-index 0` with the target camera connected.
- [ ] Start `uniscan.exe`; verify the Workspace renders without a neighboring Camera tab.

## Camera and recovery

- [ ] Discover/open the real camera and verify the live preview updates.
- [ ] Open Camera from the top action bar, close it, and verify Workspace remains responsive.
- [ ] Capture one page and a delayed multi-shot burst; cancel one burst midway.
- [ ] Close UniScan with pages present, reopen it, and verify order/selection/images restore.
- [ ] While UniScan is open, start a second copy against the same local session; confirm it stops
      with a clear single-writer message, leaves the first copy untouched, and starts normally
      after the first copy closes.
- [ ] Close the camera and application; confirm no camera lock remains in another app.

## Import, review, and export

- [ ] Import images by picker, drag-and-drop, and clipboard.
- [ ] Import a multi-page PDF and cancel a second large import midway.
- [ ] Review inline auto-crop, manual corners, page reorder/delete, and lighting correction.
- [ ] In inline wave editing, add/remove points, drag a point in both axes, and drag the complete
      center curve vertically; verify the corrected preview updates after each operation.
- [ ] Change processing controls and Preview without Apply; confirm export still uses the previous
      committed page. Apply, then confirm preview and export use the new full-resolution result.
- [ ] Start Apply on several large pages and cancel midway; confirm the background job remains
      responsive and every page, recipe, and diagnostic rolls back to its pre-Apply state.
- [ ] Export PNG/JPEG pages and a merged PDF; verify order and page count.
- [ ] Convert a PDF with different input-render and output PDF DPI values; verify both values in the
      report. Commit an A4/Letter page, change export DPI, confirm GUI export refuses the
      physical-size mismatch, then Apply at the new DPI and export successfully.
- [ ] Try an oversized raster/PDF page and an A4/Letter allocation over the safe pixel limit;
      confirm processing fails closed before publication and preserves existing outputs.
- [ ] Run `uniscan.exe convert --input INPUT --output OUTPUT.pdf --strict-detect`.
- [ ] Force a failed conversion over an existing output and verify the old output is preserved.

## Optional Office Lens models

- [ ] With a valid quad model and no classifier file, run `doctor` and an explicit Office Lens mode;
      confirm the classifier is reported as disabled, the overall check remains successful, and
      quad detection works.
- [ ] Add a classifier file that exists but cannot be loaded; confirm `doctor` reports a
      non-blocking warning and explicit Office Lens modes still work.
- [ ] Remove the quad model; confirm `doctor` reports Office Lens as disabled and the Office Lens
      boundary backend does not start.
- [ ] Add a quad-model file that exists but cannot be loaded; confirm `doctor` reports a blocking
      failure and the Office Lens boundary backend does not start.

## Removal

- [ ] Delete the extracted `uniscan` directory.
- [ ] Optionally delete `%LOCALAPPDATA%\UniScan` to remove autosaved sessions.
- [ ] Confirm UniScan added no service, scheduled task, registry autorun, or system-wide package.
