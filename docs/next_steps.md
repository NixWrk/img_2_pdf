# Next Steps

This roadmap keeps UniScan focused on the pre-OCR stage: acquire pages, correct document
geometry, clean images, and export images or a plain merged PDF.

## Milestone 1 — Pipeline hardening (P0)

### 1. Stream multi-page PDFs page by page

- Replace full-document `render_pdf_pages()` use in the headless flow with page iteration.
- Keep only the current source page and processed output in memory.
- Add a high-volume regression test using a generated multi-page PDF.

Done when memory use no longer grows linearly with PDF page count and page order/output remain
identical to the current pipeline.

### 2. Make output commits atomic

- Build PDFs and image sets in a staging location.
- Replace the destination only after successful processing.
- Preserve an existing valid output when conversion fails or is cancelled.
- Emit a concise machine-readable run report with processed, detected, and fallback page counts.

Done when forced failures leave no partial PDF and return a documented non-zero exit code.

### 3. Expose detector policy in the production CLI

- Add `--backend auto|office_lens_onnx|cv_hybrid|paddleocr_uvdoc`.
- Add `--strict-detect` to fail instead of silently keeping an undetected original page.
- Record the selected backend and fallback reason per page in the run report.

Done when backend choice and fallback behavior have deterministic CLI tests.

### 4. Finish repository hygiene

- Remove the unused tracked `naps2-7.5.3-win.exe` after confirming no distribution requirement.
- Add a top-level `LICENSE` matching the MIT package metadata.
- Normalize formatting in a dedicated mechanical commit.

Done when the repository contains no unexplained binary executable and `ruff format --check .`
passes.

## Milestone 2 — Continuous verification (P1)

### 1. Add CI

- Test Python 3.11 and the newest supported Python on Windows and Linux headless runners.
- Run `ruff check`, `ruff format --check`, `pytest`, `pip check`, and wheel build/contents checks.
- Verify that both bundled ONNX models can create inference sessions from the built wheel.

Done when every pull request receives reproducible lint, test, inference-smoke, and packaging
results.

### 2. Raise meaningful coverage

- Separate GUI-independent controllers from Tk widgets.
- Cover CLI failures, PDF iteration, cancellation, output rollback, and detector fallback reports.
- Add a fake camera implementation for deterministic camera-service tests.

Done when non-GUI modules reach at least 80% coverage and total coverage reaches at least 60%
without excluding important error paths.

## Milestone 3 — GUI reliability and recoverability (P1)

- Add session autosave and restore after an application crash.
- Test background import/export cancellation and window-close cleanup.
- Add drag-and-drop and clipboard image import.
- Create a short manual smoke checklist for camera discovery, preview, capture, review, and export.

Done when an interrupted session can be restored and the manual Windows smoke checklist passes on
at least one real camera.

## Milestone 4 — Scan quality (P2)

- Add automatic orientation correction without introducing OCR into this repository.
- Prototype glare and shadow reduction behind an opt-in setting.
- Create a versioned, legally redistributable benchmark corpus for documents, whiteboards,
  photographs, books, and difficult lighting.
- Track crop success, corner error, latency, and fallback rate instead of relying only on visual
  comparison.

Done when quality changes are supported by before/after metrics and do not regress the bundled
Office Lens baseline.

## Milestone 5 — Windows release (P2)

- Produce a versioned Windows artifact with the ONNX models included.
- Add startup diagnostics for missing camera/runtime capabilities.
- Add changelog, release checklist, clean-machine install test, and uninstall instructions.
- Decide whether code signing is required before public distribution.

Done when a clean Windows machine can install, run the GUI, execute `uniscan convert`, and remove
the application without a development Python environment.

## Recommended implementation order

1. CI plus repository hygiene.
2. Streaming PDF input plus atomic outputs.
3. Detector selection and structured run reports.
4. Coverage improvements and fake camera tests.
5. Session recovery and GUI workflow improvements.
6. Scan-quality experiments.
7. Windows packaging and release automation.
