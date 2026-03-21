# OCR Testing Plan

## Goal

Build a reliable OCR validation path for the project and keep it separate from crop/rectification testing.

OCR must be tested on four axes:

1. Dependency readiness.
2. Engine selection and fallback behavior.
3. Searchable PDF output quality.
4. Real-world fixture regression and performance.

## Current OCR Surface

The repo currently exposes these OCR engines in `src/uniscan/ocr/engine.py`:

1. `pytesseract`
2. `ocrmypdf`
3. `paddleocr`
4. `pymupdf`
5. `surya`
6. `mineru`

Searchable PDF export is currently wired for:

1. `pytesseract`
2. `ocrmypdf`
3. `pymupdf`

The remaining engines are selectable and should be treated as OCR readiness / extraction candidates until searchable-PDF support is added.

## Test Strategy

### 1. Dependency Checks

Goal:
Verify that `detect_ocr_dependencies()` and `detect_ocr_engine_status()` report the correct readiness state without requiring the full external stack.

Tests:

1. Missing Python modules.
2. Missing executables in `PATH`.
3. Engine-specific ready / not-ready states.
4. Searchable-PDF capability flags per engine.

### 2. Engine Contract

Goal:
Ensure each engine name is stable and handled explicitly.

Tests:

1. Unsupported engine raises `ValueError`.
2. Engine labels remain stable for UI text.
3. Selectable engines remain in the expected list.
4. Searchable-PDF engines are the only ones allowed for PDF export without extra wiring.

### 3. Searchable PDF Export

Goal:
Verify the three wired engines produce the expected PDF orchestration.

Tests:

1. `pytesseract` path merges page PDFs.
2. `ocrmypdf` path builds a temp input PDF and invokes the CLI.
3. `pymupdf` path requires OCR-capable `Pixmap` support.
4. Export rejects empty input.
5. Export rejects unsupported engines that are not wired yet.

### 4. Fixture Regression

Goal:
Catch OCR regressions on real content, not only synthetic mocks.

Fixture buckets:

1. Clean Latin text.
2. Mixed Russian / English text.
3. Low-contrast scans.
4. Skewed / perspective-distorted pages after crop.
5. Multi-page PDF import sources.

Expected assertions:

1. Page count is preserved.
2. Output PDF opens successfully.
3. OCR text is not empty for known-text fixtures.
4. Searchable PDF contains text layer when the engine claims it should.
5. Engine-specific failures are surfaced with clear error messages.

### 5. Performance and Memory

Goal:
Keep OCR predictable on large batches.

Tests:

1. Time a small fixed fixture set per engine.
2. Ensure large batches do not load everything into RAM at once where possible.
3. Confirm progress and cancellation work in background export jobs.
4. Confirm OCR does not block unrelated UI tabs.

### 6. UI Flow

Goal:
Verify OCR stays a separate export concern and does not leak into crop/review processing.

Tests:

1. OCR toggle respects dependency status.
2. OCR engine dropdown disables unavailable engines.
3. Searchable PDF export uses the selected engine only at export time.
4. Review processing remains crop-only.

## Commit Plan

1. `docs(plan): add OCR testing roadmap`
Document the OCR test matrix, fixtures, and staged rollout.

2. `test(ocr): strengthen dependency and engine-status coverage`
Add or refine unit tests around readiness, labels, and capability flags.

3. `test(ocr): add searchable PDF export contract tests`
Cover the existing searchable-PDF engines with deterministic orchestration tests.

4. `test(ocr): add fixture-based OCR smoke tests`
Add small real-image fixtures for one or more supported OCR engines.

5. `perf(ocr): add batch timing and cancellation checks`
Validate OCR remains usable on larger batches and does not monopolize the UI thread.

6. `feat(ocr): wire remaining engines progressively`
Add searchable-PDF support or extraction glue for `paddleocr`, `surya`, and `mineru` only after test coverage exists.

## Exit Criteria

1. Every selectable OCR engine has a defined readiness test.
2. The three wired searchable-PDF engines have export tests.
3. At least one real fixture suite exists for regressions.
4. OCR failures are reported clearly to the UI.
5. The OCR path remains isolated from crop/rectification logic.
