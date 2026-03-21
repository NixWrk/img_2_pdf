# Auto Crop Benchmark Sketch Plan

## Goal

Keep one stable crop backend for production use and preserve the benchmark flow as a regression check.

Current result from the real batch test:

1. `paddleocr_uvdoc` is the only backend that stayed reliable on the full photo series.
2. The OpenCV variants are useful as references, but they are not the active production path.
3. `camscan` is not strong enough on this dataset to keep in the active path.

Input: one folder with source photos.

Output: one production PDF from `paddleocr_uvdoc`, plus optional reference PDFs when explicitly requested.

## Scope

This is a validation sketch, not the final production batch pipeline.

The sketch should:

1. Keep folder order exactly as listed by the existing natural-sort loader.
2. Use `paddleocr_uvdoc` as the default and only active backend.
3. Allow explicit opt-in runs for reference backends during diagnostics.
4. Export one merged PDF per selected backend.
5. Write clear backend-specific output names.
6. Fail gracefully when an optional backend is unavailable.

## Commit Plan

1. `docs(plan): add benchmark sketch roadmap for crop backends`
Document the benchmark goal, output contract, and staged implementation.

2. `feat(cv): add optional UVDoc backend adapter`
Add a third detector backend using PaddleOCR `TextImageUnwarping (UVDoc)` with repo-local model cache and safe fallback behavior.

3. `feat(cli): add batch crop benchmark sketch`
Add a CLI command that takes an input folder and output directory, processes the folder through the selected backend set, and writes PDFs.

4. `test(cli): cover backend benchmark flow`
Add deterministic tests for backend selection, output naming, processing order, and PDF export orchestration without depending on live model downloads.

5. `chore(config): make paddleocr_uvdoc the default crop backend`
Reduce the active crop path to the one backend that passed the real-series test consistently.

## Acceptance Criteria

1. `uniscan benchmark-crop --input <folder> --output <dir>` produces a `paddleocr_uvdoc` PDF by default.
2. The output order matches the folder order from the source directory.
3. Missing optional backends do not break explicit diagnostic runs.
4. Tests validate the orchestration layer and backend adapter contract.
