# Auto Crop Benchmark Sketch Plan

## Goal

Add a small reproducible benchmark flow for comparing three document-cropping backends on the same input folder:

1. `camscan`
2. `opencv_quad`
3. `uvdoc`

Input: one folder with source photos.

Output: three PDFs with cropped/rectified pages, one PDF per backend, so the user can compare accuracy visually on the same batch.

## Scope

This is a validation sketch, not the final production batch pipeline.

The sketch should:

1. Keep folder order exactly as listed by the existing natural-sort loader.
2. Run each backend independently on the same images.
3. Export one merged PDF per backend.
4. Write clear backend-specific output names.
5. Fail gracefully when an optional backend is unavailable.

## Commit Plan

1. `docs(plan): add benchmark sketch roadmap for crop backends`
Document the benchmark goal, output contract, and staged implementation.

2. `feat(cv): add optional UVDoc backend adapter`
Add a third detector backend using PaddleOCR `TextImageUnwarping (UVDoc)` with repo-local model cache and safe fallback behavior.

3. `feat(cli): add batch crop benchmark sketch`
Add a CLI command that takes an input folder and output directory, processes the folder through all three backends, and writes three PDFs.

4. `test(cli): cover backend benchmark flow`
Add deterministic tests for backend selection, output naming, processing order, and PDF export orchestration without depending on live model downloads.

## Acceptance Criteria

1. `uniscan benchmark-crop --input <folder> --output <dir>` produces backend-specific PDFs.
2. The output order matches the folder order from the source directory.
3. Missing optional backends do not break other backend runs.
4. Tests validate the orchestration layer and backend adapter contract.
