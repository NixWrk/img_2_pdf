# OCR -> Searchable PDF Execution Plan (2026-03-30)

This roadmap defines the implementation order for searchable PDF text-layer
integration for:

1. `chandra`
2. `olmocr` (standalone path, no dependency on other OCR models)
3. `surya`

## Guiding Constraints

1. Separate virtual environment per engine is mandatory.
2. `olmocr` path must remain standalone.
3. Deliver in small, testable commits.

## Phase Order

## Phase 0: Shared Foundation

Goal: create one reusable pipeline for `OCR output -> hOCR -> searchable PDF`.

Commit sequence:

1. `feat(ocr-core): add canonical OCR schema and validators`
2. `feat(pdf-layer): add hOCR builder and ocrmypdf glue pipeline`
3. `feat(cli): add searchable-pdf command with engine switch`
4. `test(core): add pipeline smoke tests and fixtures`
5. `docs(core): describe OCR->hOCR->searchable PDF flow`

Exit criteria:

1. Shared CLI path exists.
2. hOCR generation is tested with synthetic fixtures.

## Phase 1: Chandra First (quality leader)

Goal: get first production-grade path online with best quality baseline.

Commit sequence:

1. `chore(chandra): add isolated venv setup and pinned requirements`
2. `feat(chandra): implement output adapter to canonical schema`
3. `feat(chandra): normalize page coordinates and reading order`
4. `feat(chandra): generate hOCR from canonical output`
5. `feat(chandra): build searchable PDF path via ocrmypdf`
6. `test(chandra): add integration tests on GOST and oldbook`
7. `docs(chandra): add usage, limits, and quality notes`

Exit criteria:

1. Searchable PDF is generated on both target documents.
2. Search and copy-paste are usable.

## Phase 2: Surya Integration

Goal: add second independent path through the same shared core.

Commit sequence:

1. `chore(surya): add isolated venv setup and pinned requirements`
2. `feat(surya): implement adapter from Surya lines/polygons to canonical schema`
3. `feat(surya): add polygon->bbox normalization and rotation handling`
4. `feat(surya): generate hOCR from normalized Surya output`
5. `feat(surya): build searchable PDF path via ocrmypdf`
6. `test(surya): add integration tests on GOST and oldbook`
7. `docs(surya): add usage, caveats, and quality notes`

Exit criteria:

1. Output quality is benchmarked against Chandra baseline.
2. No regressions in shared pipeline.

## Phase 3: OLMOCR Standalone Path

Goal: create searchable PDF from `olmocr` without fallback to Chandra/Surya text or geometry.

Commit sequence:

1. `chore(olmocr): add isolated venv setup and docker backend config`
2. `feat(olmocr): parse ocrflux/olmocr outputs into per-page text model`
3. `feat(olmocr): add standalone page layout detector (CV-based line boxes)`
4. `feat(olmocr): align page text to detected boxes with confidence scoring`
5. `feat(olmocr): generate hOCR without external OCR engines`
6. `feat(olmocr): build searchable PDF path via ocrmypdf from standalone hOCR`
7. `feat(olmocr): add strict mode (no fallback to other OCR models)`
8. `test(olmocr): add regression tests for GOST and oldbook`
9. `docs(olmocr): document standalone architecture and tuning params`

Exit criteria:

1. `olmocr` searchable PDF path works independently.
2. Failures are explicit per page with quality scores.

## Phase 4: Stabilization and Release

Goal: unify runtime ergonomics and lock benchmark evidence.

Commit sequence:

1. `chore(env): add one-click setup scripts for per-engine venvs`
2. `feat(benchmark): add final matrix runner for chandra/olmocr/surya searchable-pdf outputs`
3. `docs(benchmark): publish final comparison report and artifacts index`
4. `chore(release): tag searchable-pdf-matrix-v1`

Exit criteria:

1. Reproducible setup on target workstation.
2. Documented final ranking and benchmark methodology.
