# OCR Benchmark Plan

## Goal

Run one benchmark flow that executes all registered OCR engines on the same sampled PDF pages and produces comparable artifacts and metrics.

Canonical fixture:
`J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf`

The fixture is large, so benchmark execution must stay sampled / ranged:

1. Never materialize the full PDF into memory.
2. Render only sampled windows for tests and routine benchmark runs.
3. Keep full-document runs optional and explicit.

## Engine Matrix

All engines must go through the benchmark runner:

1. `pytesseract` (searchable PDF)
2. `ocrmypdf` (searchable PDF)
3. `pymupdf` (searchable PDF)
4. `paddleocr` (text extraction artifact)
5. `surya` (text extraction artifact)
6. `mineru` (text extraction artifact)

## Output Contract

Each run must generate:

1. Per-engine artifact (`.pdf` for searchable engines, `.txt` for extraction engines).
2. Unified JSON report with:
   `engine`, `status`, `sample_pages`, `elapsed_seconds`, `text_chars`, `artifact_path`, `error`, `note`.
3. Stable summary text for CLI output.

## Execution Policy

Default sampled window:

1. First `N` pages.
2. Middle `N` pages.
3. Last `N` pages.

Required behavior:

1. Every engine is included in the report.
2. Missing dependencies are explicit and machine-readable.
3. Engine failures do not break the whole benchmark run.
4. A strict mode can fail the CLI run when any engine is not `ok`.

## Commit Plan

1. `docs(plan): define all-engine OCR benchmark contract`
Expand benchmark requirements to all engines, add output contract and strict-mode rule.

2. `feat(ocr-bench): wire all-engine adapter runner`
Route `paddleocr`, `surya`, and `mineru` through extraction adapters so each engine is executed by one unified runner.

3. `feat(ocr-bench): add strict mode and readiness policy`
Add CLI strict mode and deterministic handling for missing dependencies / per-engine failures.

4. `test(ocr-bench): cover all-engine routing`
Add tests that verify searchable engines, extraction engines, and strict-mode behavior.

5. `perf(ocr-bench): add memory metric hook`
Add per-engine memory delta metric where available (best-effort and optional).

## Exit Criteria

1. `benchmark-ocr` can run with all engines in one command.
2. Every engine appears in JSON output with deterministic `ok/error` status.
3. Searchable and extraction engines emit the correct artifact type.
4. Tests cover all-engine routing and strict-mode exit behavior.
