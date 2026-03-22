"""Build a human-readable OCR comparison package in ./outputs."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_summary(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported summary payload format: {type(payload)!r}")


def _safe_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "run"


def _extract_pdf_text(pdf_path: Path) -> str:
    fitz_error: Exception | None = None
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        try:
            parts = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        text = "\n".join(part for part in parts if part)
        if text.strip():
            return text
    except Exception as exc:  # pragma: no cover - runtime fallback
        fitz_error = exc

    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as exc:  # pragma: no cover - runtime fallback
        if fitz_error is not None:
            raise RuntimeError(f"PDF text extraction failed via fitz ({fitz_error}) and pypdf ({exc}).") from exc
        raise RuntimeError(f"PDF text extraction failed via pypdf: {exc}") from exc


def _extract_text(artifact_path: Path) -> str:
    suffix = artifact_path.suffix.lower()
    if suffix == ".txt":
        return artifact_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(artifact_path)
    raise RuntimeError(f"Unsupported artifact extension for text extraction: {artifact_path.suffix}")


def _normalize_for_similarity(text: str, max_chars: int = 200_000) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:max_chars]


def _pairwise_similarity(engine_texts: dict[str, str]) -> list[tuple[str, str, float]]:
    engines = sorted(engine_texts.keys())
    result: list[tuple[str, str, float]] = []
    for idx, left in enumerate(engines):
        left_text = _normalize_for_similarity(engine_texts[left])
        for right in engines[idx + 1 :]:
            right_text = _normalize_for_similarity(engine_texts[right])
            ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
            result.append((left, right, ratio))
    return result


def _render_report(
    *,
    run_dir: Path,
    source_root: Path,
    summary_rows: list[dict[str, Any]],
    extracted_rows: list[dict[str, Any]],
    similarity_rows: list[tuple[str, str, float]],
) -> str:
    lines: list[str] = []
    lines.append("# OCR Comparison Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Source run: `{source_root}`")
    lines.append(f"- Output bundle: `{run_dir}`")
    lines.append("")

    lines.append("## Engine Summary")
    lines.append("")
    lines.append("| Engine | Status | Elapsed (s) | Text chars (benchmark) | Memory delta (MB) | Artifact |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| {engine} | {status} | {elapsed} | {chars} | {mem} | `{artifact}` |".format(
                engine=row.get("engine", ""),
                status=row.get("status", ""),
                elapsed=row.get("elapsed_seconds", ""),
                chars=row.get("text_chars", ""),
                mem=row.get("memory_delta_mb", ""),
                artifact=row.get("artifact_path", ""),
            )
        )
    lines.append("")

    lines.append("## Extracted Text Files")
    lines.append("")
    lines.append("| Engine | Extracted chars | Text file | Note |")
    lines.append("|---|---:|---|---|")
    for row in extracted_rows:
        lines.append(
            "| {engine} | {chars} | `{text_file}` | {note} |".format(
                engine=row["engine"],
                chars=row["extracted_chars"],
                text_file=row["text_file"],
                note=row["note"] or "",
            )
        )
    lines.append("")

    if similarity_rows:
        lines.append("## Pairwise Similarity (normalized text)")
        lines.append("")
        lines.append("| Engine A | Engine B | Similarity |")
        lines.append("|---|---|---:|")
        for left, right, ratio in sorted(similarity_rows, key=lambda item: item[2], reverse=True):
            lines.append(f"| {left} | {right} | {ratio:.4f} |")
        lines.append("")

    lines.append("## Snippets")
    lines.append("")
    for row in extracted_rows:
        lines.append(f"### {row['engine']}")
        lines.append("")
        lines.append(f"Source: `{row['artifact']}`")
        lines.append("")
        snippet = row["snippet"].strip()
        if snippet:
            lines.append("```text")
            lines.append(snippet)
            lines.append("```")
        else:
            lines.append("_No text extracted._")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package OCR comparison report into ./outputs.")
    parser.add_argument(
        "--input-root",
        default=str(_repo_root() / "artifacts" / "ocr_latest_matrix_full_run_final"),
        help="Path to OCR matrix run folder that contains summary.json and per-engine folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_repo_root() / "outputs"),
        help="Target root for comparison bundles.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional custom output folder name. Default: <input-folder>_compare_<timestamp>.",
    )
    args = parser.parse_args()

    source_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary_path = source_root / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"summary.json not found: {summary_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{source_root.name}_compare_{timestamp}"
    run_name = _safe_slug(args.run_name) if args.run_name.strip() else default_name
    run_dir = output_root / run_name
    results_copy_dir = run_dir / "results"
    texts_dir = run_dir / "texts"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    texts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, results_copy_dir)

    summary_rows = _load_summary(summary_path)
    extracted_rows: list[dict[str, Any]] = []
    engine_texts: dict[str, str] = {}

    for row in summary_rows:
        engine = str(row.get("engine", "")).strip() or "unknown"
        artifact_rel = str(row.get("artifact_path", "")).strip()
        artifact_path = Path(artifact_rel)
        if artifact_rel and not artifact_path.is_absolute():
            artifact_path = _repo_root() / artifact_path

        text_output_path = texts_dir / f"{_safe_slug(engine)}.txt"
        note = ""
        extracted_text = ""
        if not artifact_rel:
            note = "no artifact path"
        elif not artifact_path.exists():
            note = f"artifact missing: {artifact_path}"
        else:
            try:
                extracted_text = _extract_text(artifact_path)
            except Exception as exc:
                note = f"extract failed: {exc}"

        text_output_path.write_text(extracted_text, encoding="utf-8")
        if extracted_text:
            engine_texts[engine] = extracted_text

        extracted_rows.append(
            {
                "engine": engine,
                "artifact": artifact_rel,
                "text_file": str(text_output_path.relative_to(run_dir)),
                "extracted_chars": len(extracted_text),
                "note": note,
                "snippet": extracted_text[:1200],
            }
        )

    similarity_rows = _pairwise_similarity(engine_texts)
    report_md = _render_report(
        run_dir=run_dir,
        source_root=source_root,
        summary_rows=summary_rows,
        extracted_rows=extracted_rows,
        similarity_rows=similarity_rows,
    )

    report_path = run_dir / "ocr_comparison_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    metadata = {
        "source_root": str(source_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "summary_rows": summary_rows,
        "extracted_rows": [
            {
                "engine": row["engine"],
                "artifact": row["artifact"],
                "text_file": row["text_file"],
                "extracted_chars": row["extracted_chars"],
                "note": row["note"],
            }
            for row in extracted_rows
        ],
        "pairwise_similarity": [
            {"engine_a": left, "engine_b": right, "ratio": ratio}
            for left, right, ratio in similarity_rows
        ],
    }
    (run_dir / "ocr_comparison_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Comparison bundle: {run_dir}")
    print(f"Report: {report_path}")
    print(f"Results copy: {results_copy_dir}")
    print(f"Texts: {texts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
