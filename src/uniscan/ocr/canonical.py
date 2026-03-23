"""Canonical OCR packaging for equal-condition engine comparison."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

from uniscan.io import imwrite_unicode, render_pdf_page_indices

from .benchmark import _run_extraction_engine, resolve_pdf_page_indices
from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_VALUES,
    SEARCHABLE_PDF_ENGINES,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)


@dataclass(slots=True)
class CanonicalOcrResult:
    engine: str
    status: str
    elapsed_seconds: float
    sample_pages: list[int]
    text_chars: int
    canonical_dir: str | None
    searchable_pdf_path: str | None
    error: str | None = None


def _pdf_page_count(pdf_path: Path) -> int:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    try:
        return int(doc.page_count)
    finally:
        doc.close()


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
    except Exception as exc:
        fitz_error = exc

    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        if fitz_error is not None:
            raise RuntimeError(f"PDF extraction failed via fitz ({fitz_error}) and pypdf ({exc}).") from exc
        raise RuntimeError(f"PDF extraction failed via pypdf ({exc}).") from exc


def _extract_page_text(
    engine: str,
    image_path: Path,
    *,
    lang: str,
    work_dir: Path,
) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    if engine in SEARCHABLE_PDF_ENGINES:
        page_pdf = work_dir / f"{image_path.stem}_{engine}.pdf"
        output_pdf = image_paths_to_searchable_pdf(
            [image_path],
            out_pdf=page_pdf,
            lang=lang,
            engine_name=engine,
        )
        return _extract_pdf_text(output_pdf)

    text, _chars = _run_extraction_engine(
        engine,
        [image_path],
        lang=lang,
        work_dir=work_dir,
        which_fn=shutil.which,
        run_cmd=subprocess.run,
    )
    return text


def _build_text_only_searchable_pdf(page_texts: Sequence[str], *, out_pdf: Path) -> Path:
    import fitz  # type: ignore

    width, height = fitz.paper_size("a4")
    margin = 36.0
    line_height = 10.0
    wrap_width = 110
    font_size = 8.0

    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        y = margin

        for page_number, text in enumerate(page_texts, start=1):
            block_lines = [f"[SOURCE PAGE {page_number:04d}]"]
            block_lines.append("")
            if text.strip():
                for raw_line in text.splitlines():
                    wrapped = textwrap.wrap(raw_line, width=wrap_width, break_long_words=True) or [""]
                    block_lines.extend(wrapped)
            else:
                block_lines.append("<EMPTY>")
            block_lines.append("")

            for line in block_lines:
                if y > (height - margin):
                    page = doc.new_page(width=width, height=height)
                    y = margin
                page.insert_text((margin, y), line, fontsize=font_size, fontname="cour")
                y += line_height

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_pdf))
    finally:
        doc.close()

    return out_pdf


def run_ocr_canonical_package(
    *,
    pdf_path: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
    sample_size: int = 5,
    page_numbers: Sequence[int] | None = None,
    dpi: int = 160,
    lang: str = "eng",
) -> list[CanonicalOcrResult]:
    """Run OCR engines and package outputs into a canonical comparable format."""
    resolved_pdf = Path(pdf_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    source_pages_dir = resolved_output / "source_pages"
    canonical_root = resolved_output / "canonical"
    searchable_root = resolved_output / "searchable_pdf"
    source_pages_dir.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)
    searchable_root.mkdir(parents=True, exist_ok=True)

    page_count = _pdf_page_count(resolved_pdf)
    sample_pages = resolve_pdf_page_indices(
        page_count,
        sample_size=sample_size,
        page_numbers=page_numbers,
    )
    if not sample_pages:
        raise ValueError("No PDF pages available for canonical OCR packaging.")

    rendered = render_pdf_page_indices(resolved_pdf, sample_pages, dpi=dpi)
    sampled_images: list[Path] = []
    for idx, (_name, image) in enumerate(rendered, start=1):
        out_path = source_pages_dir / f"page_{idx:04d}.png"
        if not imwrite_unicode(out_path, image):
            raise RuntimeError(f"Failed to write source page image: {out_path}")
        sampled_images.append(out_path)

    selected_engines = tuple(engines) if engines is not None else OCR_ENGINE_VALUES
    results: list[CanonicalOcrResult] = []

    with tempfile.TemporaryDirectory(prefix="uniscan_canonical_ocr_") as tmp:
        tmp_root = Path(tmp)
        for engine_name in selected_engines:
            engine = engine_name.strip().lower()
            start = perf_counter()
            engine_dir = canonical_root / engine
            engine_dir.mkdir(parents=True, exist_ok=True)
            searchable_pdf_path = searchable_root / f"{engine}.pdf"

            try:
                status = detect_ocr_engine_status(engine)
                if not status.ready:
                    missing = ", ".join(status.missing) if status.missing else "unknown"
                    raise RuntimeError(f"Engine is not ready: {missing}")

                page_texts: list[str] = []
                total_chars = 0
                for page_idx, image_path in enumerate(sampled_images, start=1):
                    page_work = tmp_root / engine / f"page_{page_idx:04d}"
                    text = _extract_page_text(
                        engine,
                        image_path,
                        lang=lang,
                        work_dir=page_work,
                    )
                    page_texts.append(text)
                    total_chars += len(text)
                    (engine_dir / f"page_{page_idx:04d}.txt").write_text(text, encoding="utf-8")

                (engine_dir / "all_pages.txt").write_text("\n\n".join(page_texts), encoding="utf-8")
                _build_text_only_searchable_pdf(page_texts, out_pdf=searchable_pdf_path)

                results.append(
                    CanonicalOcrResult(
                        engine=engine,
                        status="ok",
                        elapsed_seconds=perf_counter() - start,
                        sample_pages=[page + 1 for page in sample_pages],
                        text_chars=total_chars,
                        canonical_dir=str(engine_dir),
                        searchable_pdf_path=str(searchable_pdf_path),
                    )
                )
            except Exception as exc:
                results.append(
                    CanonicalOcrResult(
                        engine=engine,
                        status="error",
                        elapsed_seconds=perf_counter() - start,
                        sample_pages=[page + 1 for page in sample_pages],
                        text_chars=0,
                        canonical_dir=str(engine_dir),
                        searchable_pdf_path=None,
                        error=str(exc),
                    )
                )

    json_path = resolved_output / "canonical_summary.json"
    csv_path = resolved_output / "canonical_summary.csv"
    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "engine",
                "status",
                "elapsed_seconds",
                "sample_pages",
                "text_chars",
                "canonical_dir",
                "searchable_pdf_path",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return results


def summarize_ocr_canonical_package(results: Sequence[CanonicalOcrResult]) -> str:
    lines: list[str] = []
    for row in results:
        if row.status == "ok":
            lines.append(
                f"{row.engine}: ok {row.elapsed_seconds:.2f}s text={row.text_chars} pdf={row.searchable_pdf_path}"
            )
        else:
            lines.append(
                f"{row.engine}: error {row.elapsed_seconds:.2f}s {row.error or 'unknown error'}"
            )
    return "\n".join(lines)
