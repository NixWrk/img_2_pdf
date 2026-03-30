"""Build searchable PDFs from existing OCR text artifacts (artifact-first mode)."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Sequence


@dataclass(slots=True)
class ArtifactSearchableResult:
    document: str
    engine: str
    status: str
    source_pdf_path: str | None
    text_artifact_path: str
    searchable_pdf_path: str | None
    page_count: int
    text_chars: int
    elapsed_seconds: float
    error: str | None = None


_PAGE_MARKER_RE = re.compile(r"^\s*\[SOURCE PAGE\s+(\d+)\]\s*$", re.IGNORECASE)


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_artifact_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(
            f"Invalid artifact filename '{path.name}'. Expected '<document>__<engine>.txt'."
        )
    document, engine = stem.rsplit("__", 1)
    document = document.strip()
    engine = engine.strip().lower()
    if not document or not engine:
        raise ValueError(
            f"Invalid artifact filename '{path.name}'. Expected '<document>__<engine>.txt'."
        )
    return document, engine


def _split_text_to_pages(text: str, page_count: int) -> list[str]:
    if page_count <= 0:
        return []
    if page_count == 1:
        return [text]

    marker_pages: dict[int, list[str]] = {}
    current_page: int | None = None
    preamble: list[str] = []

    for line in text.splitlines():
        match = _PAGE_MARKER_RE.match(line)
        if match:
            current_page = int(match.group(1))
            marker_pages.setdefault(current_page, [])
            continue
        if current_page is None:
            preamble.append(line)
        else:
            marker_pages.setdefault(current_page, []).append(line)

    if marker_pages:
        pages: list[str] = []
        for page_idx in range(1, page_count + 1):
            if page_idx == 1 and preamble:
                source = preamble + marker_pages.get(page_idx, [])
            else:
                source = marker_pages.get(page_idx, [])
            pages.append("\n".join(source).strip())
        return pages

    # Many OCR tools emit form-feed as page separator.
    if "\f" in text:
        chunks = [chunk.strip() for chunk in text.split("\f")]
        if len(chunks) == page_count:
            return chunks

    lines = text.splitlines()
    if not lines:
        return [""] * page_count

    total_chars = max(len(text), 1)
    target_chars = max(total_chars // page_count, 1)
    pages: list[str] = []
    current_lines: list[str] = []
    current_chars = 0

    for idx, line in enumerate(lines):
        remaining_lines = len(lines) - idx
        remaining_pages = page_count - len(pages)
        can_finalize = (
            len(pages) < (page_count - 1)
            and current_lines
            and current_chars >= target_chars
            and remaining_lines >= (remaining_pages - 1)
        )
        if can_finalize:
            pages.append("\n".join(current_lines).strip())
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += len(line) + 1

    pages.append("\n".join(current_lines).strip())

    if len(pages) < page_count:
        pages.extend([""] * (page_count - len(pages)))
    elif len(pages) > page_count:
        overflow = pages[page_count - 1 :]
        pages = pages[: page_count - 1] + ["\n".join(part for part in overflow if part)]
    return pages


def _extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(part for part in parts if part.strip())


def _resolve_text_layer_font_path() -> Path:
    env_path = os.getenv("UNISCAN_TEXT_LAYER_FONT", "").strip()
    candidates = [
        env_path,
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for raw_path in candidates:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No unicode-capable TTF font found for searchable text layer. "
        "Set UNISCAN_TEXT_LAYER_FONT to a valid .ttf path."
    )


def _register_overlay_font(font_path: Path) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Deterministic alias for one process.
    font_alias = "UniscanTextLayerFont"
    try:
        pdfmetrics.getFont(font_alias)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_alias, str(font_path)))
    return font_alias


def _wrap_line_to_width(
    line: str,
    *,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    from reportlab.pdfbase import pdfmetrics

    if not line:
        return [""]
    if pdfmetrics.stringWidth(line, font_name, font_size) <= max_width:
        return [line]

    parts: list[str] = []
    start = 0
    while start < len(line):
        low = start + 1
        high = len(line)
        best = start + 1
        while low <= high:
            mid = (low + high) // 2
            chunk = line[start:mid]
            width = pdfmetrics.stringWidth(chunk, font_name, font_size)
            if width <= max_width:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= start:
            best = start + 1
        parts.append(line[start:best])
        start = best
        while start < len(line) and line[start] == " ":
            start += 1
    return parts


def _wrap_text_to_width(
    text: str,
    *,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    wrapped: list[str] = []
    for source_line in text.splitlines():
        wrapped.extend(
            _wrap_line_to_width(
                source_line,
                font_name=font_name,
                font_size=font_size,
                max_width=max_width,
            )
        )
    return wrapped or [""]


def _build_overlay_page(
    *,
    page_width: float,
    page_height: float,
    text: str,
    font_name: str,
):
    from pypdf import PdfReader
    from reportlab.pdfgen import canvas

    margin = 4.0
    max_width = max(page_width - margin * 2.0, 1.0)
    candidate_sizes = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4)

    chosen_size = candidate_sizes[-1]
    chosen_leading = max(chosen_size * 1.05, 0.4)
    chosen_lines: list[str] = _wrap_text_to_width(
        text,
        font_name=font_name,
        font_size=chosen_size,
        max_width=max_width,
    )

    for font_size in candidate_sizes:
        leading = max(font_size * 1.05, 0.4)
        max_lines = max(int((page_height - margin * 2.0) / leading), 1)
        lines = _wrap_text_to_width(
            text,
            font_name=font_name,
            font_size=font_size,
            max_width=max_width,
        )
        if len(lines) <= max_lines:
            chosen_size = font_size
            chosen_leading = leading
            chosen_lines = lines
            break
        chosen_size = font_size
        chosen_leading = leading
        chosen_lines = lines[:max_lines]

    packet = BytesIO()
    pdf_canvas = canvas.Canvas(packet, pagesize=(page_width, page_height), pageCompression=1)
    text_obj = pdf_canvas.beginText(margin, max(page_height - margin - chosen_size, margin))
    text_obj.setFont(font_name, chosen_size)
    text_obj.setLeading(chosen_leading)
    text_obj.setTextRenderMode(3)  # invisible selectable text
    for line in chosen_lines:
        text_obj.textLine(line)
    pdf_canvas.drawText(text_obj)
    pdf_canvas.save()

    packet.seek(0)
    return PdfReader(packet).pages[0]


def _build_searchable_pdf_from_text(
    *,
    source_pdf: Path,
    text: str,
    out_pdf: Path,
) -> tuple[int, int]:
    from pypdf import PdfReader, PdfWriter

    font_path = _resolve_text_layer_font_path()
    font_name = _register_overlay_font(font_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()

    page_count = len(reader.pages)
    page_texts = _split_text_to_pages(text, page_count)
    for page_idx, source_page in enumerate(reader.pages):
        page_text = page_texts[page_idx] if page_idx < len(page_texts) else ""
        if page_text.strip():
            page_width = float(source_page.mediabox.width)
            page_height = float(source_page.mediabox.height)
            overlay_page = _build_overlay_page(
                page_width=page_width,
                page_height=page_height,
                text=page_text,
                font_name=font_name,
            )
            source_page.merge_page(overlay_page)
        writer.add_page(source_page)

    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return page_count, len(text)


def run_artifact_searchable_package(
    *,
    compare_dir: Path,
    pdf_root: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
) -> list[ArtifactSearchableResult]:
    resolved_compare = Path(compare_dir)
    resolved_pdf_root = Path(pdf_root)
    resolved_output = Path(output_dir)

    if not resolved_compare.exists():
        raise FileNotFoundError(f"Compare dir not found: {resolved_compare}")
    if not resolved_pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {resolved_pdf_root}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    allowed_engines = None if engines is None else {engine.strip().lower() for engine in engines if engine.strip()}

    artifact_files = sorted(path for path in resolved_compare.glob("*.txt") if path.name.lower() != "sources_map.txt")

    pdf_index: dict[str, Path] = {}
    for pdf_path in resolved_pdf_root.rglob("*.pdf"):
        key = _normalize_key(pdf_path.stem)
        pdf_index.setdefault(key, pdf_path)

    results: list[ArtifactSearchableResult] = []

    for artifact_path in artifact_files:
        start = perf_counter()
        try:
            document, engine = _parse_artifact_filename(artifact_path)
        except Exception as exc:
            results.append(
                ArtifactSearchableResult(
                    document=artifact_path.stem,
                    engine="unknown",
                    status="error",
                    source_pdf_path=None,
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=str(exc),
                )
            )
            continue

        if allowed_engines is not None and engine not in allowed_engines:
            continue

        source_pdf = pdf_index.get(_normalize_key(document))
        if source_pdf is None:
            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="error",
                    source_pdf_path=None,
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=f"Source PDF not found in pdf_root for document '{document}'.",
                )
            )
            continue

        try:
            text = artifact_path.read_text(encoding="utf-8", errors="ignore")
            out_pdf = resolved_output / document / f"{document}__{engine}_searchable.pdf"
            page_count, text_chars = _build_searchable_pdf_from_text(
                source_pdf=source_pdf,
                text=text,
                out_pdf=out_pdf,
            )
            extracted = _extract_pdf_text(out_pdf)
            if not extracted.strip():
                raise RuntimeError("Output PDF has empty extracted text layer.")

            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="ok",
                    source_pdf_path=str(source_pdf),
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=str(out_pdf),
                    page_count=page_count,
                    text_chars=text_chars,
                    elapsed_seconds=perf_counter() - start,
                )
            )
        except Exception as exc:
            results.append(
                ArtifactSearchableResult(
                    document=document,
                    engine=engine,
                    status="error",
                    source_pdf_path=str(source_pdf),
                    text_artifact_path=str(artifact_path),
                    searchable_pdf_path=None,
                    page_count=0,
                    text_chars=0,
                    elapsed_seconds=perf_counter() - start,
                    error=str(exc),
                )
            )

    summary_json = resolved_output / "artifact_searchable_summary.json"
    summary_csv = resolved_output / "artifact_searchable_summary.csv"
    summary_json.write_text(
        json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "document",
                "engine",
                "status",
                "source_pdf_path",
                "text_artifact_path",
                "searchable_pdf_path",
                "page_count",
                "text_chars",
                "elapsed_seconds",
                "error",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))

    return results


def summarize_artifact_searchable_package(results: Sequence[ArtifactSearchableResult]) -> str:
    lines: list[str] = []
    for row in results:
        if row.status == "ok":
            lines.append(
                f"{row.document} [{row.engine}]: ok {row.elapsed_seconds:.2f}s "
                f"pages={row.page_count} text={row.text_chars} pdf={row.searchable_pdf_path}"
            )
        else:
            lines.append(
                f"{row.document} [{row.engine}]: error {row.elapsed_seconds:.2f}s "
                f"{row.error or 'unknown error'}"
            )
    return "\n".join(lines)
