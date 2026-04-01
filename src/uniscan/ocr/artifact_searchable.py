"""Build searchable PDFs from existing OCR text artifacts (artifact-first mode)."""

from __future__ import annotations

import csv
import json
import math
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


@dataclass(slots=True)
class CompareTxtBuildResult:
    engine: str
    status: str
    source_artifact_path: str | None
    compare_txt_path: str | None
    error: str | None = None


_PAGE_MARKER_RE = re.compile(r"^\s*\[SOURCE PAGE\s+(\d+)\]\s*$", re.IGNORECASE | re.MULTILINE)


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


def _has_explicit_page_markers(text: str) -> bool:
    if "\f" in text:
        return True
    return bool(_PAGE_MARKER_RE.search(text))


def _split_lines_to_pages_by_weights(
    lines: Sequence[str],
    *,
    page_count: int,
    page_weights: Sequence[float] | None,
) -> list[str]:
    if page_count <= 0:
        return []
    if not lines:
        return [""] * page_count

    if page_weights is None or len(page_weights) != page_count:
        weights = [1.0] * page_count
    else:
        weights = [max(float(value), 0.0) for value in page_weights]
        if sum(weights) <= 0:
            weights = [1.0] * page_count

    total_weight = float(sum(weights))
    total_lines = len(lines)
    raw_counts = [(total_lines * weight / total_weight) for weight in weights]
    counts = [int(math.floor(value)) for value in raw_counts]
    used = sum(counts)
    remainder = total_lines - used

    if remainder > 0:
        order = sorted(
            range(page_count),
            key=lambda idx: (
                raw_counts[idx] - counts[idx],
                raw_counts[idx],
                -idx,
            ),
            reverse=True,
        )
        for idx in order[:remainder]:
            counts[idx] += 1

    pages: list[str] = []
    cursor = 0
    for count in counts:
        chunk = list(lines[cursor : cursor + count])
        cursor += count
        pages.append("\n".join(chunk).strip())

    if cursor < total_lines:
        tail = "\n".join(lines[cursor:]).strip()
        if pages:
            pages[-1] = f"{pages[-1]}\n{tail}".strip() if pages[-1] else tail
    if len(pages) < page_count:
        pages.extend([""] * (page_count - len(pages)))
    return pages[:page_count]


def _estimate_page_split_weights(
    page_line_boxes: Sequence[Sequence[tuple[float, float, float, float]]],
) -> list[float]:
    if not page_line_boxes:
        return []

    counts = [float(len(items)) for items in page_line_boxes]
    positive = sorted(value for value in counts if value > 0)
    if not positive:
        return [1.0] * len(counts)

    # Trim upper tail so one noisy page does not consume most text lines.
    trim_count = max(1, int(len(positive) * 0.8))
    trimmed = positive[:trim_count]
    median = trimmed[len(trimmed) // 2]
    high_clip = max(4.0, median * 2.5)
    return [min(value, high_clip) if value > 0 else 0.1 for value in counts]


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


def _estimate_page_line_bboxes(
    *,
    page,
) -> list[tuple[float, float, float, float]]:
    try:
        import fitz  # type: ignore
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return []

    matrix = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    if pix.width <= 0 or pix.height <= 0:
        return []

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img[:, :, 0]

    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Keep character strokes while removing isolated salt noise.
    binary_inv = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    # Connect nearby characters along the same baseline into denser text rows.
    binary_inv = cv2.morphologyEx(
        binary_inv,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
        iterations=1,
    )

    row_ink = np.count_nonzero(binary_inv, axis=1)
    min_row_ink = max(6, int(pix.width * 0.004))
    active_rows = row_ink >= min_row_ink
    if not np.any(active_rows):
        return []

    runs: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, active in enumerate(active_rows):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            runs.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(active_rows) - 1))

    if not runs:
        return []

    merged_runs: list[tuple[int, int]] = []
    for y0, y1 in runs:
        if not merged_runs:
            merged_runs.append((y0, y1))
            continue
        prev_y0, prev_y1 = merged_runs[-1]
        # Merge tiny vertical gaps to keep one box per text line.
        if (y0 - prev_y1) <= 2:
            merged_runs[-1] = (prev_y0, y1)
        else:
            merged_runs.append((y0, y1))

    raw_boxes: list[tuple[float, float, float, float]] = []
    min_box_width = max(10, int(pix.width * 0.05))
    for y0, y1 in merged_runs:
        h = y1 - y0 + 1
        if h < 3 or h > int(pix.height * 0.20):
            continue
        band = binary_inv[y0 : y1 + 1, :]
        col_ink = np.count_nonzero(band, axis=0)
        min_col_ink = max(1, int(h * 0.12))
        active_cols = np.where(col_ink >= min_col_ink)[0]
        if active_cols.size == 0:
            continue
        x0 = int(active_cols[0])
        x1 = int(active_cols[-1]) + 1
        if (x1 - x0) < min_box_width:
            continue
        # Small padding keeps descenders/ascenders inside the target box.
        py = max(1, int(h * 0.15))
        px = max(1, int((x1 - x0) * 0.01))
        raw_boxes.append(
            (
                float(max(0, x0 - px)),
                float(max(0, y0 - py)),
                float(min(pix.width, x1 + px)),
                float(min(pix.height, y1 + py)),
            )
        )

    if not raw_boxes:
        return []

    raw_boxes.sort(key=lambda item: (item[1], item[0]))
    scale_x = float(page.rect.width) / float(pix.width)
    scale_y = float(page.rect.height) / float(pix.height)
    result: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in raw_boxes:
        bx0 = max(0.0, x0 * scale_x)
        by0 = max(0.0, y0 * scale_y)
        bx1 = min(float(page.rect.width), x1 * scale_x)
        by1 = min(float(page.rect.height), y1 * scale_y)
        if bx1 <= bx0 or by1 <= by0:
            continue
        result.append((bx0, by0, bx1, by1))
    return result


def _split_page_text_lines(text: str) -> list[str]:
    def _clean_overlay_line(raw_line: str) -> str:
        # Strip lightweight markdown/html markers so they do not leak into the
        # selectable text layer (<b>, <math>, etc.).
        line = re.sub(r"</?[^>\n]+>", "", raw_line)
        line = line.replace("\u00a0", " ")
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    lines: list[str] = []
    for raw in text.splitlines():
        line = _clean_overlay_line(raw)
        if not line:
            continue
        if line.startswith("[SOURCE PAGE"):
            continue
        lines.append(line)
    return lines


def _assign_lines_to_boxes(
    lines: Sequence[str],
    boxes: Sequence[tuple[float, float, float, float]],
) -> list[tuple[tuple[float, float, float, float], str]]:
    if not lines:
        return []
    if not boxes:
        return []

    normalized_boxes = list(boxes)
    if len(normalized_boxes) > 1:
        heights = sorted(max(item[3] - item[1], 1.0) for item in normalized_boxes)
        median_h = float(heights[len(heights) // 2]) if heights else 10.0
        y_threshold = max(2.0, median_h * 0.7)
        rows: list[dict[str, float]] = []
        for x0, y0, x1, y1 in sorted(normalized_boxes, key=lambda item: (item[1], item[0])):
            cy = (y0 + y1) / 2.0
            target: dict[str, float] | None = None
            for row in reversed(rows[-6:]):
                if abs(cy - row["cy"]) <= y_threshold:
                    target = row
                    break
            if target is None:
                rows.append(
                    {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "cy": cy,
                        "count": 1.0,
                    }
                )
            else:
                count = target["count"]
                target["x0"] = min(target["x0"], x0)
                target["y0"] = min(target["y0"], y0)
                target["x1"] = max(target["x1"], x1)
                target["y1"] = max(target["y1"], y1)
                target["cy"] = (target["cy"] * count + cy) / (count + 1.0)
                target["count"] = count + 1.0

        normalized_boxes = [
            (row["x0"], row["y0"], row["x1"], row["y1"])
            for row in sorted(rows, key=lambda row: (row["y0"], row["x0"]))
        ]

    if len(lines) <= len(normalized_boxes):
        if len(lines) == 1:
            x0 = min(item[0] for item in normalized_boxes)
            y0 = min(item[1] for item in normalized_boxes)
            x1 = max(item[2] for item in normalized_boxes)
            y1 = max(item[3] for item in normalized_boxes)
            return [((x0, y0, x1, y1), lines[0])]

        assignments: list[tuple[tuple[float, float, float, float], str]] = []
        span = max(len(normalized_boxes) - 1, 1)
        for idx, line in enumerate(lines):
            rel = idx / max(len(lines) - 1, 1)
            box_idx = int(round(rel * span))
            assignments.append((normalized_boxes[box_idx], line))
        return assignments

    assignments: list[tuple[tuple[float, float, float, float], str]] = []
    cursor = 0
    for box_idx, box in enumerate(normalized_boxes):
        remaining_lines = len(lines) - cursor
        remaining_boxes = len(normalized_boxes) - box_idx
        take = max(1, int(math.ceil(remaining_lines / remaining_boxes)))
        chunk = lines[cursor : cursor + take]
        cursor += take
        assignments.append((box, "\n".join(chunk)))
        if cursor >= len(lines):
            break
    if cursor < len(lines) and assignments:
        last_box, last_text = assignments[-1]
        tail = "\n".join(lines[cursor:])
        assignments[-1] = (last_box, f"{last_text}\n{tail}".strip())
    return assignments


def _build_overlay_page(
    *,
    page_width: float,
    page_height: float,
    placements: Sequence[tuple[tuple[float, float, float, float], str]],
    font_name: str,
):
    from pypdf import PdfReader
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics

    packet = BytesIO()
    pdf_canvas = canvas.Canvas(packet, pagesize=(page_width, page_height), pageCompression=1)

    for bbox, text in placements:
        if not text.strip():
            continue
        x0, y0, x1, y1 = bbox
        overlay_lines = [item.strip() for item in text.splitlines() if item.strip()]
        if not overlay_lines:
            continue

        line_height = max((y1 - y0) / max(len(overlay_lines), 1), 0.5)
        for line_idx, line in enumerate(overlay_lines):
            sub_y0 = y0 + (line_height * line_idx)
            sub_y1 = y0 + (line_height * (line_idx + 1))
            width = max(x1 - x0 - 0.6, 0.5)
            height = max(sub_y1 - sub_y0 - 0.4, 0.4)
            font_size = max(min(height * 0.80, 32.0), 0.12)
            natural_width = max(pdfmetrics.stringWidth(line, font_name, font_size), 0.01)
            horiz_scale = max(min((width / natural_width) * 100.0, 800.0), 10.0)
            baseline_y = max(page_height - sub_y1 + (height - font_size) * 0.55, 0.2)

            text_obj = pdf_canvas.beginText(max(x0 + 0.2, 0.2), baseline_y)
            text_obj.setFont(font_name, font_size)
            text_obj.setTextRenderMode(3)  # invisible selectable text
            if abs(horiz_scale - 100.0) > 0.5:
                text_obj.setHorizScale(horiz_scale)
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
    import fitz  # type: ignore

    font_path = _resolve_text_layer_font_path()
    font_name = _register_overlay_font(font_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    layout_doc = fitz.open(str(source_pdf))

    page_count = len(reader.pages)
    page_line_boxes: list[list[tuple[float, float, float, float]]] = []
    page_layout_sizes: list[tuple[float, float]] = []
    for page_idx in range(page_count):
        layout_page = layout_doc[page_idx]
        page_line_boxes.append(_estimate_page_line_bboxes(page=layout_page))
        page_layout_sizes.append((float(layout_page.rect.width), float(layout_page.rect.height)))

    page_texts = _split_text_to_pages(text, page_count)
    marker_pages_detected = bool(_PAGE_MARKER_RE.search(text)) or ("\f" in text)
    if not marker_pages_detected:
        all_lines = _split_page_text_lines(text)
        page_weights = _estimate_page_split_weights(page_line_boxes)
        page_texts = _split_lines_to_pages_by_weights(
            all_lines,
            page_count=page_count,
            page_weights=page_weights,
        )

    try:
        for page_idx, source_page in enumerate(reader.pages):
            page_text = page_texts[page_idx] if page_idx < len(page_texts) else ""
            if page_text.strip():
                crop_box = source_page.cropbox
                media_box = source_page.mediabox
                crop_x0 = float(crop_box.left)
                crop_y0 = float(crop_box.bottom)
                crop_width = float(crop_box.width)
                crop_height = float(crop_box.height)
                layout_width, layout_height = page_layout_sizes[page_idx]
                page_width = layout_width if layout_width > 0 else (crop_width if crop_width > 0 else float(media_box.width))
                page_height = layout_height if layout_height > 0 else (crop_height if crop_height > 0 else float(media_box.height))
                line_boxes = page_line_boxes[page_idx]
                page_lines = _split_page_text_lines(page_text)
                placements = _assign_lines_to_boxes(page_lines, line_boxes)
                if not placements and page_lines:
                    placements = [((4.0, 4.0, page_width - 4.0, page_height - 4.0), " ".join(page_lines))]
                overlay_page = _build_overlay_page(
                    page_width=page_width,
                    page_height=page_height,
                    placements=placements,
                    font_name=font_name,
                )
                if abs(crop_x0) > 1e-6 or abs(crop_y0) > 1e-6:
                    source_page.merge_translated_page(overlay_page, crop_x0, crop_y0)
                else:
                    source_page.merge_page(overlay_page)
            writer.add_page(source_page)
    finally:
        layout_doc.close()

    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return page_count, len(text)


def run_artifact_searchable_package(
    *,
    compare_dir: Path,
    pdf_root: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
    require_page_markers: bool = False,
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
            if require_page_markers and not _has_explicit_page_markers(text):
                raise ValueError(
                    "TXT artifact has no explicit page markers. "
                    "Expected '[SOURCE PAGE N]' blocks or form-feed separators."
                )
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


def _load_compare_source_rows(benchmark_root: Path) -> tuple[list[dict], list[str]]:
    summary_path = benchmark_root / "summary.json"
    if summary_path.exists():
        try:
            raw_summary = summary_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_summary = summary_path.read_text(encoding="utf-8-sig")
        if raw_summary.startswith("\ufeff"):
            raw_summary = raw_summary.lstrip("\ufeff")
        payload = json.loads(raw_summary)
        if not isinstance(payload, list):
            raise ValueError("Benchmark summary.json must contain a list of engine rows.")
        return [row for row in payload if isinstance(row, dict)], [f"summary_json={summary_path}"]

    report_paths = sorted(
        (path for path in benchmark_root.rglob("*_ocr_benchmark.json") if path.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    payload: list[dict] = []
    seen_engines: set[str] = set()
    source_reports: list[str] = []
    for report_path in report_paths:
        try:
            report_raw = report_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            report_raw = report_path.read_text(encoding="utf-8-sig")
        try:
            report_payload = json.loads(report_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(report_payload, dict):
            continue
        results = report_payload.get("results")
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, dict):
                continue
            engine = str(row.get("engine") or "").strip().lower()
            if not engine or engine in seen_engines:
                continue
            payload.append(dict(row))
            seen_engines.add(engine)
            source_reports.append(str(report_path))

    if not payload:
        raise FileNotFoundError(
            f"Benchmark summary.json not found: {summary_path} and no *_ocr_benchmark.json reports found under {benchmark_root}"
        )
    unique_reports = list(dict.fromkeys(source_reports))
    return payload, [f"discovered_reports={len(unique_reports)}"] + unique_reports


def build_compare_txt_from_benchmark(
    *,
    benchmark_root: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
) -> list[CompareTxtBuildResult]:
    resolved_root = Path(benchmark_root)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    payload, source_map_lines = _load_compare_source_rows(resolved_root)

    allowed_engines = None if engines is None else {item.strip().lower() for item in engines if item.strip()}
    results: list[CompareTxtBuildResult] = []

    for row in payload:
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine") or "").strip().lower()
        if not engine:
            continue
        if allowed_engines is not None and engine not in allowed_engines:
            continue

        status = str(row.get("status") or "").strip().lower()
        artifact_raw = str(row.get("artifact_path") or "").strip()
        if status != "ok":
            results.append(
                CompareTxtBuildResult(
                    engine=engine,
                    status="error",
                    source_artifact_path=artifact_raw or None,
                    compare_txt_path=None,
                    error=f"engine status is '{status or 'unknown'}'",
                )
            )
            continue
        if not artifact_raw:
            results.append(
                CompareTxtBuildResult(
                    engine=engine,
                    status="error",
                    source_artifact_path=None,
                    compare_txt_path=None,
                    error="artifact_path is empty in summary row",
                )
            )
            continue

        source_path = Path(artifact_raw)
        if source_path.suffix.lower() == ".pdf":
            txt_sidecar = source_path.with_suffix(".txt")
            if txt_sidecar.exists():
                source_path = txt_sidecar

        if source_path.suffix.lower() != ".txt" or not source_path.exists():
            engine_dir = resolved_root / engine
            fallback_candidates: list[Path] = []
            if engine_dir.exists():
                fallback_candidates.extend(sorted(engine_dir.glob(f"*_{engine}.txt")))
                if not fallback_candidates:
                    fallback_candidates.extend(sorted(engine_dir.rglob(f"*_{engine}.txt")))
            if fallback_candidates:
                source_path = fallback_candidates[0]

        if source_path.suffix.lower() != ".txt" or not source_path.exists():
            results.append(
                CompareTxtBuildResult(
                    engine=engine,
                    status="error",
                    source_artifact_path=str(source_path),
                    compare_txt_path=None,
                    error="source text artifact not found",
                )
            )
            continue

        suffix = f"_{engine}"
        stem = source_path.stem
        document = stem[: -len(suffix)] if stem.lower().endswith(suffix) else stem
        if not document.strip():
            results.append(
                CompareTxtBuildResult(
                    engine=engine,
                    status="error",
                    source_artifact_path=str(source_path),
                    compare_txt_path=None,
                    error=f"cannot resolve document name from artifact '{source_path.name}'",
                )
            )
            continue

        compare_path = resolved_output / f"{document}__{engine}.txt"
        compare_path.write_text(source_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        source_map_lines.append(str(source_path))
        results.append(
            CompareTxtBuildResult(
                engine=engine,
                status="ok",
                source_artifact_path=str(source_path),
                compare_txt_path=str(compare_path),
                error=None,
            )
        )

    (resolved_output / "sources_map.txt").write_text("\n".join(source_map_lines) + "\n", encoding="utf-8")
    return results


def summarize_compare_txt_build(results: Sequence[CompareTxtBuildResult]) -> str:
    lines: list[str] = []
    for row in results:
        if row.status == "ok":
            lines.append(f"{row.engine}: ok compare_txt={row.compare_txt_path}")
        else:
            lines.append(f"{row.engine}: error {row.error or 'unknown error'}")
    return "\n".join(lines)
