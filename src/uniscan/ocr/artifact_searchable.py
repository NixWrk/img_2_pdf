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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[tuple[float, float, float, float, float, float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 20:
            continue
        if h < 4 or h > (pix.height * 0.25):
            continue
        if w < 2:
            continue
        x0 = float(x)
        y0 = float(y)
        x1 = float(x + w)
        y1 = float(y + h)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        components.append((x0, y0, x1, y1, cx, cy, float(h)))

    if not components:
        return []

    heights = [item[6] for item in components]
    median_h = float(sorted(heights)[len(heights) // 2]) if heights else 10.0
    y_threshold = max(4.0, median_h * 0.6)
    gap_threshold = max(20.0, median_h * 2.5)

    components.sort(key=lambda item: (item[5], item[0]))
    lines: list[dict[str, object]] = []
    for comp in components:
        x0, y0, x1, y1, _cx, cy, _h = comp
        target: dict[str, object] | None = None
        for line in reversed(lines[-8:]):
            line_cy = float(line["cy"])
            if abs(cy - line_cy) <= y_threshold:
                target = line
                break
        if target is None:
            target = {"items": [], "cy": cy}
            lines.append(target)
        target_items = target["items"]
        assert isinstance(target_items, list)
        target_items.append((x0, y0, x1, y1))
        target["cy"] = (float(target["cy"]) + cy) / 2.0

    raw_boxes: list[tuple[float, float, float, float]] = []
    for line in lines:
        items = list(line["items"])
        if not items:
            continue
        items.sort(key=lambda item: item[0])
        seg_x0, seg_y0, seg_x1, seg_y1 = items[0]
        prev_x1 = seg_x1
        for x0, y0, x1, y1 in items[1:]:
            if (x0 - prev_x1) > gap_threshold:
                if (seg_x1 - seg_x0) >= 2.0 and (seg_y1 - seg_y0) >= 2.0:
                    raw_boxes.append((seg_x0, seg_y0, seg_x1, seg_y1))
                seg_x0, seg_y0, seg_x1, seg_y1 = x0, y0, x1, y1
            else:
                seg_x0 = min(seg_x0, x0)
                seg_y0 = min(seg_y0, y0)
                seg_x1 = max(seg_x1, x1)
                seg_y1 = max(seg_y1, y1)
            prev_x1 = x1
        if (seg_x1 - seg_x0) >= 2.0 and (seg_y1 - seg_y0) >= 2.0:
            raw_boxes.append((seg_x0, seg_y0, seg_x1, seg_y1))

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
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
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
        assignments.append((box, " ".join(chunk)))
        if cursor >= len(lines):
            break
    if cursor < len(lines) and assignments:
        last_box, last_text = assignments[-1]
        tail = " ".join(lines[cursor:])
        assignments[-1] = (last_box, f"{last_text} {tail}".strip())
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

    packet = BytesIO()
    pdf_canvas = canvas.Canvas(packet, pagesize=(page_width, page_height), pageCompression=1)
    candidate_sizes = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4)

    for bbox, text in placements:
        if not text.strip():
            continue
        x0, y0, x1, y1 = bbox
        width = max(x1 - x0 - 1.0, 1.0)
        height = max(y1 - y0 - 1.0, 1.0)
        chosen_size = candidate_sizes[-1]
        chosen_leading = max(chosen_size * 1.05, 0.4)
        chosen_lines = [text]
        fitted = False

        for font_size in candidate_sizes:
            leading = max(font_size * 1.05, 0.4)
            max_lines = max(int(height / leading), 1)
            lines = _wrap_text_to_width(
                text,
                font_name=font_name,
                font_size=font_size,
                max_width=width,
            )
            if len(lines) <= max_lines:
                chosen_size = font_size
                chosen_leading = leading
                chosen_lines = lines
                fitted = True
                break

        if not fitted:
            # Preserve all text even for very small detected boxes.
            # We intentionally shrink invisible text size instead of dropping lines.
            min_font = 0.1
            lines = _wrap_text_to_width(
                text,
                font_name=font_name,
                font_size=min_font,
                max_width=width,
            )
            chosen_lines = lines
            if len(lines) == 0:
                continue
            leading = max(height / max(len(lines), 1), 0.08)
            chosen_leading = leading
            chosen_size = max(min(leading / 1.05, min_font), 0.05)

        baseline_y = max(page_height - y0 - chosen_size, 0.5)
        text_obj = pdf_canvas.beginText(max(x0 + 0.5, 0.5), baseline_y)
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
    import fitz  # type: ignore

    font_path = _resolve_text_layer_font_path()
    font_name = _register_overlay_font(font_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    layout_doc = fitz.open(str(source_pdf))

    page_count = len(reader.pages)
    page_line_boxes: list[list[tuple[float, float, float, float]]] = []
    for page_idx in range(page_count):
        page_line_boxes.append(_estimate_page_line_bboxes(page=layout_doc[page_idx]))

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
                page_width = float(source_page.mediabox.width)
                page_height = float(source_page.mediabox.height)
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


def build_compare_txt_from_benchmark(
    *,
    benchmark_root: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
) -> list[CompareTxtBuildResult]:
    resolved_root = Path(benchmark_root)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark summary.json not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark summary.json must contain a list of engine rows.")

    allowed_engines = None if engines is None else {item.strip().lower() for item in engines if item.strip()}
    source_map_lines = [f"Created: {summary_path.stat().st_mtime}"]
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
