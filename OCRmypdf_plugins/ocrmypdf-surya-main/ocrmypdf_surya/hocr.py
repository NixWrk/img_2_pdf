"""hOCR helpers for Surya OCR output."""

from __future__ import annotations

from html import escape
from typing import Any


def _safe_bbox(raw_bbox: Any, page_width: int, page_height: int) -> tuple[int, int, int, int]:
    try:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError("bbox must contain 4 items")
        x0, y0, x1, y1 = (int(float(raw_bbox[0])), int(float(raw_bbox[1])), int(float(raw_bbox[2])), int(float(raw_bbox[3])))
    except Exception:
        return (0, 0, max(1, int(page_width)), max(1, int(page_height)))

    page_width = max(1, int(page_width))
    page_height = max(1, int(page_height))
    x0 = max(0, min(x0, page_width - 1))
    y0 = max(0, min(y0, page_height - 1))
    x1 = max(x0 + 1, min(x1, page_width))
    y1 = max(y0 + 1, min(y1, page_height))
    return (x0, y0, x1, y1)


def build_hocr(
    *,
    page_width: int,
    page_height: int,
    lines: list[dict[str, Any]],
    language: str = "en",
) -> str:
    """Build hOCR document from normalized Surya line/word dictionaries."""
    page_width = max(1, int(page_width))
    page_height = max(1, int(page_height))
    language = (language or "en").strip() or "en"

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{escape(language)}" lang="{escape(language)}">',
        "<head>",
        "<title></title>",
        '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
        '<meta name="ocr-system" content="surya-ocr via ocrmypdf-surya" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf" />',
        "</head>",
        "<body>",
        f'<div class="ocr_page" id="page_1" title="bbox 0 0 {page_width} {page_height}">',
    ]

    line_id = 1
    word_id = 1
    carea_id = 1
    par_id = 1

    for line in lines:
        line_bbox = _safe_bbox(line.get("bbox"), page_width, page_height)
        lx0, ly0, lx1, ly1 = line_bbox
        out.append(f'<div class="ocr_carea" id="carea_{carea_id}" title="bbox {lx0} {ly0} {lx1} {ly1}">')
        out.append(f'<p class="ocr_par" id="par_{par_id}" title="bbox {lx0} {ly0} {lx1} {ly1}">')
        out.append(
            f'<span class="ocr_line" id="line_{line_id}" title="bbox {lx0} {ly0} {lx1} {ly1}; baseline 0 0">'
        )

        words = line.get("words") or []
        if words:
            for idx, word in enumerate(words):
                wx0, wy0, wx1, wy1 = _safe_bbox(word.get("bbox"), page_width, page_height)
                wconf = int(max(0.0, min(1.0, float(word.get("confidence", 0.0) or 0.0))) * 100)
                text = escape(str(word.get("text") or ""))
                out.append(
                    f'<span class="ocrx_word" id="word_{word_id}" title="bbox {wx0} {wy0} {wx1} {wy1}; x_wconf {wconf}">{text}</span>'
                )
                if idx < len(words) - 1:
                    out.append(" ")
                word_id += 1
        else:
            wconf = int(max(0.0, min(1.0, float(line.get("confidence", 0.0) or 0.0))) * 100)
            text = escape(str(line.get("text") or ""))
            out.append(
                f'<span class="ocrx_word" id="word_{word_id}" title="bbox {lx0} {ly0} {lx1} {ly1}; x_wconf {wconf}">{text}</span>'
            )
            word_id += 1

        out.append("</span>")
        out.append("</p>")
        out.append("</div>")
        line_id += 1
        carea_id += 1
        par_id += 1

    out.extend(["</div>", "</body>", "</html>"])
    return "\n".join(out)
