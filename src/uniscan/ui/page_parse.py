"""Helpers for parsing page selection strings in UI."""

from __future__ import annotations

import re


def parse_page_numbers_text(raw_text: str) -> tuple[int, ...] | None:
    """Parse 1-based page numbers from text like '3,9' or '3 9;12'."""
    raw = raw_text.strip()
    if not raw:
        return None

    tokens = [part for part in re.split(r"[\s,;]+", raw) if part]
    if not tokens:
        return None

    pages: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            page = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid page value: {token}") from exc
        if page < 1:
            raise ValueError(f"Invalid page value: {page}. Page numbers must be >= 1.")
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return tuple(pages)

