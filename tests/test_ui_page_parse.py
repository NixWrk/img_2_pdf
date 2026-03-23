from __future__ import annotations

import pytest

from uniscan.ui.page_parse import parse_page_numbers_text


def test_parse_page_numbers_text_accepts_common_separators() -> None:
    assert parse_page_numbers_text("3,9") == (3, 9)
    assert parse_page_numbers_text("3 9;12") == (3, 9, 12)
    assert parse_page_numbers_text("  5 \n 7  ") == (5, 7)


def test_parse_page_numbers_text_deduplicates_preserving_order() -> None:
    assert parse_page_numbers_text("3,9,3,12,9") == (3, 9, 12)


def test_parse_page_numbers_text_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        parse_page_numbers_text("0")
    with pytest.raises(ValueError, match="Invalid page value: abc"):
        parse_page_numbers_text("3,abc")


def test_parse_page_numbers_text_empty_returns_none() -> None:
    assert parse_page_numbers_text("") is None
    assert parse_page_numbers_text("   ") is None

