import cv2
import numpy as np
import pytest

from uniscan.core.layout import detect_content_box, layout_document_page


def _page() -> np.ndarray:
    image = np.full((600, 400, 3), 255, dtype=np.uint8)
    for y in range(180, 430, 45):
        cv2.putText(
            image,
            "document line",
            (75, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def test_detect_content_box_excludes_page_margins() -> None:
    box, confidence, reason = detect_content_box(_page())

    assert reason is None
    assert 50 < box.x < 100
    assert 140 < box.y < 190
    assert box.width < 300
    assert box.height < 320
    assert confidence > 0.5


def test_layout_places_content_on_a4_with_alignment_and_margin() -> None:
    result, diagnostics = layout_document_page(
        _page(),
        method="a4",
        dpi=100,
        margin_mm=10,
        horizontal_alignment="left",
        vertical_alignment="top",
    )

    assert result.shape[:2] == (1169, 827)
    assert diagnostics.applied is True
    assert diagnostics.target_width == 827
    assert diagnostics.target_height == 1169
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ink_y, ink_x = np.where(gray < 200)
    assert ink_x.min() >= 38
    assert ink_y.min() >= 38


def test_layout_none_is_identity_and_blank_page_is_safe() -> None:
    blank = np.full((200, 150), 255, dtype=np.uint8)

    unchanged, disabled = layout_document_page(blank, method="none")
    laid_out, diagnostics = layout_document_page(blank, method="letter", dpi=72)

    assert unchanged is blank
    assert disabled.reason == "disabled"
    assert diagnostics.reason == "no_content"
    assert laid_out.shape[:2] == (792, 612)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "missing"}, "Unsupported page layout"),
        ({"horizontal_alignment": "middle"}, "horizontal alignment"),
        ({"vertical_alignment": "middle"}, "vertical alignment"),
        ({"dpi": 20}, "DPI"),
        ({"margin_mm": 90}, "margin"),
    ],
)
def test_layout_validates_options(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        layout_document_page(_page(), **kwargs)
