import cv2
import numpy as np
import pytest

from uniscan.core.orientation import (
    ORIENTATION_METHOD_AUTO,
    ORIENTATION_METHOD_NONE,
    estimate_page_orientation,
    orient_document,
)


def _text_page() -> np.ndarray:
    image = np.full((720, 520, 3), 255, dtype=np.uint8)
    lines = (
        "Document page",
        "quickly aligns",
        "baseline glyphs",
        "properly oriented",
        "local processing",
        "keeps text safe",
        "without any OCR",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (35, 100 + index * 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return image


@pytest.mark.parametrize(
    ("rotation", "expected_correction"),
    [
        (None, 0),
        (cv2.ROTATE_90_CLOCKWISE, 270),
        (cv2.ROTATE_180, 180),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, 90),
    ],
)
def test_estimate_page_orientation_for_text_layout(rotation, expected_correction) -> None:
    image = _text_page()
    if rotation is not None:
        image = cv2.rotate(image, rotation)

    diagnostics = estimate_page_orientation(image)

    assert diagnostics.angle_degrees == expected_correction
    assert diagnostics.line_count >= 5
    assert diagnostics.confidence > 0.2


def test_orient_document_restores_sideways_page() -> None:
    upright = _text_page()
    sideways = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)

    corrected, diagnostics = orient_document(sideways, method=ORIENTATION_METHOD_AUTO)

    assert diagnostics.applied is True
    assert diagnostics.angle_degrees == 270
    assert corrected.shape == upright.shape
    assert np.array_equal(corrected, upright)


@pytest.mark.parametrize("angle", (90, 180, 270))
def test_orient_document_supports_explicit_right_angle_rotation(angle) -> None:
    source = _text_page()

    rotated, diagnostics = orient_document(source, method=str(angle))

    assert diagnostics.applied is True
    assert diagnostics.angle_degrees == angle
    assert diagnostics.confidence == 1.0
    assert diagnostics.reason == "forced"
    assert np.array_equal(
        rotated,
        cv2.rotate(
            source,
            {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[angle],
        ),
    )


def test_orientation_is_noop_for_ambiguous_and_disabled_images() -> None:
    ambiguous = np.full((300, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(ambiguous, (40, 70), (180, 230), (0, 0, 0), 3)

    unchanged, automatic = orient_document(ambiguous, method=ORIENTATION_METHOD_AUTO)
    disabled, disabled_diagnostics = orient_document(ambiguous, method=ORIENTATION_METHOD_NONE)

    assert automatic.applied is False
    assert automatic.reason in {"insufficient_layout", "ambiguous_text_axis"}
    assert np.array_equal(unchanged, ambiguous)
    assert disabled is ambiguous
    assert disabled_diagnostics.reason == "disabled"


def test_orientation_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported orientation method"):
        orient_document(_text_page(), method="missing")
