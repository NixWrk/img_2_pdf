from __future__ import annotations

import numpy as np

from uniscan.core.boundary_review import (
    BOUNDARY_NOT_DETECTED_REASON,
    LARGE_DARK_BORDER_REASON,
    assess_boundary_review,
)


def _clean_page() -> np.ndarray:
    image = np.full((700, 500, 3), 238, dtype=np.uint8)
    image[80:620:45, 70:430] = 90
    return image


def _page8_like_bad_boundary() -> np.ndarray:
    image = _clean_page()
    image[:150, :] = 24
    image[:, :70] = 18
    return image


def test_clean_page_does_not_need_boundary_review() -> None:
    diagnostics = assess_boundary_review(
        _clean_page(),
        detection_enabled=True,
        detected=True,
    )

    assert diagnostics.needs_review is False
    assert diagnostics.reasons == ()
    assert diagnostics.dark_border_fraction < 0.12


def test_page8_like_dark_border_is_flagged_for_review() -> None:
    diagnostics = assess_boundary_review(
        _page8_like_bad_boundary(),
        detection_enabled=True,
        detected=True,
    )

    assert diagnostics.needs_review is True
    assert diagnostics.reasons == (LARGE_DARK_BORDER_REASON,)
    assert diagnostics.dark_border_fraction >= 0.12


def test_detector_miss_is_flagged_even_without_a_dark_border() -> None:
    diagnostics = assess_boundary_review(
        _clean_page(),
        detection_enabled=True,
        detected=False,
    )

    assert diagnostics.needs_review is True
    assert BOUNDARY_NOT_DETECTED_REASON in diagnostics.reasons


def test_pending_proposal_is_not_double_flagged() -> None:
    diagnostics = assess_boundary_review(
        _page8_like_bad_boundary(),
        detection_enabled=True,
        detected=True,
        proposal_only=True,
    )

    assert diagnostics.needs_review is False
    assert diagnostics.dark_border_fraction == 0.0
