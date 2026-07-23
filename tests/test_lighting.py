from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.core import lighting
from uniscan.core.lighting import (
    SHADOW_METHOD_AUTO,
    SHADOW_METHOD_CLASSICAL,
    SHADOW_METHOD_DOCSHADOW,
    SHADOW_METHOD_NONE,
    remove_document_shadows,
)


def _page(width: int = 700, height: int = 900) -> np.ndarray:
    page = np.full((height, width, 3), 242, np.uint8)
    for y in range(60, height - 60, 40):
        cv2.rectangle(page, (60, y), (width - 60, y + 14), (28, 28, 30), -1)
    return page


def _shadowed_page() -> np.ndarray:
    page = _page()
    ramp = np.linspace(0.35, 1.0, page.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def test_none_is_an_identity_stage() -> None:
    page = _shadowed_page()

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_NONE)

    assert diagnostics.applied is False
    assert diagnostics.reason == "disabled"
    np.testing.assert_array_equal(result, page)


def test_unsupported_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported shadow removal method"):
        remove_document_shadows(_page(), method="magic")


def test_classical_method_reports_its_own_measurements() -> None:
    page = _shadowed_page()

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_CLASSICAL)

    assert diagnostics.applied is True
    assert diagnostics.selected_method == SHADOW_METHOD_CLASSICAL
    assert diagnostics.unevenness_after < diagnostics.unevenness_before
    assert result.shape == page.shape


def test_auto_leaves_an_evenly_lit_page_untouched(monkeypatch) -> None:
    page = _page()
    calls: list[bool] = []
    monkeypatch.setattr(
        lighting,
        "_docshadow_candidate",
        lambda _image: (calls.append(True), (None, "unused"))[1],
    )

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    # No shadow means no model run at all, not merely a rejected candidate.
    assert diagnostics.applied is False
    assert diagnostics.reason == "no_shadow_detected"
    assert calls == []
    np.testing.assert_array_equal(result, page)


def test_auto_prefers_the_model_and_reports_it(monkeypatch) -> None:
    page = _shadowed_page()
    corrected = _page()
    monkeypatch.setattr(lighting, "_docshadow_candidate", lambda _image: (corrected, None))

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    assert diagnostics.applied is True
    assert diagnostics.selected_method == SHADOW_METHOD_DOCSHADOW
    assert diagnostics.shadow_after < diagnostics.shadow_before
    assert diagnostics.duration_ms >= 0.0
    np.testing.assert_array_equal(result, corrected)


def test_auto_falls_back_to_the_classical_path_when_the_model_is_missing(monkeypatch) -> None:
    page = _shadowed_page()
    monkeypatch.setattr(
        lighting, "_docshadow_candidate", lambda _image: (None, "docshadow_model_unavailable")
    )

    _result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    assert diagnostics.applied is True
    assert diagnostics.selected_method == SHADOW_METHOD_CLASSICAL
    assert diagnostics.reason == "docshadow_model_unavailable"


def test_auto_rejects_a_candidate_that_washes_the_ink_out(monkeypatch) -> None:
    page = _shadowed_page()
    washed = np.full_like(page, 250)  # evenly lit, but the text is gone
    monkeypatch.setattr(lighting, "_docshadow_candidate", lambda _image: (washed, None))
    monkeypatch.setattr(lighting, "correct_illumination", lambda image: washed)

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    assert diagnostics.applied is False
    assert diagnostics.selected_method == SHADOW_METHOD_NONE
    assert "contrast_lost" in diagnostics.reason
    np.testing.assert_array_equal(result, page)


def test_auto_rejects_a_candidate_that_does_not_even_the_page(monkeypatch) -> None:
    page = _shadowed_page()
    monkeypatch.setattr(lighting, "_docshadow_candidate", lambda image: (image.copy(), None))
    monkeypatch.setattr(lighting, "correct_illumination", lambda image: image.copy())

    _result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    assert diagnostics.applied is False
    assert "docshadow_rejected:lighting_not_improved" in diagnostics.reason
    assert "classical_rejected:lighting_not_improved" in diagnostics.reason


def test_explicit_docshadow_reports_a_missing_model_without_raising(monkeypatch) -> None:
    page = _shadowed_page()
    monkeypatch.setattr(
        lighting, "_docshadow_candidate", lambda _image: (None, "docshadow_model_unavailable")
    )

    result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_DOCSHADOW)

    assert diagnostics.applied is False
    assert diagnostics.reason == "docshadow_model_unavailable"
    np.testing.assert_array_equal(result, page)


def test_model_failure_is_contained(monkeypatch) -> None:
    page = _shadowed_page()

    def explode(_image):
        raise RuntimeError("runtime blew up")

    monkeypatch.setattr("uniscan.core.docshadow.is_available", lambda: True)
    monkeypatch.setattr("uniscan.core.docshadow.remove_shadows", explode)

    _result, diagnostics = remove_document_shadows(page, method=SHADOW_METHOD_AUTO)

    # The stage degrades to the classical path instead of failing the page.
    assert "docshadow_failed:RuntimeError" in diagnostics.reason
    assert diagnostics.selected_method == SHADOW_METHOD_CLASSICAL
