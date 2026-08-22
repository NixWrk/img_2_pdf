from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.ui.app import UnifiedScanApp
from uniscan.ui.app import (
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    _clamp_preview_zoom,
    _preview_crop_bounds,
    _preview_zoom_pan,
)


def test_preview_zoom_is_clamped_to_supported_range() -> None:
    assert _clamp_preview_zoom(0.2) == PREVIEW_ZOOM_MIN
    assert _clamp_preview_zoom(99.0) == PREVIEW_ZOOM_MAX
    assert _clamp_preview_zoom(2.5) == 2.5


def test_preview_crop_uses_fit_at_one_and_centered_source_at_two() -> None:
    shape = (800, 1200, 3)
    viewport = (600, 400)
    assert _preview_crop_bounds(shape, viewport, 1.0, (0.5, 0.5)) == (0, 0, 1200, 800)
    assert _preview_crop_bounds(shape, viewport, 2.0, (0.5, 0.5)) == (
        300,
        200,
        900,
        600,
    )
    assert _preview_crop_bounds(shape, viewport, 2.0, (0.0, 0.0)) == (0, 0, 600, 400)
    assert _preview_crop_bounds(shape, viewport, 2.0, (1.0, 1.0)) == (
        600,
        400,
        1200,
        800,
    )


def test_preview_zoom_keeps_cursor_source_point_stable() -> None:
    shape = (800, 1200, 3)
    viewport = (600, 400)
    cursor = (450.0, 100.0)
    zoom, pan = _preview_zoom_pan(
        shape,
        viewport,
        1.0,
        2.0,
        (0.5, 0.5),
        cursor,
    )
    assert zoom == 2.0
    left, top, right, bottom = _preview_crop_bounds(shape, viewport, zoom, pan)
    anchored_x = left + cursor[0] / viewport[0] * (right - left)
    anchored_y = top + cursor[1] / viewport[1] * (bottom - top)
    assert anchored_x == pytest.approx(900.0, abs=1.0)
    assert anchored_y == pytest.approx(200.0, abs=1.0)


def test_preview_crop_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="positive"):
        _preview_crop_bounds((0, 1200, 3), (600, 400), 1.0, (0.5, 0.5))


def test_space_hold_toggles_cached_original_without_reprocessing() -> None:
    renders: list[bool] = []
    app = SimpleNamespace(
        preview_mode_var=SimpleNamespace(get=lambda: "Preview"),
        page_preview_before_image=np.zeros((4, 4, 3), dtype=np.uint8),
        page_preview_after_image=np.ones((4, 4, 3), dtype=np.uint8),
        preview_hold_original=False,
        review_preview_generation=17,
        preview_zoom=2.0,
        preview_pan=(0.25, 0.75),
        focus_get=lambda: None,
        _preview_space_shortcut_is_allowed=lambda: UnifiedScanApp._preview_space_shortcut_is_allowed(app),
        _release_preview_hold_original=lambda: UnifiedScanApp._release_preview_hold_original(app),
        _render_cached_review_previews=lambda **kwargs: renders.append(kwargs["force"]),
    )
    event = SimpleNamespace(keysym="space")

    assert UnifiedScanApp._on_preview_hold_key_press(app, event) == "break"
    assert app.preview_hold_original is True
    assert renders == [True]
    assert UnifiedScanApp._on_preview_hold_key_press(app, event) == "break"
    assert renders == [True]

    assert UnifiedScanApp._on_preview_hold_key_release(app, event) == "break"
    assert app.preview_hold_original is False
    assert renders == [True, True]
    assert app.preview_mode_var.get() == "Preview"
    assert app.review_preview_generation == 17
    assert app.preview_zoom == 2.0
    assert app.preview_pan == (0.25, 0.75)


def test_space_hold_does_not_steal_text_input_focus() -> None:
    renders: list[bool] = []
    app = SimpleNamespace(
        preview_mode_var=SimpleNamespace(get=lambda: "Preview"),
        page_preview_before_image=np.zeros((4, 4, 3), dtype=np.uint8),
        page_preview_after_image=np.ones((4, 4, 3), dtype=np.uint8),
        preview_hold_original=False,
        focus_get=lambda: SimpleNamespace(winfo_class=lambda: "Entry"),
        _preview_space_shortcut_is_allowed=lambda: UnifiedScanApp._preview_space_shortcut_is_allowed(app),
        _render_cached_review_previews=lambda **kwargs: renders.append(kwargs["force"]),
    )

    assert UnifiedScanApp._on_preview_hold_key_press(app, SimpleNamespace(keysym="space")) is None
    assert app.preview_hold_original is False
    assert renders == []


def test_space_hold_waits_until_preview_is_cached() -> None:
    app = SimpleNamespace(
        preview_mode_var=SimpleNamespace(get=lambda: "Preview"),
        page_preview_before_image=np.zeros((4, 4, 3), dtype=np.uint8),
        page_preview_after_image=None,
        preview_hold_original=False,
        focus_get=lambda: None,
        _preview_space_shortcut_is_allowed=lambda: True,
    )

    assert UnifiedScanApp._on_preview_hold_key_press(
        app,
        SimpleNamespace(keysym="space"),
    ) is None
    assert app.preview_hold_original is False
