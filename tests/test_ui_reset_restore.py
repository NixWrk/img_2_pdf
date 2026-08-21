from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uniscan.ui.app import UnifiedScanApp
from uniscan.session import CROP_STATE_APPLIED


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Button:
    def __init__(self):
        self.state = None

    def configure(self, *, state):
        self.state = state


def _deskew_app(entry):
    app = object.__new__(UnifiedScanApp)
    app.deskew_method_var = _Var("Manual angle")
    app.manual_deskew_angle_var = _Var(4.2)
    app.manual_deskew_summary_var = _Var("Manual deskew: +4.2 degrees")
    app._single_selected_entry = lambda: (0, entry)
    app.preview_calls = 0
    app.update_page_preview = lambda: setattr(app, "preview_calls", app.preview_calls + 1)
    app.status = ""
    app._set_status = lambda value: setattr(app, "status", value)
    return app


def test_deskew_reset_is_draft_only_and_restore_uses_committed_recipe() -> None:
    entry = SimpleNamespace(
        revision=7,
        current_image=object(),
        committed_processing=SimpleNamespace(
            recipe=SimpleNamespace(
                to_request=lambda: SimpleNamespace(
                    deskew_method="manual", deskew_angle_degrees=-1.7
                )
            )
        ),
    )
    app = _deskew_app(entry)
    original_current = entry.current_image
    original_committed = entry.committed_processing

    app._reset_deskew_controls()
    assert app.deskew_method_var.get() == "Hybrid (recommended)"
    assert app.manual_deskew_angle_var.get() == 0.0
    assert app.preview_calls == 1

    app._restore_deskew_controls()
    assert app.deskew_method_var.get() == "Manual angle"
    assert app.manual_deskew_angle_var.get() == -1.7
    assert app.manual_deskew_summary_var.get() == "Manual deskew: -1.7 degrees"
    assert app.preview_calls == 2
    assert entry.committed_processing.recipe.to_request().deskew_angle_degrees == -1.7
    assert entry.revision == 7
    assert entry.current_image is original_current
    assert entry.committed_processing is original_committed


def test_deskew_restore_requires_single_committed_page() -> None:
    entry = SimpleNamespace(committed_processing=None)
    app = _deskew_app(entry)
    app._single_selected_entry = lambda: (None, None)

    assert app._deskew_restore_available() is False
    app._restore_deskew_controls()
    assert "requires one committed page" in app.status
    assert app.preview_calls == 0


def test_waves_restore_baseline_precedence_saved_then_legacy_then_recipe() -> None:
    saved = ((0.5, ((0.0, 0.1), (0.5, 0.12), (1.0, 0.1))),)
    legacy = ((0.0, 0.0), (0.5, 0.02), (1.0, 0.0))
    recipe_model = {"control_points": ((0.0, 0.0), (0.5, 0.03), (1.0, 0.0))}
    committed = SimpleNamespace(recipe=SimpleNamespace(dewarp_model=recipe_model))

    entry = SimpleNamespace(
        dewarp_control_curves=saved,
        dewarp_control_points=legacy,
        committed_processing=committed,
    )
    assert UnifiedScanApp._saved_dewarp_curves(entry) == saved

    entry.dewarp_control_curves = None
    restored_legacy = UnifiedScanApp._saved_dewarp_curves(entry)
    assert restored_legacy == tuple((anchor, legacy) for anchor in (0.25, 0.5, 0.75))

    entry.dewarp_control_points = None
    restored_recipe = UnifiedScanApp._saved_dewarp_curves(entry)
    assert restored_recipe == tuple(
        (anchor, recipe_model["control_points"]) for anchor in (0.25, 0.5, 0.75)
    )

    entry.committed_processing = None
    entry.dewarp_control_curves = saved
    assert UnifiedScanApp._saved_dewarp_curves(entry) is None


def _stage_app(entry):
    app = object.__new__(UnifiedScanApp)
    for name, value in {
        "preprocess_preset_var": "Whiteboard",
        "preprocess_contrast_var": 0.8,
        "preprocess_brightness_var": -5,
        "preprocess_denoise_var": 1,
        "preprocess_threshold_var": 120,
        "binarization_method_var": "Sauvola (uneven light)",
        "binarization_window_var": 41,
        "binarization_k_var": 0.5,
        "despeckle_strength_var": "Strong",
        "postprocess_var": "Grayscale",
        "shadow_method_var": "Classical",
        "page_layout_var": "A4",
        "page_margin_mm_var": 2.0,
        "page_align_x_var": "left",
        "page_align_y_var": "top",
        "export_pdf_dpi_var": 150,
        "deskew_method_var": "Manual angle",
        "dewarp_method_var": "Page model (UVDoc)",
    }.items():
        setattr(app, name, _Var(value))
    app._binarization_k_custom = True
    app._single_selected_entry = lambda: (0, entry)
    app.preview_calls = 0
    app.update_page_preview = lambda: setattr(app, "preview_calls", app.preview_calls + 1)
    app.status = ""
    app._set_status = lambda value: setattr(app, "status", value)
    return app


def _recipe_request(**overrides):
    values = {
        "shadow_method": "docshadow",
        "postprocess_name": "Binary",
        "preprocess_settings": SimpleNamespace(
            contrast=1.4,
            brightness=7,
            denoise=3,
            threshold=180,
            binarization_method="sauvola",
            binarization_window=35,
            binarization_k=0.31,
            despeckle_strength="conservative",
        ),
        "page_layout": "letter",
        "page_margin_mm": 6.5,
        "horizontal_alignment": "right",
        "vertical_alignment": "bottom",
        "page_dpi": 600,
        "deskew_method": "manual",
        "dewarp_method": "uvdoc",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cleanup_reset_is_complete_document_baseline_and_restore_is_scoped() -> None:
    request = _recipe_request()
    entry = SimpleNamespace(
        committed_processing=SimpleNamespace(recipe=SimpleNamespace(to_request=lambda: request))
    )
    app = _stage_app(entry)
    original_upstream = (app.shadow_method_var.get(), app.page_layout_var.get())

    app._reset_cleanup_controls()
    assert [
        var.get()
        for var in (
            app.preprocess_preset_var,
            app.preprocess_contrast_var,
            app.preprocess_brightness_var,
            app.preprocess_denoise_var,
            app.preprocess_threshold_var,
            app.binarization_method_var,
            app.binarization_window_var,
            app.binarization_k_var,
            app.despeckle_strength_var,
            app.postprocess_var,
        )
    ] == ["Document", 1.25, 10, 4, 170, "None", 31, 0.2, "None", "None"]
    assert app._binarization_k_custom is False

    app._restore_cleanup_controls()
    assert app.preprocess_contrast_var.get() == 1.4
    assert app.binarization_method_var.get() == "Sauvola (uneven light)"
    assert app.despeckle_strength_var.get() == "Conservative"
    assert app.postprocess_var.get() == "Binary"
    assert app.shadow_method_var.get() == original_upstream[0]
    assert app.page_layout_var.get() == original_upstream[1]
    assert app.preview_calls == 2


def test_lighting_and_layout_restore_exact_committed_controls_only() -> None:
    request = _recipe_request()
    entry = SimpleNamespace(
        committed_processing=SimpleNamespace(recipe=SimpleNamespace(to_request=lambda: request))
    )
    app = _stage_app(entry)
    entry.revision = 11
    original_committed = entry.committed_processing
    app._reset_lighting_controls()
    assert app.shadow_method_var.get() == "Automatic (validated)"
    app._restore_lighting_controls()
    assert app.shadow_method_var.get() == "Model (DocShadow)"

    app._reset_layout_controls()
    assert [
        var.get()
        for var in (
            app.page_layout_var,
            app.page_margin_mm_var,
            app.page_align_x_var,
            app.page_align_y_var,
            app.export_pdf_dpi_var,
        )
    ] == ["Keep source page", 10.0, "center", "center", 300]
    app._restore_layout_controls()
    assert [
        var.get()
        for var in (
            app.page_layout_var,
            app.page_margin_mm_var,
            app.page_align_x_var,
            app.page_align_y_var,
            app.export_pdf_dpi_var,
        )
    ] == ["Letter", 6.5, "right", "bottom", 600]
    assert app.deskew_method_var.get() == "Manual angle"
    assert app.dewarp_method_var.get() == "Page model (UVDoc)"
    assert entry.revision == 11
    assert entry.committed_processing is original_committed


def test_perspective_restore_requires_proven_coordinate_space_and_prefers_applied_contour() -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    applied = np.array([[5, 6], [150, 8], [148, 92], [7, 95]], dtype=np.float32)
    recipe = np.array([[10, 11], [140, 12], [139, 88], [12, 90]], dtype=np.float32)
    entry = SimpleNamespace(
        crop_state=CROP_STATE_APPLIED,
        detected_contour=applied,
        committed_processing=SimpleNamespace(recipe=SimpleNamespace(perspective_points=recipe)),
    )
    restored = UnifiedScanApp._perspective_restore_points(entry, image, from_current_geometry=False)
    assert np.array_equal(restored, applied)
    current = UnifiedScanApp._perspective_restore_points(entry, image, from_current_geometry=True)
    assert np.array_equal(current, recipe)
    invalid = SimpleNamespace(
        crop_state=CROP_STATE_APPLIED,
        detected_contour=np.array([[-1, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        committed_processing=None,
    )
    assert (
        UnifiedScanApp._perspective_restore_points(invalid, image, from_current_geometry=False)
        is None
    )


def test_restore_button_states_follow_selection_and_committed_state_without_mutation() -> None:
    entry = SimpleNamespace(entry_id="page-1", revision=4, committed_processing=None)
    app = object.__new__(UnifiedScanApp)
    selected = []
    app.session = SimpleNamespace(entries=[entry], can_undo_deletion=False)
    app._selected_entry_indices = lambda: list(selected)
    app._single_selected_entry = lambda: (0, entry) if selected == [0] else (None, None)
    app._deskew_restore_available = lambda: False
    app.move_pages_up_button = None
    app.move_pages_down_button = None
    app.delete_pages_button = None
    app.undo_delete_button = None
    app.deskew_restore_button = None
    app.lighting_restore_button = _Button()
    app.cleanup_restore_button = _Button()
    app.layout_restore_button = _Button()
    app.page_context_menu = None
    app.apply_split_button = None
    original_revision = entry.revision

    app._update_page_action_states()
    assert [
        app.lighting_restore_button.state,
        app.cleanup_restore_button.state,
        app.layout_restore_button.state,
    ] == ["disabled", "disabled", "disabled"]

    entry.committed_processing = SimpleNamespace(recipe=object())
    selected.append(0)
    app._update_page_action_states()
    assert [
        app.lighting_restore_button.state,
        app.cleanup_restore_button.state,
        app.layout_restore_button.state,
    ] == ["normal", "normal", "normal"]

    selected.clear()
    app._update_page_action_states()
    assert [
        app.lighting_restore_button.state,
        app.cleanup_restore_button.state,
        app.layout_restore_button.state,
    ] == ["disabled", "disabled", "disabled"]
    assert entry.revision == original_revision
