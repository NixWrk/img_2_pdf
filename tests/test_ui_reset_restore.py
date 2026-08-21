from __future__ import annotations

from types import SimpleNamespace

from uniscan.ui.app import UnifiedScanApp


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


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
