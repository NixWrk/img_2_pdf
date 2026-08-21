from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from uniscan.core.dewarp import DewarpModel
from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.core.preprocess import PreprocessSettings
from uniscan.session import CommittedPageProcessing
from uniscan.ui.review_pipeline import build_pipeline_cards, _pending_changed_stage
from uniscan.ui.stage_state import PipelineStage, StageMode, StageStatus


def _entry(*, committed=None, crop_state="none", contour=None, revision=4):
    return SimpleNamespace(
        revision=revision,
        crop_state=crop_state,
        detected_contour=contour,
        detected_backend=None,
        review_reasons=(),
        committed_processing=committed,
    )


def _recipe():
    return SimpleNamespace(
        dewarp_method="auto",
        deskew_method="hybrid",
        shadow_method="auto",
        postprocess_name="None",
        preprocess_settings=PreprocessSettings(contrast=1.25, brightness=10, denoise=4),
        page_layout="a4",
    )


def test_adapter_returns_ordered_cards_with_human_reason_copy() -> None:
    committed = SimpleNamespace(
        recipe=_recipe(),
        diagnostics={
            "dewarp": {"applied": False, "method": "auto", "reason": "curvature_below_threshold"},
            "deskew_method": "hybrid",
            "deskew_angle_degrees": 0.0,
            "deskew_reason": "angle_below_threshold",
            "deskew_confidence": 0.8,
            "deskew_line_count": 12,
            "shadow": {"applied": False, "method": "auto", "reason": "no_shadow_detected"},
            "despeckle": {"applied": True, "strength": "conservative", "removed_components": 3},
            "layout": {"applied": True, "method": "a4", "reason": None},
        },
    )
    cards = build_pipeline_cards(
        _entry(committed=committed, crop_state="applied", contour=object()),
    )

    assert [card.stage for card in cards] == list(PipelineStage)
    assert cards[0].state.status is StageStatus.APPLIED
    assert cards[1].state.status is StageStatus.NOT_NEEDED
    assert cards[1].reason.summary == "No wave correction was needed."
    assert cards[3].reason.summary == "Lighting is already even."
    assert cards[4].state.status is StageStatus.APPLIED
    assert cards[4].reason.summary == "Cleanup applied."
    assert cards[-1].state.status is StageStatus.APPLIED
    assert cards[-1].reason.summary == "Committed result is ready for export."
    assert cards[1].state.mode is StageMode.AUTO
    assert cards[1].controls == ("Auto", "Off", "Edit")


def test_pending_candidate_is_visible_and_does_not_claim_durable_commit() -> None:
    entry = _entry(crop_state="proposed", contour=object(), revision=7)
    cards = build_pipeline_cards(
        entry,
        pending_request=_recipe(),
        pending_diagnostics={
            "dewarp": {
                "applied": True,
                "method": "auto",
                "reason": "user_adjusted_model",
                "curvature_rms_px": 1.2,
            }
        },
    )

    result = cards[-1]
    waves = cards[1]
    assert result.state.status is StageStatus.EDITED
    assert result.state.candidate_revision == 8
    assert result.state.committed_revision is None
    assert result.reason.summary == "Result has a manual edit."
    assert waves.state.status is StageStatus.EDITED
    assert waves.reason.summary == "Your manual wave adjustment is active."
    assert cards[0].state.status is StageStatus.APPLIED


def test_persisted_applied_crop_without_recipe_is_export_committed() -> None:
    cards = build_pipeline_cards(_entry(crop_state="applied", contour=object()))

    perspective = cards[0]
    assert perspective.state.status is StageStatus.EDITED
    assert perspective.state.candidate_revision == 4
    assert perspective.state.committed_revision == 4
    assert cards[-1].state.status is StageStatus.APPLIED
    assert cards[-1].state.committed_revision == 4
    assert cards[-1].reason.summary == "Committed result is ready for export."


def test_manual_applied_crop_is_presented_as_edited() -> None:
    entry = _entry(
        committed=SimpleNamespace(recipe=_recipe(), diagnostics={}), crop_state="applied"
    )
    entry.detected_backend = "manual"

    perspective = build_pipeline_cards(entry)[0]

    assert perspective.state.status is StageStatus.EDITED


def test_off_modes_are_explicit_and_reasons_are_not_raw_codes() -> None:
    recipe = SimpleNamespace(
        dewarp_method="none",
        deskew_method="none",
        shadow_method="none",
        postprocess_name="None",
        preprocess_settings=None,
        page_layout="none",
    )
    committed = SimpleNamespace(recipe=recipe, diagnostics={})
    cards = build_pipeline_cards(_entry(committed=committed))

    for stage in (
        PipelineStage.WAVES,
        PipelineStage.DESKEW,
        PipelineStage.LIGHTING,
        PipelineStage.LAYOUT,
    ):
        card = next(item for item in cards if item.stage is stage)
        assert card.state.mode is StageMode.OFF
        assert card.state.status is StageStatus.NOT_NEEDED
        assert card.reason.summary
        assert "disabled" not in card.reason.summary.lower()


def test_model_failure_is_error_and_insufficient_evidence_is_rejected() -> None:
    committed = SimpleNamespace(
        recipe=_recipe(),
        diagnostics={
            "dewarp": {
                "applied": False,
                "method": "auto",
                "reason": "uvdoc_model_unavailable",
            },
            "deskew_method": "hybrid",
            "deskew_angle_degrees": 0.0,
            "deskew_reason": "no_lines",
        },
    )

    cards = build_pipeline_cards(_entry(committed=committed))

    assert cards[1].state.status is StageStatus.ERROR
    assert cards[2].state.status is StageStatus.REJECTED


def test_pending_request_without_diagnostics_is_running_not_applied() -> None:
    committed = SimpleNamespace(recipe=_recipe(), diagnostics={})

    cards = build_pipeline_cards(
        _entry(committed=committed),
        pending_request=_recipe(),
    )

    assert cards[1].state.status is StageStatus.RUNNING
    assert cards[1].reason.summary == "Checking waves…"
    assert cards[-1].state.status is StageStatus.RUNNING
    assert cards[-1].reason.summary == "Checking result…"


def test_failed_candidate_preserves_committed_evidence_and_marks_result_error() -> None:
    committed = SimpleNamespace(
        recipe=_recipe(),
        diagnostics={
            "dewarp": {
                "applied": False,
                "method": "auto",
                "reason": "curvature_below_threshold",
            }
        },
    )

    cards = build_pipeline_cards(
        _entry(committed=committed),
        pending_request=_recipe(),
        pending_error="worker failed",
    )

    assert cards[1].state.status is StageStatus.NOT_NEEDED
    assert cards[1].reason.summary == "No wave correction was needed."
    assert cards[-1].state.status is StageStatus.ERROR
    assert cards[-1].state.committed_revision == 4
    assert cards[-1].state.candidate_revision == 5
    assert cards[-1].reason.summary == "Automatic processing could not finish."


def test_adapter_accepts_real_committed_recipe_and_diagnostics() -> None:
    image = np.full((24, 32, 3), 180, dtype=np.uint8)
    request = PageProcessingRequest(
        dewarp_method="none",
        deskew_method="none",
        shadow_method="none",
        preprocess_settings=None,
        page_layout="none",
    )
    result = process_document_page(image, request)
    committed = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )

    cards = build_pipeline_cards(
        _entry(committed=committed, crop_state="applied", contour=object()),
    )

    assert cards[1].state.mode is StageMode.OFF
    assert cards[2].state.mode is StageMode.OFF
    assert cards[3].state.mode is StageMode.OFF
    assert cards[-1].reason.summary == "Committed result is ready for export."


def test_pending_invalidation_marks_only_downstream_until_full_recompute() -> None:
    committed = SimpleNamespace(recipe=_recipe(), diagnostics={})
    pending = _recipe()
    pending.deskew_method = "manual"
    cards = build_pipeline_cards(
        _entry(committed=committed),
        pending_request=pending,
    )
    assert cards[1].state.status is StageStatus.IDLE
    assert cards[2].state.status is StageStatus.RUNNING
    assert cards[3].state.status is StageStatus.STALE
    assert cards[3].reason.summary == "This stage needs to be rerun after an earlier edit."
    assert cards[4].state.status is StageStatus.STALE
    assert cards[5].state.status is StageStatus.STALE
    assert cards[-1].state.status is StageStatus.STALE


def test_pending_changed_stage_mapping_ignores_preview_dpi_and_covers_geometry_inputs() -> None:
    committed = SimpleNamespace(
        orientation_method="none",
        perspective_points=None,
        dewarp_method="auto",
        dewarp_model=None,
        dewarp_already_applied=False,
        auto_dewarp_uvdoc=False,
        auto_dewarp_uvdoc_grid=True,
        deskew_method="hybrid",
        deskew_angle_degrees=None,
        shadow_method="auto",
        postprocess_name="None",
        preprocess_settings=SimpleNamespace(contrast=1.0),
        page_layout="a4",
        page_margin_mm=10.0,
        horizontal_alignment="center",
        vertical_alignment="center",
        page_dpi=300,
    )
    preview_values = vars(committed).copy()
    preview_values["page_dpi"] = 100
    preview = SimpleNamespace(**preview_values)
    assert _pending_changed_stage(committed, preview, perspective_candidate=False) is None
    committed.dewarp_model = {
        "method": "auto",
        "control_points": ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)),
        "source": "automatic",
        "line_count": 0,
        "control_curves": None,
    }
    model_values = vars(preview).copy()
    model_values["dewarp_model"] = DewarpModel(
        method="auto",
        control_points=((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)),
    )
    model_preview = SimpleNamespace(**model_values)
    assert _pending_changed_stage(committed, model_preview, perspective_candidate=False) is None

    orientation_values = vars(preview).copy()
    orientation_values["orientation_method"] = "90"
    orientation = SimpleNamespace(**orientation_values)
    assert (
        _pending_changed_stage(committed, orientation, perspective_candidate=False)
        is PipelineStage.PERSPECTIVE
    )
    uvdoc_values = vars(preview).copy()
    uvdoc_values["auto_dewarp_uvdoc"] = True
    uvdoc = SimpleNamespace(**uvdoc_values)
    assert (
        _pending_changed_stage(committed, uvdoc, perspective_candidate=False) is PipelineStage.WAVES
    )
    assert (
        _pending_changed_stage(committed, preview, perspective_candidate=True)
        is PipelineStage.PERSPECTIVE
    )


def test_preprocess_semantics_map_equal_values_to_none_and_edits_to_cleanup() -> None:
    committed = _recipe()
    assert _pending_changed_stage(committed, _recipe(), perspective_candidate=False) is None
    for changed in (
        replace(committed.preprocess_settings, contrast=1.4),
        replace(committed.preprocess_settings, binarization_method="sauvola"),
        replace(committed.preprocess_settings, despeckle_strength="normal"),
    ):
        pending = _recipe()
        pending.preprocess_settings = changed
        assert (
            _pending_changed_stage(committed, pending, perspective_candidate=False)
            is PipelineStage.CLEANUP
        )
