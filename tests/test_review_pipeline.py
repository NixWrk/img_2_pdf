from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.session import CommittedPageProcessing
from uniscan.ui.review_pipeline import build_pipeline_cards
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
        preprocess_settings=object(),
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
