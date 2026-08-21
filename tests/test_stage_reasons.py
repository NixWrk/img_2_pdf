from __future__ import annotations

import pytest

from uniscan.ui.stage_reasons import StageReason, describe_stage_reason
from uniscan.ui.stage_state import PipelineStage


def test_known_reason_uses_short_summary_and_details() -> None:
    result = describe_stage_reason(PipelineStage.WAVES, "curvature_below_threshold")

    assert isinstance(result, StageReason)
    assert result.known is True
    assert result.summary == "No wave correction was needed."
    assert "safe correction threshold" in result.details


def test_composite_reason_keeps_summary_short_and_explains_all_known_parts() -> None:
    result = describe_stage_reason(
        PipelineStage.LIGHTING,
        "docshadow_model_unavailable;classical_rejected:lighting_not_improved",
    )

    assert result.known is True
    assert result.summary == "The lighting model is unavailable."
    assert "optional DocShadow model" in result.details
    assert "did not improve" in result.details


def test_real_hybrid_deskew_fallback_is_explained() -> None:
    result = describe_stage_reason(
        PipelineStage.DESKEW,
        "hough_no_lines;min_area_selected",
    )

    assert result.known is True
    assert result.summary == "Deskew used a safe fallback angle."
    assert "minimum-area estimate" in result.details


def test_prefix_reason_and_unknown_suffix_are_reported_without_technical_main_copy() -> None:
    result = describe_stage_reason(
        PipelineStage.WAVES,
        "textline_fallback:insufficient_text_lines;future_reason",
    )

    assert result.known is False
    assert result.summary == "Wave correction used a safe fallback."
    assert "future_reason" in result.details
    assert "future_reason" not in result.summary


def test_stage_specific_prefix_is_not_mapped_to_the_wrong_stage() -> None:
    result = describe_stage_reason(PipelineStage.CLEANUP, "docshadow_failed:RuntimeError")

    assert result.known is False
    assert result.summary == "Cleanup was kept unchanged."


def test_unknown_reason_uses_stage_fallback() -> None:
    result = describe_stage_reason(PipelineStage.DESKEW, "new_backend:unhandled")

    assert result.known is False
    assert result.summary == "Deskew was kept unchanged."
    assert "new_backend:unhandled" in result.details


@pytest.mark.parametrize("stage", list(PipelineStage))
def test_missing_reason_has_safe_fallback_for_every_stage(stage: PipelineStage) -> None:
    result = describe_stage_reason(stage, None)

    assert result.known is False
    assert result.summary
    assert result.details


def test_metrics_are_details_only_and_have_readable_labels() -> None:
    result = describe_stage_reason(
        PipelineStage.DESKEW,
        "hough_selected",
        {"angle_degrees": 1.25, "confidence": 0.932},
    )

    assert result.summary == "Deskew found a usable text angle."
    assert "Angle: 1.25°" in result.details
    assert "Confidence: 0.932" in result.details
    assert result.summary.count("1.25") == 0


def test_stage_and_metrics_are_validated() -> None:
    with pytest.raises(TypeError, match="PipelineStage"):
        describe_stage_reason("waves", "disabled")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metrics"):
        describe_stage_reason(PipelineStage.WAVES, "disabled", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        describe_stage_reason(PipelineStage.WAVES, "disabled", {"confidence": float("inf")})


@pytest.mark.parametrize(
    "stage,code",
    [
        (PipelineStage.PERSPECTIVE, "boundary_not_detected"),
        (PipelineStage.WAVES, "uvdoc_no_result"),
        (PipelineStage.DESKEW, "insufficient_lines"),
        (PipelineStage.LIGHTING, "no_shadow_detected"),
        (PipelineStage.CLEANUP, "no_isolated_specks"),
        (PipelineStage.LAYOUT, "no_content"),
    ],
)
def test_each_pipeline_stage_has_real_diagnostic_copy(stage: PipelineStage, code: str) -> None:
    result = describe_stage_reason(stage, code)

    assert result.known is True
    assert result.summary
    assert result.details
