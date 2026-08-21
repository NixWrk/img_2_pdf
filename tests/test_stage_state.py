from __future__ import annotations

import pytest

from uniscan.ui.stage_state import (
    PipelineStage,
    StageMode,
    StageState,
    StageStatus,
    downstream_stages,
    invalidate_downstream,
)


def test_default_state_is_explicit_and_immutable() -> None:
    state = StageState()
    assert state.mode is StageMode.AUTO
    assert state.status is StageStatus.IDLE
    assert state.input_revision == 0
    assert state.candidate_revision is None
    assert state.committed_revision is None
    assert dict(state.metrics) == {}
    with pytest.raises(TypeError):
        state.metrics["confidence"] = 0.9  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        state.status = StageStatus.RUNNING  # type: ignore[misc]


def test_preview_apply_and_reject_keep_commit_boundary_explicit() -> None:
    state = StageState(input_revision=3)
    running = state.transition(StageStatus.RUNNING, candidate_revision=4)
    applied_preview = running.transition(StageStatus.APPLIED, candidate_revision=4)
    edited = applied_preview.transition(
        StageStatus.EDITED, candidate_revision=5, metrics={"confidence": 0.92}
    )
    applied = edited.transition(
        StageStatus.APPLIED, candidate_revision=5, committed_revision=5
    )
    assert edited.committed_revision is None
    assert applied.status is StageStatus.APPLIED
    assert applied.candidate_revision == applied.committed_revision == 5
    rejected = applied.transition(StageStatus.RUNNING, candidate_revision=6).transition(
        StageStatus.REJECTED,
        candidate_revision=6,
        reason_code="quality_below_threshold",
    )
    assert rejected.committed_revision == 5
    assert rejected.candidate_revision == 6


def test_status_reason_and_revision_invariants_are_strict() -> None:
    with pytest.raises(TypeError):
        StageState(mode="auto")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StageState(status="idle")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason_code"):
        StageState(status=StageStatus.ERROR)
    with pytest.raises(ValueError, match="non-negative"):
        StageState(input_revision=-1)
    with pytest.raises(ValueError, match="input_revision"):
        StageState(input_revision=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="newer"):
        StageState(candidate_revision=2, committed_revision=3)
    with pytest.raises(ValueError, match="applied"):
        StageState(status=StageStatus.APPLIED)
    with pytest.raises(ValueError, match="edited"):
        StageState(status=StageStatus.EDITED)
    with pytest.raises(ValueError, match="finite"):
        StageState(metrics={"confidence": float("nan")})
    with pytest.raises(ValueError, match="invalid stage transition"):
        StageState().transition(StageStatus.APPLIED)


def test_mode_can_be_switched_without_losing_result_state() -> None:
    state = StageState(
        mode=StageMode.AUTO,
        status=StageStatus.APPLIED,
        candidate_revision=2,
        committed_revision=2,
    )
    off = state.with_mode(StageMode.OFF)
    assert off.mode is StageMode.OFF
    assert off.status is StageStatus.APPLIED
    assert off.committed_revision == 2


def test_applied_candidate_can_remain_uncommitted() -> None:
    state = StageState(
        status=StageStatus.APPLIED,
        candidate_revision=2,
        committed_revision=None,
    )
    assert state.candidate_revision == 2
    assert state.committed_revision is None


def test_downstream_stages_are_transitive_and_ordered() -> None:
    assert downstream_stages(PipelineStage.PERSPECTIVE) == (
        PipelineStage.WAVES,
        PipelineStage.DESKEW,
        PipelineStage.LIGHTING,
        PipelineStage.CLEANUP,
        PipelineStage.LAYOUT,
        PipelineStage.RESULT,
    )
    assert downstream_stages(PipelineStage.WAVES, include_self=True)[0] is PipelineStage.WAVES
    assert downstream_stages(PipelineStage.RESULT) == ()


def test_invalidate_downstream_marks_materialised_results_stale_only() -> None:
    states = {
        PipelineStage.PERSPECTIVE: StageState(
            status=StageStatus.APPLIED, candidate_revision=1, committed_revision=1
        ),
        PipelineStage.WAVES: StageState(
            status=StageStatus.APPLIED, candidate_revision=1, committed_revision=1
        ),
        PipelineStage.DESKEW: StageState(),
        PipelineStage.RESULT: StageState(
            status=StageStatus.ERROR, reason_code="model_failed", candidate_revision=1
        ),
    }
    invalidated = invalidate_downstream(
        states, PipelineStage.PERSPECTIVE, input_revision=2
    )
    assert invalidated[PipelineStage.PERSPECTIVE] == states[PipelineStage.PERSPECTIVE]
    assert invalidated[PipelineStage.WAVES].status is StageStatus.STALE
    assert invalidated[PipelineStage.WAVES].reason_code == "upstream_changed"
    assert invalidated[PipelineStage.WAVES].input_revision == 2
    assert invalidated[PipelineStage.DESKEW].status is StageStatus.IDLE
    assert invalidated[PipelineStage.RESULT].status is StageStatus.STALE
    assert states[PipelineStage.WAVES].status is StageStatus.APPLIED


def test_invalidation_rejects_wrong_mapping_types() -> None:
    with pytest.raises(TypeError):
        invalidate_downstream({}, "perspective")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        invalidate_downstream(
            {"waves": StageState()}, PipelineStage.PERSPECTIVE  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        invalidate_downstream(
            {PipelineStage.WAVES: object()}, PipelineStage.PERSPECTIVE  # type: ignore[arg-type]
        )
