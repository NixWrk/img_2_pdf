"""Pure adapter from session diagnostics to the Review pipeline strip."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Mapping

from .stage_reasons import StageReason, describe_stage_reason
from .stage_state import PipelineStage, StageMode, StageState, StageStatus


_TITLES = {
    PipelineStage.PERSPECTIVE: "Perspective",
    PipelineStage.WAVES: "Waves",
    PipelineStage.DESKEW: "Deskew",
    PipelineStage.LIGHTING: "Lighting",
    PipelineStage.CLEANUP: "Cleanup",
    PipelineStage.LAYOUT: "Layout",
    PipelineStage.RESULT: "Result",
}
_CONTROLS = {
    PipelineStage.PERSPECTIVE: ("Auto", "Edit"),
    PipelineStage.WAVES: ("Auto", "Off", "Edit"),
    PipelineStage.DESKEW: ("Auto", "Off", "Edit"),
    PipelineStage.LIGHTING: ("Auto", "Off"),
    PipelineStage.CLEANUP: ("Preset", "Edit"),
    PipelineStage.LAYOUT: ("Off", "Edit"),
    PipelineStage.RESULT: (),
}
_STATUS_LABELS = {
    StageStatus.IDLE: "Idle",
    StageStatus.RUNNING: "Running",
    StageStatus.NOT_NEEDED: "Not needed",
    StageStatus.APPLIED: "Applied",
    StageStatus.REJECTED: "Rejected",
    StageStatus.EDITED: "Edited",
    StageStatus.STALE: "Stale",
    StageStatus.ERROR: "Error",
}
_NONE_METHODS = {None, "", "none", "off", "disabled"}
_NOT_NEEDED_REASONS = {
    "angle_below_threshold",
    "curvature_below_threshold",
    "disabled",
    "no_content",
    "no_isolated_specks",
    "no_shadow_detected",
}


@dataclass(frozen=True, slots=True)
class PipelineCard:
    """Display-ready state for one pipeline stage."""

    stage: PipelineStage
    title: str
    state: StageState
    reason: StageReason
    controls: tuple[str, ...]

    @property
    def status_label(self) -> str:
        return _STATUS_LABELS[self.state.status]

    @property
    def mode_label(self) -> str:
        if not self.controls:
            return ""
        return "Auto" if self.state.mode is StageMode.AUTO else "Off"


def _mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        payload = asdict(value)
        if isinstance(payload, Mapping):
            return payload
    return {}


def _value(payload: Mapping[str, object], name: str, default: object = None) -> object:
    return payload.get(name, default)


def _metrics(payload: Mapping[str, object]) -> dict[str, str | int | float | bool | None]:
    result: dict[str, str | int | float | bool | None] = {}
    excluded = {"applied", "method", "selected_method", "reason", "strength"}
    for key, value in payload.items():
        if key in excluded or not isinstance(value, (str, int, float, bool, type(None))):
            continue
        result[key] = value
    return result


def _mode(method: object) -> StageMode:
    return StageMode.OFF if method in _NONE_METHODS else StageMode.AUTO


def _state_from_diagnostic(
    stage: PipelineStage,
    payload: Mapping[str, object],
    *,
    mode: StageMode,
    input_revision: int,
    candidate_revision: int | None,
    committed_revision: int | None,
    pending: bool,
) -> StageState:
    reason = _value(payload, "reason")
    reason_code = str(reason) if isinstance(reason, str) and reason else None
    applied = _value(payload, "applied") is True
    method = _value(payload, "method")
    if (
        mode is StageMode.OFF
        or (method is not None and method in _NONE_METHODS)
        or reason_code == "disabled"
    ):
        status = StageStatus.NOT_NEEDED
        reason_code = reason_code or "disabled"
    elif (
        applied and reason_code and (reason_code == "user_override" or "user_adjust" in reason_code)
    ):
        status = StageStatus.EDITED
    elif applied:
        status = StageStatus.APPLIED
    elif reason_code in _NOT_NEEDED_REASONS:
        status = StageStatus.NOT_NEEDED
    elif reason_code and ("failed" in reason_code or "unavailable" in reason_code):
        status = StageStatus.ERROR
    elif reason_code:
        status = StageStatus.REJECTED
    elif pending:
        status = StageStatus.RUNNING
    else:
        status = StageStatus.IDLE
    return StageState(
        mode=mode,
        status=status,
        reason_code=reason_code,
        metrics=_metrics(payload),
        input_revision=input_revision,
        candidate_revision=candidate_revision,
        committed_revision=committed_revision,
    )


def _reason_for(stage: PipelineStage, state: StageState) -> StageReason:
    if state.reason_code:
        return describe_stage_reason(stage, state.reason_code, state.metrics)
    title = _TITLES[stage]
    if state.status is StageStatus.IDLE:
        return StageReason(f"{title} is waiting.", "No candidate has been checked yet.", True)
    if state.status is StageStatus.RUNNING:
        return StageReason(f"Checking {title.lower()}…", "Automatic processing is running.", True)
    if state.status is StageStatus.APPLIED:
        if stage is PipelineStage.RESULT:
            if (
                state.committed_revision is not None
                and state.committed_revision == state.candidate_revision
            ):
                return StageReason(
                    "Committed result is ready for export.",
                    "The visible result matches the durable export generation.",
                    True,
                )
            return StageReason(
                "Candidate result is ready to apply.",
                "The preview is not part of export until it is committed.",
                True,
            )
        return StageReason(f"{title} applied.", "The accepted correction is active.", True)
    if state.status is StageStatus.EDITED:
        return StageReason(
            f"{title} has a manual edit.",
            "The edited candidate is not part of export until it is committed.",
            True,
        )
    if state.status is StageStatus.STALE:
        return describe_stage_reason(stage, "upstream_changed", state.metrics)
    return describe_stage_reason(stage, state.reason_code, state.metrics)


def _cleanup_mode(recipe: object, payload: Mapping[str, object]) -> StageMode:
    if _value(payload, "strength") not in _NONE_METHODS:
        return StageMode.AUTO
    if getattr(recipe, "postprocess_name", None) not in _NONE_METHODS:
        return StageMode.AUTO
    if getattr(recipe, "preprocess_settings", None) is not None:
        return StageMode.AUTO
    return StageMode.OFF


def build_pipeline_cards(
    entry: object,
    *,
    pending_request: object | None = None,
    pending_diagnostics: object | None = None,
    pending_error: str | None = None,
) -> tuple[PipelineCard, ...]:
    """Build seven Review cards from committed evidence and an optional candidate."""

    if entry is None:
        raise TypeError("entry is required")
    committed = getattr(entry, "committed_processing", None)
    committed_payload = _mapping(getattr(committed, "diagnostics", None))
    committed_recipe = getattr(committed, "recipe", None)
    recipe = pending_request if pending_request is not None else committed_recipe
    input_revision = getattr(entry, "revision", 0)
    if (
        isinstance(input_revision, bool)
        or not isinstance(input_revision, int)
        or input_revision < 0
    ):
        raise ValueError("entry revision must be a non-negative integer")
    pending = (
        pending_request is not None or pending_diagnostics is not None or pending_error is not None
    )
    candidate_revision = input_revision + 1 if pending else (input_revision if committed else None)
    committed_revision = input_revision if committed else None
    diagnostics = (
        _mapping(pending_diagnostics)
        if pending_diagnostics is not None
        else committed_payload
        if pending_error is not None
        else ({} if pending_request is not None else committed_payload)
    )

    crop_state = str(getattr(entry, "crop_state", "none"))
    contour = getattr(entry, "detected_contour", None)
    review_reasons = tuple(getattr(entry, "review_reasons", ()) or ())
    backend = getattr(entry, "detected_backend", None)
    if crop_state == "proposed" and contour is not None:
        perspective_status = StageStatus.APPLIED
        perspective_reason = review_reasons[0] if review_reasons else "applied_by_detection_backend"
    elif crop_state == "applied" or getattr(entry, "detected_backend", None):
        perspective_status = (
            StageStatus.EDITED
            if backend == "manual" or not (committed or pending)
            else StageStatus.APPLIED
        )
        perspective_reason = "applied_by_detection_backend"
    else:
        perspective_status = StageStatus.REJECTED
        perspective_reason = review_reasons[0] if review_reasons else "boundary_not_detected"
    perspective_candidate_revision = (
        candidate_revision if candidate_revision is not None else input_revision
    )
    perspective_committed_revision = (
        input_revision if crop_state == "applied" or committed else None
    )
    perspective_state = StageState(
        status=perspective_status,
        reason_code=perspective_reason,
        input_revision=input_revision,
        candidate_revision=(
            perspective_candidate_revision
            if perspective_status in {StageStatus.APPLIED, StageStatus.EDITED}
            else candidate_revision
        ),
        committed_revision=perspective_committed_revision,
    )
    cards: list[PipelineCard] = [
        PipelineCard(
            PipelineStage.PERSPECTIVE,
            "Perspective",
            perspective_state,
            _reason_for(PipelineStage.PERSPECTIVE, perspective_state),
            _CONTROLS[PipelineStage.PERSPECTIVE],
        )
    ]

    def add(stage: PipelineStage, payload: Mapping[str, object], *, mode: StageMode) -> None:
        state = _state_from_diagnostic(
            stage,
            payload,
            mode=mode,
            input_revision=input_revision,
            candidate_revision=candidate_revision,
            committed_revision=committed_revision,
            pending=pending and pending_error is None and not payload,
        )
        cards.append(
            PipelineCard(stage, _TITLES[stage], state, _reason_for(stage, state), _CONTROLS[stage])
        )

    dewarp = _mapping(_value(diagnostics, "dewarp"))
    add(PipelineStage.WAVES, dewarp, mode=_mode(getattr(recipe, "dewarp_method", None)))
    deskew = {
        "method": _value(diagnostics, "deskew_method", getattr(recipe, "deskew_method", None)),
        "applied": abs(float(_value(diagnostics, "deskew_angle_degrees", 0.0) or 0.0)) > 1e-6,
        "reason": _value(diagnostics, "deskew_reason"),
        "angle_degrees": _value(diagnostics, "deskew_angle_degrees"),
        "confidence": _value(diagnostics, "deskew_confidence"),
        "line_count": _value(diagnostics, "deskew_line_count"),
    }
    add(PipelineStage.DESKEW, deskew, mode=_mode(getattr(recipe, "deskew_method", None)))
    shadow = _mapping(_value(diagnostics, "shadow"))
    add(PipelineStage.LIGHTING, shadow, mode=_mode(getattr(recipe, "shadow_method", None)))
    despeckle = _mapping(_value(diagnostics, "despeckle"))
    add(PipelineStage.CLEANUP, despeckle, mode=_cleanup_mode(recipe, despeckle))
    layout = _mapping(_value(diagnostics, "layout"))
    add(PipelineStage.LAYOUT, layout, mode=_mode(getattr(recipe, "page_layout", None)))

    stored_result_ready = crop_state == "applied" and not bool(
        getattr(entry, "needs_review", False)
    )
    candidate_has_manual_edit = any(card.state.status is StageStatus.EDITED for card in cards)
    result_status = (
        StageStatus.ERROR
        if pending_error is not None
        else StageStatus.RUNNING
        if pending and pending_diagnostics is None
        else StageStatus.EDITED
        if pending and candidate_has_manual_edit
        else StageStatus.APPLIED
        if pending or committed or stored_result_ready
        else StageStatus.IDLE
    )
    result_candidate_revision = (
        candidate_revision
        if candidate_revision is not None
        else input_revision
        if stored_result_ready
        else None
    )
    result_committed_revision = (
        committed_revision
        if committed_revision is not None
        else input_revision
        if stored_result_ready
        else None
    )
    result_state = StageState(
        status=result_status,
        reason_code="model_failed" if pending_error is not None else None,
        input_revision=input_revision,
        candidate_revision=result_candidate_revision,
        committed_revision=result_committed_revision,
    )
    cards.append(
        PipelineCard(
            PipelineStage.RESULT,
            "Result",
            result_state,
            _reason_for(PipelineStage.RESULT, result_state),
            _CONTROLS[PipelineStage.RESULT],
        )
    )
    return tuple(cards)


__all__ = ["PipelineCard", "build_pipeline_cards"]
