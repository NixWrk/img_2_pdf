"""Pure, immutable state model for the document-processing pipeline.

The UI can adopt this module incrementally.  ``input_revision`` identifies
the upstream input, ``candidate_revision`` the latest preview generation,
and ``committed_revision`` the generation accepted for export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Final, Mapping, TypeAlias


class StageMode(str, Enum):
    AUTO = "auto"
    OFF = "off"


class StageStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    NOT_NEEDED = "not_needed"
    APPLIED = "applied"
    REJECTED = "rejected"
    EDITED = "edited"
    STALE = "stale"
    ERROR = "error"


class PipelineStage(str, Enum):
    """Canonical left-to-right order used for downstream invalidation."""

    PERSPECTIVE = "perspective"
    WAVES = "waves"
    DESKEW = "deskew"
    LIGHTING = "lighting"
    CLEANUP = "cleanup"
    LAYOUT = "layout"
    RESULT = "result"


MetricValue: TypeAlias = str | int | float | bool | None
_UNSET: Final = object()


PIPELINE_DEPENDENCIES: Final[Mapping[PipelineStage, tuple[PipelineStage, ...]]] = MappingProxyType(
    {
        PipelineStage.PERSPECTIVE: (PipelineStage.WAVES,),
        PipelineStage.WAVES: (PipelineStage.DESKEW,),
        PipelineStage.DESKEW: (PipelineStage.LIGHTING,),
        PipelineStage.LIGHTING: (PipelineStage.CLEANUP,),
        PipelineStage.CLEANUP: (PipelineStage.LAYOUT,),
        PipelineStage.LAYOUT: (PipelineStage.RESULT,),
        PipelineStage.RESULT: (),
    }
)


_ALLOWED_TRANSITIONS: Final[Mapping[StageStatus, frozenset[StageStatus]]] = MappingProxyType(
    {
        StageStatus.IDLE: frozenset(
            {StageStatus.IDLE, StageStatus.RUNNING, StageStatus.NOT_NEEDED, StageStatus.ERROR}
        ),
        StageStatus.RUNNING: frozenset(
            {
                StageStatus.RUNNING,
                StageStatus.NOT_NEEDED,
                StageStatus.APPLIED,
                StageStatus.REJECTED,
                StageStatus.EDITED,
                StageStatus.ERROR,
                StageStatus.STALE,
            }
        ),
        StageStatus.NOT_NEEDED: frozenset(
            {StageStatus.NOT_NEEDED, StageStatus.RUNNING, StageStatus.EDITED, StageStatus.STALE}
        ),
        StageStatus.APPLIED: frozenset(
            {
                StageStatus.APPLIED,
                StageStatus.RUNNING,
                StageStatus.EDITED,
                StageStatus.NOT_NEEDED,
                StageStatus.STALE,
            }
        ),
        StageStatus.REJECTED: frozenset(
            {StageStatus.REJECTED, StageStatus.RUNNING, StageStatus.EDITED, StageStatus.STALE}
        ),
        StageStatus.EDITED: frozenset(
            {
                StageStatus.EDITED,
                StageStatus.RUNNING,
                StageStatus.APPLIED,
                StageStatus.REJECTED,
                StageStatus.ERROR,
                StageStatus.STALE,
            }
        ),
        StageStatus.STALE: frozenset(
            {StageStatus.STALE, StageStatus.RUNNING, StageStatus.NOT_NEEDED, StageStatus.ERROR}
        ),
        StageStatus.ERROR: frozenset({StageStatus.ERROR, StageStatus.RUNNING, StageStatus.STALE}),
    }
)


def _validate_revision(name: str, value: int | None, *, allow_none: bool = True) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a non-negative integer{suffix}")


def _validate_reason(status: StageStatus, reason_code: str | None) -> None:
    if reason_code is not None and (not isinstance(reason_code, str) or not reason_code.strip()):
        raise ValueError("reason_code must be a non-empty string or None")
    if status in {StageStatus.NOT_NEEDED, StageStatus.REJECTED, StageStatus.ERROR} and not reason_code:
        raise ValueError(f"{status.value} requires reason_code")


def _freeze_metrics(metrics: Mapping[str, MetricValue]) -> Mapping[str, MetricValue]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    normalized: dict[str, MetricValue] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric keys must be non-empty strings")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"metric {key!r} has unsupported value type")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class StageState:
    """Validated state for one processing stage."""

    mode: StageMode = StageMode.AUTO
    status: StageStatus = StageStatus.IDLE
    reason_code: str | None = None
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    input_revision: int = 0
    candidate_revision: int | None = None
    committed_revision: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, StageMode):
            raise TypeError("mode must be a StageMode")
        if not isinstance(self.status, StageStatus):
            raise TypeError("status must be a StageStatus")
        _validate_revision("input_revision", self.input_revision, allow_none=False)
        _validate_revision("candidate_revision", self.candidate_revision)
        _validate_revision("committed_revision", self.committed_revision)
        if (
            self.candidate_revision is not None
            and self.committed_revision is not None
            and self.committed_revision > self.candidate_revision
        ):
            raise ValueError("committed_revision cannot be newer than candidate_revision")
        _validate_reason(self.status, self.reason_code)
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        if self.status is StageStatus.APPLIED and self.candidate_revision is None:
            raise ValueError("applied state requires candidate_revision")
        if self.status is StageStatus.EDITED and self.candidate_revision is None:
            raise ValueError("edited state requires candidate_revision")

    def transition(
        self,
        status: StageStatus,
        *,
        reason_code: str | None | object = _UNSET,
        metrics: Mapping[str, MetricValue] | object = _UNSET,
        input_revision: int | object = _UNSET,
        candidate_revision: int | None | object = _UNSET,
        committed_revision: int | None | object = _UNSET,
    ) -> StageState:
        """Return a new state after one legal lifecycle transition."""

        if not isinstance(status, StageStatus):
            raise TypeError("status must be a StageStatus")
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid stage transition: {self.status.value} -> {status.value}")
        return StageState(
            mode=self.mode,
            status=status,
            reason_code=self.reason_code if reason_code is _UNSET else reason_code,
            metrics=self.metrics if metrics is _UNSET else metrics,
            input_revision=self.input_revision if input_revision is _UNSET else input_revision,
            candidate_revision=(
                self.candidate_revision if candidate_revision is _UNSET else candidate_revision
            ),
            committed_revision=(
                self.committed_revision if committed_revision is _UNSET else committed_revision
            ),
        )

    def with_mode(self, mode: StageMode) -> StageState:
        if not isinstance(mode, StageMode):
            raise TypeError("mode must be a StageMode")
        return StageState(
            mode=mode,
            status=self.status,
            reason_code=self.reason_code,
            metrics=self.metrics,
            input_revision=self.input_revision,
            candidate_revision=self.candidate_revision,
            committed_revision=self.committed_revision,
        )

    def mark_stale(self, *, input_revision: int | None = None) -> StageState:
        return self.transition(
            StageStatus.STALE,
            reason_code="upstream_changed",
            input_revision=self.input_revision if input_revision is None else input_revision,
        )


def downstream_stages(stage: PipelineStage, *, include_self: bool = False) -> tuple[PipelineStage, ...]:
    """Return all descendants of ``stage`` in pipeline order."""

    if not isinstance(stage, PipelineStage):
        raise TypeError("stage must be a PipelineStage")
    result: list[PipelineStage] = [stage] if include_self else []
    pending = list(PIPELINE_DEPENDENCIES[stage])
    seen = set(result)
    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        result.append(current)
        pending.extend(PIPELINE_DEPENDENCIES[current])
    return tuple(result)


def invalidate_downstream(
    states: Mapping[PipelineStage, StageState],
    changed_stage: PipelineStage,
    *,
    input_revision: int | None = None,
) -> dict[PipelineStage, StageState]:
    """Copy ``states`` with every materialised descendant marked stale."""

    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping")
    if not isinstance(changed_stage, PipelineStage):
        raise TypeError("changed_stage must be a PipelineStage")
    for stage, state in states.items():
        if not isinstance(stage, PipelineStage):
            raise TypeError("state keys must be PipelineStage values")
        if not isinstance(state, StageState):
            raise TypeError("state values must be StageState values")
    result = dict(states)
    for stage in downstream_stages(changed_stage):
        state = result.get(stage)
        if state is not None and state.status is not StageStatus.IDLE:
            result[stage] = state.mark_stale(input_revision=input_revision)
    return result


__all__ = [
    "MetricValue",
    "PIPELINE_DEPENDENCIES",
    "PipelineStage",
    "StageMode",
    "StageState",
    "StageStatus",
    "downstream_stages",
    "invalidate_downstream",
]
