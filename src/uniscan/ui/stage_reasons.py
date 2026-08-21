"""User-facing explanations for automatic pipeline decisions.

The processing modules intentionally keep diagnostic reason codes terse and
machine-readable.  This module is the small presentation boundary between
those diagnostics and the UI.  It does not run processing, mutate state, or
depend on any widget toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .stage_state import MetricValue, PipelineStage


@dataclass(frozen=True, slots=True)
class StageReason:
    """Copy shown for one diagnostic reason code."""

    summary: str
    details: str
    known: bool


_COPY = tuple[str, str]


_COMMON: dict[str, _COPY] = {
    "disabled": ("This stage is turned off.", "Automatic processing was disabled for this stage."),
    "upstream_changed": (
        "This stage needs to be rerun after an earlier edit.",
        "An upstream stage changed, so this result is no longer current.",
    ),
    "quality_below_threshold": (
        "The proposed correction was not reliable enough to keep.",
        "The quality check did not meet the acceptance threshold.",
    ),
    "model_failed": (
        "Automatic processing could not finish.",
        "The model reported a processing error; the source image was kept unchanged.",
    ),
    "empty_image": ("There is no image to process.", "The input image was empty."),
    "image_too_small": (
        "The image is too small for a reliable correction.",
        "The input does not contain enough pixels for this stage to make a safe decision.",
    ),
}


_STAGE_COPY: dict[PipelineStage, dict[str, _COPY]] = {
    PipelineStage.PERSPECTIVE: {
        "applied_by_detection_backend": (
            "Perspective correction is already applied.",
            "The document detector supplied the accepted page boundary.",
        ),
        "boundary_not_detected": (
            "No page boundary was detected.",
            "The original framing was kept so no page content is lost.",
        ),
        "large_dark_border_region": (
            "The page boundary needs a quick review.",
            "A large dark border may be part of the capture rather than the page.",
        ),
        "empty_output": (
            "The correction produced no usable page.",
            "The proposed perspective result was empty, so the source was kept.",
        ),
        "no document boundary detected": (
            "No page boundary was detected.",
            "The detector did not find a reliable document outline.",
        ),
        "rejected landscape crop from portrait source": (
            "The wide crop was rejected to protect page content.",
            "The full portrait source was kept instead of publishing a destructive crop.",
        ),
    },
    PipelineStage.WAVES: {
        "curvature_below_threshold": (
            "No wave correction was needed.",
            "Detected page curvature was below the safe correction threshold.",
        ),
        "insufficient_text_lines": (
            "There are not enough text lines for wave correction.",
            "The text-line model could not find enough evidence to build a safe curve.",
        ),
        "uvdoc_model_unavailable": (
            "The wave-correction model is unavailable.",
            "The optional UVDoc model could not be loaded, so the page was kept unchanged.",
        ),
        "uvdoc_no_result": (
            "The wave-correction model found no usable result.",
            "UVDoc returned no accepted page geometry.",
        ),
        "docscanner_model_unavailable": (
            "The wave-correction model is unavailable.",
            "The optional DocScanner model could not be loaded.",
        ),
        "docscanner_no_result": (
            "The wave-correction model found no usable result.",
            "DocScanner returned no accepted page geometry.",
        ),
        "uvdoc_not_flatter": (
            "The proposed wave correction was not flatter.",
            "The candidate did not improve the measured page curvature.",
        ),
        "user_adjusted_model": (
            "Your manual wave adjustment is active.",
            "The model was updated from the control points you edited.",
        ),
        "uvdoc_with_user_adjustment": (
            "Your manual wave adjustment is active.",
            "The accepted UVDoc result includes your control-point adjustment.",
        ),
        "docscanner_with_user_adjustment": (
            "Your manual wave adjustment is active.",
            "The accepted DocScanner result includes your control-point adjustment.",
        ),
        "applied_by_detection_backend": (
            "Wave correction is already applied.",
            "The detection backend supplied the accepted page geometry.",
        ),
    },
    PipelineStage.DESKEW: {
        "hough_selected": (
            "Deskew found a usable text angle.",
            "The Hough line estimate supplied the accepted rotation.",
        ),
        "no_lines": (
            "No reliable text lines were found.",
            "Deskew was skipped because there was not enough line evidence.",
        ),
        "insufficient_lines": (
            "There are not enough text lines for deskew.",
            "Deskew was skipped because the line evidence was insufficient.",
        ),
        "no_foreground": (
            "No page foreground was found.",
            "Deskew could not separate page content from the background.",
        ),
        "angle_out_of_range": (
            "The detected angle was outside the safe range.",
            "Deskew did not apply an angle that could crop or distort the page.",
        ),
        "angle_below_threshold": (
            "No deskew was needed.",
            "The detected rotation was below the correction threshold.",
        ),
        "user_override": (
            "Your manual deskew angle is active.",
            "The result uses the angle supplied in the manual control.",
        ),
        "hough_no_lines": (
            "Deskew used a safe fallback angle.",
            "No reliable Hough lines were found, so the minimum-area estimate was used.",
        ),
        "hough_insufficient_lines": (
            "Deskew used a safe fallback angle.",
            "Hough line evidence was insufficient, so the minimum-area estimate was used.",
        ),
        "hough_no_foreground": (
            "Deskew used a safe fallback angle.",
            "The Hough method found no foreground, so the minimum-area estimate was used.",
        ),
        "hough_angle_out_of_range": (
            "Deskew used a safe fallback angle.",
            "The Hough angle was outside the safe range, so the minimum-area estimate was used.",
        ),
        "min_area_selected": (
            "Deskew used a safe fallback angle.",
            "The minimum-area estimate supplied the accepted rotation.",
        ),
    },
    PipelineStage.LIGHTING: {
        "no_shadow_detected": (
            "Lighting is already even.",
            "No measurable shadow pattern needed correction.",
        ),
        "docshadow_model_unavailable": (
            "The lighting model is unavailable.",
            "The optional DocShadow model could not be loaded.",
        ),
        "docshadow_rejected:lighting_not_improved": (
            "The proposed lighting correction was not better.",
            "The model result did not improve the measured lighting enough.",
        ),
        "classical_rejected:lighting_not_improved": (
            "The proposed lighting correction was not better.",
            "The fallback result did not improve the measured lighting enough.",
        ),
        "docshadow_rejected:excessive_glare": (
            "The proposed lighting correction was rejected.",
            "The candidate introduced too much glare.",
        ),
        "classical_rejected:excessive_glare": (
            "The proposed lighting correction was rejected.",
            "The fallback candidate introduced too much glare.",
        ),
        "docshadow_rejected:excessive_clipping": (
            "The proposed lighting correction was rejected.",
            "The candidate clipped too much page detail.",
        ),
        "classical_rejected:excessive_clipping": (
            "The proposed lighting correction was rejected.",
            "The fallback candidate clipped too much page detail.",
        ),
    },
    PipelineStage.CLEANUP: {
        "no_isolated_specks": (
            "No isolated specks were detected.",
            "Cleanup found no small components that could be safely removed.",
        ),
    },
    PipelineStage.LAYOUT: {
        "no_content": (
            "No page content was detected.",
            "The layout stage could not find a content box to place on the page.",
        ),
    },
}


_PREFIX_COPY: dict[PipelineStage, dict[str, _COPY]] = {
    PipelineStage.WAVES: {
        "textline_rejected:": (
            "The text-line wave correction was rejected.",
            "The candidate did not pass the geometry quality check.",
        ),
        "textline_fallback:": (
            "Wave correction used a safe fallback.",
            "The text-line model did not have enough evidence, so another method was tried.",
        ),
        "textline:": (
            "Wave correction used text-line evidence.",
            "The text-line model supplied part of the automatic geometry decision.",
        ),
        "uvdoc_rejected:": (
            "The wave-correction proposal was rejected.",
            "The UVDoc candidate did not pass the geometry quality check.",
        ),
        "uvdoc_unavailable:": (
            "The wave-correction model is unavailable.",
            "UVDoc could not be used for this page.",
        ),
    },
    PipelineStage.LIGHTING: {
        "docshadow_failed:": (
            "Automatic lighting correction could not finish.",
            "The DocShadow model reported an error, so the source lighting was kept.",
        ),
        "docshadow_rejected:": (
            "The proposed lighting correction was rejected.",
            "The model candidate did not pass a lighting quality check.",
        ),
        "classical_rejected:": (
            "The proposed lighting correction was rejected.",
            "The fallback candidate did not pass a lighting quality check.",
        ),
    },
}


_DEFAULT_SUMMARY: dict[PipelineStage, str] = {
    PipelineStage.PERSPECTIVE: "Perspective correction was kept unchanged.",
    PipelineStage.WAVES: "Wave correction was kept unchanged.",
    PipelineStage.DESKEW: "Deskew was kept unchanged.",
    PipelineStage.LIGHTING: "Lighting correction was kept unchanged.",
    PipelineStage.CLEANUP: "Cleanup was kept unchanged.",
    PipelineStage.LAYOUT: "Page layout was kept unchanged.",
    PipelineStage.RESULT: "The final result is not ready yet.",
}


_METRIC_LABELS = {
    "confidence": "Confidence",
    "angle_degrees": "Angle",
    "curvature_before_px": "Curvature before",
    "curvature_after_px": "Curvature after",
    "shadow_fraction": "Shadow area",
    "glare_fraction": "Glare area",
    "removed_components": "Removed components",
    "protected_components": "Protected components",
    "content_confidence": "Content confidence",
    "duration_ms": "Duration",
}


def _lookup(stage: PipelineStage, code: str) -> _COPY | None:
    stage_copy = _STAGE_COPY.get(stage, {})
    if code in stage_copy:
        return stage_copy[code]
    if code in _COMMON:
        return _COMMON[code]
    for prefix, copy in _PREFIX_COPY.get(stage, {}).items():
        if code.startswith(prefix):
            return copy
    if stage is PipelineStage.PERSPECTIVE and code.endswith("no candidate"):
        return (
            "No page boundary was detected.",
            "The detector did not find a reliable document candidate.",
        )
    return None


def _format_value(value: MetricValue) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _format_metrics(metrics: Mapping[str, MetricValue] | None) -> str:
    if metrics is None:
        return ""
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping or None")
    parts: list[str] = []
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric keys must be non-empty strings")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"metric {key!r} has unsupported value type")
        label = _METRIC_LABELS.get(key, key.replace("_", " ").capitalize())
        suffix = "°" if key == "angle_degrees" else (" ms" if key == "duration_ms" else "")
        parts.append(f"{label}: {_format_value(value)}{suffix}")
    return "; ".join(parts)


def describe_stage_reason(
    stage: PipelineStage,
    reason_code: str | None,
    metrics: Mapping[str, MetricValue] | None = None,
) -> StageReason:
    """Return concise UI copy for a diagnostic reason and optional evidence.

    Composite diagnostic strings are common in automatic fallbacks.  Their
    first recognized part determines the short summary, while all parts are
    retained in details.  Unknown codes never leak into the main summary.
    """

    if not isinstance(stage, PipelineStage):
        raise TypeError("stage must be a PipelineStage")
    if reason_code is not None and not isinstance(reason_code, str):
        raise TypeError("reason_code must be a string or None")
    normalized = reason_code.strip() if reason_code is not None else ""
    if not normalized:
        result = StageReason(
            summary=_DEFAULT_SUMMARY[stage],
            details="The processing report did not include a reason.",
            known=False,
        )
    else:
        parts = [part.strip() for part in normalized.split(";") if part.strip()]
        copies = [_lookup(stage, part) for part in parts]
        first = next((copy for copy in copies if copy is not None), None)
        if first is None:
            result = StageReason(
                summary=_DEFAULT_SUMMARY[stage],
                details=(
                    f"The processing report returned an unrecognized reason code: {normalized!r}."
                ),
                known=False,
            )
        else:
            detail_parts = [copy[1] for copy in copies if copy is not None]
            unknown_parts = [part for part, copy in zip(parts, copies) if copy is None]
            if unknown_parts:
                detail_parts.append("Additional diagnostic: " + "; ".join(unknown_parts))
            result = StageReason(
                summary=first[0],
                details=" ".join(dict.fromkeys(detail_parts)),
                known=not unknown_parts,
            )
    metric_details = _format_metrics(metrics)
    if metric_details:
        result = StageReason(
            summary=result.summary,
            details=f"{result.details} Evidence: {metric_details}.",
            known=result.known,
        )
    return result


__all__ = ["StageReason", "describe_stage_reason"]
