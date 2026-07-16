"""One GUI-independent controller for every post-detection document stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time

import numpy as np

from .cleanup import (
    DESPECKLE_CHOICES,
    BINARIZATION_NONE,
    DESPECKLE_NONE,
    DespeckleDiagnostics,
    LightingDiagnostics,
    analyze_lighting,
)
from .dewarp import (
    DEWARP_METHOD_CHOICES,
    DEWARP_METHOD_NONE,
    DewarpDiagnostics,
    DewarpModel,
    dewarp_document,
)
from .layout import (
    PAGE_LAYOUT_CHOICES,
    PAGE_LAYOUT_NONE,
    PageLayoutDiagnostics,
    layout_document_page,
)
from .layout import ContentBox
from .orientation import (
    ORIENTATION_METHOD_CHOICES,
    ORIENTATION_METHOD_NONE,
    OrientationDiagnostics,
    orient_document,
)
from .postprocess import POSTPROCESSING_OPTIONS, grayscale
from .preprocess import (
    DESKEW_METHOD_CHOICES,
    DESKEW_METHOD_NONE,
    PreprocessSettings,
    SkewEstimate,
    apply_enhancements_with_diagnostics,
    deskew_document_with_diagnostics,
)
from uniscan.storage.stage_cache import ProcessingStageCache

CancelCb = Callable[[], bool]
PROCESSING_ALGORITHM_VERSION = 3


@dataclass(slots=True)
class PageProcessingRequest:
    """All settings required to reproduce a processed page."""

    orientation_method: str = ORIENTATION_METHOD_NONE
    deskew_method: str = DESKEW_METHOD_NONE
    dewarp_method: str = DEWARP_METHOD_NONE
    dewarp_model: DewarpModel | None = None
    dewarp_already_applied: bool = False
    uvdoc_cache_home: Path | None = None
    auto_dewarp_uvdoc: bool = False
    postprocess_name: str = "None"
    preprocess_settings: PreprocessSettings | None = None
    page_layout: str = PAGE_LAYOUT_NONE
    page_dpi: int = 300
    page_margin_mm: float = 10.0
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"
    lighting_diagnostics: bool = False
    stage_cache: ProcessingStageCache | None = field(default=None, repr=False)
    source_fingerprint: str | None = None
    cancel_cb: CancelCb | None = field(default=None, repr=False)


@dataclass(slots=True, frozen=True)
class PageProcessingDiagnostics:
    orientation: OrientationDiagnostics
    deskew_method: str
    deskew_angle_degrees: float
    deskew_selected_method: str
    deskew_confidence: float
    deskew_line_count: int
    deskew_reason: str
    dewarp: DewarpDiagnostics
    despeckle: DespeckleDiagnostics
    layout: PageLayoutDiagnostics
    lighting: LightingDiagnostics | None
    stage_durations_ms: dict[str, float]
    cache_hits: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class PageProcessingResult:
    image: np.ndarray
    diagnostics: PageProcessingDiagnostics


def _cancelled(request: PageProcessingRequest) -> None:
    if request.cancel_cb is not None and request.cancel_cb():
        raise RuntimeError("Cancelled by user.")


def _uses_unidentified_uvdoc(request: PageProcessingRequest) -> bool:
    """Whether this run may invoke model-backed UVDoc without a stable model identity."""
    if request.dewarp_already_applied:
        # The already-warped pixels are the controller input and therefore part
        # of the source fingerprint.
        return False
    # This is intentionally conservative for auto: before executing the stage
    # we cannot prove that deterministic textline correction will win, so
    # model-independent persistent hits are disabled through all downstream
    # stages whenever UVDoc is allowed.
    return request.dewarp_method == "paddleocr_uvdoc" or (
        request.dewarp_method == "auto" and request.auto_dewarp_uvdoc
    )


def _timed(stage: str, durations: dict[str, float], operation):
    started = time.perf_counter()
    result = operation()
    durations[stage] = round((time.perf_counter() - started) * 1000.0, 3)
    return result


def _run_stage(
    *,
    stage: str,
    image: np.ndarray,
    upstream_key: str,
    options: dict[str, object],
    operation,
    encode_diagnostics,
    decode_diagnostics,
    cacheable: bool,
    request: PageProcessingRequest,
    durations: dict[str, float],
    cache_hits: list[str],
):
    _cancelled(request)
    cache = request.stage_cache
    key = (
        cache.stage_key(
            upstream_key,
            stage,
            {"version": PROCESSING_ALGORITHM_VERSION, **options},
        )
        if cache is not None
        else upstream_key
    )
    started = time.perf_counter()
    if cache is not None and cacheable:
        cached = cache.get(key)
        _cancelled(request)
        if cached is not None:
            cached_image, metadata = cached
            try:
                diagnostics = decode_diagnostics(metadata)
            except (KeyError, OverflowError, TypeError, ValueError):
                # The generic cache can validate its envelope, but only this
                # stage knows the semantic shape of its diagnostics. Treat a
                # stale or malformed payload as a miss and repair it below.
                cache.reject_hit(key)
            else:
                durations[stage] = round((time.perf_counter() - started) * 1000.0, 3)
                cache_hits.append(stage)
                _cancelled(request)
                return cached_image, diagnostics, key

    output, diagnostics = operation()
    durations[stage] = round((time.perf_counter() - started) * 1000.0, 3)
    _cancelled(request)
    if cache is not None and cacheable:
        cache.put(key, output, encode_diagnostics(diagnostics))
        _cancelled(request)
    return output, diagnostics, key


def _strict_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Invalid cached {field_name}.")
    return value


def _strict_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid cached {field_name}.")
    return value


def _finite_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid cached {field_name}.")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"Invalid cached {field_name}.") from exc
    if (
        not math.isfinite(normalized)
        or (minimum is not None and normalized < minimum)
        or (maximum is not None and normalized > maximum)
    ):
        raise ValueError(f"Invalid cached {field_name}.")
    return normalized


def _optional_reason(value: object, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Invalid cached {field_name}.")
    return value


def _orientation_from_dict(payload: dict[str, object]) -> OrientationDiagnostics:
    diagnostics = OrientationDiagnostics(**payload)
    if diagnostics.method not in ORIENTATION_METHOD_CHOICES:
        raise ValueError("Invalid cached orientation method.")
    _strict_bool(diagnostics.applied, field_name="orientation applied flag")
    if type(diagnostics.angle_degrees) is not int or diagnostics.angle_degrees not in {
        0,
        90,
        180,
        270,
    }:
        raise ValueError("Invalid cached orientation angle.")
    _finite_float(
        diagnostics.confidence,
        field_name="orientation confidence",
        minimum=0.0,
        maximum=1.0,
    )
    _strict_int(diagnostics.line_count, field_name="orientation line count")
    _optional_reason(diagnostics.reason, field_name="orientation reason")
    return diagnostics


def _layout_from_dict(payload: dict[str, object]) -> PageLayoutDiagnostics:
    values = dict(payload)
    content_box = values.pop("content_box")
    if not isinstance(content_box, dict):
        raise ValueError("Invalid cached content box.")
    diagnostics = PageLayoutDiagnostics(
        content_box=ContentBox(**content_box),
        **values,
    )
    if diagnostics.method not in PAGE_LAYOUT_CHOICES:
        raise ValueError("Invalid cached layout method.")
    _strict_bool(diagnostics.applied, field_name="layout applied flag")
    for field_name in ("x", "y", "width", "height"):
        _strict_int(
            getattr(diagnostics.content_box, field_name), field_name=f"content box {field_name}"
        )
    _finite_float(
        diagnostics.content_confidence,
        field_name="content confidence",
        minimum=0.0,
        maximum=1.0,
    )
    _finite_float(diagnostics.scale, field_name="layout scale", minimum=0.0)
    _strict_int(diagnostics.target_width, field_name="layout target width")
    _strict_int(diagnostics.target_height, field_name="layout target height")
    _optional_reason(diagnostics.reason, field_name="layout reason")
    return diagnostics


def _deskew_from_dict(payload: dict[str, object]) -> dict[str, object]:
    """Validate cached deskew diagnostics before they reach report assembly."""
    estimate = SkewEstimate(**payload)
    if estimate.method not in DESKEW_METHOD_CHOICES:
        raise ValueError("Invalid cached deskew method.")
    if (
        estimate.selected_method is not None
        and estimate.selected_method not in DESKEW_METHOD_CHOICES
    ):
        raise ValueError("Invalid cached selected deskew method.")
    _optional_reason(estimate.reason, field_name="deskew reason")
    _strict_int(estimate.line_count, field_name="deskew line count")
    angle = _finite_float(estimate.angle_degrees, field_name="deskew angle")
    confidence = _finite_float(
        estimate.confidence,
        field_name="deskew confidence",
        minimum=0.0,
        maximum=1.0,
    )
    return {
        **asdict(estimate),
        "angle_degrees": angle,
        "confidence": confidence,
    }


def _dewarp_from_dict(payload: dict[str, object]) -> DewarpDiagnostics:
    diagnostics = DewarpDiagnostics(**payload)
    if diagnostics.method not in DEWARP_METHOD_CHOICES:
        raise ValueError("Invalid cached dewarp method.")
    if diagnostics.selected_method not in DEWARP_METHOD_CHOICES:
        raise ValueError("Invalid cached selected dewarp method.")
    _strict_bool(diagnostics.applied, field_name="dewarp applied flag")
    _strict_int(diagnostics.line_count, field_name="dewarp line count")
    for field_name in (
        "max_displacement_px",
        "curvature_rms_px",
        "curvature_before_px",
        "curvature_after_px",
        "blank_border_before",
        "blank_border_after",
        "edge_ink_before",
        "edge_ink_after",
        "aspect_change",
        "duration_ms",
    ):
        _finite_float(getattr(diagnostics, field_name), field_name=f"dewarp {field_name}")
    _optional_reason(diagnostics.reason, field_name="dewarp reason")
    return diagnostics


def _despeckle_from_dict(payload: dict[str, object]) -> DespeckleDiagnostics:
    diagnostics = DespeckleDiagnostics(**payload)
    if diagnostics.strength not in DESPECKLE_CHOICES:
        raise ValueError("Invalid cached despeckle strength.")
    _strict_bool(diagnostics.applied, field_name="despeckle applied flag")
    for field_name in (
        "candidate_components",
        "removed_components",
        "removed_pixels",
        "protected_components",
    ):
        _strict_int(getattr(diagnostics, field_name), field_name=f"despeckle {field_name}")
    _optional_reason(diagnostics.reason, field_name="despeckle reason")
    return diagnostics


def process_document_page(
    image: np.ndarray,
    request: PageProcessingRequest,
) -> PageProcessingResult:
    """Run orientation, deskew, dewarp, cleanup, and layout in the canonical order."""
    if image.size == 0:
        raise ValueError("Cannot process an empty page image.")
    if request.dewarp_already_applied and request.dewarp_method == DEWARP_METHOD_NONE:
        raise ValueError("dewarp_already_applied is incompatible with dewarp_method='none'.")
    durations: dict[str, float] = {}
    cache_hits: list[str] = []
    cache = request.stage_cache
    cache_safe_after_dewarp = not _uses_unidentified_uvdoc(request)
    upstream_key = ""
    if cache is not None:
        upstream_key = request.source_fingerprint or cache.fingerprint_image(image)

    _cancelled(request)
    oriented, orientation, upstream_key = _run_stage(
        stage="orientation",
        image=image,
        upstream_key=upstream_key,
        options={"method": request.orientation_method},
        operation=lambda: orient_document(image, method=request.orientation_method),
        encode_diagnostics=asdict,
        decode_diagnostics=_orientation_from_dict,
        cacheable=request.orientation_method != ORIENTATION_METHOD_NONE,
        request=request,
        durations=durations,
        cache_hits=cache_hits,
    )
    _cancelled(request)

    def deskew_stage():
        output, estimate = deskew_document_with_diagnostics(
            oriented,
            method=request.deskew_method,
        )
        return output, asdict(estimate)

    deskewed, deskew_payload, upstream_key = _run_stage(
        stage="deskew",
        image=oriented,
        upstream_key=upstream_key,
        options={"method": request.deskew_method, "diagnostics_version": 2},
        operation=deskew_stage,
        encode_diagnostics=lambda payload: payload,
        decode_diagnostics=_deskew_from_dict,
        cacheable=request.deskew_method != DESKEW_METHOD_NONE,
        request=request,
        durations=durations,
        cache_hits=cache_hits,
    )
    deskew_angle = float(deskew_payload["angle_degrees"])
    _cancelled(request)
    if request.dewarp_already_applied:
        dewarped = deskewed
        dewarp = DewarpDiagnostics(
            method=request.dewarp_method,
            applied=True,
            selected_method=request.dewarp_method,
            reason="applied_by_detection_backend",
        )
        durations["dewarp"] = 0.0
        if cache is not None:
            upstream_key = cache.stage_key(
                upstream_key,
                "dewarp",
                {"version": 1, "already_applied": True, "method": request.dewarp_method},
            )
    else:
        dewarped, dewarp, upstream_key = _run_stage(
            stage="dewarp",
            image=deskewed,
            upstream_key=upstream_key,
            options={
                "method": request.dewarp_method,
                "model": asdict(request.dewarp_model) if request.dewarp_model is not None else None,
                "auto_uvdoc": request.auto_dewarp_uvdoc,
                "uvdoc_cache": (
                    str(request.uvdoc_cache_home) if request.uvdoc_cache_home is not None else None
                ),
            },
            operation=lambda: dewarp_document(
                deskewed,
                method=request.dewarp_method,
                uvdoc_cache_home=request.uvdoc_cache_home,
                auto_use_uvdoc=request.auto_dewarp_uvdoc,
                model=request.dewarp_model,
            ),
            encode_diagnostics=asdict,
            decode_diagnostics=_dewarp_from_dict,
            cacheable=(request.dewarp_method != DEWARP_METHOD_NONE and cache_safe_after_dewarp),
            request=request,
            durations=durations,
            cache_hits=cache_hits,
        )

    _cancelled(request)
    if request.postprocess_name not in POSTPROCESSING_OPTIONS:
        raise ValueError(f"Unsupported postprocess mode: {request.postprocess_name}")
    postprocess = POSTPROCESSING_OPTIONS[request.postprocess_name]

    def cleanup_stage():
        settings = request.preprocess_settings
        will_binarize = settings is not None and (
            settings.apply_threshold or settings.binarization_method != BINARIZATION_NONE
        )
        # Black-and-white postprocess already applies an adaptive threshold.
        # If the cleanup settings request a binarizer, start from grayscale so
        # fixed/Otsu/Sauvola/Wolf is applied exactly once.
        postprocessed = (
            grayscale(dewarped)
            if request.postprocess_name == "Black and White" and will_binarize
            else postprocess(dewarped)
        )
        if request.preprocess_settings is None:
            return postprocessed, DespeckleDiagnostics(
                strength=DESPECKLE_NONE,
                applied=False,
                reason="disabled",
            )
        return apply_enhancements_with_diagnostics(postprocessed, request.preprocess_settings)

    cleaned, despeckle, upstream_key = _run_stage(
        stage="cleanup",
        image=dewarped,
        upstream_key=upstream_key,
        options={
            "postprocess": request.postprocess_name,
            "preprocess": (
                asdict(request.preprocess_settings)
                if request.preprocess_settings is not None
                else None
            ),
        },
        operation=cleanup_stage,
        encode_diagnostics=asdict,
        decode_diagnostics=_despeckle_from_dict,
        cacheable=cache_safe_after_dewarp
        and (request.postprocess_name != "None" or request.preprocess_settings is not None),
        request=request,
        durations=durations,
        cache_hits=cache_hits,
    )
    _cancelled(request)
    laid_out, layout, _upstream_key = _run_stage(
        stage="layout",
        image=cleaned,
        upstream_key=upstream_key,
        options={
            "method": request.page_layout,
            "dpi": request.page_dpi,
            "margin_mm": request.page_margin_mm,
            "align_x": request.horizontal_alignment,
            "align_y": request.vertical_alignment,
        },
        operation=lambda: layout_document_page(
            cleaned,
            method=request.page_layout,
            dpi=request.page_dpi,
            margin_mm=request.page_margin_mm,
            horizontal_alignment=request.horizontal_alignment,
            vertical_alignment=request.vertical_alignment,
        ),
        encode_diagnostics=asdict,
        decode_diagnostics=_layout_from_dict,
        cacheable=cache_safe_after_dewarp and request.page_layout != PAGE_LAYOUT_NONE,
        request=request,
        durations=durations,
        cache_hits=cache_hits,
    )
    lighting = None
    if request.lighting_diagnostics:
        _cancelled(request)
        lighting = _timed("lighting", durations, lambda: analyze_lighting(dewarped))
    _cancelled(request)
    return PageProcessingResult(
        image=laid_out,
        diagnostics=PageProcessingDiagnostics(
            orientation=orientation,
            deskew_method=request.deskew_method,
            deskew_angle_degrees=round(float(deskew_angle), 3),
            deskew_selected_method=str(
                deskew_payload.get("selected_method") or request.deskew_method
            ),
            deskew_confidence=round(float(deskew_payload.get("confidence", 0.0)), 3),
            deskew_line_count=int(deskew_payload.get("line_count", 0)),
            deskew_reason=str(deskew_payload.get("reason", "unknown")),
            dewarp=dewarp,
            despeckle=despeckle,
            layout=layout,
            lighting=lighting,
            stage_durations_ms=durations,
            cache_hits=tuple(cache_hits),
        ),
    )
