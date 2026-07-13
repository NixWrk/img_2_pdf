"""One GUI-independent controller for every post-detection document stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time

import numpy as np

from .cleanup import (
    BINARIZATION_NONE,
    DESPECKLE_NONE,
    DespeckleDiagnostics,
    LightingDiagnostics,
    analyze_lighting,
)
from .dewarp import (
    DEWARP_METHOD_NONE,
    DewarpDiagnostics,
    DewarpModel,
    dewarp_document,
)
from .layout import PAGE_LAYOUT_NONE, PageLayoutDiagnostics, layout_document_page
from .layout import ContentBox
from .orientation import (
    ORIENTATION_METHOD_NONE,
    OrientationDiagnostics,
    orient_document,
)
from .postprocess import POSTPROCESSING_OPTIONS, grayscale
from .preprocess import (
    DESKEW_METHOD_NONE,
    PreprocessSettings,
    apply_enhancements_with_diagnostics,
    deskew_document_with_diagnostics,
)
from uniscan.storage.stage_cache import ProcessingStageCache

CancelCb = Callable[[], bool]
PROCESSING_ALGORITHM_VERSION = 2


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
        if cached is not None:
            cached_image, metadata = cached
            diagnostics = decode_diagnostics(metadata)
            durations[stage] = round((time.perf_counter() - started) * 1000.0, 3)
            cache_hits.append(stage)
            return cached_image, diagnostics, key

    output, diagnostics = operation()
    durations[stage] = round((time.perf_counter() - started) * 1000.0, 3)
    if cache is not None and cacheable:
        cache.put(key, output, encode_diagnostics(diagnostics))
    return output, diagnostics, key


def _layout_from_dict(payload: dict[str, object]) -> PageLayoutDiagnostics:
    values = dict(payload)
    content_box = values.pop("content_box")
    if not isinstance(content_box, dict):
        raise ValueError("Invalid cached content box.")
    return PageLayoutDiagnostics(
        content_box=ContentBox(**content_box),
        **values,
    )


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
        decode_diagnostics=lambda payload: OrientationDiagnostics(**payload),
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
        decode_diagnostics=lambda payload: payload,
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
            decode_diagnostics=lambda payload: DewarpDiagnostics(**payload),
            cacheable=request.dewarp_method != DEWARP_METHOD_NONE,
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
        decode_diagnostics=lambda payload: DespeckleDiagnostics(**payload),
        cacheable=request.postprocess_name != "None" or request.preprocess_settings is not None,
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
        cacheable=request.page_layout != PAGE_LAYOUT_NONE,
        request=request,
        durations=durations,
        cache_hits=cache_hits,
    )
    lighting = None
    if request.lighting_diagnostics:
        lighting = _timed("lighting", durations, lambda: analyze_lighting(dewarped))
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
