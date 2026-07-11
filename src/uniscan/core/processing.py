"""One GUI-independent controller for every post-detection document stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from .cleanup import (
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
from .orientation import (
    ORIENTATION_METHOD_NONE,
    OrientationDiagnostics,
    orient_document,
)
from .postprocess import POSTPROCESSING_OPTIONS
from .preprocess import (
    DESKEW_METHOD_NONE,
    PreprocessSettings,
    apply_enhancements_with_diagnostics,
    deskew_document,
)

CancelCb = Callable[[], bool]


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
    cancel_cb: CancelCb | None = field(default=None, repr=False)


@dataclass(slots=True, frozen=True)
class PageProcessingDiagnostics:
    orientation: OrientationDiagnostics
    deskew_method: str
    deskew_angle_degrees: float
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


def process_document_page(
    image: np.ndarray,
    request: PageProcessingRequest,
) -> PageProcessingResult:
    """Run orientation, deskew, dewarp, cleanup, and layout in the canonical order."""
    if image.size == 0:
        raise ValueError("Cannot process an empty page image.")
    durations: dict[str, float] = {}

    _cancelled(request)
    oriented, orientation = _timed(
        "orientation",
        durations,
        lambda: orient_document(image, method=request.orientation_method),
    )
    _cancelled(request)
    deskewed, deskew_angle = _timed(
        "deskew",
        durations,
        lambda: deskew_document(oriented, method=request.deskew_method),
    )
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
    else:
        dewarped, dewarp = _timed(
            "dewarp",
            durations,
            lambda: dewarp_document(
                deskewed,
                method=request.dewarp_method,
                uvdoc_cache_home=request.uvdoc_cache_home,
                auto_use_uvdoc=request.auto_dewarp_uvdoc,
                model=request.dewarp_model,
            ),
        )

    _cancelled(request)
    if request.postprocess_name not in POSTPROCESSING_OPTIONS:
        raise ValueError(f"Unsupported postprocess mode: {request.postprocess_name}")
    postprocess = POSTPROCESSING_OPTIONS[request.postprocess_name]

    def cleanup_stage():
        postprocessed = postprocess(dewarped)
        if request.preprocess_settings is None:
            return postprocessed, DespeckleDiagnostics(
                strength=DESPECKLE_NONE,
                applied=False,
                reason="disabled",
            )
        return apply_enhancements_with_diagnostics(postprocessed, request.preprocess_settings)

    cleaned, despeckle = _timed("cleanup", durations, cleanup_stage)
    _cancelled(request)
    laid_out, layout = _timed(
        "layout",
        durations,
        lambda: layout_document_page(
            cleaned,
            method=request.page_layout,
            dpi=request.page_dpi,
            margin_mm=request.page_margin_mm,
            horizontal_alignment=request.horizontal_alignment,
            vertical_alignment=request.vertical_alignment,
        ),
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
            dewarp=dewarp,
            despeckle=despeckle,
            layout=layout,
            lighting=lighting,
            stage_durations_ms=durations,
        ),
    )
