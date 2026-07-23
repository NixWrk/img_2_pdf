"""Shadow and illumination correction as a validated pipeline stage.

Geometry is corrected first, then lighting, which is the order the page
processing controller runs. Like the dewarp stage, a candidate has to prove
itself against measurable evidence — here the lighting diagnostics — or the
page is left as it was.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from uniscan.core.cleanup import LightingDiagnostics, analyze_lighting
from uniscan.core.preprocess import correct_illumination

SHADOW_METHOD_NONE = "none"
SHADOW_METHOD_AUTO = "auto"
SHADOW_METHOD_CLASSICAL = "classical"
SHADOW_METHOD_DOCSHADOW = "docshadow"
SHADOW_METHOD_CHOICES = (
    SHADOW_METHOD_NONE,
    SHADOW_METHOD_AUTO,
    SHADOW_METHOD_CLASSICAL,
    SHADOW_METHOD_DOCSHADOW,
)

# Below this there is no shadow worth correcting, and running a model would
# only risk flattening a page that is already evenly lit. Measured on the
# shadow diagnostics rather than on unevenness, which stays high on a clean
# page simply because text is dark.
_MIN_SHADOW_FRACTION = 0.02
# A candidate must leave the page measurably more even than it found it.
_REQUIRED_IMPROVEMENT = 0.9
# Correction must not wash the page out: ink has to stay dark and the ink to
# paper separation has to survive. Glare and clipped-pixel counts are
# deliberately not used here — pushing a shadowed background up towards white
# is the goal of this stage, and those metrics read that as damage.
_MIN_CONTRAST_RATIO = 0.85
_MAX_INK_RISE = 60.0


@dataclass(slots=True, frozen=True)
class ShadowDiagnostics:
    """Explain whether and how the lighting was corrected."""

    method: str
    applied: bool
    selected_method: str = SHADOW_METHOD_NONE
    unevenness_before: float = 0.0
    unevenness_after: float = 0.0
    shadow_before: float = 0.0
    shadow_after: float = 0.0
    glare_after: float = 0.0
    duration_ms: float = 0.0
    reason: str | None = None


def _ink_and_contrast(image: np.ndarray) -> tuple[float, float]:
    """Darkness of the ink and its separation from the paper."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = float(np.percentile(gray, 5))
    paper = float(np.percentile(gray, 95))
    return ink, paper - ink


def _rejection_reason(
    source: np.ndarray,
    candidate: np.ndarray,
    before: LightingDiagnostics,
    after: LightingDiagnostics,
) -> str | None:
    if after.unevenness > before.unevenness * _REQUIRED_IMPROVEMENT:
        return "lighting_not_improved"
    ink_before, contrast_before = _ink_and_contrast(source)
    ink_after, contrast_after = _ink_and_contrast(candidate)
    if contrast_after < contrast_before * _MIN_CONTRAST_RATIO:
        return "contrast_lost"
    if ink_after > ink_before + _MAX_INK_RISE:
        return "ink_washed_out"
    return None


def _docshadow_candidate(image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Run the bundled model, or report why it could not be used."""
    # Imported lazily so a missing optional runtime cannot break module import.
    from uniscan.core import docshadow

    if not docshadow.is_available():
        return None, "docshadow_model_unavailable"
    try:
        return docshadow.remove_shadows(image), None
    except Exception as exc:  # the stage must stay a safe no-op
        return None, f"docshadow_failed:{type(exc).__name__}"


def remove_document_shadows(
    image: np.ndarray,
    *,
    method: str = SHADOW_METHOD_NONE,
) -> tuple[np.ndarray, ShadowDiagnostics]:
    """Even out page lighting without changing geometry or page content."""
    normalized = method.strip().lower()
    if normalized not in SHADOW_METHOD_CHOICES:
        raise ValueError(f"Unsupported shadow removal method: {method}")
    started = time.perf_counter()
    if normalized == SHADOW_METHOD_NONE:
        return image, ShadowDiagnostics(method=normalized, applied=False, reason="disabled")

    before = analyze_lighting(image)

    def finish(
        result: np.ndarray,
        *,
        selected: str,
        after: LightingDiagnostics | None,
        reason: str | None,
    ) -> tuple[np.ndarray, ShadowDiagnostics]:
        measured = after if after is not None else before
        return result, ShadowDiagnostics(
            method=normalized,
            applied=selected != SHADOW_METHOD_NONE,
            selected_method=selected,
            unevenness_before=round(before.unevenness, 6),
            unevenness_after=round(measured.unevenness, 6),
            shadow_before=round(before.shadow_fraction, 6),
            shadow_after=round(measured.shadow_fraction, 6),
            glare_after=round(measured.glare_fraction, 6),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            reason=reason,
        )

    if normalized == SHADOW_METHOD_CLASSICAL:
        candidate = correct_illumination(image)
        return finish(
            candidate,
            selected=SHADOW_METHOD_CLASSICAL,
            after=analyze_lighting(candidate),
            reason=None,
        )

    if normalized == SHADOW_METHOD_DOCSHADOW:
        candidate, failure = _docshadow_candidate(image)
        if candidate is None:
            return finish(image.copy(), selected=SHADOW_METHOD_NONE, after=None, reason=failure)
        return finish(
            candidate,
            selected=SHADOW_METHOD_DOCSHADOW,
            after=analyze_lighting(candidate),
            reason=None,
        )

    # Automatic: only act on a page that measurably needs it, prefer the model,
    # and accept a candidate only when the evidence improves.
    if before.shadow_fraction < _MIN_SHADOW_FRACTION:
        return finish(
            image.copy(),
            selected=SHADOW_METHOD_NONE,
            after=None,
            reason="no_shadow_detected",
        )

    reasons: list[str] = []
    candidate, failure = _docshadow_candidate(image)
    if candidate is not None:
        after = analyze_lighting(candidate)
        rejection = _rejection_reason(image, candidate, before, after)
        if rejection is None:
            return finish(candidate, selected=SHADOW_METHOD_DOCSHADOW, after=after, reason=None)
        reasons.append(f"docshadow_rejected:{rejection}")
    elif failure:
        reasons.append(failure)

    fallback = correct_illumination(image)
    after = analyze_lighting(fallback)
    rejection = _rejection_reason(image, fallback, before, after)
    if rejection is None:
        return finish(
            fallback,
            selected=SHADOW_METHOD_CLASSICAL,
            after=after,
            reason=";".join(reasons) or None,
        )
    reasons.append(f"classical_rejected:{rejection}")
    return finish(
        image.copy(),
        selected=SHADOW_METHOD_NONE,
        after=None,
        reason=";".join(reasons),
    )


__all__ = [
    "SHADOW_METHOD_AUTO",
    "SHADOW_METHOD_CHOICES",
    "SHADOW_METHOD_CLASSICAL",
    "SHADOW_METHOD_DOCSHADOW",
    "SHADOW_METHOD_NONE",
    "ShadowDiagnostics",
    "remove_document_shadows",
]
