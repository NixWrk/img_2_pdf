"""Preprocessing presets and enhancement helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessSettings:
    contrast: float = 1.0
    brightness: int = 0
    denoise: int = 0
    threshold: int = 170
    apply_threshold: bool = False


PREPROCESS_PRESETS: dict[str, PreprocessSettings] = {
    "Custom": PreprocessSettings(),
    "Document": PreprocessSettings(contrast=1.25, brightness=10, denoise=4, threshold=170, apply_threshold=False),
    "Whiteboard": PreprocessSettings(contrast=1.35, brightness=20, denoise=5, threshold=185, apply_threshold=False),
    "Photo": PreprocessSettings(contrast=1.05, brightness=0, denoise=2, threshold=170, apply_threshold=False),
    "B/W High Contrast": PreprocessSettings(
        contrast=1.45,
        brightness=8,
        denoise=4,
        threshold=165,
        apply_threshold=True,
    ),
}


def apply_enhancements(image: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    """Apply denoise, contrast/brightness, and optional binary threshold."""
    out = image
    denoise = max(0, int(settings.denoise))
    if denoise > 0:
        if len(out.shape) == 2:
            out = cv2.fastNlMeansDenoising(out, None, h=float(denoise))
        else:
            out = cv2.fastNlMeansDenoisingColored(
                out,
                None,
                h=float(denoise),
                hColor=float(denoise),
                templateWindowSize=7,
                searchWindowSize=21,
            )

    out = cv2.convertScaleAbs(out, alpha=float(settings.contrast), beta=int(settings.brightness))

    if settings.apply_threshold:
        if len(out.shape) == 3:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        else:
            gray = out
        _, out = cv2.threshold(gray, int(settings.threshold), 255, cv2.THRESH_BINARY)

    return out
