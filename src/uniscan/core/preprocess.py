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


@dataclass(slots=True, frozen=True)
class LensModeProfile:
    preset_name: str
    postprocess_name: str


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


LENS_MODE_PROFILES: dict[str, LensModeProfile] = {
    "Document": LensModeProfile(preset_name="Document", postprocess_name="Grayscale"),
    "Whiteboard": LensModeProfile(preset_name="Whiteboard", postprocess_name="Grayscale"),
    "Photo": LensModeProfile(preset_name="Photo", postprocess_name="None"),
    "B/W": LensModeProfile(preset_name="B/W High Contrast", postprocess_name="Black and White"),
}

LENS_MODE_CUSTOM = "Custom"
LENS_MODE_VALUES: tuple[str, ...] = tuple([*LENS_MODE_PROFILES.keys(), LENS_MODE_CUSTOM])


def resolve_lens_mode_profile(mode_name: str) -> LensModeProfile | None:
    if mode_name == LENS_MODE_CUSTOM:
        return None
    return LENS_MODE_PROFILES.get(mode_name)


def infer_lens_mode(preset_name: str, postprocess_name: str) -> str:
    for mode_name, profile in LENS_MODE_PROFILES.items():
        if profile.preset_name == preset_name and profile.postprocess_name == postprocess_name:
            return mode_name
    return LENS_MODE_CUSTOM


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


def deskew_document(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Try to estimate and correct document skew.

    Returns `(deskewed_image, applied_angle_degrees)`.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        return image, 0.0

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)
