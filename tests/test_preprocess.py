import cv2
import numpy as np
import pytest

from uniscan.core.preprocess import (
    DESKEW_METHOD_HOUGH,
    DESKEW_METHOD_MIN_AREA,
    LENS_MODE_CUSTOM,
    LENS_MODE_PROFILES,
    PREPROCESS_PRESETS,
    PreprocessSettings,
    apply_enhancements,
    correct_illumination,
    deskew_document,
    estimate_document_skew,
    infer_lens_mode,
    resolve_lens_mode_profile,
)


def _color_img() -> np.ndarray:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    return img


def test_preprocess_presets_exist() -> None:
    assert "Custom" in PREPROCESS_PRESETS
    assert "Document" in PREPROCESS_PRESETS
    assert "B/W High Contrast" in PREPROCESS_PRESETS


def test_lens_mode_profiles_exist() -> None:
    assert "Document" in LENS_MODE_PROFILES
    assert "Whiteboard" in LENS_MODE_PROFILES
    assert "Photo" in LENS_MODE_PROFILES
    assert "B/W" in LENS_MODE_PROFILES


def test_resolve_lens_mode_profile_handles_custom() -> None:
    assert resolve_lens_mode_profile(LENS_MODE_CUSTOM) is None
    profile = resolve_lens_mode_profile("Document")
    assert profile is not None
    assert profile.preset_name == "Document"
    assert profile.postprocess_name == "Grayscale"


def test_infer_lens_mode_returns_custom_for_non_profile_combo() -> None:
    assert infer_lens_mode("Document", "None") == LENS_MODE_CUSTOM
    assert infer_lens_mode("Photo", "None") == "Photo"


def test_apply_enhancements_keeps_shape_for_color() -> None:
    out = apply_enhancements(
        _color_img(),
        PreprocessSettings(contrast=1.2, brightness=10, denoise=2, apply_threshold=False),
    )
    assert out.shape == (20, 30, 3)


def test_apply_enhancements_threshold_returns_binary() -> None:
    out = apply_enhancements(
        _color_img(),
        PreprocessSettings(
            contrast=1.0, brightness=0, denoise=0, threshold=100, apply_threshold=True
        ),
    )
    assert out.ndim == 2
    unique = set(np.unique(out).tolist())
    assert unique.issubset({0, 255})


def test_correct_illumination_reduces_brightness_gradient() -> None:
    gradient = np.tile(np.linspace(45, 220, 240, dtype=np.uint8), (160, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    corrected = correct_illumination(image)

    before_spread = abs(float(image[:, :40].mean()) - float(image[:, -40:].mean()))
    after_spread = abs(float(corrected[:, :40].mean()) - float(corrected[:, -40:].mean()))
    assert corrected.shape == image.shape
    assert after_spread < before_spread * 0.5


def test_apply_enhancements_can_correct_illumination() -> None:
    gradient = np.tile(np.linspace(60, 200, 120, dtype=np.uint8), (80, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    settings = PreprocessSettings(correct_illumination=True)

    assert not np.array_equal(apply_enhancements(image, settings), image)


def test_deskew_document_returns_angle_for_rotated_content() -> None:
    base = np.full((160, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (40, 60), (180, 100), (0, 0, 0), -1)
    m = cv2.getRotationMatrix2D((110, 80), 17.0, 1.0)
    rotated = cv2.warpAffine(
        base, m, (220, 160), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )

    fixed, angle = deskew_document(rotated)
    assert fixed.shape == rotated.shape
    assert abs(angle) > 1.0


def test_hough_deskew_estimates_rotated_text_lines() -> None:
    base = np.full((420, 620, 3), 255, dtype=np.uint8)
    for y in range(80, 350, 45):
        cv2.line(base, (70, y), (550, y), (0, 0, 0), 5)
    matrix = cv2.getRotationMatrix2D((310, 210), 8.0, 1.0)
    rotated = cv2.warpAffine(
        base,
        matrix,
        (620, 420),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )

    estimate = estimate_document_skew(rotated, method=DESKEW_METHOD_HOUGH)

    assert estimate.line_count >= 3
    assert estimate.confidence > 0.1
    assert estimate.angle_degrees == pytest.approx(-8.0, abs=1.0)


def test_min_area_deskew_and_invalid_method() -> None:
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 70), (200, 110), (0, 0, 0), -1)

    estimate = estimate_document_skew(image, method=DESKEW_METHOD_MIN_AREA)

    assert abs(estimate.angle_degrees) < 0.1
    with pytest.raises(ValueError, match="Unsupported deskew method"):
        estimate_document_skew(image, method="missing")
