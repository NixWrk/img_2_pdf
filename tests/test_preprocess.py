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
    SkewEstimate,
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
    assert profile.postprocess_name == "None"
    grayscale = resolve_lens_mode_profile("Grayscale")
    assert grayscale is not None
    assert grayscale.preset_name == "Document"
    assert grayscale.postprocess_name == "Grayscale"
    whiteboard = resolve_lens_mode_profile("Whiteboard")
    assert whiteboard is not None
    assert whiteboard.postprocess_name == "None"


def test_infer_lens_mode_returns_custom_for_non_profile_combo() -> None:
    assert infer_lens_mode("Document", "None") == "Document"
    assert infer_lens_mode("Document", "Grayscale") == "Grayscale"
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


def test_negative_brightness_clips_instead_of_inverting_dark_tones() -> None:
    image = np.array([[0, 40, 80, 120]], dtype=np.uint8)

    out = apply_enhancements(
        image,
        PreprocessSettings(contrast=1.0, brightness=-80),
    )

    np.testing.assert_array_equal(out, np.array([[0, 0, 0, 40]], dtype=np.uint8))
    assert np.all(np.diff(out.astype(np.int16), axis=1) >= 0)


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


def test_apply_enhancements_supports_adaptive_binarization_and_despeckle() -> None:
    image = np.full((120, 180, 3), 240, dtype=np.uint8)
    cv2.putText(image, "Text", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    image[10, 10] = 0
    settings = PreprocessSettings(
        binarization_method="sauvola",
        binarization_window=31,
        despeckle_strength="conservative",
    )

    result = apply_enhancements(image, settings)

    assert result.ndim == 2
    assert set(np.unique(result).tolist()).issubset({0, 255})
    assert result[10, 10] == 255


def test_deskew_document_returns_angle_for_rotated_content() -> None:
    base = np.full((160, 220, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (40, 60), (180, 100), (0, 0, 0), -1)
    m = cv2.getRotationMatrix2D((110, 80), 17.0, 1.0)
    rotated = cv2.warpAffine(
        base, m, (220, 160), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )

    fixed, angle = deskew_document(rotated)
    assert fixed.shape[0] >= rotated.shape[0]
    assert fixed.shape[1] >= rotated.shape[1]
    assert abs(angle) > 1.0


def test_deskew_expands_canvas_to_preserve_tight_corner_content(monkeypatch) -> None:
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    image[2:22, 2:22] = 0
    image[2:22, -22:-2] = 0
    image[-22:-2, 2:22] = 0
    image[-22:-2, -22:-2] = 0
    dark_pixels_before = int(np.count_nonzero(np.all(image < 64, axis=2)))

    monkeypatch.setattr(
        "uniscan.core.preprocess.estimate_document_skew",
        lambda *_args, **_kwargs: SkewEstimate(
            angle_degrees=20.0,
            method="hybrid",
            confidence=1.0,
            selected_method="hough",
        ),
    )

    fixed, angle = deskew_document(image)

    dark_pixels_after = int(np.count_nonzero(np.all(fixed < 64, axis=2)))
    assert angle == 20.0
    assert fixed.shape[:2] == (149, 185)
    assert dark_pixels_after >= dark_pixels_before * 0.9


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
