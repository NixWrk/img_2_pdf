import cv2
import numpy as np
import pytest

from uniscan.core.cleanup import (
    BINARIZATION_CHOICES,
    analyze_lighting,
    binarize_document,
    despeckle_document,
)


def _uneven_document() -> np.ndarray:
    background = np.tile(np.linspace(135, 245, 420, dtype=np.uint8), (260, 1))
    image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    for y in range(55, 225, 42):
        cv2.putText(
            image,
            "Uneven document line",
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
    return image


@pytest.mark.parametrize("method", BINARIZATION_CHOICES[1:])
def test_binarization_methods_return_binary_grayscale(method) -> None:
    result = binarize_document(_uneven_document(), method=method, window_size=35)

    assert result.ndim == 2
    assert set(np.unique(result).tolist()).issubset({0, 255})
    assert np.count_nonzero(result == 0) > 100
    assert np.count_nonzero(result == 255) > 100


def test_adaptive_binarization_handles_both_sides_of_gradient() -> None:
    image = _uneven_document()

    sauvola = binarize_document(image, method="sauvola", window_size=31)
    wolf = binarize_document(image, method="wolf", window_size=31)

    assert np.count_nonzero(sauvola[:, : image.shape[1] // 2] == 0) > 100
    assert np.count_nonzero(sauvola[:, image.shape[1] // 2 :] == 0) > 100
    assert np.count_nonzero(wolf[:, : image.shape[1] // 2] == 0) > 100
    assert np.count_nonzero(wolf[:, image.shape[1] // 2 :] == 0) > 100


def test_despeckle_removes_isolated_noise_and_preserves_nearby_dot() -> None:
    image = np.full((180, 260), 255, dtype=np.uint8)
    cv2.rectangle(image, (70, 80), (180, 100), 0, -1)
    image[20, 20] = 0
    image[150, 230] = 0
    image[77, 185] = 0  # Punctuation-sized ink close to the text body.

    result, diagnostics = despeckle_document(image, strength="conservative")

    assert result[20, 20] == 255
    assert result[150, 230] == 255
    assert result[77, 185] == 0
    assert diagnostics.removed_components == 2
    assert diagnostics.protected_components >= 1


def test_lighting_diagnostics_separate_shadow_and_glare() -> None:
    image = np.full((320, 480, 3), 210, dtype=np.uint8)
    image[:, :150] = 115
    cv2.circle(image, (360, 150), 25, (255, 255, 255), -1)

    diagnostics = analyze_lighting(image)

    assert diagnostics.shadow_fraction > 0.1
    assert diagnostics.glare_fraction > 0.001
    assert diagnostics.unevenness > 0.2
    assert "uneven_shadow" in diagnostics.warnings
    assert "possible_glare" in diagnostics.warnings


def test_cleanup_validates_methods_and_parameters() -> None:
    image = _uneven_document()
    assert binarize_document(image, method="none") is image
    with pytest.raises(ValueError, match="Unsupported binarization"):
        binarize_document(image, method="missing")
    with pytest.raises(ValueError, match="window"):
        binarize_document(image, method="sauvola", window_size=1)
    with pytest.raises(ValueError, match="Sauvola k"):
        binarize_document(image, method="sauvola", k=2.0)
    with pytest.raises(ValueError, match="Unsupported despeckle"):
        despeckle_document(image, strength="missing")
