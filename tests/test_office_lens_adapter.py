from __future__ import annotations

import numpy as np

from uniscan.office_lens import (
    CLASSIFIER_LABELS,
    CLASSIFIER_MODEL,
    OfficeLensOnnx,
    QUAD_MODEL,
    preprocess_classifier,
    preprocess_quad_mask,
)


def test_office_lens_models_are_packaged() -> None:
    assert QUAD_MODEL.exists()
    assert CLASSIFIER_MODEL.exists()


def test_office_lens_preprocessors_match_model_input_shapes() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    quad_tensor = preprocess_quad_mask(image)
    classifier_tensor = preprocess_classifier(image)

    assert quad_tensor.shape == (1, 256, 256, 3)
    assert quad_tensor.dtype == np.float32
    assert classifier_tensor.shape == (1, 3, 256, 256)
    assert classifier_tensor.dtype == np.float32


def test_office_lens_models_run_inference() -> None:
    image = np.full((320, 240, 3), 245, dtype=np.uint8)
    image[20:-20, 20:-20] = 255

    result = OfficeLensOnnx().process_image(image, mode="document")

    assert result.classification.label in CLASSIFIER_LABELS
    assert result.mask_result.mask.shape == (256, 256)
    assert np.isfinite(result.mask_result.mask).all()
