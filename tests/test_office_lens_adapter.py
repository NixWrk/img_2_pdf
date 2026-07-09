from __future__ import annotations

import numpy as np

from uniscan.office_lens import (
    CLASSIFIER_MODEL,
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
