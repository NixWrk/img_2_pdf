from __future__ import annotations

import numpy as np
import pytest

from uniscan.office_lens import (
    CLASSIFIER_LABELS,
    CLASSIFIER_MODEL,
    OfficeLensOnnx,
    QUAD_MODEL,
    preprocess_classifier,
    preprocess_quad_mask,
)


class _TensorInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, path: str, providers: list[str], created: list[str]) -> None:
        self.path = path
        self.providers = providers
        created.append(path)

    def get_inputs(self):
        return [_TensorInfo("input")]

    def get_outputs(self):
        return [_TensorInfo("output")]

    def run(self, _outputs, _inputs):
        if self.path.endswith(QUAD_MODEL.name):
            mask = np.zeros((1, 256, 256, 1), dtype=np.float32)
            mask[:, 30:225, 35:220, :] = 0.9
            return [mask]
        return [np.array([[0.8, 0.1, 0.1]], dtype=np.float32)]


class _FakeRuntime:
    def __init__(self, created: list[str]) -> None:
        self.created = created

    def InferenceSession(self, path: str, providers: list[str]):
        return _FakeSession(path, providers, self.created)


def test_office_lens_requires_explicit_byom_directory(monkeypatch) -> None:
    monkeypatch.delenv("UNISCAN_OFFICE_LENS_MODEL_DIR", raising=False)
    monkeypatch.setattr(
        "uniscan.office_lens.adapter._load_onnxruntime",
        lambda: _FakeRuntime([]),
    )

    with pytest.raises(RuntimeError, match="not bundled"):
        OfficeLensOnnx()


def test_office_lens_preprocessors_match_model_input_shapes() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    quad_tensor = preprocess_quad_mask(image)
    classifier_tensor = preprocess_classifier(image)

    assert quad_tensor.shape == (1, 256, 256, 3)
    assert quad_tensor.dtype == np.float32
    assert classifier_tensor.shape == (1, 3, 256, 256)
    assert classifier_tensor.dtype == np.float32


def test_office_lens_sessions_are_fakeable_and_classifier_is_lazy(tmp_path, monkeypatch) -> None:
    quad_model = tmp_path / QUAD_MODEL.name
    classifier_model = tmp_path / CLASSIFIER_MODEL.name
    quad_model.write_bytes(b"licensed quad supplied by user")
    classifier_model.write_bytes(b"licensed classifier supplied by user")
    created: list[str] = []
    monkeypatch.setattr(
        "uniscan.office_lens.adapter._load_onnxruntime",
        lambda: _FakeRuntime(created),
    )
    image = np.full((320, 240, 3), 245, dtype=np.uint8)
    image[20:-20, 20:-20] = 255
    runner = OfficeLensOnnx(
        quad_model=quad_model,
        classifier_model=classifier_model,
    )

    explicit_result = runner.process_image(image, mode="document")

    assert created == [str(quad_model)]
    assert explicit_result.classification.label == "Document"
    assert explicit_result.classification.scores == {}
    assert explicit_result.mask_result.mask.shape == (256, 256)

    auto_result = runner.process_image(image, mode="auto")

    assert created == [str(quad_model), str(classifier_model)]
    assert auto_result.classification.label in CLASSIFIER_LABELS
    assert np.isfinite(auto_result.mask_result.mask).all()
