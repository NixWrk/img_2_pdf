from __future__ import annotations

import hashlib
import io

import pytest

from uniscan import model_assets
from uniscan.model_assets import ModelAsset


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _asset(payload: bytes) -> ModelAsset:
    return ModelAsset(
        name="test",
        filename="test.onnx",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="MIT",
        source="https://example.invalid/source",
        url="https://example.invalid/test.onnx",
    )


def test_download_model_asset_publishes_only_a_verified_file(tmp_path, monkeypatch) -> None:
    payload = b"pinned-model-payload"
    asset = _asset(payload)
    monkeypatch.setattr(model_assets, "model_asset", lambda _name: asset)
    monkeypatch.setattr(model_assets, "urlopen", lambda *_args, **_kwargs: _Response(payload))

    result = model_assets.download_model_asset("test", tmp_path)

    assert result.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))


def test_download_model_asset_keeps_existing_file_on_hash_failure(tmp_path, monkeypatch) -> None:
    expected = b"expected"
    existing = b"old-data"
    asset = _asset(expected)
    destination = tmp_path / asset.filename
    destination.write_bytes(existing)
    monkeypatch.setattr(model_assets, "model_asset", lambda _name: asset)
    monkeypatch.setattr(
        model_assets,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"tampered"),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_assets.download_model_asset("test", tmp_path)

    assert destination.read_bytes() == existing
    assert not list(tmp_path.glob("*.part"))


def test_bundled_uvdoc_assets_match_the_manifest() -> None:
    assert model_assets.verify_model_asset("uvdoc_graph").name == "UVDoc_grid.onnx"
    assert model_assets.verify_model_asset("uvdoc_data").name == "UVDoc_grid.onnx.data"
