import numpy as np
import pytest

from uniscan.storage.stage_cache import ProcessingStageCache


def test_stage_cache_round_trip_and_stats(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024, max_entries=4)
    image = np.full((80, 120, 3), 173, dtype=np.uint8)
    source = cache.fingerprint_image(image)
    key = cache.stage_key(source, "cleanup", {"method": "otsu"})

    assert cache.get(key) is None
    assert cache.put(key, image, {"applied": True}) is True
    restored = cache.get(key)

    assert restored is not None
    restored_image, metadata = restored
    np.testing.assert_array_equal(restored_image, image)
    assert metadata == {"applied": True}
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.writes == 1


def test_stage_cache_key_invalidates_downstream_options() -> None:
    source = "a" * 64
    first = ProcessingStageCache.stage_key(source, "dewarp", {"method": "none"})
    second = ProcessingStageCache.stage_key(source, "dewarp", {"method": "auto"})
    downstream_first = ProcessingStageCache.stage_key(first, "cleanup", {"method": "otsu"})
    downstream_second = ProcessingStageCache.stage_key(second, "cleanup", {"method": "otsu"})

    assert first != second
    assert downstream_first != downstream_second


def test_stage_cache_prunes_oldest_entries(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024, max_entries=2)
    image = np.full((40, 50), 200, dtype=np.uint8)
    keys = [cache.stage_key("a" * 64, "stage", {"index": index}) for index in range(3)]

    for index, key in enumerate(keys):
        cache.put(key, image, {"index": index})

    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None
    assert cache.stats.evictions == 1


def test_stage_cache_discards_corrupt_metadata(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    cache.put(key, image, {"valid": True})
    metadata_path = cache.root_dir / f"{key}.json"
    metadata_path.write_bytes(b"\xff\xfe broken")

    assert cache.get(key) is None
    assert not (cache.root_dir / f"{key}.png").exists()
    assert not metadata_path.exists()


def test_stage_cache_discards_valid_json_with_wrong_shape(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    cache.put(key, image, {"valid": True})
    metadata_path = cache.root_dir / f"{key}.json"
    metadata_path.write_text("[]", encoding="utf-8")

    assert cache.get(key) is None
    assert not (cache.root_dir / f"{key}.png").exists()
    assert not metadata_path.exists()


def test_stage_cache_discards_oversized_image_before_decode(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    assert cache.put(key, image, {"valid": True}) is True
    image_path = cache.root_dir / f"{key}.png"
    metadata_path = cache.root_dir / f"{key}.json"

    def reject_size(*_args, **_kwargs) -> None:
        raise RuntimeError("test pixel limit exceeded")

    monkeypatch.setattr("uniscan.io.loaders._validated_pixel_count", reject_size)

    assert cache.get(key) is None
    assert not image_path.exists()
    assert not metadata_path.exists()
    assert cache.stats.misses == 1


def test_stage_cache_discards_corrupt_image_and_metadata_pair(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    assert cache.put(key, image, {"valid": True}) is True
    image_path = cache.root_dir / f"{key}.png"
    metadata_path = cache.root_dir / f"{key}.json"
    image_path.write_bytes(b"not a PNG")

    assert cache.get(key) is None
    assert not image_path.exists()
    assert not metadata_path.exists()
    assert cache.stats.misses == 1


def test_stage_cache_rejects_invalid_limits(tmp_path) -> None:
    with pytest.raises(ValueError, match="1 MiB"):
        ProcessingStageCache(tmp_path / "small", max_bytes=100)
    with pytest.raises(ValueError, match="positive"):
        ProcessingStageCache(tmp_path / "entries", max_entries=0)
