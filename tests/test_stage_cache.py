from pathlib import Path

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


def test_stage_cache_round_trips_geometry_map_under_same_key(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024, max_entries=4)
    image = np.full((40, 50), 173, dtype=np.uint8)
    key = cache.stage_key(cache.fingerprint_image(image), "dewarp", {"method": "textline"})
    map_x, map_y = np.meshgrid(
        np.arange(50, dtype=np.float32) + 0.25,
        np.arange(40, dtype=np.float32) - 0.5,
    )

    assert cache.put(key, image, {"applied": True}) is True
    assert cache.put_backward_map(key, map_x, map_y) is True
    restored = cache.get_backward_map(key)

    assert restored is not None
    np.testing.assert_array_equal(restored[0], map_x)
    np.testing.assert_array_equal(restored[1], map_y)
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0

    cache.discard(key)
    assert not (cache.root_dir / f"{key}.npz").exists()


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


def test_stage_cache_discard_removes_pair_and_is_idempotent(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    assert cache.put(key, image, {"valid": True}) is True
    image_path = cache.root_dir / f"{key}.png"
    metadata_path = cache.root_dir / f"{key}.json"

    cache.discard(key)
    cache.discard(key)

    assert not image_path.exists()
    assert not metadata_path.exists()


def test_stage_cache_lru_touch_failure_keeps_valid_hit(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    assert cache.put(key, image, {"valid": True}) is True

    def fail_touch(*_args, **_kwargs) -> None:
        raise PermissionError("read-only cache")

    monkeypatch.setattr("uniscan.storage.stage_cache.os.utime", fail_touch)

    restored = cache.get(key)

    assert restored is not None
    np.testing.assert_array_equal(restored[0], image)
    assert restored[1] == {"valid": True}
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0
    assert (cache.root_dir / f"{key}.png").exists()
    assert (cache.root_dir / f"{key}.json").exists()


def test_rejected_cache_entry_stays_a_miss_when_cleanup_is_locked(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    assert cache.put(key, image, {"valid": True}) is True
    monkeypatch.setattr(cache, "_discard_paths", lambda *_args: None)

    cache.reject_hit(key)

    assert cache.get(key) is None
    assert cache.stats.misses == 2
    assert cache.put(key, image, {"valid": "repaired"}) is True
    restored = cache.get(key)
    assert restored is not None
    assert restored[1] == {"valid": "repaired"}


def test_stage_cache_temp_cleanup_failure_is_fail_soft(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    image = np.full((40, 50), 200, dtype=np.uint8)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "otsu"})
    real_unlink = Path.unlink

    def fail_temp_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.endswith(".tmp"):
            raise PermissionError("locked temporary file")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

    assert cache.put(key, image, {"valid": True}) is True
    assert cache.get(key) is not None


def test_stage_cache_writes_encoded_buffer_without_tobytes_copy(tmp_path, monkeypatch) -> None:
    class EncodedBuffer(np.ndarray):
        def tobytes(self, *_args, **_kwargs):
            raise AssertionError("encoded cache buffer must not be copied")

    encoded = np.arange(48, dtype=np.uint8).view(EncodedBuffer)
    monkeypatch.setattr("uniscan.storage.stage_cache.cv2.imencode", lambda *_args: (True, encoded))
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "test"})

    assert cache.put(key, np.zeros((4, 4), dtype=np.uint8), {"valid": True}) is True

    assert (cache.root_dir / f"{key}.png").read_bytes() == bytes(range(48))


def test_stage_cache_refuses_non_finite_metadata(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "cache", max_bytes=1024 * 1024)
    key = cache.stage_key("a" * 64, "cleanup", {"method": "test"})

    assert (
        cache.put(
            key,
            np.zeros((4, 4), dtype=np.uint8),
            {"confidence": float("nan")},
        )
        is False
    )
    assert not list(cache.root_dir.iterdir())


def test_stage_cache_rejects_invalid_limits(tmp_path) -> None:
    with pytest.raises(ValueError, match="1 MiB"):
        ProcessingStageCache(tmp_path / "small", max_bytes=100)
    with pytest.raises(ValueError, match="positive"):
        ProcessingStageCache(tmp_path / "entries", max_entries=0)
