import cv2
import numpy as np
import pytest

from uniscan.core.dewarp import (
    DewarpDiagnostics,
    DewarpModel,
    apply_dewarp_model,
    dewarp_model_backward_map,
)
from uniscan.core.geometry import (
    compose_backward_maps,
    perspective_backward_map,
    render_backward_map,
    rotation_backward_map,
    warp_perspective_from_points,
    identity_backward_map,
)
from uniscan.core.preprocess import deskew_document_with_diagnostics
from uniscan.core.postprocess import grayscale
from uniscan.core.preprocess import PreprocessSettings, apply_enhancements
from uniscan.core.processing import (
    PROCESSING_ALGORITHM_VERSION,
    PageProcessingRequest,
    process_document_page,
)
from uniscan.storage.stage_cache import ProcessingStageCache


def _sideways_page() -> np.ndarray:
    image = np.full((720, 520, 3), 255, dtype=np.uint8)
    for index, text in enumerate(
        ("Document page", "quickly aligns", "baseline glyphs", "properly oriented", "local text")
    ):
        cv2.putText(
            image,
            text,
            (35, 110 + index * 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def test_processing_controller_runs_canonical_stages() -> None:
    request = PageProcessingRequest(
        orientation_method="auto",
        deskew_method="none",
        dewarp_method="none",
        postprocess_name="Grayscale",
        preprocess_settings=PreprocessSettings(
            binarization_method="otsu",
            despeckle_strength="conservative",
        ),
        page_layout="a4",
        page_dpi=100,
        page_margin_mm=10,
        lighting_diagnostics=True,
    )

    result = process_document_page(_sideways_page(), request)

    assert result.image.shape == (1169, 827)
    assert set(np.unique(result.image).tolist()).issubset({0, 255})
    assert result.diagnostics.orientation.angle_degrees == 270
    assert result.diagnostics.deskew_selected_method == "none"
    assert result.diagnostics.deskew_confidence == 1.0
    assert result.diagnostics.deskew_reason == "disabled"
    assert result.diagnostics.layout.applied is True
    assert result.diagnostics.lighting is not None
    assert set(result.diagnostics.stage_durations_ms) == {
        "orientation",
        "deskew",
        "dewarp",
        "cleanup",
        "layout",
        "lighting",
        "lighting_diagnostics",
    }


def test_processing_runs_dewarp_before_deskew(monkeypatch) -> None:
    calls: list[str] = []

    def fake_dewarp(source, **_kwargs):
        calls.append("dewarp")
        return source.copy(), DewarpDiagnostics(method="textline", applied=False)

    def fake_deskew(source, **_kwargs):
        from uniscan.core.preprocess import SkewEstimate

        calls.append("deskew")
        return source.copy(), SkewEstimate(0.0, "hybrid", 1.0, reason="test")

    monkeypatch.setattr("uniscan.core.processing.dewarp_document", fake_dewarp)
    monkeypatch.setattr(
        "uniscan.core.processing.deskew_document_with_diagnostics",
        fake_deskew,
    )

    process_document_page(
        np.full((80, 100, 3), 220, dtype=np.uint8),
        PageProcessingRequest(dewarp_method="textline", deskew_method="hybrid"),
    )

    assert calls == ["dewarp", "deskew"]


def test_crop_dewarp_and_deskew_render_authoritative_pixels_once() -> None:
    size = 480
    checker = ((np.indices((size, size)).sum(axis=0) // 3) % 2 * 255).astype(np.uint8)
    source = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    points = ((18.0, 12.0), (461.0, 21.0), (452.0, 466.0), (25.0, 457.0))
    model = DewarpModel(
        method="textline",
        control_points=((0.0, 0.0), (0.5, 0.018), (1.0, 0.0)),
        source="user",
    )
    request = PageProcessingRequest(
        perspective_points=points,
        dewarp_method="textline",
        dewarp_model=model,
        deskew_method="manual",
        deskew_angle_degrees=1.6,
    )

    composed = process_document_page(source, request)
    reference_map = perspective_backward_map(source, np.asarray(points, dtype=np.float32))
    reference_map = compose_backward_maps(
        reference_map,
        dewarp_model_backward_map(reference_map.output_size, model),
    )
    reference_map = compose_backward_maps(
        reference_map,
        rotation_backward_map(reference_map.output_size, 1.6),
    )
    single_pass_reference = render_backward_map(source, reference_map)
    cropped = warp_perspective_from_points(source, np.asarray(points, dtype=np.float32))
    dewarped = apply_dewarp_model(cropped, model)
    chained, _diagnostics = deskew_document_with_diagnostics(
        dewarped,
        method="manual",
        manual_angle_degrees=1.6,
    )

    def sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    assert composed.diagnostics.geometry_resample_count == 1
    assert composed.image.shape == chained.shape
    np.testing.assert_array_equal(composed.image, single_pass_reference)
    assert sharpness(composed.image) >= sharpness(chained) * 1.15


def test_upstream_boundary_map_composes_with_controller_geometry() -> None:
    source = np.zeros((180, 220, 3), dtype=np.uint8)
    for y in range(20, 165, 12):
        cv2.line(source, (18, y), (202, y), (255, 255, 255), 2)
    points = np.float32([[12, 8], [210, 15], [202, 173], [18, 168]])
    upstream = perspective_backward_map(source, points)
    preview = render_backward_map(source, upstream, interpolation=cv2.INTER_LINEAR)
    model = DewarpModel(
        method="textline",
        control_points=((0.0, 0.0), (0.5, 0.015), (1.0, 0.0)),
        source="user",
    )

    result = process_document_page(
        preview,
        PageProcessingRequest(
            geometry_source=source,
            upstream_backward_map=upstream,
            dewarp_method="textline",
            dewarp_model=model,
            deskew_method="manual",
            deskew_angle_degrees=-1.2,
        ),
    )

    expected_map = compose_backward_maps(
        upstream,
        dewarp_model_backward_map(upstream.output_size, model),
    )
    expected_map = compose_backward_maps(
        expected_map,
        rotation_backward_map(expected_map.output_size, -1.2),
    )
    expected = render_backward_map(source, expected_map)
    assert result.diagnostics.geometry_resample_count == 1
    np.testing.assert_array_equal(result.image, expected)


def test_processing_controller_honors_preapplied_dewarp(monkeypatch) -> None:
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    dewarp_calls = 0

    def unexpected_dewarp(*_args, **_kwargs):
        nonlocal dewarp_calls
        dewarp_calls += 1
        raise AssertionError("pre-applied dewarp must not run twice")

    monkeypatch.setattr("uniscan.core.processing.dewarp_document", unexpected_dewarp)
    result = process_document_page(
        image,
        PageProcessingRequest(
            dewarp_method="paddleocr_uvdoc",
            dewarp_already_applied=True,
        ),
    )

    assert result.image is image
    assert result.diagnostics.dewarp.applied is True
    assert result.diagnostics.dewarp.reason == "applied_by_detection_backend"
    assert dewarp_calls == 0


def test_processing_controller_rejects_preapplied_dewarp_with_none_method() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        process_document_page(
            np.full((100, 120, 3), 220, dtype=np.uint8),
            PageProcessingRequest(dewarp_method="none", dewarp_already_applied=True),
        )


def test_black_and_white_with_requested_binarizer_thresholds_once() -> None:
    gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (80, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    settings = PreprocessSettings(
        contrast=1.0,
        brightness=0,
        threshold=100,
        apply_threshold=True,
    )

    result = process_document_page(
        image,
        PageProcessingRequest(
            postprocess_name="Black and White",
            preprocess_settings=settings,
        ),
    )

    expected = apply_enhancements(grayscale(image), settings)
    np.testing.assert_array_equal(result.image, expected)


def test_processing_controller_checks_cancellation_between_stages() -> None:
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        process_document_page(
            np.full((100, 120, 3), 220, dtype=np.uint8),
            PageProcessingRequest(cancel_cb=cancel),
        )


def test_processing_cancellation_after_native_stage_prevents_cache_write(
    tmp_path, monkeypatch
) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    cancelled = False
    from uniscan.core import processing

    real_orient = processing.orient_document

    def orient_then_cancel(*args, **kwargs):
        nonlocal cancelled
        result = real_orient(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(processing, "orient_document", orient_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        process_document_page(
            _sideways_page(),
            PageProcessingRequest(
                orientation_method="auto",
                stage_cache=cache,
                cancel_cb=lambda: cancelled,
            ),
        )

    assert cache.stats.writes == 0


def test_processing_cancellation_during_final_lighting_is_not_lost(monkeypatch) -> None:
    cancelled = False
    from uniscan.core import processing

    real_analyze = processing.analyze_lighting

    def analyze_then_cancel(*args, **kwargs):
        nonlocal cancelled
        result = real_analyze(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(processing, "analyze_lighting", analyze_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        process_document_page(
            np.full((100, 120, 3), 220, dtype=np.uint8),
            PageProcessingRequest(
                lighting_diagnostics=True,
                cancel_cb=lambda: cancelled,
            ),
        )


def test_processing_controller_rejects_unknown_postprocess() -> None:
    with pytest.raises(ValueError, match="Unsupported postprocess"):
        process_document_page(
            np.full((100, 120, 3), 220, dtype=np.uint8),
            PageProcessingRequest(postprocess_name="missing"),
        )


def test_processing_cache_reuses_upstream_and_invalidates_downstream(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = _sideways_page()
    request = PageProcessingRequest(
        orientation_method="auto",
        postprocess_name="Grayscale",
        preprocess_settings=PreprocessSettings(binarization_method="otsu"),
        page_layout="a4",
        page_dpi=100,
        stage_cache=cache,
    )

    first = process_document_page(image, request)
    second = process_document_page(image, request)

    assert first.diagnostics.cache_hits == ()
    assert second.diagnostics.cache_hits == ("orientation", "cleanup", "layout")
    np.testing.assert_array_equal(second.image, first.image)

    request.preprocess_settings = PreprocessSettings(
        contrast=1.2,
        binarization_method="otsu",
    )
    cleanup_changed = process_document_page(image, request)
    assert cleanup_changed.diagnostics.cache_hits == ("orientation",)

    layout_only_request = PageProcessingRequest(
        orientation_method="auto",
        postprocess_name="Grayscale",
        preprocess_settings=PreprocessSettings(binarization_method="otsu"),
        page_layout="a4",
        page_dpi=100,
        page_margin_mm=15,
        stage_cache=cache,
    )
    layout_changed = process_document_page(image, layout_only_request)
    assert layout_changed.diagnostics.cache_hits == ("orientation", "cleanup")


def test_processing_cache_reuses_dewarp_map_without_repeating_inference(
    tmp_path, monkeypatch
) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    calls = 0

    def fake_with_map(source, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            source.copy(),
            DewarpDiagnostics(
                method="textline",
                applied=True,
                selected_method="textline",
            ),
            identity_backward_map((source.shape[1], source.shape[0])),
        )

    monkeypatch.setattr(
        "uniscan.core.processing.dewarp_document_with_map",
        fake_with_map,
    )
    request = PageProcessingRequest(
        dewarp_method="textline",
        deskew_method="manual",
        deskew_angle_degrees=1.0,
        stage_cache=cache,
    )

    first = process_document_page(image, request)
    second = process_document_page(image, request)

    assert calls == 1
    assert second.diagnostics.cache_hits == ("dewarp", "deskew")
    np.testing.assert_array_equal(second.image, first.image)


def test_processing_cache_does_not_reuse_previous_algorithm_version(tmp_path) -> None:
    assert PROCESSING_ALGORITHM_VERSION > 1
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = _sideways_page()
    source_key = cache.fingerprint_image(image)
    old_key = cache.stage_key(
        source_key,
        "orientation",
        {"version": 1, "method": "auto"},
    )
    stale = np.zeros_like(image)
    assert cache.put(
        old_key,
        stale,
        {
            "method": "auto",
            "applied": False,
            "angle_degrees": 0,
            "confidence": 1.0,
            "line_count": 0,
            "reason": "stale-v1-entry",
        },
    )

    result = process_document_page(
        image,
        PageProcessingRequest(
            orientation_method="auto",
            stage_cache=cache,
        ),
    )

    assert "orientation" not in result.diagnostics.cache_hits
    assert result.diagnostics.orientation.reason != "stale-v1-entry"
    assert np.any(result.image != 0)


def test_processing_cache_repairs_semantically_invalid_diagnostics(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = _sideways_page()
    key = cache.stage_key(
        cache.fingerprint_image(image),
        "orientation",
        {"version": PROCESSING_ALGORITHM_VERSION, "method": "auto"},
    )
    assert cache.put(key, image, {"method": "auto"}) is True
    request = PageProcessingRequest(orientation_method="auto", stage_cache=cache)

    repaired = process_document_page(image, request)
    reused = process_document_page(image, request)

    assert "orientation" not in repaired.diagnostics.cache_hits
    assert reused.diagnostics.cache_hits == ("orientation",)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    np.testing.assert_array_equal(reused.image, repaired.image)


def test_processing_cache_repairs_invalid_deskew_diagnostics(tmp_path) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = _sideways_page()
    key = cache.stage_key(
        cache.fingerprint_image(image),
        "deskew",
        {
            "version": PROCESSING_ALGORITHM_VERSION,
            "method": "hybrid",
            "diagnostics_version": 2,
        },
    )
    assert cache.put(key, image, {"method": "hybrid"}) is True
    request = PageProcessingRequest(deskew_method="hybrid", stage_cache=cache)

    repaired = process_document_page(image, request)
    reused = process_document_page(image, request)

    assert "deskew" not in repaired.diagnostics.cache_hits
    assert reused.diagnostics.cache_hits == ("deskew",)
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    np.testing.assert_array_equal(reused.image, repaired.image)


@pytest.mark.parametrize(
    ("decoder_name", "payload"),
    (
        (
            "_orientation_from_dict",
            {"method": "auto", "applied": False, "confidence": float("nan")},
        ),
        (
            "_orientation_from_dict",
            {
                "method": "auto",
                "applied": False,
                "confidence": 10**400,
            },
        ),
        (
            "_dewarp_from_dict",
            {"method": "auto", "applied": False, "max_displacement_px": float("nan")},
        ),
        (
            "_despeckle_from_dict",
            {"strength": "normal", "applied": True, "removed_pixels": -1},
        ),
        (
            "_layout_from_dict",
            {
                "method": "a4",
                "applied": True,
                "content_box": {"x": 0, "y": 0, "width": 10, "height": 10},
                "scale": float("nan"),
            },
        ),
    ),
)
def test_cached_diagnostic_decoders_reject_semantic_corruption(decoder_name, payload) -> None:
    from uniscan.core import processing

    decoder = getattr(processing, decoder_name)

    with pytest.raises(ValueError, match="Invalid cached"):
        decoder(payload)


@pytest.mark.parametrize(
    ("dewarp_method", "auto_dewarp_uvdoc"),
    (("paddleocr_uvdoc", False), ("auto", True)),
)
def test_processing_cache_does_not_reuse_uvdoc_without_model_identity(
    tmp_path, monkeypatch, dewarp_method: str, auto_dewarp_uvdoc: bool
) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    calls = 0

    def model_backed_dewarp(source, *, method, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            np.full_like(source, 200 - calls),
            DewarpDiagnostics(
                method=method,
                applied=True,
                selected_method=method,
            ),
        )

    monkeypatch.setattr("uniscan.core.processing.dewarp_document", model_backed_dewarp)
    request = PageProcessingRequest(
        dewarp_method=dewarp_method,
        auto_dewarp_uvdoc=auto_dewarp_uvdoc,
        postprocess_name="Grayscale",
        page_layout="a4",
        page_dpi=100,
        stage_cache=cache,
    )

    first = process_document_page(image, request)
    second = process_document_page(image, request)

    assert calls == 2
    assert second.diagnostics.cache_hits == ()
    assert not np.array_equal(first.image, second.image)
    assert cache.stats.writes == 0


def test_processing_cache_keys_bundled_uvdoc_by_model_content(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    first_model = tmp_path / "first.onnx"
    second_model = tmp_path / "second.onnx"
    first_model.write_bytes(b"first-uvdoc")
    second_model.write_bytes(b"second-uvdoc")
    monkeypatch.setenv("UNISCAN_UVDOC_MODEL", str(first_model))
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    calls = 0

    def model_backed_dewarp(source, **_kwargs):
        nonlocal calls
        calls += 1
        return source.copy(), DewarpDiagnostics(method="uvdoc", applied=True)

    monkeypatch.setattr("uniscan.core.processing.dewarp_document", model_backed_dewarp)
    request = PageProcessingRequest(dewarp_method="uvdoc", stage_cache=cache)

    process_document_page(image, request)
    cached = process_document_page(image, request)
    monkeypatch.setenv("UNISCAN_UVDOC_MODEL", str(second_model))
    changed = process_document_page(image, request)

    assert calls == 2
    assert "dewarp" in cached.diagnostics.cache_hits
    assert "dewarp" not in changed.diagnostics.cache_hits


def test_processing_cache_keys_docshadow_by_model_content(tmp_path, monkeypatch) -> None:
    cache = ProcessingStageCache(tmp_path / "stages", max_bytes=64 * 1024 * 1024)
    first_model = tmp_path / "first.onnx"
    second_model = tmp_path / "second.onnx"
    first_model.write_bytes(b"first-docshadow")
    second_model.write_bytes(b"second-docshadow")
    monkeypatch.setenv("UNISCAN_DOCSHADOW_MODEL", str(first_model))
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
    calls = 0

    def model_backed_lighting(source, *, method):
        from uniscan.core.lighting import ShadowDiagnostics

        nonlocal calls
        calls += 1
        return source.copy(), ShadowDiagnostics(method=method, applied=True)

    monkeypatch.setattr("uniscan.core.processing.remove_document_shadows", model_backed_lighting)
    request = PageProcessingRequest(shadow_method="docshadow", stage_cache=cache)

    process_document_page(image, request)
    cached = process_document_page(image, request)
    monkeypatch.setenv("UNISCAN_DOCSHADOW_MODEL", str(second_model))
    changed = process_document_page(image, request)

    assert calls == 2
    assert "lighting" in cached.diagnostics.cache_hits
    assert "lighting" not in changed.diagnostics.cache_hits
