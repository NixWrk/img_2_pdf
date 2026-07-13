import cv2
import numpy as np
import pytest

from uniscan.core.dewarp import DewarpDiagnostics
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
    }


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
