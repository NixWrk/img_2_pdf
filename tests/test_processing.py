import cv2
import numpy as np
import pytest

from uniscan.core.preprocess import PreprocessSettings
from uniscan.core.processing import PageProcessingRequest, process_document_page
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


def test_processing_controller_honors_preapplied_dewarp() -> None:
    image = np.full((100, 120, 3), 220, dtype=np.uint8)
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
