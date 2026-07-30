from __future__ import annotations

import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from uniscan.core.pipeline import PageResult
from uniscan.core.processing import PageProcessingRequest, process_document_page
from uniscan.diagnostics import DiagnosticCheck, DiagnosticReport
from uniscan.io.loaders import imwrite_unicode
from uniscan.session import (
    CROP_STATE_APPLIED,
    CROP_STATE_PROPOSED,
    CaptureSession,
    CommittedPageProcessing,
    UnsafeSessionLockError,
    acquire_autosave_lock,
    create_persistent_session,
    load_or_create_session,
)
from uniscan.io.camera_service import CameraMode
from uniscan.storage import PageStore
from uniscan.ui.app import (
    RESOLUTIONS,
    UnifiedScanApp,
    _ApplyPageSnapshot,
    _StagedAppliedPage,
    _add_dewarp_control_point,
    _compose_split_preview,
    _detection_summary,
    _entry_has_crop_proposal,
    _load_import_preferences,
    _save_import_preferences,
    _entry_needs_crop_review,
    _fit_image_to_box,
    _move_dewarp_guide_anchor,
    _move_dewarp_control_point,
    _perspective_source_image,
    _remove_dewarp_control_point,
    _split_at_ratio,
    _split_spread_pair,
    run_app,
)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


def _app_for_processing() -> UnifiedScanApp:
    app = object.__new__(UnifiedScanApp)
    values = {
        "preprocess_preset_var": "Custom",
        "preprocess_contrast_var": 1.0,
        "preprocess_brightness_var": 0,
        "preprocess_denoise_var": 0,
        "preprocess_threshold_var": 170,
        "shadow_method_var": "None",
        "orientation_method_var": "Off",
        "binarization_method_var": "None",
        "binarization_window_var": 31,
        "binarization_k_var": 0.2,
        "despeckle_strength_var": "None",
        "postprocess_var": "Grayscale",
        "lens_mode_var": "Custom",
        "dewarp_method_var": "None",
        "deskew_method_var": "Hybrid (recommended)",
        "manual_deskew_angle_var": 0.0,
        "manual_deskew_summary_var": "Manual deskew: 0.0 degrees",
        "stage_settings_var": "Stage settings: document defaults",
        "page_layout_var": "Keep source page",
        "export_pdf_dpi_var": 300,
        "page_margin_mm_var": 10.0,
        "page_align_x_var": "center",
        "page_align_y_var": "center",
    }
    for name, value in values.items():
        setattr(app, name, _Var(value))
    app._binarization_k_custom = False
    app._loading_page_recipe = False
    app.processing_cache = None
    app.camera = None
    app._last_processing_cache_hits = ()
    return app


def _read_image(path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


def test_import_preferences_round_trip_and_corruption_fails_safe(tmp_path) -> None:
    path = tmp_path / "state" / "import_preferences.json"

    assert _load_import_preferences(path) == (300, False)
    assert _save_import_preferences(path, pdf_dpi=600, split_spreads=True) is True
    assert _load_import_preferences(path) == (600, True)
    assert not list(path.parent.glob(".import_preferences.json.stage-*"))

    path.write_text(
        '{"schemaVersion": 1, "pdfDpi": "600", "splitSpreads": 1}',
        encoding="utf-8",
    )
    assert _load_import_preferences(path) == (300, False)
    path.write_text("not json", encoding="utf-8")
    assert _load_import_preferences(path) == (300, False)
    path.write_bytes(b"\xff\xfe\xfa")
    assert _load_import_preferences(path) == (300, False)


def test_import_preferences_reject_unsafe_values(tmp_path) -> None:
    path = tmp_path / "import_preferences.json"

    assert _save_import_preferences(path, pdf_dpi=71, split_spreads=False) is False
    assert _save_import_preferences(path, pdf_dpi=300, split_spreads=1) is False
    assert path.exists() is False


def test_import_preferences_ignore_stage_cleanup_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "import_preferences.json"
    real_unlink = type(path).unlink

    def fail_stage_cleanup(self, *args, **kwargs):
        if self.name.startswith(".import_preferences.json.stage-"):
            raise OSError("cleanup denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "unlink", fail_stage_cleanup)

    assert _save_import_preferences(path, pdf_dpi=450, split_spreads=False) is True
    assert _load_import_preferences(path) == (450, False)


def test_export_uses_each_pages_committed_current_not_pending_global_settings(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    image = np.zeros((42, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 1] = 100
    image[:, :, 2] = 230
    grayscale_entry = session.add_image(name="grayscale.png", image=image)
    color_entry = session.add_image(name="color.png", image=image)

    app.postprocess_var.set("Grayscale")
    app._reprocess_entry_from_original(grayscale_entry)
    committed_grayscale = grayscale_entry.current_image.copy()
    app.postprocess_var.set("None")
    app._reprocess_entry_from_original(color_entry)
    committed_color = color_entry.current_image.copy()
    assert committed_grayscale.ndim == 2
    assert committed_color.ndim == 3

    # This is only a pending preview setting.  Export must not reapply it to
    # pages that were committed independently with Apply preview.
    app.postprocess_var.set("Black and White")
    snapshot_dir, snapshots = app._snapshot_entries_for_export([grayscale_entry, color_entry])
    # Concurrent UI edits cannot change the generation selected for export.
    grayscale_entry.current_image = np.full_like(committed_grayscale, 255)
    color_entry.current_image = np.zeros_like(committed_color)
    try:
        paths = app._render_export_paths(
            snapshots,
            stage_dir=tmp_path / "export-stage",
            emit=lambda **_kwargs: None,
            is_cancelled=lambda: False,
            job_name="test",
        )
    finally:
        snapshot_dir.cleanup()

    np.testing.assert_array_equal(_read_image(paths[0]), committed_grayscale)
    np.testing.assert_array_equal(_read_image(paths[1]), committed_color)


def test_export_after_restart_uses_durable_applied_current(tmp_path) -> None:
    app = _app_for_processing()
    manifest = tmp_path / "autosave.json"
    session = create_persistent_session(tmp_path)
    image = np.zeros((42, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 15
    image[:, :, 1] = 90
    image[:, :, 2] = 240
    entry = session.add_image(name="applied.png", image=image)
    app.postprocess_var.set("Grayscale")
    app._reprocess_entry_from_original(entry)
    committed = entry.current_image.copy()
    session.save_manifest(manifest)
    session.close(preserve=True)

    restored, was_restored = load_or_create_session(manifest)
    assert was_restored is True
    assert restored.entries[0].committed_processing is not None
    assert restored.entries[0].committed_processing.recipe.postprocess_name == "Grayscale"
    app.postprocess_var.set("None")
    snapshot_dir, snapshots = app._snapshot_entries_for_export(restored.entries)
    try:
        paths = app._render_export_paths(
            snapshots,
            stage_dir=tmp_path / "restart-export-stage",
            emit=lambda **_kwargs: None,
            is_cancelled=lambda: False,
            job_name="test",
        )
    finally:
        snapshot_dir.cleanup()

    np.testing.assert_array_equal(_read_image(paths[0]), committed)


def test_export_renderer_honors_cancellation_before_writing(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="page", image=np.zeros((10, 10, 3), dtype=np.uint8))
    snapshot_dir, snapshots = app._snapshot_entries_for_export([entry])

    try:
        with pytest.raises(RuntimeError, match="Cancelled by user"):
            app._render_export_paths(
                snapshots,
                stage_dir=tmp_path / "export-stage",
                emit=lambda **_kwargs: None,
                is_cancelled=lambda: True,
                job_name="test",
            )
    finally:
        snapshot_dir.cleanup()
    assert not list(tmp_path.glob("*.png"))


def test_export_renderer_rejects_corrupt_committed_snapshot(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="page", image=np.zeros((10, 10, 3), dtype=np.uint8))
    snapshot_dir, snapshots = app._snapshot_entries_for_export([entry])
    snapshots[0].current_path.write_bytes(b"not an image")

    try:
        with pytest.raises(RuntimeError, match="Cannot read committed page"):
            app._render_export_paths(
                snapshots,
                stage_dir=tmp_path / "export-stage",
                emit=lambda **_kwargs: None,
                is_cancelled=lambda: False,
                job_name="test",
            )
    finally:
        snapshot_dir.cleanup()


def test_export_renderer_propagates_bounded_decoder_failure(tmp_path, monkeypatch) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="oversized", image=np.zeros((10, 10, 3), dtype=np.uint8))
    snapshot_dir, snapshots = app._snapshot_entries_for_export([entry])

    def reject(_path):
        raise RuntimeError("safe input limit: 150,000,000 pixels")

    monkeypatch.setattr("uniscan.ui.app.imread_unicode", reject)
    try:
        with pytest.raises(RuntimeError, match="safe input limit"):
            app._render_export_paths(
                snapshots,
                stage_dir=tmp_path / "export-stage",
                emit=lambda **_kwargs: None,
                is_cancelled=lambda: False,
                job_name="test",
            )
    finally:
        snapshot_dir.cleanup()


def test_wolf_uses_controller_default_until_user_overrides_k() -> None:
    app = _app_for_processing()
    app.update_page_preview = lambda: None
    app.binarization_method_var.set("Wolf (uneven light)")

    app._on_binarization_method_change("Wolf (uneven light)")
    settings = app._current_preprocess_settings()

    assert app.binarization_k_var.get() == 0.5
    assert settings.binarization_k is None
    app._on_binarization_k_change(0.35)
    app.binarization_k_var.set(0.35)
    assert app._current_preprocess_settings().binarization_k == 0.35


def test_whiteboard_preset_handler_preserves_colour_and_lens_mode() -> None:
    app = _app_for_processing()
    app.update_page_preview = lambda: None
    app.preprocess_preset_var.set("Whiteboard")
    app.postprocess_var.set("Grayscale")

    app.on_preprocess_preset_change("Whiteboard")

    assert app.postprocess_var.get() == "None"
    assert app.lens_mode_var.get() == "Whiteboard"
    colour = np.zeros((12, 16, 3), dtype=np.uint8)
    colour[:, :, 2] = 220
    assert app._apply_postprocess(colour).ndim == 3


def test_grayscale_lens_mode_is_not_reset_to_document_colour() -> None:
    app = _app_for_processing()
    app.update_page_preview = lambda: None
    statuses: list[str] = []
    app._set_status = statuses.append

    app.on_lens_mode_change("Grayscale")

    assert app.preprocess_preset_var.get() == "Document"
    assert app.postprocess_var.get() == "Grayscale"
    assert app.lens_mode_var.get() == "Grayscale"
    assert statuses == ["Lens mode set to Grayscale."]


def test_geometry_change_replays_committed_stage_recipe_not_pending_controls(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    image = np.zeros((30, 50, 3), dtype=np.uint8)
    image[:, :, 0] = 20
    image[:, :, 1] = 110
    image[:, :, 2] = 230
    entry = session.add_image(name="page", image=image)
    app.postprocess_var.set("Grayscale")
    app._reprocess_entry_from_original(entry)
    previous_committed = entry.committed_processing
    assert previous_committed is not None

    app.postprocess_var.set("Black and White")
    app.page_layout_var.set("A4")
    app.dewarp_method_var.set("Automatic (validated)")
    entry.original_image = cv2.rotate(entry.original_image, cv2.ROTATE_90_CLOCKWISE)
    app._reprocess_after_geometry_change(entry, previous_committed)

    assert entry.current_image.ndim == 2
    assert entry.committed_processing is not None
    recipe = entry.committed_processing.recipe
    assert recipe.postprocess_name == "Grayscale"
    assert recipe.page_layout == "none"
    assert recipe.orientation_method == "none"
    assert recipe.deskew_method == "hybrid"
    assert recipe.dewarp_method == "none"


def test_geometry_change_without_committed_processing_keeps_new_original(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="page", image=np.zeros((20, 30, 3), np.uint8))
    rotated = np.full((30, 20, 3), 123, np.uint8)

    entry.original_image = rotated
    app._reprocess_after_geometry_change(entry, None)

    np.testing.assert_array_equal(entry.current_image, rotated)
    assert entry.committed_processing is None


def test_fast_preview_request_is_lightweight_but_export_stays_full_dpi() -> None:
    app = _app_for_processing()
    app.export_pdf_dpi_var.set(360)

    assert app._processing_request(preview=True).page_dpi == 100
    assert app._processing_request(preview=False).page_dpi == 360


def test_processing_request_keeps_automatic_stages_and_one_manual_override() -> None:
    app = _app_for_processing()
    app.orientation_method_var.set("Automatic (conservative)")
    app.deskew_method_var.set("Manual angle")
    app.manual_deskew_angle_var.set(-1.3)
    app.dewarp_method_var.set("Automatic (validated)")
    app.shadow_method_var.set("Automatic (validated)")

    request = app._processing_request(preview=False)

    assert request.orientation_method == "auto"
    assert request.deskew_method == "manual"
    assert request.deskew_angle_degrees == pytest.approx(-1.3)
    assert request.dewarp_method == "auto"
    assert request.shadow_method == "auto"


def test_selecting_processed_page_loads_its_recipe_before_stage_edit(tmp_path) -> None:
    app = _app_for_processing()
    session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = session.add_image(name="page", image=np.full((30, 40, 3), 180, np.uint8))
    request = PageProcessingRequest(
        orientation_method="auto",
        deskew_method="manual",
        deskew_angle_degrees=1.7,
        dewarp_method="none",
        shadow_method="auto",
        postprocess_name="Grayscale",
    )
    result = process_document_page(entry.original_image, request)
    entry.current_image = result.image
    entry.committed_processing = CommittedPageProcessing.from_result(
        request,
        result.diagnostics,
        result.image,
    )
    app.session = session
    app.page_listbox = SimpleNamespace(curselection=lambda: (0,))
    app.preprocess_contrast_var.set(1.8)
    app.preprocess_brightness_var.set(40)
    app.preprocess_denoise_var.set(7)

    app._sync_controls_from_single_committed_page()

    assert app.orientation_method_var.get() == "Automatic (conservative)"
    assert app.deskew_method_var.get() == "Manual angle"
    assert app.manual_deskew_angle_var.get() == pytest.approx(1.7)
    assert app.shadow_method_var.get() == "Automatic (validated)"
    assert app.postprocess_var.get() == "Grayscale"
    assert app.preprocess_contrast_var.get() == pytest.approx(1.0)
    assert app.preprocess_brightness_var.get() == 0
    assert app.preprocess_denoise_var.get() == 0
    assert "loaded from page" in app.stage_settings_var.get()


def test_restored_uvdoc_page_with_dewarp_none_is_identity(tmp_path) -> None:
    app = _app_for_processing()
    app.dewarp_method_var.set("None")
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image_with_contour(
        name="uvdoc",
        raw_image=np.zeros((20, 30, 3), dtype=np.uint8),
        warped_image=np.full((18, 28, 3), 200, dtype=np.uint8),
        contour=None,
        backend="paddleocr_uvdoc",
    )

    request = app._processing_request(entry=entry, preview=False)

    assert request.dewarp_already_applied is False
    assert app._process_review_page(entry.original_image, entry=entry, preview=False).image.shape[
        :2
    ] == (18, 28)


def test_capture_frame_disables_hidden_auto_split(monkeypatch) -> None:
    app = _app_for_processing()

    class _ForbiddenTkVar:
        def get(self):
            raise AssertionError("Tk variable was read from worker code")

    app.import_two_page_mode_var = _ForbiddenTkVar()
    seen: list[bool] = []

    def fake_process(_items, *, options):
        seen.append(options.two_page_mode)
        return []

    monkeypatch.setattr("uniscan.ui.app.process_loaded_items", fake_process)

    assert app._process_capture_frame(np.zeros((10, 10, 3), dtype=np.uint8), "burst") == []
    assert seen == [False]


def test_ingest_keeps_detected_crop_as_uncommitted_proposal(tmp_path) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    raw = np.full((12, 16, 3), 25, dtype=np.uint8)
    warped = np.full((8, 10, 3), 220, dtype=np.uint8)
    contour = np.float32([[1, 1], [14, 1], [14, 10], [1, 10]])

    app._ingest_page_results(
        [PageResult("page", raw, warped, warped, contour, "cv_hybrid", True, None)]
    )

    entry = app.session.entries[0]
    np.testing.assert_array_equal(entry.raw_image, raw)
    np.testing.assert_array_equal(entry.original_image, raw)
    np.testing.assert_array_equal(entry.current_image, raw)
    np.testing.assert_array_equal(entry.detected_contour, contour)
    assert entry.detected_backend == "cv_hybrid"
    assert entry.crop_state == CROP_STATE_PROPOSED
    assert _entry_has_crop_proposal(entry) is True


def test_crop_proposal_previews_then_commits_only_through_apply(tmp_path) -> None:
    app = _app_for_processing()
    app.postprocess_var.set("None")
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    raw = np.zeros((14, 18, 3), dtype=np.uint8)
    raw[2:12, 3:15] = (40, 120, 220)
    contour = np.float32([[3, 2], [14, 2], [14, 11], [3, 11]])
    app._ingest_page_results(
        [PageResult("page", raw, raw[2:12, 3:15], raw, contour, "cv_hybrid", True, None)]
    )
    entry = app.session.entries[0]
    expected = app._review_after_image(entry, raw)

    # Previewing a proposal leaves every durable image on the raw generation.
    np.testing.assert_array_equal(entry.original_image, raw)
    np.testing.assert_array_equal(entry.current_image, raw)

    app._apply_perspective_crop(
        entry,
        source_image=entry.raw_image,
        points=entry.detected_contour,
        backend=entry.detected_backend,
    )

    np.testing.assert_array_equal(entry.original_image, expected)
    np.testing.assert_array_equal(entry.current_image, expected)
    np.testing.assert_array_equal(entry.detected_contour, contour)
    assert entry.detected_backend == "cv_hybrid"
    assert entry.crop_state == CROP_STATE_APPLIED
    assert _entry_has_crop_proposal(entry) is False


def test_crop_apply_all_rolls_back_every_page_when_later_commit_fails(
    tmp_path, monkeypatch
) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    raw_images = [np.full((14, 18, 3), value, dtype=np.uint8) for value in (30, 90)]
    contour = np.float32([[2, 2], [15, 2], [15, 11], [2, 11]])
    for index, raw in enumerate(raw_images):
        app._ingest_page_results(
            [PageResult(str(index), raw, raw, raw, contour, "cv_hybrid", True, None)]
        )
    entries = list(app.session.entries)
    snapshots = [
        (
            entry.original_image.copy(),
            entry.current_image.copy(),
            entry.detected_contour.copy(),
            entry.detected_backend,
            entry.crop_state,
            entry.revision,
        )
        for entry in entries
    ]
    real_replace = app.session.replace_entry_image
    failure_pending = True

    def fail_second_once(entry_id, **kwargs):
        nonlocal failure_pending
        if entry_id == entries[1].entry_id and failure_pending:
            failure_pending = False
            raise RuntimeError("simulated second-page commit failure")
        return real_replace(entry_id, **kwargs)

    monkeypatch.setattr(app.session, "replace_entry_image", fail_second_once)
    proposals = [
        (entry, entry.raw_image, entry.detected_contour, entry.detected_backend)
        for entry in entries
    ]

    with pytest.raises(RuntimeError, match="second-page commit failure"):
        app._apply_perspective_crops(proposals)

    for entry, snapshot in zip(entries, snapshots, strict=True):
        original, current, old_contour, backend, crop_state, revision = snapshot
        np.testing.assert_array_equal(entry.original_image, original)
        np.testing.assert_array_equal(entry.current_image, current)
        np.testing.assert_array_equal(entry.detected_contour, old_contour)
        assert entry.detected_backend == backend
        assert entry.crop_state == crop_state
        assert entry.revision == revision


def test_failed_detection_does_not_leave_success_backend_status(tmp_path) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    raw = np.zeros((8, 9, 3), dtype=np.uint8)

    app._ingest_page_results(
        [PageResult("page", raw, raw, raw, None, "cv_hybrid", False, "no boundary")]
    )

    entry = app.session.entries[0]
    assert entry.detected_backend is None
    assert entry.crop_state != CROP_STATE_PROPOSED
    assert _entry_needs_crop_review(entry) is True
    assert (
        _entry_needs_crop_review(
            SimpleNamespace(detected_contour=None, detected_backend="intentional_split")
        )
        is False
    )


def test_preview_error_is_explicit_and_stale_generation_is_ignored() -> None:
    app = _app_for_processing()
    app._closing = False
    app.review_preview_generation = 2
    app.review_preview_threads = []
    app.page_preview_after_photo = "old-photo"
    statuses: list[str] = []
    app._set_status = statuses.append

    class _Inner:
        def configure(self, **_kwargs):
            pass

    class _Label:
        _label = _Inner()

        def __init__(self):
            self.text = "old"

        def configure(self, **kwargs):
            self.text = kwargs.get("text", self.text)

    app.page_preview_after_label = _Label()

    app._handle_review_preview_result(1, None, None, "stale failure")
    assert app.page_preview_after_label.text == "old"
    assert app.page_preview_after_photo == "old-photo"

    app._handle_review_preview_result(2, None, None, "new failure")
    assert app.page_preview_after_label.text == "Preview failed: new failure"
    assert app.page_preview_after_photo is None
    assert statuses == ["Preview failed: new failure"]


def test_preview_reports_applied_and_rejected_wave_correction() -> None:
    app = _app_for_processing()
    app._closing = False
    app.review_preview_generation = 1
    app.review_preview_threads = []
    app.page_preview_after_photo = None
    app.page_preview_after_label = SimpleNamespace(configure=lambda **_kwargs: None)
    app.geometry_summary_var = _Var("Wave preview: pending")
    app._to_ctk_photo_for_label = lambda _image, _label: "preview-photo"
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    applied = SimpleNamespace(
        dewarp=SimpleNamespace(
            applied=True,
            selected_method="textline",
            line_count=7,
            max_displacement_px=4.25,
            reason=None,
        )
    )
    app._handle_review_preview_result(1, image, applied, None)
    assert app.geometry_summary_var.get() == "Wave preview: textline, 7 lines, 4.2px"

    rejected = SimpleNamespace(
        dewarp=SimpleNamespace(
            applied=False,
            selected_method="none",
            line_count=0,
            max_displacement_px=0.0,
            reason="insufficient_line_evidence",
        )
    )
    app._handle_review_preview_result(1, image, rejected, None)
    assert app.geometry_summary_var.get() == ("Wave preview unchanged: insufficient line evidence")


def test_gui_spread_split_replays_warped_ratio_on_raw(monkeypatch) -> None:
    raw = np.zeros((20, 1000, 3), dtype=np.uint8)
    warped = np.zeros((20, 800, 3), dtype=np.uint8)
    calls: list[np.ndarray] = []

    def fake_split(image, *, fallback):
        assert fallback == "none"
        calls.append(image)
        return [image[:, :480], image[:, 480:]]

    monkeypatch.setattr("uniscan.ui.app.split_spread_accurate", fake_split)
    pair = _split_spread_pair(raw, warped)

    assert pair is not None
    raw_halves, warped_halves = pair
    assert len(calls) == 1 and calls[0] is warped
    assert [part.shape[1] for part in warped_halves] == [480, 320]
    assert [part.shape[1] for part in raw_halves] == [600, 400]


def test_split_preview_composes_two_pages_without_changing_source() -> None:
    source = np.full((100, 240, 3), 180, np.uint8)
    left, right = _split_at_ratio(source, 0.4)

    composed = _compose_split_preview(left, right)

    assert source.shape == (100, 240, 3)
    assert left.shape == (100, 96, 3)
    assert right.shape == (100, 144, 3)
    assert composed.shape[0] == 100
    assert composed.shape[1] > source.shape[1]
    assert np.all(composed[:, 96:104] == 48)


def test_fit_image_to_box_only_resizes_the_presentation_copy() -> None:
    source = np.arange(120 * 240 * 3, dtype=np.uint8).reshape(120, 240, 3)

    fitted = _fit_image_to_box(source, 80, 80)

    assert fitted.shape == (40, 80, 3)
    assert source.shape == (120, 240, 3)


def test_second_perspective_pass_uses_corrected_page_geometry() -> None:
    raw = np.zeros((80, 120, 3), np.uint8)
    corrected = np.zeros((60, 90, 3), np.uint8)
    entry = SimpleNamespace(raw_image=raw, original_image=corrected)

    assert _perspective_source_image(entry, from_current_geometry=False) is raw
    assert _perspective_source_image(entry, from_current_geometry=True) is corrected


def test_import_sources_accept_files_and_multiple_folders_without_duplicates(tmp_path) -> None:
    first_folder = tmp_path / "first"
    second_folder = tmp_path / "second"
    first_folder.mkdir()
    second_folder.mkdir()
    first_image = first_folder / "page1.png"
    second_pdf = second_folder / "book.pdf"
    first_image.write_bytes(b"image")
    second_pdf.write_bytes(b"pdf")
    (first_folder / "ignore.txt").write_text("ignore", encoding="utf-8")

    sources = UnifiedScanApp._expand_import_sources([first_folder, second_folder, first_image])

    assert sources == [first_image, second_pdf]


def test_dewarp_points_and_visual_guide_can_be_moved_independently() -> None:
    points = [(0.0, 0.0), (0.5, 0.02), (1.0, 0.0)]

    added = _add_dewarp_control_point(points, 0.25, -0.05)
    assert added == 1
    assert points[1] == (0.25, -0.05)

    _move_dewarp_control_point(points, added, 0.4, 0.1)
    assert points[1] == (0.4, 0.1)
    _move_dewarp_control_point(points, added, 0.9, 0.5)
    assert points[1][0] < points[2][0]
    assert points[1][1] == 0.24

    before = list(points)
    anchor = _move_dewarp_guide_anchor(points, 0.5, -0.48)
    assert anchor == pytest.approx(0.02)
    assert points == before

    top_anchor = _move_dewarp_guide_anchor([(0.0, -0.08), (1.0, 0.04)], 0.5, -1.0)
    bottom_anchor = _move_dewarp_guide_anchor([(0.0, -0.08), (1.0, 0.04)], 0.5, 1.0)
    assert top_anchor == pytest.approx(0.08)
    assert bottom_anchor == pytest.approx(0.96)

    assert _remove_dewarp_control_point(points, 1) is True
    assert len(points) == 3
    assert _remove_dewarp_control_point(points, 1) is False


def test_committing_previewed_split_creates_adjacent_pages_with_prior_appearance(tmp_path) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    image = np.zeros((60, 100, 3), np.uint8)
    image[:, :, 1] = 120
    image[:, :, 2] = 230
    entry = app.session.add_image(name="spread", image=image)
    app.postprocess_var.set("Grayscale")
    app._reprocess_entry_from_original(entry)

    right_entry = app._commit_entry_split(0, entry, 0.45)

    assert [item.name for item in app.session.entries] == ["spread [L]", "spread [R]"]
    assert entry.original_image.shape[1] == 45
    assert right_entry.original_image.shape[1] == 55
    assert entry.current_image.ndim == 2
    assert right_entry.current_image.ndim == 2
    assert entry.committed_processing.recipe.postprocess_name == "Grayscale"
    assert right_entry.committed_processing.recipe.postprocess_name == "Grayscale"

    assert entry.detected_backend == "intentional_split"
    assert right_entry.detected_backend == "intentional_split"
    assert _entry_needs_crop_review(entry) is False
    assert _entry_needs_crop_review(right_entry) is False


def test_drag_captures_selected_page_ids_before_listbox_changes_selection() -> None:
    app = object.__new__(UnifiedScanApp)
    app.session = SimpleNamespace(
        entries=[
            SimpleNamespace(entry_id="a"),
            SimpleNamespace(entry_id="b"),
            SimpleNamespace(entry_id="c"),
        ]
    )
    app._page_index_at_y = lambda _y: 1
    app._selected_entry_indices = lambda: [0, 1]

    app._on_page_drag_start(SimpleNamespace(y=25))

    assert app.page_drag_state["entry_ids"] == ("a", "b")


def test_detection_summary_never_calls_fallback_detected() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    fallback = PageResult("p", image, image, image, None, "cv_hybrid", False, "no candidate")
    detected = PageResult("q", image, image, image, None, "cv_hybrid", True, None)

    fallback_text = _detection_summary([fallback])
    mixed_text = _detection_summary([detected, fallback])
    assert "detected boundaries" not in fallback_text
    assert "unchanged" in fallback_text
    assert "detected 1" in mixed_text and "fallback" in mixed_text


def test_pdf_layout_dpi_mismatch_is_rejected(tmp_path) -> None:
    app = _app_for_processing()
    app.page_layout_var.set("A4")
    app.export_pdf_dpi_var.set(300)
    session = CaptureSession(store=PageStore(root_dir=tmp_path))
    entry = session.add_image(name="a4", image=np.full((20, 30, 3), 200, np.uint8))
    app._reprocess_entry_from_original(entry)

    app._validate_pdf_layout_dpi([entry], 300)
    with pytest.raises(RuntimeError, match="physical A4/Letter size"):
        app._validate_pdf_layout_dpi([entry], 200)


def test_quick_export_preserves_scope_and_dpi_while_requesting_path() -> None:
    app = _app_for_processing()
    app.session = SimpleNamespace(entries=[object()])
    app.export_scope_var = _Var("Selected pages")
    app.export_pdf_dpi_var.set(420)
    app.export_pdf_path_var = _Var("old-output.pdf")
    observed: list[tuple[str, int, str]] = []
    app.export_to_pdf = lambda: observed.append(
        (
            app.export_scope_var.get(),
            app.export_pdf_dpi_var.get(),
            app.export_pdf_path_var.get(),
        )
    )

    app.quick_export_pdf()

    assert observed == [("Selected pages", 420, "")]


def test_burst_releases_existing_handle_and_commits_staged_raw_frames(
    tmp_path, monkeypatch
) -> None:
    app = _app_for_processing()
    app.camera_shots_var = _Var(2)
    app.camera_delay_var = _Var(0.0)
    app.camera_index_var = _Var(3)
    app.camera_resolution = (640, 480)
    app.preview_job = None
    app.stop_preview = lambda: None
    app._update_camera_health = lambda: None
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    app.refresh_page_list = lambda keep_index=None: None
    app.go_to_review_tab = lambda: None
    statuses: list[str] = []
    app._set_status = statuses.append

    class PersistentCamera:
        released = False

        def release(self) -> None:
            self.released = True

    persistent = PersistentCamera()
    app.camera = persistent
    frames = [
        np.full((8, 9, 3), 11, dtype=np.uint8),
        np.full((8, 9, 3), 22, dtype=np.uint8),
    ]
    created = []

    class FakeBurstCamera:
        MAX_BURST_SHOTS = 20

        def __init__(self, *, index, resolution):
            assert persistent.released is True
            assert index == 3
            assert resolution == (640, 480)
            self.released = False
            created.append(self)

        def open(self) -> None:
            pass

        def iter_burst(
            self,
            *,
            shots,
            delay_sec,
            cancel_cb,
            on_progress,
            first_frame,
        ):
            # Legacy path: no live stream, so there is no press-time frame.
            assert first_frame is None
            assert shots == 2
            assert delay_sec == 0.0
            assert cancel_cb() is False
            for index, frame in enumerate(frames, start=1):
                on_progress(index, shots)
                yield index, frame

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr("uniscan.ui.app.CameraService", FakeBurstCamera)

    def process_frame(frame, base_name):
        assert created[-1].released is True
        warped = np.full_like(frame, 255)
        contour = np.float32([[0, 0], [8, 0], [8, 7], [0, 7]])
        return [PageResult(base_name, frame, warped, warped, contour, "cv_hybrid", True, None)]

    app._process_capture_frame = process_frame

    def run_job(_name, worker, on_done, *, on_error=None):
        try:
            payload = worker(lambda **_kwargs: None, lambda: False)
            on_done(payload)
        except Exception:
            if on_error is not None:
                on_error()
            raise
        return True

    app._start_background_job = run_job

    app.capture_burst()

    assert app.camera is None
    assert app.burst_camera is None
    assert app._burst_is_active() is False
    assert created[0].released is True
    assert len(app.session.entries) == 2
    np.testing.assert_array_equal(app.session.entries[0].original_image, frames[0])
    np.testing.assert_array_equal(app.session.entries[1].current_image, frames[1])
    assert statuses[-1].startswith("Burst captured 2 page(s)")


def test_closing_camera_cancels_and_releases_active_burst() -> None:
    app = object.__new__(UnifiedScanApp)
    app._camera_state_lock = __import__("threading").RLock()
    app._burst_capture_active = True
    app.job_cancel_event = __import__("threading").Event()
    app.camera = None
    app.preview_job = None
    app.stop_preview = lambda: None
    app._update_camera_health = lambda: None
    app._set_status = lambda _text: None

    class ActiveCamera:
        released = False

        def release(self) -> None:
            self.released = True

    active = ActiveCamera()
    app.burst_camera = active

    app.close_camera()

    assert app.job_cancel_event.is_set()
    assert active.released is True
    assert app.burst_camera is active
    assert app._burst_is_active() is True


def test_burst_reuses_streaming_camera_and_keeps_preview(tmp_path) -> None:
    app = _app_for_processing()
    app.camera_shots_var = _Var(2)
    app.camera_delay_var = _Var(0.0)
    app.camera_index_var = _Var(3)
    app.camera_resolution = (640, 480)
    app.preview_job = "tick"
    stop_calls: list[bool] = []
    app.stop_preview = lambda: stop_calls.append(True)
    app._update_camera_health = lambda: None
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    app.refresh_page_list = lambda keep_index=None: None
    statuses: list[str] = []
    app._set_status = statuses.append

    frames = [
        np.full((8, 9, 3), 11, dtype=np.uint8),
        np.full((8, 9, 3), 22, dtype=np.uint8),
    ]

    class StreamingCamera:
        index = 3
        resolution = (640, 480)
        released = False

        def is_streaming(self) -> bool:
            return True

        def latest_frame(self):
            return frames[0]

        def iter_burst(self, *, shots, delay_sec, cancel_cb, on_progress, first_frame):
            assert shots == 2
            assert delay_sec == 0.0
            assert cancel_cb() is False
            # The shutter frame is grabbed on the UI thread at press time.
            assert first_frame is frames[0]
            for frame_index, frame in enumerate(frames, start=1):
                on_progress(frame_index, shots)
                yield frame_index, frame

        def release(self) -> None:
            self.released = True

    camera = StreamingCamera()
    app.camera = camera

    def process_frame(frame, base_name):
        warped = np.full_like(frame, 255)
        contour = np.float32([[0, 0], [8, 0], [8, 7], [0, 7]])
        return [PageResult(base_name, frame, warped, warped, contour, "cv_hybrid", True, None)]

    app._process_capture_frame = process_frame

    def run_job(_name, worker, on_done, *, on_error=None):
        try:
            payload = worker(lambda **_kwargs: None, lambda: False)
            on_done(payload)
        except Exception:
            if on_error is not None:
                on_error()
            raise
        return True

    app._start_background_job = run_job

    app.capture_burst()

    # The open streaming camera is reused: never released, preview untouched.
    assert app.camera is camera
    assert camera.released is False
    assert stop_calls == []
    assert app.burst_camera is None
    assert app._burst_is_active() is False
    assert len(app.session.entries) == 2
    np.testing.assert_array_equal(app.session.entries[0].original_image, frames[0])
    assert statuses[-1].startswith("Burst captured 2 page(s)")


def test_shared_burst_camera_requires_matching_streaming_camera() -> None:
    app = _app_for_processing()

    class Cam:
        def __init__(self, index, resolution, *, streaming=True):
            self.index = index
            self.resolution = resolution
            self._streaming = streaming

        def is_streaming(self) -> bool:
            return self._streaming

    app.camera = None
    assert app._shared_burst_camera(3, (640, 480)) is None
    app.camera = Cam(2, (640, 480))
    assert app._shared_burst_camera(3, (640, 480)) is None
    app.camera = Cam(3, (1280, 720))
    assert app._shared_burst_camera(3, (640, 480)) is None
    app.camera = Cam(3, (640, 480), streaming=False)
    assert app._shared_burst_camera(3, (640, 480)) is None
    match = Cam(3, (640, 480))
    app.camera = match
    assert app._shared_burst_camera(3, (640, 480)) is match


def test_grab_still_frame_takes_the_on_screen_frame() -> None:
    app = _app_for_processing()
    on_screen = np.full((4, 4, 3), 7, dtype=np.uint8)

    class StreamCam:
        def latest_frame(self):
            return on_screen

        def read_frame(self):
            raise AssertionError("a live stream must supply the shot directly")

    # No waiting and no device reconfiguration: the shot is what was on screen.
    assert app._grab_still_frame(StreamCam()) is on_screen

    stale = np.full((4, 4, 3), 3, dtype=np.uint8)

    class DirectCam:
        def latest_frame(self):
            return None

        def read_frame(self):
            return stale

    assert app._grab_still_frame(DirectCam()) is stale


def test_device_menu_lists_system_names_and_resolves_them_to_indices() -> None:
    app = _app_for_processing()
    app.camera_index_var = _Var(1)
    app.camera_device_names = ["c922 Pro Stream Webcam", "Integrated Webcam"]
    app.camera_device_indices = []

    assert app._device_menu_values() == ["c922 Pro Stream Webcam", "Integrated Webcam"]
    assert app._device_menu_selection() == "Integrated Webcam"
    assert app._index_for_device_label("c922 Pro Stream Webcam") == 0
    assert app._index_for_device_label("Integrated Webcam") == 1
    assert app._index_for_device_label("Not a device") is None

    # Selecting by label drives the same path as selecting by index.
    selected: list[str] = []
    app._on_camera_index_selected = selected.append
    app._on_camera_device_selected("c922 Pro Stream Webcam")
    app._on_camera_device_selected("Not a device")
    assert selected == ["0"]


def test_device_menu_falls_back_to_indices_without_system_names() -> None:
    app = _app_for_processing()
    app.camera_index_var = _Var(0)
    app.camera_device_names = []
    app.camera_device_indices = []

    values = app._device_menu_values()
    assert values[:3] == ["Camera 0", "Camera 1", "Camera 2"]
    assert app._device_menu_selection() == "Camera 0"
    assert app._index_for_device_label("Camera 2") == 2


def test_device_menu_uses_probed_indices_and_disambiguates_equal_names() -> None:
    app = _app_for_processing()
    app.camera_index_var = _Var(2)
    app.camera_device_names = ["USB Camera", "USB Camera", "USB Camera"]
    # Only two of the three enumerated devices actually opened.
    app.camera_device_indices = [0, 2]

    assert app._device_menu_values() == ["USB Camera", "USB Camera (2)"]
    assert app._device_menu_selection() == "USB Camera (2)"
    assert app._index_for_device_label("USB Camera (2)") == 2
    assert app._index_for_device_label("USB Camera") == 0


def test_camera_modes_round_trip_through_the_cache(tmp_path) -> None:
    app = _app_for_processing()
    app.autosave_path = tmp_path / "state" / "autosave.json"
    modes = [
        CameraMode(requested=(3264, 2448), granted=(2304, 1536), fps=2.0),
        CameraMode(requested=(1920, 1080), granted=(1920, 1080), fps=30.0),
    ]

    app._save_camera_modes(0, modes)
    app._save_camera_modes(1, [CameraMode((640, 480), (640, 480), 30.0)])

    assert app._load_camera_modes(0) == modes
    assert app._load_camera_modes(1)[0].granted == (640, 480)
    assert app._load_camera_modes(7) == []  # unmeasured device

    # A cached device selects its best real-time mode without re-probing.
    app.camera_resolution = (3264, 2448)
    app._camera_resolution_chosen = False
    assert app._apply_cached_camera_modes(0) is True
    assert app.camera_resolution == (1920, 1080)

    # An explicit user choice survives the cache being applied again.
    app.camera_resolution = (2304, 1536)
    app._camera_resolution_chosen = True
    assert app._apply_cached_camera_modes(0) is True
    assert app.camera_resolution == (2304, 1536)


def test_failed_mode_detection_opens_once_instead_of_looping() -> None:
    app = _app_for_processing()
    app.camera_index_var = _Var(0)
    app.camera_modes = []
    app._camera_modes_probed_index = None
    app._camera_opening = False
    app._camera_modes_probing = False
    app._burst_is_active = lambda: False
    app._apply_cached_camera_modes = lambda _index: False
    app._set_status = lambda _text: None

    detections: list[object] = []

    def fake_detect(*, on_done=None) -> None:
        detections.append(on_done)
        # A device that cannot be measured still records the attempt.
        app._camera_modes_probed_index = 0
        if on_done is not None:
            on_done()

    app._detect_camera_modes_async = fake_detect
    opened: list[bool] = []
    app._camera_open_generation = 0
    app._update_camera_health = lambda *_a, **_k: None
    app._show_preview_placeholder = lambda _text: None
    app.after = lambda _delay, _cb: opened.append(True) or "job"
    app._max_camera_resolution = lambda: (1920, 1080)

    app._open_camera_async()

    # Detection is attempted once; the retry then proceeds to open the device.
    assert len(detections) == 1
    assert opened == [True]


def test_camera_modes_cache_ignores_corrupt_entries(tmp_path) -> None:
    app = _app_for_processing()
    app.autosave_path = tmp_path / "state" / "autosave.json"
    app.autosave_path.parent.mkdir(parents=True)
    app._camera_modes_path.write_text('{"0": [{"granted": [1, 2]}]}', encoding="utf-8")

    assert app._load_camera_modes(0) == []
    assert app._apply_cached_camera_modes(0) is False


def test_resolution_menu_shows_measured_modes_and_parses_their_labels() -> None:
    app = _app_for_processing()
    app.camera_modes = [
        CameraMode(requested=(3264, 2448), granted=(2304, 1536), fps=2.0),
        CameraMode(requested=(1920, 1080), granted=(1920, 1080), fps=30.0),
    ]
    app.camera_resolution = (1920, 1080)

    values = app._resolution_menu_values()
    assert values == ["2304x1536 - 2.0 fps (slow)", "1920x1080 - 30 fps"]
    assert app._resolution_menu_selection() == "1920x1080 - 30 fps"

    # Selecting a labelled mode applies just its resolution.
    applied: list[tuple[int, int]] = []
    app._burst_is_active = lambda: False
    app._set_status = lambda _text: None
    app._camera_opening = True  # stop before the async apply, keep the parse
    app._apply_resolution_string("2304x1536 - 2.0 fps (slow)")
    assert app._camera_resolution_chosen is True
    del applied

    # Without measurements the plain preset list is offered.
    app.camera_modes = []
    assert app._resolution_menu_values() == RESOLUTIONS
    assert app._resolution_menu_selection() == "1920x1080"


def test_capture_one_runs_on_shared_capture_job() -> None:
    app = _app_for_processing()
    calls: list[tuple[int, float]] = []
    app._start_capture_job = lambda *, shots, delay_sec: calls.append((shots, delay_sec))

    app.capture_one()

    assert calls == [(1, 0.0)]


def test_capture_job_skips_while_previous_job_runs() -> None:
    app = _app_for_processing()
    app.job_thread = object()
    statuses: list[str] = []
    app._set_status = statuses.append

    app._start_capture_job(shots=1, delay_sec=0.0)

    assert statuses and "Busy" in statuses[-1]


def test_shared_burst_camera_requires_the_capture_resolution() -> None:
    app = _app_for_processing()

    class Cam:
        def __init__(self, index, resolution, *, streaming=True):
            self.index = index
            self.resolution = resolution
            self._streaming = streaming

        def is_streaming(self) -> bool:
            return self._streaming

    # Preview and capture share one resolution, so a stream already at the
    # capture size is exactly what a shot should be taken from.
    match = Cam(3, (3264, 2448))
    app.camera = match
    assert app._shared_burst_camera(3, (3264, 2448)) is match

    app.camera = Cam(3, (1280, 720))
    assert app._shared_burst_camera(3, (3264, 2448)) is None
    app.camera = Cam(4, (3264, 2448))
    assert app._shared_burst_camera(3, (3264, 2448)) is None
    app.camera = Cam(3, (3264, 2448), streaming=False)
    assert app._shared_burst_camera(3, (3264, 2448)) is None


def test_camera_resolution_commits_only_after_success(monkeypatch) -> None:
    app = object.__new__(UnifiedScanApp)
    app.camera_index_var = _Var(2)
    app.camera_resolution = (640, 480)
    app.camera = None

    class FakeCamera:
        fail = False
        rollback_fail = False

        def __init__(self, *, index, resolution):
            self.index = index
            self.resolution = resolution
            self.released = False

        def open(self):
            if self.fail:
                raise RuntimeError("unsupported resolution")

        def release(self):
            self.released = True

        def set_resolution(self, resolution):
            self.resolution = resolution
            if resolution == (9999, 9999) or (self.rollback_fail and resolution == (1280, 720)):
                raise RuntimeError("unsupported resolution")

    monkeypatch.setattr("uniscan.ui.app.CameraService", FakeCamera)
    app._set_camera_resolution((1280, 720))
    assert app.camera_resolution == (1280, 720)
    assert app.camera.resolution == (1280, 720)

    app.camera = None
    FakeCamera.fail = True
    with pytest.raises(RuntimeError, match="unsupported resolution"):
        app._set_camera_resolution((9999, 9999))
    assert app.camera_resolution == (1280, 720)
    assert app.camera is None

    FakeCamera.fail = False
    app.camera = FakeCamera(index=2, resolution=(1280, 720))
    with pytest.raises(RuntimeError, match="unsupported resolution"):
        app._set_camera_resolution((9999, 9999))
    assert app.camera is not None
    assert app.camera.resolution == (1280, 720)
    assert app.camera_resolution == (1280, 720)

    FakeCamera.rollback_fail = True
    with pytest.raises(RuntimeError, match="unsupported resolution"):
        app._set_camera_resolution((9999, 9999))
    assert app.camera is None
    assert app.camera_resolution == (1280, 720)


def test_constructor_failure_releases_autosave_lock(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "autosave.json"

    def fail_after_lock(self):
        lock = acquire_autosave_lock(manifest)
        object.__setattr__(self, "_autosave_lock", lock)
        raise RuntimeError("forced constructor failure")

    monkeypatch.setattr(UnifiedScanApp, "_initialize", fail_after_lock)
    with pytest.raises(RuntimeError, match="forced constructor failure"):
        UnifiedScanApp()

    acquire_autosave_lock(manifest).release()


def test_run_app_reports_unsafe_lock_without_traceback(monkeypatch, capsys) -> None:
    def fail_startup():
        raise UnsafeSessionLockError("unsafe test lock")

    monkeypatch.setattr("uniscan.ui.app.UnifiedScanApp", fail_startup)
    assert run_app() == 2
    assert "UniScan startup failed: unsafe test lock" in capsys.readouterr().err


def test_staged_apply_rejects_stale_page_without_mutation(tmp_path) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entry = app.session.add_image(name="page", image=np.full((8, 9, 3), 30, np.uint8))
    previous = tmp_path / "previous.png"
    result_path = tmp_path / "result.png"
    assert imwrite_unicode(previous, entry.current_image)
    assert imwrite_unicode(result_path, np.full((8, 9, 3), 200, np.uint8))
    request = PageProcessingRequest()
    result = process_document_page(entry.original_image, request)
    committed = CommittedPageProcessing.from_result(request, result.diagnostics, result.image)
    snapshot = _ApplyPageSnapshot(
        entry.entry_id,
        entry.name,
        entry.original_path,
        previous,
        entry.revision,
        request,
        None,
    )
    staged = _StagedAppliedPage(entry.entry_id, result_path, committed, ())
    before = entry.current_image.copy()
    entry.revision += 1

    with pytest.raises(RuntimeError, match="Page changed while processing"):
        app._commit_staged_apply([snapshot], [staged])
    np.testing.assert_array_equal(entry.current_image, before)


def test_staged_apply_rolls_back_prior_pages_on_commit_failure(tmp_path) -> None:
    app = _app_for_processing()
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    entries = [
        app.session.add_image(name="one", image=np.full((8, 9, 3), 30, np.uint8)),
        app.session.add_image(name="two", image=np.full((8, 9, 3), 60, np.uint8)),
    ]
    request = PageProcessingRequest()
    snapshots = []
    staged = []
    old_images = [entry.current_image.copy() for entry in entries]
    for index, entry in enumerate(entries):
        previous = tmp_path / f"previous-{index}.png"
        result_path = tmp_path / f"result-{index}.png"
        assert imwrite_unicode(previous, old_images[index])
        if index == 0:
            assert imwrite_unicode(result_path, np.full((8, 9, 3), 210, np.uint8))
        else:
            result_path.write_bytes(b"not-a-png")
        result = process_document_page(entry.original_image, request)
        committed = CommittedPageProcessing.from_result(
            request,
            result.diagnostics,
            result.image,
        )
        snapshots.append(
            _ApplyPageSnapshot(
                entry.entry_id,
                entry.name,
                entry.original_path,
                previous,
                entry.revision,
                request,
                None,
            )
        )
        staged.append(_StagedAppliedPage(entry.entry_id, result_path, committed, ()))

    with pytest.raises(RuntimeError, match="Processed page is unreadable"):
        app._commit_staged_apply(snapshots, staged)
    for entry, old_image in zip(entries, old_images):
        np.testing.assert_array_equal(entry.current_image, old_image)
        assert entry.committed_processing is None
        assert entry.revision == 0


def test_gui_import_consumes_pdf_pages_lazily_and_stages_to_disk(tmp_path, monkeypatch) -> None:
    app = _app_for_processing()
    app.import_pdf_dpi_var = _Var(150)
    app.import_two_page_mode_var = _Var(True)
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    app.refresh_page_list = lambda keep_index=None: None
    app.go_to_review_tab = lambda: None
    app._set_status = lambda _text: None
    yielded = 0
    processed = 0

    def lazy_items(_paths, **_kwargs):
        nonlocal yielded
        for index in range(3):
            yielded += 1
            yield f"pdf-{index}", np.full((6, 7, 3), index + 1, np.uint8)

    def process_one(items, *, options, cancel_cb):
        nonlocal processed
        assert len(items) == 1
        assert options.two_page_mode is True
        assert yielded == processed + 1
        processed += 1
        name, image = items[0]
        return [PageResult(name, image, image, image, None, "cv_hybrid", True, None)]

    def run_job(_name, worker, on_done, *, on_error):
        try:
            result = worker(lambda **_kwargs: None, lambda: False)
            on_done(result)
        except Exception:
            on_error()
            raise
        return True

    monkeypatch.setattr("uniscan.ui.app.iter_input_items", lazy_items)
    monkeypatch.setattr("uniscan.ui.app.process_loaded_items", process_one)
    app._start_background_job = run_job

    app._import_paths(paths=[tmp_path / "large.pdf"])

    assert yielded == processed == 3
    assert len(app.session.entries) == 3


def test_gui_import_honors_cancellation_after_final_staging_encode(tmp_path, monkeypatch) -> None:
    app = _app_for_processing()
    app.import_pdf_dpi_var = _Var(150)
    app.import_two_page_mode_var = _Var(False)
    app.session = CaptureSession(store=PageStore(root_dir=tmp_path / "store"))
    app._set_status = lambda _text: None
    cancelled = False
    errors: list[str] = []

    image = np.full((6, 7, 3), 90, np.uint8)
    monkeypatch.setattr(
        "uniscan.ui.app.iter_input_items",
        lambda *_args, **_kwargs: iter((("page", image),)),
    )
    monkeypatch.setattr(
        "uniscan.ui.app.process_loaded_items",
        lambda items, **_kwargs: [
            PageResult("page", items[0][1], items[0][1], items[0][1], None, None, False, None)
        ],
    )
    real_write = imwrite_unicode
    writes = 0

    def cancel_after_staging_write(path, pixels) -> bool:
        nonlocal cancelled, writes
        written = real_write(path, pixels)
        writes += 1
        if writes == 1:
            cancelled = True
        return written

    def run_job(_name, worker, _on_done, *, on_error):
        try:
            worker(lambda **_kwargs: None, lambda: cancelled)
        except RuntimeError as exc:
            errors.append(str(exc))
            on_error()
        return True

    monkeypatch.setattr("uniscan.ui.app.imwrite_unicode", cancel_after_staging_write)
    app._start_background_job = run_job

    app._import_paths(paths=[tmp_path / "page.png"])

    assert writes == 1
    assert errors == ["Cancelled by user."]
    assert app.session.entries == []


def test_staged_apply_honors_cancellation_after_result_encode(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.png"
    previous = tmp_path / "previous.png"
    pixels = np.full((8, 9, 3), 80, np.uint8)
    assert imwrite_unicode(source, pixels)
    assert imwrite_unicode(previous, pixels)
    snapshot = _ApplyPageSnapshot(
        "entry",
        "page",
        source,
        previous,
        0,
        PageProcessingRequest(),
        None,
    )
    cancelled = False
    real_write = imwrite_unicode

    def write_then_cancel(path, image) -> bool:
        nonlocal cancelled
        written = real_write(path, image)
        cancelled = True
        return written

    monkeypatch.setattr("uniscan.ui.app.imwrite_unicode", write_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        UnifiedScanApp._stage_apply_pages(
            [snapshot],
            emit=lambda **_kwargs: None,
            is_cancelled=lambda: cancelled,
        )


def test_background_job_stays_busy_until_queued_completion_is_consumed(monkeypatch) -> None:
    app = object.__new__(UnifiedScanApp)
    app.job_thread = SimpleNamespace(is_alive=lambda: False)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "uniscan.ui.app.messagebox.showwarning",
        lambda title, text: warnings.append((title, text)),
    )

    started = app._start_background_job("second", lambda *_args: None, lambda _result: None)

    assert started is False
    assert warnings == [("Busy", "Another background job is already running.")]


def test_background_completion_releases_slot_before_callback() -> None:
    app = object.__new__(UnifiedScanApp)
    app._closing = False
    app.job_thread = SimpleNamespace(is_alive=lambda: False)
    app.job_queue = queue.Queue()
    app.winfo_exists = lambda: False
    app.cancel_task_button = SimpleNamespace(configure=lambda **_kwargs: None)
    app._set_status = lambda _text: None
    callback_state: list[object] = []
    app.job_queue.put(
        (
            "done",
            (lambda _result: callback_state.append(app.job_thread), "result", "test"),
        )
    )

    app._poll_job_queue()

    assert callback_state == [None]


def test_autosave_tick_skips_unchanged_manifest(tmp_path) -> None:
    app = object.__new__(UnifiedScanApp)
    entry = SimpleNamespace(
        entry_id="page",
        revision=0,
        name="page",
        selected=False,
        detected_backend=None,
    )
    saves: list[object] = []
    app.session = SimpleNamespace(
        entries=[entry],
        has_recoverable_state=True,
        save_manifest=lambda path: saves.append(path),
    )
    app.autosave_path = tmp_path / "autosave.json"
    app._last_autosave_signature = None
    app.winfo_exists = lambda: False
    app._set_status = lambda _text: None

    app._autosave_tick()
    app._autosave_tick()
    entry.selected = True
    app._autosave_tick()

    assert saves == [app.autosave_path, app.autosave_path]


def test_startup_diagnostics_status_lists_only_blocking_failures() -> None:
    app = object.__new__(UnifiedScanApp)
    app._closing = False
    app.winfo_exists = lambda: False
    app.job_queue = __import__("queue").Queue()
    statuses: list[str] = []
    app._set_status = statuses.append
    app.job_queue.put(
        (
            "diagnostics",
            DiagnosticReport(
                python="test",
                platform="test",
                checks=(
                    DiagnosticCheck("required", False, "broken", blocking=True),
                    DiagnosticCheck("optional-warning", False, "warning", blocking=False),
                ),
            ),
        )
    )

    app._poll_job_queue()

    assert statuses == ["Startup diagnostics failed: required. Run 'uniscan doctor'."]
