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
    CaptureSession,
    CommittedPageProcessing,
    UnsafeSessionLockError,
    acquire_autosave_lock,
    create_persistent_session,
    load_or_create_session,
)
from uniscan.storage import PageStore
from uniscan.ui.app import (
    UnifiedScanApp,
    _ApplyPageSnapshot,
    _StagedAppliedPage,
    _add_dewarp_control_point,
    _compose_split_preview,
    _detection_summary,
    _fit_image_to_box,
    _move_dewarp_control_point,
    _perspective_source_image,
    _remove_dewarp_control_point,
    _shift_dewarp_control_points,
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
        "preprocess_illumination_var": False,
        "binarization_method_var": "None",
        "binarization_window_var": 31,
        "binarization_k_var": 0.2,
        "despeckle_strength_var": "None",
        "postprocess_var": "Grayscale",
        "lens_mode_var": "Custom",
        "dewarp_method_var": "None",
        "page_layout_var": "Keep source page",
        "export_pdf_dpi_var": 300,
        "page_margin_mm_var": 10.0,
        "page_align_x_var": "center",
        "page_align_y_var": "center",
    }
    for name, value in values.items():
        setattr(app, name, _Var(value))
    app._binarization_k_custom = False
    app.processing_cache = None
    app._last_processing_cache_hits = ()
    return app


def _read_image(path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


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


def test_geometry_change_replays_committed_appearance_not_pending_controls(tmp_path) -> None:
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
    assert recipe.deskew_method == "none"
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


def test_capture_frame_uses_plain_two_page_snapshot_not_tk_var(monkeypatch) -> None:
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

    assert (
        app._process_capture_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            "burst",
            two_page_mode=True,
        )
        == []
    )
    assert seen == [True]


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


def test_dewarp_points_can_be_added_moved_removed_and_shifted_as_a_line() -> None:
    points = [(0.0, 0.0), (0.5, 0.02), (1.0, 0.0)]

    added = _add_dewarp_control_point(points, 0.25, -0.05)
    assert added == 1
    assert points[1] == (0.25, -0.05)

    _move_dewarp_control_point(points, added, 0.4, 0.1)
    assert points[1] == (0.4, 0.1)
    _move_dewarp_control_point(points, added, 0.9, 0.5)
    assert points[1][0] < points[2][0]
    assert points[1][1] == 0.24

    before = [value for _x, value in points]
    applied = _shift_dewarp_control_points(points, -0.08)
    assert applied == pytest.approx(-0.08)
    assert [value for _x, value in points] == pytest.approx([value - 0.08 for value in before])

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
    app.import_two_page_mode_var = _Var(False)
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

    def cancel_after_second_write(path, pixels) -> bool:
        nonlocal cancelled, writes
        written = real_write(path, pixels)
        writes += 1
        if writes == 2:
            cancelled = True
        return written

    def run_job(_name, worker, _on_done, *, on_error):
        try:
            worker(lambda **_kwargs: None, lambda: cancelled)
        except RuntimeError as exc:
            errors.append(str(exc))
            on_error()
        return True

    monkeypatch.setattr("uniscan.ui.app.imwrite_unicode", cancel_after_second_write)
    app._start_background_job = run_job

    app._import_paths(paths=[tmp_path / "page.png"])

    assert writes == 2
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
