from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniscan.session import CaptureSession, create_persistent_session, load_or_create_session
from uniscan.storage import PageStore
from uniscan.ui.app import UnifiedScanApp


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
    # pages that were committed independently with Apply processing.
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

    app._handle_review_preview_result(1, None, "stale failure")
    assert app.page_preview_after_label.text == "old"
    assert app.page_preview_after_photo == "old-photo"

    app._handle_review_preview_result(2, None, "new failure")
    assert app.page_preview_after_label.text == "Preview failed: new failure"
    assert app.page_preview_after_photo is None
    assert statuses == ["Preview failed: new failure"]
