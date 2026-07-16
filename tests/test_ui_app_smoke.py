from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="Tk smoke test needs Windows desktop or Xvfb",
)


def _pump_until(app, predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            app.update()
        except Exception:
            if predicate():
                return
            raise
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for GUI background work.")


def test_gui_constructs_with_all_tabs_and_closes_cleanly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UNISCAN_STATE_DIR", str(tmp_path / "state"))
    from uniscan.ui import app as app_module

    app = app_module.UnifiedScanApp()
    preview_release = threading.Event()
    close_release = threading.Event()
    close_started = False
    close_completed = False
    try:
        app.withdraw()
        app.update()
        assert app.tabs.get() == app.tab_review_name
        assert app.status_var.get() == "Ready"
        assert len(app.session) == 0
        assert app.preprocess_preset_var.get() == "Document"
        assert app.dewarp_method_var.get() == "Automatic (validated)"
        assert app.geometry_summary_var.get() == "Wave preview: pending"
        assert app.apply_processing_button.cget("text") == "Apply preview to pages"
        assert app.deskew_method_var.get() == "Hybrid (recommended)"
        assert app.binarization_method_var.get() == "None"
        assert app.despeckle_strength_var.get() == "None"
        assert app.page_layout_var.get() == "Keep source page"
        assert app.preview_mode_var.get() == "Processed"
        assert app.page_preview_after_frame.winfo_manager() == "grid"
        assert app.page_preview_before_frame.winfo_manager() == ""
        app.preprocess_illumination_var.set(True)
        assert app._current_preprocess_settings().correct_illumination is True
        assert app.import_files_entry.winfo_exists()
        assert app._drag_drop_error is None
        assert app.toolbar_export_button.cget("state") == "disabled"
        assert app.cancel_task_button.cget("state") == "disabled"
        assert app.page_listbox.bind("<Delete>")
        assert app.page_listbox.bind("<Control-Right>")

        source = tmp_path / "drop.png"
        source.write_bytes(b"placeholder")
        imports: list[list] = []
        monkeypatch.setattr(app, "_import_paths", lambda *, paths: imports.append(paths))
        assert app._on_drop_files(SimpleNamespace(data=f"{{{source.as_posix()}}}")) == "break"
        assert imports[-1] == [source]
        monkeypatch.setattr(
            "uniscan.ui.app.filedialog.askopenfilenames",
            lambda **_kwargs: (str(source),),
        )
        app.quick_add_files()
        assert imports[-1] == [source]

        first = np.full((90, 120, 3), 220, dtype=np.uint8)
        second = np.full((90, 120, 3), 180, dtype=np.uint8)
        app.session.add_image(name="first.png", image=first)
        app.session.add_image(name="second.png", image=second)
        app.refresh_page_list(keep_index=1)
        app.update()
        assert app.page_count_var.get() == "2 pages"
        assert app.toolbar_export_button.cget("state") == "normal"
        assert app._single_selected_entry()[1].name == "second.png"
        app.binarization_method_var.set("Sauvola (uneven light)")
        app.despeckle_strength_var.set("Conservative")
        settings = app._current_preprocess_settings()
        assert settings.binarization_method == "sauvola"
        assert settings.despeckle_strength == "conservative"
        app.page_layout_var.set("A4")
        laid_out, layout_diagnostics = app._apply_page_layout(second, preview=True)
        assert laid_out.shape[:2] == (1169, 827)
        assert layout_diagnostics.applied is True
        app.export_pdf_dpi_var.set(100)
        selected_entry = app._single_selected_entry()[1]
        app._reprocess_entry_from_original(selected_entry)
        assert selected_entry.current_image.shape[:2] == (1169, 827)
        assert app.processing_cache.stats.writes >= 1
        app.analyze_selected_page_lighting()
        assert app.lighting_summary_var.get().startswith("Shadow ")
        app.binarization_method_var.set("None")
        app.despeckle_strength_var.set("None")
        app.page_layout_var.set("Keep source page")
        app._reprocess_entry_from_original(selected_entry)
        app.clear_processing_cache()
        assert not list(app.processing_cache.root_dir.glob("*.png"))
        app.open_manual_corners_editor()
        app.update()
        corner_editors = [
            child
            for child in app.winfo_children()
            if hasattr(child, "title") and child.title() == "Perspective corners"
        ]
        assert len(corner_editors) == 1
        assert app.grab_current() == corner_editors[0]
        assert app.corner_source_canvas.winfo_exists()
        assert app.corner_preview_canvas.winfo_exists()
        corner_editors[0].destroy()
        app.open_dewarp_points_editor()
        app.update()
        dewarp_editors = [
            child
            for child in app.winfo_children()
            if hasattr(child, "title") and child.title() == "Wave correction"
        ]
        assert len(dewarp_editors) == 1
        dewarp_editors[0].destroy()
        app.preview_mode_var.set("Original")
        app._on_preview_mode_change()
        assert app.page_preview_before_frame.winfo_manager() == "grid"
        assert app.page_preview_after_frame.winfo_manager() == ""
        app.preview_mode_var.set("Compare")
        app._on_preview_mode_change()
        _pump_until(app, lambda: app.page_preview_after_photo is not None)
        assert app.page_preview_before_frame.winfo_manager() == "grid"
        assert app.page_preview_after_frame.winfo_manager() == "grid"
        assert app.geometry_summary_var.get() != "Wave preview: pending"
        app.move_selected_up()
        assert app.session.entries[0].name == "second.png"
        app.move_selected_down()
        assert app.session.entries[1].name == "second.png"
        app.select_all_pages()
        assert all(entry.selected for entry in app.session.entries)
        app.clear_page_selection()
        assert not any(entry.selected for entry in app.session.entries)
        app.select_all_pages()
        app.delete_selected_pages()
        assert len(app.session) == 0
        assert app.page_count_var.get() == "0 pages"

        started = threading.Event()
        seen_shapes: list[tuple[int, ...]] = []
        real_process = app_module.process_document_page

        def slow_process(image, request):
            seen_shapes.append(image.shape)
            started.set()
            preview_release.wait(timeout=5)
            return real_process(image, request)

        wide_source = np.full((120, 2400, 3), 180, dtype=np.uint8)
        wide_entry = app.session.add_image(name="wide", image=wide_source)
        app.refresh_page_list(keep_index=0)
        app._cancel_review_page_preview()
        app.page_preview_after_photo = None
        monkeypatch.setattr(app_module, "process_document_page", slow_process)

        before = time.monotonic()
        app.update_page_preview()
        elapsed = time.monotonic() - before

        assert elapsed < 0.15
        assert "Preparing fast preview" in app.page_preview_after_label.cget("text")
        _pump_until(app, started.is_set)
        assert seen_shapes == [wide_entry.preview_original_image.shape]
        assert seen_shapes[0][1] < wide_source.shape[1]
        preview_release.set()
        _pump_until(
            app,
            lambda: (
                app.page_preview_after_photo is not None
                and not any(thread.is_alive() for thread in app.review_preview_threads)
            ),
        )
        monkeypatch.setattr(app_module, "process_document_page", real_process)

        app.select_all_pages()
        app.delete_selected_pages()
        app.session.add_image(
            name="close-page",
            image=np.full((20, 30, 3), 180, dtype=np.uint8),
        )
        worker = threading.Thread(target=close_release.wait, daemon=True)
        worker.start()
        app.job_thread = worker

        app._on_close()
        close_started = True
        assert app._close_wait_job is not None
        assert app.session.store.session_dir.exists()
        assert not app.autosave_path.exists()

        close_release.set()
        _pump_until(app, lambda: app._close_wait_job is None)
        close_completed = True
        assert app.autosave_path.exists()
    finally:
        preview_release.set()
        close_release.set()
        if not close_started:
            app._on_close()
            close_started = True
        if not close_completed:
            _pump_until(app, lambda: app._close_wait_job is None)
