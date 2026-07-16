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
        assert app.export_tab_pdf_button.cget("state") == "disabled"
        assert app.export_tab_files_button.cget("state") == "disabled"
        assert app.cancel_task_button.cget("state") == "disabled"
        assert app.move_pages_up_button.cget("state") == "disabled"
        assert app.move_pages_down_button.cget("state") == "disabled"
        assert app.delete_pages_button.cget("state") == "disabled"
        assert app.page_listbox.bind("<Delete>")
        assert app.page_listbox.bind("<Control-Right>")
        assert app.page_listbox.bind("<B1-Motion>")
        assert app.page_listbox.bind("<Button-3>")

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
        assert app.export_tab_pdf_button.cget("state") == "normal"
        assert app.export_tab_files_button.cget("state") == "normal"
        assert app._single_selected_entry()[1].name == "second.png"
        assert app.move_pages_up_button.cget("state") == "normal"
        assert app.move_pages_down_button.cget("state") == "disabled"
        assert app.delete_pages_button.cget("state") == "normal"
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
        app.page_listbox.selection_clear(0, app_module.tk.END)
        app.page_listbox.selection_set(0)
        app.on_page_select()
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
        _pump_until(app, lambda: app.corner_resize_job is None)
        assert app.corner_meta_var.get().startswith("1/2  first.png")
        assert app.corner_prev_button.cget("state") == "disabled"
        assert app.corner_next_button.cget("state") == "normal"
        source_bounds = app.corner_source_canvas.bbox("source-image")
        preview_bounds = app.corner_preview_canvas.bbox("perspective-preview")
        assert source_bounds is not None and preview_bounds is not None
        initial_source_size = (
            source_bounds[2] - source_bounds[0],
            source_bounds[3] - source_bounds[1],
        )
        initial_corner_preview_size = (
            preview_bounds[2] - preview_bounds[0],
            preview_bounds[3] - preview_bounds[1],
        )
        corner_editors[0].geometry("1400x980")
        app.update()
        _pump_until(
            app,
            lambda: (
                app.corner_resize_job is None
                and app.corner_source_canvas.bbox("source-image") is not None
                and (
                    app.corner_source_canvas.bbox("source-image")[2]
                    - app.corner_source_canvas.bbox("source-image")[0]
                    > initial_source_size[0]
                )
            ),
        )
        resized_source_bounds = app.corner_source_canvas.bbox("source-image")
        resized_preview_bounds = app.corner_preview_canvas.bbox("perspective-preview")
        assert resized_source_bounds is not None and resized_preview_bounds is not None
        assert resized_source_bounds[2] - resized_source_bounds[0] > initial_source_size[0]
        assert resized_source_bounds[3] - resized_source_bounds[1] > initial_source_size[1]
        assert (
            resized_preview_bounds[2] - resized_preview_bounds[0]
            > initial_corner_preview_size[0]
        )
        assert (
            resized_preview_bounds[3] - resized_preview_bounds[1]
            > initial_corner_preview_size[1]
        )
        app.corner_next_button.invoke()
        app.update()
        assert app.corner_meta_var.get().startswith("2/2  second.png")
        assert app.corner_prev_button.cget("state") == "normal"
        assert app.corner_next_button.cget("state") == "disabled"
        corner_editors[0].destroy()
        app.page_listbox.selection_clear(0, app_module.tk.END)
        app.page_listbox.selection_set(1)
        app.on_page_select()
        app.open_dewarp_points_editor()
        app.update()
        dewarp_editors = [
            child
            for child in app.winfo_children()
            if hasattr(child, "title") and child.title() == "Wave correction"
        ]
        assert len(dewarp_editors) == 1
        _pump_until(app, lambda: app.dewarp_resize_job is None)
        initial_dewarp_bounds = app.dewarp_source_canvas.bbox("dewarp-source")
        assert initial_dewarp_bounds is not None
        initial_dewarp_size = (
            initial_dewarp_bounds[2] - initial_dewarp_bounds[0],
            initial_dewarp_bounds[3] - initial_dewarp_bounds[1],
        )
        dewarp_editors[0].geometry("1400x980")
        app.update()
        _pump_until(
            app,
            lambda: (
                app.dewarp_resize_job is None
                and app.dewarp_source_canvas.bbox("dewarp-source") is not None
                and (
                    app.dewarp_source_canvas.bbox("dewarp-source")[2]
                    - app.dewarp_source_canvas.bbox("dewarp-source")[0]
                    > initial_dewarp_size[0]
                )
            ),
        )
        resized_dewarp_bounds = app.dewarp_source_canvas.bbox("dewarp-source")
        resized_dewarp_preview_bounds = app.dewarp_preview_canvas.bbox("dewarp-preview")
        assert resized_dewarp_bounds is not None
        assert resized_dewarp_preview_bounds is not None
        assert resized_dewarp_bounds[2] - resized_dewarp_bounds[0] > initial_dewarp_size[0]
        assert resized_dewarp_bounds[3] - resized_dewarp_bounds[1] > initial_dewarp_size[1]
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
        app.deiconify()
        _pump_until(app, lambda: app.review_preview_resize_job is None)
        initial_preview_size = app.page_preview_after_photo.cget("size")
        app.geometry("1560x980")
        app.update()
        _pump_until(
            app,
            lambda: (
                app.review_preview_resize_job is None
                and app.page_preview_after_photo.cget("size") != initial_preview_size
            ),
        )
        resized_preview_size = app.page_preview_after_photo.cget("size")
        assert resized_preview_size[0] > initial_preview_size[0]
        assert resized_preview_size[1] > initial_preview_size[1]
        app.move_selected_up()
        assert app.session.entries[0].name == "second.png"
        assert app.move_pages_up_button.cget("state") == "disabled"
        assert app.move_pages_down_button.cget("state") == "normal"
        app.move_selected_down()
        assert app.session.entries[1].name == "second.png"
        start_bounds = app.page_listbox.bbox(1)
        target_bounds = app.page_listbox.bbox(0)
        assert start_bounds is not None and target_bounds is not None
        app.page_listbox.event_generate(
            "<ButtonPress-1>",
            x=start_bounds[0] + 8,
            y=start_bounds[1] + start_bounds[3] // 2,
        )
        app.update()
        app.page_listbox.event_generate(
            "<B1-Motion>",
            x=target_bounds[0] + 8,
            y=target_bounds[1] + 2,
            state=0x0100,
        )
        app.update()
        app.page_listbox.event_generate(
            "<ButtonRelease-1>",
            x=target_bounds[0] + 8,
            y=target_bounds[1] + 2,
        )
        app.update()
        assert [entry.name for entry in app.session.entries] == ["second.png", "first.png"]
        start_bounds = app.page_listbox.bbox(0)
        assert start_bounds is not None
        drop_y = app.page_listbox.winfo_height() - 2
        app.page_listbox.event_generate(
            "<ButtonPress-1>",
            x=start_bounds[0] + 8,
            y=start_bounds[1] + start_bounds[3] // 2,
        )
        app.update()
        app.page_listbox.event_generate(
            "<B1-Motion>",
            x=start_bounds[0] + 8,
            y=drop_y,
            state=0x0100,
        )
        app.update()
        app.page_listbox.event_generate(
            "<ButtonRelease-1>",
            x=start_bounds[0] + 8,
            y=drop_y,
        )
        app.update()
        assert [entry.name for entry in app.session.entries] == ["first.png", "second.png"]
        app.withdraw()
        app.select_all_pages()
        assert all(entry.selected for entry in app.session.entries)
        assert app.move_pages_up_button.cget("state") == "disabled"
        assert app.move_pages_down_button.cget("state") == "disabled"
        app.clear_page_selection()
        assert not any(entry.selected for entry in app.session.entries)
        assert app.delete_pages_button.cget("state") == "disabled"
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
        app.update_page_preview()
        overlap_deadline = time.monotonic() + 0.35
        while time.monotonic() < overlap_deadline:
            app.update()
            time.sleep(0.01)
        assert seen_shapes == [wide_entry.preview_original_image.shape]
        preview_release.set()
        _pump_until(
            app,
            lambda: (
                app.page_preview_after_photo is not None
                and not any(thread.is_alive() for thread in app.review_preview_threads)
            ),
        )
        assert seen_shapes == [
            wide_entry.preview_original_image.shape,
            wide_entry.preview_original_image.shape,
        ]
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

        # The resize/concurrency checks make this smoke test long enough for
        # the periodic autosave tick; reset it to isolate close-time behavior.
        app.autosave_path.unlink(missing_ok=True)
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
