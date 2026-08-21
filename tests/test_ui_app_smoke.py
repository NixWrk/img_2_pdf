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


def _assert_round_magnifier_at(canvas, x_pos: int, y_pos: int) -> None:
    rings = canvas.find_withtag("geometry-magnifier-ring")
    assert len(rings) == 1
    assert canvas.type(rings[0]) == "oval"
    left, top, right, bottom = canvas.coords(rings[0])
    assert (left + right) / 2 == pytest.approx(x_pos, abs=1)
    assert (top + bottom) / 2 == pytest.approx(y_pos, abs=1)
    assert right - left == pytest.approx(bottom - top, abs=1)


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
        assert app.apply_processing_button.cget("text") == "Apply candidate for export"
        assert app.deskew_method_var.get() == "Hybrid (recommended)"
        assert app.binarization_method_var.get() == "None"
        assert app.despeckle_strength_var.get() == "None"
        assert app.page_layout_var.get() == "Keep source page"
        assert app.preview_mode_var.get() == "Preview"
        assert app.crop_warning_var.get() == ""
        assert app.export_readiness_var.get() == "No pages to export"
        assert app.page_preview_after_frame.winfo_manager() == "grid"
        assert app.page_preview_before_frame.winfo_manager() == ""
        assert app.tabs._name_list == [app.tab_review_name, app.tab_camera_name]
        app.shadow_method_var.set("Classical")
        assert app._processing_request().shadow_method == "classical"
        assert app._current_preprocess_settings().correct_illumination is False
        assert app._drag_drop_error is None
        assert app.toolbar_add_files_button.cget("text") == "+ Add files"
        assert app.toolbar_add_folder_button.cget("text") == "Add folder"
        assert app.toolbar_paste_button.cget("text") == "Paste"
        assert app.toolbar_camera_button.cget("text") == "Camera"
        assert app.toolbar_export_pdf_button.cget("state") == "disabled"
        assert app.toolbar_export_options_button.cget("state") == "disabled"
        assert app.cancel_task_button.cget("state") == "disabled"
        assert app.move_pages_up_button.cget("state") == "disabled"
        assert app.move_pages_down_button.cget("state") == "disabled"
        assert app.delete_pages_button.cget("state") == "disabled"
        assert app.undo_delete_button.cget("state") == "disabled"
        assert app.page_listbox.bind("<Delete>")
        assert app.page_listbox.bind("<Control-z>")
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
        app.toolbar_add_files_button.invoke()
        assert imports[-1] == [source]

        first = np.full((90, 120, 3), 220, dtype=np.uint8)
        second = np.full((90, 120, 3), 180, dtype=np.uint8)
        first_entry = app.session.add_image(name="first.png", image=first)
        second_entry = app.session.add_image(name="second.png", image=second)
        app.refresh_page_list(keep_index=1)
        app.update()
        assert app.page_count_var.get() == "2 pages"
        assert app.crop_warning_var.get() == "⚠ 2 pages need crop review"
        assert app.export_readiness_var.get() == "Export: 0 ready · 0 warnings · 2 blocked"
        assert app.toolbar_export_pdf_button.cget("state") == "normal"
        assert app.toolbar_export_options_button.cget("state") == "normal"
        readiness_warnings: list[tuple[str, str]] = []
        save_requests: list[bool] = []
        monkeypatch.setattr(
            app_module.messagebox,
            "showwarning",
            lambda title, message: readiness_warnings.append((title, message)),
        )
        monkeypatch.setattr(
            app_module.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: save_requests.append(True) or "",
        )
        app.export_to_pdf()
        assert readiness_warnings and readiness_warnings[-1][0] == "Export blocked"
        assert "2 blocked" in readiness_warnings[-1][1]
        assert save_requests == []
        assert app.status_var.get() == "Export blocked: resolve page readiness issues."
        export_calls: list[tuple[str, str, int | str]] = []
        monkeypatch.setattr(
            app,
            "export_to_pdf",
            lambda: export_calls.append(
                ("pdf", app.export_scope_var.get(), app.export_pdf_dpi_var.get())
            ),
        )
        monkeypatch.setattr(
            app,
            "export_to_files",
            lambda: export_calls.append(
                ("images", app.export_scope_var.get(), app.export_format_var.get())
            ),
        )
        app.toolbar_export_pdf_button.invoke()
        assert export_calls[-1] == ("pdf", "All pages", 300)
        app.open_export_dialog()
        app.export_dialog_mode_var.set("Images")
        app.export_dialog_scope_var.set("Selected pages")
        app.export_dialog_format_var.set("jpg")
        app.export_custom_button.invoke()
        assert export_calls[-1] == ("images", "Selected pages", "jpg")
        assert app.export_dialog_window is None
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
        app.deiconify()
        app.geometry("1100x720")
        app.update()
        app.open_manual_corners_editor()
        app.update()
        assert app.corner_editor_window is not None
        assert app.inline_editor_host.winfo_manager() == "grid"
        assert app.workspace_preview_frame.winfo_manager() == ""
        assert app.corner_source_canvas.winfo_exists()
        assert app.corner_preview_canvas.winfo_exists()
        _pump_until(app, lambda: app.corner_resize_job is None)
        assert app.corner_meta_var.get().startswith("1/2  first.png")
        assert app.corner_prev_button.cget("state") == "disabled"
        assert app.corner_next_button.cget("state") == "normal"
        source_bounds = app.corner_source_canvas.bbox("source-image")
        preview_bounds = app.corner_preview_canvas.bbox("perspective-preview")
        assert source_bounds is not None and preview_bounds is not None
        assert app.corner_editor_state["last_corrected_source_shape"] == first.shape
        corner_point = app.corner_editor_state["points"][0]
        corner_x = int(
            float(corner_point[0]) / float(app.corner_editor_state["scale_x"])
            + float(app.corner_editor_state["offset_x"])
        )
        corner_y = int(
            float(corner_point[1]) / float(app.corner_editor_state["scale_y"])
            + float(app.corner_editor_state["offset_y"])
        )
        app.corner_source_canvas.event_generate("<ButtonPress-1>", x=corner_x, y=corner_y)
        app.update()
        assert app.corner_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.corner_source_canvas, corner_x, corner_y)
        app.corner_source_canvas.event_generate(
            "<B1-Motion>", x=corner_x + 8, y=corner_y + 8, state=0x0100
        )
        app.update()
        assert app.corner_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.corner_source_canvas, corner_x + 8, corner_y + 8)
        app.corner_source_canvas.event_generate("<ButtonRelease-1>", x=corner_x + 8, y=corner_y + 8)
        app.update()
        assert app.corner_source_canvas.bbox("geometry-magnifier") is None
        assert first_entry.entry_id in app.corner_editor_state["dirty_entry_ids"]
        initial_source_size = (
            source_bounds[2] - source_bounds[0],
            source_bounds[3] - source_bounds[1],
        )
        initial_corner_preview_size = (
            preview_bounds[2] - preview_bounds[0],
            preview_bounds[3] - preview_bounds[1],
        )
        app.geometry("1500x920")
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
            resized_preview_bounds[2] - resized_preview_bounds[0] > initial_corner_preview_size[0]
        )
        assert (
            resized_preview_bounds[3] - resized_preview_bounds[1] > initial_corner_preview_size[1]
        )
        app.corner_next_button.invoke()
        app.update()
        assert first_entry.entry_id in app.corner_editor_state["dirty_entry_ids"]
        assert first_entry.detected_backend is None
        np.testing.assert_array_equal(first_entry.original_image, first)
        assert app.corner_meta_var.get().startswith("2/2  second.png")
        assert app.corner_prev_button.cget("state") == "normal"
        assert app.corner_next_button.cget("state") == "disabled"
        second_point = app.corner_editor_state["points"][0]
        second_x = int(
            float(second_point[0]) / float(app.corner_editor_state["scale_x"])
            + float(app.corner_editor_state["offset_x"])
        )
        second_y = int(
            float(second_point[1]) / float(app.corner_editor_state["scale_y"])
            + float(app.corner_editor_state["offset_y"])
        )
        app.corner_source_canvas.event_generate("<ButtonPress-1>", x=second_x, y=second_y)
        app.corner_source_canvas.event_generate(
            "<B1-Motion>", x=second_x + 6, y=second_y + 6, state=0x0100
        )
        app.corner_source_canvas.event_generate("<ButtonRelease-1>", x=second_x + 6, y=second_y + 6)
        app.update()
        assert second_entry.entry_id in app.corner_editor_state["dirty_entry_ids"]
        app.corner_apply_button.invoke()
        app.update()
        assert second_entry.entry_id not in app.corner_editor_state["dirty_entry_ids"]
        assert second_entry.detected_backend == "manual"
        app.corner_close_button.invoke()
        assert first_entry.detected_backend is None
        np.testing.assert_array_equal(first_entry.original_image, first)
        assert app.inline_editor_host.winfo_manager() == ""
        assert app.workspace_preview_frame.winfo_manager() == "grid"
        app.page_listbox.selection_clear(0, app_module.tk.END)
        app.page_listbox.selection_set(1)
        app.on_page_select()
        app.geometry("1100x720")
        app.open_dewarp_points_editor()
        app.update()
        assert app.dewarp_editor_window is not None
        assert app.inline_editor_host.winfo_manager() == "grid"
        _pump_until(app, lambda: app.dewarp_resize_job is None)
        initial_dewarp_bounds = app.dewarp_source_canvas.bbox("dewarp-source")
        assert initial_dewarp_bounds is not None
        assert (
            app.dewarp_editor_state["last_corrected_source_shape"]
            == second_entry.original_image.shape
        )
        initial_dewarp_size = (
            initial_dewarp_bounds[2] - initial_dewarp_bounds[0],
            initial_dewarp_bounds[3] - initial_dewarp_bounds[1],
        )
        initial_point_count = len(app.dewarp_editor_state["points"])
        app.dewarp_add_point_button.invoke()
        display_width = int(app.dewarp_editor_state["display_width"])
        display_height = int(app.dewarp_editor_state["display_height"])
        offset_x = int(app.dewarp_editor_state["offset_x"])
        offset_y = int(app.dewarp_editor_state["offset_y"])
        app.dewarp_source_canvas.event_generate(
            "<Button-1>",
            x=offset_x + int(display_width * 0.33),
            y=offset_y + int(display_height * 0.45),
        )
        app.update()
        assert len(app.dewarp_editor_state["points"]) == initial_point_count + 1
        app.dewarp_remove_point_button.invoke()
        assert len(app.dewarp_editor_state["points"]) == initial_point_count

        assert len(app.dewarp_editor_state["curves"]) == 3
        top_curve = app.dewarp_editor_state["curves"][0]
        before_shift = list(top_curve["points"])
        before_anchor = float(top_curve["anchor"])
        curve_x = 0.44
        curve_y = float(
            app_module.interpolate_control_curve(
                before_shift,
                np.asarray([curve_x], dtype=np.float32),
            )[0]
        )
        press_x = offset_x + int(display_width * curve_x)
        press_y = offset_y + int(display_height * (before_anchor + curve_y))
        app.dewarp_source_canvas.event_generate("<ButtonPress-1>", x=press_x, y=press_y)
        app.update()
        assert app.dewarp_editor_state["active_curve"] == 0
        assert app.dewarp_editor_state["points"] is top_curve["points"]
        assert app.dewarp_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.dewarp_source_canvas, press_x, press_y)
        target_y = offset_y
        app.dewarp_source_canvas.event_generate("<B1-Motion>", x=press_x, y=target_y, state=0x0100)
        app.update()
        assert app.dewarp_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.dewarp_source_canvas, press_x, target_y)
        app.dewarp_source_canvas.event_generate("<ButtonRelease-1>", x=press_x, y=target_y)
        app.update()
        assert app.dewarp_source_canvas.bbox("geometry-magnifier") is None
        assert app.dewarp_editor_state["points"] == before_shift
        guide_anchor = float(app.dewarp_editor_state["guide_anchor"])
        assert guide_anchor < before_anchor
        assert min(guide_anchor + point[1] for point in before_shift) == pytest.approx(
            0.0, abs=0.01
        )
        overlay_bounds = app.dewarp_source_canvas.bbox("dewarp-overlay")
        assert overlay_bounds is not None and overlay_bounds[1] < offset_y

        app.geometry("1500x920")
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
        preview_pixels = app.dewarp_editor_state["last_corrected"].copy()
        app.dewarp_apply_points_button.invoke()
        np.testing.assert_array_equal(second_entry.current_image, preview_pixels)
        assert second_entry.dewarp_control_curves is not None
        assert len(second_entry.dewarp_control_curves) == 3
        saved_curves = second_entry.dewarp_control_curves
        app.open_dewarp_points_editor()
        app.update()
        _pump_until(app, lambda: app.dewarp_resize_job is None)
        reopened_curves = tuple(
            (float(curve["anchor"]), tuple(curve["points"]))
            for curve in app.dewarp_editor_state["curves"]
        )
        assert reopened_curves == saved_curves
        app.dewarp_close_button.invoke()
        app.open_review_processing_dialog()
        app.update()
        assert app.review_processing_window is not None
        assert app.inline_editor_host.winfo_manager() == "grid"
        app.review_processing_close_button.invoke()
        assert app.inline_editor_host.winfo_manager() == ""
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
        assert app.page_preview_after_state.cget("text") == "Committed — export ready"
        app.postprocess_var.set("Black and White")
        app.update_page_preview()
        _pump_until(
            app,
            lambda: app.page_preview_after_state.cget("text") == "Candidate — not exported",
        )
        app._sync_controls_from_single_committed_page()
        app.deiconify()
        app.geometry("1100x720")
        app.update()
        _pump_until(
            app,
            lambda: (
                app.page_preview_after_label.winfo_width() > 1
                and app.page_preview_after_label.winfo_height() > 1
            ),
        )
        _pump_until(app, lambda: app.review_preview_resize_job is None)
        initial_preview_size = app.page_preview_after_photo.cget("size")
        app.geometry("1500x920")
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
        # Camera lives on an in-window tab now; switching to it starts the
        # preview and leaving it releases the device.
        preview_calls: list[bool] = []
        app.start_preview = lambda: preview_calls.append(True)
        app.toolbar_camera_button.invoke()
        app.update()
        assert app.tabs.get() == app.tab_camera_name
        assert app.tabs._name_list == [app.tab_review_name, app.tab_camera_name]
        assert preview_calls == [True]
        released: list[bool] = []
        app.camera = SimpleNamespace(release=lambda: released.append(True))
        app.go_to_review_tab()
        app.update()
        assert app.tabs.get() == app.tab_review_name
        assert app.camera is None
        assert released == [True]
        app.update()
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
        monkeypatch.setattr("uniscan.ui.app.messagebox.askyesno", lambda *_args: True)
        app.delete_selected_pages()
        assert len(app.session) == 0
        assert app.page_count_var.get() == "0 pages"
        assert app.undo_delete_button.cget("state") == "normal"
        app.undo_last_page_deletion()
        assert len(app.session) == 2
        assert all(entry.selected for entry in app.session.entries)
        app.delete_selected_pages()
        assert len(app.session) == 0

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


def test_split_workflow_previews_two_pages_before_mutating_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UNISCAN_STATE_DIR", str(tmp_path / "split-state"))
    from uniscan.ui import app as app_module

    app = app_module.UnifiedScanApp()
    try:
        app.withdraw()
        spread = np.full((360, 720, 3), 235, np.uint8)
        spread[:, 350:370] = 25
        for x0, x1 in ((40, 330), (390, 680)):
            for y in range(50, 330, 28):
                app_module.cv2.line(spread, (x0, y), (x1, y), (50, 50, 50), 3)
        app.session.add_image(name="spread", image=spread)
        app.refresh_page_list(keep_index=0)
        app.deiconify()
        app.geometry("1100x720")
        app.update()

        app.open_split_editor()
        app.update()
        _pump_until(app, lambda: app.split_resize_job is None)
        assert app.split_editor_window is not None
        assert app.split_editor_state["source_shape"] == spread.shape
        display_width = int(app.split_editor_state["display_width"])
        display_height = int(app.split_editor_state["display_height"])
        offset_x = int(app.split_editor_state["offset_x"])
        offset_y = int(app.split_editor_state["offset_y"])
        line_x = offset_x + int(float(app.split_editor_state["ratio"]) * (display_width - 1))
        drag_y = offset_y + display_height // 2
        target_x = offset_x + int(0.4 * (display_width - 1))
        app.split_source_canvas.event_generate("<ButtonPress-1>", x=line_x, y=drag_y)
        app.update()
        assert app.split_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.split_source_canvas, line_x, drag_y)
        app.split_source_canvas.event_generate("<B1-Motion>", x=target_x, y=drag_y, state=0x0100)
        app.update()
        assert float(app.split_editor_state["ratio"]) == pytest.approx(0.4, abs=0.01)
        assert app.split_source_canvas.bbox("geometry-magnifier") is not None
        _assert_round_magnifier_at(app.split_source_canvas, target_x, drag_y)
        app.split_source_canvas.event_generate("<ButtonRelease-1>", x=target_x, y=drag_y)
        app.update()
        assert app.split_source_canvas.bbox("geometry-magnifier") is None
        app.split_editor_preview_button.invoke()

        assert len(app.session.entries) == 1
        assert app.preview_mode_var.get() == "Compare"
        assert app.apply_split_button.cget("state") == "normal"
        assert app.split_preview_var.get().startswith("Split: 2 pages")
        assert app.pending_split_ratio == pytest.approx(0.4, abs=0.01)
        _pump_until(
            app,
            lambda: (
                app.page_preview_after_image is not None
                and "2 output pages" in app.page_preview_after_title.cget("text")
            ),
            timeout=15.0,
        )
        assert app.page_preview_before_image.shape[:2] == spread.shape[:2]
        assert app.page_preview_after_image.shape[1] > spread.shape[1]

        app.apply_previewed_spread_split()

        assert [entry.name for entry in app.session.entries] == ["spread [L]", "spread [R]"]
        assert app.page_listbox.curselection() == (0,)
        assert app.apply_split_button.cget("state") == "disabled"
        assert app.preview_mode_var.get() == "Preview"
    finally:
        app._on_close()
        _pump_until(app, lambda: app._close_wait_job is None)
