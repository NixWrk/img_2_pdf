from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="Tk smoke test needs Windows desktop or Xvfb",
)


def test_gui_constructs_with_all_tabs_and_closes_cleanly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UNISCAN_STATE_DIR", str(tmp_path / "state"))
    from uniscan.ui.app import UnifiedScanApp

    app = UnifiedScanApp()
    try:
        app.withdraw()
        app.update()
        assert app.tabs.get() == app.tab_review_name
        assert app.status_var.get() == "Ready"
        assert len(app.session) == 0
        assert app.preprocess_preset_var.get() == "Document"
        assert app.dewarp_method_var.get() == "None"
        assert app.deskew_method_var.get() == "Hybrid (recommended)"
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
        app.open_dewarp_points_editor()
        app.update()
        dewarp_editors = [
            child
            for child in app.winfo_children()
            if hasattr(child, "title") and child.title() == "Adjust dewarp control points"
        ]
        assert len(dewarp_editors) == 1
        dewarp_editors[0].destroy()
        preview_calls: list[str] = []
        original_preview = app._review_after_image

        def tracked_preview(entry, before):
            preview_calls.append(entry.name)
            return original_preview(entry, before)

        monkeypatch.setattr(app, "_review_after_image", tracked_preview)
        app.preview_mode_var.set("Original")
        app._on_preview_mode_change()
        assert preview_calls == []
        assert app.page_preview_before_frame.winfo_manager() == "grid"
        assert app.page_preview_after_frame.winfo_manager() == ""
        app.preview_mode_var.set("Compare")
        app._on_preview_mode_change()
        assert preview_calls == ["second.png"]
        assert app.page_preview_before_frame.winfo_manager() == "grid"
        assert app.page_preview_after_frame.winfo_manager() == "grid"
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
    finally:
        app._on_close()

    assert not app.autosave_path.exists()
