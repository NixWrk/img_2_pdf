from __future__ import annotations

import os

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
        assert app.tabs.get() == app.tab_import_name
        assert app.status_var.get() == "Ready"
        assert len(app.session) == 0
        assert app.preprocess_preset_var.get() == "Document"
        app.preprocess_illumination_var.set(True)
        assert app._current_preprocess_settings().correct_illumination is True
        assert app.import_files_entry.winfo_exists()
        assert app._drag_drop_error is None

        first = np.full((90, 120, 3), 220, dtype=np.uint8)
        second = np.full((90, 120, 3), 180, dtype=np.uint8)
        app.session.add_image(name="first.png", image=first)
        app.session.add_image(name="second.png", image=second)
        app.refresh_page_list(keep_index=1)
        app.update()
        assert app._single_selected_entry()[1].name == "second.png"
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
    finally:
        app._on_close()

    assert not app.autosave_path.exists()
