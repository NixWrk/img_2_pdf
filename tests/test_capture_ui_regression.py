from __future__ import annotations

import json

from scripts.capture_ui_regression import (
    CaptureSpec,
    FOCUS_SCENE_WIDGET,
    _pipeline_capture_is_consistent,
    _preview_capture_is_idle,
    capture_filename,
    capture_matrix,
    manifest_entry,
)


def test_capture_matrix_is_stable_and_covers_the_manual_review_axes() -> None:
    matrix = capture_matrix()
    assert len(matrix) == 12
    assert {item.theme for item in matrix} == {"Light", "Dark"}
    assert {(item.width, item.height) for item in matrix} == {(1280, 800), (1024, 680)}
    assert {item.scene for item in matrix} == {"workspace", "advanced", "keyboard-focus"}
    assert matrix == capture_matrix()


def test_capture_filename_is_safe_and_descriptive() -> None:
    spec = CaptureSpec("Dark", 1024, 680, "keyboard-focus")
    assert capture_filename(spec) == "dark-1024x680-keyboard-focus.png"
    assert "/" not in capture_filename(spec)
    assert "\\" not in capture_filename(spec)


def test_manifest_entry_has_requested_evidence_fields() -> None:
    spec = CaptureSpec("Light", 1280, 800, "workspace")
    row = manifest_entry(spec, file="light-1280x800-workspace.png", head="abc123", pixel_size=(1278, 798))
    json.dumps(row)
    assert row["scene"] == "workspace"
    assert row["theme"] == "Light"
    assert row["width"] == 1280
    assert row["height"] == 800
    assert row["file"] == "light-1280x800-workspace.png"
    assert row["HEAD"] == "abc123"
    assert row["pixel_size"] == {"width": 1278, "height": 798}


def test_capture_invariants_cancel_running_preview_before_screenshot() -> None:
    class Worker:
        def __init__(self, alive: bool) -> None:
            self._alive = alive

        def is_alive(self) -> bool:
            return self._alive

    class FakeApp:
        review_preview_job = "after-id"
        review_preview_threads = [Worker(True)]
        review_preview_thread = None

    assert not _preview_capture_is_idle(FakeApp())
    FakeApp.review_preview_job = None
    FakeApp.review_preview_threads = []
    assert _preview_capture_is_idle(FakeApp())


def test_keyboard_focus_scene_targets_visible_styled_preview_control() -> None:
    assert FOCUS_SCENE_WIDGET == "preview_fit_button"


def test_capture_fixture_requires_real_pipeline_cards_for_one_selection() -> None:
    assert _pipeline_capture_is_consistent(selected_count=1, card_count=7)
    assert _pipeline_capture_is_consistent(selected_count=1, card_count=8)
    assert not _pipeline_capture_is_consistent(selected_count=0, card_count=7)
    assert not _pipeline_capture_is_consistent(selected_count=1, card_count=1)
