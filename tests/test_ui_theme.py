from __future__ import annotations

import pytest

from uniscan.ui.stage_state import StageStatus
from uniscan.ui.theme import (
    COLORS,
    bind_focus_ring,
    component_style,
    resolve_pair,
    status_presentation,
)


def _relative_luminance(value: str) -> float:
    channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_foundation_tokens_match_the_accepted_visual_audit() -> None:
    assert COLORS["surface.canvas"] == ("#F3F5F7", "#181A1D")
    assert COLORS["surface.panel"] == ("#FFFFFF", "#22262B")
    assert COLORS["surface.raised"] == ("#F8FAFC", "#2A2F35")
    assert COLORS["border.default"] == ("#CBD5E1", "#434B55")
    assert COLORS["text.primary"] == ("#0F172A", "#F8FAFC")
    assert COLORS["text.secondary"] == ("#475569", "#D6DCE4")
    assert COLORS["text.muted"] == ("#64748B", "#A7B0BC")
    assert COLORS["action.primary"] == ("#1D5F94", "#4EA3E3")
    assert COLORS["focus"] == ("#0B6BCB", "#7CC4FF")
    assert COLORS["danger"] == ("#B42318", "#FF7B72")
    assert COLORS["warning"] == ("#8A4B00", "#F2B84B")
    assert COLORS["success"] == ("#157347", "#59D98E")
    assert COLORS["edited"] == ("#6D28D9", "#C4A7FF")


@pytest.mark.parametrize("mode", (0, 1))
def test_normal_text_and_status_chips_meet_wcag_aa(mode: int) -> None:
    panel = COLORS["surface.panel"][mode]
    raised = COLORS["surface.raised"][mode]
    for token in ("text.primary", "text.secondary", "text.muted"):
        assert _contrast(COLORS[token][mode], panel) >= 4.5
    for token in ("text.primary", "text.secondary", "text.muted"):
        assert _contrast(COLORS[token][mode], raised) >= 4.5
    for status in StageStatus:
        presentation = status_presentation(status)
        assert _contrast(presentation.foreground[mode], presentation.tint[mode]) >= 4.5


@pytest.mark.parametrize(
    ("background", "foreground"),
    (
        ("action.primary", "action.text"),
        ("action.hover", "action.text"),
        ("danger", "danger.text"),
        ("danger.hover", "danger.text"),
        ("success", "success.text"),
        ("success.hover", "success.text"),
    ),
)
@pytest.mark.parametrize("mode", (0, 1))
def test_filled_actions_have_contrasting_text(
    background: str,
    foreground: str,
    mode: int,
) -> None:
    assert _contrast(COLORS[background][mode], COLORS[foreground][mode]) >= 4.5


def test_raw_tk_color_resolution_is_explicit() -> None:
    assert resolve_pair("surface.canvas", "Light") == "#F3F5F7"
    assert resolve_pair("surface.canvas", "dark") == "#181A1D"
    assert resolve_pair(COLORS["focus"], 0) == "#0B6BCB"
    assert resolve_pair(COLORS["focus"], 1) == "#7CC4FF"
    with pytest.raises(ValueError, match="appearance_mode"):
        resolve_pair("surface.canvas", "system")


def test_stage_status_semantics_do_not_depend_on_color_alone() -> None:
    rejected = status_presentation(StageStatus.REJECTED)
    stale = status_presentation("Stale")
    applied = status_presentation("applied")

    assert (rejected.glyph, rejected.label, rejected.foreground) == (
        "x",
        "Rejected",
        COLORS["danger"],
    )
    assert (stale.glyph, stale.label, stale.foreground) == (
        "~",
        "Stale",
        COLORS["warning"],
    )
    assert (applied.glyph, applied.label) == ("+", "Applied")
    assert len({status_presentation(status).glyph for status in StageStatus}) > 1


def test_component_styles_are_local_copies_with_explicit_states() -> None:
    secondary = component_style("secondary_button")
    assert secondary == {
        "fg_color": COLORS["surface.raised"],
        "hover_color": COLORS["tint.neutral"],
        "text_color": COLORS["text.primary"],
        "text_color_disabled": COLORS["text.muted"],
        "border_color": COLORS["border.default"],
        "border_width": 1,
    }
    secondary["border_width"] = 99
    assert component_style("secondary_button")["border_width"] == 1
    assert component_style("segmented")["selected_color"] == COLORS["action.primary"]
    assert component_style("option_menu")["dropdown_fg_color"] == COLORS["surface.panel"]
    assert component_style("progress")["progress_color"] == COLORS["focus"]
    with pytest.raises(ValueError, match="unknown component style"):
        component_style("global-theme-mutation")


@pytest.mark.parametrize("mode", (0, 1))
def test_component_state_pairs_meet_wcag_aa(mode: int) -> None:
    pairs = (
        (COLORS["action.primary"], COLORS["action.text"]),
        (COLORS["action.hover"], COLORS["action.text"]),
        (COLORS["surface.raised"], COLORS["text.primary"]),
        (COLORS["tint.neutral"], COLORS["text.primary"]),
        (COLORS["surface.panel"], COLORS["text.primary"]),
    )
    for background, foreground in pairs:
        assert _contrast(background[mode], foreground[mode]) >= 4.5


def test_focus_ring_is_local_and_restores_the_component_style() -> None:
    class Widget:
        def __init__(self) -> None:
            self.options = {"border_color": COLORS["border.default"], "border_width": 0}
            self.bindings = {}

        def cget(self, name: str):
            return self.options[name]

        def configure(self, **options) -> None:
            self.options.update(options)

        def bind(self, sequence: str, command, add: str) -> None:
            assert add == "+"
            self.bindings[sequence] = command

    widget = Widget()
    assert bind_focus_ring(widget) is widget
    widget.bindings["<FocusIn>"]()
    assert widget.options["border_color"] == COLORS["focus"]
    assert widget.options["border_width"] == 1
    widget.bindings["<FocusOut>"]()
    assert widget.options["border_color"] == COLORS["border.default"]
    assert widget.options["border_width"] == 0
