"""Small, shared visual vocabulary for the Tk/CustomTkinter UI.

Pairs are ordered ``(light, dark)`` because that is also the shape accepted by
CustomTkinter.  Raw Tk widgets cannot consume a pair, so ``resolve_pair`` is
the one place where the current appearance mode is reduced to a concrete
colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, TypeAlias

import customtkinter as ctk

from .stage_state import StageStatus

ColorPair: TypeAlias = tuple[str, str]


# Calm Technical Dark keeps the document brighter than the application chrome.
# These pairs are the compact token proposal accepted in the visual UX audit.
COLORS: Final[Mapping[str, ColorPair]] = {
    "surface.canvas": ("#F3F5F7", "#181A1D"),
    "surface.panel": ("#FFFFFF", "#22262B"),
    "surface.raised": ("#F8FAFC", "#2A2F35"),
    "border.default": ("#CBD5E1", "#434B55"),
    "text.primary": ("#0F172A", "#F8FAFC"),
    "text.secondary": ("#475569", "#D6DCE4"),
    "text.muted": ("#64748B", "#A7B0BC"),
    "action.primary": ("#1D5F94", "#4EA3E3"),
    "action.hover": ("#174D78", "#78BDEC"),
    "action.text": ("#FFFFFF", "#0F172A"),
    "focus": ("#0B6BCB", "#7CC4FF"),
    "danger": ("#B42318", "#FF7B72"),
    "danger.hover": ("#8E1C13", "#FF9A94"),
    "danger.text": ("#FFFFFF", "#0F172A"),
    "warning": ("#8A4B00", "#F2B84B"),
    "success": ("#157347", "#59D98E"),
    "success.hover": ("#105C39", "#78E2A5"),
    "success.text": ("#FFFFFF", "#0F172A"),
    "edited": ("#6D28D9", "#C4A7FF"),
    # Tints are intentionally separate from foregrounds: they provide a
    # semantic chip background without making colour the only status cue.
    "tint.neutral": ("#E2E8F0", "#343B44"),
    "tint.info": ("#DBEAFE", "#123247"),
    "tint.success": ("#DCFCE7", "#173A2B"),
    "tint.warning": ("#FEF3C7", "#493716"),
    "tint.edited": ("#F3E8FF", "#342653"),
    "tint.danger": ("#FEE2E2", "#4A2328"),
}

TYPOGRAPHY: Final[Mapping[str, tuple[int, str]]] = {
    "display": (24, "bold"),
    "heading": (18, "bold"),
    "section": (15, "bold"),
    "body": (13, "normal"),
    "caption": (12, "normal"),
    "metrics": (12, "normal"),
    "status": (12, "bold"),
}

COMPONENT_STYLES: Final[Mapping[str, Mapping[str, object]]] = {
    "primary_button": {
        "fg_color": COLORS["action.primary"],
        "hover_color": COLORS["action.hover"],
        "text_color": COLORS["action.text"],
        "text_color_disabled": COLORS["text.muted"],
    },
    "secondary_button": {
        "fg_color": COLORS["surface.raised"],
        "hover_color": COLORS["tint.neutral"],
        "text_color": COLORS["text.primary"],
        "text_color_disabled": COLORS["text.muted"],
        "border_color": COLORS["border.default"],
        "border_width": 1,
    },
    "danger_button": {
        "fg_color": COLORS["danger"],
        "hover_color": COLORS["danger.hover"],
        "text_color": COLORS["danger.text"],
        "text_color_disabled": COLORS["text.muted"],
    },
    "success_button": {
        "fg_color": COLORS["success"],
        "hover_color": COLORS["success.hover"],
        "text_color": COLORS["success.text"],
        "text_color_disabled": COLORS["text.muted"],
    },
    "segmented": {
        "fg_color": COLORS["surface.raised"],
        "selected_color": COLORS["action.primary"],
        "selected_hover_color": COLORS["action.hover"],
        "unselected_color": COLORS["surface.panel"],
        "unselected_hover_color": COLORS["tint.neutral"],
        "text_color": COLORS["text.primary"],
        "text_color_disabled": COLORS["text.muted"],
        "border_width": 1,
    },
    "option_menu": {
        "fg_color": COLORS["surface.raised"],
        "button_color": COLORS["action.primary"],
        "button_hover_color": COLORS["action.hover"],
        "text_color": COLORS["text.primary"],
        "text_color_disabled": COLORS["text.muted"],
        "dropdown_fg_color": COLORS["surface.panel"],
        "dropdown_hover_color": COLORS["tint.neutral"],
        "dropdown_text_color": COLORS["text.primary"],
    },
    "progress": {
        "fg_color": COLORS["tint.neutral"],
        "progress_color": COLORS["focus"],
        "border_color": COLORS["border.default"],
        "border_width": 1,
    },
    "focus_ring": {
        "border_color": COLORS["focus"],
        "border_width": 1,
    },
}


def resolve_pair(
    pair_or_name: ColorPair | str,
    appearance_mode: str | int | None = None,
) -> str:
    """Resolve a colour pair (or token name) for a raw Tk widget.

    ``appearance_mode`` accepts CustomTkinter's ``"Light"``/``"Dark"`` or
    their numeric equivalents (0/1).  If omitted, the active CustomTkinter
    mode is used, with Light as a safe fallback for test doubles/headless use.
    """

    pair = COLORS[pair_or_name] if isinstance(pair_or_name, str) else pair_or_name
    if len(pair) != 2:
        raise ValueError("a colour pair must contain light and dark values")
    mode = appearance_mode
    if mode is None:
        try:
            mode = ctk.get_appearance_mode()
        except (AttributeError, RuntimeError):
            mode = "Light"
    if isinstance(mode, str):
        normalized = mode.lower()
        if normalized not in {"light", "dark"}:
            raise ValueError("appearance_mode must be Light/Dark or 0/1")
        index = 1 if normalized == "dark" else 0
    elif mode in (0, 1):
        index = int(mode)
    else:
        raise ValueError("appearance_mode must be Light/Dark or 0/1")
    return pair[index]


def color_pair(name: str) -> ColorPair:
    """Return a named pair for CustomTkinter options."""

    return COLORS[name]


def component_style(name: str) -> dict[str, object]:
    """Return isolated kwargs for one CustomTkinter component instance."""

    try:
        return dict(COMPONENT_STYLES[name])
    except KeyError:
        raise ValueError(f"unknown component style: {name!r}") from None


def bind_focus_ring(widget):
    """Add a component-local keyboard focus ring without mutating the global theme."""

    resting_color = widget.cget("border_color")
    resting_width = widget.cget("border_width")
    focus = component_style("focus_ring")

    def show(_event=None) -> None:
        widget.configure(**focus)

    def hide(_event=None) -> None:
        widget.configure(border_color=resting_color, border_width=resting_width)

    widget.bind("<FocusIn>", show, add="+")
    widget.bind("<FocusOut>", hide, add="+")
    return widget


@dataclass(frozen=True, slots=True)
class StageStatusPresentation:
    """Text and colours used by a status row/chip."""

    glyph: str
    label: str
    foreground: ColorPair
    tint: ColorPair


_STATUS_PRESENTATIONS: Final[Mapping[StageStatus, StageStatusPresentation]] = {
    StageStatus.IDLE: StageStatusPresentation(
        "-", "Idle", COLORS["text.secondary"], COLORS["tint.neutral"]
    ),
    StageStatus.RUNNING: StageStatusPresentation(
        ">", "Running", COLORS["action.primary"], COLORS["tint.info"]
    ),
    StageStatus.NOT_NEEDED: StageStatusPresentation(
        "=", "Not needed", COLORS["text.secondary"], COLORS["tint.neutral"]
    ),
    StageStatus.APPLIED: StageStatusPresentation(
        "+", "Applied", COLORS["success"], COLORS["tint.success"]
    ),
    StageStatus.REJECTED: StageStatusPresentation(
        "x", "Rejected", COLORS["danger"], COLORS["tint.danger"]
    ),
    StageStatus.EDITED: StageStatusPresentation(
        "~", "Edited", COLORS["edited"], COLORS["tint.edited"]
    ),
    StageStatus.STALE: StageStatusPresentation(
        "~", "Stale", COLORS["warning"], COLORS["tint.warning"]
    ),
    StageStatus.ERROR: StageStatusPresentation(
        "!", "Error", COLORS["danger"], COLORS["tint.danger"]
    ),
}


def status_presentation(status: StageStatus | str) -> StageStatusPresentation:
    """Return a stable, non-colour-only presentation for a stage status."""

    if isinstance(status, str):
        normalized = status.strip().lower().replace(" ", "_")
        for key, item in _STATUS_PRESENTATIONS.items():
            if normalized in {key.value, item.label.lower().replace(" ", "_")}:
                return item
        raise ValueError(f"unknown stage status: {status!r}")
    try:
        return _STATUS_PRESENTATIONS[status]
    except KeyError:
        raise ValueError(f"unknown stage status: {status!r}") from None


__all__ = [
    "COLORS",
    "COMPONENT_STYLES",
    "ColorPair",
    "StageStatusPresentation",
    "TYPOGRAPHY",
    "bind_focus_ring",
    "color_pair",
    "component_style",
    "resolve_pair",
    "status_presentation",
]
