"""Read-only Tk renderer for the Review pipeline cards."""

from __future__ import annotations

import customtkinter as ctk

from .review_pipeline import PipelineCard
from .theme import COLORS, TYPOGRAPHY, status_presentation


def _clear(frame: ctk.CTkFrame) -> None:
    for child in frame.winfo_children():
        child.destroy()


def _font(role: str) -> tuple[str, int, str]:
    size, weight = TYPOGRAPHY[role]
    return ("Segoe UI", size, weight)


def _render_signature(
    cards: tuple[PipelineCard, ...],
    placeholder: str,
) -> tuple[object, ...]:
    return (
        placeholder,
        tuple(
            (
                card.title,
                card.mode_label,
                card.status_label,
                card.reason.summary,
                card.controls,
            )
            for card in cards
        ),
    )


def _mode_text(card: PipelineCard) -> str:
    alternatives = tuple(control for control in card.controls if control != card.mode_label)
    text = f"Mode: {card.mode_label}"
    if alternatives:
        text += " | " + " / ".join(alternatives)
    return text


def _configure_card(widgets: tuple[object, ...], card: PipelineCard) -> None:
    presentation = status_presentation(card.state.status)
    _, title_label, mode_label, status_label, reason_label = widgets
    title_label.configure(text=card.title)
    mode_label.configure(text=_mode_text(card))
    status_label.configure(
        text=f"{presentation.glyph} {presentation.label}",
        fg_color=presentation.tint,
        text_color=presentation.foreground,
    )
    reason_label.configure(text=card.reason.summary)


def _render_card(parent: ctk.CTkFrame, card: PipelineCard) -> tuple[object, ...]:
    card_frame = ctk.CTkFrame(
        parent,
        width=156,
        height=130,
        corner_radius=8,
        fg_color=COLORS["surface.raised"],
        border_width=1,
        border_color=COLORS["border.default"],
    )
    card_frame.pack(side=ctk.LEFT, padx=(0, 6), pady=2)
    card_frame.pack_propagate(False)
    title_label = ctk.CTkLabel(
        card_frame,
        text="",
        anchor="w",
        text_color=COLORS["text.primary"],
        font=_font("section"),
    )
    title_label.pack(fill=ctk.X, padx=8, pady=(5, 0))
    mode_label = ctk.CTkLabel(
        card_frame,
        text="",
        anchor="w",
        text_color=COLORS["text.secondary"],
        font=_font("caption"),
    )
    mode_label.pack(fill=ctk.X, padx=8)
    status_label = ctk.CTkLabel(
        card_frame,
        text="",
        anchor="w",
        height=22,
        corner_radius=6,
        font=_font("status"),
    )
    status_label.pack(fill=ctk.X, padx=8, pady=(2, 1))
    reason_label = ctk.CTkLabel(
        card_frame,
        text="",
        anchor="w",
        justify="left",
        wraplength=140,
        text_color=COLORS["text.secondary"],
        font=_font("caption"),
    )
    reason_label.pack(fill=ctk.X, padx=8, pady=(1, 3))
    widgets = (card_frame, title_label, mode_label, status_label, reason_label)
    _configure_card(widgets, card)
    return widgets


def render_pipeline_strip(
    frame: ctk.CTkFrame,
    cards: tuple[PipelineCard, ...],
    *,
    placeholder: str = "Select one page to inspect the pipeline",
) -> None:
    """Render cards or an honest selection placeholder into ``frame``."""

    signature = _render_signature(cards, placeholder)
    if getattr(frame, "_uniscan_render_signature", None) == signature:
        return
    card_widgets = getattr(frame, "_uniscan_card_widgets", ())
    if cards and len(card_widgets) == len(cards):
        for widgets, card in zip(card_widgets, cards, strict=True):
            _configure_card(widgets, card)
        frame._uniscan_render_signature = signature
        return
    frame._uniscan_render_signature = signature
    _clear(frame)
    frame._uniscan_card_widgets = ()
    if not cards:
        ctk.CTkLabel(
            frame,
            text=placeholder,
            anchor="w",
            text_color=COLORS["text.secondary"],
            font=_font("body"),
        ).pack(fill=ctk.X, padx=10, pady=10)
        return
    frame._uniscan_card_widgets = tuple(_render_card(frame, card) for card in cards)


__all__ = ["render_pipeline_strip"]
