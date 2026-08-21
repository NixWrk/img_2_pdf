"""Read-only Tk renderer for the Review pipeline cards."""

from __future__ import annotations

import customtkinter as ctk

from .review_pipeline import PipelineCard


_STATUS_COLORS = {
    "Idle": ("#6b7280", "#a1a1aa"),
    "Running": ("#2563eb", "#60a5fa"),
    "Not needed": ("#6b7280", "#a1a1aa"),
    "Applied": ("#15803d", "#4ade80"),
    "Rejected": ("#b45309", "#fbbf24"),
    "Edited": ("#7c3aed", "#c4b5fd"),
    "Stale": ("#b45309", "#fbbf24"),
    "Error": ("#b91c1c", "#f87171"),
}


def _clear(frame: ctk.CTkFrame) -> None:
    for child in frame.winfo_children():
        child.destroy()


def _render_card(parent: ctk.CTkFrame, card: PipelineCard) -> None:
    card_frame = ctk.CTkFrame(parent, width=140, height=94, corner_radius=8)
    card_frame.pack(side=ctk.LEFT, padx=(0, 6), pady=2)
    card_frame.pack_propagate(False)
    ctk.CTkLabel(
        card_frame,
        text=card.title,
        anchor="w",
        font=ctk.CTkFont(size=11, weight="bold"),
    ).pack(fill=ctk.X, padx=8, pady=(5, 0))
    status_text = f"{card.mode_label} · {card.status_label}" if card.controls else card.status_label
    ctk.CTkLabel(
        card_frame,
        text=status_text,
        anchor="w",
        text_color=_STATUS_COLORS.get(card.status_label),
        font=ctk.CTkFont(size=10, weight="bold"),
    ).pack(fill=ctk.X, padx=8)
    ctk.CTkLabel(
        card_frame,
        text=card.reason.summary,
        anchor="w",
        justify="left",
        wraplength=124,
        font=ctk.CTkFont(size=9),
    ).pack(fill=ctk.X, padx=8)
    if card.controls:
        ctk.CTkLabel(
            card_frame,
            text="Available: " + "/".join(card.controls),
            anchor="w",
            text_color=("#71717a", "#a1a1aa"),
            font=ctk.CTkFont(size=8),
        ).pack(fill=ctk.X, padx=8, pady=(0, 3))


def render_pipeline_strip(
    frame: ctk.CTkFrame,
    cards: tuple[PipelineCard, ...],
    *,
    placeholder: str = "Select one page to inspect the pipeline",
) -> None:
    """Render cards or an honest selection placeholder into ``frame``."""

    _clear(frame)
    if not cards:
        ctk.CTkLabel(
            frame,
            text=placeholder,
            anchor="w",
            text_color=("#60646c", "#a1a4ab"),
        ).pack(fill=ctk.X, padx=10, pady=10)
        return
    for card in cards:
        _render_card(frame, card)


__all__ = ["render_pipeline_strip"]
