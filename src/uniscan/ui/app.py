"""Unified application shell."""

from __future__ import annotations

import customtkinter as ctk
import tkinter as tk


class UnifiedScanApp(ctk.CTk):
    """Main window shell for all future merged functionality."""

    def __init__(self) -> None:
        super().__init__()
        self.title("UniScan")
        self.geometry("1280x800")
        self.minsize(1024, 680)

        self._build_ui()

    def _build_ui(self) -> None:
        container = ctk.CTkFrame(self)
        container.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        title = ctk.CTkLabel(
            container,
            text="UniScan - Unified Document Processing",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.pack(anchor="w", padx=12, pady=(12, 8))

        self.tabs = ctk.CTkTabview(container)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

        for name in ("Capture", "Import", "Pages", "Export", "Jobs"):
            tab = self.tabs.add(name)
            body = ctk.CTkLabel(
                tab,
                text=f"{name} module: implementation in progress",
                anchor="w",
            )
            body.pack(fill=ctk.X, padx=16, pady=16)

        status_frame = ctk.CTkFrame(container)
        status_frame.pack(fill=ctk.X, padx=12, pady=(0, 12))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=ctk.X, padx=10, pady=8)


def run_app() -> int:
    app = UnifiedScanApp()
    app.mainloop()
    return 0
