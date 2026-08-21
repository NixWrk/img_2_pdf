# UniScan shared UI components

UniScan is a Python 3.11 desktop application built with CustomTkinter and native Tk widgets. It
does not have a separate component library: most visible widgets are constructed inline in
`UnifiedScanApp._build_ui`, `_build_capture_tab`, `_build_pages_tab`, and the inline editor
methods in `src/uniscan/ui/app.py`.

## CameraHealth status component

- Path: `src/uniscan/ui/camera_health.py`
- Purpose: maps camera lifecycle states to a visible label and semantic colour.
- Props: `is_open`, `is_previewing`, `is_opening`, `error_text`, `detail`.

```python
"""Camera health status helpers for UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CameraHealth:
    label: str
    color: str


def camera_health_state(
    *,
    is_open: bool,
    is_previewing: bool,
    is_opening: bool = False,
    error_text: str | None = None,
    detail: str | None = None,
) -> CameraHealth:
    """Map camera state to a status label; ``detail`` (e.g. "1920x1080 @ 28 fps")
    is appended to the open/previewing labels."""
    if error_text:
        return CameraHealth(label="Camera: Error", color="#d94f4f")
    if is_opening:
        return CameraHealth(label="Camera: Opening...", color="#b8860b")
    suffix = f" ({detail})" if detail else ""
    if is_previewing:
        return CameraHealth(label=f"Camera: Previewing{suffix}", color="#2f9e44")
    if is_open:
        return CameraHealth(label=f"Camera: Open{suffix}", color="#0b7285")
    return CameraHealth(label="Camera: Closed", color="#6c757d")
```

## Preview mode selector

- Path: `src/uniscan/ui/app.py:1165`
- Purpose: switches the central viewer between processed, original, and side-by-side comparison.
- Props: `preview_mode_var`; command `_on_preview_mode_change`.

```python
preview_toolbar = ctk.CTkFrame(preview, fg_color="transparent")
preview_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
ctk.CTkLabel(
    preview_toolbar,
    text="Preview",
    font=ctk.CTkFont(size=15, weight="bold"),
).pack(side=ctk.LEFT)
self.preview_mode_selector = ctk.CTkSegmentedButton(
    preview_toolbar,
    values=["Processed", "Original", "Compare"],
    variable=self.preview_mode_var,
    command=self._on_preview_mode_change,
)
self.preview_mode_selector.pack(side=ctk.RIGHT)
```

## Page list

- Path: `src/uniscan/ui/app.py:1063`
- Purpose: multi-selection, reorder, context menu, keyboard actions, and page status text.
- Props: session entries and the selected index set.

```python
self.page_listbox = tk.Listbox(
    left,
    selectmode=tk.EXTENDED,
    width=30,
    height=8,
    bg="#202225",
    fg="#f2f2f2",
    selectbackground="#1f6aa5",
    selectforeground="#ffffff",
    activestyle="none",
    relief=tk.FLAT,
    borderwidth=0,
    highlightthickness=1,
    highlightbackground="#45484d",
    highlightcolor="#1f6aa5",
    font=("Segoe UI", 11),
    exportselection=False,
)
self.page_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
self.page_listbox.bind("<<ListboxSelect>>", self.on_page_select)
self.page_listbox.bind("<ButtonPress-1>", self._on_page_drag_start, add="+")
self.page_listbox.bind("<B1-Motion>", self._on_page_drag_motion, add="+")
self.page_listbox.bind("<ButtonRelease-1>", self._on_page_drag_end, add="+")
self.page_listbox.bind("<Button-3>", self._show_page_context_menu)
```

## Persistent task status

- Path: `src/uniscan/ui/app.py:811`
- Purpose: text-only progress/status surface plus cancellation affordance.
- Props: `status_var`, background-job state.

```python
self.status_frame = ctk.CTkFrame(container)
self.status_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=(0, 12))
status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var, anchor="w")
status_label.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=10, pady=7)
self.cancel_task_button = ctk.CTkButton(
    self.status_frame,
    text="Cancel task",
    width=90,
    height=26,
    fg_color="transparent",
    border_width=1,
    command=self.cancel_current_job,
    state=tk.DISABLED,
)
self.cancel_task_button.pack(side=ctk.RIGHT, padx=8, pady=5)
```

## Geometry canvases

- Paths: `src/uniscan/ui/app.py:4503`, `src/uniscan/ui/app.py:5187`,
  `src/uniscan/ui/app.py:5673`.
- Purpose: original/corrected panes for perspective, split, and wave editors.
- Appearance: native Tk canvases use a hard-coded black background; control points and lines are
  painted by editor-local callbacks.

