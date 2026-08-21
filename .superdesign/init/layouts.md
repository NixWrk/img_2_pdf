# UniScan layouts

## Application shell

- Path: `src/uniscan/ui/app.py:718`
- Description: one 1280×800 desktop window with persistent header, toolbar, bottom status bar, and
  a tab view containing Workspace and Camera. Workspace itself is a three-column editor.
- Full source of the shell method:

```python
def _build_ui(self) -> None:
    container = ctk.CTkFrame(self)
    container.pack(fill=ctk.BOTH, expand=True, padx=12, pady=12)

    header = ctk.CTkFrame(container, fg_color=("#dbdbdb", "#2b2b2b"))
    header.pack(fill=ctk.X, padx=12, pady=(10, 4))
    brand = ctk.CTkFrame(header, fg_color="transparent")
    brand.pack(side=ctk.LEFT)
    ctk.CTkLabel(
        brand,
        text="UniScan",
        font=ctk.CTkFont(size=24, weight="bold"),
    ).pack(anchor="w")
    ctk.CTkLabel(
        brand,
        text="Capture, clean and export documents",
        text_color=("#60646c", "#a0a4ab"),
    ).pack(anchor="w")
    ctk.CTkLabel(
        header,
        textvariable=self.camera_health_var,
        text_color=("#60646c", "#a0a4ab"),
    ).pack(side=ctk.RIGHT, padx=(8, 0))
    ctk.CTkLabel(
        header,
        textvariable=self.page_count_var,
        text_color=("#60646c", "#a0a4ab"),
    ).pack(side=ctk.RIGHT, padx=8)

    toolbar = ctk.CTkFrame(container)
    toolbar.pack(fill=ctk.X, padx=12, pady=(6, 4))
    self.toolbar_add_files_button = ctk.CTkButton(
        toolbar, text="+ Add files", width=110, command=self.quick_add_files
    )
    self.toolbar_add_files_button.pack(side=ctk.LEFT, padx=(8, 4), pady=8)
    self.toolbar_add_folder_button = ctk.CTkButton(
        toolbar, text="Add folder", width=105, fg_color="transparent",
        border_width=1, command=self.quick_add_folder,
    )
    self.toolbar_add_folder_button.pack(side=ctk.LEFT, padx=4, pady=8)
    self.toolbar_paste_button = ctk.CTkButton(
        toolbar, text="Paste", width=80, fg_color="transparent",
        border_width=1, command=self.import_from_clipboard,
    )
    self.toolbar_paste_button.pack(side=ctk.LEFT, padx=4, pady=8)
    self.toolbar_camera_button = ctk.CTkButton(
        toolbar, text="Camera", width=90, fg_color="transparent",
        border_width=1, command=self.go_to_camera_tab,
    )
    self.toolbar_camera_button.pack(side=ctk.LEFT, padx=4, pady=8)
    self.toolbar_import_options_button = ctk.CTkButton(
        toolbar, text="Import options...", width=125, fg_color="transparent",
        border_width=1, command=self.open_import_options_dialog,
    )
    self.toolbar_import_options_button.pack(side=ctk.LEFT, padx=4, pady=8)
    self.toolbar_export_pdf_button = ctk.CTkButton(
        toolbar, text="Export PDF", width=120, fg_color="#2f855a",
        hover_color="#276749", command=self.quick_export_pdf,
    )
    self.toolbar_export_pdf_button.pack(side=ctk.RIGHT, padx=(4, 8), pady=8)
    self.toolbar_export_options_button = ctk.CTkButton(
        toolbar, text="Export options...", width=135, fg_color="transparent",
        border_width=1, command=self.open_export_dialog,
    )
    self.toolbar_export_options_button.pack(side=ctk.RIGHT, padx=4, pady=8)

    self.status_frame = ctk.CTkFrame(container)
    self.status_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=(0, 12))
    status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var, anchor="w")
    status_label.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=10, pady=7)
    self.cancel_task_button = ctk.CTkButton(
        self.status_frame, text="Cancel task", width=90, height=26,
        fg_color="transparent", border_width=1, command=self.cancel_current_job,
        state=tk.DISABLED,
    )
    self.cancel_task_button.pack(side=ctk.RIGHT, padx=8, pady=5)

    self.tabs = ctk.CTkTabview(container, command=self._sync_camera_to_active_tab)
    self.tabs.pack(fill=ctk.BOTH, expand=True, padx=12, pady=(4, 8))
    self.pages_tab = self.tabs.add(self.tab_review_name)
    self.camera_tab = self.tabs.add(self.tab_camera_name)
    self._build_pages_tab(self.pages_tab)
    self._build_capture_tab(self.camera_tab)
    self.tabs.set(self.tab_review_name)
```

## Workspace layout

- Path: `src/uniscan/ui/app.py:1033`
- Structure: fixed 290 px page rail | flexible preview | fixed 270–280 px scrollable processing
  rail.
- Center header: Preview plus segmented `Processed / Original / Compare` modes.
- Right rail: sequential geometry buttons, then document/output/preset selectors, stage-specific
  controls, Preview/Advanced, and Apply.
- Inline editors temporarily replace the three-column workspace host.

## Camera layout

- Path: `src/uniscan/ui/app.py:861`
- Structure: 340 px scrollable control rail | flexible camera preview.
- Primary action: green `Capture Page`; burst, experimental live detection, device and capture mode
  selection are placed underneath.

