# UniScan routes and UI states

UniScan has no URL router. Navigation is stateful desktop navigation inside one `UnifiedScanApp`.

| Route/state | Entry method | Layout | Exit/transition |
| --- | --- | --- | --- |
| Workspace | `_build_pages_tab` / `go_to_review_tab` | three-column document editor | Camera tab or an inline editor |
| Camera | `_build_capture_tab` / `go_to_camera_tab` | controls + live preview | Open Workspace |
| Import options | `open_import_options_dialog` | 420×245 transient dialog | Cancel / Save |
| Export options | `open_export_dialog` | transient dialog with PDF/images modes | Cancel / Export |
| Perspective editor | `_open_corner_editor_dialog` | inline source and corrected canvases | Done / close callback |
| Spread split editor | `open_split_editor` | inline source and output panes | Cancel / Create 2 pages |
| Wave editor | `open_dewarp_points_editor` | inline source and corrected canvases | Reset / Apply points / Done |
| Advanced processing | `open_review_processing_dialog` | inline advanced parameter form | Close / Preview / Apply |

## Application entry

```python
# src/uniscan/cli.py
from uniscan.ui import run_app
return run_app()

# src/uniscan/ui/app.py
def run_app() -> int:
    try:
        app = UnifiedScanApp()
    except (SessionInUseError, UnsafeSessionLockError, OSError) as exc:
        print(f"UniScan startup failed: {exc}", file=sys.stderr)
        return 2
    app.mainloop()
    return 0
```

## Keyboard navigation

- Global: `Ctrl+O`, `Ctrl+Shift+O`, `Ctrl+Shift+C`, `Ctrl+E`, `F5`.
- Page list: `Delete`, `Ctrl+Left/Right`, `Alt+Up/Down`, `Ctrl+A`.
- No code-level bindings were found for Undo/Redo, zoom, fit-to-page, stage traversal, or editor
  keyboard manipulation.

