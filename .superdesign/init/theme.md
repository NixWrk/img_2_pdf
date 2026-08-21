# UniScan theme

## Compact token summary

### Framework defaults

- Component system: CustomTkinter 5.2 plus native Tk widgets.
- Appearance mode: CustomTkinter default `System`; the app does not call
  `set_appearance_mode` or expose a theme switch.
- Base colour theme: CustomTkinter default blue; no explicit `set_default_color_theme` call.
- Typeface: CustomTkinter platform default; native page list explicitly uses Segoe UI 11.
- Type sizes: brand 24 bold; section titles 15–18 bold; warning 11; default body otherwise.
- Outer spacing: 12 px shell; 8–10 px widget padding; 4–6 px compact control gaps.
- Minimum window: 1024×680; default 1280×800.
- Radii/shadows: inherited from CustomTkinter; no application token layer.
- Breakpoints/responsive rules: none; desktop grid uses fixed side rails.

### Explicit colours

| Semantic role | Light | Dark / fixed |
| --- | --- | --- |
| Header surface | `#dbdbdb` | `#2b2b2b` |
| Secondary text | `#60646c` | `#a0a4ab` |
| Primary success/export | `#2f855a` | same |
| Primary success hover | `#276749` | same |
| Destructive | `#b42318` | same |
| Destructive hover | `#912018` | same |
| Review warning | `#8a5a00` | `#d6a84b` |
| Camera error | `#d94f4f` | same |
| Camera opening | `#b8860b` | same |
| Camera previewing | `#2f9e44` | same |
| Camera open | `#0b7285` | same |
| Camera closed | `#6c757d` | same |
| Native list surface | — | `#202225` fixed |
| Native list text | — | `#f2f2f2` fixed |
| Selected row / focus | — | `#1f6aa5` fixed |
| Canvas | — | black fixed |

### Current semantic gaps

- No token distinction exists for pipeline states `Auto`, `Off`, `Not needed`, `Applied`,
  `Rejected`, or `Edited`.
- Camera state colours are semantic but status relies on unverified contrast in both appearance
  modes.
- List and canvas widgets are fixed dark even if the operating system selects light appearance.
- Focus, disabled, error, loading, and hover mostly inherit widget defaults instead of a documented
  application system.

## Raw source excerpts

```python
# src/uniscan/ui/app.py
self.title("UniScan")
self.geometry("1280x800")
self.minsize(1024, 680)

header = ctk.CTkFrame(container, fg_color=("#dbdbdb", "#2b2b2b"))
ctk.CTkLabel(
    brand,
    text="UniScan",
    font=ctk.CTkFont(size=24, weight="bold"),
)
ctk.CTkLabel(
    brand,
    text="Capture, clean and export documents",
    text_color=("#60646c", "#a0a4ab"),
)

self.toolbar_export_pdf_button = ctk.CTkButton(
    toolbar,
    text="Export PDF",
    width=120,
    fg_color="#2f855a",
    hover_color="#276749",
    command=self.quick_export_pdf,
)
```

```python
# src/uniscan/ui/camera_health.py
if error_text:
    return CameraHealth(label="Camera: Error", color="#d94f4f")
if is_opening:
    return CameraHealth(label="Camera: Opening...", color="#b8860b")
if is_previewing:
    return CameraHealth(label=f"Camera: Previewing{suffix}", color="#2f9e44")
if is_open:
    return CameraHealth(label=f"Camera: Open{suffix}", color="#0b7285")
return CameraHealth(label="Camera: Closed", color="#6c757d")
```

