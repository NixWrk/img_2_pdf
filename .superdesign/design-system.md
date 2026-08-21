# UniScan current design system and product context

## Product

UniScan is a Windows-first desktop document scanner for importing photos/PDFs or capturing from a
camera, correcting document geometry and lighting non-destructively, comparing the original and
processed result, applying recipes to selected/all pages, and exporting a merged PDF or images.
The primary user is an operator digitising photographed pages, books, receipts, tables, or mixed
multi-page documents while preserving text, edges, and page order.

## Current information architecture

1. Persistent header and toolbar: import files/folder/clipboard, Camera, import options, Export PDF,
   export options.
2. Workspace: page selection and ordering → preview/compare → processing controls.
3. Camera tab: capture controls and live preview.
4. Inline specialist editors: perspective, split, wave correction, advanced processing.
5. Persistent bottom task/status surface with cancellation.

## Current visual system

- Platform: CustomTkinter with native Tk widgets.
- Appearance: follows OS through CustomTkinter's default System mode; no visible theme switch.
- Typography: platform default, Segoe UI 11 for the native page list; title 24 bold; section titles
  15–18 bold.
- Spacing: dense desktop tool, generally 4/6/8/10/12 px increments.
- Primary action: default CustomTkinter blue; capture/export are green `#2f855a` with hover
  `#276749`.
- Destructive: `#b42318`, hover `#912018`.
- Secondary text: `#60646c` light / `#a0a4ab` dark.
- Canvas and native page list are fixed dark (`black`, `#202225`) regardless of OS appearance.
- Side rails are fixed around 270–340 px; central preview flexes.

## Interaction principles already present

- Pages remain in one recoverable session and autosave.
- Preview is non-destructive; Apply commits full-resolution processing.
- Export publishes committed pages rather than silently replaying uncommitted controls.
- Long work runs in the background with status messages and cancellation.
- Geometry editors show source and corrected panes and preserve full-resolution pixels.

## Constraints for any Superdesign reproduction

- Reproduce the existing desktop tool before proposing changes.
- Keep the UniScan name as text; there is no repository logo asset.
- Use source screenshots as temporary references, not Brand Assets.
- Treat all generated HTML as design reference for a Tkinter implementation, not production code.
- Do not invent pipeline decisions, results, model confidence, or document imagery.
- Do not create visual variants unless the user explicitly approves the generation round.

