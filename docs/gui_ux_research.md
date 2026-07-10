# GUI UX research and redesign notes

## Reference patterns

### NAPS2

The [official NAPS2 desktop screenshot](https://www.naps2.com/images/naps2-desktop-win.png)
keeps the current document visible as a page workspace. Scan, import, save, rotate, reorder, and
delete are persistent actions around that workspace. Scanner profile details stay in a narrow
context panel instead of becoming a separate mandatory step.

Useful pattern for UniScan: the user works on one document and chooses actions; they do not have
to remember which wizard step owns a page.

### ScanTailor Advanced

[ScanTailor Advanced](https://github.com/4lex4/scantailor-advanced) separates specialist page
operations into focused processing stages while keeping page thumbnails and the current page in
view. It is powerful, but its full stage pipeline would be excessive for UniScan's simpler
capture-to-PDF workflow.

Useful pattern for UniScan: keep common processing visible and place specialist correction tools
behind progressive disclosure.

## Problems observed in the previous UniScan GUI

- Four numbered tabs treated import, camera, review, and export as a strict wizard even though
  users frequently move between those actions.
- The initial Import page used most of the window for empty space while hiding the document pages.
- Review put the page list, eight rows of page tools, processing controls, and navigation into one
  fixed-height left panel. Processing controls were clipped at the standard 1280x800 window size.
- File selection required two steps: choose paths, then click a second import button.
- Export similarly required visiting another tab and managing a path field before the common PDF
  action.
- The page list used a bright default Tk listbox inside a dark application.
- Background jobs supported cancellation internally but exposed no Cancel control.

## First redesign implemented

![UniScan Workspace v1](images/workspace-v1.png)

- Workspace is now the default and keeps pages, original/processed previews, processing, and
  common export controls visible together.
- A persistent action bar provides Add files, Add folder, Paste, Camera, and Export PDF.
- File/folder actions import immediately; drops on the page list or preview import immediately.
- Common page operations remain beside the page list. Rare crop/replace/deskew/retake/split tools
  moved into a dedicated Page tools dialog.
- Processing moved into its own scrollable panel, so every setting remains reachable at smaller
  window heights.
- Camera controls are scrollable and return directly to Workspace.
- The page list now follows the dark theme and shows a live page count.
- The status bar exposes Cancel task.

## Second iteration implemented

- The center workspace now defaults to a single large processed preview instead of permanently
  squeezing original and processed pages side by side.
- `Processed`, `Original`, and `Compare` modes are always visible above the preview. Original-only
  mode skips the processing preview computation, which keeps inspection responsive.
- Common keyboard actions are available: `Ctrl+O` add files, `Ctrl+Shift+O` add a folder,
  `Ctrl+Shift+C` capture one frame, `Ctrl+E` export PDF, `F5` refresh preview, and page-list
  shortcuts for delete, rotate, reorder, and select all.

## Next UX iterations

1. Replace the text list with thumbnail cards while preserving multi-select and keyboard access.
2. Add an explicit New/Discard session action and a compact recent-session recovery surface.
3. Add Russian/English interface localization after labels and workflows stabilize.
4. Run the manual workflow with a real camera and record task time/click count for common jobs.
