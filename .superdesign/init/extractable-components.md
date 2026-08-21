# UniScan extractable components

The codebase is a Tk desktop application, so extraction means reproducing visible structures as
Petite-Vue templates for the Superdesign canvas; it does not imply that those HTML components can
be reused directly by the Python application.

## AppShell

- Source: `src/uniscan/ui/app.py:718`
- Category: layout
- Description: header, import/camera/export toolbar, tab body, and persistent task status.
- Extractable props: `activeTab`, `pageCount`, `cameraState`, `taskStatus`, `taskCancelable`.
- Hardcoded: UniScan wordmark text, toolbar action labels, two tab labels.

## WorkspaceThreePane

- Source: `src/uniscan/ui/app.py:1033`
- Category: layout
- Description: page rail, original/processed viewer, and scrollable processing rail.
- Extractable props: `selectedPage`, `previewMode`, `pageCount`, `applyToAll`, `isBusy`.
- Hardcoded: panel labels, order of common page actions, stage control labels.

## ProcessingPipeline

- Source: `src/uniscan/ui/app.py:1217`
- Category: layout
- Description: geometry sequence followed by appearance, cleanup, layout, preview, and Apply.
- Extractable props: `orientationState`, `perspectiveState`, `splitState`, `dewarpState`,
  `deskewState`, `lightingState`, `cleanupState`, `layoutState`.
- Hardcoded: stage order and English labels.

## PreviewCompare

- Source: `src/uniscan/ui/app.py:1158`
- Category: basic
- Description: segmented mode selector and original/processed image panes.
- Extractable props: `mode`, `originalImage`, `processedImage`, `isLoading`, `message`.
- Hardcoded: `Processed`, `Original`, `Compare` labels.

## PageRail

- Source: `src/uniscan/ui/app.py:1039`
- Category: layout
- Description: multi-select page list with reorder, rotate, delete, replace, and retake actions.
- Extractable props: `pages`, `selectedIds`, `warningCount`.
- Hardcoded: action labels and warning copy.

## CameraPanel

- Source: `src/uniscan/ui/app.py:861`
- Category: layout
- Description: camera health, capture/burst controls, experimental edge detection, device and mode
  selectors, and live preview.
- Extractable props: `cameraState`, `device`, `resolution`, `shots`, `delay`, `liveDetection`.
- Hardcoded: control labels and green Capture Page action.

## TaskStatus

- Source: `src/uniscan/ui/app.py:811`
- Category: basic
- Description: persistent task message and conditional cancel button.
- Extractable props: `message`, `progress`, `cancelable`.
- Hardcoded: `Cancel task` label.

