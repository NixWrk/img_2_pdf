# UniScan page dependency trees

## Workspace

Entry: `src/uniscan/ui/app.py` → `UnifiedScanApp._build_pages_tab`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/ui/page_parse.py`
  - `src/uniscan/ui/import_sources.py`
  - `src/uniscan/ui/overlays.py`
  - `src/uniscan/ui/camera_health.py`
  - `src/uniscan/ui/live_detect.py`
  - `src/uniscan/session/capture_session.py`
    - `src/uniscan/session/autosave.py`
    - `src/uniscan/storage/page_store.py`
  - `src/uniscan/storage/stage_cache.py`
  - `src/uniscan/core/processing.py`
    - `src/uniscan/core/orientation.py`
    - `src/uniscan/core/dewarp.py`
      - `src/uniscan/core/uvdoc.py`
      - `src/uniscan/core/docscanner.py`
    - `src/uniscan/core/preprocess.py`
    - `src/uniscan/core/lighting.py`
    - `src/uniscan/core/cleanup.py`
    - `src/uniscan/core/layout.py`
  - `src/uniscan/core/pipeline.py`
  - `src/uniscan/core/scanner_adapter.py`
  - `src/uniscan/core/spread.py`
  - `src/uniscan/export/exporters.py`

## Camera

Entry: `src/uniscan/ui/app.py` → `UnifiedScanApp._build_capture_tab`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/io/camera_service.py`
  - `src/uniscan/ui/live_detect.py`
  - `src/uniscan/ui/camera_health.py`
  - `src/uniscan/ui/overlays.py`
  - `src/uniscan/core/pipeline.py`

## Perspective editor

Entry: `src/uniscan/ui/app.py` → `_open_corner_editor_dialog`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/core/geometry.py`
  - `src/uniscan/core/scanner_adapter.py`
  - `src/uniscan/ui/overlays.py`

## Spread split editor

Entry: `src/uniscan/ui/app.py` → `open_split_editor`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/core/spread.py`

## Wave editor

Entry: `src/uniscan/ui/app.py` → `open_dewarp_points_editor`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/core/dewarp.py`
  - `src/uniscan/core/processing.py`

## Import/export dialogs

Entry methods: `open_import_options_dialog`, `open_export_dialog`

Dependencies:

- `src/uniscan/ui/app.py`
  - `src/uniscan/io/loaders.py`
  - `src/uniscan/export/exporters.py`

