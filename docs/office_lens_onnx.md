# Optional Office Lens ONNX adapter

This repository retains an experimental adapter for compatible Office Lens ONNX models, but it
does **not** distribute model weights extracted from the Microsoft application. No explicit grant
allowing this MIT project to redistribute those weights has been documented. The production
`auto` detector therefore uses the OpenCV hybrid backend.

Use this adapter only with compatible weights that you obtained lawfully and are permitted to use.
Installing the optional adapter does not grant rights to any model files.

The weights were present in repository history before the BYOM policy. Removing them from the
current tree and release artifacts does not erase older Git objects, tags, forks, or downloaded
copies. Before representing a public repository mirror as purged, its owner must coordinate a
history rewrite and force-update, invalidate affected tags/artifacts, and allow host garbage
collection. That destructive repository operation is intentionally not performed by a normal code
commit.

## Runtime Flow

```text
BGR image from UniScan
  -> RGB conversion for Office Lens models
  -> triclass_doc_classifier.ort: Document / Photo / Whiteboard
  -> mnv2_ep42_wb_quant.ort: 256x256 document/whiteboard quad mask
  -> OpenCV mask-to-quad and image-edge refinement
  -> perspective warp
  -> optional Office Lens cleanup variants
  -> BGR image back to the UniScan pipeline
  -> existing image/PDF export
```

## Files

- `src/uniscan/office_lens/adapter.py` - optional BYOM inference adapter.
- `src/uniscan/office_lens/cli.py` - standalone image runner.
- `src/uniscan/office_lens/models/README.md` - expected local filenames and provenance policy.
- `src/uniscan/core/scanner_adapter.py` - integrates the optional backend when explicitly selected.

## Public API

```python
from uniscan.office_lens import OfficeLensOnnx

runner = OfficeLensOnnx()
result = runner.process_file("page.jpg", mode="auto", padding_percent=0.0)
```

Useful result fields:

- `result.classification.label` - `Document`, `Photo`, or `Whiteboard`.
- `result.mask_result.quad` - selected page quadrilateral in source-image coordinates.
- `result.warped` - RGB perspective-corrected page.
- `result.enhancement.image` - RGB cleaned image for the resolved mode.
- `result.enhancement.variants` - extra outputs such as grayscale and black-and-white.

## Integrated Backend

Install the optional runtime and use the backend directly through the scanner adapter:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[office-lens]"
```

```python
from uniscan.core.scanner_adapter import DETECTOR_BACKEND_OFFICE_LENS_ONNX, scan_with_document_detector

scan = scan_with_document_detector(image_bgr, backends=(DETECTOR_BACKEND_OFFICE_LENS_ONNX,))
```

The backend returns `ScanOutput.warped` in BGR, matching the rest of UniScan. It keeps the full
Office Lens `PipelineResult` in `ScanOutput.raw_result` for diagnostics. A missing optional runtime
or either model is an actionable backend error; `auto` does not select this backend.

## Model Path Override

Point the adapter at your external, licensed model folder:

```powershell
set UNISCAN_OFFICE_LENS_MODEL_DIR=D:\models\office_lens
```

The folder must contain:

- `mnv2_ep42_wb_quant.ort`
- `triclass_doc_classifier.ort`

## Dependencies

The adapter needs the optional `onnxruntime` extra plus UniScan's `opencv-python`, `numpy`, and
`pillow` runtime. ONNX Runtime is intentionally absent from the standard portable build because
no ONNX detector weights are distributed with it.
