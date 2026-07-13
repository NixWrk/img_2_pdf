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

## Runtime flows

The adapter has two deliberately different flows.

The standalone `OfficeLensOnnx.process_image()` / `process_file()` flow resolves a mode, detects
the page quadrilateral, warps it, and applies cleanup:

```text
RGB image (process_image) or input file converted to RGB (process_file)
  -> mode=auto only: triclass_doc_classifier.ort -> Document / Photo / Whiteboard
     explicit document/photo/whiteboard mode: use that mode without classifier inference
  -> required mnv2_ep42_wb_quant.ort: 256x256 document/whiteboard quad mask
  -> OpenCV mask-to-quad and image-edge refinement
  -> perspective warp
  -> optional Office Lens cleanup variants
  -> RGB PipelineResult; the standalone CLI can then save output files/report
```

The integrated `office_lens_onnx` scanner backend uses only boundary detection:

```text
BGR image from UniScan
  -> RGB conversion
  -> required mnv2_ep42_wb_quant.ort quad mask
  -> OpenCV mask-to-quad and image-edge refinement
  -> perspective warp
  -> BGR ScanOutput for the normal UniScan processing/export pipeline
```

The integrated flow intentionally skips the classifier and Office Lens cleanup because only its
quadrilateral is consumed. The normal `convert --backend auto` policy remains the offline
`cv_hybrid` detector and does not select this adapter.

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

Construction always requires the licensed quad model and ONNX Runtime. The classifier is loaded
lazily only when `mode="auto"` is processed. Pass `mode="document"`, `"photo"`, or
`"whiteboard"` to use a fixed cleanup mode without the classifier model.

Useful result fields:

- `result.classification.label` - the inferred label in `auto` mode, or the explicitly requested
  mode title otherwise. In an explicit mode, this field does not prove classifier inference and
  `result.classification.scores` is empty.
- `result.mask_result.quad` - selected page quadrilateral in source-image coordinates.
- `result.warped` - RGB perspective-corrected page, or `None` if no quad was found.
- `result.enhancement.image` - RGB cleaned image for the resolved mode when a quad was found.
- `result.enhancement.variants` - extra outputs such as grayscale and black-and-white when cleanup
  ran.

## Integrated Backend

Install the optional runtime and use the backend directly through the scanner adapter:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[office-lens]"
```

```python
from uniscan.core.scanner_adapter import DETECTOR_BACKEND_OFFICE_LENS_ONNX, scan_with_document_detector

scan = scan_with_document_detector(image_bgr, backends=(DETECTOR_BACKEND_OFFICE_LENS_ONNX,))
```

The backend returns `ScanOutput.warped` in BGR, matching the rest of UniScan. It keeps the
`QuadMaskResult` in `ScanOutput.raw_result` for diagnostics. Missing ONNX Runtime or the quad model
is an actionable backend error. The classifier is not loaded by this integrated flow, and `auto`
does not select this backend.

## Model Path Override

Point the adapter at your external, licensed model folder:

```powershell
$env:UNISCAN_OFFICE_LENS_MODEL_DIR = "D:\models\office_lens"
```

The folder must contain the quad model used by every flow:

- `mnv2_ep42_wb_quant.ort`

It must also contain `triclass_doc_classifier.ort` only for standalone `mode="auto"`. Explicit
`document`, `photo`, and `whiteboard` modes and the integrated boundary detector do not need it.

## Dependencies

The adapter needs the optional `onnxruntime` extra plus UniScan's `opencv-python`, `numpy`, and
`pillow` runtime. ONNX Runtime is intentionally absent from the standard portable build because
no ONNX detector weights are distributed with it. `uniscan doctor` reports a missing quad model as
disabled and an installed but unloadable quad model as a blocking failure. A missing classifier is
reported as disabled; an installed but unloadable classifier is a non-blocking warning because
explicit modes remain available.
