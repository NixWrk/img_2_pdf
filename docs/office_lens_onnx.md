# Office Lens ONNX Adapter

This is the PC-portable part extracted from the Office Lens APK analysis. The Android activity layer, native Android `.so` libraries, OCR, and live camera pieces are not required for this adapter.

## Runtime Flow

```text
BGR image from img_2_pdf
  -> RGB conversion for Office Lens models
  -> triclass_doc_classifier.ort: Document / Photo / Whiteboard
  -> mnv2_ep42_wb_quant.ort: 256x256 document/whiteboard quad mask
  -> OpenCV mask-to-quad and image-edge refinement
  -> perspective warp
  -> optional Office Lens cleanup variants
  -> BGR image back to img_2_pdf pipeline
  -> existing image/PDF export
```

## Files

- `src/uniscan/office_lens/adapter.py` - reusable API copied from the PC adapter and adjusted for package-relative models.
- `src/uniscan/office_lens/cli.py` - standalone image runner.
- `src/uniscan/office_lens/models/mnv2_ep42_wb_quant.ort` - quad mask model.
- `src/uniscan/office_lens/models/triclass_doc_classifier.ort` - mode classifier.
- `src/uniscan/office_lens/model_metadata/*.json` - inspection and CPU benchmark metadata from the extraction pass.
- `src/uniscan/core/scanner_adapter.py` - integrates `office_lens_onnx` as the default document-detection backend, with PaddleOCR UVDoc and OpenCV hybrid fallback.

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

Use the backend directly through the existing scanner adapter:

```python
from uniscan.core.scanner_adapter import DETECTOR_BACKEND_OFFICE_LENS_ONNX, scan_with_document_detector

scan = scan_with_document_detector(image_bgr, backends=(DETECTOR_BACKEND_OFFICE_LENS_ONNX,))
```

The backend returns `ScanOutput.warped` in BGR, matching the rest of `img_2_pdf`. It keeps the full Office Lens `PipelineResult` in `ScanOutput.raw_result` for diagnostics.

## Model Path Override

By default models are loaded from `src/uniscan/office_lens/models`. For experiments, point the adapter at another model folder:

```powershell
set UNISCAN_OFFICE_LENS_MODEL_DIR=D:\models\office_lens
```

The folder must contain:

- `mnv2_ep42_wb_quant.ort`
- `triclass_doc_classifier.ort`

## Dependencies

The adapter directly needs `onnxruntime`, `opencv-python`, and `numpy`; it can use `pillow` for
EXIF-aware file reads. All four are declared as package runtime dependencies in `pyproject.toml`.
