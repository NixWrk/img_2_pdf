# Bundled model weights

## UVDoc grid rectifier

- Files: `UVDoc_grid.onnx` (graph) and `UVDoc_grid.onnx.data` (external weights, ~30 MB).
  Both are required; ONNX Runtime resolves the data file next to the graph.
- Source: <https://huggingface.co/fredcallagan/uvdoc-grid-onnx>, an ONNX export of the reference
  UVDoc model with a wrapper that keeps only the 2D grid output.
- Upstream model and code: <https://github.com/tanguymagne/UVDoc> (MIT), from *UVDoc: Neural
  Grid-based Document Unwarping*, SIGGRAPH Asia 2023.
- Licenses: upstream UVDoc is MIT; the ONNX export is published under Apache-2.0. `LICENSE` in
  this directory is the export's license text. Both permit redistribution, so unlike the Office
  Lens adapter these weights ship with UniScan.

### Interface

| | |
| --- | --- |
| Input `image` | `(1, 3, 720, 496)` float32, RGB, `[0, 1]` |
| Output `grid_2d` | `(1, 2, 45, 31)` float32, `[-1, 1]` |

The output is a backward sampling grid: channel 0 is x, channel 1 is y, laid out as 45 rows by
31 columns (verified against the exported graph — x rises along the column axis, y along the row
axis). `uniscan.core.uvdoc` upsamples it to the page size, converts it to pixel coordinates and
feeds `cv2.remap`, so the original full-resolution image is sampled directly.

Set `UNISCAN_UVDOC_MODEL` to use a different UVDoc ONNX file.

## Adding other weights

Do not add model binaries to this repository without documented provenance and compatible terms.
Weights that cannot be redistributed belong behind an environment variable instead, the way
`uniscan.office_lens` handles the optional Office Lens models.
