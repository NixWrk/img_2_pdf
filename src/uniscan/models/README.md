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
- Pinned identity: graph SHA-256
  `044b406398e8accf2b8c896043f22d318221e443bb095c55d184e97f3eb003f7`; external data SHA-256
  `3a58a04944e59578200d7b0ed02ccb3ade934e9d88efa21454fbb8b53fa40f02`.

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

## DocShadow shadow remover

- File: `docshadow_sd7k.onnx` (~120 MB), the SD7K-trained variant.
- Source: <https://github.com/fabio-sim/DocShadow-ONNX-TensorRT> (MIT), an ONNX export of
  <https://github.com/CXH-Research/DocShadow-SD7K> (MIT), from *High-Resolution Document Shadow
  Removal via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net*,
  ICCV 2023. `DOCSHADOW-LICENSE` is the export's licence text.
- Delivery: immutable upstream release asset `v1.0.0`; SHA-256
  `8c09b9320a0fb3c53806cdf9cb8410b3706c77caed81f4ec1cb0bf2cbd14b049`, size
  `120041674` bytes. It is downloaded atomically by `scripts/download_model_assets.py` and is
  accepted only after both size and SHA-256 match; it is not stored as a Git blob.
- Interface: input `image` `(batch, 3, height, width)` float32 RGB in `[0, 1]`; output `result`
  in the same layout. Both spatial dimensions are dynamic.

`uniscan.core.docshadow` runs it at 256x256 and keeps only the ratio between its result and its
input as a smoothed, single-channel illumination map, which is then applied to the full-resolution
page. Running the network at page resolution would be several times slower for no measurable gain,
and using its output directly would resample every glyph.

Set `UNISCAN_DOCSHADOW_MODEL` to use a different DocShadow ONNX file.

## DocScanner-L grid rectifier

- File: `DocScanner-L-grid-opset17.onnx` (34,100,351 bytes), kept as an external release asset.
- Source: official DocScanner commit `54f6063a61a52e4ce4012832e943d1871a9c3c66` and its
  `DocScanner-L.pth` / `seg.pth` checkpoints. Upstream permits non-commercial use with attribution
  and share-alike conditions; commercial use requires the author's permission.
- Pinned identity: SHA-256
  `9fdebcb4067afb09d66b6637f3fd1036ba7952bbf0656778d44c2bd1c2c067f4`.
- Interface: input `image` `(1, 3, 288, 288)` float32 RGB in `[0, 1]`; output `grid`
  `(1, 2, 288, 288)` float32 in normalized backward-sampling coordinates.

The opset-17 graph includes the official U²-Net page mask and all 12 DocScanner-L recurrent
iterations. Against PyTorch 2.1.1/cu118 on a real DocUNet input, ONNX Runtime differed by at most
`2.98e-7` in the grid. UniScan then applies that grid to the original page pixels. Set
`UNISCAN_DOCSCANNER_MODEL` to the verified external graph; environment overrides are still required
to match the pinned size and SHA-256.

Until the graph is attached to a UniScan release, a staged release URL can be verified and installed
without changing the manifest:

```text
python scripts/download_model_assets.py --asset docscanner_l_grid --url RELEASE_ASSET_URL
```

Machine-readable provenance and exact identities are in `manifest.json`. Bundled default assets
are verified again before the first ONNX Runtime session is created. Environment overrides are
allowed for development and their content SHA-256 becomes part of processing-cache keys.

## Adding other weights

Model quality is ranked independently of licence family. Every candidate still needs documented
provenance, a content identity and recorded terms. Use `uniscan benchmark-models` to compare outputs
from incompatible frameworks before adding a production dependency. Winning weights may be bundled,
downloaded with an exact SHA-256, or kept behind an external/BYOM adapter according to their actual
distribution and use terms.
