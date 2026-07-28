# Optional Office Lens model directory

UniScan does not distribute the Office Lens model weights. Their original application terms do
not provide this project with an explicit redistribution grant.

The adapter remains available for users who independently have compatible, lawfully obtained
weights and install the `office-lens` extra. Set `UNISCAN_OFFICE_LENS_MODEL_DIR` to a directory
containing both files:

- `mnv2_ep42_wb_quant.ort`
- `triclass_doc_classifier.ort`

The normal `auto` detector does not depend on these files and uses the OpenCV hybrid backend.
Office Lens may enter the same quality-first tournament as every other candidate. Before changing
its production delivery, record the exact weights, provenance, licence metadata and permitted use.
