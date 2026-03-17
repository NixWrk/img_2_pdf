"""Core processing primitives for unified scanner."""

from .geometry import order_quad_points, warp_perspective_from_points
from .pipeline import PipelineOptions, build_pdf_from_images, process_loaded_items, split_spread
from .preprocess import PREPROCESS_PRESETS, PreprocessSettings, apply_enhancements
from .postprocess import POSTPROCESSING_OPTIONS
from .scanner_adapter import ScanAdapterError, scan_with_document_detector

__all__ = [
    "PipelineOptions",
    "PREPROCESS_PRESETS",
    "POSTPROCESSING_OPTIONS",
    "PreprocessSettings",
    "ScanAdapterError",
    "apply_enhancements",
    "build_pdf_from_images",
    "order_quad_points",
    "process_loaded_items",
    "scan_with_document_detector",
    "split_spread",
    "warp_perspective_from_points",
]
