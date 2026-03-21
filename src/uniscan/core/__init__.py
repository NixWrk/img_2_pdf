"""Core processing primitives for unified scanner."""

from .geometry import order_quad_points, warp_perspective_from_points
from .pipeline import PipelineOptions, build_pdf_from_images, process_loaded_items, split_spread
from .preprocess import (
    LENS_MODE_CUSTOM,
    LENS_MODE_PROFILES,
    LENS_MODE_VALUES,
    PREPROCESS_PRESETS,
    PreprocessSettings,
    apply_enhancements,
    deskew_document,
    infer_lens_mode,
    resolve_lens_mode_profile,
)
from .postprocess import POSTPROCESSING_OPTIONS
from .scanner_adapter import (
    DETECTOR_BACKEND_CAMSCAN,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    DEFAULT_ACTIVE_DOCUMENT_BACKENDS,
    ScanAdapterError,
    probe_detector_backend,
    scan_with_document_detector,
)

__all__ = [
    "LENS_MODE_CUSTOM",
    "LENS_MODE_PROFILES",
    "LENS_MODE_VALUES",
    "PipelineOptions",
    "PREPROCESS_PRESETS",
    "POSTPROCESSING_OPTIONS",
    "PreprocessSettings",
    "DETECTOR_BACKEND_CAMSCAN",
    "DETECTOR_BACKEND_CV_HYBRID",
    "DETECTOR_BACKEND_OPENCV",
    "DETECTOR_BACKEND_OPENCV_HOUGH",
    "DETECTOR_BACKEND_OPENCV_MINRECT",
    "DETECTOR_BACKEND_PADDLEOCR_UVDOC",
    "DETECTOR_BACKEND_UVDOC",
    "DEFAULT_ACTIVE_DOCUMENT_BACKENDS",
    "ScanAdapterError",
    "apply_enhancements",
    "build_pdf_from_images",
    "deskew_document",
    "infer_lens_mode",
    "order_quad_points",
    "process_loaded_items",
    "probe_detector_backend",
    "resolve_lens_mode_profile",
    "scan_with_document_detector",
    "split_spread",
    "warp_perspective_from_points",
]
