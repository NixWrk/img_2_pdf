"""Utility tools for offline benchmarking and maintenance flows."""

from .batch_pipeline import (
    DESKEW_METHOD_CHOICES,
    DEWARP_METHOD_CHOICES,
    DETECTOR_POLICY_CHOICES,
    LENS_MODE_CHOICES,
    ORIENTATION_METHOD_CHOICES,
    PAGE_LAYOUT_CHOICES,
    BINARIZATION_CHOICES,
    DESPECKLE_CHOICES,
    BatchPipelineResult,
    PageRunReport,
    resolve_input_paths,
    run_batch_pipeline,
)
from .crop_benchmark import (
    BackendBenchmarkResult,
    run_crop_benchmark,
    summarize_benchmark_results,
)
from .quality_benchmark import (
    DEFAULT_QUALITY_BACKENDS,
    QualityBackendResult,
    QualityBenchmarkReport,
    QualityPageResult,
    run_quality_benchmark,
    summarize_quality_report,
    validate_quality_baseline,
)
from .geometry_benchmark import (
    load_geometry_manifest,
    run_geometry_benchmark,
    summarize_geometry_report,
    validate_geometry_baseline,
)

__all__ = [
    "DESKEW_METHOD_CHOICES",
    "DEWARP_METHOD_CHOICES",
    "LENS_MODE_CHOICES",
    "ORIENTATION_METHOD_CHOICES",
    "PAGE_LAYOUT_CHOICES",
    "BINARIZATION_CHOICES",
    "DESPECKLE_CHOICES",
    "DETECTOR_POLICY_CHOICES",
    "BatchPipelineResult",
    "PageRunReport",
    "BackendBenchmarkResult",
    "resolve_input_paths",
    "run_batch_pipeline",
    "run_crop_benchmark",
    "summarize_benchmark_results",
    "DEFAULT_QUALITY_BACKENDS",
    "QualityPageResult",
    "QualityBackendResult",
    "QualityBenchmarkReport",
    "run_quality_benchmark",
    "summarize_quality_report",
    "validate_quality_baseline",
    "load_geometry_manifest",
    "run_geometry_benchmark",
    "summarize_geometry_report",
    "validate_geometry_baseline",
]
