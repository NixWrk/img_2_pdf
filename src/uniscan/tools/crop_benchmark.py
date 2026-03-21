"""Batch benchmark sketch for comparing document-cropping backends."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_CAMSCAN,
    DETECTOR_BACKEND_CV_HYBRID,
    DETECTOR_BACKEND_OPENCV,
    DETECTOR_BACKEND_OPENCV_HOUGH,
    DETECTOR_BACKEND_OPENCV_MINRECT,
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
    DETECTOR_BACKEND_UVDOC,
    ScanAdapterError,
    probe_detector_backend,
    scan_with_document_detector,
)
from uniscan.export.exporters import export_image_paths_as_pdf
from uniscan.io.loaders import load_input_items, list_supported_in_folder, imwrite_unicode

DEFAULT_BENCHMARK_BACKENDS = (
    DETECTOR_BACKEND_PADDLEOCR_UVDOC,
)


@dataclass(slots=True)
class BackendBenchmarkResult:
    """Summary of one backend run in the benchmark sketch."""

    backend: str
    output_pdf: Path | None
    total_pages: int
    detected_pages: int
    error: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_camscan_root() -> Path | None:
    """Return the local vendored camscan root if it exists."""
    candidate = _repo_root() / "camscan_suhren"
    if (candidate / "camscan" / "scanner.py").exists():
        return candidate
    return None


def _iter_loaded_input_paths(input_paths: Sequence[Path], *, pdf_dpi: int):
    if not input_paths:
        raise ValueError("No supported files found for benchmark input.")

    for source_path in input_paths:
        for loaded in load_input_items([source_path], pdf_dpi=pdf_dpi):
            yield loaded


def _resolve_benchmark_input_paths(
    *,
    input_dir: Path,
    output_dir: Path,
    backends: Sequence[str],
) -> tuple[Path, ...]:
    input_paths = list_supported_in_folder(input_dir)
    if input_dir.resolve() != output_dir.resolve():
        return tuple(input_paths)

    reserved_output_names = {f"{input_dir.name}_{backend}.pdf" for backend in backends}
    filtered = [path for path in input_paths if path.name not in reserved_output_names]
    return tuple(filtered)


def _run_single_backend(
    *,
    input_dir: Path,
    input_paths: Sequence[Path],
    output_dir: Path,
    backend: str,
    pdf_dpi: int,
    scanner_root: Path | None,
    uvdoc_cache_home: Path | None,
) -> BackendBenchmarkResult:
    probe_detector_backend(
        backend,
        scanner_root=scanner_root,
        uvdoc_cache_home=uvdoc_cache_home,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / f"{input_dir.name}_{backend}.pdf"
    total_pages = 0
    detected_pages = 0

    with tempfile.TemporaryDirectory(prefix=f"uniscan_benchmark_{backend}_") as tmp:
        tmp_dir = Path(tmp)
        image_paths: list[Path] = []

        for total_pages, (_name, image) in enumerate(
            _iter_loaded_input_paths(input_paths, pdf_dpi=pdf_dpi),
            start=1,
        ):
            scan_output = scan_with_document_detector(
                image,
                enabled=True,
                scanner_root=scanner_root,
                backends=(backend,),
                uvdoc_cache_home=uvdoc_cache_home,
            )
            working = scan_output.warped if scan_output.warped is not None else image
            if scan_output.detected:
                detected_pages += 1

            image_path = tmp_dir / f"{total_pages:05d}.png"
            if not imwrite_unicode(image_path, working):
                raise RuntimeError(f"Failed to write benchmark page image: {image_path}")
            image_paths.append(image_path)

        if not image_paths:
            raise ValueError(f"No pages were processed for backend: {backend}")

        export_image_paths_as_pdf(image_paths, out_pdf=output_pdf, dpi=pdf_dpi)

    return BackendBenchmarkResult(
        backend=backend,
        output_pdf=output_pdf,
        total_pages=total_pages,
        detected_pages=detected_pages,
    )


def run_crop_benchmark(
    *,
    input_dir: Path,
    output_dir: Path,
    backends: Sequence[str] | None = DEFAULT_BENCHMARK_BACKENDS,
    pdf_dpi: int = 300,
    scanner_root: Path | None = None,
    uvdoc_cache_home: Path | None = None,
) -> list[BackendBenchmarkResult]:
    """Run the crop benchmark sketch and return one result per backend."""
    resolved_input = Path(input_dir)
    resolved_output = Path(output_dir)
    resolved_backends = tuple(backends) if backends is not None else DEFAULT_BENCHMARK_BACKENDS
    resolved_input_paths = _resolve_benchmark_input_paths(
        input_dir=resolved_input,
        output_dir=resolved_output,
        backends=resolved_backends,
    )
    resolved_scanner_root = scanner_root
    if resolved_scanner_root is None and DETECTOR_BACKEND_CAMSCAN in resolved_backends:
        resolved_scanner_root = default_camscan_root()

    results: list[BackendBenchmarkResult] = []
    for backend in resolved_backends:
        try:
            result = _run_single_backend(
                input_dir=resolved_input,
                input_paths=resolved_input_paths,
                output_dir=resolved_output,
                backend=backend,
                pdf_dpi=int(pdf_dpi),
                scanner_root=resolved_scanner_root,
                uvdoc_cache_home=uvdoc_cache_home,
            )
        except (ScanAdapterError, RuntimeError, ValueError) as exc:
            result = BackendBenchmarkResult(
                backend=backend,
                output_pdf=None,
                total_pages=0,
                detected_pages=0,
                error=str(exc),
            )
        results.append(result)
    return results


def summarize_benchmark_results(results: Sequence[BackendBenchmarkResult]) -> str:
    """Format a concise summary for CLI output."""
    lines: list[str] = []
    for result in results:
        if result.error:
            lines.append(f"{result.backend}: failed - {result.error}")
            continue
        lines.append(
            f"{result.backend}: {result.output_pdf} "
            f"(detected {result.detected_pages}/{result.total_pages})"
        )
    return "\n".join(lines)
