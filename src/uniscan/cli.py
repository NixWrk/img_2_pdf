"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uniscan.diagnostics import diagnostics_json, format_diagnostics, run_diagnostics
from uniscan.tools import (
    DESKEW_METHOD_CHOICES,
    DEWARP_METHOD_CHOICES,
    DEFAULT_QUALITY_BACKENDS,
    BINARIZATION_CHOICES,
    DESPECKLE_CHOICES,
    DETECTOR_POLICY_CHOICES,
    LENS_MODE_CHOICES,
    ORIENTATION_METHOD_CHOICES,
    PAGE_LAYOUT_CHOICES,
    run_batch_pipeline,
    run_crop_benchmark,
    run_geometry_benchmark,
    run_quality_benchmark,
    summarize_benchmark_results,
    summarize_geometry_report,
    summarize_quality_report,
    validate_quality_baseline,
    validate_geometry_baseline,
)


def main(argv: list[str] | None = None) -> int:
    """Run unified scanner application."""
    parser = argparse.ArgumentParser(prog="uniscan")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check runtime dependencies, bundled models, storage, and optionally a camera.",
    )
    doctor_parser.add_argument("--camera", action="store_true", help="Open and read camera 0.")
    doctor_parser.add_argument("--camera-index", type=int, default=0)
    doctor_parser.add_argument("--json", action="store_true", help="Write machine-readable JSON.")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Process image/PDF inputs and write one merged PDF.",
    )
    convert_parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="One or more image/PDF files or folders.",
    )
    convert_parser.add_argument("--output", required=True, type=Path, help="Output PDF path.")
    convert_parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Also write processed page images to this directory.",
    )
    convert_parser.add_argument(
        "--image-format",
        choices=("png", "jpg", "jpeg", "webp", "tif", "tiff"),
        default="png",
        help="Format used with --images-dir.",
    )
    convert_parser.add_argument(
        "--mode",
        choices=LENS_MODE_CHOICES,
        default="document",
        help="Document cleanup profile.",
    )
    convert_parser.add_argument(
        "--backend",
        choices=DETECTOR_POLICY_CHOICES,
        default="auto",
        help="Document detector policy.",
    )
    convert_parser.add_argument(
        "--strict-detect",
        action="store_true",
        help="Fail when a page boundary cannot be detected.",
    )
    convert_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: <output>.report.json).",
    )
    convert_parser.add_argument(
        "--uvdoc-cache",
        type=Path,
        default=None,
        help="Optional cache directory for PaddleOCR UVDoc weights.",
    )
    convert_parser.add_argument("--pdf-dpi", type=int, default=300, help="Input/output PDF DPI.")
    convert_parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable document boundary detection and perspective correction.",
    )
    convert_parser.add_argument(
        "--two-page",
        action="store_true",
        help="Split book spreads into left and right pages.",
    )
    convert_parser.add_argument(
        "--illumination-correction",
        action="store_true",
        help="Opt in to local shadow and glare correction.",
    )
    convert_parser.add_argument(
        "--orientation",
        choices=ORIENTATION_METHOD_CHOICES,
        default="none",
        help="Correct 0/90/180/270 page orientation without OCR.",
    )
    convert_parser.add_argument(
        "--deskew",
        choices=DESKEW_METHOD_CHOICES,
        default="none",
        help="Correct small page rotation after boundary detection.",
    )
    convert_parser.add_argument(
        "--dewarp",
        choices=DEWARP_METHOD_CHOICES,
        default="none",
        help="Correct local page waves independently from boundary detection.",
    )
    convert_parser.add_argument(
        "--auto-dewarp-uvdoc",
        action="store_true",
        help="Allow --dewarp auto to use optional UVDoc (may initialize its model cache).",
    )
    convert_parser.add_argument(
        "--page-layout",
        choices=PAGE_LAYOUT_CHOICES,
        default="none",
        help="Place detected content on a standard output page.",
    )
    convert_parser.add_argument(
        "--page-margin-mm",
        type=float,
        default=10.0,
        help="Uniform margin for standard page layout.",
    )
    convert_parser.add_argument(
        "--align-x",
        choices=("left", "center", "right"),
        default="center",
        help="Horizontal content alignment on a standard page.",
    )
    convert_parser.add_argument(
        "--align-y",
        choices=("top", "center", "bottom"),
        default="center",
        help="Vertical content alignment on a standard page.",
    )
    convert_parser.add_argument(
        "--binarization",
        choices=BINARIZATION_CHOICES,
        default="none",
        help="Document binarization algorithm.",
    )
    convert_parser.add_argument(
        "--binarization-window",
        type=int,
        default=31,
        help="Local window for Sauvola/Wolf (even values are rounded up).",
    )
    convert_parser.add_argument(
        "--binarization-k",
        type=float,
        default=None,
        help="Optional Sauvola/Wolf coefficient from 0 to 1.",
    )
    convert_parser.add_argument(
        "--despeckle",
        choices=DESPECKLE_CHOICES,
        default="none",
        help="Remove only isolated specks at the selected strength.",
    )
    convert_parser.add_argument(
        "--lighting-diagnostics",
        action="store_true",
        help="Measure shadows, possible glare, and clipped pixels in the JSON report.",
    )
    convert_parser.add_argument(
        "--stage-cache-dir",
        type=Path,
        default=None,
        help="Optional persistent cache for post-detection processing stages.",
    )
    convert_parser.add_argument(
        "--stage-cache-max-mb",
        type=int,
        default=512,
        help="Maximum persistent stage-cache size in MiB.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-crop",
        help="Compare crop backends on one input folder and write one PDF per backend.",
    )
    benchmark_parser.add_argument("--input", required=True, type=Path, help="Input folder path.")
    benchmark_parser.add_argument("--output", required=True, type=Path, help="Output folder path.")
    benchmark_parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="Target DPI for generated PDFs.",
    )
    benchmark_parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backend names to run. Defaults to office_lens_onnx.",
    )
    benchmark_parser.add_argument(
        "--scanner-root",
        type=Path,
        default=None,
        help="Optional root directory for vendored camscan backend.",
    )
    benchmark_parser.add_argument(
        "--uvdoc-cache",
        type=Path,
        default=None,
        help="Optional cache directory for PaddleOCR UVDoc weights.",
    )

    quality_parser = subparsers.add_parser(
        "benchmark-quality",
        help="Measure crop success, corner error, latency, and fallback rate.",
    )
    quality_parser.add_argument("--input", required=True, type=Path, help="Corpus folder.")
    quality_parser.add_argument("--output", required=True, type=Path, help="JSON report path.")
    quality_parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_QUALITY_BACKENDS),
        help="Detector backends to measure.",
    )
    quality_parser.add_argument(
        "--corner-tolerance",
        type=float,
        default=0.08,
        help="Maximum mean corner error as an image-diagonal ratio.",
    )
    quality_parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional committed threshold file; regressions return exit code 2.",
    )
    quality_parser.add_argument(
        "--scanner-root",
        type=Path,
        default=None,
        help="Optional root directory for a vendored camscan backend.",
    )
    quality_parser.add_argument(
        "--uvdoc-cache",
        type=Path,
        default=None,
        help="Optional cache directory for PaddleOCR UVDoc weights.",
    )

    geometry_parser = subparsers.add_parser(
        "benchmark-geometry",
        help="Measure orientation, deskew, dewarp quality, and latency.",
    )
    geometry_parser.add_argument("--input", required=True, type=Path, help="Corpus folder.")
    geometry_parser.add_argument("--output", required=True, type=Path, help="JSON report path.")
    geometry_parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional committed threshold file; regressions return exit code 2.",
    )

    args = parser.parse_args(argv)
    if args.version:
        from uniscan import __version__

        print(__version__)
        return 0
    if args.command == "doctor":
        report = run_diagnostics(check_camera=args.camera, camera_index=args.camera_index)
        print(diagnostics_json(report) if args.json else format_diagnostics(report))
        return 0 if report.ok else 1
    if args.command == "convert":
        try:
            result = run_batch_pipeline(
                inputs=args.input,
                output_pdf=args.output,
                images_dir=args.images_dir,
                image_format=args.image_format,
                report_path=args.report,
                pdf_dpi=args.pdf_dpi,
                detect_document=not args.no_detect,
                detector_policy=args.backend,
                strict_detect=args.strict_detect,
                two_page_mode=args.two_page,
                lens_mode=args.mode,
                illumination_correction=args.illumination_correction,
                orientation_method=args.orientation,
                deskew_method=args.deskew,
                dewarp_method=args.dewarp,
                auto_dewarp_uvdoc=args.auto_dewarp_uvdoc,
                page_layout=args.page_layout,
                page_margin_mm=args.page_margin_mm,
                horizontal_alignment=args.align_x,
                vertical_alignment=args.align_y,
                binarization_method=args.binarization,
                binarization_window=args.binarization_window,
                binarization_k=args.binarization_k,
                despeckle_strength=args.despeckle,
                lighting_diagnostics=args.lighting_diagnostics,
                stage_cache_dir=args.stage_cache_dir,
                stage_cache_max_mb=args.stage_cache_max_mb,
                uvdoc_cache_home=args.uvdoc_cache,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"uniscan: error: {exc}", file=sys.stderr)
            return 2
        print(
            f"Wrote {result.total_pages} page(s) to {result.output_pdf} "
            f"(detected {result.detected_pages}/{result.total_pages})."
        )
        if result.image_outputs:
            print(f"Wrote {len(result.image_outputs)} image(s) to {args.images_dir}.")
        print(f"Report: {result.report_path}")
        return 0
    if args.command == "benchmark-crop":
        results = run_crop_benchmark(
            input_dir=args.input,
            output_dir=args.output,
            backends=tuple(args.backends) if args.backends else None,
            pdf_dpi=args.pdf_dpi,
            scanner_root=args.scanner_root,
            uvdoc_cache_home=args.uvdoc_cache,
        )
        print(summarize_benchmark_results(results))
        return 0 if any(result.output_pdf is not None for result in results) else 1
    if args.command == "benchmark-quality":
        try:
            report = run_quality_benchmark(
                corpus_dir=args.input,
                output_path=args.output,
                backends=tuple(args.backends),
                corner_tolerance_ratio=args.corner_tolerance,
                scanner_root=args.scanner_root,
                uvdoc_cache_home=args.uvdoc_cache,
            )
            print(summarize_quality_report(report))
            if args.baseline is not None:
                failures = validate_quality_baseline(report, args.baseline)
                if failures:
                    print("Quality baseline regressions:", file=sys.stderr)
                    for failure in failures:
                        print(f"- {failure}", file=sys.stderr)
                    return 2
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"uniscan: error: {exc}", file=sys.stderr)
            return 2
        return 0 if any(result.error is None for result in report.backends) else 1
    if args.command == "benchmark-geometry":
        try:
            report = run_geometry_benchmark(corpus_dir=args.input, output_path=args.output)
            print(summarize_geometry_report(report))
            if args.baseline is not None:
                failures = validate_geometry_baseline(report, args.baseline)
                if failures:
                    print("Geometry baseline regressions:", file=sys.stderr)
                    for failure in failures:
                        print(f"- {failure}", file=sys.stderr)
                    return 2
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"uniscan: error: {exc}", file=sys.stderr)
            return 2
        return 0
    from uniscan.ui import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
