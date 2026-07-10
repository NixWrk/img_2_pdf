"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uniscan.tools import (
    LENS_MODE_CHOICES,
    run_batch_pipeline,
    run_crop_benchmark,
    summarize_benchmark_results,
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

    args = parser.parse_args(argv)
    if args.version:
        from uniscan import __version__

        print(__version__)
        return 0
    if args.command == "convert":
        try:
            result = run_batch_pipeline(
                inputs=args.input,
                output_pdf=args.output,
                images_dir=args.images_dir,
                image_format=args.image_format,
                pdf_dpi=args.pdf_dpi,
                detect_document=not args.no_detect,
                two_page_mode=args.two_page,
                lens_mode=args.mode,
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
    from uniscan.ui import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
