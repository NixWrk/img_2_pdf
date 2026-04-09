"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse
from pathlib import Path

from uniscan.tools import run_crop_benchmark, summarize_benchmark_results
from uniscan.ui import run_app


def main(argv: list[str] | None = None) -> int:
    """Run unified scanner application."""
    parser = argparse.ArgumentParser(prog="uniscan")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

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
        help="Backend names to run. Defaults to paddleocr_uvdoc.",
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
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
