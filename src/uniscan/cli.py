"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse

from uniscan.ui import run_app


def main() -> int:
    """Run unified scanner application."""
    parser = argparse.ArgumentParser(prog="uniscan")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    args = parser.parse_args()
    if args.version:
        from uniscan import __version__

        print(__version__)
        return 0
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
