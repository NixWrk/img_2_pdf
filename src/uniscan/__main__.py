"""Module entrypoint for ``python -m uniscan``."""

from __future__ import annotations

from uniscan.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
