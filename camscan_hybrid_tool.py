#!/usr/bin/env python3
"""Compatibility launcher for the package-based UniScan application."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uniscan.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
