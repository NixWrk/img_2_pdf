"""Download pinned release model assets after SHA-256 verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uniscan.model_assets import (  # noqa: E402
    MODEL_DIR,
    download_model_asset,
    model_asset,
    verify_model_asset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--target", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--url",
        help="Release-asset URL override for one SHA-pinned manifest entry.",
    )
    parser.add_argument("--check", action="store_true", help="Verify without downloading.")
    args = parser.parse_args()
    names = list(dict.fromkeys(args.asset or ["docshadow_sd7k"]))
    if args.url and len(names) != 1:
        parser.error("--url requires exactly one --asset")
    for name in names:
        path = (
            verify_model_asset(name, args.target / model_asset(name).filename)
            if args.check
            else download_model_asset(name, args.target, url=args.url)
        )
        print(f"verified {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
