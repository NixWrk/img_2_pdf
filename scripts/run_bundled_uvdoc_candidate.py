"""Run the pinned bundled UVDoc graph for a model tournament corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from uniscan.tools.geometry_candidate import run_bundled_uvdoc_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = run_bundled_uvdoc_candidate(
        corpus_dir=args.corpus,
        output_dir=args.output,
        on_progress=lambda current, total, case_id: print(
            f"UVDoc {current}/{total}: {case_id}", flush=True
        ),
    )
    print(path)


if __name__ == "__main__":
    main()
