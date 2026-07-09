"""CLI for the Office Lens ONNX adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import OfficeLensOnnx, save_pipeline_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uniscan-office-lens",
        description="Run the Android-free Office Lens ONNX/OpenCV document pipeline on an image.",
    )
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument("--out", default="office_lens_out", help="Output directory.")
    parser.add_argument(
        "--mode",
        choices=("auto", "document", "whiteboard", "photo"),
        default="auto",
        help="Cleanup mode. auto uses the extracted Office Lens classifier.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Optional quad padding before perspective warp. Example: 0.02 adds 2 percent.",
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    output_dir = Path(args.out)

    runner = OfficeLensOnnx()
    result = runner.process_file(image_path, mode=args.mode, padding_percent=args.padding)
    report = save_pipeline_outputs(image_path, result, output_dir)

    report_path = output_dir / f"{image_path.stem}_onnx_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
