"""Measure what the current geometry stage order costs on one real document page.

Three independent probes back `docs/geometry_stage_order_audit_2026-08-15.md`:

``resample``
    How much high-frequency detail one, two and three interpolation passes remove
    from the same image. Isolates the cost of the chain length itself.

``order``
    The same page half through ``deskew -> dewarp`` (the shipped order) and
    ``dewarp -> deskew``, plus rectification before the boundary crop.

``gate``
    What validated automatic dewarp actually selects and rejects per page half.

Rendering must match ``uniscan.io.loaders._render_pdf_page`` exactly: PDFium
returns BGRA and the loader asks for ``rev_byteorder``. Rendering without it
swaps red and blue, which changes the grayscale projection every geometry and
OCR metric here is built on, and therefore changes detector decisions.

Tesseract is optional; without ``--tesseract`` the OCR columns are omitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from uniscan.core.dewarp import dewarp_document, measure_dewarp_quality
from uniscan.core.pipeline import PipelineOptions, process_loaded_items
from uniscan.core.preprocess import deskew_document
from uniscan.core import uvdoc

DIFFICULT_SPREADS = (2, 4, 6, 10)  # zero-based; source PDF pages 3, 5, 7 and 11


def render_page(pdf_path: Path, page_index: int, dpi: int) -> np.ndarray:
    """Render one page the way the production loader does."""
    document = pdfium.PdfDocument(str(pdf_path))
    bitmap = document[page_index].render(
        scale=dpi / 72.0,
        rev_byteorder=True,
        fill_color=(255, 255, 255, 255),
    )
    try:
        array = np.array(bitmap.to_numpy(), copy=True)
    finally:
        bitmap.close()
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: a proxy for retained high-frequency detail."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def split_halves(image: np.ndarray, name: str) -> list[np.ndarray]:
    pages = process_loaded_items(
        [(name, image)],
        options=PipelineOptions(detect_document=True, two_page_mode=True),
    )
    return [page.warped for page in pages]


def ocr_probe(image: np.ndarray, *, tesseract: Path, work_dir: Path, tag: str) -> dict[str, float]:
    """Words, alphanumeric characters and character-weighted confidence."""
    path = work_dir / f"{tag}.png"
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode probe image for {tag}.")
    path.write_bytes(buffer.tobytes())
    completed = subprocess.run(
        [str(tesseract), str(path), "stdout", "-l", "rus+eng", "--psm", "1", "tsv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    words = 0
    characters = 0
    weighted = 0.0
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        text = fields[11].strip()
        if confidence < 0 or not text:
            continue
        alphanumeric = sum(1 for character in text if character.isalnum())
        if alphanumeric == 0:
            continue
        words += 1
        characters += alphanumeric
        weighted += confidence * alphanumeric
    return {
        "words": words,
        "alnumChars": characters,
        "meanConfidence": round(weighted / characters, 2) if characters else 0.0,
    }


def describe(
    image: np.ndarray,
    tag: str,
    *,
    tesseract: Path | None,
    work_dir: Path,
) -> dict[str, object]:
    quality = measure_dewarp_quality(image)
    row: dict[str, object] = {
        "tag": tag,
        "size": f"{image.shape[1]}x{image.shape[0]}",
        "curvature": quality.curvature_rms_px,
        "lines": quality.line_count,
        "perspective": quality.perspective_score,
        "edgeInk": quality.edge_ink_ratio,
        "sharpness": sharpness(image),
    }
    if tesseract is not None:
        row.update(ocr_probe(image, tesseract=tesseract, work_dir=work_dir, tag=tag))
    return row


def probe_resample(source: np.ndarray) -> list[dict[str, object]]:
    """Cost of repeated interpolation, with the geometry held at identity."""

    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    height, width = source.shape[:2]
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    once = rotate(source, 1.7)
    twice = rotate(once, -1.7)
    return [
        {"pass": "source", "sharpness": sharpness(source)},
        {"pass": "1 rotation", "sharpness": sharpness(once)},
        {"pass": "2 rotations (net zero)", "sharpness": sharpness(twice)},
        {"pass": "3 rotations", "sharpness": sharpness(rotate(twice, 1.7))},
        {
            "pass": "identity remap, exact grid",
            "sharpness": sharpness(cv2.remap(source, map_x, map_y, cv2.INTER_CUBIC)),
        },
        {
            "pass": "remap, half-pixel offset",
            "sharpness": sharpness(
                cv2.remap(
                    source,
                    map_x + 0.5,
                    map_y + 0.5,
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            ),
        },
    ]


def probe_order(
    source: np.ndarray,
    *,
    tesseract: Path | None,
    work_dir: Path,
) -> list[dict[str, object]]:
    """Compare stage orders on the same detected page halves."""
    rows: list[dict[str, object]] = []
    halves = split_halves(source, "probe")
    rectified_halves = split_halves(uvdoc.dewarp(source), "probe-uvdoc-first")

    for index, half in enumerate(halves):
        side = "L" if index == 0 else "R"
        rows.append(describe(half, f"{side}_v0_no_dewarp", tesseract=tesseract, work_dir=work_dir))

        deskewed, angle = deskew_document(half, method="hybrid")
        rows.append(
            {
                **describe(
                    uvdoc.dewarp(deskewed),
                    f"{side}_v1_deskew_then_uvdoc",
                    tesseract=tesseract,
                    work_dir=work_dir,
                ),
                "deskewAngle": round(angle, 3),
            }
        )

        rectified, angle = deskew_document(uvdoc.dewarp(half), method="hybrid")
        rows.append(
            {
                **describe(
                    rectified,
                    f"{side}_v2_uvdoc_then_deskew",
                    tesseract=tesseract,
                    work_dir=work_dir,
                ),
                "deskewAngle": round(angle, 3),
            }
        )

        if index < len(rectified_halves):
            before_crop, angle = deskew_document(rectified_halves[index], method="hybrid")
            rows.append(
                {
                    **describe(
                        before_crop,
                        f"{side}_v3_uvdoc_before_crop",
                        tesseract=tesseract,
                        work_dir=work_dir,
                    ),
                    "deskewAngle": round(angle, 3),
                    "note": (
                        "comparable only when rectify-first produced the same number of halves; "
                        f"it produced {len(rectified_halves)}"
                    ),
                }
            )
    return rows


def probe_gate(pdf_path: Path, dpi: int) -> list[dict[str, object]]:
    """What validated automatic dewarp selects and rejects, in the shipped order."""
    rows: list[dict[str, object]] = []
    for page_index in DIFFICULT_SPREADS:
        source = render_page(pdf_path, page_index, dpi)
        for page in process_loaded_items(
            [(f"p{page_index + 1}", source)],
            options=PipelineOptions(detect_document=True, two_page_mode=True),
        ):
            deskewed, angle = deskew_document(page.warped, method="hybrid")
            _, diagnostics = dewarp_document(deskewed, method="auto")
            rows.append(
                {
                    "page": page.name,
                    "deskewAngle": round(angle, 2),
                    "applied": diagnostics.applied,
                    "selected": diagnostics.selected_method,
                    "curvatureBefore": diagnostics.curvature_before_px,
                    "curvatureAfter": diagnostics.curvature_after_px,
                    "perspectiveBefore": diagnostics.perspective_before,
                    "perspectiveAfter": diagnostics.perspective_after,
                    "durationMs": round(diagnostics.duration_ms),
                    "reason": diagnostics.reason or "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Source document.")
    parser.add_argument("--page", type=int, default=3, help="One-based page for the order probe.")
    parser.add_argument("--dpi", type=int, default=216)
    parser.add_argument("--tesseract", type=Path, help="Tesseract executable for the OCR probe.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory for probe images.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--probe",
        choices=("resample", "order", "gate", "all"),
        default="all",
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    source = render_page(args.pdf, args.page - 1, args.dpi)
    print(
        f"{args.pdf.name} page {args.page} at {args.dpi} DPI: {source.shape[1]}x{source.shape[0]}",
        flush=True,
    )

    report: dict[str, object] = {
        "pdf": args.pdf.name,
        "page": args.page,
        "dpi": args.dpi,
    }

    if args.probe in ("resample", "all"):
        rows = probe_resample(source)
        report["resample"] = rows
        print("\nInterpolation passes (same image, geometry held at identity)")
        for row in rows:
            print(f"  {row['pass']:<28} {row['sharpness']:>8.1f}")

    if args.probe in ("order", "all"):
        rows = probe_order(source, tesseract=args.tesseract, work_dir=args.work_dir)
        report["order"] = rows
        print("\nStage order on the detected halves")
        for row in rows:
            line = (
                f"  {row['tag']:<28} {row['size']:>11} curv {row['curvature']:>6.3f} "
                f"persp {row['perspective']:>7.4f} sharp {row['sharpness']:>7.1f}"
            )
            if "alnumChars" in row:
                line += f" chars {row['alnumChars']:>5} conf {row['meanConfidence']:>6.2f}"
            print(line)

    if args.probe in ("gate", "all"):
        rows = probe_gate(args.pdf, args.dpi)
        report["gate"] = rows
        print("\nValidated automatic dewarp decisions")
        for row in rows:
            print(
                f"  {row['page'][-14:]:<16} applied={str(row['applied']):<5} "
                f"selected={row['selected']:<9} "
                f"curv {row['curvatureBefore']:.3f}->{row['curvatureAfter']:.3f} "
                f"{row['durationMs']:>5}ms  {row['reason']}"
            )

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
