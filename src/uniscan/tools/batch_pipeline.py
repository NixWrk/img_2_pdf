"""Headless input-to-PDF pipeline built from the production scanner primitives."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from uniscan.core.pipeline import PipelineOptions, process_loaded_items
from uniscan.core.postprocess import POSTPROCESSING_OPTIONS
from uniscan.core.preprocess import (
    PREPROCESS_PRESETS,
    apply_enhancements,
    resolve_lens_mode_profile,
)
from uniscan.export import export_image_paths_as_files, export_image_paths_as_pdf
from uniscan.io import (
    IMG_EXTS,
    PDF_EXTS,
    imwrite_unicode,
    list_supported_in_folder,
    load_input_items,
)


LENS_MODE_CHOICES = ("none", "document", "whiteboard", "photo", "b/w")


@dataclass(slots=True, frozen=True)
class BatchPipelineResult:
    """Summary of one completed headless conversion."""

    output_pdf: Path
    input_files: tuple[Path, ...]
    image_outputs: tuple[Path, ...]
    total_pages: int
    detected_pages: int


def resolve_input_paths(inputs: Sequence[Path], *, output_pdf: Path) -> tuple[Path, ...]:
    """Expand files and folders while preserving argument and natural folder order."""
    if not inputs:
        raise ValueError("At least one input file or folder is required.")

    output_resolved = output_pdf.with_suffix(".pdf").resolve()
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")

        if path.is_dir():
            candidates = list_supported_in_folder(path)
        elif path.is_file():
            if path.suffix.lower() not in (IMG_EXTS | PDF_EXTS):
                raise ValueError(f"Unsupported input: {path}")
            if path.resolve() == output_resolved:
                raise ValueError("Output PDF cannot also be an explicit input file.")
            candidates = [path]
        else:
            raise ValueError(f"Input is neither a file nor a folder: {path}")

        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if candidate_resolved == output_resolved or candidate_resolved in seen:
                continue
            seen.add(candidate_resolved)
            resolved.append(candidate)

    if not resolved:
        raise ValueError("No supported image or PDF inputs were found.")
    return tuple(resolved)


def _resolve_processing(mode: str):
    normalized = mode.strip().lower()
    if normalized == "none":
        return POSTPROCESSING_OPTIONS["None"], None

    profiles_by_key = {
        name.lower(): profile
        for name, profile in (
            (name, resolve_lens_mode_profile(name))
            for name in ("Document", "Whiteboard", "Photo", "B/W")
        )
    }
    profile = profiles_by_key.get(normalized)
    if profile is None:
        raise ValueError(f"Unsupported lens mode: {mode}")
    return POSTPROCESSING_OPTIONS[profile.postprocess_name], PREPROCESS_PRESETS[profile.preset_name]


def run_batch_pipeline(
    *,
    inputs: Sequence[Path],
    output_pdf: Path,
    images_dir: Path | None = None,
    image_format: str = "png",
    pdf_dpi: int = 300,
    detect_document: bool = True,
    two_page_mode: bool = False,
    lens_mode: str = "document",
) -> BatchPipelineResult:
    """Run the complete headless pre-OCR pipeline and write a merged PDF."""
    dpi = int(pdf_dpi)
    if dpi < 72:
        raise ValueError("PDF DPI must be >= 72.")

    output_pdf = Path(output_pdf).with_suffix(".pdf")
    input_files = resolve_input_paths(inputs, output_pdf=output_pdf)
    if images_dir is not None:
        images_resolved = Path(images_dir).resolve()
        input_dirs = {path.parent.resolve() for path in input_files}
        if images_resolved in input_dirs:
            raise ValueError("Images output directory cannot be an input directory.")

    postprocess, preprocess_settings = _resolve_processing(lens_mode)
    options = PipelineOptions(
        detect_document=bool(detect_document),
        two_page_mode=bool(two_page_mode),
        postprocess_name="None",
    )

    detected_pages = 0
    with tempfile.TemporaryDirectory(prefix="uniscan_convert_") as tmp:
        staging_dir = Path(tmp)
        staged_paths: list[Path] = []

        for source_path in input_files:
            loaded_items = load_input_items([source_path], pdf_dpi=dpi)
            page_results = process_loaded_items(loaded_items, options=options)
            for page in page_results:
                current = postprocess(page.current)
                if preprocess_settings is not None:
                    current = apply_enhancements(current, preprocess_settings)
                page_path = staging_dir / f"{len(staged_paths) + 1:05d}.png"
                if not imwrite_unicode(page_path, current):
                    raise RuntimeError(f"Failed to write processed page: {page_path}")
                staged_paths.append(page_path)
                if page.backend is not None:
                    detected_pages += 1

        if not staged_paths:
            raise ValueError("The input did not produce any pages.")

        written_pdf = export_image_paths_as_pdf(staged_paths, out_pdf=output_pdf, dpi=dpi)
        image_outputs: tuple[Path, ...] = ()
        if images_dir is not None:
            image_outputs = tuple(
                export_image_paths_as_files(
                    staged_paths,
                    output_dir=Path(images_dir),
                    ext=image_format,
                    base_name="page",
                )
            )

    return BatchPipelineResult(
        output_pdf=written_pdf,
        input_files=input_files,
        image_outputs=image_outputs,
        total_pages=len(staged_paths),
        detected_pages=detected_pages,
    )
