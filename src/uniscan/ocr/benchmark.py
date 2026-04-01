"""OCR benchmark helpers for sampled PDF fixtures."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from uniscan.io import imwrite_unicode, render_pdf_page_indices

from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_OLMOCR,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_VALUES,
    SEARCHABLE_PDF_ENGINES,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)
from .preprocessing import _strip_markdown

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PADDLE_CACHE_HOME = _REPO_ROOT / ".paddlex_cache"
_DEFAULT_HF_CACHE_HOME = _REPO_ROOT / ".hf_cache"
_DEFAULT_MODELSCOPE_CACHE_HOME = _REPO_ROOT / ".modelscope_cache"
_DEFAULT_SURYA_MODEL_CACHE_HOME = _REPO_ROOT / ".surya_cache"
_DEFAULT_YOLO_CONFIG_HOME = _REPO_ROOT / ".ultralytics"
_DEFAULT_RUNTIME_TMP_HOME = _REPO_ROOT / ".tmp_runtime"


@dataclass(slots=True)
class OcrBenchmarkResult:
    engine: str
    status: str
    sample_pages: list[int]
    elapsed_seconds: float
    artifact_path: str | None
    text_chars: int
    memory_delta_mb: float | None = None
    error: str | None = None
    note: str | None = None

    @property
    def label(self) -> str:
        return OCR_ENGINE_LABELS.get(self.engine, self.engine)


def sample_pdf_page_indices(page_count: int, *, sample_size: int = 5) -> list[int]:
    """Pick an evenly distributed page sample without loading the whole PDF."""
    if page_count <= 0:
        return []

    target = max(1, int(sample_size))
    if page_count <= target:
        return list(range(page_count))

    if target == 1:
        return [0]

    indices: list[int] = []
    for index in range(target):
        # Even spread across [0, page_count - 1], including both ends.
        page_index = round(index * (page_count - 1) / (target - 1))
        if page_index not in indices:
            indices.append(page_index)

    if len(indices) < target:
        for page_index in range(page_count):
            if page_index in indices:
                continue
            indices.append(page_index)
            if len(indices) == target:
                break

    return indices


def resolve_pdf_page_indices(
    page_count: int,
    *,
    sample_size: int = 5,
    page_numbers: Sequence[int] | None = None,
) -> list[int]:
    """Resolve 0-based page indices either from explicit page numbers or sampled spread."""
    if page_count <= 0:
        return []

    if page_numbers is not None:
        resolved: list[int] = []
        seen: set[int] = set()
        for raw_page in page_numbers:
            page = int(raw_page)
            if page < 1:
                raise ValueError(f"Invalid page number: {page}. Page numbers must be >= 1.")
            page_index = page - 1
            if page_index >= page_count:
                raise ValueError(
                    f"Invalid page number: {page}. PDF has {page_count} pages (valid range is 1..{page_count})."
                )
            if page_index in seen:
                continue
            seen.add(page_index)
            resolved.append(page_index)

        if not resolved:
            raise ValueError("No valid page numbers were provided.")
        return resolved

    return sample_pdf_page_indices(page_count, sample_size=sample_size)


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _collect_text_strings(value: Any) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        try:
            return [value.decode("utf-8", errors="ignore")]
        except Exception:
            return []
    if isinstance(value, dict):
        for item in value.values():
            texts.extend(_collect_text_strings(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_collect_text_strings(item))
        return texts

    for attr in ("text", "rec_text", "transcription", "content", "label"):
        if hasattr(value, attr):
            texts.extend(_collect_text_strings(getattr(value, attr)))
    return texts


_PADDLEOCR_LANG_MAP: dict[str, str] = {
    "eng": "en",
    "en": "en",
    "english": "en",
    "rus": "ru",
    "ru": "ru",
    "russian": "ru",
    "deu": "german",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "jpn": "japan",
    "kor": "korean",
    "ara": "ar",
}


def _paddleocr_lang(lang: str) -> str:
    """Map Tesseract-style language codes to PaddleOCR identifiers.

    Handles multi-language specs like ``rus+eng`` by returning the first
    mapped language (PaddleOCR does not support multi-lang in a single call).
    """
    for part in lang.split("+"):
        normalized = part.strip().lower()
        if normalized in _PADDLEOCR_LANG_MAP:
            return _PADDLEOCR_LANG_MAP[normalized]
    # Fallback: return the first component as-is.
    return lang.split("+")[0].strip().lower()


def _render_sample_paths(
    pdf_path: Path,
    sample_pages: Sequence[int],
    *,
    dpi: int,
    tmp_dir: Path,
) -> list[Path]:
    rendered = render_pdf_page_indices(pdf_path, sample_pages, dpi=dpi)
    image_paths: list[Path] = []
    for idx, (_name, image) in enumerate(rendered, start=1):
        out_path = tmp_dir / f"{idx:05d}.png"
        if not imwrite_unicode(out_path, image):
            raise RuntimeError(f"Failed to write sampled page image: {out_path}")
        image_paths.append(out_path)
    return image_paths


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    try:
        parts: list[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                parts.append(page_text)
        return "\n".join(parts)
    finally:
        doc.close()


def _extract_pdf_text_chars(pdf_path: Path) -> int:
    return len(_extract_pdf_text(pdf_path))


def _memory_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _memory_delta_mb(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 3)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _configure_chandra_runtime_device() -> str:
    """Resolve TORCH_DEVICE for Chandra before importing chandra.settings."""

    explicit = (os.environ.get("TORCH_DEVICE") or "").strip()
    prefer_gpu = _env_bool("UNISCAN_CHANDRA_PREFER_GPU", default=True)
    require_gpu = _env_bool("UNISCAN_CHANDRA_REQUIRE_GPU", default=False)

    def _cuda_probe() -> tuple[bool, str]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            return False, f"torch import failed: {exc}"
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:
            return False, f"torch.cuda probe failed: {exc}"
        if available:
            return True, f"torch={getattr(torch, '__version__', 'unknown')}"
        return False, f"torch={getattr(torch, '__version__', 'unknown')} cuda_available=False"

    if explicit:
        if explicit.lower().startswith("cuda"):
            has_cuda, detail = _cuda_probe()
            if not has_cuda and require_gpu:
                raise RuntimeError(
                    f"Chandra GPU mode requires CUDA, but TORCH_DEVICE='{explicit}' is unusable ({detail})."
                )
        return explicit

    if not prefer_gpu and not require_gpu:
        os.environ.setdefault("TORCH_DEVICE", "cpu")
        return os.environ["TORCH_DEVICE"]

    has_cuda, detail = _cuda_probe()
    if has_cuda:
        os.environ["TORCH_DEVICE"] = "cuda:0"
        return "cuda:0"

    if require_gpu:
        raise RuntimeError(
            "Chandra GPU mode is required, but CUDA is unavailable. "
            f"Install a CUDA-enabled torch build in the Chandra venv ({detail})."
        )

    os.environ.setdefault("TORCH_DEVICE", "cpu")
    return os.environ["TORCH_DEVICE"]


def _artifact_path_for_engine(output_dir: Path, pdf_stem: str, engine: str) -> Path:
    suffix = ".pdf" if engine in SEARCHABLE_PDF_ENGINES else ".txt"
    return output_dir / f"{pdf_stem}_{engine}{suffix}"


def _markerized_pages_text(
    *,
    page_texts: Sequence[str],
    source_pages_1based: Sequence[int],
) -> str:
    blocks: list[str] = []
    for page_no, text in zip(source_pages_1based, page_texts, strict=True):
        blocks.append(f"[SOURCE PAGE {page_no:04d}]")
        if text:
            blocks.append(text.rstrip())
        blocks.append("")
    payload = "\n".join(blocks).strip()
    return payload + "\n" if payload else ""


def _write_pagewise_text_artifacts(
    *,
    output_dir: Path,
    engine: str,
    pdf_path: Path,
    source_pages_1based: Sequence[int],
    page_texts: Sequence[str],
    aggregate_path: Path,
    page_metadata: Sequence[dict[str, Any]] | None = None,
) -> tuple[int, Path]:
    engine_dir = output_dir / engine
    engine_dir.mkdir(parents=True, exist_ok=True)

    pages_payload: list[dict[str, Any]] = []
    metadata_by_page: dict[int, dict[str, Any]] = {}
    if page_metadata:
        for item in page_metadata:
            if not isinstance(item, dict):
                continue
            source_page = item.get("source_page")
            if isinstance(source_page, int):
                metadata_by_page[source_page] = item

    total_chars = 0
    for source_page, text in zip(source_pages_1based, page_texts, strict=True):
        page_file = engine_dir / f"page_{source_page:04d}.txt"
        page_file.write_text(text, encoding="utf-8")
        chars = len(text)
        total_chars += chars
        page_info: dict[str, Any] = {
            "source_page": source_page,
            "file": page_file.name,
            "text_chars": chars,
        }
        page_meta = metadata_by_page.get(source_page, {})
        sidecar_specs = (
            ("surya_page_lines_path", "surya_text_lines", "surya"),
            ("chandra_page_lines_path", "chandra_text_lines", "chandra"),
        )
        for key, geometry_type, suffix in sidecar_specs:
            sidecar_raw = page_meta.get(key)
            if not isinstance(sidecar_raw, str):
                continue
            sidecar_src = Path(sidecar_raw)
            if not sidecar_src.exists():
                continue
            sidecar_name = f"page_{source_page:04d}.{suffix}.json"
            sidecar_dst = engine_dir / sidecar_name
            try:
                shutil.copy2(sidecar_src, sidecar_dst)
                page_info["geometry_file"] = sidecar_name
                page_info["geometry_type"] = geometry_type
                break
            except Exception:
                continue

        pages_payload.append(page_info)

    markerized = _markerized_pages_text(
        page_texts=page_texts,
        source_pages_1based=source_pages_1based,
    )
    (engine_dir / "all_pages.txt").write_text(markerized, encoding="utf-8")
    aggregate_path.write_text(markerized, encoding="utf-8")

    pages_index = {
        "pdf_path": str(pdf_path),
        "engine": engine,
        "pages": pages_payload,
        "total_text_chars": total_chars,
        "aggregate_file": "all_pages.txt",
        "aggregate_has_page_markers": True,
    }
    pages_json_path = engine_dir / "pages.json"
    pages_json_path.write_text(
        json.dumps(pages_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return total_chars, pages_json_path


def _run_extraction_engine_pagewise(
    engine: str,
    image_paths: Sequence[Path],
    *,
    source_pages_1based: Sequence[int],
    lang: str,
    work_dir: Path,
    which_fn,
    run_cmd,
) -> tuple[list[str], int, list[dict[str, Any]], list[dict[str, Any]]]:
    if len(image_paths) != len(source_pages_1based):
        raise ValueError("image_paths and source_pages_1based lengths must match.")

    page_texts: list[str] = []
    total_chars = 0
    page_errors: list[dict[str, Any]] = []
    page_metadata: list[dict[str, Any]] = []
    for image_path, source_page in zip(image_paths, source_pages_1based, strict=True):
        page_work_dir = work_dir / f"page_{source_page:04d}"
        try:
            text, chars = _run_extraction_engine(
                engine,
                [image_path],
                lang=lang,
                work_dir=page_work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
            page_texts.append(text)
            total_chars += int(chars)
            if engine == OCR_ENGINE_SURYA:
                sidecar = page_work_dir / "surya_page_lines.json"
                if sidecar.exists():
                    page_metadata.append(
                        {
                            "source_page": source_page,
                            "surya_page_lines_path": str(sidecar),
                        }
                    )
            if engine == OCR_ENGINE_CHANDRA:
                sidecar = page_work_dir / "chandra_page_lines.json"
                if sidecar.exists():
                    page_metadata.append(
                        {
                            "source_page": source_page,
                            "chandra_page_lines_path": str(sidecar),
                        }
                    )
        except Exception as exc:
            page_texts.append("")
            page_errors.append(
                {
                    "source_page": source_page,
                    "image": str(image_path),
                    "error": str(exc),
                }
            )

    if page_errors and not any(text.strip() for text in page_texts):
        preview = "; ".join(f"p{item['source_page']}: {item['error']}" for item in page_errors[:3])
        raise RuntimeError(f"all sampled pages failed for {engine}: {preview}")
    return page_texts, total_chars, page_errors, page_metadata


def _module_presence_probe(name: str):
    """Import-probe compatible callable without importing heavyweight modules."""
    if importlib.util.find_spec(name) is None:
        raise ImportError(name)
    return object()


def _create_runtime_work_dir(*, prefix: str) -> Path:
    root_raw = (os.environ.get("UNISCAN_RUNTIME_TMP") or "").strip()
    root = Path(root_raw) if root_raw else _DEFAULT_RUNTIME_TMP_HOME
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(64):
        candidate = root / f"{prefix}{uuid.uuid4().hex[:12]}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate runtime work dir under '{root}'.")


def _run_paddleocr_direct(image_paths: Sequence[Path], *, lang: str) -> tuple[str, int]:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_DEFAULT_PADDLE_CACHE_HOME))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_USE_PIR_TRT", "false")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang=_paddleocr_lang(lang),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    collected: list[str] = []
    for path in image_paths:
        result = ocr.ocr(str(path))
        collected.extend(_collect_text_strings(result))

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_surya_module_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    run_cmd,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for Surya OCR.")

    input_dir = work_dir / "surya_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    # Surya CLI consumes a directory. For pagewise mode we must isolate input
    # files, otherwise every call re-processes all sibling pages and pollutes
    # per-page artifacts with foreign text.
    ordered_names: list[str] = []
    for idx, image_path in enumerate(image_paths, start=1):
        src = Path(image_path)
        if not src.exists():
            continue
        if len(image_paths) == 1:
            target_name = src.name
        else:
            target_name = f"{idx:04d}_{src.name}"
        target = input_dir / target_name
        shutil.copy2(src, target)
        ordered_names.append(target_name)

    if not ordered_names:
        raise RuntimeError("Surya input directory is empty after staging images.")

    output_root = work_dir / "surya_out"
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", str(_DEFAULT_SURYA_MODEL_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
    # Surya opens images via PIL internally — lift the decompression-bomb
    # guard so it can process high-resolution pages.
    try:
        from PIL import Image as _PIL_Image  # type: ignore
        _PIL_Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    from surya.scripts.ocr_text import ocr_text_cli

    args = [
        str(input_dir),
        "--output_dir",
        str(output_root),
    ]
    try:
        ocr_text_cli.main(args=args, standalone_mode=False)
    except SystemExit as exc:
        if int(getattr(exc, "code", 1) or 0) != 0:
            raise RuntimeError(f"surya.scripts.ocr_text exited with code {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"surya.scripts.ocr_text failed: {exc}") from exc

    results_json = output_root / input_dir.name / "results.json"
    if not results_json.exists():
        raise RuntimeError(f"Surya did not produce results file: {results_json}")

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    collected: list[str] = []
    sidecar_images: list[dict[str, Any]] = []
    consumed = False
    if isinstance(payload, dict):
        for image_name in ordered_names:
            pages = payload.get(image_name)
            if not isinstance(pages, list):
                # Some builds key by stem, not full filename.
                pages = payload.get(Path(image_name).stem)
            if not isinstance(pages, list):
                continue
            consumed = True
            image_payload: dict[str, Any] = {"image_name": image_name, "pages": []}
            for page in pages:
                if not isinstance(page, dict):
                    continue
                page_payload: dict[str, Any] = {}
                image_bbox = page.get("image_bbox")
                if (
                    isinstance(image_bbox, (list, tuple))
                    and len(image_bbox) == 4
                    and all(isinstance(item, (int, float)) for item in image_bbox)
                ):
                    page_payload["image_bbox"] = [float(item) for item in image_bbox]

                line_payload: list[dict[str, Any]] = []
                for line in page.get("text_lines", []):
                    if isinstance(line, dict):
                        text = (line.get("text") or "").strip()
                        bbox = line.get("bbox")
                        if (
                            text
                            and isinstance(bbox, (list, tuple))
                            and len(bbox) == 4
                            and all(isinstance(item, (int, float)) for item in bbox)
                        ):
                            line_payload.append(
                                {
                                    "text": text,
                                    "bbox": [float(item) for item in bbox],
                                }
                            )
                        if text:
                            collected.append(text)
                if line_payload:
                    page_payload["text_lines"] = line_payload
                if page_payload:
                    image_payload["pages"].append(page_payload)
            if image_payload["pages"]:
                sidecar_images.append(image_payload)

    # Fallback for unknown payload layouts.
    if not consumed:
        for pages in payload.values():
            if not isinstance(pages, list):
                continue
            for page in pages:
                if not isinstance(page, dict):
                    continue
                for line in page.get("text_lines", []):
                    if isinstance(line, dict):
                        text = (line.get("text") or "").strip()
                        if text:
                            collected.append(text)
    if sidecar_images:
        sidecar_path = work_dir / "surya_page_lines.json"
        sidecar_path.write_text(
            json.dumps({"images": sidecar_images}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    text = "\n".join(collected)
    return text, len(text)


def _run_mineru_module_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    run_cmd,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for MinerU OCR.")

    input_dir = image_paths[0].parent
    output_root = work_dir / "mineru_out"
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(_DEFAULT_YOLO_CONFIG_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # MinerU only supports "en" and "ch"; map common Tesseract codes.
    _first_lang = lang.split("+")[0].strip().lower()
    if _first_lang in {"eng", "en", "english"}:
        mineru_lang = "en"
    elif _first_lang in {"chi_sim", "chi_tra", "ch", "chinese"}:
        mineru_lang = "ch"
    else:
        # Unsupported language — fall back to "en" which at least handles
        # Latin subset; MinerU has no Cyrillic/Russian model.
        import warnings
        warnings.warn(
            f"MinerU does not support language '{lang}'; falling back to 'en'.",
            stacklevel=2,
        )
        mineru_lang = "en"
    # MinerU converts input images to PDF internally via PIL — lift the
    # decompression-bomb guard before importing the module.
    try:
        from PIL import Image as _PIL_Image  # type: ignore
        _PIL_Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    from mineru.cli.client import main as mineru_main

    args = [
        "-p",
        str(input_dir),
        "-o",
        str(output_root),
        "-m",
        "ocr",
        "-b",
        "pipeline",
        "-l",
        mineru_lang,
    ]
    try:
        mineru_main.main(args=args, standalone_mode=False)
    except SystemExit as exc:
        if int(getattr(exc, "code", 1) or 0) != 0:
            raise RuntimeError(f"mineru.cli.client exited with code {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"mineru.cli.client failed: {exc}") from exc

    # Primary path: Markdown exported by MinerU.
    text_parts: list[str] = []
    for path in sorted(output_root.rglob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        cleaned = _strip_markdown(raw)
        if cleaned:
            text_parts.append(cleaned)

    # Some MinerU builds emit empty markdown but keep OCR text in
    # *_content_list.json. Use it only as a fallback when markdown is empty.
    if not text_parts:
        for path in sorted(output_root.rglob("*_content_list.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    text_parts.append(text)

    if not text_parts:
        raise RuntimeError("MinerU finished without text artifacts.")

    text = "\n".join(text_parts)
    return text, len(text)


def _run_text_engine_from_cli(
    image_paths: Sequence[Path],
    *,
    engine: str,
    lang: str,
    candidates: Sequence[tuple[str, ...]],
    which_fn,
    run_cmd,
) -> tuple[str, int]:
    collected: list[str] = []
    errors: list[str] = []

    for image_path in image_paths:
        page_text: str | None = None
        for template in candidates:
            binary = template[0]
            binary_path = which_fn(binary) or which_fn(f"{binary}.exe")
            if binary_path is None:
                continue
            args = [str(binary_path)] + [part.format(image=str(image_path), lang=lang) for part in template[1:]]
            proc = run_cmd(args, capture_output=True, text=True)
            if int(getattr(proc, "returncode", 1)) == 0:
                stdout = getattr(proc, "stdout", "") or ""
                stderr = getattr(proc, "stderr", "") or ""
                combined = (stdout + "\n" + stderr).strip()
                if engine == OCR_ENGINE_SURYA and binary in {"marker_single", "marker"}:
                    marker_text = _extract_marker_cli_text(combined)
                    page_text = marker_text if marker_text else combined
                else:
                    page_text = combined
                if page_text:
                    collected.append(page_text)
                break
            stderr = (getattr(proc, "stderr", "") or "").strip()
            stdout = (getattr(proc, "stdout", "") or "").strip()
            details = stderr or stdout or "unknown cli error"
            errors.append(f"{binary}: {details}")

        if page_text is None:
            if not errors:
                raise RuntimeError(f"Engine '{engine}' has no runnable CLI candidates in PATH.")
            raise RuntimeError(f"Engine '{engine}' failed on {image_path.name}: {' | '.join(errors)}")

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _extract_marker_cli_text(log_blob: str) -> str:
    """Extract OCR text from marker CLI logs by reading saved markdown files."""
    matches = re.findall(r"Saved markdown to\s+([^\r\n]+)", log_blob)
    if not matches:
        return ""

    collected: list[str] = []
    for raw_path in matches:
        marker_path = Path(raw_path.strip().strip("'\""))
        if marker_path.is_file() and marker_path.suffix.lower() == ".md":
            md_files = [marker_path]
        elif marker_path.is_dir():
            md_files = sorted(marker_path.glob("*.md"))
        else:
            md_files = []

        for md_file in md_files:
            try:
                raw_md = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            cleaned = _strip_markdown(raw_md)
            if cleaned:
                collected.append(cleaned)

    return "\n".join(part for part in collected if part and not part.isspace())


def _run_surya_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    module_error: Exception | None = None
    try:
        return _run_surya_module_cli(image_paths, lang=lang, work_dir=work_dir, run_cmd=run_cmd)
    except Exception as exc:
        module_error = exc

    candidates = (
        ("surya_ocr", "{image}", "--lang", "{lang}"),
        ("surya_ocr", "--input", "{image}", "--lang", "{lang}"),
        ("surya_ocr", "--image", "{image}", "--lang", "{lang}"),
        ("marker_single", "{image}"),
        ("marker", "{image}"),
    )
    try:
        return _run_text_engine_from_cli(
            image_paths,
            engine=OCR_ENGINE_SURYA,
            lang=lang,
            candidates=candidates,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _run_mineru_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    module_error: Exception | None = None
    try:
        return _run_mineru_module_cli(image_paths, lang=lang, work_dir=work_dir, run_cmd=run_cmd)
    except Exception as exc:
        module_error = exc

    candidates = (
        ("mineru", "{image}", "--lang", "{lang}"),
        ("mineru", "--input", "{image}", "--lang", "{lang}"),
        ("magic-pdf", "{image}", "--lang", "{lang}"),
        ("magic-pdf", "--input", "{image}", "--lang", "{lang}"),
    )
    try:
        return _run_text_engine_from_cli(
            image_paths,
            engine=OCR_ENGINE_MINERU,
            lang=lang,
            candidates=candidates,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _run_chandra_module(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
) -> tuple[str, int]:
    """Run Chandra OCR via direct Python module import (preferred path)."""
    if len(image_paths) == 0:
        raise ValueError("No images for Chandra OCR.")

    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    selected_device = _configure_chandra_runtime_device()
    # Chandra uses PIL internally — lift the decompression-bomb guard.
    try:
        from PIL import Image as _PIL_Image  # type: ignore
        _PIL_Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass

    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.input import load_image

    def _chunk_text(raw_content: Any) -> str:
        if raw_content is None:
            return ""
        text = re.sub(r"<[^>]+>", " ", str(raw_content))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _safe_bbox(raw_bbox: Any, width: int, height: int) -> list[float] | None:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            x0 = float(raw_bbox[0])
            y0 = float(raw_bbox[1])
            x1 = float(raw_bbox[2])
            y1 = float(raw_bbox[3])
        except Exception:
            return None
        x0 = max(0.0, min(x0, float(width)))
        y0 = max(0.0, min(y0, float(height)))
        x1 = max(0.0, min(x1, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    _ = lang
    model = InferenceManager(method="hf")
    if _env_bool("UNISCAN_CHANDRA_REQUIRE_GPU", default=False):
        model_device = str(getattr(getattr(model, "model", None), "device", ""))
        if not model_device.lower().startswith("cuda"):
            raise RuntimeError(
                "Chandra model loaded without CUDA device while GPU mode is required "
                f"(TORCH_DEVICE={selected_device!r}, model_device={model_device!r})."
            )

    collected: list[str] = []
    sidecar_images: list[dict[str, Any]] = []
    for image_path in image_paths:
        pil_image = load_image(str(image_path))
        width, height = pil_image.size
        batch = [BatchInputItem(image=pil_image, prompt_type="ocr_layout")]
        results = model.generate(batch, include_images=False, include_headers_footers=False)

        page_texts: list[str] = []
        page_lines: list[dict[str, Any]] = []
        for result in results:
            chunks = getattr(result, "chunks", None)
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    line_text = _chunk_text(chunk.get("content"))
                    if not line_text:
                        continue
                    page_texts.append(line_text)
                    bbox = _safe_bbox(chunk.get("bbox"), width, height)
                    if bbox is not None:
                        page_lines.append({"text": line_text, "bbox": bbox})
            if not page_texts:
                md = getattr(result, "markdown", "") or ""
                md = _strip_markdown(md.strip())
                if md:
                    for line in md.splitlines():
                        line = line.strip()
                        if line:
                            page_texts.append(line)

        if not page_texts:
            raise RuntimeError(f"Chandra OCR produced no text for {image_path.name}.")

        collected.append("\n".join(page_texts))
        if page_lines:
            sidecar_images.append(
                {
                    "image_name": image_path.name,
                    "pages": [
                        {
                            "image_bbox": [0.0, 0.0, float(width), float(height)],
                            "text_lines": page_lines,
                        }
                    ],
                }
            )

    if sidecar_images:
        sidecar_path = work_dir / "chandra_page_lines.json"
        sidecar_path.write_text(
            json.dumps({"images": sidecar_images}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_chandra_cli(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    """Run Chandra OCR via CLI binary (fallback path)."""
    if len(image_paths) == 0:
        raise ValueError("No images for Chandra OCR.")

    chandra_cmd = which_fn("chandra") or which_fn("chandra.exe")
    if not chandra_cmd:
        raise RuntimeError("Chandra CLI was not found in PATH.")

    collected: list[str] = []
    errors: list[str] = []
    for image_path in image_paths:
        page_output = work_dir / image_path.stem
        page_output.mkdir(parents=True, exist_ok=True)
        candidates = (
            [str(chandra_cmd), str(image_path), str(page_output), "--method", "hf"],
            [str(chandra_cmd), str(image_path), str(page_output)],
        )
        run_ok = False
        for command in candidates:
            proc = run_cmd(command, capture_output=True, text=True)
            if int(getattr(proc, "returncode", 1)) == 0:
                run_ok = True
                break
            stderr = (getattr(proc, "stderr", "") or "").strip()
            stdout = (getattr(proc, "stdout", "") or "").strip()
            details = stderr or stdout or "unknown cli error"
            errors.append(details)

        if not run_ok:
            raise RuntimeError(
                f"Chandra OCR failed on {image_path.name}: " + " | ".join(errors[-2:])
            )

        page_texts: list[str] = []
        for pattern in ("*.md", "*.txt", "*.json", "*.html"):
            for artifact in sorted(page_output.rglob(pattern)):
                try:
                    text = artifact.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if text:
                    page_texts.append(text)
        if not page_texts:
            raise RuntimeError(f"Chandra OCR produced no text artifacts for {image_path.name}.")
        collected.append("\n".join(page_texts))

    text = "\n".join(part for part in collected if part and not part.isspace())
    return text, len(text)


def _run_chandra_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    # Primary: direct Python module import (no CLI binary needed).
    module_error: Exception | None = None
    try:
        return _run_chandra_module(image_paths, lang=lang, work_dir=work_dir)
    except Exception as exc:
        module_error = exc

    # Fallback: CLI binary via shutil.which.
    try:
        return _run_chandra_cli(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    except Exception as cli_exc:
        if module_error is not None:
            raise RuntimeError(f"{module_error} | fallback: {cli_exc}") from cli_exc
        raise


def _collect_olmocr_workspace_text(workspace: Path) -> tuple[str, int]:
    markdown_candidates: list[Path] = []
    markdown_dir = workspace / "markdown"
    patterns = ("*.md", "*.markdown", "*.mmd", "*.txt")
    if markdown_dir.exists():
        for pattern in patterns:
            markdown_candidates.extend(sorted(markdown_dir.rglob(pattern)))
    if not markdown_candidates:
        for pattern in patterns:
            markdown_candidates.extend(sorted(workspace.rglob(pattern)))
    # Preserve first-seen order and drop duplicates when multiple glob patterns match.
    markdown_candidates = list(dict.fromkeys(markdown_candidates))

    text_parts: list[str] = []

    def _load_json_relaxed(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Some OCRFlux outputs contain bare backslashes in markdown fields
            # (for example "\_" or "\("), which are invalid JSON escapes.
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
            if fixed == raw:
                raise
            return json.loads(fixed)

    def _append_payload_text(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        for key in (
            "text",
            "content",
            "document_text",
            "document_markdown",
            "natural_text",
            "markdown",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(_strip_markdown(value.strip()))
        for key in ("page_texts", "pages", "page_markdown", "page_text"):
            value = payload.get(key)
            for extracted in _collect_text_strings(value):
                cleaned = _strip_markdown(extracted.strip())
                if cleaned:
                    text_parts.append(cleaned)

    for md_path in markdown_candidates:
        try:
            raw = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        cleaned = _strip_markdown(raw)
        if cleaned:
            text_parts.append(cleaned)

    # Fallback for formats that keep text in JSON/JSONL payloads.
    for json_path in sorted(workspace.rglob("*.json")):
        try:
            payload = _load_json_relaxed(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(payload, dict):
            _append_payload_text(payload)
        elif isinstance(payload, list):
            for item in payload:
                _append_payload_text(item)

    for jsonl_path in sorted(workspace.rglob("*.jsonl")):
        try:
            lines = jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = _load_json_relaxed(line)
            except Exception:
                continue
            _append_payload_text(payload)

    for compressed_path in sorted(workspace.rglob("*.jsonl.zst")):
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            break
        try:
            with compressed_path.open("rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    raw = reader.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = _load_json_relaxed(line)
            except Exception:
                continue
            _append_payload_text(payload)

    if not text_parts:
        raise RuntimeError("olmOCR finished without markdown/text artifacts.")

    text = "\n".join(part for part in text_parts if part and not part.isspace())
    return text, len(text)


def _render_images_to_pdf(image_paths: Sequence[Path], out_pdf: Path) -> None:
    if len(image_paths) == 0:
        raise ValueError("No images to render into PDF.")
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("olmOCR docker fallback requires PyMuPDF. Install with: pip install pymupdf") from exc

    output_doc = fitz.open()
    try:
        for image_path in image_paths:
            image_doc = fitz.open(str(image_path))
            try:
                image_pdf = fitz.open("pdf", image_doc.convert_to_pdf())
                try:
                    output_doc.insert_pdf(image_pdf)
                finally:
                    image_pdf.close()
            finally:
                image_doc.close()
        output_doc.save(str(out_pdf))
    finally:
        output_doc.close()


def _run_olmocr_docker(
    image_paths: Sequence[Path],
    *,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    docker_cmd = which_fn("docker") or which_fn("docker.exe")
    if not docker_cmd:
        raise RuntimeError("docker is not available in PATH for olmOCR docker fallback.")

    docker_root = work_dir / "olmocr_docker"
    data_dir = docker_root / "data"
    work_root = docker_root / "work"
    workspace_dir = work_root / "ws"
    for directory in (data_dir, work_root):
        directory.mkdir(parents=True, exist_ok=True)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)

    input_pdf = data_dir / "input.pdf"
    _render_images_to_pdf(image_paths, input_pdf)

    image = (os.environ.get("UNISCAN_OLMOCR_DOCKER_IMAGE") or "chatdoc/ocrflux:latest").strip()
    gpu = (os.environ.get("UNISCAN_OLMOCR_DOCKER_GPU") or "all").strip()
    model = (os.environ.get("UNISCAN_OLMOCR_DOCKER_MODEL") or "").strip()
    workers = (os.environ.get("UNISCAN_OLMOCR_DOCKER_WORKERS") or "1").strip()
    gpu_mem_util = (os.environ.get("UNISCAN_OLMOCR_DOCKER_GPU_MEM_UTIL") or "").strip()
    pages_per_group = (os.environ.get("UNISCAN_OLMOCR_DOCKER_PAGES_PER_GROUP") or "").strip()
    max_page_retries = (os.environ.get("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_RETRIES") or "").strip()
    # ocrflux default is 1/250 (~0.004), which is too strict for noisy scans.
    # In page-wise mode (single page per invocation), any fallback would discard
    # the whole page unless we relax the threshold to 1.0.
    default_error_rate = "1.0" if len(image_paths) <= 1 else "0.10"
    max_page_error_rate = (
        os.environ.get("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_ERROR_RATE") or default_error_rate
    ).strip()

    cache_dir_raw = (os.environ.get("UNISCAN_OLMOCR_DOCKER_CACHE") or str(_REPO_ROOT / ".hf_cache_ocrflux")).strip()
    cache_dir = Path(cache_dir_raw)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mount_data = data_dir.resolve().as_posix()
    mount_work = work_root.resolve().as_posix()
    mount_cache = cache_dir.resolve().as_posix()

    command: list[str] = [str(docker_cmd), "run", "--rm"]
    if gpu and gpu.lower() != "none":
        command.extend(["--gpus", gpu])
    command.extend(
        [
            "-v",
            f"{mount_data}:/data:ro",
            "-v",
            f"{mount_work}:/work",
            "-v",
            f"{mount_cache}:/root/.cache/huggingface",
            image,
            "/work/ws",
            "--task",
            "pdf2markdown",
            "--data",
            "/data/input.pdf",
        ]
    )
    if workers:
        command.extend(["--workers", workers])
    if pages_per_group:
        command.extend(["--pages_per_group", pages_per_group])
    if max_page_retries:
        command.extend(["--max_page_retries", max_page_retries])
    if max_page_error_rate:
        command.extend(["--max_page_error_rate", max_page_error_rate])
    if model:
        command.extend(["--model", model])
    if gpu_mem_util:
        command.extend(["--gpu_memory_utilization", gpu_mem_util])

    proc = run_cmd(command, capture_output=True, text=True)
    stderr = (getattr(proc, "stderr", "") or "").strip()
    stdout = (getattr(proc, "stdout", "") or "").strip()
    if int(getattr(proc, "returncode", 1)) != 0:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        stdout = (getattr(proc, "stdout", "") or "").strip()
        details = stderr or stdout or "unknown docker olmOCR error"
        raise RuntimeError(f"docker olmOCR failed: {details}")

    if not workspace_dir.exists():
        raise RuntimeError("docker olmOCR finished but did not create workspace.")

    try:
        return _collect_olmocr_workspace_text(workspace_dir)
    except Exception as exc:
        details = stderr or stdout
        file_hints: list[str] = []
        try:
            for path in sorted(workspace_dir.rglob("*")):
                if not path.is_file():
                    continue
                file_hints.append(str(path.relative_to(workspace_dir)))
                if len(file_hints) >= 25:
                    break
        except Exception:
            pass
        if details:
            details = details[-2000:]
            if file_hints:
                raise RuntimeError(
                    f"{exc} | docker output tail: {details} | workspace files: {', '.join(file_hints)}"
                ) from exc
            raise RuntimeError(f"{exc} | docker output tail: {details}") from exc
        if file_hints:
            raise RuntimeError(f"{exc} | workspace files: {', '.join(file_hints)}") from exc
        raise


def _run_olmocr_direct(
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> tuple[str, int]:
    if len(image_paths) == 0:
        raise ValueError("No images for olmOCR.")

    backend = (os.environ.get("UNISCAN_OLMOCR_BACKEND") or "auto").strip().lower()
    if backend not in {"auto", "local", "docker"}:
        raise ValueError("UNISCAN_OLMOCR_BACKEND must be one of: auto, local, docker")

    workspace = work_dir / "olmocr_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # olmOCR language selection is model-driven; keep API compatible with
    # other engines by accepting `lang` but not forcing a language flag.
    _ = lang

    server = (os.environ.get("UNISCAN_OLMOCR_SERVER") or "").strip()
    model = (os.environ.get("UNISCAN_OLMOCR_MODEL") or "").strip()
    api_key = (os.environ.get("UNISCAN_OLMOCR_API_KEY") or "").strip()

    base_args = [str(workspace), "--markdown", "--pdfs", *[str(path) for path in image_paths]]
    if server:
        base_args += ["--server", server]
    if model:
        base_args += ["--model", model]
    if api_key:
        base_args += ["--api_key", api_key]

    errors: list[str] = []
    if backend in {"auto", "local"}:
        command_candidates: list[list[str]] = []
        olmocr_cmd = which_fn("olmocr") or which_fn("olmocr.exe")
        if olmocr_cmd:
            command_candidates.append([str(olmocr_cmd), *base_args])
        command_candidates.append([sys.executable, "-m", "olmocr.pipeline", *base_args])

        if not command_candidates:
            errors.append("local olmOCR command is not available in PATH.")
        else:
            for command in command_candidates:
                command_env = dict(os.environ)
                command_path = Path(command[0])
                if command_path.exists():
                    bin_dir = str(command_path.resolve().parent)
                    current_path = command_env.get("PATH", "")
                    command_env["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir
                proc = run_cmd(command, capture_output=True, text=True, env=command_env)
                if int(getattr(proc, "returncode", 1)) == 0:
                    return _collect_olmocr_workspace_text(workspace)
                stderr = (getattr(proc, "stderr", "") or "").strip()
                stdout = (getattr(proc, "stdout", "") or "").strip()
                details = stderr or stdout or "unknown olmOCR error"
                errors.append(f"{command[0]}: {details}")

    if backend in {"auto", "docker"}:
        try:
            return _run_olmocr_docker(
                image_paths,
                work_dir=work_dir,
                which_fn=which_fn,
                run_cmd=run_cmd,
            )
        except Exception as exc:
            errors.append(str(exc))

    if not errors:
        raise RuntimeError("olmOCR failed for unknown reason.")
    raise RuntimeError("olmOCR failed: " + " | ".join(errors))


def _run_extraction_engine(
    engine: str,
    image_paths: Sequence[Path],
    *,
    lang: str,
    work_dir: Path,
    which_fn,
    run_cmd,
) -> tuple[str, int]:
    if engine == OCR_ENGINE_PADDLEOCR:
        return _run_paddleocr_direct(image_paths, lang=lang)
    if engine == OCR_ENGINE_SURYA:
        return _run_surya_direct(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    if engine == OCR_ENGINE_MINERU:
        return _run_mineru_direct(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    if engine == OCR_ENGINE_CHANDRA:
        return _run_chandra_direct(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    if engine == OCR_ENGINE_OLMOCR:
        return _run_olmocr_direct(
            image_paths,
            lang=lang,
            work_dir=work_dir,
            which_fn=which_fn,
            run_cmd=run_cmd,
        )
    raise ValueError(f"Unsupported extraction engine: {engine}")


def _make_result(
    *,
    engine: str,
    status: str,
    sample_pages: Sequence[int],
    elapsed_seconds: float,
    artifact_path: Path | None,
    text_chars: int,
    memory_delta_mb: float | None,
    error: str | None = None,
    note: str | None = None,
) -> OcrBenchmarkResult:
    return OcrBenchmarkResult(
        engine=engine,
        status=status,
        sample_pages=[page + 1 for page in sample_pages],
        elapsed_seconds=elapsed_seconds,
        artifact_path=None if artifact_path is None else str(artifact_path),
        text_chars=text_chars,
        memory_delta_mb=memory_delta_mb,
        error=error,
        note=note,
    )


def run_ocr_benchmark(
    *,
    pdf_path: Path,
    output_dir: Path,
    engines: Sequence[str] | None = None,
    sample_size: int = 5,
    page_numbers: Sequence[int] | None = None,
    dpi: int = 160,
    lang: str = "eng",
    import_module=None,
    which_fn=shutil.which,
    run_cmd=subprocess.run,
) -> list[OcrBenchmarkResult]:
    """Run a sampled OCR benchmark against a PDF fixture."""
    resolved_pdf = Path(pdf_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    page_count = _pdf_page_count(resolved_pdf)
    sample_pages = resolve_pdf_page_indices(
        page_count,
        sample_size=sample_size,
        page_numbers=page_numbers,
    )
    if not sample_pages:
        raise ValueError("No PDF pages available for OCR benchmark.")

    selected_engines = tuple(engines) if engines is not None else OCR_ENGINE_VALUES
    import_probe = import_module or _module_presence_probe
    results: list[OcrBenchmarkResult] = []
    source_pages_1based = [page + 1 for page in sample_pages]

    tmp_dir = _create_runtime_work_dir(prefix="uniscan_ocr_benchmark_")
    try:
        sampled_image_paths = _render_sample_paths(
            resolved_pdf,
            sample_pages,
            dpi=dpi,
            tmp_dir=tmp_dir,
        )

        for engine_name in selected_engines:
            engine = engine_name.strip().lower()
            start = perf_counter()
            rss_before = _memory_rss_mb()
            artifact_path = _artifact_path_for_engine(resolved_output, resolved_pdf.stem, engine)
            try:
                engine_status = detect_ocr_engine_status(
                    engine,
                    import_module=import_probe,
                    which_fn=which_fn,
                )
            except Exception as exc:
                elapsed = perf_counter() - start
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        error=str(exc),
                        note="status detection failed",
                    )
                )
                continue

            if not engine_status.ready:
                elapsed = perf_counter() - start
                missing = ", ".join(engine_status.missing) if engine_status.missing else "unknown"
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        note=f"missing: {missing}",
                    )
                )
                continue

            try:
                if engine in SEARCHABLE_PDF_ENGINES:
                    output_pdf = image_paths_to_searchable_pdf(
                        sampled_image_paths,
                        out_pdf=artifact_path,
                        lang=lang,
                        engine_name=engine,
                    )
                    extracted_text = _extract_pdf_text(output_pdf)
                    text_chars = len(extracted_text)
                    # Keep native searchable PDF artifact and also write plain text
                    # sidecar to simplify downstream comparisons.
                    output_pdf.with_suffix(".txt").write_text(extracted_text, encoding="utf-8")
                    elapsed = perf_counter() - start
                    results.append(
                        _make_result(
                            engine=engine,
                            status="ok",
                            sample_pages=sample_pages,
                            elapsed_seconds=elapsed,
                            artifact_path=output_pdf,
                            text_chars=text_chars,
                            memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        )
                    )
                    continue

                # Keep extraction engines page-aware: persist per-page files and
                # write markerized aggregate text that preserves source page ids.
                page_texts, text_chars, page_errors, page_metadata = _run_extraction_engine_pagewise(
                    engine,
                    sampled_image_paths,
                    source_pages_1based=source_pages_1based,
                    lang=lang,
                    work_dir=tmp_dir / f"{engine}_work",
                    which_fn=which_fn,
                    run_cmd=run_cmd,
                )
                _write_pagewise_text_artifacts(
                    output_dir=resolved_output,
                    engine=engine,
                    pdf_path=resolved_pdf,
                    source_pages_1based=source_pages_1based,
                    page_texts=page_texts,
                    aggregate_path=artifact_path,
                    page_metadata=page_metadata,
                )
                elapsed = perf_counter() - start
                results.append(
                    _make_result(
                        engine=engine,
                        status="ok",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=text_chars,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        note=(
                            f"partial page failures: {len(page_errors)} / {len(source_pages_1based)}"
                            if page_errors
                            else None
                        ),
                    )
                )
            except Exception as exc:
                elapsed = perf_counter() - start
                results.append(
                    _make_result(
                        engine=engine,
                        status="error",
                        sample_pages=sample_pages,
                        elapsed_seconds=elapsed,
                        artifact_path=artifact_path,
                        text_chars=0,
                        memory_delta_mb=_memory_delta_mb(rss_before, _memory_rss_mb()),
                        error=str(exc),
                    )
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report_path = resolved_output / f"{resolved_pdf.stem}_ocr_benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "pdf_path": str(resolved_pdf),
                "page_count": page_count,
                "sample_pages": [page + 1 for page in sample_pages],
                "results": [asdict(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return results


def summarize_ocr_benchmark(results: Sequence[OcrBenchmarkResult]) -> str:
    """Format a concise human-readable benchmark summary."""
    lines: list[str] = []
    for result in results:
        memory_part = "" if result.memory_delta_mb is None else f" mem={result.memory_delta_mb:+.2f}MB"
        if result.status == "ok":
            lines.append(
                f"{result.engine}: ok {result.elapsed_seconds:.2f}s "
                f"text={result.text_chars}{memory_part} artifact={result.artifact_path}"
            )
            continue
        lines.append(
            f"{result.engine}: error {result.elapsed_seconds:.2f}s "
            f"{result.error or result.note or 'unknown error'}{memory_part}"
        )
    return "\n".join(lines)
