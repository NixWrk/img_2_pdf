"""OCR benchmark helpers for sampled PDF fixtures."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from uniscan.io import imwrite_unicode, render_pdf_page_indices

from .engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_VALUES,
    SEARCHABLE_PDF_ENGINES,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PADDLE_CACHE_HOME = _REPO_ROOT / ".paddlex_cache"
_DEFAULT_HF_CACHE_HOME = _REPO_ROOT / ".hf_cache"
_DEFAULT_MODELSCOPE_CACHE_HOME = _REPO_ROOT / ".modelscope_cache"
_DEFAULT_SURYA_MODEL_CACHE_HOME = _REPO_ROOT / ".surya_cache"
_DEFAULT_YOLO_CONFIG_HOME = _REPO_ROOT / ".ultralytics"


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


def _paddleocr_lang(lang: str) -> str:
    """Map OCR language codes to PaddleOCR language identifiers."""
    normalized = lang.strip().lower()
    if normalized in {"eng", "en", "english"}:
        return "en"
    return normalized


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


def _extract_pdf_text_chars(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF import requires PyMuPDF. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    try:
        total = 0
        for page in doc:
            total += len(page.get_text("text"))
        return total
    finally:
        doc.close()


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


def _artifact_path_for_engine(output_dir: Path, pdf_stem: str, engine: str) -> Path:
    suffix = ".pdf" if engine in SEARCHABLE_PDF_ENGINES else ".txt"
    return output_dir / f"{pdf_stem}_{engine}{suffix}"


def _module_presence_probe(name: str):
    """Import-probe compatible callable without importing heavyweight modules."""
    if importlib.util.find_spec(name) is None:
        raise ImportError(name)
    return object()


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

    input_dir = image_paths[0].parent
    output_root = work_dir / "surya_out"
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", str(_DEFAULT_SURYA_MODEL_CACHE_HOME))
    os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_CACHE_HOME))
    os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE_HOME))
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

    mineru_lang = "en" if lang.strip().lower() in {"eng", "en", "english"} else "ch"
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

    text_parts: list[str] = []
    for suffix in ("*.md", "*.txt", "*.json"):
        for path in sorted(output_root.rglob(suffix)):
            try:
                payload = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if payload:
                text_parts.append(payload)

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
                page_text = ((getattr(proc, "stdout", "") or "") + "\n" + (getattr(proc, "stderr", "") or "")).strip()
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

    with tempfile.TemporaryDirectory(prefix="uniscan_ocr_benchmark_") as tmp:
        tmp_dir = Path(tmp)
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
                    text_chars = _extract_pdf_text_chars(output_pdf)
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

                text, text_chars = _run_extraction_engine(
                    engine,
                    sampled_image_paths,
                    lang=lang,
                    work_dir=tmp_dir / f"{engine}_work",
                    which_fn=which_fn,
                    run_cmd=run_cmd,
                )
                artifact_path.write_text(text, encoding="utf-8")
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
