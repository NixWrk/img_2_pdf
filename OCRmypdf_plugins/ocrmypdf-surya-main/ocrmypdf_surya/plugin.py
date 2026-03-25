"""Surya OCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from ocrmypdf import OrientationConfidence, hookimpl
from ocrmypdf.exceptions import BadArgsError, MissingDependencyError
from ocrmypdf.pluginspec import OcrEngine

from .hocr import build_hocr

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger(__name__)


_RUNTIME_CACHE: dict[str, Any] = {}


def _device_from_options(options) -> str:
    return str(getattr(options, "surya_device", "auto") or "auto").strip().lower()


def _resolve_surya_runtime(*, device: str) -> Any:
    cached = _RUNTIME_CACHE.get(device)
    if cached is not None:
        return cached

    if device != "auto":
        os.environ["TORCH_DEVICE"] = device

    from surya.common.surya.schema import TASK_NAMES, TaskNames
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    runtime = SimpleNamespace(
        task_names=set(TASK_NAMES),
        default_task=TaskNames.ocr_with_boxes,
        detection_predictor=det_predictor,
        recognition_predictor=rec_predictor,
    )
    _RUNTIME_CACHE[device] = runtime
    return runtime


def _safe_confidence(value: Any) -> float:
    try:
        conf = float(value or 0.0)
    except Exception:
        return 0.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def _safe_bbox(raw_bbox: Any, width: int, height: int) -> list[int]:
    width = max(1, int(width))
    height = max(1, int(height))
    try:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError("bbox must have 4 values")
        x0, y0, x1, y1 = (int(float(raw_bbox[0])), int(float(raw_bbox[1])), int(float(raw_bbox[2])), int(float(raw_bbox[3])))
    except Exception:
        return [0, 0, width, height]

    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def _line_to_payload(*, line: Any, width: int, height: int) -> dict[str, Any]:
    line_bbox = _safe_bbox(getattr(line, "bbox", None), width, height)
    line_conf = _safe_confidence(getattr(line, "confidence", 0.0))
    line_text = str(getattr(line, "text", "") or "").strip()

    words_payload: list[dict[str, Any]] = []
    words = getattr(line, "words", None) or []
    for word in words:
        text = str(getattr(word, "text", "") or "").strip()
        if not text:
            continue
        words_payload.append(
            {
                "text": text,
                "bbox": _safe_bbox(getattr(word, "bbox", None), width, height),
                "confidence": _safe_confidence(getattr(word, "confidence", 0.0)),
            }
        )

    if not words_payload and line_text:
        words_payload = [
            {
                "text": line_text,
                "bbox": line_bbox,
                "confidence": line_conf,
            }
        ]

    if not line_text and words_payload:
        line_text = " ".join(word["text"] for word in words_payload if word.get("text"))

    return {
        "text": line_text,
        "bbox": line_bbox,
        "confidence": line_conf,
        "words": words_payload,
    }


def _predict_surya_lines(input_file: Path, options) -> tuple[list[dict[str, Any]], str, int, int]:
    device = _device_from_options(options)
    runtime = _resolve_surya_runtime(device=device)

    requested_task = str(getattr(options, "surya_task_name", "") or "").strip()
    task_name = requested_task if requested_task else runtime.default_task
    if task_name not in runtime.task_names:
        task_name = runtime.default_task

    disable_math = bool(getattr(options, "surya_disable_math", False))

    with Image.open(input_file) as image:
        pil_image = image.convert("RGB")
        width, height = pil_image.size

    predictions = runtime.recognition_predictor(
        [pil_image],
        task_names=[task_name],
        det_predictor=runtime.detection_predictor,
        math_mode=not disable_math,
        return_words=True,
    )

    if not predictions:
        return [], "", width, height

    page_prediction = predictions[0]
    lines_payload: list[dict[str, Any]] = []
    plain_lines: list[str] = []
    for line in getattr(page_prediction, "text_lines", []) or []:
        payload = _line_to_payload(line=line, width=width, height=height)
        if payload["text"]:
            plain_lines.append(payload["text"])
        if payload["words"] or payload["text"]:
            lines_payload.append(payload)

    return lines_payload, "\n".join(plain_lines), width, height


class SuryaOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        try:
            return importlib.metadata.version("surya-ocr")
        except Exception:
            return "unknown"

    @staticmethod
    def creator_tag(options) -> str:
        return f"Surya {SuryaOcrEngine.version()}"

    def __str__(self) -> str:
        return f"Surya {self.version()}"

    @staticmethod
    def languages(options) -> set[str]:
        selected = set(str(lang).strip() for lang in (getattr(options, "languages", None) or []) if str(lang).strip())
        if selected:
            return selected
        # Surya is multilingual and autodetect-oriented; keep baseline compatibility.
        return {"eng", "rus"}

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        lines_payload, plain_text, page_width, page_height = _predict_surya_lines(input_file, options)
        language = "en"
        selected = getattr(options, "languages", None) or []
        if selected:
            language = str(selected[0]).strip() or "en"

        hocr = build_hocr(
            page_width=page_width,
            page_height=page_height,
            lines=lines_payload,
            language=language,
        )
        output_hocr.write_text(hocr, encoding="utf-8")
        output_text.write_text(plain_text, encoding="utf-8")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        tmp_hocr = output_pdf.with_suffix(".surya.hocr")
        SuryaOcrEngine.generate_hocr(input_file, tmp_hocr, output_text, options)

        from ocrmypdf.hocrtransform import HocrTransform

        with Image.open(input_file) as image:
            dpi = image.info.get("dpi", (300, 300))[0]

        transform = HocrTransform(hocr_filename=tmp_hocr, dpi=dpi)
        transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True,
        )

        if not bool(getattr(options, "surya_debug_keep_hocr", False)):
            try:
                tmp_hocr.unlink(missing_ok=True)
            except Exception:
                log.debug("Could not delete temporary hOCR file: %s", tmp_hocr)


@hookimpl
def add_options(parser):
    group = parser.add_argument_group("Surya", "Options for Surya OCR engine")
    group.add_argument(
        "--surya-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for Surya models (default: auto).",
    )
    group.add_argument(
        "--surya-disable-math",
        action="store_true",
        help="Disable Surya math recognition mode.",
    )
    group.add_argument(
        "--surya-task-name",
        default="ocr_with_boxes",
        help="Surya task name (default: ocr_with_boxes).",
    )
    group.add_argument(
        "--surya-debug-keep-hocr",
        action="store_true",
        help="Keep intermediate hOCR file next to output PDF.",
    )


@hookimpl
def check_options(options):
    if importlib.util.find_spec("surya") is None:
        raise MissingDependencyError(
            "surya-ocr is not installed. Install with: pip install surya-ocr"
        )

    if importlib.util.find_spec("surya.recognition") is None:
        raise MissingDependencyError(
            "surya.recognition is missing. Check your surya-ocr installation."
        )

    device = _device_from_options(options)
    if device == "cuda":
        try:
            import torch
        except Exception as exc:
            raise MissingDependencyError("PyTorch is required for Surya CUDA mode.") from exc
        if not torch.cuda.is_available():
            raise BadArgsError("Requested --surya-device=cuda but CUDA is not available.")


@hookimpl
def get_ocr_engine(options=None):
    return SuryaOcrEngine()
