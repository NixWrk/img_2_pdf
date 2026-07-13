from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from uniscan.core.geometry import order_quad_points

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - OpenCV fallback is enough for runtime use.
    Image = None
    ImageOps = None


PACKAGE_DIR = Path(__file__).resolve().parent
_MODEL_DIR_ENV = os.environ.get("UNISCAN_OFFICE_LENS_MODEL_DIR")
# The extracted weights are intentionally not distributed.  These exported
# paths remain useful to diagnostics, but construction resolves the environment
# again so applications may configure BYOM after importing this module.
MODEL_DIR = Path(_MODEL_DIR_ENV).expanduser() if _MODEL_DIR_ENV else PACKAGE_DIR / "models"
QUAD_MODEL = MODEL_DIR / "mnv2_ep42_wb_quant.ort"
CLASSIFIER_MODEL = MODEL_DIR / "triclass_doc_classifier.ort"

CLASSIFIER_LABELS = ("Document", "Photo", "Whiteboard")
PROCESSING_MODES = ("document", "photo", "whiteboard")
RequestedMode = Literal["auto", "document", "photo", "whiteboard"]


def _load_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Office Lens ONNX requires the optional 'office-lens' extra (onnxruntime)."
        ) from exc
    return onnxruntime


def _resolve_model_path(explicit: str | Path | None, filename: str) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
    else:
        configured_dir = os.environ.get("UNISCAN_OFFICE_LENS_MODEL_DIR")
        if not configured_dir:
            raise RuntimeError(
                "Office Lens models are not bundled. Set UNISCAN_OFFICE_LENS_MODEL_DIR "
                "to a directory containing licensed model files."
            )
        path = Path(configured_dir).expanduser() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Office Lens model file is missing: {path}")
    return path


@dataclass(frozen=True)
class Classification:
    label: str
    scores: dict[str, float]


@dataclass(frozen=True)
class QuadMaskResult:
    mask: np.ndarray
    quad: np.ndarray | None
    threshold: float
    mask_quad: np.ndarray | None = None
    image_quad: np.ndarray | None = None


@dataclass(frozen=True)
class EnhancementResult:
    mode: str
    image: np.ndarray
    variants: dict[str, np.ndarray]


@dataclass(frozen=True)
class PipelineResult:
    image_width: int
    image_height: int
    classification: Classification
    mask_result: QuadMaskResult
    mode: str
    warped: np.ndarray | None
    enhancement: EnhancementResult | None


def _read_rgb(path: str | Path) -> np.ndarray:
    if Image is not None and ImageOps is not None:
        try:
            with Image.open(path) as image:
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        except Exception:
            pass

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_image(path: str | Path, image: np.ndarray) -> None:
    output = image
    if image.ndim == 3:
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), output)


def _resize_rgb(image_rgb: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_quad_mask(image_rgb: np.ndarray) -> np.ndarray:
    resized = _resize_rgb(image_rgb)
    return (resized.astype(np.float32) / 255.0)[None, :, :, :]


def preprocess_classifier(image_rgb: np.ndarray) -> np.ndarray:
    resized = _resize_rgb(image_rgb)
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[None, :, :, :]


def _order_quad(points: np.ndarray) -> np.ndarray:
    return order_quad_points(points)


def _clamp_quad(quad: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    clipped = quad.copy().astype(np.float32)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image_width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image_height - 1)
    return clipped


def _expand_quad(
    quad: np.ndarray,
    image_width: int,
    image_height: int,
    padding_percent: float = 0.0,
) -> np.ndarray:
    if padding_percent == 0:
        return _clamp_quad(quad, image_width, image_height)
    center = quad.astype(np.float32).mean(axis=0)
    expanded = center + (quad.astype(np.float32) - center) * (1.0 + padding_percent)
    return _clamp_quad(expanded, image_width, image_height)


def _quad_area(quad: np.ndarray | None) -> float:
    if quad is None:
        return 0.0
    return abs(float(cv2.contourArea(quad.astype(np.float32))))


def _quad_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    first_ordered = _order_quad(first).astype(np.float32)
    second_ordered = _order_quad(second).astype(np.float32)
    intersection_area, _ = cv2.intersectConvexConvex(first_ordered, second_ordered)
    union_area = _quad_area(first_ordered) + _quad_area(second_ordered) - float(intersection_area)
    if union_area <= 0:
        return 0.0
    return float(intersection_area) / union_area


def _quad_image_score(quad: np.ndarray | None, image_rgb: np.ndarray) -> float:
    if quad is None:
        return -1.0

    height, width = image_rgb.shape[:2]
    image_area = float(width * height)
    area = _quad_area(quad)
    area_ratio = area / image_area
    if area_ratio < 0.035 or area_ratio > 0.92:
        return -1.0

    ordered = _order_quad(quad)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
    if cv2.countNonZero(mask) == 0:
        return -1.0

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1][mask > 0]
    value = hsv[:, :, 2][mask > 0]
    center = ordered.mean(axis=0)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    center_distance = float(np.linalg.norm(center - image_center) / np.linalg.norm(image_center))
    border_margin = max(3.0, min(width, height) * 0.01)
    border_touches = sum(
        bool(
            point[0] <= border_margin
            or point[1] <= border_margin
            or point[0] >= width - border_margin
            or point[1] >= height - border_margin
        )
        for point in ordered
    )
    border_penalty = border_touches * 0.22
    oversized_penalty = max(area_ratio - 0.72, 0.0) * 1.5

    return (
        min(area_ratio, 0.75) * 1.5
        + float(value.mean()) / 255.0
        - float(saturation.mean()) / 510.0
        - center_distance * 0.2
        - border_penalty
        - oversized_penalty
    )


def detect_bright_document_quad(image_rgb: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    low_saturation = saturation < max(70, int(np.percentile(saturation, 45)))
    bright = value > max(110, int(otsu_threshold))
    combined = np.where((otsu_mask > 0) & (low_saturation | bright), 255, 0).astype(np.uint8)

    kernel = np.ones((13, 13), dtype=np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: np.ndarray | None = None
    best_score = -1.0
    for contour in contours:
        if cv2.contourArea(contour) < image_rgb.shape[0] * image_rgb.shape[1] * 0.035:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.int32)):
            quad = _order_quad(approx)
        else:
            quad = _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))

        score = _quad_image_score(quad, image_rgb)
        if score > best_score:
            best_score = score
            best = quad.astype(np.float32)

    if best is None:
        return None
    return _clamp_quad(best, image_rgb.shape[1], image_rgb.shape[0])


def detect_edge_document_quad(image_rgb: np.ndarray, max_edge: int = 1200) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    scale = min(max_edge / float(max(width, height)), 1.0)
    if scale < 1.0:
        work = cv2.resize(
            image_rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        work = image_rgb

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    low = int(max(20, np.percentile(gray, 35) * 0.66))
    high = int(min(255, max(low * 2, np.percentile(gray, 85))))
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: np.ndarray | None = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < work.shape[0] * work.shape[1] * 0.025:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True).reshape(-1, 2)
        candidates: list[np.ndarray] = []
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.int32)):
            candidates.append(_order_quad(approx))
        candidates.append(_order_quad(cv2.boxPoints(cv2.minAreaRect(contour))))

        for candidate in candidates:
            quad = candidate / scale if scale > 0 else candidate
            score = _quad_image_score(quad, image_rgb)
            if score > best_score:
                best_score = score
                best = quad.astype(np.float32)

    if best is None:
        return None
    return _clamp_quad(best, width, height)


def detect_image_document_quad(image_rgb: np.ndarray) -> np.ndarray | None:
    candidates = [
        detect_bright_document_quad(image_rgb),
        detect_edge_document_quad(image_rgb),
    ]
    best = max(candidates, key=lambda quad: _quad_image_score(quad, image_rgb))
    if _quad_image_score(best, image_rgb) < 0:
        return None
    return best


def choose_quad(
    mask_quad: np.ndarray | None,
    image_quad: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    if mask_quad is None:
        return image_quad
    if image_quad is None:
        return _clamp_quad(mask_quad, image_width, image_height)

    image_area = float(image_width * image_height)
    mask_area = _quad_area(mask_quad)
    image_area_quad = _quad_area(image_quad)
    if image_area_quad < image_area * 0.035:
        return _clamp_quad(mask_quad, image_width, image_height)

    area_ratio = image_area_quad / mask_area if mask_area > 0 else 0.0
    overlap = _quad_iou(mask_quad, image_quad)
    # Similar area alone is not evidence that both detectors found the same
    # object: two disjoint rectangles can have an area ratio of exactly 1.0.
    if overlap >= 0.15 and 0.35 <= area_ratio <= 1.3:
        return _clamp_quad(image_quad, image_width, image_height)
    return _clamp_quad(mask_quad, image_width, image_height)


def _candidate_score(quad_256: np.ndarray, mask: np.ndarray) -> float:
    area = abs(float(cv2.contourArea(quad_256.astype(np.float32))))
    image_area = float(mask.shape[0] * mask.shape[1])
    if area < image_area * 0.03 or area > image_area * 0.95:
        return -1.0

    edge_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.polylines(
        edge_mask, [quad_256.astype(np.int32).reshape((-1, 1, 2))], True, 255, 3, cv2.LINE_AA
    )
    edge_strength = float(mask[edge_mask > 0].mean()) if np.any(edge_mask) else 0.0
    area_score = min(area / image_area, 0.65)
    return edge_strength * 2.0 + area_score


def _quad_candidates(binary: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 16:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True).reshape(-1, 2)
        if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.int32)):
            candidates.append(_order_quad(approx))
        candidates.append(_order_quad(cv2.boxPoints(cv2.minAreaRect(contour))))

    points = cv2.findNonZero(binary)
    if points is not None and len(points) >= 4:
        candidates.append(_order_quad(cv2.boxPoints(cv2.minAreaRect(points))))
    return candidates


def mask_to_quad(
    mask: np.ndarray, image_width: int, image_height: int
) -> tuple[np.ndarray | None, float]:
    normalized = mask.astype(np.float32)
    if normalized.size == 0 or float(normalized.max()) <= 0:
        return None, 0.0

    kernel = np.ones((5, 5), dtype=np.uint8)
    thresholds = [
        0.55,
        0.45,
        0.35,
        0.25,
        max(float(normalized.max()) * 0.35, float(np.percentile(normalized, 85))),
    ]
    best_quad: np.ndarray | None = None
    best_score = -1.0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        binary = (normalized >= threshold).astype(np.uint8) * 255
        if cv2.countNonZero(binary) < 32:
            continue
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        for candidate in _quad_candidates(dilated):
            score = _candidate_score(candidate, normalized)
            if score > best_score:
                best_score = score
                best_quad = candidate
                best_threshold = threshold

    if best_quad is None:
        return None, 0.0

    scale = np.array([image_width / 256.0, image_height / 256.0], dtype=np.float32)
    return _clamp_quad(best_quad * scale, image_width, image_height), best_threshold


def mode_from_classification(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in PROCESSING_MODES:
        return normalized
    return "document"


def _normalize_luminance(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    sigma = max(12.0, min(gray.shape[:2]) / 22.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=220.0 + 20.0 * strength)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _clahe_luminance(image_rgb: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    gray_mean = float(channel_means.mean())
    scales = gray_mean / np.maximum(channel_means, 1.0)
    balanced = image * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def enhance_photo(image_rgb: np.ndarray) -> EnhancementResult:
    enhanced = _clahe_luminance(image_rgb, clip_limit=1.25)
    return EnhancementResult(mode="photo", image=enhanced, variants={"enhanced": enhanced})


def enhance_document(image_rgb: np.ndarray) -> EnhancementResult:
    balanced = _gray_world_white_balance(image_rgb)
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    normalized_l = _normalize_luminance(l_channel, strength=1.0)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(normalized_l)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    block_size = max(31, int(min(gray.shape[:2]) / 28) | 1)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        11,
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)

    return EnhancementResult(
        mode="document",
        image=enhanced,
        variants={
            "enhanced": enhanced,
            "gray": gray,
            "bw": bw,
        },
    )


def enhance_whiteboard(image_rgb: np.ndarray) -> EnhancementResult:
    balanced = _gray_world_white_balance(image_rgb)
    sigma = max(24.0, min(image_rgb.shape[:2]) / 14.0)
    background = cv2.GaussianBlur(balanced, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 1)
    normalized = cv2.divide(balanced, background, scale=232.0)
    mixed = cv2.addWeighted(balanced, 0.68, normalized, 0.32, 0)

    lab = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.06, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.02, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    return EnhancementResult(
        mode="whiteboard",
        image=enhanced,
        variants={
            "enhanced": enhanced,
            "gray": gray,
        },
    )


def enhance_page(image_rgb: np.ndarray, mode: RequestedMode | str) -> EnhancementResult:
    normalized = mode.lower()
    if normalized == "document":
        return enhance_document(image_rgb)
    if normalized == "whiteboard":
        return enhance_whiteboard(image_rgb)
    if normalized == "photo":
        return enhance_photo(image_rgb)
    raise ValueError(f"Unsupported processing mode: {mode}")


def warp_document_rgb(
    image_rgb: np.ndarray,
    quad: np.ndarray,
    padding_percent: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    image_height, image_width = image_rgb.shape[:2]
    ordered = _order_quad(_expand_quad(quad, image_width, image_height, padding_percent))
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])

    width = max(1, int(round(max(top_width, bottom_width))))
    height = max(1, int(round(max(left_height, right_height))))
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    warped = cv2.warpPerspective(image_rgb, transform, (width, height), flags=cv2.INTER_CUBIC)
    return warped, (width, height)


class OfficeLensOnnx:
    def __init__(
        self,
        quad_model: str | Path | None = None,
        classifier_model: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        selected_providers = providers or ["CPUExecutionProvider"]
        self._runtime = _load_onnxruntime()
        self._providers = selected_providers
        resolved_quad_model = _resolve_model_path(quad_model, QUAD_MODEL.name)
        self._classifier_model = classifier_model
        self.quad_session = self._runtime.InferenceSession(
            str(resolved_quad_model), providers=selected_providers
        )
        self.quad_input = self.quad_session.get_inputs()[0].name
        self.quad_output = self.quad_session.get_outputs()[0].name
        self.classifier_session = None
        self.classifier_input: str | None = None
        self.classifier_output: str | None = None

    def _ensure_classifier_session(self):
        if self.classifier_session is None:
            classifier_model = _resolve_model_path(
                self._classifier_model,
                CLASSIFIER_MODEL.name,
            )
            self.classifier_session = self._runtime.InferenceSession(
                str(classifier_model),
                providers=self._providers,
            )
            self.classifier_input = self.classifier_session.get_inputs()[0].name
            self.classifier_output = self.classifier_session.get_outputs()[0].name
        return self.classifier_session

    def classify(self, image_rgb: np.ndarray) -> Classification:
        session = self._ensure_classifier_session()
        assert self.classifier_input is not None
        assert self.classifier_output is not None
        tensor = preprocess_classifier(image_rgb)
        scores = session.run([self.classifier_output], {self.classifier_input: tensor})[0][0]
        score_map = {label: float(scores[index]) for index, label in enumerate(CLASSIFIER_LABELS)}
        return Classification(label=max(score_map, key=score_map.get), scores=score_map)

    def predict_quad_mask(self, image_rgb: np.ndarray) -> QuadMaskResult:
        tensor = preprocess_quad_mask(image_rgb)
        output = self.quad_session.run([self.quad_output], {self.quad_input: tensor})[0]
        mask = output[0, :, :, 0].astype(np.float32)
        mask_quad, threshold = mask_to_quad(mask, image_rgb.shape[1], image_rgb.shape[0])
        image_quad = detect_image_document_quad(image_rgb)
        quad = choose_quad(mask_quad, image_quad, image_rgb.shape[1], image_rgb.shape[0])
        return QuadMaskResult(
            mask=mask, quad=quad, threshold=threshold, mask_quad=mask_quad, image_quad=image_quad
        )

    def process_image(
        self,
        image_rgb: np.ndarray,
        mode: RequestedMode = "auto",
        padding_percent: float = 0.0,
    ) -> PipelineResult:
        classification = (
            self.classify(image_rgb)
            if mode == "auto"
            else Classification(label=str(mode).title(), scores={})
        )
        mask_result = self.predict_quad_mask(image_rgb)
        resolved_mode = mode_from_classification(classification.label) if mode == "auto" else mode

        warped: np.ndarray | None = None
        enhancement: EnhancementResult | None = None
        if mask_result.quad is not None:
            warped, _ = warp_document_rgb(
                image_rgb, mask_result.quad, padding_percent=padding_percent
            )
            enhancement = enhance_page(warped, resolved_mode)

        return PipelineResult(
            image_width=int(image_rgb.shape[1]),
            image_height=int(image_rgb.shape[0]),
            classification=classification,
            mask_result=mask_result,
            mode=str(resolved_mode),
            warped=warped,
            enhancement=enhancement,
        )

    def process_file(
        self,
        image_path: str | Path,
        mode: RequestedMode = "auto",
        padding_percent: float = 0.0,
    ) -> PipelineResult:
        return self.process_image(_read_rgb(image_path), mode=mode, padding_percent=padding_percent)

    def analyze_file(self, image_path: str | Path) -> dict[str, Any]:
        image_rgb = _read_rgb(image_path)
        result = self.process_image(image_rgb)
        return result_to_report(image_path, result)


def result_to_report(image_path: str | Path, result: PipelineResult) -> dict[str, Any]:
    mask_result = result.mask_result
    report: dict[str, Any] = {
        "image": str(image_path),
        "width": result.image_width,
        "height": result.image_height,
        "classification": {
            "label": result.classification.label,
            "scores": result.classification.scores,
        },
        "mode": result.mode,
        "quadMask": {
            "shape": list(mask_result.mask.shape),
            "min": float(mask_result.mask.min()),
            "max": float(mask_result.mask.max()),
            "mean": float(mask_result.mask.mean()),
            "threshold": float(mask_result.threshold),
            "quad": None if mask_result.quad is None else mask_result.quad.round(2).tolist(),
            "maskQuad": None
            if mask_result.mask_quad is None
            else mask_result.mask_quad.round(2).tolist(),
            "imageQuad": None
            if mask_result.image_quad is None
            else mask_result.image_quad.round(2).tolist(),
        },
    }
    if result.warped is not None:
        report["warped"] = {
            "width": int(result.warped.shape[1]),
            "height": int(result.warped.shape[0]),
        }
    if result.enhancement is not None:
        report["cleanup"] = {
            "mode": result.enhancement.mode,
            "variants": list(result.enhancement.variants.keys()),
        }
    return report


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    normalized = np.clip(mask, 0.0, 1.0)
    cv2.imwrite(str(output_path), (normalized * 255.0).astype(np.uint8))


def save_overlay(image_path: str | Path, quad: np.ndarray | None, output_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if quad is not None:
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
        for index, point in enumerate(quad.astype(np.int32)):
            cv2.circle(image, tuple(point), 6, (0, 0, 255), -1)
            cv2.putText(
                image, str(index), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
    cv2.imwrite(str(output_path), image)


def save_pipeline_outputs(
    image_path: str | Path,
    result: PipelineResult,
    output_dir: str | Path,
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = result_to_report(image_path, result)
    mask_path = output_dir / f"{image_path.stem}_quad_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_quad_overlay.png"
    save_mask(result.mask_result.mask, mask_path)
    save_overlay(image_path, result.mask_result.quad, overlay_path)
    report["outputs"] = {
        "mask": str(mask_path),
        "overlay": str(overlay_path),
    }

    if result.warped is not None:
        warped_path = output_dir / f"{image_path.stem}_warped.png"
        _write_image(warped_path, result.warped)
        report["outputs"]["warped"] = str(warped_path)

    if result.enhancement is not None:
        cleanup_outputs: dict[str, str] = {}
        for variant_name, image in result.enhancement.variants.items():
            suffix = "enhanced" if variant_name == "enhanced" else f"enhanced_{variant_name}"
            variant_path = output_dir / f"{image_path.stem}_{suffix}.png"
            _write_image(variant_path, image)
            cleanup_outputs[variant_name] = str(variant_path)
        report["outputs"]["cleanup"] = cleanup_outputs

    return report


def warp_document(
    image_path: str | Path, quad: np.ndarray, output_path: str | Path
) -> tuple[int, int]:
    image_rgb = _read_rgb(image_path)
    warped, size = warp_document_rgb(image_rgb, quad)
    _write_image(output_path, warped)
    return size
