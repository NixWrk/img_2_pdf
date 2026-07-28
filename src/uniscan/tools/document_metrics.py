"""Geometry and OCR metrics used by standard document benchmark profiles."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


DOCUNET_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_docunet_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    target_area: int = 598400,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the DocUNet target-area and paired-size preprocessing."""
    if target_area <= 0:
        raise ValueError("DocUNet target area must be positive.")
    reference_gray = _gray_u8(reference)
    candidate_gray = _gray_u8(candidate)
    height, width = reference_gray.shape
    scale = math.sqrt(float(target_area) / float(height * width))
    target_height = max(1, int(math.ceil(height * scale)))
    target_width = max(1, int(math.ceil(width * scale)))
    size = (target_width, target_height)
    return (
        cv2.resize(reference_gray, size, interpolation=cv2.INTER_CUBIC),
        cv2.resize(candidate_gray, size, interpolation=cv2.INTER_CUBIC),
    )


def _ssim_gray(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = reference.astype(np.float32) / 255.0
    y = candidate.astype(np.float32) / 255.0
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_x**2
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_y**2
    sigma_xy = (
        cv2.GaussianBlur(x * y, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_x * mu_y
    )
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    values = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / np.maximum(
        denominator, np.finfo(np.float32).eps
    )
    return float(np.clip(np.mean(values), -1.0, 1.0))


def docunet_ms_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    target_area: int = 598400,
) -> float:
    """Compute the DocUNet five-level weighted SSIM with OpenCV pyramids.

    The protocol and weights match DocUNet. Numerical output is identified as a
    Python/OpenCV reproduction because MATLAB ``ssim`` and ``impyramid`` versions differ.
    """
    x, y = resize_docunet_pair(reference, candidate, target_area=target_area)
    scores: list[float] = []
    for index in range(len(DOCUNET_MS_SSIM_WEIGHTS)):
        scores.append(_ssim_gray(x, y))
        if index + 1 < len(DOCUNET_MS_SSIM_WEIGHTS):
            x = cv2.pyrDown(x)
            y = cv2.pyrDown(y)
    return float(sum(weight * score for weight, score in zip(DOCUNET_MS_SSIM_WEIGHTS, scores)))


def axis_aligned_distortion_from_flow(reference: np.ndarray, flow: np.ndarray) -> float:
    """Apply the published AAD equations to a reference-to-candidate flow field."""
    gray = _gray_u8(reference).astype(np.float32)
    if flow.shape != (*gray.shape, 2):
        raise ValueError("AAD flow must have shape (height, width, 2).")
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    gx_max = float(np.max(gx))
    gy_max = float(np.max(gy))
    if gx_max > 0:
        gx /= gx_max
    if gy_max > 0:
        gy /= gy_max
    vx = flow[..., 0]
    vy = flow[..., 1]
    epsilon = np.finfo(np.float32).eps
    row_mean = np.sum(vy * gy, axis=1) / (np.sum(gy, axis=1) + epsilon)
    column_mean = np.sum(vx * gx, axis=0) / (np.sum(gx, axis=0) + epsilon)
    row_deviation = gy * np.abs(vy - row_mean[:, None])
    column_deviation = gx * np.abs(vx - column_mean[None, :])
    return float(np.mean(np.hypot(row_deviation, column_deviation)))


def aad_opencv_dis_proxy(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    target_area: int = 598400,
) -> float:
    """Compute AAD equations with OpenCV DIS flow; this is not official SIFTflow AAD."""
    reference_gray, candidate_gray = resize_docunet_pair(
        reference, candidate, target_area=target_area
    )
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = estimator.calc(reference_gray, candidate_gray, None)
    return axis_aligned_distortion_from_flow(reference_gray, flow)


def levenshtein_distance(reference: str, candidate: str) -> int:
    """Return character edit distance using linear memory."""
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


def tesseract_version(executable: str | Path) -> str:
    """Return the first Tesseract version line or fail with a useful error."""
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Cannot execute Tesseract {executable}: {exc}") from exc
    output = (result.stdout or result.stderr).decode("utf-8", errors="replace").strip()
    if result.returncode != 0 or not output:
        raise ValueError(f"Tesseract version check failed ({result.returncode}): {output}")
    return output.splitlines()[0]


def tesseract_text(
    image_path: Path,
    *,
    executable: str | Path,
    language: str | None = None,
) -> str:
    """Recognize one image through the Tesseract CLI with no hidden preprocessing."""
    command = [str(executable), str(image_path), "stdout"]
    if language:
        command.extend(("-l", language))
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Tesseract failed for {image_path}: {exc}") from exc
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Tesseract failed for {image_path} ({result.returncode}): {details}")
    return result.stdout.decode("utf-8", errors="replace")
