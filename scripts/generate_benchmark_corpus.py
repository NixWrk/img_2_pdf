"""Generate the deterministic, original UniScan crop benchmark corpus."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "corpus_v1"

CASES = (
    ("document", "document", [[110, 50], [505, 72], [540, 420], [82, 400]]),
    ("whiteboard", "whiteboard", [[50, 90], [590, 60], [610, 410], [35, 430]]),
    ("photograph", "photograph", [[100, 75], [550, 110], [520, 405], [75, 380]]),
    ("book", "book", [[55, 45], [575, 55], [610, 430], [28, 420]]),
    ("difficult-lighting", "difficult_lighting", [[130, 55], [520, 80], [570, 410], [85, 430]]),
)


def _page(category: str) -> np.ndarray:
    page = np.full((560, 420, 3), 246, dtype=np.uint8)
    ink = (45, 45, 45)
    cv2.rectangle(page, (0, 0), (419, 559), (220, 220, 220), 5)
    cv2.putText(page, category.replace("_", " ").upper(), (28, 55), 0, 0.8, ink, 2)
    for y, width in ((105, 345), (135, 315), (165, 350), (230, 325), (260, 350), (290, 280)):
        cv2.line(page, (32, y), (32 + width, y), ink, 3)
    cv2.rectangle(page, (32, 330), (180, 470), (95, 135, 190), -1)
    cv2.circle(page, (288, 400), 65, (90, 175, 115), -1)
    if category == "whiteboard":
        page[:] = (238, 244, 242)
        cv2.line(page, (40, 110), (350, 210), (40, 80, 210), 8)
        cv2.line(page, (70, 380), (340, 250), (50, 160, 80), 8)
        cv2.circle(page, (205, 270), 72, (180, 80, 60), 7)
    elif category == "photograph":
        yy, xx = np.mgrid[:560, :420]
        page[:, :, 0] = np.clip(50 + xx * 0.35, 0, 255)
        page[:, :, 1] = np.clip(70 + yy * 0.25, 0, 255)
        page[:, :, 2] = np.clip(180 - xx * 0.2, 0, 255)
        cv2.circle(page, (210, 260), 110, (40, 180, 240), -1)
    elif category == "book":
        cv2.line(page, (210, 8), (210, 550), (80, 80, 80), 10)
        cv2.line(page, (70, 100), (180, 100), ink, 3)
        cv2.line(page, (240, 100), (350, 100), ink, 3)
    return page


def _scene(category: str, corners: list[list[int]]) -> np.ndarray:
    height, width = 480, 640
    yy, xx = np.mgrid[:height, :width]
    background = np.empty((height, width, 3), dtype=np.uint8)
    texture = ((xx * 3 + yy * 2) % 19).astype(np.uint8)
    background[:] = (36, 45, 52)
    background = cv2.add(background, cv2.merge((texture, texture, texture)))

    source = _page(category)
    source_quad = np.float32([[0, 0], [419, 0], [419, 559], [0, 559]])
    destination = np.float32(corners)
    transform = cv2.getPerspectiveTransform(source_quad, destination)
    warped = cv2.warpPerspective(source, transform, (width, height))
    mask = cv2.warpPerspective(np.full(source.shape[:2], 255, np.uint8), transform, (width, height))
    background[mask > 0] = warped[mask > 0]

    if category == "difficult_lighting":
        shadow = np.linspace(0.35, 1.15, width, dtype=np.float32)[None, :, None]
        background = np.clip(background.astype(np.float32) * shadow, 0, 255).astype(np.uint8)
        cv2.circle(background, (470, 145), 75, (255, 255, 255), -1)
    return background


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case_id, category, corners in CASES:
        filename = f"{case_id}.png"
        if not cv2.imwrite(str(OUTPUT / filename), _scene(category, corners)):
            raise RuntimeError(f"Cannot write {filename}")
        manifest_cases.append(
            {"id": case_id, "category": category, "image": filename, "corners": corners}
        )
    manifest = {
        "schemaVersion": 1,
        "version": "1.0.0",
        "license": "MIT",
        "generator": "scripts/generate_benchmark_corpus.py",
        "cases": manifest_cases,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
