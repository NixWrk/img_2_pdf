"""Reproduce and validate UniScan's pinned DocScanner-L ONNX grid asset.

This script intentionally lives outside the normal UniScan runtime. Run it in
an isolated environment containing PyTorch, ONNX, ONNX Runtime, OpenCV and
Pillow, against the exact official DocScanner source commit recorded below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn


SOURCE_COMMIT = "54f6063a61a52e4ce4012832e943d1871a9c3c66"
CHECKPOINTS = {
    "DocScanner-L.pth": (
        29_328_510,
        "1d907965aa5d8e99ea8d0891fb66d13bc4f23838547bac6f568d01d480ff8c8a",
    ),
    "seg.pth": (
        4_715_923,
        "cb79fdec55a5ed435dc74d8112aa9285d8213bae475022f711c709744fb19dd4",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(repo: Path) -> None:
    result = subprocess.run(  # noqa: S603 - fixed git operation, no shell
        [
            "git",
            "-c",
            f"safe.directory={repo.as_posix()}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"DocScanner source mismatch: expected {SOURCE_COMMIT}, got {actual}")


def _verified_checkpoint(directory: Path, name: str) -> Path:
    path = directory / name
    expected_size, expected_hash = CHECKPOINTS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
        raise RuntimeError(f"Pinned checkpoint verification failed: {path}")
    return path


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError(f"Unexpected checkpoint payload: {path}")
    return state


def _load_segmentation(model: nn.Module, path: Path) -> dict[str, int]:
    current = model.state_dict()
    upstream = _load_state(path)
    matched = {key[6:]: value for key, value in upstream.items() if key[6:] in current}
    ignored = set(upstream) - {key for key in upstream if key[6:] in current}
    missing = set(current) - set(matched)
    if missing:
        raise RuntimeError(f"Segmentation checkpoint leaves {len(missing)} tensors unloaded")
    model.load_state_dict(matched, strict=True)
    return {"loaded": len(matched), "ignored": len(ignored)}


def _load_rectifier(model: nn.Module, path: Path) -> dict[str, int]:
    current = model.state_dict()
    upstream = _load_state(path)
    matched = {key: value for key, value in upstream.items() if key in current}
    missing = set(current) - set(matched)
    if missing:
        raise RuntimeError(f"Rectifier checkpoint leaves {len(missing)} tensors unloaded")
    model.load_state_dict(matched, strict=True)
    return {"loaded": len(matched), "ignored": len(set(upstream) - set(matched))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    _verify_source(repo)
    sys.path.insert(0, str(repo))
    from model import DocScanner  # noqa: PLC0415
    from seg import U2NETP  # noqa: PLC0415

    class GridModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Keep the upstream/export-spike attribute names: ONNX embeds them
            # into graph names, so changing them changes the release bytes.
            self.msk = U2NETP(3, 1)
            self.bm = DocScanner()

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            mask, *_ = self.msk(image)
            masked = (mask > 0.5).float() * image
            backward_map = self.bm(masked, iters=12, test_mode=True)
            return (2 * (backward_map / 286.8) - 1) * 0.99

    model = GridModel()
    load = {
        "segmentation": _load_segmentation(
            model.msk, _verified_checkpoint(args.checkpoints, "seg.pth")
        ),
        "rectification": _load_rectifier(
            model.bm,
            _verified_checkpoint(args.checkpoints, "DocScanner-L.pth"),
        ),
    }
    model.eval()
    source = np.array(Image.open(args.input).convert("RGB"), dtype=np.float64) / 255.0
    sample = cv2.resize(source, (288, 288)).transpose(2, 0, 1)[None].astype(np.float32)
    tensor = torch.from_numpy(sample)
    with torch.inference_mode():
        expected = model(tensor).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.onnx.export(
        model,
        tensor,
        str(args.output),
        input_names=["image"],
        output_names=["grid"],
        opset_version=17,
        do_constant_folding=True,
    )
    export_ms = (time.perf_counter() - started) * 1000.0
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    actual = session.run(["grid"], {"image": sample})[0]
    ort_ms = (time.perf_counter() - started) * 1000.0
    absolute = np.abs(actual - expected)
    report = {
        "sourceCommit": SOURCE_COMMIT,
        "checkpointLoad": load,
        "torch": torch.__version__,
        "opset": 17,
        "nodes": len(graph.graph.node),
        "inputShape": list(sample.shape),
        "outputShape": list(actual.shape),
        "exportMs": export_ms,
        "ortCpuMs": ort_ms,
        "meanAbsoluteError": float(absolute.mean()),
        "maxAbsoluteError": float(absolute.max()),
        "allClose1e4": bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-4)),
        "bytes": args.output.stat().st_size,
        "sha256": _sha256(args.output),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["allClose1e4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
