"""Capture a deterministic Windows desktop matrix for manual UI regression review.

This is intentionally an opt-in visual check, not a golden-pixel CI test.  The
application is launched with an isolated temporary ``UNISCAN_STATE_DIR`` and
all screenshots are written below the explicit output directory supplied by
the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCENES = ("workspace", "advanced", "keyboard-focus")
WINDOW_SIZES = ((1280, 800), (1024, 680))
THEMES = ("Light", "Dark")
FOCUS_SCENE_WIDGET = "preview_fit_button"


@dataclass(frozen=True)
class CaptureSpec:
    """One requested screenshot in the manual review matrix."""

    theme: str
    width: int
    height: int
    scene: str


def capture_matrix() -> tuple[CaptureSpec, ...]:
    """Return the stable Light/Dark × size × scene capture matrix."""
    return tuple(
        CaptureSpec(theme, width, height, scene)
        for theme in THEMES
        for width, height in WINDOW_SIZES
        for scene in SCENES
    )


def capture_filename(spec: CaptureSpec) -> str:
    """Return a filesystem-safe, stable filename for *spec*."""
    return f"{spec.theme.lower()}-{spec.width}x{spec.height}-{spec.scene}.png"


def manifest_entry(
    spec: CaptureSpec,
    *,
    file: str,
    head: str,
    pixel_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable manifest row."""
    entry = {**asdict(spec), "file": file, "HEAD": head}
    if pixel_size is not None:
        entry["pixel_size"] = {"width": pixel_size[0], "height": pixel_size[1]}
    return entry


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _synthetic_pages() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    pages = []
    for page_index, accent in enumerate((54, 92), start=1):
        image = np.full((900, 640, 3), 242, dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (610, 870), (255, 255, 255), -1)
        cv2.rectangle(image, (30, 30), (610, 870), (accent, accent, accent), 3)
        for row in range(7):
            top = 125 + row * 88
            cv2.line(image, (90, top), (550, top), (55, 55, 55), 3)
            cv2.line(image, (90, top + 42), (430 + row * 12, top + 42), (120, 120, 120), 2)
        cv2.putText(
            image,
            f"UNISCAN UX PAGE {page_index}",
            (72, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        pages.append(image)
    return tuple(pages)


def _pump(app: Any, *, seconds: float = 0.35) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.01)
    app.update()


def _preview_capture_is_idle(app: Any) -> bool:
    """Return whether no asynchronous review preview can mutate a capture."""
    scheduled = getattr(app, "review_preview_job", None)
    threads = list(getattr(app, "review_preview_threads", ()))
    active = getattr(app, "review_preview_thread", None)
    if active is not None and active not in threads:
        threads.append(active)
    return scheduled is None and not any(thread.is_alive() for thread in threads)


def _pipeline_capture_is_consistent(*, selected_count: int, card_count: int) -> bool:
    """Require a selected page and real stage cards, not the empty placeholder."""
    return selected_count == 1 and card_count >= 7


def _disable_async_preview_generation(app: Any) -> None:
    """Disable only this app instance's preview trigger during capture.

    The production app remains unchanged.  The capture harness uses cached
    representative pixels, so launching an inference worker would add no
    evidence and would make screenshots timing-dependent.
    """
    app._cancel_review_page_preview(refresh_pipeline=False)
    app.update_page_preview = lambda: None


def _stabilize_preview(app: Any) -> None:
    """Cancel and drain preview work, then restore the deterministic cache.

    Capture scenes intentionally exercise the existing UI only.  They must not
    wait for or display a production inference result, because a late worker
    result would make the same screenshot differ between runs.
    """
    app._cancel_review_page_preview(refresh_pipeline=False)
    deadline = time.monotonic() + 5.0
    while not _preview_capture_is_idle(app) and time.monotonic() < deadline:
        app.update()
        time.sleep(0.02)
    if not _preview_capture_is_idle(app):
        raise RuntimeError("Preview worker did not stop before the UI capture.")

    app.page_preview_before_image = app._ux_capture_before_image.copy()
    app.page_preview_after_image = app._ux_capture_after_image.copy()
    app.preview_hold_original = False
    app._set_preview_result_state("Candidate — not exported", kind="candidate")
    app._layout_page_previews()
    app._refresh_pipeline_strip()
    app._render_cached_review_previews(force=True)
    app.update_idletasks()
    app.update()
    if not _preview_capture_is_idle(app):
        raise RuntimeError("Preview work was scheduled while preparing the UI capture.")
    if not _pipeline_capture_is_consistent(
        selected_count=len(app._selected_entry_indices()),
        card_count=len(app.pipeline_strip.winfo_children()),
    ):
        raise RuntimeError("Capture fixture selection and pipeline cards are inconsistent.")


def _seed_app(app: Any) -> None:
    import numpy as np

    pages = _synthetic_pages()
    entries = [app.session.add_image(name=f"ux-page-{index}.png", image=page) for index, page in enumerate(pages, 1)]
    app.refresh_page_list(keep_index=0, update_preview=False)
    app.preview_mode_var.set("Preview")
    app.preview_view_entry_id = entries[0].entry_id
    before = pages[0]
    after = np.clip(before.astype(np.int16) + np.array([4, 0, -4]), 0, 255).astype(np.uint8)
    app.page_preview_before_image = before
    app.page_preview_after_image = after
    app._ux_capture_before_image = before.copy()
    app._ux_capture_after_image = after.copy()
    app.page_preview_after_title.configure(text="Preview")
    app._set_preview_result_state("Candidate — not exported", kind="candidate")
    app._layout_page_previews()
    app._refresh_pipeline_strip()
    app._render_cached_review_previews(force=True)
    app.update_idletasks()
    if not _pipeline_capture_is_consistent(
        selected_count=len(app._selected_entry_indices()),
        card_count=len(app.pipeline_strip.winfo_children()),
    ):
        raise RuntimeError("Capture fixture selection and pipeline cards are inconsistent.")
    app.update()


def _close_advanced(app: Any) -> None:
    button = getattr(app, "review_processing_close_button", None)
    if button is not None and button.winfo_exists():
        button.invoke()
    else:
        app._hide_inline_geometry_editor()
    app.update()


def _prepare_scene(app: Any, scene: str) -> None:
    _close_advanced(app)
    app.tabs.set(app.tab_review_name)
    if scene == "advanced":
        app.open_review_processing_dialog()
    elif scene == "keyboard-focus":
        # The preview toolbar is visible in both supported window sizes and
        # preview_fit_button is bound to the shared focus-ring style.
        focus_target = getattr(app, FOCUS_SCENE_WIDGET)
        focus_target.focus_set()
        app.update_idletasks()
        if not focus_target.winfo_ismapped() or focus_target.winfo_width() <= 0:
            raise RuntimeError("Keyboard-focus target is not visible in the capture scene.")
    else:
        app.page_listbox.focus_set()
    app.update()


def _capture_window(app: Any, *, output: Path, spec: CaptureSpec) -> tuple[str, tuple[int, int]]:
    from PIL import ImageGrab

    app.geometry(f"{spec.width}x{spec.height}+40+40")
    app.update_idletasks()
    app.update()
    _prepare_scene(app, spec.scene)
    _pump(app, seconds=0.05)
    _stabilize_preview(app)
    left, top = app.winfo_rootx(), app.winfo_rooty()
    width, height = app.winfo_width(), app.winfo_height()
    try:
        image = ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not capture the UniScan window. Keep a Windows desktop visible "
            "and ensure the window is not minimized."
        ) from exc
    filename = capture_filename(spec)
    image.save(output / filename)
    return filename, image.size


def run_capture(output: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Launch UniScan in isolated state and capture the complete matrix."""
    if os.name != "nt":
        raise RuntimeError(
            "UI regression capture requires a Windows desktop; it is intentionally "
            "not a headless or CI operation."
        )
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    head = _git_head(root)
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    import customtkinter as ctk
    from uniscan.ui.app import UnifiedScanApp

    with tempfile.TemporaryDirectory(prefix="uniscan-ui-regression-") as state_dir:
        previous_state = os.environ.get("UNISCAN_STATE_DIR")
        os.environ["UNISCAN_STATE_DIR"] = state_dir
        app = None
        try:
            app = UnifiedScanApp()
            _disable_async_preview_generation(app)
            _seed_app(app)
            app.deiconify()
            app.lift()
            app.focus_force()
            app.update()
            rows: list[dict[str, Any]] = []
            for spec in capture_matrix():
                ctk.set_appearance_mode(spec.theme)
                filename, pixel_size = _capture_window(app, output=output, spec=spec)
                rows.append(
                    manifest_entry(
                        spec,
                        file=filename,
                        head=head,
                        pixel_size=pixel_size,
                    )
                )
        finally:
            if app is not None:
                app._on_close()
                deadline = time.monotonic() + 6.0
                while time.monotonic() < deadline:
                    try:
                        exists = bool(app.winfo_exists())
                    except Exception:
                        exists = False
                    if not exists:
                        break
                    app.update()
                    time.sleep(0.02)
                try:
                    still_exists = bool(app.winfo_exists())
                except Exception:
                    still_exists = False
                if still_exists:
                    raise RuntimeError("UniScan did not finish closing its isolated session.")
            if previous_state is None:
                os.environ.pop("UNISCAN_STATE_DIR", None)
            else:
                os.environ["UNISCAN_STATE_DIR"] = previous_state

    manifest = {
        "tool": "scripts/capture_ui_regression.py",
        "HEAD": head,
        "matrix": rows,
        "notes": [
            "Opt-in manual review captures; no golden pixel thresholds are used in CI.",
            "The app session was isolated in a temporary UNISCAN_STATE_DIR.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for PNG captures and manifest.json; never defaults to the repo.",
    )
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error(
            "Windows desktop required: this opt-in tool uses Tk and Pillow ImageGrab "
            "and does not run in headless CI."
        )
    try:
        manifest = run_capture(args.output_dir)
    except Exception as exc:
        print(f"UI regression capture failed: {exc}", file=sys.stderr)
        return 2
    print(f"Captured {len(manifest['matrix'])} scenes to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
