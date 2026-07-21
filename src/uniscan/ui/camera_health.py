"""Camera health status helpers for UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CameraHealth:
    label: str
    color: str


def camera_health_state(
    *,
    is_open: bool,
    is_previewing: bool,
    is_opening: bool = False,
    error_text: str | None = None,
    detail: str | None = None,
) -> CameraHealth:
    """Map camera state to a status label; ``detail`` (e.g. "1920x1080 @ 28 fps")
    is appended to the open/previewing labels."""
    if error_text:
        return CameraHealth(label="Camera: Error", color="#d94f4f")
    if is_opening:
        return CameraHealth(label="Camera: Opening...", color="#b8860b")
    suffix = f" ({detail})" if detail else ""
    if is_previewing:
        return CameraHealth(label=f"Camera: Previewing{suffix}", color="#2f9e44")
    if is_open:
        return CameraHealth(label=f"Camera: Open{suffix}", color="#0b7285")
    return CameraHealth(label="Camera: Closed", color="#6c757d")
