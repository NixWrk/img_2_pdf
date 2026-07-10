"""Camera health status helpers for UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CameraHealth:
    label: str
    color: str


def camera_health_state(
    *, is_open: bool, is_previewing: bool, error_text: str | None = None
) -> CameraHealth:
    if error_text:
        return CameraHealth(label="Camera: Error", color="#d94f4f")
    if is_previewing:
        return CameraHealth(label="Camera: Previewing", color="#2f9e44")
    if is_open:
        return CameraHealth(label="Camera: Open", color="#0b7285")
    return CameraHealth(label="Camera: Closed", color="#6c757d")
