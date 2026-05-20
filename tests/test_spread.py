import numpy as np

from uniscan.core.spread import detect_spread_gutter, split_spread_accurate


def _make_synthetic_spread(
    *,
    width: int = 800,
    height: int = 500,
    gutter_x: int = 400,
    gutter_width: int = 12,
    gutter_dark: int = 30,
    background: int = 235,
    text_lines: int = 18,
) -> np.ndarray:
    """Build a synthetic two-page spread with a dark vertical gutter band."""
    image = np.full((height, width, 3), background, dtype=np.uint8)
    # Draw horizontal text-like dashes on both pages so content-balance is symmetric.
    rng = np.random.default_rng(seed=42)
    line_step = max(8, height // (text_lines + 2))
    for y in range(line_step, height - line_step, line_step):
        for side in ("left", "right"):
            if side == "left":
                start_x, end_x = 30, gutter_x - gutter_width // 2 - 20
            else:
                start_x, end_x = gutter_x + gutter_width // 2 + 20, width - 30
            x = start_x
            while x < end_x:
                seg = min(rng.integers(40, 120), end_x - x)
                image[y:y + 4, x:x + seg] = 50
                x += seg + rng.integers(15, 35)
    # Dark gutter band
    lo = max(0, gutter_x - gutter_width // 2)
    hi = min(width, gutter_x + gutter_width // 2 + 1)
    image[:, lo:hi] = gutter_dark
    return image


def _make_single_page(*, width: int = 800, height: int = 500) -> np.ndarray:
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    rng = np.random.default_rng(seed=7)
    for y in range(30, height - 30, 18):
        x = 40
        while x < width - 40:
            seg = min(rng.integers(60, 160), width - 40 - x)
            image[y:y + 4, x:x + seg] = 50
            x += seg + rng.integers(20, 40)
    return image


def test_detect_gutter_centered_spread() -> None:
    image = _make_synthetic_spread(gutter_x=400)
    candidate = detect_spread_gutter(image)
    assert candidate is not None
    assert abs(candidate.x - 400) < 12
    assert candidate.confidence > 0.5


def test_detect_gutter_offset_spread() -> None:
    image = _make_synthetic_spread(width=800, gutter_x=320)
    candidate = detect_spread_gutter(image)
    assert candidate is not None
    assert abs(candidate.x - 320) < 20


def test_no_gutter_single_page_returns_none() -> None:
    image = _make_single_page()
    candidate = detect_spread_gutter(image)
    assert candidate is None


def test_aspect_gate_rejects_tall_documents() -> None:
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    candidate = detect_spread_gutter(image, min_aspect=1.3)
    assert candidate is None


def test_split_accurate_uses_detected_gutter() -> None:
    image = _make_synthetic_spread(width=800, gutter_x=420)
    halves = split_spread_accurate(image)
    assert len(halves) == 2
    # Left width should be close to gutter_x.
    left, _right = halves
    assert abs(left.shape[1] - 420) < 20


def test_split_accurate_falls_back_to_midpoint_when_no_gutter() -> None:
    image = _make_single_page(width=800)
    halves = split_spread_accurate(image, fallback="midpoint")
    assert len(halves) == 2
    assert halves[0].shape[1] == 800 // 2


def test_split_accurate_no_fallback_returns_single_when_no_gutter() -> None:
    image = _make_single_page(width=800)
    halves = split_spread_accurate(image, fallback="none")
    assert len(halves) == 1
    assert halves[0].shape[1] == 800
