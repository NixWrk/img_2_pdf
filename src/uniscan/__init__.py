"""Unified scanner package."""

import os as _os
import sys as _sys

# Must run before anything imports cv2: OpenCV reads this parameter once, when
# the videoio module initialises, so setting it later has no effect. Without
# it the Media Foundation backend spends ~28 s negotiating hardware transforms
# on every camera open; with it the same device opens in ~0.25 s and streams at
# full frame rate. Respect an explicit user override.
if _sys.platform == "win32":
    _os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

__all__ = ["__version__"]
__version__ = "0.1.0"
