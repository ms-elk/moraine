"""The fixed names of the channel between the entry points and a
running ``serve`` (DESIGN 6, 7.2).

A parser needs these before any command runs, so this module imports
nothing of the package: an entry point reads a default without loading
the server or a kernel.
"""

from __future__ import annotations

import math

DEFAULT_PORT = 8642
# The request header a CLI channel names its root in (percent-encoded),
# so a server of another root refuses the request.
ROOT_HEADER = "X-Moraine-Root"

# Named views are presets over the free camera — the single source of
# the view roster (CLI choices and server validation both read it).
# iso's elevation is exactly the historical (1, 0.8, 1) direction.
VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "iso": (45.0, math.degrees(math.atan(0.8 / math.sqrt(2.0)))),
}
