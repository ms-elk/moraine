"""``colorize`` operator.

Semantics: maps a 1-channel image through a two-color linear gradient:
``out = color_low + clamp(p, 0, 1) * (color_high - color_low)`` per pixel.
Output: a 3-channel Image the size of the input. Params: ``color_low``
(linear RGB at input 0, default [0, 0, 0]); ``color_high`` (at input 1,
default [1, 1, 1]). Input values outside [0, 1] clamp. Deterministic
and pure. Typical use: turning noise into a concrete / grime color map.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "color_low": numbers.FLOAT,
    "color_high": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "colorize"
    src = common.image_input(op, inputs, 0, channels=1)
    lo = common.rgb(op, params, "color_low", [0.0, 0.0, 0.0])
    hi = common.rgb(op, params, "color_high", [1.0, 1.0, 1.0])
    p = np.clip(src.pixels[..., 0:1].astype(np.float64), 0.0, 1.0)
    out = lo + p * (hi - lo)
    return Image(pixels=out.astype(np.float32))
