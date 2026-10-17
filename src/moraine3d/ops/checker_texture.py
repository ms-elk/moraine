"""``checker_texture`` operator.

Semantics: checkerboard color texture. Output: a 3-channel Image;
``tiles`` x ``tiles`` squares alternate between ``color_a`` (the square
containing u = v = 0) and ``color_b``. Params: ``size`` — int or [w, h]
pixels, 4..2048 (default 256); ``tiles`` (int, default 8, 1..2048 —
beyond one tile per pixel nothing is resolved);
``color_a`` (linear RGB, default [1, 1, 1]); ``color_b`` (default
[0, 0, 0]). An even ``tiles`` repeats seamlessly; an odd count shows its
seam when tiled (fine on a 0..1-UV-mapped surface). Deterministic and
pure. Primary use: verifying UV layout and texel density before real
materials.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "size": numbers.INT,
    "tiles": numbers.INT,
    "color_a": numbers.FLOAT,
    "color_b": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "checker_texture"
    w, h = common.texture_size(op, params)
    tiles = common.integer(op, params, "tiles", 8, minimum=1, maximum=2048)
    ca = common.rgb(op, params, "color_a", [1.0, 1.0, 1.0])
    cb = common.rgb(op, params, "color_b", [0.0, 0.0, 0.0])

    iu = np.floor((np.arange(w) + 0.5) / w * tiles).astype(np.int64)[None, :]
    iv = np.floor((np.arange(h) + 0.5) / h * tiles).astype(np.int64)[:, None]
    parity = ((iu + iv) % 2).astype(np.float64)[..., None]
    out = ca * (1.0 - parity) + cb * parity
    return Image(pixels=out.astype(np.float32))
