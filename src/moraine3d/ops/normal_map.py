"""``normal_map`` operator.

Semantics: converts a 1-channel height field into a tangent-space
normal map. Output: a 3-channel Image, unit normals encoded as
``n * 0.5 + 0.5``, +Y-up ("OpenGL") convention as required by glTF
(DESIGN 4.3): the green channel brightens on the *upper* (toward
smaller v) flank of a bump. Gradients are central differences with
wrap-around, so a tileable height map yields a tileable normal map.
``strength`` is the height amplitude in UV-tile units: the slope fed to
the normal is ``strength * dh`` per tile — larger reads deeper. Params:
``strength`` (default 1.0, >= 0). Deterministic and pure.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "strength": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "normal_map"
    src = common.image_input(op, inputs, 0, channels=1)
    strength = common.num(op, params, "strength", 1.0)
    if strength < 0:
        raise OpError(
            "bad_param",
            "normal_map 'strength' must be >= 0 (invert the height map "
            "upstream instead of negating strength)",
            param="strength",
        )

    hmap = src.pixels[..., 0].astype(np.float64)
    h, w = hmap.shape
    # Central differences, wrapping (tileable); per-UV-unit gradients.
    g_u = (np.roll(hmap, -1, axis=1) - np.roll(hmap, 1, axis=1)) * (w / 2.0)
    g_v = (np.roll(hmap, -1, axis=0) - np.roll(hmap, 1, axis=0)) * (h / 2.0)

    # Tangent frame: x along +u, y along image-up (-v) — the OpenGL/glTF
    # +Y convention. Height rises toward larger v => the normal tilts
    # toward image-up => positive green. See the derivation in the test.
    nx = -strength * g_u
    ny = strength * g_v
    nz = np.ones_like(nx)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx * inv_len, ny * inv_len, nz * inv_len], axis=-1)
    return Image(pixels=(n * 0.5 + 0.5).astype(np.float32))
