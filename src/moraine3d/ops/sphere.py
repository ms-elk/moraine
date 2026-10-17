"""``sphere`` operator.

Semantics: emits a UV sphere centered at the origin with ``radius`` (m,
default 0.5), ``segments`` longitude divisions (default 32, 3..4096), and
``rings`` latitude divisions (default 16, 2..4096). Smooth-shaded: shared
vertices with normals equal to the normalized position. Poles are single
vertices at +-Y closed with triangle fans; interior latitudes are quad
bands. Winding is CCW from outside. Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Mesh
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "radius": numbers.FLOAT,
    "segments": numbers.INT,
    "rings": numbers.INT,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceiling on tessellation per axis (cylinder's): beyond it the
# count is a typo, and an unbounded one surfaces as a raw allocation failure.
SEGMENTS_MAX = 4096
ATTRIBUTE_POLICY = "preserved"  # source op


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    radius = common.num("sphere", params, "radius", 0.5, positive=True)
    seg = common.integer("sphere", params, "segments", 32, minimum=3, maximum=SEGMENTS_MAX)
    rings = common.integer("sphere", params, "rings", 16, minimum=2, maximum=SEGMENTS_MAX)

    # Interior latitude rings: rings-1 of them, seg vertices each.
    lat = np.arange(1, rings) * (np.pi / rings)  # from +Y pole
    lon = np.arange(seg) * (2.0 * np.pi / seg)
    sl, cl = np.sin(lat)[:, None], np.cos(lat)[:, None]
    dirs = np.stack(
        [
            (sl * np.cos(lon)[None, :]).ravel(),
            np.broadcast_to(cl, (rings - 1, seg)).ravel(),
            (sl * -np.sin(lon)[None, :]).ravel(),
        ],
        axis=1,
    )
    n_ring = (rings - 1) * seg
    verts = np.concatenate([dirs * radius, [[0.0, radius, 0.0]], [[0.0, -radius, 0.0]]])
    normals = np.concatenate([dirs, [[0.0, 1.0, 0.0]], [[0.0, -1.0, 0.0]]])
    top, bot = n_ring, n_ring + 1

    i = np.arange(seg)
    j = (i + 1) % seg
    faces = [np.stack([np.full(seg, top), i, j], axis=1)]  # top fan
    for r in range(rings - 2):  # quad bands between ring r and r+1
        a, b = r * seg, (r + 1) * seg
        faces.append(np.stack([a + i, b + i, b + j], axis=1))
        faces.append(np.stack([a + i, b + j, a + j], axis=1))
    last = (rings - 2) * seg
    faces.append(np.stack([np.full(seg, bot), last + j, last + i], axis=1))  # bottom fan

    return Mesh(
        vertices=verts.astype(np.float32),
        faces=np.concatenate(faces).astype(np.int32),
        normals=normals.astype(np.float32),
    )
