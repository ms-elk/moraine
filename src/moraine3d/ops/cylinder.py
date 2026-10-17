"""``cylinder`` operator.

Semantics: emits a Y-axis cylinder centered at the origin (pivot at center
of volume) with ``radius`` (m, default 0.5), ``height`` (m, default 1),
and ``segments`` around the circumference (default 32, 3..4096). The side is
smooth-shaded (shared ring vertices, radial normals); the caps are hard
(duplicated ring vertices, +-Y normals) and closed with a center-vertex
fan. Vertex 0 of each ring sits at angle 0 (+X); winding is CCW from
outside. Deterministic.
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
    "height": numbers.FLOAT,
    "segments": numbers.INT,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceiling on tessellation: beyond it a primitive is a typo, and
# an unbounded count surfaces as a raw allocation failure, not a refusal.
SEGMENTS_MAX = 4096
ATTRIBUTE_POLICY = "preserved"  # source op


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    radius = common.num("cylinder", params, "radius", 0.5, positive=True)
    height = common.num("cylinder", params, "height", 1.0, positive=True)
    seg = common.integer("cylinder", params, "segments", 32, minimum=3, maximum=SEGMENTS_MAX)

    ang = np.arange(seg) * (2.0 * np.pi / seg)
    ring = np.stack([np.cos(ang), np.zeros(seg), -np.sin(ang)], axis=1)  # CCW seen from +Y
    hy = height / 2.0

    # Side: bottom ring [0:seg], top ring [seg:2seg], smooth radial normals.
    side_v = np.concatenate([ring * radius + [0, -hy, 0], ring * radius + [0, hy, 0]])
    side_n = np.concatenate([ring, ring])
    i = np.arange(seg)
    j = (i + 1) % seg
    side_f = np.concatenate(
        [np.stack([i, j, seg + j], axis=1), np.stack([i, seg + j, seg + i], axis=1)]
    )

    # Caps: duplicated rings + center vertex, hard +-Y normals.
    base = 2 * seg
    top_v = np.concatenate([ring * radius + [0, hy, 0], [[0.0, hy, 0.0]]])
    bot_v = np.concatenate([ring * radius + [0, -hy, 0], [[0.0, -hy, 0.0]]])
    top_f = np.stack([base + i, base + j, np.full(seg, base + seg)], axis=1)
    b2 = base + seg + 1
    bot_f = np.stack([b2 + j, b2 + i, np.full(seg, b2 + seg)], axis=1)

    verts = np.concatenate([side_v, top_v, bot_v])
    normals = np.concatenate(
        [side_n, np.tile([0.0, 1.0, 0.0], (seg + 1, 1)), np.tile([0.0, -1.0, 0.0], (seg + 1, 1))]
    )
    faces = np.concatenate([side_f, top_f, bot_f])
    return Mesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
    )
