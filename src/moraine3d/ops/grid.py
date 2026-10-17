"""``grid`` operator.

Semantics: emits a flat rectangle in the XZ plane at y = 0, centered at
the origin, facing +Y. ``size`` = [sx, sz] meters (default [1, 1]);
``segments`` = [nx, nz] quad divisions (default [1, 1], each 1..4096).
Vertices are shared, normals all +Y, winding CCW seen from +Y. The open
boundary
makes it the canonical base for ``extrude`` and a scatter target.
Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Mesh
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "size": numbers.FLOAT,
    "segments": numbers.INT,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceiling on divisions per axis: beyond it the count is a typo,
# and an unbounded one surfaces as a raw allocation failure, not a refusal.
SEGMENTS_MAX = 4096
ATTRIBUTE_POLICY = "preserved"  # source op


def _pair(params: dict, name: str, default: list, *, integral: bool) -> tuple:
    v = params.get(name, default)
    if not common.num_pair(v, integral=integral) or (
        integral and any(c < 1 or c > SEGMENTS_MAX for c in v)
    ) or (not integral and not all(c > 0 for c in v)):
        kind = (
            f"2 integers in 1..{SEGMENTS_MAX}" if integral else "2 positive finite numbers"
        )
        raise OpError("bad_param", f"grid '{name}' must be {kind}", param=name)
    return tuple(v)


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    sx, sz = _pair(params, "size", [1.0, 1.0], integral=False)
    nx, nz = _pair(params, "segments", [1, 1], integral=True)

    xs = np.linspace(-sx / 2.0, sx / 2.0, nx + 1)
    zs = np.linspace(-sz / 2.0, sz / 2.0, nz + 1)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")  # row-major over x then z
    verts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)

    ix, iz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    a = (ix * (nz + 1) + iz).ravel()  # (x, z) corner
    b = a + (nz + 1)  # +x
    c = b + 1  # +x +z
    d = a + 1  # +z
    # CCW seen from +Y (+Y up, right-handed): x-then-z corner order a,d,c,b.
    faces = np.concatenate([np.stack([a, d, c], axis=1), np.stack([a, c, b], axis=1)])

    return Mesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=np.tile([0.0, 1.0, 0.0], (len(verts), 1)).astype(np.float32),
    )
