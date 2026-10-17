"""``unwrap`` operator.

Semantics: assigns UVs to its single mesh input. Params (the full future
signature): ``mode`` — ``"box"`` (default) or ``"planar"``;
``uv_scale`` UV units per meter (default 1); ``axis`` — planar mode's
projection axis, ``"x" | "y" | "z"`` (default "y"). Box mode projects
each face along the dominant axis of its normal onto the two other axes,
signed so the texture reads upright and un-mirrored from outside the
surface under glTF's v-down texture origin (DESIGN 4.3): a side face maps
world +Y to smaller v, so a map with an intrinsic up — weathering running
down a wall — keeps it. Vertices shared between differently-projected
faces are split, others stay shared. Planar mode projects every vertex
along one axis; no splits.
Projection UVs overlap by construction — good enough for tiling
materials. What ``box`` and ``planar`` return is fixed; a packed
non-overlapping atlas (xatlas) arrives as a further ``mode``. Position, topology,
and normals are otherwise unchanged. Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Mesh, tri_cross
from . import common

OP_IMPL_VERSION = 2
KERNELS = ("core.mesh",)
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "mode": numbers.OTHER,
    "uv_scale": numbers.FLOAT,
    "axis": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # authors TEXCOORD_0; other slots ride along
PLACEMENT = "identity"  # seams split vertices; positions untouched

# Per (dominant axis, sign): (u axis, u sign, v axis, v sign), chosen so the
# projection reads upright and un-mirrored viewed from outside under glTF's
# v-down texture origin (DESIGN 4.3): +u is view-right, -v is view-up.
_PROJ = {
    (0, 1): (2, -1.0, 1, -1.0),  # +X: u = -z, v = -y
    (0, -1): (2, 1.0, 1, -1.0),  # -X: u = +z, v = -y
    (1, 1): (0, 1.0, 2, 1.0),  # +Y: u = +x, v = +z
    (1, -1): (0, 1.0, 2, -1.0),  # -Y: u = +x, v = -z
    (2, 1): (0, 1.0, 1, -1.0),  # +Z: u = +x, v = -y
    (2, -1): (0, -1.0, 1, -1.0),  # -Z: u = -x, v = -y
}


def _planar_uv(verts: np.ndarray, axis: int, scale: float) -> np.ndarray:
    ua, us, va, vs = _PROJ[(axis, 1)]
    return np.stack([verts[:, ua] * us, verts[:, va] * vs], axis=1) * scale


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = common.mesh_input("unwrap", inputs, 0)
    mode = common.choice("unwrap", params, "mode", "box", ("box", "planar"))
    scale = common.num("unwrap", params, "uv_scale", 1.0, positive=True)
    axis = "xyz".index(common.choice("unwrap", params, "axis", "y", ("x", "y", "z")))

    v = src.vertices.astype(np.float64)
    if mode == "planar":
        return Mesh(
            vertices=src.vertices,
            faces=src.faces,
            normals=src.normals,
            uv=_planar_uv(v, axis, scale).astype(np.float32),
            attributes=src.attributes,
        )

    # Box mode: classify faces by dominant normal axis + sign.
    f = src.faces.astype(np.int64)
    tri = v[f]
    fn = tri_cross(tri)
    dom = np.abs(fn).argmax(axis=1)
    sign = np.where(fn[np.arange(len(fn)), dom] >= 0, 1, -1)

    # Per-corner UV, then weld corners identical in (vertex index, projection)
    # so smooth regions stay shared and only projection seams split.
    corner_vi = f.ravel()
    corner_proj = np.repeat(dom * 2 + (sign < 0), 3)
    corner_uv = np.empty((len(corner_vi), 2))
    for (ax, sg), (ua, us, va, vs) in _PROJ.items():
        mask = corner_proj == (ax * 2 + (sg < 0))
        if mask.any():
            pv = v[corner_vi[mask]]
            corner_uv[mask, 0] = pv[:, ua] * us * scale
            corner_uv[mask, 1] = pv[:, va] * vs * scale

    key = np.column_stack([corner_vi, corner_proj])
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    new_vi = uniq[:, 0]
    uv = np.empty((len(uniq), 2))
    uv[inverse] = corner_uv  # identical keys carry identical uvs
    return Mesh(
        vertices=src.vertices[new_vi],
        faces=inverse.reshape(-1, 3).astype(np.int32),
        normals=None if src.normals is None else src.normals[new_vi],
        uv=uv.astype(np.float32),
        attributes=None if not src.attributes
        else {name: a[new_vi] for name, a in src.attributes.items()},
    )
