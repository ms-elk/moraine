"""``boolean`` operator.

Semantics: CSG on two or more watertight mesh inputs via manifold3d.
Param ``mode``: ``"union"`` (default), ``"difference"`` (input 0 minus
the union of the rest), or ``"intersection"``. Inputs are welded
(duplicate vertices merged) before the operation; a non-watertight input
is an actionable error, not a silent repair. Output is flat-shaded
(per-face duplicated vertices and normals) — CSG results are hard-surface
by nature. Input UVs are dropped (unwrap downstream). Deterministic.
"""

from __future__ import annotations

import manifold3d
import numpy as np

from ..core import numbers
from ..core.mesh import Mesh, flat_shaded
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ("core.mesh", "manifold3d", "trimesh")
SHADING_POLICY = "replaced"  # output is flat-shaded regardless of input
# The flat_shading warning names this op's flat output as its source
# (DESIGN 4.1). bevel also replaces shading but deliberately does not
# declare this — warning on its own flat output is a recorded follow-up
# candidate, not silently included.
FLAT_SHADING_SOURCE = True
INPUT_ARITY = (2, None)
PARAM_KINDS = {
    "mode": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "dropped"  # CSG rebuilds topology


def _to_manifold(mesh: Mesh, index: int) -> manifold3d.Manifold:
    import trimesh

    tm = trimesh.Trimesh(
        vertices=mesh.vertices.astype(np.float64), faces=mesh.faces.astype(np.int64), process=False
    )
    tm.merge_vertices()
    m = manifold3d.Manifold(
        mesh=manifold3d.Mesh(
            vert_properties=tm.vertices.astype(np.float32),
            tri_verts=tm.faces.astype(np.uint32),
        )
    )
    if m.status() != manifold3d.Error.NoError:
        raise OpError(
            "not_watertight",
            f"boolean input {index} is not a watertight manifold "
            f"({m.status().name}); close the surface (e.g. extrude an open "
            f"grid) before boolean",
            input=index,
        )
    return m


def from_manifold(m: manifold3d.Manifold) -> Mesh:
    out = m.to_mesh()
    verts = np.asarray(out.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    return flat_shaded(verts, faces)


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    mode = common.choice("boolean", params, "mode", "union", ("union", "difference", "intersection"))
    ms = [
        _to_manifold(common.mesh_input("boolean", inputs, i, hint=common.BAKE_HINT), i)
        for i in range(len(inputs))
    ]
    if mode == "union":
        acc = ms[0]
        for m in ms[1:]:
            acc = acc + m
    elif mode == "intersection":
        acc = ms[0]
        for m in ms[1:]:
            acc = acc ^ m
    else:  # difference
        rest = ms[1]
        for m in ms[2:]:
            rest = rest + m
        acc = ms[0] - rest
    if acc.is_empty():
        raise OpError(
            "empty_result",
            f"boolean {mode} produced an empty solid; the inputs likely do not overlap "
            f"as intended - check bounds with stats",
        )
    return from_manifold(acc)
