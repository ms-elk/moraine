"""Welded manifold3d solids of evaluated values (verify kernel): the
prototype-solid build the enclosure check, the overlap guard and
``embedded`` share, the world-space union for an output as placed, and
the one sentence for a surface that cannot be built as a solid.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..core.mesh import Image, Instances, Value, proto_of


class NoSolid:
    """The kernel could not build a solid from a surface; ``status`` is the
    manifold3d error name (``NotManifold`` for an open or inconsistently
    wound surface, ``NonFiniteVertex`` for NaN/inf positions), so the
    reason handed to the author names the actual defect."""

    __slots__ = ("status",)

    def __init__(self, status: str):
        self.status = status


def to_manifold(mesh_verts: np.ndarray, mesh_faces: np.ndarray):
    """Welded manifold3d solid, or a ``NoSolid`` carrying the kernel
    status when the surface cannot be built as one."""
    import manifold3d

    tm = trimesh.Trimesh(
        vertices=mesh_verts.astype(np.float64),
        faces=mesh_faces.astype(np.int64),
        process=False,
    )
    tm.merge_vertices()
    m = manifold3d.Manifold(
        mesh=manifold3d.Mesh(
            vert_properties=tm.vertices.astype(np.float32),
            tri_verts=tm.faces.astype(np.uint32),
        )
    )
    status = m.status()
    return m if status == manifold3d.Error.NoError else NoSolid(status.name)


def manifold_measures(m) -> tuple[float, float]:
    """(volume, surface area) of a manifold3d solid.

    Kernel-native accessors: the previous numpy round-trip materialized
    the whole mesh per call. Agreement with that computation is ~1e-9
    relative — orders inside the thresholds that consume these numbers
    (_ENCLOSE_VOL_RTOL, DEPTH_BAR).
    """
    return abs(float(m.volume())), float(m.surface_area())


def manifold_volume(m) -> float:
    return manifold_measures(m)[0]


_UNBUILT = object()  # "no prebuilt solid passed" (a NoSolid means unbuildable)

# The fix per manifold3d failure status: the check that catches the defect
# earlier in the report, or the property to inspect when none does.
_NO_SOLID_FIX = {
    "NotManifold": (
        "fix the watertight check first, or the face winding if that check "
        "passes (the surface must be closed and consistently wound)"
    ),
    "NonFiniteVertex": "fix the finite check first",
}


def no_solid_why(name: str, status: str) -> str:
    # One home for the sentence: the cross-pair and self-pair paths must
    # report the same defect identically (and dedupe against each other).
    fix = _NO_SOLID_FIX.get(status, f"inspect the mesh for the {status} defect")
    return f"output '{name}' cannot be built as a solid ({status}); {fix}"


def world_manifold(value: Value, name: str, proto_solid=_UNBUILT):
    """World-space manifold3d solid for an output, or (None, why-not).

    Instances become the union of the transformed prototype solid — the
    assertion is about the output as placed, all copies included.
    ``proto_solid`` is the output's prototype solid when the caller holds
    a shared memo (manifold-or-NoSolid, to_manifold's contract).
    """
    if isinstance(value, Image):
        return None, f"output '{name}' is an image; assertions apply to geometry"
    if proto_solid is _UNBUILT:
        mesh = proto_of(value)
        m = to_manifold(mesh.vertices, mesh.faces)
    else:
        m = proto_solid
    if isinstance(m, NoSolid):
        return None, no_solid_why(name, m.status)
    if isinstance(value, Instances):
        import manifold3d

        solids = [m.transform(np.ascontiguousarray(t[:3, :4])) for t in value.transforms]
        if len(solids) == 1:
            return solids[0], None
        # One balanced batch op: the sequential fold re-processes the
        # growing accumulated solid per copy (O(k^2) triangle work).
        return manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add), None
    return m, None
