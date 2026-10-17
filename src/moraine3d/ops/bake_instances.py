"""``bake_instances`` operator.

Semantics: explicitly merges its single Instances input into one Mesh —
the prototype baked at every instance transform (DESIGN 4.1, 5).
Vertices map through each instance matrix, normals through its
inverse-transpose (renormalized); faces are the
prototype's, index-offset per copy; UVs, extra attribute slots, and
face groups are tiled per copy (preserved — a group on the baked mesh
selects that region on every copy), except TANGENT directions, which
map through each instance's linear part like the geometry they frame. No params. Instance transforms must have
positive determinant (guaranteed by ``array``/``scatter``/``transform``
construction) so the shared winding stays CCW-outward; a
non-positive-determinant transform is an actionable error. A Mesh input
is an error — merging is explicit, and a mesh needs no baking (Day-One
Invariant 6: nothing merges silently, this op is the explicit opt-in).
Intended use: turning repeated cutters (bolt circles, window openings)
into a single mesh that ``boolean`` accepts. Deterministic. Example:
``examples/bolt_plate``.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Instances, Mesh
from ..errors import OpError

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS: dict = {}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # prototype slots are tiled per copy


def _bake_tangents(tan: np.ndarray, xf: np.ndarray) -> np.ndarray:
    # TANGENT is a surface direction: per copy it maps through the
    # instance's linear part (not the normals' inverse-transpose),
    # renormalized. Plain tiling would keep the prototype-space frame and
    # skew every normal map on rotated copies. Determinants are positive
    # here (checked in evaluate), so the handedness sign w tiles as-is.
    d = np.einsum("kij,nj->kni", xf[:, :3, :3], tan[:, :3].astype(np.float64))
    d = d.reshape(-1, 3)
    lens = np.linalg.norm(d, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    w = np.tile(tan[:, 3:4], (len(xf), 1))
    return np.concatenate([d / lens, w], axis=1).astype(np.float32)


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = inputs[0]
    if not isinstance(src, Instances):
        raise OpError(
            "bad_input",
            "bake_instances input is already a mesh; it merges Instances "
            "(array/scatter output) into one Mesh - feed it the instancing "
            "node, or remove it",
        )
    proto = src.prototype
    xf = np.asarray(src.transforms, dtype=np.float64)  # (k, 4, 4)
    det = np.linalg.det(xf[:, :3, :3])
    if np.any(det <= 0):
        bad = int(np.argmax(det <= 0))
        raise OpError(
            "bad_input",
            f"instance transform {bad} has non-positive determinant "
            f"({det[bad]:.6g}); the shared prototype's winding cannot flip "
            f"per instance - mirror the prototype mesh before the "
            f"array/scatter node instead",
        )

    k, n = len(xf), len(proto.vertices)
    hom = np.concatenate(
        [proto.vertices.astype(np.float64), np.ones((n, 1))], axis=1
    )  # (n, 4)
    verts = np.einsum("kij,nj->kni", xf, hom)[..., :3].reshape(k * n, 3)

    normals = None
    if proto.normals is not None:
        inv_t = np.linalg.inv(xf[:, :3, :3]).transpose(0, 2, 1)  # (k, 3, 3)
        nrm = np.einsum("kij,nj->kni", inv_t, proto.normals.astype(np.float64))
        nrm = nrm.reshape(k * n, 3)
        lens = np.linalg.norm(nrm, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        normals = (nrm / lens).astype(np.float32)

    faces = (
        proto.faces.astype(np.int64)[None] + (np.arange(k) * n)[:, None, None]
    ).reshape(-1, 3)
    m = len(proto.faces)
    return Mesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals,
        uv=None if proto.uv is None else np.tile(proto.uv, (k, 1)),
        attributes=None if not proto.attributes
        else {
            name: _bake_tangents(a, xf) if name == "TANGENT"
            else np.tile(a, (k,) + (1,) * (a.ndim - 1))
            for name, a in proto.attributes.items()
        },
        face_groups=None if not proto.face_groups
        else {
            name: (g.astype(np.int64)[None] + (np.arange(k) * m)[:, None])
            .reshape(-1).astype(np.int32)
            for name, g in proto.face_groups.items()
        },
    )
