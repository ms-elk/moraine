"""``transform`` operator.

Semantics: applies a TRS transform to its single input. Params:
``translate`` [x, y, z] meters (default [0, 0, 0]); ``rotate_deg``
[rx, ry, rz] degrees applied extrinsically in X, Y, Z order, i.e.
``M = T @ Rz @ Ry @ Rx @ S`` (default [0, 0, 0]); ``scale`` — a scalar
or [sx, sy, sz], non-zero (default 1). Mesh input: vertices map through
M, normals through the inverse-transpose (renormalized), TANGENT
directions through the linear part (renormalized; a mirroring transform
flips the handedness sign w); a
negative-determinant scale flips winding to keep faces CCW-outward.
Instances input: M pre-multiplies every instance transform (the whole
instance set moves rigidly; prototypes are untouched — instancing-aware,
Day-One Invariant 6); a negative-determinant M is rejected there, since
the shared prototype's winding cannot flip per instance — mirror the
prototype mesh before the array/scatter node instead. Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Instances, Mesh
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "translate": numbers.FLOAT,
    "rotate_deg": numbers.FLOAT,
    "scale": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"
PLACEMENT = "affine"  # `matrix(params)` below


def matrix(params: dict, op: str = "transform") -> np.ndarray:
    t = common.vec3(op, params, "translate", [0.0, 0.0, 0.0])
    r = np.deg2rad(common.vec3(op, params, "rotate_deg", [0.0, 0.0, 0.0]))
    s = common.vec3(op, params, "scale", [1.0, 1.0, 1.0], scalar_ok=True)
    if np.any(s == 0):
        raise OpError("bad_param", f"{op} 'scale' components must be non-zero", param="scale")

    cx, cy, cz = np.cos(r)
    sx, sy, sz = np.sin(r)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx @ np.diag(s)
    m[:3, 3] = t
    return m


def apply_to_mesh(mesh: Mesh, m: np.ndarray) -> Mesh:
    m3 = m[:3, :3]
    det = np.linalg.det(m3)
    verts = mesh.vertices.astype(np.float64) @ m3.T + m[:3, 3]
    normals = None
    if mesh.normals is not None:
        n = mesh.normals.astype(np.float64) @ np.linalg.inv(m3)  # (inv^T)^T applied row-wise
        lens = np.linalg.norm(n, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        normals = (n / lens).astype(np.float32)
    attributes = mesh.attributes
    if attributes and "TANGENT" in attributes:
        # TANGENT is a surface direction: it maps through the linear part
        # (not the normals' inverse-transpose), renormalized; a mirroring
        # transform flips the bitangent handedness sign w. Left verbatim,
        # a rotated mesh would keep its old-world tangent frame and skew
        # every normal map sampled through it.
        tan = attributes["TANGENT"].astype(np.float64)
        d = tan[:, :3] @ m3.T
        lens = np.linalg.norm(d, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        w = tan[:, 3:4] * (-1.0 if det < 0 else 1.0)
        attributes = {
            **attributes,
            "TANGENT": np.concatenate([d / lens, w], axis=1).astype(np.float32),
        }
    faces = mesh.faces
    if det < 0:
        faces = faces[:, ::-1].copy()
    return Mesh(
        vertices=verts.astype(np.float32), faces=faces, normals=normals, uv=mesh.uv,
        attributes=attributes,
    )


def evaluate(inputs: list, params: dict, ctx) -> Mesh | Instances:
    m = matrix(params)
    src = common.geometry_input("transform", inputs, 0)
    if isinstance(src, Instances):
        if np.linalg.det(m[:3, :3]) < 0:
            raise OpError(
                "bad_input",
                "transform with a mirroring (negative-determinant) scale cannot "
                "apply to instances - the shared prototype's winding cannot flip "
                "per instance; apply the negative scale to the prototype mesh "
                "before the array/scatter node",
                param="scale",
            )
        return Instances(prototype=src.prototype, transforms=m[None] @ src.transforms)
    return apply_to_mesh(src, m)
