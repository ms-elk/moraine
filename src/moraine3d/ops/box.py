"""``box`` operator.

Semantics: emits an axis-aligned rectangular box centered at the origin
(pivot at center of volume), with edge lengths ``size`` = [sx, sy, sz] in
meters (a scalar broadcasts to a cube — size is scale-like; translations
and rotations take no scalar) and ``segments`` = [nx, ny, nz] divisions
per axis (default [1, 1, 1], each 1..4096; the quads across all six
faces are bounded by the shared tessellation cell budget). Each face is
its own (nu + 1) x (nv + 1) vertex lattice — vertices duplicated across
faces for hard edges, per-face outward normals, CCW-wound triangles —
so the default is 24 vertices and 12 triangles. Segments do not change
the shape; they give a face interior vertices, which is what keeps a
boolean cut through that face from triangulating as corner fans of
slivers (the ``boolean`` face discipline). Deterministic; no stochastic
parameters.
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

# Per face: (normal axis, normal sign, u axis, v axis) with u x v == n,
# so (u, v) corners ordered (-,-) (+,-) (+,+) (-,+) wind CCW from outside.
_FACES = (
    (0, 1.0, 1, 2),   # +X
    (0, -1.0, 2, 1),  # -X
    (1, 1.0, 2, 0),   # +Y
    (1, -1.0, 0, 2),  # -Y
    (2, 1.0, 0, 1),   # +Z
    (2, -1.0, 1, 0),  # -Z
)

_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _segments(params: dict) -> tuple[int, int, int]:
    v = params.get("segments", [1, 1, 1])
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 3
        or not all(
            isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= SEGMENTS_MAX
            for c in v
        )
    ):
        raise OpError(
            "bad_param",
            f"box 'segments' must be 3 integers in 1..{SEGMENTS_MAX}",
            param="segments",
        )
    nx, ny, nz = v
    # The per-axis ceilings multiply across the three face pairs, so the
    # product can exceed what they admit alone - refuse before
    # allocating, per the ceilings' own rationale (the revolve precedent).
    cells = 2 * (nx * ny + ny * nz + nz * nx)
    if cells > common.TESSELLATION_CELLS_MAX:
        raise OpError(
            "bad_param",
            f"box tessellation is too large: segments {list(v)} make {cells} "
            f"face quads, over the {common.TESSELLATION_CELLS_MAX} cell budget; "
            "reduce segments",
            param="segments",
        )
    return nx, ny, nz


def _unsegmented(half: np.ndarray) -> Mesh:
    # The pre-``segments`` layout, kept verbatim so the default output
    # stays bit-identical (pinned by the frozen result hash): 4 corners
    # per face in ring order, two triangles off corner 0.
    eye = np.eye(3)
    verts = np.empty((24, 3))
    normals = np.empty((24, 3))
    for i, (na, ns, ua, va) in enumerate(_FACES):
        n = eye[na] * ns
        quad = n + _CORNERS[:, 0:1] * eye[ua] + _CORNERS[:, 1:2] * eye[va]
        verts[4 * i : 4 * i + 4] = quad
        normals[4 * i : 4 * i + 4] = n
    verts *= half

    tri_pattern = np.array([[0, 1, 2], [0, 2, 3]])
    faces = (np.arange(6)[:, None, None] * 4 + tri_pattern[None]).reshape(-1, 3)

    return Mesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
    )


def _segmented(half: np.ndarray, seg: tuple[int, int, int]) -> Mesh:
    eye = np.eye(3)
    vert_blocks, normal_blocks, face_blocks = [], [], []
    base = 0
    for na, ns, ua, va in _FACES:
        n = eye[na] * ns
        nu, nv = seg[ua], seg[va]
        us = np.linspace(-1.0, 1.0, nu + 1)
        vs = np.linspace(-1.0, 1.0, nv + 1)
        gu, gv = np.meshgrid(us, vs, indexing="ij")  # u-major lattice
        verts = n + gu.ravel()[:, None] * eye[ua] + gv.ravel()[:, None] * eye[va]
        vert_blocks.append(verts)
        normal_blocks.append(np.tile(n, (len(verts), 1)))
        iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        a = base + (iu * (nv + 1) + iv).ravel()  # (u, v) corner
        b = a + (nv + 1)  # +u
        c = b + 1  # +u +v
        d = a + 1  # +v
        # Same corner ring as the unsegmented face, (-,-) (+,-) (+,+) (-,+),
        # so u x v == n keeps the winding CCW from outside.
        face_blocks.append(np.stack([a, b, c], axis=1))
        face_blocks.append(np.stack([a, c, d], axis=1))
        base += len(verts)
    verts = np.concatenate(vert_blocks) * half
    return Mesh(
        vertices=verts.astype(np.float32),
        faces=np.concatenate(face_blocks).astype(np.int32),
        normals=np.concatenate(normal_blocks).astype(np.float32),
    )


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    half = common.vec3(
        "box", params, "size", [1.0, 1.0, 1.0], positive=True, scalar_ok=True
    ) / 2.0
    seg = _segments(params)
    if seg == (1, 1, 1):
        return _unsegmented(half)
    return _segmented(half, seg)
