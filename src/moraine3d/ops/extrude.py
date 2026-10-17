"""``extrude`` operator.

Semantics: solidifies its single mesh input by sweeping it along an
offset. Params: ``distance`` meters (default 0.1, must be > 0 — a
negative distance would turn the solid inside out; sweep the other way
with a ``direction`` vector instead); ``direction`` — the string
``"normal"`` (offset each vertex along its vertex normal; the default)
or a fixed [x, y, z] vector (normalized, scaled by ``distance``). A
vector that opposes every face normal sweeps the flipped sheet — the
output is the intended outward solid, exactly as if the input were
mirrored first; a vector tangent to any face, or opposing some faces
while aligning with others, is an actionable error (the sweep would
self-intersect). Output = offset copy of the surface (front, original
winding) + the original surface with winding flipped (back) +
hard-edged side quads along every boundary edge, wound so faces point
outward for a positive ``distance`` along the surface normal. A closed
smooth-shaded input (no boundary edges) yields a two-shell solid (a
thickened shell). Boundary edges are vertex-index edges, so a
flat-shaded *closed* surface (duplicated vertices with divergent
normals — ``box``, ``boolean``/``bevel`` output) would extrude into
per-face slabs with open wedges, not a thickened shell — it is
refused; shell a solid with a boolean difference of an inset copy, or
weld it smooth first (``smooth_normals`` with ``angle_max_deg`` above
its hardest edge — the default 30 keeps hard edges split, which stay
refused). Seam-split closed surfaces whose duplicates carry agreeing
normals (UV seams) are smooth-shaded, not flat-shaded, and extrude as
their boundary edges imply. Open duplicated-vertex sheets keep their
documented per-piece behavior; thicken smooth-shaded sheets (``grid``,
``smooth_normals`` output). Sides are flat-shaded; front/back keep the
input's shading. Input UVs are dropped (unwrap downstream).
Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Mesh, divergent_duplicates, tri_cross, vertex_normals
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 2
KERNELS = ("core.mesh",)
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "distance": numbers.FLOAT,
    "direction": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "dropped"  # topology is rebuilt (front/back/sides)

# Orientation classifier for a fixed 'direction' vector: |dot| at or
# below this is tangent. Unit-normal dots, so the guard catches exact
# tangency (degenerate side slabs); near-tangent sweeps stay legal.
_DIR_EPS = 1e-9


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Directed edges (a, b) that appear in exactly one face, face order."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[counts[inv.reshape(-1)] == 1]


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = common.mesh_input("extrude", inputs, 0)
    # > 0 only: a negative distance would put the front shell behind the
    # back shell and invert the solid (all faces wound inward). Sweeping
    # the other way is a 'direction' vector, not a negative distance.
    distance = common.num("extrude", params, "distance", 0.1, positive=True)
    direction = params.get("direction", "normal")

    v = src.vertices.astype(np.float64)
    faces = src.faces
    stored_n = src.normals

    edges = _boundary_edges(faces)
    if len(edges):
        # A flat-shaded closed surface opens every vertex-index edge
        # while the welded surface has none: extruding it would build
        # per-face slabs with open wedges, never a thickened shell.
        # Seam-split smooth surfaces (duplicates with agreeing normals)
        # are not flat-shaded and extrude as their boundary edges imply.
        uniq, welded = np.unique(v, axis=0, return_inverse=True)
        if (
            len(uniq) < len(v)
            and len(_boundary_edges(welded[faces.astype(np.int64)])) == 0
            and divergent_duplicates(src, welded=(len(uniq), welded))
        ):
            raise OpError(
                "bad_input",
                "extrude input is a flat-shaded closed surface (duplicated "
                "vertices with divergent normals); extruding it builds "
                "per-face slabs with open wedges, not a thickened shell - "
                "shell a solid with a boolean difference of an inset copy, "
                "or weld it smooth first (smooth_normals with "
                "angle_max_deg above its hardest edge; the default 30 "
                "keeps hard edges split, which stay refused)",
            )

    if isinstance(direction, str):
        if direction != "normal":
            raise OpError(
                "bad_param",
                "extrude 'direction' must be \"normal\" or an [x, y, z] vector",
                param="direction",
            )
        offset = vertex_normals(src) * distance
    else:
        d = common.vec3("extrude", params, "direction", [0.0, 1.0, 0.0])
        n = np.linalg.norm(d)
        if n == 0:
            raise OpError("bad_param", "extrude 'direction' vector must be non-zero", param="direction")
        d = d / n
        offset = np.tile(d * distance, (len(v), 1))

        # The assembly below assumes the sweep runs along the surface
        # normal; a fixed vector may not. Classify by unit face normals.
        fn = tri_cross(v[faces.astype(np.int64)])
        lens = np.linalg.norm(fn, axis=1)
        valid = lens > 0
        if not valid.any():
            raise OpError(
                "bad_input",
                "extrude input has only degenerate (zero-area) faces, so "
                "no sweep orientation exists; fix the input sheet",
            )
        dots = fn[valid] @ d / lens[valid]
        n_opp = int(np.count_nonzero(dots < -_DIR_EPS))
        n_ali = int(np.count_nonzero(dots > _DIR_EPS))
        n_tan = int(valid.sum()) - n_opp - n_ali
        if n_opp and not n_ali and not n_tan:
            # Every face opposes the sweep: the author meant the
            # mirrored sheet — flip winding so the aligned-case
            # assembly below produces the intended outward solid.
            faces = faces[:, ::-1]
            stored_n = None if stored_n is None else -stored_n
            edges = edges[:, ::-1]
        elif n_opp or n_tan:
            what = (
                f"is tangent to {n_tan} of {len(faces)} face(s)"
                if not n_opp
                else f"opposes {n_opp} and is tangent to {n_tan} of "
                f"{len(faces)} face(s)"
                if n_tan
                else f"opposes {n_opp} of {len(faces)} face(s) and aligns "
                f"with the rest"
            )
            raise OpError(
                "bad_param",
                f"extrude 'direction' {what}; the sweep would "
                f"self-intersect - extrude the flat regions separately, or "
                f"use direction \"normal\"",
                param="direction",
            )

    n_src = len(v)
    front_v = v + offset
    back_v = v
    front_n = stored_n
    back_n = None if stored_n is None else -stored_n
    # Side quad per boundary directed edge (a, b): corners back-a, back-b,
    # front-b, front-a. With CCW faces, boundary edges run CCW seen from
    # the front, which winds sides outward for positive distance.
    side_quads = np.stack(
        [back_v[edges[:, 0]], back_v[edges[:, 1]], front_v[edges[:, 1]], front_v[edges[:, 0]]],
        axis=1,
    )  # (k, 4, 3)
    k = len(edges)
    side_v = side_quads.reshape(-1, 3)
    q = np.arange(k) * 4
    side_f = np.concatenate(
        [np.stack([q, q + 1, q + 2], axis=1), np.stack([q, q + 2, q + 3], axis=1)]
    )
    sn = np.cross(side_quads[:, 1] - side_quads[:, 0], side_quads[:, 3] - side_quads[:, 0])
    lens = np.linalg.norm(sn, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    side_n = np.repeat(sn / lens, 4, axis=0)

    verts = np.concatenate([front_v, back_v, side_v])
    out_faces = np.concatenate(
        [
            faces,  # front
            faces[:, ::-1] + n_src,  # back, flipped
            (side_f + 2 * n_src) if k else np.empty((0, 3), np.int64),
        ]
    )
    normals = None
    if front_n is not None:
        normals = np.concatenate([front_n, back_n, side_n.astype(np.float32)])
    return Mesh(
        vertices=verts.astype(np.float32),
        faces=out_faces.astype(np.int32),
        normals=None if normals is None else normals.astype(np.float32),
    )
