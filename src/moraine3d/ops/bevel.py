"""``bevel`` operator.

Semantics: chamfers the sharp convex edges of its single watertight mesh
input. Params (the full future signature): ``width`` meters (default
0.01) — distance cut back along each adjacent face; ``segments`` (int,
default 1) — v1 supports only 1 (a flat chamfer); ``angle_min_deg``
(default 30) — only edges whose faces meet at more than this normal
angle are beveled. v1 implementation cuts one plane per unique sharp
convex edge (manifold3d half-space trims), which is exact for convex
solids and small widths. At the default ``angle_min_deg`` a prism or
cylinder keeps its side edges (facet angle 360/segments) and only the
rims are chamfered; each rim plane then also cuts a little into the
two neighboring side facets, visibly on a coarse prism. Lower
``angle_min_deg`` below the facet angle to chamfer the side edges
too. Corners resolve as the
intersection of their edges' planes (a shallow apex where the chamfers
meet, not a flat corner triangle); concave edges are left untouched.
Where trims meet, the kernel can leave two vertices on one point (an
edge of no length, a few float32 steps at most); those edges are
collapsed, which moves no surface. A bevel matching no
edge is an actionable error, never a silent pass-through (a wrong
``angle_min_deg`` must not read as success). Post-conditions gate the
result: the trims must leave a valid manifold, no degenerate (sliver)
triangles, and remove no more volume than the chamfers account for —
violating any (the non-convex failure, where a global plane slices
unrelated geometry) is an actionable error, never a silently broken
mesh. ``segments > 1`` is refused until a multi-segment kernel lands
behind this signature. Output is flat-shaded; UVs are dropped.
Deterministic.
"""

from __future__ import annotations

import manifold3d
import numpy as np
import trimesh

from ..core import numbers
from ..core.mesh import DEGEN_HEIGHT_TOL, Mesh, degenerate_tri_count, flat_shaded, tri_cross
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 4
KERNELS = ("core.mesh", "manifold3d", "trimesh")
SHADING_POLICY = "replaced"  # output is flat-shaded regardless of input
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "width": numbers.FLOAT,
    "segments": numbers.INT,
    "angle_min_deg": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "dropped"  # trims rebuild topology


def _collapse_null_edges(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The trimmed mesh without its edges of no length: where three
    trims meet, the kernel can emit one point as two vertices a few
    float32 steps apart, and the triangles between them have no area.
    Each such edge is collapsed onto one of its ends. Only a triangle
    the degenerate criterion rejects is touched, so a mesh that held
    none comes back as it was."""
    reach = 8.0 * float(np.spacing(np.float32(np.abs(verts).max())))
    for _ in range(8):  # a collapse can expose a neighbor; a few passes settle
        tri = verts[faces]
        edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], axis=1)
        lengths = np.sqrt((edges**2).sum(axis=2))
        longest = lengths.max(axis=1)
        area2 = np.linalg.norm(tri_cross(tri), axis=1)
        height = np.divide(area2, longest, out=np.zeros_like(area2), where=longest > 0.0)
        null = (height <= DEGEN_HEIGHT_TOL) & (lengths.min(axis=1) <= reach)
        if not null.any():
            break
        k = lengths.argmin(axis=1)[null]
        a, b = faces[null, k], faces[null, (k + 1) % 3]
        remap = np.arange(len(verts))
        remap[np.maximum(a, b)] = np.minimum(a, b)
        for _ in range(4):  # chains of collapsed vertices resolve to one end
            remap = remap[remap]
        faces = remap[faces]
        faces = faces[
            (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        ]
    used = np.unique(faces)
    index = np.full(len(verts), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    return verts[used], index[faces]


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = common.mesh_input("bevel", inputs, 0)
    width = common.num("bevel", params, "width", 0.01, positive=True)
    segments = common.integer("bevel", params, "segments", 1, minimum=1)
    angle_min = common.num("bevel", params, "angle_min_deg", 30.0, positive=True)
    if segments != 1:
        raise OpError(
            "not_implemented",
            "bevel v1 supports segments=1 (flat chamfer) only; use segments=1 - "
            "multi-segment bevel is not implemented yet",
            param="segments",
        )

    tm = trimesh.Trimesh(
        vertices=src.vertices.astype(np.float64), faces=src.faces.astype(np.int64), process=False
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
            f"bevel input is not a watertight manifold ({m.status().name}); "
            f"close the surface before bevel",
        )

    adj = tm.face_adjacency
    angles = tm.face_adjacency_angles
    convex = tm.face_adjacency_convex
    sharp = (angles > np.deg2rad(angle_min)) & convex
    if not sharp.any():
        # A deliberate pass-through is expressed by deleting the node;
        # an unchanged mesh returned as success hides a wrong angle_min.
        sharpest = (
            f"the sharpest convex edge in this input meets at "
            f"{np.rad2deg(angles[convex].max()):.1f} deg"
            if convex.any()
            else "this input has no convex edges"
        )
        raise OpError(
            "no_effect",
            f"bevel matched no convex edge sharper than "
            f"angle_min_deg={angle_min:g} ({sharpest}); lower angle_min_deg, "
            f"or remove this bevel node",
            param="angle_min_deg",
        )

    n1 = tm.face_normals[adj[sharp, 0]]
    n2 = tm.face_normals[adj[sharp, 1]]
    nb = n1 + n2
    lens = np.linalg.norm(nb, axis=1, keepdims=True)
    if np.any(lens < 1e-9):
        # Near-antipodal normals: the bisector is 0/0 and the chamfer
        # plane undefined — the sharp-edge filter only bounds the
        # dihedral from below, so a knife edge reaches this point.
        raise OpError(
            "degenerate_edge",
            "bevel input has a knife edge (adjacent faces meet at ~180 deg), "
            "so its chamfer plane is undefined; remove the sliver geometry "
            "(often a near-tangent boolean) before bevel",
        )
    nb /= lens
    p = tm.vertices[tm.face_adjacency_edges[sharp, 0]]
    sin_half = np.sqrt(np.clip((1.0 - np.einsum("ij,ij->i", n1, n2)) / 2.0, 0.0, 1.0))
    offsets = np.einsum("ij,ij->i", nb, p) - width * sin_half

    vol_in = abs(float(m.volume()))

    # Collinear edge runs share a plane; dedupe so each cut happens once.
    planes = np.unique(np.round(np.column_stack([nb, offsets]), 9), axis=0)
    for normal_x, normal_y, normal_z, off in planes:
        # trim_by_plane keeps the half-space the normal points into, so cut
        # with the flipped normal to shave the edge wedge off.
        m = m.trim_by_plane((-normal_x, -normal_y, -normal_z), -off)
    if m.is_empty():
        raise OpError(
            "empty_result",
            "bevel consumed the whole solid; 'width' is too large for this mesh",
            param="width",
        )

    trimmed = m.to_mesh()
    verts, faces = _collapse_null_edges(
        np.asarray(trimmed.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(trimmed.tri_verts, dtype=np.int64),
    )
    out = flat_shaded(verts, faces)
    # Post-conditions: each global half-space trim is exact only for the
    # convex case; on other shapes a plane derived from one edge can
    # slice unrelated geometry. Gate the result instead of flowing a
    # broken mesh on unflagged. A wedge removes at most width^2/2 per
    # meter of edge (d^2 tan(a/2) with d = width cos(a/2), maximal at a
    # 90 deg corner); the factor 4 absorbs corner interactions.
    symptoms = []
    if m.status() != manifold3d.Error.NoError:
        symptoms.append(f"kernel status {m.status().name}")
    closed = manifold3d.Manifold(
        mesh=manifold3d.Mesh(
            vert_properties=verts.astype(np.float32), tri_verts=faces.astype(np.uint32)
        )
    )
    if closed.status() != manifold3d.Error.NoError:
        symptoms.append(f"open after cleanup ({closed.status().name})")
    degen = degenerate_tri_count(out.vertices, out.faces)
    if degen:
        symptoms.append(f"{degen} degenerate triangle(s)")
    edge_pts = tm.vertices[tm.face_adjacency_edges[sharp]]
    chamfer_budget = 4.0 * width**2 * float(
        np.linalg.norm(edge_pts[:, 0] - edge_pts[:, 1], axis=1).sum()
    )
    removed = vol_in - abs(float(m.volume()))
    if removed > chamfer_budget:
        symptoms.append(
            f"removed {removed:.6g} m^3 where the chamfers account for at "
            f"most {chamfer_budget:.6g} m^3"
        )
    if symptoms:
        raise OpError(
            "broken_result",
            f"bevel produced a broken mesh on this shape "
            f"({'; '.join(symptoms)}): v1 bevel cuts one global plane per "
            f"edge, exact only for convex solids - reduce width, or bevel the "
            f"convex pieces before the boolean that made this shape",
        )
    return out
