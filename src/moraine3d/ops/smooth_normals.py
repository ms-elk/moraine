"""``smooth_normals`` operator.

Semantics: recomputes shading of its single mesh input by angle
threshold — the ecosystem-standard angle-threshold auto-smooth every
DCC tool ships. Coincident vertices are welded, then re-split so that
faces meeting across an edge
at up to ``angle_max_deg`` (default 30, > 0) share a vertex with the
corner-angle-weighted average normal (invariant to how coplanar regions
are triangulated — CSG output triangulation is arbitrary), while sharper
edges (and non-manifold edges) keep hard per-face corners. Stored input
normals are discarded
and rebuilt from geometry; face count and winding are unchanged. The
intended use is restoring curved-surface shading after ``boolean`` /
``bevel``, whose outputs are flat-shaded. Input UVs are dropped
(unwrap downstream). Known limit (shared with every angle-based
auto-smooth): near-tangent junctions — e.g. equal-radius cylinder
unions, whose crease dihedral falls to 0 at the tangent points — smooth
over and bleed shading across the seam; model the members with distinct
radii so the crease stays above the threshold. Deterministic.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..core import numbers
from ..core.mesh import Mesh, tri_cross
from . import common

OP_IMPL_VERSION = 1
KERNELS = ("core.mesh", "trimesh")
SHADING_POLICY = "replaced"  # stored input normals discarded and rebuilt
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "angle_max_deg": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "dropped"  # topology is rebuilt (weld + re-split)
PLACEMENT = "identity"  # the weld merges coincident vertices; positions untouched

# Dihedral floor, radians (~0.06 deg): below it two faces are one plane
# triangulated (boolean output splits planar faces arbitrarily; the
# dihedral is float noise), and smoothing across the edge changes no
# normal. Above it and within the threshold, an edge is a tessellated
# curve — the shading this op restores.
_COPLANAR_DIHEDRAL = 1e-3
# Sliver floor for the curvature test, height over longest edge: a
# face's stored (float32) normal is off by about (coordinate ulp /
# height), so a face thinner than this ratio of its own extent carries
# a normal error that can exceed the dihedral floor while the surface
# is planar. Such faces are left out of the test; they say nothing about
# curvature either way.
_SLIVER_ASPECT = 1e-4


def would_smooth(mesh: Mesh, angle_max_deg: float = common.CREASE_ANGLE_DEG_DEFAULT) -> bool:
    """True when this op at ``angle_max_deg`` would change some normal:
    the welded surface has an edge shared by two faces whose dihedral
    lies above the coplanar floor and within the threshold — a curved
    surface tessellated into flat faces. All-planar geometry (every
    dihedral zero or a hard crease) returns False: smoothing it is a
    no-op, so nothing is lost by leaving it flat-shaded. Faces thinner
    than ``_SLIVER_ASPECT`` are excluded from the test (their normals
    are storage noise)."""
    tm = trimesh.Trimesh(
        vertices=mesh.vertices.astype(np.float64),
        faces=mesh.faces.astype(np.int64),
        process=False,
    )
    tm.merge_vertices()
    angles = np.asarray(tm.face_adjacency_angles, dtype=np.float64)
    if len(angles) == 0:
        return False
    tri = np.asarray(tm.vertices, dtype=np.float64)[np.asarray(tm.faces, dtype=np.int64)]
    twice_area = np.linalg.norm(tri_cross(tri), axis=1)
    longest = np.sqrt(
        np.maximum.reduce([
            ((tri[:, 1] - tri[:, 0]) ** 2).sum(axis=1),
            ((tri[:, 2] - tri[:, 1]) ** 2).sum(axis=1),
            ((tri[:, 0] - tri[:, 2]) ** 2).sum(axis=1),
        ])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(longest > 0.0, twice_area / (longest * longest), 0.0)
    sound = aspect >= _SLIVER_ASPECT
    adj = np.asarray(tm.face_adjacency, dtype=np.int64)
    tested = sound[adj[:, 0]] & sound[adj[:, 1]]
    band = (angles > _COPLANAR_DIHEDRAL) & (angles <= np.deg2rad(angle_max_deg))
    return bool(np.any(band & tested))


def _connected_components(n: int, rows: np.ndarray, cols: np.ndarray) -> tuple[int, np.ndarray]:
    """Undirected connected components as (count, compact labels).

    Min-label propagation with pointer jumping — whole-array numpy rounds,
    O(log diameter) iterations, no per-node Python work (rule 1). Labels
    are compacted in first-occurrence order, so the result (and every
    downstream vertex order) is deterministic.
    """
    labels = np.arange(n, dtype=np.int64)
    while True:
        nxt = labels.copy()
        np.minimum.at(nxt, rows, labels[cols])
        np.minimum.at(nxt, cols, labels[rows])
        nxt = np.minimum(nxt, nxt[nxt])  # pointer jumping
        if np.array_equal(nxt, labels):
            break
        labels = nxt
    _, compact = np.unique(labels, return_inverse=True)
    return int(compact.max(initial=-1)) + 1, compact


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = common.mesh_input("smooth_normals", inputs, 0)
    angle = common.num(
        "smooth_normals", params, "angle_max_deg",
        common.CREASE_ANGLE_DEG_DEFAULT, positive=True,
    )

    tm = trimesh.Trimesh(
        vertices=src.vertices.astype(np.float64), faces=src.faces.astype(np.int64), process=False
    )
    tm.merge_vertices()
    verts = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)
    n_faces = len(faces)

    tri = verts[faces]
    face_cross = tri_cross(tri)
    fn_len = np.linalg.norm(face_cross, axis=1, keepdims=True)
    fn_len[fn_len == 0] = 1.0
    face_normal = face_cross / fn_len
    # Corner angles (Thuermer-Wuethrich weights): a fan normal weighted by
    # each face's interior angle at the vertex does not shift when a coplanar
    # region is re-triangulated, unlike area weighting.
    to_next = tri[:, [1, 2, 0]] - tri
    to_prev = tri[:, [2, 0, 1]] - tri
    to_next /= np.maximum(np.linalg.norm(to_next, axis=2, keepdims=True), 1e-30)
    to_prev /= np.maximum(np.linalg.norm(to_prev, axis=2, keepdims=True), 1e-30)
    corner_angle = np.arccos(np.clip(np.einsum("mki,mki->mk", to_next, to_prev), -1.0, 1.0))

    # Corners are (face, slot) pairs, id = 3*face + slot. Two corners of the
    # same vertex join when their faces share an edge no sharper than the
    # threshold; each connected component is a smoothing fan and becomes one
    # output vertex. Edges shared by != 2 faces have no adjacency entry, so
    # they stay hard.
    adj = tm.face_adjacency
    smoothed = tm.face_adjacency_angles <= np.deg2rad(angle)
    fa, fb = adj[smoothed, 0], adj[smoothed, 1]
    edge_verts = tm.face_adjacency_edges[smoothed]

    def corner(face_idx: np.ndarray, vert_idx: np.ndarray) -> np.ndarray:
        slot = (faces[face_idx] == vert_idx[:, None]).argmax(axis=1)
        return 3 * face_idx + slot

    rows = np.concatenate([corner(fa, edge_verts[:, 0]), corner(fa, edge_verts[:, 1])])
    cols = np.concatenate([corner(fb, edge_verts[:, 0]), corner(fb, edge_verts[:, 1])])
    n_comp, labels = _connected_components(3 * n_faces, rows, cols)

    corner_vertex = faces.reshape(-1)
    positions = np.zeros((n_comp, 3))
    positions[labels] = verts[corner_vertex]  # corners in a fan share one vertex
    normals = np.zeros((n_comp, 3))
    contrib = face_normal[:, None, :] * corner_angle[:, :, None]
    np.add.at(normals, labels, contrib.reshape(-1, 3))
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return Mesh(
        vertices=positions.astype(np.float32),
        faces=labels.reshape(n_faces, 3).astype(np.int32),
        normals=(normals / lens).astype(np.float32),
    )
