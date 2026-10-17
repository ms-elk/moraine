"""Mating frames — the one definition shared by the ``mate`` operator and
the ``mated`` assertion (DESIGN 5.1, 5.2).

A mating frame is derived from a tagged face group's world-space
triangles: a planar face set yields ``(origin, normal, secondary)``.

- **Planarity gate.** The area-weighted unit normal is the frame normal;
  every face normal must agree with it within ``NORMAL_ANGLE_TOL_DEG``
  (faces below a relative area floor are slivers whose normals carry no
  information and are not tested — the largest faces are always tested,
  so a huge set of equal faces never empties the test) and every vertex
  must lie within
  ``COPLANAR_TOL`` of the mean plane, else the set is ``not_planar`` (a
  mixed-surface tag would put the origin between surfaces). A zero-area
  set, or one whose face normals cancel (opposite-facing faces), is
  ``degenerate``.
- **Opening-aware origin.** The group's boundary loops are extracted
  (edges used once inside the group's face set, after welding vertices by
  exact position — flat-shaded and boolean outputs share no indices);
  loops are followed edge to edge and split at pinch vertices (a vertex
  where loops touch), so two openings or two patches meeting at a point
  stay two. A chain that leaves one pinch vertex and ends at another
  does not close, so its loops cannot be told apart — that set is
  ``ambiguous_opening`` (a refusal, never a guessed count). With CCW
  winding the outer loop has positive signed area in the frame's plane
  and each hole negative. One opening (a port, a tube end, a flange
  bore) → its centroid is the origin, the feature a joint registers on;
  no opening → the area-weighted face centroid; two or more openings →
  ``ambiguous_opening``; two or more outer loops (edge-disconnected
  patches) → ``disconnected``.
- **Secondary axis.** World +Y projected into the plane — the upright
  rule, so assemblies keep their up; a face within ``HORIZONTAL_DEG`` of
  level falls back to +Z, the asset front (DESIGN 4.3). Deterministic by
  construction; the remaining roll is the explicit ``roll_deg``
  parameter of ``mate``.
- **Solve.** ``solve`` returns the rigid transform that carries a part
  frame onto an anchor frame: part normal → −anchor normal, part
  secondary → anchor secondary, then the explicit roll about the anchor
  normal, then translation origin → origin. Horizontal mating (anchor
  +Y, part −Y) degenerates to a pure translation, consistent with
  ``place`` ``on_top``.

Pure numpy / float64; no per-vertex Python loops. Errors are
``FrameError`` with codes ``degenerate`` / ``not_planar`` /
``disconnected`` / ``ambiguous_opening``, role-free and factual — the op
prefixes the role and appends the remedy, the assertion turns them into
an unevaluatable reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import MoraineError
from .mesh import Mesh, tri_cross

# Vertex spread allowed across one mating plane, meters: above float32
# vertex noise at asset scale, below any intentional step between surfaces
# (the same bar as place's single-surface gate).
COPLANAR_TOL = 1e-4
# Per-face normal agreement with the mean normal, degrees: far above the
# float32 noise on small triangles, far below any mixed-orientation group
# (a curved side band fails, as it should — curved mating is not v1).
NORMAL_ANGLE_TOL_DEG = 1.0
# Faces below this fraction of the set's area are slivers: their normal
# direction is numerical noise, so they are excluded from the normal
# agreement test (they still contribute their negligible area elsewhere).
SLIVER_AREA_FRACTION = 1e-6
# A face whose normal is within this angle of vertical is horizontal: the
# upright reference (+Y) has no meaningful in-plane direction and the
# secondary axis falls back to +Z (asset front). The same order as the
# normal-agreement tolerance, so a face set that passes as planar cannot
# straddle the threshold copy by copy.
HORIZONTAL_DEG = 1.0
# Relative floor for |summed normal| / summed |cross|: below it the face
# normals cancel and no plane side is defined.
_CANCEL_EPS = 1e-6
# Loops whose |signed area| is below this fraction of the group area are
# numerical residue (zero-length boundary edges), not openings.
_LOOP_AREA_EPS = 1e-9
# A point closer than this (meters) to a segment's carrying line counts
# as ON it for the crossing test: collinear rim contact — a part edge
# flush with the anchor contour — produces orientation determinants that
# are pure float noise, and "grazing is not a crossing" must hold in
# floats, not just exact arithmetic. Far above f64 noise at asset scale,
# far below any real excursion the rim-distance tolerance would admit.
_CROSSING_EPS = 1e-9

_UP = np.array([0.0, 1.0, 0.0])
_FRONT = np.array([0.0, 0.0, 1.0])


# Shared-kernel version (DESIGN 4.5): ops that consume this module declare
# it in KERNELS and this integer joins their cache keys. Bump on ANY
# output-semantics change of a function an op consumes (frame detection,
# opening handling, the solve), bug fixes included — like OP_IMPL_VERSION.
KERNEL_VERSION = 1


class FrameError(MoraineError):
    """A face set that has no mating frame; ``code`` names why."""


# The terminating remedy per FrameError code — one home, so the op's error
# and the assertion's unevaluatable reason steer the author identically.
REMEDY = {
    "not_planar": "tag a single planar face set (split the tag per surface)",
    "degenerate": "tag a face set with area and one facing side",
    "disconnected": "tag one connected surface",
    "ambiguous_opening": "tag the faces around one opening",
}


@dataclass(frozen=True)
class Frame:
    origin: np.ndarray  # (3,) float64
    normal: np.ndarray  # (3,) unit
    secondary: np.ndarray  # (3,) unit, in-plane
    openings: int  # inner boundary loops found (0 or 1 when a frame exists)


def origin_word(frame: "Frame") -> str:
    """The frame's origin provenance as the report word: ``opening`` when
    the origin is the single opening's centroid, ``centroid`` for the
    area-weighted fallback (the word every consumer of a frame reports,
    so the op side and the assertion side never spell it differently)."""
    return "opening" if frame.openings == 1 else "centroid"


def group_triangles(
    mesh: Mesh, faces_idx: np.ndarray, matrix: np.ndarray | None = None
) -> np.ndarray:
    """World-space ``(m, 3, 3)`` float64 triangles of a face group; the copy
    matrix (an Instances transform) is applied first when given. Gathers
    before casting, so the cost scales with the group, not the mesh."""
    faces = mesh.faces[np.asarray(faces_idx, dtype=np.int64)].astype(np.int64)
    tri = mesh.vertices[faces].astype(np.float64)
    if matrix is not None:
        m = np.asarray(matrix, dtype=np.float64)
        tri = tri @ m[:3, :3].T + m[:3, 3]
    return tri


def secondary_axis(normal: np.ndarray) -> np.ndarray:
    """The upright rule: +Y projected into the plane; +Z when the plane is
    within HORIZONTAL_DEG of level; unit length."""
    tilt = float(np.hypot(normal[0], normal[2]))  # |projection of the normal off vertical|
    ref = _FRONT if tilt < np.sin(np.deg2rad(HORIZONTAL_DEG)) else _UP
    s = ref - float(ref @ normal) * normal
    return s / np.linalg.norm(s)


def _components(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label per node of an undirected edge list —
    min-label propagation with pointer jumping, whole-array passes only."""
    labels = np.arange(n_nodes)
    if len(edges) == 0:
        return labels
    a, b = edges[:, 0], edges[:, 1]
    while True:
        new = labels.copy()
        np.minimum.at(new, a, labels[b])
        np.minimum.at(new, b, labels[a])
        new = new[new]
        if np.array_equal(new, labels):
            return labels
        labels = new


def _loop_components(n_vertices: int, boundary: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(loop id per directed boundary edge, all chains closed)``, following
    each edge to the one edge that leaves its end vertex. A pinch vertex
    (more than one boundary edge in or out — two loops touching at a
    point) breaks the chain, so loops that merely touch stay distinct; a
    chain that leaves a pinch and returns to the same pinch still closes
    geometrically, so its signed area is its loop's. A chain that ends at
    a different pinch (or a lone edge between two pinches) does not close,
    and its signed area is meaningless — reported as ``False``."""
    a, b = boundary[:, 0], boundary[:, 1]
    m = len(boundary)
    simple = (np.bincount(a, minlength=n_vertices) == 1) & (
        np.bincount(b, minlength=n_vertices) == 1
    )
    leaving = np.full(n_vertices, -1)
    leaving[a] = np.arange(m)  # unique where the start vertex is simple
    follow = np.flatnonzero(simple[b])
    links = np.column_stack([follow, leaving[b[follow]]])
    labels = _components(m, links)
    comp = np.unique(labels, return_inverse=True)[1].reshape(-1)
    # Each component is a cycle (every vertex simple) or a path with one
    # head edge (start vertex not simple) and one tail edge (end vertex
    # not simple); the path closes iff head start == tail end.
    k = int(comp.max()) + 1 if m else 0
    head = np.full(k, -1)
    tail = np.full(k, -1)
    head[comp[~simple[a]]] = a[~simple[a]]
    tail[comp[~simple[b]]] = b[~simple[b]]
    return comp, bool(np.all(head == tail))


def _welded_boundary(tri: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed boundary edges of the live faces over exact-position
    welded vertices: ``(uniq (n, 3), boundary (m, 2))`` — edges used once
    inside the face set, CCW-directed. One home for the frame's opening
    walk and the containment test, so both see the same contour."""
    pts = tri.reshape(-1, 3)
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    fw = inv.reshape(-1, 3)[live]
    edges = np.stack([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]], axis=1).reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, key_inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[key_inv.reshape(-1)] == 1]
    return uniq, boundary[boundary[:, 0] != boundary[:, 1]]


def group_frame(tri: np.ndarray) -> Frame:
    """The mating frame of a planar face set (see module docstring)."""
    tri = np.asarray(tri, dtype=np.float64)
    if len(tri) == 0:
        raise FrameError("degenerate", "the face set is empty")
    cross = tri_cross(tri)
    twice_area = np.linalg.norm(cross, axis=1)
    total2 = float(twice_area.sum())
    if total2 <= 0.0:
        raise FrameError("degenerate", "the face set has zero area")
    nsum = cross.sum(axis=0)
    mag = float(np.linalg.norm(nsum))
    if mag <= _CANCEL_EPS * total2:
        raise FrameError(
            "degenerate",
            "the face normals cancel (summed normal ~ 0) - the set spans "
            "opposite-facing faces and has no side",
        )
    normal = nsum / mag
    live = twice_area > 0.0
    # The largest face(s) always qualify: a set of millions of equal faces
    # puts every face under the relative floor, and an empty test set
    # would be a raw reduction error, not a verdict.
    tested = twice_area >= min(SLIVER_AREA_FRACTION * total2, float(twice_area.max()))
    dots = (cross[tested] / twice_area[tested, None]) @ normal
    worst = float(np.degrees(np.arccos(np.clip(dots.min(), -1.0, 1.0))))
    if worst > NORMAL_ANGLE_TOL_DEG:
        raise FrameError(
            "not_planar",
            f"face normals spread up to {worst:.3g} deg from the mean normal",
        )
    centers = tri.mean(axis=1)
    centroid = (centers * twice_area[:, None]).sum(axis=0) / total2
    pts = tri.reshape(-1, 3)
    depth = (pts - centroid) @ normal
    span = float(depth.max() - depth.min())
    if span > COPLANAR_TOL:
        raise FrameError(
            "not_planar",
            f"vertices span {span:.6g} m across the mean plane",
        )
    secondary = secondary_axis(normal)
    tangent = np.cross(normal, secondary)

    # Boundary loops over exact-position welded vertices of the live faces.
    uniq, boundary = _welded_boundary(tri, live)
    openings = 0
    origin = centroid
    if len(boundary):
        comp, closed = _loop_components(len(uniq), boundary)
        if not closed:
            raise FrameError(
                "ambiguous_opening",
                "boundary loops pinch together at more than one vertex, so the "
                "openings cannot be told apart",
            )
        k = int(comp.max()) + 1
        u = (uniq - centroid) @ secondary
        v = (uniq - centroid) @ tangent
        a, b = boundary[:, 0], boundary[:, 1]
        cr = u[a] * v[b] - u[b] * v[a]
        area2 = np.bincount(comp, weights=cr, minlength=k)  # 2 * signed area
        floor = _LOOP_AREA_EPS * total2
        outer = np.flatnonzero(area2 > floor)
        holes = np.flatnonzero(area2 < -floor)
        if len(outer) > 1:
            raise FrameError(
                "disconnected",
                f"{len(outer)} edge-disconnected face patches in the set",
            )
        openings = int(len(holes))
        if openings > 1:
            raise FrameError(
                "ambiguous_opening",
                f"{openings} openings in the face set",
            )
        if openings == 1:
            h = int(holes[0])
            cu = np.bincount(comp, weights=(u[a] + u[b]) * cr, minlength=k)[h]
            cv = np.bincount(comp, weights=(v[a] + v[b]) * cr, minlength=k)[h]
            a2 = area2[h]
            origin = centroid + (cu / (3.0 * a2)) * secondary + (cv / (3.0 * a2)) * tangent
    return Frame(
        origin=np.asarray(origin, dtype=np.float64),
        normal=normal,
        secondary=secondary,
        openings=openings,
    )


def _points_in_loop(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ray-casting parity per point ``P (k, 2)`` against one loop's edges
    ``A→B (m, 2)``: True where the point is inside. Points on the loop are
    parity-unstable; the caller's rim-distance tolerance owns that band."""
    px, py = P[:, 0][:, None], P[:, 1][:, None]
    ax, ay = A[:, 0][None], A[:, 1][None]
    bx, by = B[:, 0][None], B[:, 1][None]
    cond = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = ax + (py - ay) * (bx - ax) / (by - ay)
    hits = cond & (px < xint)  # NaN off the cond mask compares False
    return hits.sum(axis=1) % 2 == 1


def _dist_points_segments(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance from each point ``P (k, 2)`` to the nearest of the segments
    ``A→B (m, 2)``."""
    d = B - A
    ap = P[:, None, :] - A[None]
    denom = np.maximum((d * d).sum(axis=1), 1e-300)
    t = np.clip(np.einsum("kmi,mi->km", ap, d) / denom, 0.0, 1.0)
    proj = A[None] + t[..., None] * d[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=2).min(axis=1)


def _proper_crossings(
    P1: np.ndarray, P2: np.ndarray, A: np.ndarray, B: np.ndarray,
    tol: float = 0.0,
) -> int:
    """Count segments ``P1→P2`` that strictly cross a segment ``A→B``:
    both orientation pairs strictly opposite. Grazing — an endpoint on the
    boundary, or a collinear overlap — is not a crossing; the vertex
    rim-distance test owns that band. ``tol`` widens the band to the
    caller's allowance: an endpoint within ``tol`` of the other segment's
    line cannot straddle it, so a rim-sharing edge (collinear within the
    allowance, float noise included) is containment, not a crossing —
    only an excursion beyond ``tol`` between two interior points counts."""

    def orient(sa, sb, c):
        return (sb[..., 0] - sa[..., 0]) * (c[..., 1] - sa[..., 1]) - (
            sb[..., 1] - sa[..., 1]
        ) * (c[..., 0] - sa[..., 0])

    o1 = orient(A[None], B[None], P1[:, None])
    o2 = orient(A[None], B[None], P2[:, None])
    o3 = orient(P1[:, None], P2[:, None], A[None])
    o4 = orient(P1[:, None], P2[:, None], B[None])
    # |orient(sa, sb, c)| = |sb-sa| * dist(c, line sa-sb): thresholding at
    # band * |segment| turns the orientation test into a distance test, so
    # a strict straddle needs both endpoints beyond the band on opposite
    # sides. The band is the larger of the collinear-float-noise floor and
    # the caller's tol.
    band = max(_CROSSING_EPS, tol)
    e_ab = band * np.linalg.norm(B - A, axis=1)[None]
    e_p = band * np.linalg.norm(P2 - P1, axis=1)[:, None]
    straddle_ab = ((o1 < -e_ab) & (o2 > e_ab)) | ((o1 > e_ab) & (o2 < -e_ab))
    straddle_p = ((o3 < -e_p) & (o4 > e_p)) | ((o3 > e_p) & (o4 < -e_p))
    crossing = straddle_ab & straddle_p
    return int(crossing.any(axis=1).sum())


def contained_excursion(
    anchor_tri: np.ndarray, part_tri: np.ndarray, frame: Frame,
    tol: float = 0.0,
) -> tuple[float, int]:
    """How far the part face set leaves the anchor face set's outer
    contour, measured in the anchor plane (the ``mated`` ``contained``
    mode, DESIGN 5.2): ``(outside, crossings)``.

    ``outside`` is the largest in-plane distance of a part boundary
    vertex beyond the anchor's outer boundary loop — 0.0 when every part
    boundary vertex is inside it. ``crossings`` counts part boundary
    edges that strictly cross the anchor's outer loop: a part region
    that leaves the contour between two interior vertices (a spanned
    notch of a non-convex anchor) crosses even though every vertex is
    inside, so the vertex distance alone would under-report it. ``tol``
    is the caller's rim allowance and the crossing test honors it: a
    part edge collinear with the rim within ``tol`` — the part seated
    flush with the anchor's edge — is containment, not a crossing.
    Holes in the anchor face are ignored — containment is against the
    outer extent (a blank flange may cover a bore). Both face sets must
    already carry a frame (``group_frame`` succeeded): planarity and the
    single-outer-loop shape are its guarantees, re-derived here from the
    same welded boundary.
    """
    n, s = frame.normal, frame.secondary
    t = np.cross(n, s)
    c = frame.origin

    def project(uniq):
        rel = uniq - c
        return np.column_stack([rel @ s, rel @ t])

    a_live = np.linalg.norm(tri_cross(anchor_tri), axis=1) > 0.0
    uniq_a, bnd_a = _welded_boundary(np.asarray(anchor_tri, dtype=np.float64), a_live)
    p_live = np.linalg.norm(tri_cross(part_tri), axis=1) > 0.0
    uniq_p, bnd_p = _welded_boundary(np.asarray(part_tri, dtype=np.float64), p_live)
    if not len(bnd_a) or not len(bnd_p):
        # Unreachable after group_frame: a planar face set has a boundary.
        return 0.0, 0
    pa = project(uniq_a)
    comp, _ = _loop_components(len(uniq_a), bnd_a)
    ea, eb = bnd_a[:, 0], bnd_a[:, 1]
    cr = pa[ea, 0] * pa[eb, 1] - pa[eb, 0] * pa[ea, 1]
    area2 = np.bincount(comp, weights=cr, minlength=int(comp.max()) + 1)
    outer = bnd_a[comp == int(np.argmax(area2))]
    A, B = pa[outer[:, 0]], pa[outer[:, 1]]

    pp = project(uniq_p)
    P = pp[np.unique(bnd_p)]
    inside = _points_in_loop(P, A, B)
    outside = 0.0
    if not inside.all():
        outside = float(_dist_points_segments(P[~inside], A, B).max())
    crossings = _proper_crossings(pp[bnd_p[:, 0]], pp[bnd_p[:, 1]], A, B, tol)
    return outside, crossings


def _rotation_about(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis, right-hand rule."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + s * k + (1.0 - c) * np.outer(axis, axis)


def solve(part: Frame, anchor: Frame, roll_deg: float = 0.0) -> np.ndarray:
    """The (4, 4) rigid transform that mates ``part`` onto ``anchor``:
    part normal → −anchor normal, part secondary → anchor secondary, the
    explicit roll about the anchor normal, origin onto origin."""
    n_p, s_p = part.normal, part.secondary
    n_a = -anchor.normal
    s_a = anchor.secondary
    f_p = np.column_stack([s_p, np.cross(n_p, s_p), n_p])
    f_a = np.column_stack([s_a, np.cross(n_a, s_a), n_a])
    rot = f_a @ f_p.T
    if roll_deg:
        rot = _rotation_about(anchor.normal, float(roll_deg)) @ rot
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = anchor.origin - rot @ part.origin
    return m
