"""``revolve`` operator.

Semantics: emits the solid of revolution of ``profile`` - a list of two
or more [radius, y] points (m) forming an open polyline in the +X
half-plane - turned around the +Y axis through ``arc_deg`` degrees
(default 360, above 0 and at most 360) in ``segments`` steps
(default 32, 3..4096).
The turn starts at angle 0 (+X) and runs toward -Z, and below 360
both cut faces are closed with flat caps. ``segments`` counts the
steps across the arc, so a half turn of 16 has the facets of a full
turn of 32, and at equal ``segments`` the volume is not proportional
to the arc. Order the profile so the
solid lies on its left (bottom-to-top along the outside); a profile
that would face the surface inward is refused naming the fix (reverse
the point order). Ends close automatically: an end with radius 0
becomes a pole vertex, an end with radius > 0 a flat horizontal cap
(duplicated ring, +-Y normals, center-vertex fan) - the result is
always watertight. Radii must be >= 0, with 0 only at the ends;
consecutive duplicates, identical endpoints, interior zero radii,
exact double-backs, self-intersections (the closure to the axis
included), and profiles enclosing no area are refused, as is a
rings x columns product beyond the shared 4096 x 4096 cell budget (a
column per step, and one more below 360; hard joints duplicate rings,
so the per-param ceilings alone do not bound it).
``closed`` (default false) joins the last point to the first instead:
three or more points, every radius above 0, wound counter-clockwise
around the section, which the turn carries around the axis as a ring
with no end caps. The last point does not repeat the first.
Circumferential shading is smooth (shared ring vertices); a profile
joint bending by more than ``angle_max_deg`` (default 30) is hard
(duplicated rings), a gentler one smooth (shared ring, half-plane
bisector normal). Vertex 0 of each ring sits at angle 0 (+X); winding
is CCW from outside. A turn of less than 360 degrees is refused where
the profile is too thin to cap. That turn, and any turn of a
``closed`` profile, is refused where the profile does not survive its
vertices rounding to float32 (the profile turns over or flattens at a
column, a triangle collapses or turns over, the vertices weld into an
open surface, or two walls round onto each other); the full turn of an
open profile takes neither check.
Deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from ..core import numbers, profile
from ..core.mesh import Mesh
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
# A radius this far below zero is a point an expression put on the axis
# (an interpolation's end): a nanometer, under what a float32 vertex
# resolves at any size the tool builds. It is read as 0. Nothing that
# was accepted before reads differently, so the version stands.
RADIUS_NOISE = 1e-9
KERNELS = ("core.mesh", "core.profile", "manifold3d", "trimesh")
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "profile": numbers.FLOAT,
    "segments": numbers.INT,
    "angle_max_deg": numbers.FLOAT,
    "arc_deg": numbers.FLOAT,
    "closed": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceilings on tessellation: beyond them the count is a typo,
# and an unbounded one surfaces as a raw allocation failure, not a
# refusal.
SEGMENTS_MAX = 4096
PROFILE_MAX = 4096
ATTRIBUTE_POLICY = "preserved"  # source op


def _bad(message: str) -> OpError:
    return OpError("bad_param", message, param="profile")


def _lost(what: str, closed: bool) -> OpError:
    where = "" if closed else " at this arc_deg"
    return _bad(
        f"revolve 'profile' does not survive rounding to float32{where}: "
        f"{what}; enlarge the profile, or draw it nearer the origin"
    )


def _generating(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The generating polygon: the profile closed to the axis at the
    end heights (a pole end already sits on the axis)."""
    poly = list(pts)
    if pts[-1][0] > 0:
        poly.append((0.0, pts[-1][1]))
    if pts[0][0] > 0 and (0.0, pts[0][1]) != poly[-1]:
        poly.append((0.0, pts[0][1]))
    return poly


def _profile(params: dict, closed: bool) -> list[tuple[float, float]]:
    value = params.get("profile")
    if value is None:
        raise _bad("revolve requires 'profile' - a list of [radius, y] points (m)")
    least = 3 if closed else 2
    if not isinstance(value, list) or not least <= len(value) <= PROFILE_MAX:
        raise _bad(
            f"revolve 'profile' must be a list of {least}..{PROFILE_MAX} [radius, y] points"
            + (" when closed" if closed else "")
        )
    pts: list[tuple[float, float]] = []
    for k, pt in enumerate(value):
        if not common.num_pair(pt):
            raise _bad(
                f"revolve 'profile' point {k} must be [radius, y] finite numbers"
            )
        r, y = float(pt[0]), float(pt[1])
        if -RADIUS_NOISE <= r < 0:
            r = 0.0  # a point computed onto the axis, short by rounding
        if r < 0:
            raise _bad(f"revolve 'profile' point {k} has radius < 0")
        if r == 0 and 0 < k < len(value) - 1 and not closed:
            raise _bad(
                f"revolve 'profile' point {k} has radius 0 - the surface would "
                "pinch at the axis; radius 0 is allowed only at the ends"
            )
        pts.append((r, y))
    k = profile.duplicate(pts)
    if k is not None:
        raise _bad(
            f"revolve 'profile' points {k} and {k + 1} coincide - remove "
            "the duplicate"
        )
    if pts[0] == pts[-1]:
        # The fix depends on where the repeated point sits: on the axis
        # the ends close to it, clear of it either closure can be meant.
        if pts[0][0] == 0:
            fix = (
                "drop the last point and remove closed; the ends close to the axis"
                if closed else "drop the repeated last point; the ends close to the axis"
            )
        elif closed:
            fix = "drop the last point; closed joins the ends"
        else:
            fix = (
                "drop the last point; closed = True then makes a ring, and "
                "without it the ends close to the axis"
            )
        raise _bad(f"revolve 'profile' first and last points coincide - {fix}")
    if closed:
        on_axis = [k for k, (r, _) in enumerate(pts) if r == 0]
        if on_axis:
            raise _bad(
                f"revolve 'profile' point {on_axis[0]} has radius 0 - a closed "
                "profile makes a ring clear of the axis; for a solid that reaches "
                "the axis remove closed and end the profile there"
            )
    k = profile.double_back(pts, closed=closed)
    if k is not None:
        raise _bad(
            f"revolve 'profile' doubles back exactly at point {k} - "
            "a zero-thickness fin; drop or move the point"
        )
    # The generating polygon's edges must not cross or touch - a
    # self-intersecting polygon sweeps an ill-defined solid that only
    # looks watertight.
    hit = profile.self_intersection(pts if closed else _generating(pts))
    if hit is not None:
        last = "the closing segment" if closed else "the end closure"
        names = [
            f"the segment from point {e}" if e < len(pts) - 1 else last
            for e in hit
        ]
        raise _bad(
            f"revolve 'profile' self-intersects - {names[0]} crosses "
            f"{names[1]}; the swept solid would be ill-defined"
        )
    if closed:
        area2 = profile.area2(pts)
        if area2 == 0:
            raise _bad("revolve 'profile' encloses no area - a degenerate solid")
        if area2 < 0:
            raise _bad(
                "revolve 'profile' would face the surface inward - reverse the "
                "point order (counter-clockwise around the section)"
            )
        return pts
    # Orientation gate: the generating polygon must have positive signed
    # area, so the authored winding is outward by construction. The last
    # term is the closure through the axis, from the last point across
    # to its height on the axis, along it, and out to the first.
    area2 = sum(
        pts[k][0] * pts[k + 1][1] - pts[k + 1][0] * pts[k][1]
        for k in range(len(pts) - 1)
    )
    area2 += pts[-1][0] * pts[-1][1] - pts[0][0] * pts[0][1]
    if area2 == 0:
        raise _bad("revolve 'profile' encloses no area - a degenerate solid")
    if area2 < 0:
        raise _bad(
            "revolve 'profile' would face the surface inward - reverse the "
            "point order (bottom-to-top along the outside)"
        )
    return pts


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    closed = common.boolean("revolve", params, "closed", False)
    pts = _profile(params, closed)
    seg = common.integer("revolve", params, "segments", 32, minimum=3, maximum=SEGMENTS_MAX)
    angle_max = common.num(
        "revolve", params, "angle_max_deg",
        common.CREASE_ANGLE_DEG_DEFAULT, positive=True, maximum=180.0,
    )
    arc = common.num("revolve", params, "arc_deg", 360.0, positive=True, maximum=360.0)
    # A full turn joins its last column to its first; a part turn has
    # one column more and two cut faces.
    full = arc == 360.0
    cols = seg if full else seg + 1

    n_pts = len(pts)
    # Rings: (radius, y, half-plane normal), one per normal a point
    # carries - a hard joint has two. A band runs from the last ring of
    # its lower point to the first ring of its upper one; a pole end has
    # no ring. A closed profile has one band more, from its last point
    # back to its first.
    bands = n_pts if closed else n_pts - 1
    pole_start = pts[0][0] == 0
    pole_end = pts[-1][0] == 0
    rings: list[tuple[float, float, tuple[float, float]]] = []
    band_lo = [0] * bands  # ring index at pts[k] for band k
    band_hi = [0] * bands  # ring index at the next point for band k
    for k, carried in enumerate(profile.corners(pts, angle_max, closed=closed)):
        if not carried:
            raise _bad(
                f"revolve 'profile' doubles back at point {k} - a "
                "zero-thickness fin; drop or move the point"
            )
        if (k == 0 and pole_start) or (k == n_pts - 1 and pole_end):
            continue
        if k > 0 or closed:
            band_hi[k - 1] = len(rings)
        rings.extend((*pts[k], normal) for normal in carried)
        if k < bands:
            band_lo[k] = len(rings) - 1

    # Hard joints duplicate rings, so the rings x columns product can
    # exceed what the per-param ceilings admit alone - refuse before
    # allocating, per the ceilings' own rationale.
    if len(rings) * cols > common.TESSELLATION_CELLS_MAX:
        raise OpError(
            "bad_param",
            f"revolve tessellation is too large: {len(rings)} rings x "
            f"{cols} columns exceeds the {common.TESSELLATION_CELLS_MAX} "
            "cell budget; reduce segments, profile points, or hard joints",
            param="segments",
        )

    if full:
        ang = np.arange(seg) * (2.0 * np.pi / seg)
    else:
        ang = np.arange(cols) * (math.radians(arc) / seg)
    cos, sin = np.cos(ang), np.sin(ang)
    # +Z runs against the turn. Subtracting from zero keeps the column
    # on the +X axis at z = 0.0 where negating writes -0.0, which the
    # full turn of an open profile has always carried.
    back = -sin if full and not closed else 0.0 - sin
    i = np.arange(seg)
    j = (i + 1) % cols

    def ring_xyz(r: float, y: float) -> np.ndarray:
        return np.stack([r * cos, np.full(cols, y), r * back], axis=1)

    verts = [ring_xyz(r, y) for r, y, _ in rings]
    normals = [
        np.stack([nr * cos, np.full(cols, ny), nr * back], axis=1)
        for _, _, (nr, ny) in rings
    ]
    faces: list[np.ndarray] = []

    def emit(v: np.ndarray, n: np.ndarray) -> int:
        """Append a vertex/normal block; return its base index."""
        base = sum(len(block) for block in verts)
        verts.append(v)
        normals.append(n)
        return base

    # Side bands: ring A at pts[k], ring B at pts[k + 1] - the cylinder
    # split (A_i, A_j, B_j), (A_i, B_j, B_i) is CCW-outward for every
    # band direction; a pole collapses one triangle of each quad.
    for k in range(bands):
        if k == 0 and pole_start:
            p = emit(np.array([[0.0, pts[0][1], 0.0]]), np.array([[0.0, -1.0, 0.0]]))
            b = band_hi[0] * cols
            faces.append(np.stack([np.full(seg, p), b + j, b + i], axis=1))
        elif k == n_pts - 2 and pole_end:
            p = emit(np.array([[0.0, pts[-1][1], 0.0]]), np.array([[0.0, 1.0, 0.0]]))
            a = band_lo[k] * cols
            faces.append(np.stack([a + i, a + j, np.full(seg, p)], axis=1))
        else:
            a, b = band_lo[k] * cols, band_hi[k] * cols
            faces.append(np.stack([a + i, a + j, b + j], axis=1))
            faces.append(np.stack([a + i, b + j, b + i], axis=1))

    # Caps: the closure path runs axis-out into pts[0] and pts[-1]-in to
    # the axis, so cap normals are -Y (start) / +Y (end) and winding
    # follows the sign - hard by construction (duplicated ring +
    # center-vertex fan, the cylinder cap technique).
    for (r, y), ny, pole in ((pts[0], -1.0, pole_start), (pts[-1], 1.0, pole_end)):
        if pole or closed:
            continue
        base = emit(
            np.concatenate([ring_xyz(r, y), [[0.0, y, 0.0]]]),
            np.tile([0.0, ny, 0.0], (cols + 1, 1)),
        )
        fan = [base + i, base + j, np.full(seg, base + cols)]
        faces.append(np.stack(fan if ny > 0 else fan[::-1], axis=1))

    if full and not closed:
        return Mesh(
            vertices=np.concatenate(verts).astype(np.float32),
            faces=np.concatenate(faces).astype(np.int32),
            normals=np.concatenate(normals).astype(np.float32),
        )

    poly = pts if closed else _generating(pts)
    flat = np.array(poly)
    # Adding zero leaves a point on the axis at 0.0, not at the -0.0
    # its radius times a negative cosine comes to.
    stations = np.stack(
        [flat[None, :, 0] * cos[:, None] + 0.0, np.broadcast_to(flat[:, 1], (cols, len(poly))),
         flat[None, :, 0] * back[:, None] + 0.0],
        axis=2,
    )  # (cols, q, 3)
    if not full:
        # Cut faces: the generating polygon in the half-planes of the
        # first and the last column. Counter-clockwise in [radius, y],
        # it faces back along the turn - outward at the first column,
        # so the last takes the triangles reversed.
        cap = profile.triangulate(poly)
        if cap is None:
            raise _bad(
                "revolve 'profile' is too thin to cap where arc_deg cuts it - its "
                "triangles would overlap; widen it"
            )
        for col, outward, tris in ((0, 1.0, cap), (seg, -1.0, cap[:, ::-1])):
            base = emit(
                stations[col],
                np.tile([outward * sin[col], 0.0, outward * cos[col]], (len(poly), 1)),
            )
            faces.append(base + tris)

    turned = profile.rounded_outlines_turned(stations)
    if turned:
        raise _lost(f"it turns over or flattens at {turned} column(s)", closed)
    exact = np.concatenate(verts)
    vertices = exact.astype(np.float32)
    tris = np.concatenate(faces)
    lost = profile.rounded_faces_lost(exact, tris)
    if lost:
        raise _lost(f"{lost} triangle(s) collapse or turn over", closed)
    # The weld `validate` judges a surface by: vertices it merges must
    # leave every edge to two faces.
    welded = trimesh.Trimesh(vertices=vertices, faces=tris, process=False)
    welded.merge_vertices()
    if not welded.is_watertight:
        raise _lost("its vertices weld into an open surface", closed)
    # Between two columns every wall runs along the chord, the tangent
    # at the angle halfway. Each column stands half a step askew of it,
    # so end on the chord shows the polygon that much narrower.
    half = 0.5 * math.radians(arc) / seg
    mid = ang[:seg] + half
    chord = np.stack([-np.sin(mid), np.zeros(seg), -np.cos(mid)], axis=1)
    narrow = math.cos(half)
    if profile.rounded_walls_meet(stations, chord, [(r * narrow, y) for r, y in poly]):
        raise _lost("two walls round onto each other", closed)

    return Mesh(
        vertices=vertices,
        faces=tris.astype(np.int32),
        normals=np.concatenate(normals).astype(np.float32),
    )
