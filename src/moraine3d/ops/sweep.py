"""``sweep`` operator.

Semantics: emits the solid of ``profile`` - a list of 3 to 4096 [x, y]
points (m) forming a closed polygon wound counter-clockwise, the last
point joining the first unrepeated - carried along ``path``, a list of
2 to 4096 [x, y, z] points (m). The profile's origin rides the path.
The first section stands upright: its +y is world +Y projected into
the plane across the first segment, +Z where that segment runs within
1 degree of vertical, and its +x is +y cross the segment's direction
(looking back down the path, +x is right and +y up). Each bend carries
the section by the smallest rotation that turns one segment onto the
next, so the section never twists about the path; turn the profile's
points to roll it. Bends are mitered: the ring at a bend lies in the
plane bisecting its two segments, every segment's cross-section is the
profile itself, and the volume is the profile's area times the length
its centroid travels. An open path is capped flat at both ends.
``closed`` (default false) joins the last path point to the first and
makes a loop without caps, accepted where the section carried once
around stands within 0.01 degrees of its start (every planar path).
The result is always watertight. A profile joint or a path bend
turning by more than ``angle_max_deg`` (default 30) is hard
(duplicated vertices), a gentler one smooth. A bend is the angle
between two consecutive segments' directions, 0 where the path runs
straight. Refused: a clockwise profile; coinciding consecutive points,
an exact double-back, a self-intersection, or no enclosed area in the
profile; coinciding consecutive path points, or a first and last one
that coincide; a bend beyond 150 degrees; a segment too short for the
miters at its ends; two segments not adjacent whose solids overlap or
touch; a profile too thin to cap; a section that does not survive
its vertices rounding to float32 (a triangle collapses or turns over,
two walls of the profile round onto each other, or the vertices weld
into an open surface); and a rings x profile vertices product beyond
the shared 4096 x 4096 cell budget. No UVs. Deterministic.
"""

from __future__ import annotations

import manifold3d
import numpy as np
import trimesh

from ..core import numbers, profile
from ..core.mating import secondary_axis
from ..core.mesh import Mesh, degenerate_tri_count, tri_cross, triangles_meet
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ("core.mating", "core.mesh", "core.profile", "manifold3d", "trimesh")
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "profile": numbers.FLOAT,
    "path": numbers.FLOAT,
    "closed": numbers.OTHER,
    "angle_max_deg": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceilings on tessellation: beyond them the count is a typo,
# and an unbounded one surfaces as a raw allocation failure, not a
# refusal.
PROFILE_MAX = 4096
PATH_MAX = 4096
# A miter reaches 1 / cos(bend / 2) times the section's own reach: 3.9
# times at this bend, and without bound as the bend nears 180.
BEND_MAX_DEG = 150.0
# How far the section may stand from its start after one turn of a
# closed path; the last segment takes up what is left.
RETURN_TOL_DEG = 0.01
ATTRIBUTE_POLICY = "preserved"  # source op


def _bad(param: str, message: str) -> OpError:
    return OpError("bad_param", message, param=param)


def _section(params: dict) -> list[tuple[float, float]]:
    value = params.get("profile")
    if value is None:
        raise _bad(
            "profile",
            "sweep requires 'profile' - a counter-clockwise list of [x, y] points (m)",
        )
    if not isinstance(value, list) or not 3 <= len(value) <= PROFILE_MAX:
        raise _bad(
            "profile", f"sweep 'profile' must be a list of 3..{PROFILE_MAX} [x, y] points"
        )
    pts: list[tuple[float, float]] = []
    for k, pt in enumerate(value):
        if not common.num_pair(pt):
            raise _bad("profile", f"sweep 'profile' point {k} must be [x, y] finite numbers")
        pts.append((float(pt[0]), float(pt[1])))
    k = profile.duplicate(pts, closed=True)
    if k == len(pts) - 1:
        raise _bad(
            "profile",
            "sweep 'profile' first and last points coincide - the profile "
            "closes by itself; drop the last point",
        )
    if k is not None:
        raise _bad(
            "profile",
            f"sweep 'profile' points {k} and {k + 1} coincide - remove the duplicate",
        )
    k = profile.double_back(pts, closed=True)
    if k is not None:
        raise _bad(
            "profile",
            f"sweep 'profile' doubles back exactly at point {k} - a "
            "zero-thickness fin; drop or move the point",
        )
    hit = profile.self_intersection(pts)
    if hit is not None:
        raise _bad(
            "profile",
            f"sweep 'profile' self-intersects - the segment from point {hit[0]} "
            f"crosses the segment from point {hit[1]}; the swept solid would be "
            "ill-defined",
        )
    area2 = profile.area2(pts)
    if area2 == 0:
        raise _bad("profile", "sweep 'profile' encloses no area - a degenerate solid")
    if area2 < 0:
        raise _bad(
            "profile",
            "sweep 'profile' runs clockwise - reverse the point order "
            "(counter-clockwise, seen with the path coming toward you)",
        )
    return pts


def _path(params: dict, closed: bool) -> np.ndarray:
    value = params.get("path")
    if value is None:
        raise _bad("path", "sweep requires 'path' - a list of [x, y, z] points (m)")
    least = 3 if closed else 2
    if not isinstance(value, list) or not least <= len(value) <= PATH_MAX:
        raise _bad(
            "path",
            f"sweep 'path' must be a list of {least}..{PATH_MAX} [x, y, z] points"
            + (" when closed" if closed else ""),
        )
    for k, pt in enumerate(value):
        if not (
            isinstance(pt, (list, tuple))
            and len(pt) == 3
            and all(common.is_num(c) for c in pt)
        ):
            raise _bad("path", f"sweep 'path' point {k} must be [x, y, z] finite numbers")
    pts = np.array(value, dtype=np.float64)
    same = np.flatnonzero(np.all(pts[1:] == pts[:-1], axis=1))
    if len(same):
        k = int(same[0])
        raise _bad(
            "path", f"sweep 'path' points {k} and {k + 1} coincide - remove the duplicate"
        )
    if np.array_equal(pts[0], pts[-1]):
        raise _bad(
            "path",
            "sweep 'path' first and last points coincide - drop the last point"
            + ("; closed joins them" if closed else " and set closed = True for a loop"),
        )
    return pts


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, row by row; exact near 0 and 180."""
    return np.degrees(
        np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1))
    )


def _carry(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``v`` under the smallest rotation that turns direction a onto b,
    kept square to b so the rounding of a long path does not add up."""
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s > 0.0:
        k = axis / s
        c = float(a @ b)
        v = v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c)
    v = v - float(v @ b) * b
    return v / np.linalg.norm(v)


def _lost(what: str) -> OpError:
    return _bad(
        "profile",
        f"sweep section does not survive rounding to float32: {what}; "
        "enlarge the profile, or build nearer the origin",
    )


# How many walls one pass of the triangle test sets against each other.
_WALL_PAIRS = 1 << 14


def _walls_meet(start: np.ndarray, end: np.ndarray, pairs: np.ndarray) -> bool:
    """Between the rings of s segments, (s, q, 3) each: do the walls of
    any of the given (p, 2) pairs of profile edges share a point? A wall
    is the two triangles the mesh carries for its edge."""
    q = start.shape[1]
    total = len(start) * len(pairs)
    for at in range(0, total, _WALL_PAIRS):
        s, p = np.divmod(np.arange(at, min(at + _WALL_PAIRS, total)), len(pairs))
        p = pairs[p]

        def wall(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u, w = edge, (edge + 1) % q
            au, aw = start[s, u].astype(np.float64), start[s, w].astype(np.float64)
            bu, bw = end[s, u].astype(np.float64), end[s, w].astype(np.float64)
            return np.stack([au, aw, bw], axis=1), np.stack([au, bw, bu], axis=1)

        for one in wall(p[:, 0]):
            for other in wall(p[:, 1]):
                if triangles_meet(one, other).any():
                    return True
    return False


def _segment_solid(start: np.ndarray, end: np.ndarray, cap: np.ndarray):
    """One segment of the sweep closed at both rings, as the kernel's
    solid."""
    q = len(start)
    i = np.arange(q)
    j = (i + 1) % q
    faces = np.concatenate([
        np.stack([i, j, q + j], axis=1),
        np.stack([i, q + j, q + i], axis=1),
        cap[:, ::-1],
        q + cap,
    ])
    solid = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.concatenate([start, end]).astype(np.float32),
            tri_verts=faces.astype(np.uint32),
        )
    )
    if solid.status() != manifold3d.Error.NoError:
        raise _lost("a segment's rings do not close a solid")
    return solid


# How many numbers one pass of a screen holds at once.
_SCREEN_CELLS = 1 << 22


def _folded(
    rings: np.ndarray, t: np.ndarray, plane: np.ndarray, cap: np.ndarray, closed: bool,
) -> tuple[int, int] | None:
    """The first two segments not adjacent whose solids overlap or
    touch, judged on the float32 rings the mesh carries. Screens leave
    the kernel only the pairs worth its exact answer: bounding boxes,
    then a gap along either segment's own direction - bounded from each
    ring's plane for every pair at once, then measured vertex by vertex
    for those left. The second clears a straight run however finely it
    is cut."""
    n, q = rings.shape[:2]
    m = n if closed else n - 1
    if m < 3:
        return None
    after = (np.arange(m) + 1) % n
    ends = np.stack([rings[:m], rings[after]])  # (2, m, q, 3)
    lo = ends.min(axis=(0, 2))
    hi = ends.max(axis=(0, 2))

    def near(rows: np.ndarray) -> np.ndarray:
        """Pairs (i, k), i among rows and k at least two on, whose
        bounding boxes meet - bar the pair a closed path makes adjacent."""
        meet = np.ones((len(rows), m), dtype=bool)
        for axis in range(3):
            meet &= lo[rows, None, axis] <= hi[None, :, axis]
            meet &= lo[None, :, axis] <= hi[rows, None, axis]
        meet &= np.arange(m)[None, :] >= rows[:, None] + 2
        if closed:
            meet[rows == 0, m - 1] = False
        found = np.argwhere(meet)
        found[:, 0] = rows[found[:, 0]]
        return found

    # Each segment's own extent along its direction, and a margin under
    # which a gap is the kernel's to judge, not a screen's.
    exact = rings.astype(np.float64)
    along = (ends.astype(np.float64) * t[None, :, None, :]).sum(axis=3)  # (2, m, q)
    own_lo = along.min(axis=(0, 2))
    own_hi = along.max(axis=(0, 2))
    reach = float(np.abs(rings).max())
    margin = 1e-9 * max(1.0, reach)

    # A ring lies in its plane but for rounding, so along a direction d
    # it spans its center give or take radius * sin(angle between d and
    # the plane's normal).
    center = exact.mean(axis=1)
    radius = np.linalg.norm(exact - center[:, None, :], axis=2).max(axis=1)
    slack = 2.0 * float(np.spacing(np.float32(reach)))

    def past(i: np.ndarray, k: np.ndarray, measured: bool) -> np.ndarray:
        """Does segment k's solid lie wholly past either end of segment
        i, along i's direction? Bounded from its two rings' planes, or
        measured at its vertices."""
        d = t[i]
        if measured:
            span = np.stack([exact[k], exact[after[k]]], axis=1)
            span = (span * d[:, None, None, :]).sum(axis=3).reshape(len(i), -1)
            low, high = span.min(axis=1), span.max(axis=1)
        else:
            low, high = np.full(len(i), np.inf), np.full(len(i), -np.inf)
            for ring in (k, after[k]):
                tilt = np.sqrt(np.clip(1.0 - (d * plane[ring]).sum(axis=1) ** 2, 0.0, None))
                mid = (d * center[ring]).sum(axis=1)
                spread = radius[ring] * tilt + slack
                low, high = np.minimum(low, mid - spread), np.maximum(high, mid + spread)
        return (low > own_hi[i] + margin) | (high < own_lo[i] - margin)

    rows = max(1, _SCREEN_CELLS // (16 * m))
    step = max(1, _SCREEN_CELLS // (6 * q))
    solids: dict[int, object] = {}
    for top in range(0, m - 2, rows):
        pairs = near(np.arange(top, min(top + rows, m - 2)))
        i, k = pairs.T
        pairs = pairs[~(past(i, k, False) | past(k, i, False))]
        for at in range(0, len(pairs), step):
            i, k = pairs[at:at + step].T
            close = ~(past(i, k, True) | past(k, i, True))
            for a, b in zip(i[close].tolist(), k[close].tolist()):
                for seg in (a, b):
                    if seg not in solids:
                        solids[seg] = _segment_solid(rings[seg], rings[after[seg]], cap)
                if solids[a].min_gap(solids[b], 1.0) <= 0.0:
                    return a, b
    return None


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    closed = common.boolean("sweep", params, "closed", False)
    section = _section(params)
    way = _path(params, closed)
    angle_max = common.num(
        "sweep", params, "angle_max_deg",
        common.CREASE_ANGLE_DEG_DEFAULT, positive=True, maximum=180.0,
    )
    n = len(way)
    m = n if closed else n - 1
    q = len(section)

    # Segment i runs from point i to point i + 1. Joint j is the bend at
    # point j, between its incoming segment and its outgoing one; the
    # ends of an open path are no joints.
    t = _unit(way[(np.arange(m) + 1) % n] - way[:m])
    joints = np.arange(n) if closed else np.arange(1, n - 1)
    incoming = (joints - 1) % m
    bend = _angle_deg(t[incoming], t[joints])
    sharp = np.flatnonzero(bend > BEND_MAX_DEG)
    if len(sharp):
        j = int(joints[sharp[0]])
        raise _bad(
            "path",
            f"sweep 'path' bends {bend[sharp[0]]:.1f} degrees at point {j}, beyond "
            f"{BEND_MAX_DEG:g} - a miter that sharp runs away; add a point to "
            "split the bend",
        )

    # Section axes per segment: upright at the first, carried on.
    ys = [secondary_axis(t[0])]
    for i in range(1, m):
        ys.append(_carry(ys[-1], t[i - 1], t[i]))
    if closed:
        back = _carry(ys[-1], t[m - 1], t[0])
        turned = float(_angle_deg(back, ys[0]))
        if turned > RETURN_TOL_DEG:
            raise _bad(
                "path",
                f"sweep 'path' brings the section back turned {turned:.3g} degrees "
                "- a closed path must return it to its start, as a planar one "
                "does; leave the path open, or flatten it",
            )
    y = np.array(ys)
    x = np.cross(y, t)

    # Ring vertices: one per normal a profile point carries.
    carried = profile.corners(section, angle_max, closed=True)
    held = np.array([len(c) for c in carried])
    first = np.concatenate([[0], np.cumsum(held)[:-1]])
    vert_point = np.repeat(np.arange(q), held)
    normal2 = np.array([nrm for c in carried for nrm in c])
    v = len(vert_point)

    # Ring copies along the path: a hard joint has two, one per segment.
    hard = np.zeros(n, dtype=bool)
    hard[joints] = bend > angle_max
    copies = 1 + hard.astype(np.int64)
    c_first = np.concatenate([[0], np.cumsum(copies)[:-1]])
    c_last = c_first + copies - 1
    c = int(copies.sum())
    if c * v > common.TESSELLATION_CELLS_MAX:
        raise _bad(
            "path",
            f"sweep tessellation is too large: {c} rings x {v} profile vertices "
            f"exceeds the {common.TESSELLATION_CELLS_MAX} cell budget; reduce path "
            "points, profile points, or hard joints",
        )

    # Ring j: the profile laid in the outgoing segment's frame and slid
    # along that segment into the plane bisecting the bend. The ends of
    # an open path stay square to their one segment.
    out = np.arange(n)
    into = (out - 1) % m
    if not closed:
        out[-1] = m - 1
        into[0] = 0
    sec = np.array(section)
    offset = sec[None, :, :1] * x[:, None, :] + sec[None, :, 1:] * y[:, None, :]  # (m, q, 3)
    plane = _unit(t[into] + t[out])
    slide = -(offset[out] @ plane[:, :, None])[..., 0] / (t[out] * plane).sum(axis=1)[:, None]
    if not closed:
        slide[[0, -1]] = 0.0
    rings = way[:, None, :] + offset[out] + slide[..., None] * t[out][:, None, :]  # (n, q, 3)

    run = ((rings[(np.arange(m) + 1) % n] - rings[:m]) * t[:, None, :]).sum(axis=2)
    short = np.flatnonzero((run <= 0.0).any(axis=1))
    if len(short):
        raise _bad(
            "path",
            f"sweep 'path' segment from point {int(short[0])} is too short for "
            "the miters at its ends - the section would pass through itself; "
            "lengthen it, ease its bends, or keep the profile off the inside "
            "of the bend",
        )

    cap = profile.triangulate(section)
    if cap is None:
        raise _bad(
            "profile",
            "sweep 'profile' is too thin to cap - its triangles would overlap; "
            "widen it",
        )

    # Side vertices, copy by copy: positions from the ring, normals from
    # the one segment of a hard copy or both segments of a smooth one.
    copy_point = np.repeat(np.arange(n), copies)
    seg_a = np.repeat(into, copies)
    seg_b = np.repeat(out, copies)
    seg_b[c_first[hard]] = into[hard]
    seg_a[c_last[hard]] = out[hard]
    side = normal2[None, :, :1] * x[:, None, :] + normal2[None, :, 1:] * y[:, None, :]  # (m, v, 3)
    normals = np.where(
        (seg_a == seg_b)[:, None, None], side[seg_a], _unit(side[seg_a] + side[seg_b])
    )
    verts = [rings[copy_point][:, vert_point, :].reshape(-1, 3)]
    norms = [normals.reshape(-1, 3)]

    # Walls: segment i joins the last copy at its start to the first at
    # its end, and profile edge k the last vertex of point k to the
    # first of the next - (A_u, A_w, B_w), (A_u, B_w, B_u) faces out.
    a = (c_last[:m] * v)[:, None]
    b = (c_first[(np.arange(m) + 1) % n] * v)[:, None]
    u = (first + held - 1)[None, :]
    w = first[(np.arange(q) + 1) % q][None, :]
    faces = [
        np.stack([a + u, a + w, b + w], axis=2).reshape(-1, 3),
        np.stack([a + u, b + w, b + u], axis=2).reshape(-1, 3),
    ]
    if not closed:
        base = c * v
        verts += [rings[0], rings[-1]]
        norms += [np.tile(-t[0], (q, 1)), np.tile(t[-1], (q, 1))]
        faces += [base + cap[:, ::-1], base + q + cap]

    exact = np.concatenate(verts)
    vertices = exact.astype(np.float32)
    tris = np.concatenate(faces)
    facing = (tri_cross(exact[tris]) * tri_cross(vertices.astype(np.float64)[tris])).sum(axis=1)
    lost = int((facing <= 0.0).sum()) or degenerate_tri_count(vertices, tris)
    if lost:
        raise _lost(f"{lost} triangle(s) collapse or turn over")
    # The weld `validate` judges a surface by: vertices it merges must
    # leave every edge to two faces.
    welded = trimesh.Trimesh(vertices=vertices, faces=tris, process=False)
    welded.merge_vertices()
    if not welded.is_watertight:
        raise _lost("its vertices weld into an open surface")

    # Rounding moves a vertex by half a float32 step along each axis.
    # Seen along a segment, where its walls run, that moves the section
    # by the part of each step square to the segment. Only walls that
    # stand nearer than two such moves can meet, and there the rounded
    # walls themselves decide: the triangles between the two rings,
    # which can cross between rings that each look sound.
    stepped = rings.astype(np.float32)
    nxt = (np.arange(m) + 1) % n
    half = 0.5 * np.spacing(np.abs(stepped)).astype(np.float64)  # (n, q, 3)
    square = np.sqrt(np.clip(1.0 - t ** 2, 0.0, None))  # (m, 3)
    moved = np.maximum(
        (half[:m] * square[:, None, :]).sum(axis=2).max(axis=1),
        (half[nxt] * square[:, None, :]).sum(axis=2).max(axis=1),
    )
    suspect = np.flatnonzero(profile.clearance(section) <= 2.0 * moved)
    if len(suspect):
        for pairs in profile.close_edges(section, 2.0 * float(moved[suspect].max())):
            if _walls_meet(stepped[suspect], stepped[nxt[suspect]], pairs):
                raise _lost("walls of the profile round onto each other")

    fold = _folded(stepped, t, plane, cap, closed)
    if fold is not None:
        raise _bad(
            "path",
            f"sweep 'path' folds the solid onto itself: the segments from points "
            f"{fold[0]} and {fold[1]} overlap or touch; move them apart",
        )

    return Mesh(
        vertices=vertices,
        faces=tris.astype(np.int32),
        normals=np.concatenate(norms).astype(np.float32),
    )
