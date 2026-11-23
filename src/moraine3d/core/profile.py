"""2D profile geometry shared by the operators that take an inline
profile (DESIGN 5).

A profile is a list of ``(a, b)`` float pairs. The functions answer
with facts (an index, a sign, a list of normals) and never raise for
the shape of a profile: the operator owns the wording of a refusal and
the fix it names. ``closed`` reads the list as a loop, its last point
joined to its first. The last functions follow a profile into the
solid an operator builds from it, where float32 has rounded it.
"""

from __future__ import annotations

import math

import manifold3d
import numpy as np

from .mesh import degenerate_tri_count, tri_cross, triangles_meet

# Shared-kernel version (DESIGN 4.5): ops that consume this module declare
# it in KERNELS and this integer joins their cache keys. Bump on ANY
# output-semantics change of a function an op consumes, bug fixes
# included - like OP_IMPL_VERSION.
KERNEL_VERSION = 2

Point = tuple[float, float]


def cross2(u, v):
    """z-component of the 2D cross product, broadcast over leading axes
    (numpy dropped 2D support from np.cross)."""
    u = np.asarray(u)
    v = np.asarray(v)
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _on_segment(p1, p2, q):
    """Given q collinear with segment [p1, p2], is q within its bounds?
    Vectorized over the leading axis of any argument."""
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    return np.all((lo <= q) & (q <= hi), axis=-1)


def self_intersection(poly: list[Point]) -> tuple[int, int] | None:
    """First pair of non-adjacent edges of the closed polygon that cross
    or touch (exact arithmetic; a shared vertex between non-adjacent
    edges is a pinch and counts). None when the polygon is simple."""
    n = len(poly)
    if n < 4:
        return None
    pts = np.array(poly, dtype=float)
    a = pts
    b = np.vstack([pts[1:], pts[:1]])
    for k in range(n - 2):
        m0 = k + 2
        m1 = n - 1 if k == 0 else n  # the wrap edge is adjacent to edge 0
        if m0 >= m1:
            continue
        p1, p2 = a[k], b[k]
        q1, q2 = a[m0:m1], b[m0:m1]
        r = p2 - p1
        s = q2 - q1
        d1 = cross2(r, q1 - p1)
        d2 = cross2(r, q2 - p1)
        d3 = cross2(s, p1 - q1)
        d4 = cross2(s, p2 - q1)
        bad = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
        bad |= (d1 == 0) & _on_segment(p1, p2, q1)
        bad |= (d2 == 0) & _on_segment(p1, p2, q2)
        bad |= (d3 == 0) & _on_segment(q1, q2, p1)
        bad |= (d4 == 0) & _on_segment(q1, q2, p2)
        if bad.any():
            return k, m0 + int(np.argmax(bad))
    return None


def _steps(n: int, closed: bool) -> int:
    """How many segments a list of n points has."""
    return n if closed else n - 1


def duplicate(pts: list[Point], *, closed: bool = False) -> int | None:
    """Index k of the first point that coincides with the next one."""
    n = len(pts)
    for k in range(_steps(n, closed)):
        if pts[k] == pts[(k + 1) % n]:
            return k
    return None


def double_back(pts: list[Point], *, closed: bool = False) -> int | None:
    """Index of the first point where the outline turns exactly back on
    itself: a zero-thickness fin, and the one joint whose bisector
    normal is undefined."""
    n = len(pts)
    for k in range(n if closed else n - 2):
        p0, p1, p2 = pts[k], pts[(k + 1) % n], pts[(k + 2) % n]
        d0 = (p1[0] - p0[0], p1[1] - p0[1])
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        if d0[0] * d1[1] - d0[1] * d1[0] == 0 and d0[0] * d1[0] + d0[1] * d1[1] < 0:
            return (k + 1) % n
    return None


def area2(poly: list[Point]) -> float:
    """Twice the signed area of the closed polygon: positive when its
    points run counter-clockwise, whichever point the list starts at.
    Measured from the first point, so a small polygon far from the
    origin keeps its area."""
    a0, b0 = poly[0]
    rel = [(a - a0, b - b0) for a, b in poly]
    return sum(
        rel[k][0] * rel[k + 1][1] - rel[k + 1][0] * rel[k][1]
        for k in range(1, len(rel) - 1)
    )


def _to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of p to its segment [a, b], row by row."""
    ab = b - a
    along = ((p - a) * ab).sum(axis=-1) / (ab * ab).sum(axis=-1)
    return np.linalg.norm(p - (a + np.clip(along, 0.0, 1.0)[..., None] * ab), axis=-1)


def _edge_gaps(poly: list[Point]):
    """Per edge k of the simple closed polygon: the first later edge
    that shares no point with it, and its distance to that edge and
    each one after."""
    n = len(poly)
    pts = np.array(poly, dtype=np.float64)
    pts = pts - pts[0]
    a = pts
    b = np.vstack([pts[1:], pts[:1]])
    for k in range(n - 2):
        m0 = k + 2
        m1 = n - 1 if k == 0 else n  # the wrap edge is adjacent to edge 0
        if m0 >= m1:
            continue
        p1 = np.broadcast_to(a[k], (m1 - m0, 2))
        p2 = np.broadcast_to(b[k], (m1 - m0, 2))
        q1, q2 = a[m0:m1], b[m0:m1]
        # Edges that do not cross are nearest at an end of one of them.
        yield k, m0, np.minimum.reduce([
            _to_segments(p1, q1, q2), _to_segments(p2, q1, q2),
            _to_segments(q1, p1, p2), _to_segments(q2, p1, p2),
        ])


def clearance(poly: list[Point]) -> float:
    """The least distance between two edges of the simple closed
    polygon that share no point: how close its outline comes to
    itself. Infinite for a triangle, whose edges all meet."""
    return min((float(gaps.min()) for _, _, gaps in _edge_gaps(poly)), default=math.inf)


def close_edges(poly: list[Point], within: float):
    """The pairs of edges of the simple closed polygon that share no
    point and stand no farther apart than ``within``, as (p, 2) blocks:
    one per edge that has any, holding its pairs with later edges. Edge
    k runs from point k to the next."""
    for k, m0, gaps in _edge_gaps(poly):
        at = np.flatnonzero(gaps <= within)
        if len(at):
            yield np.stack([np.full(len(at), k), m0 + at], axis=1)


# How many walls one pass of the triangle test sets against each other.
_WALL_PAIRS = 1 << 14


def _walls_meet(start: np.ndarray, end: np.ndarray, pairs: np.ndarray) -> bool:
    """Between the outlines at the two ends of s spans, (s, q, 3) each:
    do the walls of any of the given (p, 2) pairs of edges share a
    point? A wall is the two triangles a mesh carries for its edge, and
    one collapsed onto a line, as at a point on an axis, carries no
    face and is left out."""
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
                live = tri_cross(one).any(axis=1) & tri_cross(other).any(axis=1)
                if live.any() and triangles_meet(one[live], other[live]).any():
                    return True
    return False


def rounded_walls_meet(stations: np.ndarray, t: np.ndarray, seen: list[Point]) -> bool:
    """Do two walls of a solid land on each other when its vertices
    round to float32? ``stations`` (n, q, 3) holds an outline as the
    solid carries it at n places, unrounded, and ``t`` (m, 3) the unit
    direction span i runs, from station i to the next: m is n where the
    last span returns to the first station, n - 1 otherwise. ``seen``
    is the outline as every span shows it end on, the same q points:
    the section itself where the stations stand square to their spans
    or mitered between them, and narrower where they stand askew.

    Rounding moves a vertex by half a float32 step along each axis.
    Seen along a span, where its walls run, that moves the outline by
    the part of each step square to the span. Only walls that stand
    nearer than two such moves in that view can meet, and there the
    rounded walls themselves decide: the triangles between two
    stations, which can cross between stations that each look sound."""
    n, m = len(stations), len(t)
    stepped = stations.astype(np.float32)
    nxt = (np.arange(m) + 1) % n
    half = 0.5 * np.spacing(np.abs(stepped)).astype(np.float64)  # (n, q, 3)
    square = np.sqrt(np.clip(1.0 - t ** 2, 0.0, None))  # (m, 3)
    moved = np.maximum(
        (half[:m] * square[:, None, :]).sum(axis=2).max(axis=1),
        (half[nxt] * square[:, None, :]).sum(axis=2).max(axis=1),
    )
    suspect = np.flatnonzero(clearance(seen) <= 2.0 * moved)
    if len(suspect):
        for pairs in close_edges(seen, 2.0 * float(moved[suspect].max())):
            if _walls_meet(stepped[suspect], stepped[nxt[suspect]], pairs):
                return True
    return False


# How many faces one pass of the rounding check reads the corners of.
_FACES = 1 << 13


def rounded_faces_lost(exact: np.ndarray, faces: np.ndarray) -> int:
    """How many triangles of a solid do not survive its vertices
    rounding to float32: those that turn over or lose all area against
    the unrounded ``exact`` (v, 3) or, with none of those, those the
    rounded mesh carries as degenerate."""
    turned = thin = 0
    for at in range(0, len(faces), _FACES):
        was = exact[faces[at:at + _FACES]]
        now = was.astype(np.float32).astype(np.float64)
        turned += int(((tri_cross(was) * tri_cross(now)).sum(axis=1) <= 0.0).sum())
        thin += degenerate_tri_count(now.reshape(-1, 3), np.arange(now.size // 3).reshape(-1, 3))
    return turned or thin


def _area(outlines: np.ndarray) -> np.ndarray:
    """Twice the area vector of each planar outline, (s, q, 3) to
    (s, 3), measured from its first point."""
    rel = outlines[:, 1:] - outlines[:, :1]
    return np.cross(rel[:, :-1], rel[:, 1:]).sum(axis=1)


def rounded_outlines_turned(stations: np.ndarray) -> int:
    """How many of the planar outlines a solid carries, (n, q, 3)
    unrounded, turn over or lose all area when their vertices round to
    float32. Walls between outlines that have turned over each keep
    their side, weld shut, and need not meet: they bound the solid
    inside out."""
    rows = max(1, _FACES // stations.shape[1])
    turned = 0
    for at in range(0, len(stations), rows):
        was = stations[at:at + rows]
        now = was.astype(np.float32).astype(np.float64)
        turned += int(((_area(was) * _area(now)).sum(axis=1) <= 0.0).sum())
    return turned


def corners(
    pts: list[Point], angle_max_deg: float, *, closed: bool = False,
) -> list[list[Point]]:
    """Per point, the outward unit normals its vertices carry, in
    outline order. A joint bending by at most ``angle_max_deg`` is
    smooth: one normal, the bisector of its two segments' normals. A
    sharper joint is hard: two, the incoming segment's then the
    outgoing one's. An end of an open outline carries its one segment's
    normal. A segment running (da, db) has the normal (db, -da), outward
    for a counter-clockwise outline. A smooth joint whose two normals
    cancel has no bisector and carries none: the outline doubles back
    there, by less than ``double_back`` resolves."""
    n = len(pts)
    dirs: list[Point] = []
    normals: list[Point] = []
    for k in range(_steps(n, closed)):
        da = pts[(k + 1) % n][0] - pts[k][0]
        db = pts[(k + 1) % n][1] - pts[k][1]
        length = math.hypot(da, db)
        dirs.append((da / length, db / length))
        normals.append((db / length, -da / length))

    def joint(before: int, after: int) -> list[Point]:
        d0, d1 = dirs[before], dirs[after]
        bend = math.degrees(math.acos(max(-1.0, min(1.0, d0[0] * d1[0] + d0[1] * d1[1]))))
        if bend > angle_max_deg:
            return [normals[before], normals[after]]
        na, nb = normals[before][0] + normals[after][0], normals[before][1] + normals[after][1]
        length = math.hypot(na, nb)
        return [(na / length, nb / length)] if length else []

    if closed:
        return [joint(k - 1, k) for k in range(n)]
    return [[normals[0]], *(joint(k - 1, k) for k in range(1, n - 1)), [normals[-1]]]


def valid_cap(poly: list[Point], tris: np.ndarray) -> bool:
    """Do the triangles tile the counter-clockwise polygon exactly once?
    They do when there are n - 2 of them, each counter-clockwise, every
    polygon edge is one triangle's edge in the polygon's direction, and
    every other edge is shared by two triangles running opposite ways."""
    n = len(poly)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if len(tris) != n - 2 or tris.min(initial=0) < 0 or tris.max(initial=0) >= n:
        return False
    p = np.array(poly, dtype=np.float64)[tris]
    if not np.all(cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]) > 0):
        return False
    start = tris.reshape(-1)
    end = np.roll(tris, -1, axis=1).reshape(-1)
    edges = start * n + end
    if len(np.unique(edges)) != len(edges):
        return False
    boundary = np.arange(n) * n + (np.arange(n) + 1) % n
    if not np.isin(boundary, edges).all():
        return False
    # A polygon edge has no triangle on its far side; every other edge
    # has exactly one.
    return bool(np.array_equal(np.isin(end * n + start, edges), ~np.isin(edges, boundary)))


def triangulate(poly: list[Point]) -> np.ndarray | None:
    """Index triples, counter-clockwise, tiling the simple
    counter-clockwise polygon: the same triples cap every affine image
    of it. None where the kernel returns no valid tiling (a polygon
    thinner than its triangulator resolves). Handed over from its first
    point, so the triangulator sees the polygon's own size."""
    pts = np.array(poly, dtype=np.float64)
    tris = np.asarray(manifold3d.triangulate([pts - pts[0]]), dtype=np.int64).reshape(-1, 3)
    return tris if valid_cap(poly, tris) else None
