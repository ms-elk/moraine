"""2D profile geometry shared by the operators that take an inline
profile (DESIGN 5).

A profile is a list of ``(a, b)`` float pairs. The functions answer
with facts (an index, a sign, a list of normals) and never raise for
the shape of a profile: the operator owns the wording of a refusal and
the fix it names. ``closed`` reads the list as a loop, its last point
joined to its first.
"""

from __future__ import annotations

import math

import manifold3d
import numpy as np

# Shared-kernel version (DESIGN 4.5): ops that consume this module declare
# it in KERNELS and this integer joins their cache keys. Bump on ANY
# output-semantics change of a function an op consumes, bug fixes
# included - like OP_IMPL_VERSION.
KERNEL_VERSION = 1

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
    points run counter-clockwise, whichever point the list starts at."""
    n = len(poly)
    return sum(
        poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1]
        for k in range(n)
    )


def corners(
    pts: list[Point], angle_max_deg: float, *, closed: bool = False,
) -> list[list[Point]]:
    """Per point, the outward unit normals its vertices carry, in
    outline order. A joint bending by at most ``angle_max_deg`` is
    smooth: one normal, the bisector of its two segments' normals. A
    sharper joint is hard: two, the incoming segment's then the
    outgoing one's. An end of an open outline carries its one segment's
    normal. A segment running (da, db) has the normal (db, -da), outward
    for a counter-clockwise outline."""
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
        return [(na / length, nb / length)]

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
    thinner than its triangulator resolves)."""
    tris = np.asarray(
        manifold3d.triangulate([np.array(poly, dtype=np.float64)]), dtype=np.int64
    ).reshape(-1, 3)
    return tris if valid_cap(poly, tris) else None
