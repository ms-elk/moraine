"""``revolve`` operator.

Semantics: emits the solid of revolution of ``profile`` - a list of two
or more [radius, y] points (m) forming an open polyline in the +X
half-plane - swept a full 360 degrees around the +Y axis with
``segments`` steps (default 32, 3..4096). Order the profile so the
solid lies on its left (bottom-to-top along the outside); a profile
that would face the surface inward is refused naming the fix (reverse
the point order). Ends close automatically: an end with radius 0
becomes a pole vertex, an end with radius > 0 a flat horizontal cap
(duplicated ring, +-Y normals, center-vertex fan) - the result is
always watertight. Radii must be >= 0, with 0 only at the ends;
consecutive duplicates, identical endpoints, interior zero radii,
exact double-backs, self-intersections (the closure to the axis
included), and profiles enclosing no area are refused, as is a
rings x segments product beyond the shared 4096 x 4096 cell budget
(hard joints duplicate rings, so the per-param ceilings alone do not
bound it).
Circumferential shading is smooth (shared ring vertices); a profile
joint bending by more than ``angle_max_deg`` (default 30) is hard
(duplicated rings), a gentler one smooth (shared ring, half-plane
bisector normal). Vertex 0 of each ring sits at angle 0 (+X); winding
is CCW from outside. Deterministic.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import numbers
from ..core.mesh import Mesh
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
# A radius this far below zero is a point an expression put on the axis
# (an interpolation's end): a nanometer, under what a float32 vertex
# resolves at any size the tool builds. It is read as 0. Nothing that
# was accepted before reads differently, so the version stands.
RADIUS_NOISE = 1e-9
KERNELS = ()
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "profile": numbers.FLOAT,
    "segments": numbers.INT,
    "angle_max_deg": numbers.FLOAT,
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


def _cross2(u, v):
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


def _self_intersection(poly: list[tuple[float, float]]) -> tuple[int, int] | None:
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
        d1 = _cross2(r, q1 - p1)
        d2 = _cross2(r, q2 - p1)
        d3 = _cross2(s, p1 - q1)
        d4 = _cross2(s, p2 - q1)
        bad = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
        bad |= (d1 == 0) & _on_segment(p1, p2, q1)
        bad |= (d2 == 0) & _on_segment(p1, p2, q2)
        bad |= (d3 == 0) & _on_segment(q1, q2, p1)
        bad |= (d4 == 0) & _on_segment(q1, q2, p2)
        if bad.any():
            return k, m0 + int(np.argmax(bad))
    return None


def _profile(params: dict) -> list[tuple[float, float]]:
    value = params.get("profile")
    if value is None:
        raise _bad("revolve requires 'profile' - a list of [radius, y] points (m)")
    if not isinstance(value, list) or not 2 <= len(value) <= PROFILE_MAX:
        raise _bad(
            f"revolve 'profile' must be a list of 2..{PROFILE_MAX} [radius, y] points"
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
        if r == 0 and 0 < k < len(value) - 1:
            raise _bad(
                f"revolve 'profile' point {k} has radius 0 - the surface would "
                "pinch at the axis; radius 0 is allowed only at the ends"
            )
        pts.append((r, y))
    for k in range(len(pts) - 1):
        if pts[k] == pts[k + 1]:
            raise _bad(
                f"revolve 'profile' points {k} and {k + 1} coincide - remove "
                "the duplicate"
            )
    if pts[0] == pts[-1]:
        raise _bad(
            "revolve 'profile' first and last points coincide - the ends "
            "close automatically; leave the polyline open (or compose "
            "solids with boolean)"
        )
    # Exact double-backs (adjacent segments exactly opposite) are a
    # zero-thickness fin, and the one joint whose bisector normal is
    # undefined.
    for k in range(len(pts) - 2):
        d0 = (pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
        d1 = (pts[k + 2][0] - pts[k + 1][0], pts[k + 2][1] - pts[k + 1][1])
        if d0[0] * d1[1] - d0[1] * d1[0] == 0 and d0[0] * d1[0] + d0[1] * d1[1] < 0:
            raise _bad(
                f"revolve 'profile' doubles back exactly at point {k + 1} - "
                "a zero-thickness fin; drop or move the point"
            )
    # The generating polygon: the profile closed to the axis at the end
    # heights (a pole end already sits on the axis). Its edges must not
    # cross or touch - a self-intersecting polygon sweeps an ill-defined
    # solid that only looks watertight.
    poly = list(pts)
    if pts[-1][0] > 0:
        poly.append((0.0, pts[-1][1]))
    if pts[0][0] > 0 and (0.0, pts[0][1]) != poly[-1]:
        poly.append((0.0, pts[0][1]))
    hit = _self_intersection(poly)
    if hit is not None:
        names = [
            f"the segment from point {e}" if e < len(pts) - 1 else "the end closure"
            for e in hit
        ]
        raise _bad(
            f"revolve 'profile' self-intersects - {names[0]} crosses "
            f"{names[1]}; the swept solid would be ill-defined"
        )
    # Orientation gate: the generating polygon must have positive signed
    # area, so the authored winding is outward by construction.
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
    pts = _profile(params)
    seg = common.integer("revolve", params, "segments", 32, minimum=3, maximum=SEGMENTS_MAX)
    angle_max = common.num(
        "revolve", params, "angle_max_deg",
        common.CREASE_ANGLE_DEG_DEFAULT, positive=True, maximum=180.0,
    )

    n_pts = len(pts)
    # Per-segment direction and half-plane band normal (rotate by -90):
    # for segment direction (dr, dy) the outward normal is (dy, -dr).
    dirs = []
    band_n2 = []
    for k in range(n_pts - 1):
        dr = pts[k + 1][0] - pts[k][0]
        dy = pts[k + 1][1] - pts[k][1]
        length = math.hypot(dr, dy)
        dirs.append((dr / length, dy / length))
        band_n2.append((dy / length, -dr / length))

    # Rings: (radius, y, half-plane normal). A shared ring joins two
    # bands with the bisector normal; a hard joint duplicates the ring.
    rings: list[tuple[float, float, tuple[float, float]]] = []
    band_lo = [0] * (n_pts - 1)  # ring index at pts[k] for band k
    band_hi = [0] * (n_pts - 1)  # ring index at pts[k + 1] for band k
    pole_start = pts[0][0] == 0
    pole_end = pts[-1][0] == 0
    if not pole_start:
        band_lo[0] = len(rings)
        rings.append((*pts[0], band_n2[0]))
    for k in range(1, n_pts - 1):
        d0, d1 = dirs[k - 1], dirs[k]
        bend = math.degrees(math.acos(max(-1.0, min(1.0, d0[0] * d1[0] + d0[1] * d1[1]))))
        if bend <= angle_max:
            nr, ny = band_n2[k - 1][0] + band_n2[k][0], band_n2[k - 1][1] + band_n2[k][1]
            length = math.hypot(nr, ny)
            band_hi[k - 1] = band_lo[k] = len(rings)
            rings.append((*pts[k], (nr / length, ny / length)))
        else:
            band_hi[k - 1] = len(rings)
            rings.append((*pts[k], band_n2[k - 1]))
            band_lo[k] = len(rings)
            rings.append((*pts[k], band_n2[k]))
    if not pole_end:
        band_hi[-1] = len(rings)
        rings.append((*pts[-1], band_n2[-1]))

    # Hard joints duplicate rings, so the rings x segments product can
    # exceed what the per-param ceilings admit alone - refuse before
    # allocating, per the ceilings' own rationale.
    if len(rings) * seg > common.TESSELLATION_CELLS_MAX:
        raise OpError(
            "bad_param",
            f"revolve tessellation is too large: {len(rings)} rings x "
            f"{seg} segments exceeds the {common.TESSELLATION_CELLS_MAX} "
            "cell budget; reduce segments, profile points, or hard joints",
            param="segments",
        )

    ang = np.arange(seg) * (2.0 * np.pi / seg)
    cos, sin = np.cos(ang), np.sin(ang)
    i = np.arange(seg)
    j = (i + 1) % seg

    def ring_xyz(r: float, y: float) -> np.ndarray:
        return np.stack([r * cos, np.full(seg, y), -r * sin], axis=1)

    verts = [ring_xyz(r, y) for r, y, _ in rings]
    normals = [
        np.stack([nr * cos, np.full(seg, ny), -nr * sin], axis=1)
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
    for k in range(n_pts - 1):
        if k == 0 and pole_start:
            p = emit(np.array([[0.0, pts[0][1], 0.0]]), np.array([[0.0, -1.0, 0.0]]))
            b = band_hi[0] * seg
            faces.append(np.stack([np.full(seg, p), b + j, b + i], axis=1))
        elif k == n_pts - 2 and pole_end:
            p = emit(np.array([[0.0, pts[-1][1], 0.0]]), np.array([[0.0, 1.0, 0.0]]))
            a = band_lo[k] * seg
            faces.append(np.stack([a + i, a + j, np.full(seg, p)], axis=1))
        else:
            a, b = band_lo[k] * seg, band_hi[k] * seg
            faces.append(np.stack([a + i, a + j, b + j], axis=1))
            faces.append(np.stack([a + i, b + j, b + i], axis=1))

    # Caps: the closure path runs axis-out into pts[0] and pts[-1]-in to
    # the axis, so cap normals are -Y (start) / +Y (end) and winding
    # follows the sign - hard by construction (duplicated ring +
    # center-vertex fan, the cylinder cap technique).
    for (r, y), ny, pole in ((pts[0], -1.0, pole_start), (pts[-1], 1.0, pole_end)):
        if pole:
            continue
        base = emit(
            np.concatenate([ring_xyz(r, y), [[0.0, y, 0.0]]]),
            np.tile([0.0, ny, 0.0], (seg + 1, 1)),
        )
        cols = [base + i, base + j, np.full(seg, base + seg)]
        faces.append(np.stack(cols if ny > 0 else cols[::-1], axis=1))

    return Mesh(
        vertices=np.concatenate(verts).astype(np.float32),
        faces=np.concatenate(faces).astype(np.int32),
        normals=np.concatenate(normals).astype(np.float32),
    )
