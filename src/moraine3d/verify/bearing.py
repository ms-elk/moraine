"""Shared-face-area kernel (contact's ``min_area`` — DESIGN 5.2).

Pure geometry over float64 arrays, the measure behind the bearing
form: the area two triangle sets share is the sum, over the triangle
pairs whose planes are anti-parallel within ``ANGLE_FLOOR_DEG`` and
within ``tol`` of each other along the normal, of the pair's
intersection area in that plane. Each side's triangles partition its
own surface, so the sum is the area of the overlap of the two face
sets and no union of polygons is ever formed. An edge or a corner
contact shares no area; a curved seat shares area only where the two
tessellations put a facet on the same plane.
"""

from __future__ import annotations

import numpy as np

from .pairs import grid_registration

# Anti-parallelism floor for two facets to count as one bearing plane:
# a numerics floor the tool owns, the same number as mated's default
# angle tolerance (DESIGN 5.2).
ANGLE_FLOOR_DEG = 0.01
_EPS = 1e-30  # squared-length floor: below this a normal is a point
_REL = 1e-9  # inclusive-boundary slack, relative to the pair's extent


def shared_area(ta: np.ndarray, tb: np.ndarray, tol: float) -> float:
    """The face area ``ta`` (n, 3, 3) and ``tb`` (m, 3, 3) share within
    ``tol`` along the normal, in square meters. Zero when either side
    is empty or no pair is coplanar and anti-parallel."""
    if len(ta) == 0 or len(tb) == 0:
        return 0.0
    na, oka = _unit_normals(ta)
    nb, okb = _unit_normals(tb)
    ta, na = ta[oka], na[oka]
    tb, nb = tb[okb], nb[okb]
    if len(ta) == 0 or len(tb) == 0:
        return 0.0
    ia, ib = _box_pairs(ta.min(axis=1) - tol, ta.max(axis=1) + tol, tb.min(axis=1), tb.max(axis=1))
    if len(ia) == 0:
        return 0.0
    A, B, NA, NB = ta[ia], tb[ib], na[ia], nb[ib]
    anti = np.einsum("ij,ij->i", NA, NB) <= -np.cos(np.radians(ANGLE_FLOOR_DEG))
    A, B, NA, NB = A[anti], B[anti], NA[anti], NB[anti]
    if len(A) == 0:
        return 0.0
    # Coplanar within tol, measured from both planes (a long thin facet
    # tilted inside the angle floor can read further from the other's
    # plane at its far vertex).
    gap_b = np.abs(np.einsum("ij,ikj->ik", NA, B - A[:, :1])).max(axis=1)
    gap_a = np.abs(np.einsum("ij,ikj->ik", NB, A - B[:, :1])).max(axis=1)
    flat = (gap_b <= tol) & (gap_a <= tol)
    A, B, NA = A[flat], B[flat], NA[flat]
    if len(A) == 0:
        return 0.0
    u, v = _plane_basis(NA)
    origin = A[:, 0]
    A2 = np.stack([np.einsum("ikj,ij->ik", A - origin[:, None], u),
                   np.einsum("ikj,ij->ik", A - origin[:, None], v)], axis=-1)
    B2 = np.stack([np.einsum("ikj,ij->ik", B - origin[:, None], u),
                   np.einsum("ikj,ij->ik", B - origin[:, None], v)], axis=-1)
    return float(_tri_tri_area(_ccw(A2), _ccw(B2)).sum())


def _unit_normals(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    l2 = np.einsum("ij,ij->i", n, n)
    ok = l2 > _EPS
    n = n / np.sqrt(np.where(ok, l2, 1.0))[:, None]
    return n, ok


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal (u, v) in each normal's plane, from the axis the
    normal leans on least (deterministic, no branch on near-ties that
    matters: any axis not parallel to n serves)."""
    helper = np.zeros_like(n)
    helper[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    u = np.cross(n, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(n, u)
    return u, v


def _ccw(t2: np.ndarray) -> np.ndarray:
    """Triangles (n, 3, 2) wound counterclockwise, the orientation the
    inside tests assume."""
    d = t2[:, 1] - t2[:, 0]
    e = t2[:, 2] - t2[:, 0]
    cw = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0] < 0
    out = t2.copy()
    out[cw, 1], out[cw, 2] = t2[cw, 2], t2[cw, 1]
    return out


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(p: np.ndarray, t: np.ndarray, slack: np.ndarray) -> np.ndarray:
    """Whether points ``p`` (n, k, 2) lie in or on the CCW triangles ``t``
    (n, 3, 2): on or left of every edge, within ``slack`` (n,) of the
    edge line, so a vertex resting on the other's edge counts."""
    ok = np.ones(p.shape[:2], dtype=bool)
    for i in range(3):
        q0, q1 = t[:, i], t[:, (i + 1) % 3]
        e = q1 - q0
        el = np.sqrt(np.einsum("ij,ij->i", e, e))
        el = np.where(el > 0, el, 1.0)
        s = _cross2(e[:, None, :], p - q0[:, None, :]) / el[:, None]
        ok &= s >= -slack[:, None]
    return ok


def _tri_tri_area(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Intersection area per pair of CCW triangles (n, 3, 2): the
    intersection of two convex sets is convex and its vertices are the
    vertices of each inside the other plus the edge crossings, so the
    area is the shoelace over those points in angular order."""
    n = len(A)
    scale = np.maximum(np.abs(A).max(axis=(1, 2)), np.abs(B).max(axis=(1, 2)))
    slack = _REL * np.maximum(scale, 1e-12)
    pts = np.zeros((n, 15, 2))
    mask = np.zeros((n, 15), dtype=bool)
    pts[:, 0:3] = A
    mask[:, 0:3] = _inside(A, B, slack)
    pts[:, 3:6] = B
    mask[:, 3:6] = _inside(B, A, slack)
    k = 6
    for i in range(3):
        p, r = A[:, i], A[:, (i + 1) % 3] - A[:, i]
        for j in range(3):
            q, s = B[:, j], B[:, (j + 1) % 3] - B[:, j]
            den = _cross2(r, s)
            par = np.abs(den) <= _EPS
            den = np.where(par, 1.0, den)
            w = q - p
            t = _cross2(w, s) / den
            uu = _cross2(w, r) / den
            hit = ~par & (t >= -_REL) & (t <= 1 + _REL) & (uu >= -_REL) & (uu <= 1 + _REL)
            pts[:, k] = p + t[:, None] * r
            mask[:, k] = hit
            k += 1
    m = mask.sum(axis=1)
    cnt = np.maximum(m, 1)[:, None]
    centroid = (pts * mask[:, :, None]).sum(axis=1) / cnt
    rel = pts - centroid[:, None, :]
    ang = np.where(mask, np.arctan2(rel[:, :, 1], rel[:, :, 0]), np.inf)
    order = np.argsort(ang, axis=1, kind="stable")
    srt = np.take_along_axis(rel, order[:, :, None], axis=1)
    idx = np.arange(15)[None, :]
    live = idx < m[:, None]
    nxt = np.where(idx + 1 < m[:, None], idx + 1, 0)
    nxt_pts = np.take_along_axis(srt, nxt[:, :, None], axis=1)
    terms = np.where(live, _cross2(srt, nxt_pts), 0.0)
    area = 0.5 * np.abs(terms.sum(axis=1))
    # The inclusive slack lets an edge-on-edge pair form a sliver no
    # wider than the slack along the shared edge: a polygon whose area
    # fits in a strip of that width around its own perimeter shares a
    # line, not a face. The bound is the polygon's, so a small seat on
    # a large face keeps its area.
    sides = np.where(live, np.linalg.norm(nxt_pts - srt, axis=2), 0.0)
    perimeter = sides.sum(axis=1)
    return np.where(area <= slack * perimeter, 0.0, area)


def _box_pairs(la, ha, lb, hb) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs whose boxes overlap (``la``/``ha`` already inflated),
    through the shared grid (pairs.grid_registration): a pair that
    overlaps shares a cell, and the exact box test below removes the
    collisions the hash admits."""
    ia_r, ka, ib_r, kb = grid_registration(la, ha, lb, hb)
    order = np.argsort(kb, kind="stable")
    kb_s, ib_s = kb[order], ib_r[order]
    starts = np.searchsorted(kb_s, ka, "left")
    counts = (np.searchsorted(kb_s, ka, "right") - starts).astype(np.int64)
    ia = np.repeat(ia_r, counts)
    pos = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    ib = ib_s[np.repeat(starts, counts) + pos]
    if len(ia) == 0:
        return ia, ib
    key = np.unique(ia * len(lb) + ib)
    ia, ib = key // len(lb), key % len(lb)
    hit = np.all((ha[ia] >= lb[ib]) & (la[ia] <= hb[ib]), axis=1)
    return ia[hit], ib[hit]
