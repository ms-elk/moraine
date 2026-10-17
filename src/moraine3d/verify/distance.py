"""Minimum-surface-distance kernel (contact / clearance — DESIGN 5.2).

Pure geometry over float64 arrays: validate.py turns these numbers into
check entries and fix sentences. A *source* is the per-output world-
triangle view ``(proto_tris, transforms | None, copy_lo, copy_hi)`` built
by ``world_tri_source``; copies are culled per query box and never
merged (instancing stays instancing).
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Image, Instances, Value, proto_of
from .pairs import (
    grid_registration, last_uses, overlapping_copy_pairs, release_after, transformed_corner_boxes,
)


# The distance kinds are threshold predicates: they never need a global
# nearest-pair query, only "is any triangle pair within tol". Triangle
# AABBs inflated by tol and joined through a uniform grid give a sound
# broad phase (any pair within tol shares a cell); the exact narrow phase
# runs on the survivors. Everything is float64 — tol sits below float32
# resolution at asset scale.

_DIST_EPS = 1e-30  # squared-length floor: below this an edge/normal is a point


def _safe_div(num, den):
    """Elementwise num/den with degenerate denominators neutralized. One
    shared guard so the vertex-face and edge-edge features agree on what
    counts as degenerate."""
    return num / np.where(np.abs(den) > _DIST_EPS, den, 1.0)


def source_why(value: Value, name: str) -> str | None:
    """Why an output cannot be measured — an image, no triangles,
    non-finite vertices or transforms — or None. The predicates
    ``world_tri_source`` applies before its gather, alone: a kind that
    reads only bounds asks this and never materializes the triangles."""
    if isinstance(value, Image):
        return f"output '{name}' is an image; assertions apply to geometry"
    proto = proto_of(value)
    if len(proto.faces) == 0:
        return f"output '{name}' has no triangles; nothing to measure"
    if not np.isfinite(proto.vertices).all():
        return f"output '{name}' has non-finite vertices; fix the finite check first"
    if isinstance(value, Instances) and not np.isfinite(value.transforms).all():
        return (
            f"output '{name}' has non-finite instance transforms; "
            f"fix the finite check first"
        )
    return None


def world_tri_source(value: Value, name: str):
    """Per-output world-triangle source, or (None, why-not).

    Built once per output and cheap to hold: the prototype triangles plus,
    for Instances, per-copy transforms and world AABBs — copies are culled
    per query box and tiled chunk-wise (`_tris_in_box`). Unevaluatable
    cases fail with the reason: image outputs, empty (zero-triangle)
    outputs, and non-finite data. Watertightness is NOT required — the
    distance kinds read raw surfaces (DESIGN 5.2).
    """
    why = source_why(value, name)
    if why is not None:
        return None, why
    proto = proto_of(value)
    v = proto.vertices.astype(np.float64)
    f = proto.faces.astype(np.int64)
    ptris = v[f]  # (m, 3, 3)
    if not isinstance(value, Instances):
        return (ptris, None, None, None), None
    xf = np.asarray(value.transforms, dtype=np.float64)
    clo, chi = transformed_corner_boxes(v, xf)
    return (ptris, xf, clo, chi), None


def _tris_in_box(source, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """World triangles from a source, culled to AABB-overlap with [lo, hi].

    Culling is conservative-outward (AABB tests, the Instances boxes from
    transformed corner boxes), so it only discards triangles provably
    outside the box — sound for any query whose minimizer lies inside.
    Instances tile chunk-wise so the tiling transient stays bounded; the
    surviving set itself is materialized, so an assert between a dense
    scatter and a neighbor whose box covers it still scales with the
    survivors.
    """
    ptris, xf, clo, chi = source
    if xf is None:
        keep = np.all((ptris.max(axis=1) >= lo) & (ptris.min(axis=1) <= hi), axis=1)
        return ptris[keep]
    keep = np.all((chi >= lo) & (clo <= hi), axis=1)
    if not keep.any():
        return np.empty((0, 3, 3), dtype=np.float64)
    xs = xf[keep]
    out = []
    step = max(1, (1 << 18) // max(len(ptris), 1))
    for s in range(0, len(xs), step):
        xc = xs[s : s + step]
        w = np.einsum("kij,mtj->kmti", xc[:, :3, :3], ptris) + xc[:, None, None, :3, 3]
        tris = w.reshape(-1, 3, 3)
        tkeep = np.all((tris.max(axis=1) >= lo) & (tris.min(axis=1) <= hi), axis=1)
        out.append(tris[tkeep])
    return out[0] if len(out) == 1 else np.concatenate(out)


def copy_source(src, i: int):
    """A world-triangle source restricted to copy ``i`` of an instanced
    source — the per-copy view the quantifiers share with ``mated``."""
    ptris, xf, clo, chi = src
    return (ptris, xf[i : i + 1], clo[i : i + 1], chi[i : i + 1])


def copy_bounds(src, i: int) -> tuple[np.ndarray, np.ndarray]:
    return src[2][i], src[3][i]


def cull_source(src, lo: np.ndarray, hi: np.ndarray):
    """A source reduced to the world triangles within the box [lo, hi],
    paired with that set's own bounds — a plain-mesh source (the Instances
    copies are materialized), valid only for queries whose minimizer lies
    inside the box it was cut to."""
    tris = _tris_in_box(src, lo, hi)
    if len(tris):
        bounds = (tris.min(axis=(0, 1)), tris.max(axis=(0, 1)))
    else:
        bounds = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    return (tris, None, None, None), bounds

def min_candidate_distance(ta, tb, inflate: float) -> float | None:
    """Minimum surface distance over all triangle pairs within `inflate` of
    each other, or None when the broad phase proves no such pair exists.
    Exact whenever the true minimum is <= inflate (the minimizing pair is
    then always a candidate — a pair within `inflate` always shares a grid
    cell, because AABBs register in every cell they span); otherwise a
    sound upper bound over the candidates it did find.

    The grid join runs block-wise with a shrinking threshold: each block's
    pairs are pruned against the best distance seen so far (a pair whose
    AABB gap exceeds it cannot improve the minimum), so neither the join
    nor the narrow phase degenerates toward the full cross product when
    the inflation is large relative to the actual gap — the regime every
    failing contact's exact-distance search runs in.
    """
    if len(ta) == 0 or len(tb) == 0:
        return None
    la_raw, ha_raw = ta.min(axis=1), ta.max(axis=1)
    lb, hb = tb.min(axis=1), tb.max(axis=1)
    la, ha = la_raw - inflate, ha_raw + inflate
    # The grid join (pairs.grid_registration): the gap test below
    # removes the hash collisions it admits.
    ia_r, ka, ib_r, kb = grid_registration(la, ha, lb, hb, min_cell=float(inflate))
    order = np.argsort(kb, kind="stable")
    kb_s, ib_s = kb[order], ib_r[order]
    starts = np.searchsorted(kb_s, ka, "left")
    matches = (np.searchsorted(kb_s, ka, "right") - starts).astype(np.int64)
    cum = np.cumsum(matches)
    best = np.inf
    thresh = float(inflate)
    chunk = 1 << 16
    blk = 0
    while blk < len(ka):
        base = int(cum[blk - 1]) if blk else 0
        end = max(int(np.searchsorted(cum, base + chunk, side="right")), blk + 1)
        c = matches[blk:end]
        n = int(c.sum())
        if n:
            rep_a = np.repeat(ia_r[blk:end], c)
            off = np.arange(n, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
            rep_b = ib_s[np.repeat(starts[blk:end], c) + off]
            # Euclidean AABB gap lower-bounds the pair's true distance, so
            # pruning pairs whose gap exceeds the running best (or the
            # inflation) never discards a pair that could set the minimum.
            gap = np.maximum(
                0.0, np.maximum(lb[rep_b] - ha_raw[rep_a], la_raw[rep_a] - hb[rep_b])
            )
            ok = np.einsum("ij,ij->i", gap, gap) <= thresh * thresh
            if ok.any():
                pk = rep_a[ok] * np.int64(len(tb)) + rep_b[ok]
                pairs = np.unique(pk)
                d = tri_pair_dist(ta[pairs // len(tb)], tb[pairs % len(tb)])
                best = min(best, float(d.min()))
                thresh = min(thresh, best)
        blk = end
    return None if np.isinf(best) else best


# The two feature-distance routines below follow the published code in
# sections 5.1.5 and 5.1.9 of the book named here, vectorized over numpy
# arrays. The book's software license grants use for any purpose on the
# condition that this credit appears with the code: from Real-Time
# Collision Detection by Christer Ericson, published by Morgan Kaufmann
# Publishers, (c) 2005 Elsevier Inc. THIRD_PARTY_NOTICES.md carries the
# same credit for the distribution.
def _point_tri_dist(p, a, b, c):
    """Distance from points to triangles (Ericson 5.1.5 region method),
    elementwise over (n, 3) arrays. Degenerate triangles neutralize the
    face-interior branch (a sliver has no interior); their edges and
    vertices stay covered by the pair minimum's other feature terms.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    t_ab = np.clip(_safe_div(d1, d1 - d3), 0.0, 1.0)
    t_ac = np.clip(_safe_div(d2, d2 - d6), 0.0, 1.0)
    t_bc = np.clip(_safe_div(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)
    denom = va + vb + vc
    v_f = _safe_div(vb, denom)
    w_f = _safe_div(vc, denom)
    face_ok = np.abs(denom) > _DIST_EPS

    candidates = [
        (
            (d1 <= 0.0) & (d2 <= 0.0),
            a,
        ),
        (
            (d3 >= 0.0) & (d4 <= d3),
            b,
        ),
        (
            (d6 >= 0.0) & (d5 <= d6),
            c,
        ),
        (
            (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
            a + t_ab[:, None] * ab,
        ),
        (
            (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
            a + t_ac[:, None] * ac,
        ),
        (
            (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
            b + t_bc[:, None] * (c - b),
        ),
    ]
    closest = a + v_f[:, None] * ab + w_f[:, None] * ac  # face region default
    dist = np.where(
        face_ok, np.linalg.norm(p - closest, axis=1), np.inf
    )
    settled = np.zeros(len(p), dtype=bool)
    for mask, point in candidates:
        use = mask & ~settled
        if use.any():
            dist[use] = np.linalg.norm(p[use] - point[use], axis=1)
        settled |= mask
    return dist


def seg_seg_dist(p1, q1, p2, q2):
    """Distance between segments (Ericson 5.1.9 clamped parametrization),
    elementwise over (n, 3) arrays; near-parallel and degenerate segments
    fall back through the clamp masks, never through a raw division.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b

    s = np.where(
        denom > _DIST_EPS * np.maximum(a * e, 1.0),
        np.clip(_safe_div(b * f - c * e, denom), 0.0, 1.0),
        0.0,
    )
    # Second segment a point: the closest point on the first is the
    # projection of that point, clamped (the parallel branch above left
    # s at 0, which is only the endpoint).
    s = np.where(
        e > _DIST_EPS,
        s,
        np.where(a > _DIST_EPS, np.clip(_safe_div(-c, a), 0.0, 1.0), 0.0),
    )
    t = np.where(e > _DIST_EPS, _safe_div(b * s + f, e), 0.0)
    t_cl = np.clip(t, 0.0, 1.0)
    s = np.where(
        t != t_cl,
        np.where(a > _DIST_EPS, np.clip(_safe_div(b * t_cl - c, a), 0.0, 1.0), 0.0),
        np.where(a > _DIST_EPS, s, 0.0),
    )
    cp1 = p1 + s[:, None] * d1
    cp2 = p2 + t_cl[:, None] * d2
    return np.linalg.norm(cp1 - cp2, axis=1)


def _edge_pierces_face(tri_e, tri_f):
    """True where any edge of tri_e crosses tri_f's interior transversally.

    The feature distances alone miss exactly this configuration (a rod
    through a large face: every vertex-face and edge-edge term is large
    while the true surface distance is 0), so the pair distance clamps to
    0 where this fires. Coplanar overlap needs no branch here — its
    feature distance is already 0. Degenerate faces have no interior.
    """
    f0, f1, f2 = tri_f[:, 0], tri_f[:, 1], tri_f[:, 2]
    n = np.cross(f1 - f0, f2 - f0)
    valid = np.einsum("ij,ij->i", n, n) > _DIST_EPS
    crossed = np.zeros(len(tri_e), dtype=bool)
    for i in range(3):
        p = tri_e[:, i]
        q = tri_e[:, (i + 1) % 3]
        hp = np.einsum("ij,ij->i", p - f0, n)
        hq = np.einsum("ij,ij->i", q - f0, n)
        transversal = ((hp > 0) & (hq < 0)) | ((hp < 0) & (hq > 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            tpar = np.where(transversal, hp / np.where(transversal, hp - hq, 1.0), 0.0)
        x = p + tpar[:, None] * (q - p)
        inside = (
            (np.einsum("ij,ij->i", np.cross(f1 - f0, x - f0), n) >= 0)
            & (np.einsum("ij,ij->i", np.cross(f2 - f1, x - f1), n) >= 0)
            & (np.einsum("ij,ij->i", np.cross(f0 - f2, x - f2), n) >= 0)
        )
        crossed |= transversal & inside
    return crossed & valid


def tri_pair_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact surface distance per triangle pair: the minimum over the six
    vertex-triangle and nine edge-edge features — complete for disjoint
    pairs — clamped to 0 for transversally crossing pairs (see
    `_edge_pierces_face`).
    """
    d = np.full(len(A), np.inf)
    for i in range(3):
        d = np.minimum(d, _point_tri_dist(A[:, i], B[:, 0], B[:, 1], B[:, 2]))
        d = np.minimum(d, _point_tri_dist(B[:, i], A[:, 0], A[:, 1], A[:, 2]))
    for i in range(3):
        p1, q1 = A[:, i], A[:, (i + 1) % 3]
        for j in range(3):
            d = np.minimum(d, seg_seg_dist(p1, q1, B[:, j], B[:, (j + 1) % 3]))
    crossed = _edge_pierces_face(A, B) | _edge_pierces_face(B, A)
    return np.where(crossed, 0.0, d)


def surface_sample_bound(src_a, src_b) -> float:
    """Deterministic upper bound on the surface distance: alternating
    nearest-vertex descent between on-surface sample sets (for Instances,
    a strided subset of each copy's transformed prototype vertices — every
    sample lies on the real surface, so any sample pair bounds the true
    minimum from above)."""

    def samples(source):
        ptris, xf, _, _ = source
        pts = ptris.reshape(-1, 3)
        if xf is None:
            return pts
        step = max(1, len(pts) // 32)
        sub = pts[::step]
        w = np.einsum("kij,mj->kmi", xf[:, :3, :3], sub) + xf[:, None, :3, 3]
        return w.reshape(-1, 3)

    va, vb = samples(src_a), samples(src_b)
    pa = va[0]
    pb = vb[np.argmin(((vb - pa) ** 2).sum(axis=1))]
    for _ in range(3):
        pa = va[np.argmin(((va - pb) ** 2).sum(axis=1))]
        pb = vb[np.argmin(((vb - pa) ** 2).sum(axis=1))]
    return float(np.linalg.norm(pa - pb))


def aabb_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Euclidean distance between two AABBs — a sound lower bound on the
    surface distance of anything they contain."""
    d = np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))
    return float(np.linalg.norm(d))


def surface_distance(src_a, bounds_a, src_b, bounds_b, tol: float):
    """Minimum surface distance between two triangle sources, or None
    when it is provably above ``tol`` (bounds gap) or no candidate pair
    falls inside the ``tol`` inflation. The one kernel entry for the pair
    check, the per-copy quantifier, and the copy-pair clearance."""
    (lo_a, hi_a), (lo_b, hi_b) = bounds_a, bounds_b
    if aabb_gap(lo_a, hi_a, lo_b, hi_b) > tol:
        return None
    ta = _tris_in_box(src_a, lo_b - tol, hi_b + tol)
    tb = _tris_in_box(src_b, lo_a - tol, hi_a + tol)
    return min_candidate_distance(ta, tb, tol)


def exact_surface_distance(src_a, bounds_a, src_b, bounds_b, upper=None) -> float:
    """The true minimum surface distance, for an actionable failure
    message. Any candidate minimum already bounds it from above; otherwise
    a nearest-vertex descent bound does. One pass at that inflation
    captures the minimizing pair exactly."""
    (lo_a, hi_a), (lo_b, hi_b) = bounds_a, bounds_b
    bound = upper if upper is not None else surface_sample_bound(src_a, src_b)
    bound = bound * (1.0 + 1e-9) + 1e-12
    ta = _tris_in_box(src_a, lo_b - bound, hi_b + bound)
    tb = _tris_in_box(src_b, lo_a - bound, hi_a + bound)
    exact = min_candidate_distance(ta, tb, bound)
    if exact is None:
        # Unreachable in a sound broad phase: the pair realizing `bound` is
        # always a candidate at inflation `bound`. Substituting a bounds
        # figure here would silently mask a kernel bug behind a plausible
        # number — fail loudly instead.
        raise RuntimeError(
            "internal: distance search lost the minimizing pair below its "
            "proven upper bound - broad-phase soundness bug, please report"
        )
    return exact


def _copy_world_tris(ptris: np.ndarray, xf: np.ndarray, i: int, memo: dict) -> np.ndarray:
    """One copy's world triangles, built once per copy (the copy_solid
    memo of the no_overlap path); per pair only a box cull remains."""
    if i not in memo:
        x = xf[i]
        memo[i] = np.einsum("ij,mtj->mti", x[:3, :3], ptris) + x[:3, 3]
    return memo[i]


def closest_copy_pair(src, inflate: float, known: dict | None = None):
    """The closest pair of copies of an instanced source, ``(i, j,
    distance)`` — the number a green ``self`` clearance would otherwise
    withhold (DESIGN 5.2) — or None below two copies. A doubling box
    search: the copy pairs whose boxes come within ``inflate`` are
    measured in order of box gap, pruned against the running best (a box
    gap lower-bounds the pair's distance), and the inflation doubles
    until the best found is provably the minimum — any pair closer than
    a best that fits inside the inflation would have been a candidate —
    or every pair has been a candidate. ``known`` seeds the search with
    pairs already priced exactly (``self_clearance_pairs``' third
    result), and a pair is settled once across rounds: priced exactly
    when its search finds a distance within its inflation, or proven
    above the running best when it does not (the kernel's number above
    its inflation is a bound, never taken for the distance). The
    per-copy world memo is dropped between rounds, so a dense scatter
    never holds every copy's geometry at once. Deterministic;
    logarithmically many rounds, and never the full k^2 narrow phase."""
    ptris, xf, clo, chi = src
    k = len(xf)
    if k < 2:
        return None
    span = float(np.linalg.norm(chi.max(axis=0) - clo.min(axis=0)))
    settled: set[tuple[int, int]] = set(known or ())
    best: tuple[int, int, float] | None = None
    for (i, j), d in (known or {}).items():
        if best is None or d < best[2]:
            best = (i, j, d)
    inflate = max(float(inflate), 1e-9)
    while True:
        total, ii, jj = overlapping_copy_pairs(clo, chi, inflate=inflate)
        world: dict[int, np.ndarray] = {}
        if len(ii):
            gaps = np.linalg.norm(
                np.maximum(0.0, np.maximum(clo[jj] - chi[ii], clo[ii] - chi[jj])), axis=1
            )
            for p in np.argsort(gaps, kind="stable"):
                if best is not None and gaps[p] >= best[2]:
                    break
                i, j = int(ii[p]), int(jj[p])
                if (i, j) in settled:
                    continue
                ta = _copy_world_tris(ptris, xf, i, world)
                tb = _copy_world_tris(ptris, xf, j, world)
                bound = surface_sample_bound((ta, None, None, None), (tb, None, None, None))
                up = bound * (1.0 + 1e-9) + 1e-12
                if best is not None:
                    up = min(up, best[2])
                d = min_candidate_distance(ta, tb, up)
                settled.add((i, j))
                if d is not None and d <= up and (best is None or d < best[2]):
                    best = (i, j, d)
        if best is not None and best[2] <= inflate:
            return best
        if inflate >= span or len(ii) == total:
            return best
        inflate *= 2.0


def self_clearance_pairs(src, tol: float):
    """Copy-pair clearance over one instanced source (DESIGN 5.2 'self' on
    clearance): ``(pairs_checked, failing [(i, j, distance)], known {(i,
    j): distance})`` — ``known`` being the candidate pairs the pass
    priced exactly, the seed for ``closest_copy_pair``: a pair whose
    search finds a distance within its inflation (the kernel's
    exactness condition). A pair the search prices above its inflation
    is only proven clear of ``tol``; that number is a bound, not the
    distance, and the pair is left out of ``known`` for the closest-pair
    search to measure. Copies stay per-copy (never merged); pairs whose
    boxes are farther apart than ``tol`` are proven clear by the broad
    phase; ``>= tol`` passes (inclusive, as the cross-output bound)."""
    ptris, xf, clo, chi = src
    total, ii, jj = overlapping_copy_pairs(clo, chi, inflate=tol)
    world: dict[int, np.ndarray] = {}
    known: dict[tuple[int, int], float] = {}
    last = last_uses(ii, jj, len(xf))

    def copy_tris(i: int) -> np.ndarray:
        return _copy_world_tris(ptris, xf, i, world)

    def near(tris: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        keep = np.all((tris.max(axis=1) >= lo - tol) & (tris.min(axis=1) <= hi + tol), axis=1)
        return tris[keep]

    failing: list[tuple[int, int, float]] = []
    for p, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
        ta = near(copy_tris(i), clo[j], chi[j])
        tb = near(copy_tris(j), clo[i], chi[i])
        release_after(world, last, p, i, j)
        if len(ta) == 0 or len(tb) == 0:
            continue
        # A sampled upper bound sizes the grid to the pair's actual gap:
        # the exact minimum falls out cheaply when it is below tol, and a
        # pair whose bound already clears tol is searched at tol — the
        # threshold it has to beat. (The kernel is exact below its
        # inflation and the bound is above the minimum, so an inflation of
        # min(bound, tol) resolves the predicate either way; a result
        # above the inflation is a bound over other candidates, so only a
        # result within it is the pair's distance.)
        bound = surface_sample_bound((ta, None, None, None), (tb, None, None, None))
        inflate = min(bound * (1.0 + 1e-9) + 1e-12, tol)
        d = min_candidate_distance(ta, tb, inflate)
        if d is None or d > inflate:
            continue
        known[(i, j)] = d
        if d < tol:
            failing.append((i, j, d))
    return total, failing, known
