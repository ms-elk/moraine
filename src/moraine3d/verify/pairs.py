"""Instanced copy-pair broad phase (DESIGN 5.2 copy passes).

The per-copy machinery the copy passes share: the overlap guard's and
embedded's in ``validate`` (manifold solids per copy) and the
``clearance`` self check in ``distance`` (world triangles per copy).
Copies stay per-copy — instancing is never merged — so both paths
enumerate candidate copy pairs from transformed prototype AABBs here,
then resolve the survivors with their own exact kernel, holding each
copy's world geometry only until its last pair.
"""

from __future__ import annotations

import numpy as np


def transformed_corner_boxes(v: np.ndarray, xf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-copy world AABBs from the prototype's transformed AABB corners:
    conservative (a superset of each copy's exact bounds), so sound for
    culling and for gap early-outs — one home for the distance and the
    copy-pair paths."""
    plo, phi = v.min(axis=0), v.max(axis=0)
    corners = np.stack(
        np.meshgrid([plo[0], phi[0]], [plo[1], phi[1]], [plo[2], phi[2]], indexing="ij"),
        axis=-1,
    ).reshape(8, 3)
    wc = np.einsum("kij,cj->kci", xf[:, :3, :3], corners) + xf[:, None, :3, 3]
    return wc.min(axis=1), wc.max(axis=1)


def overlapping_copy_pairs(clo: np.ndarray, chi: np.ndarray, inflate: float = 0.0):
    """``(total_pairs, ii, jj)``: the copy pairs whose boxes come within
    ``inflate`` of each other on every axis; the rest are provably farther
    apart than ``inflate`` (``inflate`` 0 = plain AABB overlap)."""
    k = len(clo)
    total = k * (k - 1) // 2
    # Row blocks bound the (rows, k, 3) boolean transient — a dense
    # scatter's k^2 pairs would otherwise allocate gigabytes at once.
    step = max(1, (1 << 22) // max(k, 1))
    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    for s in range(0, k, step):
        lo_s, hi_s = clo[s : s + step], chi[s : s + step]
        overlap = np.all(
            (lo_s[:, None, :] - inflate <= chi[None, :, :])
            & (clo[None, :, :] <= hi_s[:, None, :] + inflate),
            axis=2,
        )
        rows, cols = np.nonzero(overlap)
        keep = cols > rows + s  # each unordered pair once, i < j
        ii_parts.append(rows[keep] + s)
        jj_parts.append(cols[keep])
    if not ii_parts:
        return total, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return total, np.concatenate(ii_parts), np.concatenate(jj_parts)


def grid_registration(la, ha, lb, hb, *, min_cell: float = 1e-9):
    """Register two AABB sets (``la``/``ha`` and ``lb``/``hb``, (n, 3)
    each) in one uniform grid and return ``(ia, ka, ib, kb)``: per
    registration the box index and its cell's hash key, so a join on
    equal keys yields every pair of boxes that share a cell — a
    superset of the pairs that overlap, since a box registers in every
    cell it spans. One home for the distance kernel's threshold join
    and the bearing kernel's overlap join.

    The cell starts at the median extent (never below ``min_cell``) and
    doubles until the registrations fit a cap: a few huge boxes (a
    floor slab) would otherwise span thousands of median-sized cells.
    Doubling shrinks spans geometrically, so this terminates fast and
    stays deterministic. The span totals are summed in float64 — int64
    products wrap on extreme extent ratios and would silently break the
    cap, and with it the shared-cell soundness guarantee. The hash is
    the classic spatial one; collisions only add candidates, which the
    caller's exact test removes."""
    ext = np.concatenate([ha - la, hb - lb])
    cell = max(float(np.median(ext)), float(min_cell), 1e-9)
    origin = np.minimum(la.min(axis=0), lb.min(axis=0))
    limit = 8 * (len(la) + len(lb)) + 1024
    while True:
        c0_a = np.floor((la - origin) / cell).astype(np.int64)
        span_a = np.floor((ha - origin) / cell).astype(np.int64) - c0_a + 1
        c0_b = np.floor((lb - origin) / cell).astype(np.int64)
        span_b = np.floor((hb - origin) / cell).astype(np.int64) - c0_b + 1
        total = float(span_a.astype(np.float64).prod(axis=1).sum()) + float(
            span_b.astype(np.float64).prod(axis=1).sum()
        )
        if total <= limit:
            break
        cell *= 2.0

    def register(c0, span):
        counts = span.prod(axis=1)  # safe: the cap bounds every product
        idx = np.repeat(np.arange(len(c0), dtype=np.int64), counts)
        off = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        syz = np.repeat(span[:, 1] * span[:, 2], counts)
        sz = np.repeat(span[:, 2], counts)
        cx = np.repeat(c0[:, 0], counts) + off // syz
        cy = np.repeat(c0[:, 1], counts) + (off % syz) // sz
        cz = np.repeat(c0[:, 2], counts) + off % sz
        return idx, cx * 73856093 ^ cy * 19349663 ^ cz * 83492791

    ia_r, ka = register(c0_a, span_a)
    ib_r, kb = register(c0_b, span_b)
    return ia_r, ka, ib_r, kb


def last_uses(ii: np.ndarray, jj: np.ndarray, k: int) -> np.ndarray:
    """Per copy, the index of the last pair that uses it (``-1`` if none):
    the per-copy world memos of the self paths are dropped after it, so a
    dense scatter never holds every copy's geometry at once."""
    last = np.full(k, -1, dtype=np.int64)
    steps = np.arange(len(ii), dtype=np.int64)
    np.maximum.at(last, ii, steps)
    np.maximum.at(last, jj, steps)
    return last


def release_after(memo: dict, last: np.ndarray, step: int, i: int, j: int) -> None:
    for c in (i, j):
        if last[c] == step:
            memo.pop(c, None)
