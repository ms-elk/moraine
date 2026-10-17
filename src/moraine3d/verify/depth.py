"""Penetration depth of intersection solids (verify kernel, DESIGN 5.2):
the one measurement behind the overlap guard's bar and ``embedded``'s
band — mean depth over the load-bearing surface of A ∩ B — with the
copy-pair broad phase and exact pass the two share over an instanced
output. The bar itself (DEPTH_BAR) and its rationale live
with the assert vocabulary.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Instances, proto_of, tri_cross
from ..core.asserts.common import DEPTH_BAR
from . import solids
from .pairs import last_uses, overlapping_copy_pairs, release_after, transformed_corner_boxes

# Kernel-exact coplanar residue floor, meters. Intersecting two solids
# whose surfaces share a stored plane (a cutter seated in the pocket it
# cut; flush construction) yields zero-thickness sheets attached to the
# real intersection body: their area joins the 2V/A denominator while
# adding no volume, so identical intersection volumes can read far apart
# in depth. A component or an intersection this thin is residue and
# leaves the measure (DESIGN 5.2). The floor separates scales, not
# defects: residue is kernel-exact (<< 1e-12 m — both operands store
# the identical float32 plane), while storage-noise slabs are ulp-scale
# (>= ~1e-7 m at meter coordinates) and stay in the denominator under
# the DEPTH_BAR rationale.
_RESIDUE_THICKNESS = 1e-8

# Normal cell for grouping a solid's faces by plane direction. Faces of
# one kernel-exact plane agree to ~1e-15 (both operands stored the same
# float32 plane), so a cell nine decades wider never splits a plane;
# within a cell, a plane is the faces whose offsets lie within the
# residue floor of the group's first: two faces closer than 1e-8 m are
# one plane (residue), a storage-noise slab (>= ~1e-7 m) is two.
_NORMAL_CELL = 1e-6


def _plane_groups(cell: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces grouped by plane: ``(order, starts, ends)`` such that
    ``order[starts[i]:ends[i]]`` is one plane — the same normal cell,
    offsets within _RESIDUE_THICKNESS of the group's first. The split is
    by step first; the rare group whose span still exceeds the floor (a
    chain of sub-floor steps) is re-cut from its first offset, so a
    chain never merges planes farther apart than the floor."""
    order = np.lexsort((offset, cell[:, 2], cell[:, 1], cell[:, 0]))
    cell_s, d_s = cell[order], offset[order]
    breaks = np.flatnonzero(
        np.any(cell_s[1:] != cell_s[:-1], axis=1) | (np.diff(d_s) > _RESIDUE_THICKNESS)
    ) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(order)]))
    wide = np.flatnonzero(d_s[ends - 1] - d_s[starts] > _RESIDUE_THICKNESS)
    if len(wide):
        cuts = []
        for g in wide:
            i, e = int(starts[g]), int(ends[g])
            while i < e:
                j = i + int(np.searchsorted(d_s[i:e], d_s[i] + _RESIDUE_THICKNESS, side="right"))
                cuts.append(i)
                i = j
        starts = np.unique(np.concatenate((starts, cuts)))
        ends = np.concatenate((starts[1:], [len(order)]))
    return order, starts, ends


def _residue_free_area(solid) -> float:
    """Surface area of a solid minus its kernel-exact coplanar residue:
    in each plane, the overlap of the faces of one orientation with
    the faces of the other is a zero-thickness sheet, and its area
    leaves on both sides. The overlap is the 2D intersection of the two
    face sets in the plane's own frame (the kernel's 2D boolean), so a
    face only partly covered by a sheet keeps the rest of its area. No
    ray, no erosion: nothing here depends on the solid's position or
    size, and the one tolerance is the plane grouping (_NORMAL_CELL,
    _RESIDUE_THICKNESS)."""
    from manifold3d import CrossSection, FillRule

    mesh = solid.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    if len(f) == 0:
        return 0.0
    tri = v[f]
    cross = tri_cross(tri)
    norms = np.linalg.norm(cross, axis=1)
    ok = norms > 0.0
    if not ok.any():
        return 0.0
    tri, cross, norms = tri[ok], cross[ok], norms[ok]
    areas = 0.5 * norms
    normal = cross / norms[:, None]
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])  # plane: n·x = offset
    # One key per unoriented plane: flip each normal so its largest
    # component is positive; the flip sign is the face's side. The
    # largest component is far from zero, so its sign survives any
    # noise smaller than itself.
    side = np.sign(normal[np.arange(len(normal)), np.argmax(np.abs(normal), axis=1)])
    key_n = normal * side[:, None]
    cell = np.round(key_n / _NORMAL_CELL).astype(np.int64)
    order, starts, ends = _plane_groups(cell, offset * side)
    # Only a plane with faces of both orientations can hold a sheet.
    front_s = side[order] > 0
    n_front = np.add.reduceat(front_s, starts)
    n_back = np.add.reduceat(~front_s, starts)
    mixed = np.flatnonzero((n_front > 0) & (n_back > 0))

    def polygons(idx: np.ndarray, u: np.ndarray, w: np.ndarray) -> list:
        pts = np.stack([tri[idx] @ u, tri[idx] @ w], axis=-1)  # (m, 3, 2)
        # Counterclockwise in the (u, w) frame regardless of side, so
        # the fill rule sees every triangle as covered once.
        twice = (pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1]) - (
            pts[:, 2, 0] - pts[:, 0, 0]
        ) * (pts[:, 1, 1] - pts[:, 0, 1])
        pts[twice < 0] = pts[twice < 0][:, ::-1]
        return list(pts)

    residue = 0.0
    for g in mixed:
        group = order[starts[g] : ends[g]]
        n = key_n[group[0]]
        seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, seed)
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        front = CrossSection(polygons(group[side[group] > 0], u, w), FillRule.NonZero)
        back = CrossSection(polygons(group[side[group] < 0], u, w), FillRule.NonZero)
        residue += 2.0 * abs((front ^ back).area())
    return max(float(areas.sum()) - residue, 0.0)


def overlap_depth(inter, vol: float, area: float) -> float:
    """Mean penetration depth of an intersection solid over its
    load-bearing surface (DESIGN 5.2).

    Coplanar contact (flush construction; a cutter seated in the pocket
    it cut) attaches kernel-exact zero-thickness sheets to the real
    intersection body: their area joins a plain 2V/A denominator while
    adding no volume, so identical intersection volumes read far apart
    in depth. Layered measure: detached sheets leave by component
    (decompose), and sheets fused into a real body's triangulation
    leave by their exact overlap in the plane (_residue_free_area).
    """
    if area <= 0.0:
        return 0.0
    if 2.0 * vol / area < _RESIDUE_THICKNESS:
        return 0.0  # residue only: flush contact, no material to measure
    vol_r = area_r = eff = 0.0
    for c in inter.decompose():
        v_c, a_c = solids.manifold_measures(c)
        if a_c <= 0.0 or 2.0 * v_c / a_c < _RESIDUE_THICKNESS:
            continue  # a detached residue sheet
        vol_r += v_c
        area_r += a_c
        eff += _residue_free_area(c)
    if area_r <= 0.0:
        return 0.0
    return 2.0 * vol_r / eff if eff > 0.0 else 2.0 * vol_r / area_r


def intersection_depth(inter) -> tuple[float, float]:
    """(volume, mean depth) of an intersection solid — the one
    measurement behind the guard's bar and embedded's band; an empty
    intersection is (0, 0)."""
    if inter.is_empty():
        return 0.0, 0.0
    vol, area = solids.manifold_measures(inter)
    return vol, overlap_depth(inter, vol, area)


def overlap_exceeds(inter) -> tuple[float, float, bool]:
    """(volume, mean depth, exceeds) of an intersection solid, against
    the guard's depth bar."""
    vol, depth = intersection_depth(inter)
    return vol, depth, depth > DEPTH_BAR


def under_depth_bar(lo_a, hi_a, lo_b, hi_b) -> np.ndarray:
    """Box pairs whose overlap box is thinner than the depth bar on some
    axis (disjoint pairs included): the intersection sits inside that
    slab, and a solid inside a slab of extent e has 2V/A <= e, so the
    pair is under the guard's bar — and under embedded's defect floor —
    before any solid is built. Conservative boxes keep it sound."""
    ext = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
    return ext.min(axis=-1) < DEPTH_BAR


def copy_pair_candidates(value: Instances):
    """The copy pairs of an instanced output the exact kernel would have
    to resolve — ``(total_pairs, ii, jj, transforms)``: transformed-AABB
    overlap, minus the pairs whose box overlap is under the depth bar on
    some axis (flush-built kits leave most touching pairs there, cleared
    like disjoint ones before any copy solid exists). The broad phase the
    guard's copy pass, ``embedded``'s copy forms, and the free-overlap
    declaration's count share."""
    proto = proto_of(value)
    v = proto.vertices.astype(np.float64)
    xf = np.asarray(value.transforms, dtype=np.float64)
    empty = np.empty(0, dtype=np.int64)
    if len(xf) < 2:
        return 0, empty, empty, xf
    clo, chi = transformed_corner_boxes(v, xf)
    total, ii, jj = overlapping_copy_pairs(clo, chi)
    if len(ii):
        keep = ~under_depth_bar(clo[ii], chi[ii], clo[jj], chi[jj])
        ii, jj = ii[keep], jj[keep]
    return total, ii, jj, xf


def copy_pair_depths(value: Instances, name: str, proto_solid):
    """Copy-pair intersections over one instanced output (DESIGN 5.2).

    Returns ``(pairs_checked, overlapping [(i, j, vol, depth)], why)``
    — every copy pair with a non-empty exact intersection and its
    measured volume/depth; the caller applies its own criterion, so one
    measurement backs both sides of the depth axis (the overlap
    guard's copy pass and embedded's copy forms). The copies stay per-copy
    solids (instancing is never merged — the union `solids.world_manifold`
    builds makes mutual interpenetration invisible): broad phase on
    transformed prototype-AABB corners, the exact kernel on surviving
    pairs. Mirrors the cross-pair watertight policy: a non-watertight
    prototype fails only when AABB overlap leaves copy pairs the exact
    kernel would have to resolve — zero copies, one copy, or provably
    disjoint copies pass without a solid.
    """
    total, ii, jj, xf = copy_pair_candidates(value)
    k = len(xf)
    if len(ii) == 0:
        return total, [], None
    if isinstance(proto_solid, solids.NoSolid):
        # Only the AABB-cleared pairs count as checked; the survivors
        # would need the exact kernel, which needs a solid.
        return int(total - len(ii)), [], solids.no_solid_why(name, proto_solid.status)
    built: dict[int, object] = {}
    last = last_uses(ii, jj, k)

    def copy_solid(i: int):
        if i not in built:
            built[i] = proto_solid.transform(np.ascontiguousarray(xf[i][:3, :4]))
        return built[i]

    overlapping: list[tuple[int, int, float, float]] = []
    for p, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
        inter = copy_solid(i) ^ copy_solid(j)
        if not inter.is_empty():
            overlapping.append((i, j, *intersection_depth(inter)))
        release_after(built, last, p, i, j)
    return total, overlapping, None


def copy_components(k: int, edges) -> list[list[int]]:
    """Connected components of ``k`` copies under ``edges`` (union-find),
    each as a sorted member list, ordered by first member."""
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
