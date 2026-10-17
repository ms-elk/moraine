"""``contact`` (DESIGN 5.2): the minimum surface distance between two
subjects — measured on world-space triangles, crossing surfaces
counting as 0 — is at most ``tol`` (default the parse side's
DEFAULT_TOL), both bounds inclusive; with the overlap guard this
asserts resting contact. An instanced subject is the union of its
copies, so the plain form is existential (one near copy passes for
all); ``every`` checks each copy of the named outputs and reports the
checked counts (a group's plain member is its one copy, judged apart
from the other members), and an entry left existential over instanced
outputs names them under ``unchecked.instanced_copies``; one whose
``every`` judged no more than the union also draws a top-level warning. ``distance`` rides every evaluable entry, pass or fail.
``min_area`` is the bearing form: the face area the two subjects
share within ``tol`` along the normal (``bearing.shared_area``) must
be at least that many square meters, so an edge or corner contact,
distance 0 like a face, no longer passes for a seat; the entry then
carries ``area`` and, under ``every``, the copy sharing least as
``area_min``. The storage-noise floor applies to the side length of
the area (the square roots are compared).
"""

from __future__ import annotations

import json
import math

from ...core.asserts import contact as spec
from ...core.sites import where
from ..bearing import shared_area
from ..distance import (
    copy_bounds,
    copy_source,
    cull_source,
    exact_surface_distance,
    surface_distance,
    surface_sample_bound,
)
from . import common


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill a contact entry: see the module docstring; ``every`` names
    failing copies as ``name[i]`` (``detail.failing_copies``) and the
    loosest copy rides the entry as ``every_max``."""
    s = common.PairScope(ctx, entry, a, spec.DEFAULT_TOL)
    tol = s.tol
    if s.why is not None:
        entry["ok"] = False
        entry["fix"] = s.why
        return
    fix: list[str] = []
    detail: dict = {}
    every = a.get("every") or []
    every_names = [m for n in every for m in ctx.members(n)]

    def side(n: str) -> list[str]:
        return s.mem_a if n in s.mem_a else ctx.members(s.name_b)

    # A plain output named alone is its side: the union check is its
    # one copy. A group's member is judged apart from the other members.
    quantified = [n for n in every_names if s.srcs[n][1] is not None or len(side(n)) > 1]
    min_area = a.get("min_area")
    if min_area is not None:
        # The union's shared area rides every bearing entry; it is the
        # verdict's measure where no instanced side is quantified.
        entry["min_area"] = min_area
        area = sum(
            _area_between(s, p, q, tol)
            for p in s.mem_a for q in ctx.members(s.name_b)
        )
        entry["area"] = round(area, 12)
        if not quantified and _short(entry, area, min_area):
            advice = common.owner_advice(
                entry, "seat the part on a face", "lower min_area", knob="min_area"
            )
            fix.append(
                f"'{s.name_a}' and '{s.name_b}' share {area:.4g} m^2 of face "
                f"(min_area {min_area:.6g} m^2); an edge or corner contact has "
                f"no area - {advice}"
            )
    if not quantified:
        # The union check; when `every` quantifies an instanced side the
        # per-copy minima below imply it (their minimum is the union's).
        broad = s.pair_broad(s.name_a, s.name_b)
        best = s.pair_best(broad)
        if best is not None and best <= tol:
            entry["distance"] = round(best, 12)  # exact: within the inflation
        else:
            exact = s.pair_exact(broad)
            entry["distance"] = round(exact, 12)
            if common.misses(entry, exact - tol):
                advice = common.owner_advice(entry, "close the gap", "set an explicit tol")
                fix.append(
                    f"closest approach between '{s.name_a}' and '{s.name_b}' is "
                    f"{exact:.4g} m (contact tol {tol:.6g} m); {advice}"
                )
    if every:
        counts: dict[str, int] = {}
        failing_copies: list[tuple[str, float]] = []
        loosest: tuple[str, float] | None = None  # the copy farthest from touching
        closest: float | None = None  # the union's closest approach
        bare: list[tuple[str, float]] = []  # copies sharing too little face
        least: tuple[str, float] | None = None  # the copy sharing least
        for n in every_names:
            src = s.srcs[n]
            if n not in quantified:
                # Its one copy is the union: the pair check above, or,
                # where the other side is quantified, what that side's
                # per-copy checks imply.
                counts[n] = 1
                continue
            plain = src[1] is None

            def copy(i: int, src=src, n=n, plain=plain):
                if plain:
                    return src, s.bounds[n], n
                return copy_source(src, i), copy_bounds(src, i), f"{n}[{i}]"

            # The other side's members (a group's union is the minimum
            # over them; an output is its own one member).
            others = ctx.members(s.name_b) if n in s.mem_a else s.mem_a
            k = 1 if plain else len(src[1])
            counts[n] = k
            # The other output culled once to within tol of this output's
            # union box (sound for every copy's threshold test); each
            # copy culls that again by its own box.
            lo, hi = s.bounds[n]
            near = {q: cull_source(s.srcs[q], lo - tol, hi + tol) for q in others}
            pending: list[tuple[int, dict[str, float | None]]] = []
            for i in range(k):
                cs, cb, label = copy(i)
                if min_area is None:
                    ds = {q: surface_distance(cs, cb, near[q][0], near[q][1], tol) for q in others}
                else:
                    # This copy's triangles within tol of each partner's
                    # box, built once and read by both measures (the
                    # threshold's minimizer lies inside that box, so the
                    # distance over the culled set is the copy's).
                    shared, ds = 0.0, {}
                    for q in others:
                        cq, cqb = cull_source(cs, s.bounds[q][0] - tol, s.bounds[q][1] + tol)
                        tq = cull_source(near[q][0], cb[0] - tol, cb[1] + tol)[0][0]
                        shared += shared_area(cq[0], tq, tol)
                        ds[q] = surface_distance(cq, cqb, near[q][0], near[q][1], tol)
                    if least is None or shared < least[1]:
                        least = (label, shared)
                    if _short(entry, shared, min_area):
                        bare.append((label, shared))
                found = [d for d in ds.values() if d is not None]
                d = min(found) if found else None
                if d is None or d > tol:
                    pending.append((i, ds))
                    continue
                if loosest is None or d > loosest[1]:
                    loosest = (label, d)  # exact: within the inflation
                if closest is None or d < closest:
                    closest = d
            if pending:
                # Exact distances for the message: one cull of the other
                # side at the largest proven reach serves every failing
                # copy (each minimizer lies within its own bound of its
                # copy, hence inside the union box inflated by the max);
                # every (copy, member) pair searches under its own bound.
                uppers = {
                    (i, q): ds[q] if ds[q] is not None
                    else surface_sample_bound(copy(i)[0], s.srcs[q])
                    for i, ds in pending for q in others
                }
                reach = max(uppers.values()) * (1.0 + 1e-9) + 1e-12
                wide = {q: cull_source(s.srcs[q], lo - reach, hi + reach) for q in others}
                for i, _ds in pending:
                    cs, cb, label = copy(i)
                    exact = min(
                        exact_surface_distance(
                            cs, cb, wide[q][0], wide[q][1], upper=uppers[(i, q)],
                        )
                        for q in others
                    )
                    if common.misses(entry, exact - tol):
                        failing_copies.append((label, exact))
                    if loosest is None or exact > loosest[1]:
                        loosest = (label, exact)
                    if closest is None or exact < closest:
                        closest = exact
        entry["every"] = counts
        if closest is not None and "distance" not in entry:
            # The union's minimum is the per-copy minimum, every copy
            # measured exactly: the entry carries `distance` like an
            # unquantified one (DESIGN 5.2).
            entry["distance"] = round(closest, 12)
        if loosest is not None:
            # The loosest copy rides the entry pass or fail (DESIGN
            # 5.2): on a green entry it is the contact closest to
            # not being one.
            entry["every_max"] = [loosest[0], round(loosest[1], 12)]
        if failing_copies:
            listed = common.listed(failing_copies, lambda f: f"'{f[0]}' is {f[1]:.4g} m away")
            advice = common.owner_advice(entry, "close the gaps", "set an explicit tol")
            fix.append(
                f"{len(failing_copies)} copies not touching the other output: "
                f"{listed} (contact tol {tol:.6g} m); {advice}"
            )
            detail["failing_copies"] = [[c, round(d, 12)] for c, d in failing_copies]
        if least is not None:
            # The copy sharing least rides the entry pass or fail: on a
            # green entry it is the seat closest to not being one.
            entry["area_min"] = [least[0], round(least[1], 12)]
        if bare:
            listed = common.listed(bare, lambda f: f"'{f[0]}' shares {f[1]:.4g} m^2")
            advice = common.owner_advice(
                entry, "seat the copies on a face", "lower min_area", knob="min_area"
            )
            fix.append(
                f"{len(bare)} copies share less than min_area {min_area:.6g} m^2 "
                f"of face with the other output: {listed}; an edge or corner "
                f"contact has no area - {advice}"
            )
            detail["bare_copies"] = [[c, round(v, 12)] for c, v in bare]
    common.unchecked_instanced(entry, "instanced_copies", [n for n in s.inst if n not in every_names])
    common.conclude(entry, fix, detail)


def _area_between(s: common.PairScope, p: str, q: str, tol: float) -> float:
    """The face area members ``p`` and ``q`` share, each culled to within
    ``tol`` of the other's box first (a minimizer of the overlap lies in
    both boxes, so the cull drops nothing that bears)."""
    (lo_p, hi_p), (lo_q, hi_q) = s.bounds[p], s.bounds[q]
    tp = cull_source(s.srcs[p], lo_q - tol, hi_q + tol)[0][0]
    tq = cull_source(s.srcs[q], lo_p - tol, hi_p + tol)[0][0]
    return shared_area(tp, tq, tol)


def _short(entry: dict, area: float, min_area: float) -> bool:
    """Whether ``area`` misses ``min_area`` beyond the storage-noise
    floor, applied to the side length (a face 1e-6 m short on one side
    is the same float32 residue a length comparison forgives)."""
    return common.misses(entry, math.sqrt(min_area) - math.sqrt(max(area, 0.0)))


def _quantified_a_side(entry: dict) -> bool:
    """Whether the entry's ``every`` judged anything apart from the
    union: more than one copy of an output, or the members of a group
    one by one."""
    grouped = {
        m for members in (entry.get("members") or {}).values()
        if len(members) > 1 for m in members
    }
    return any(k > 1 or n in grouped for n, k in (entry.get("every") or {}).items())


def report_warnings(entries: list[dict]) -> list[dict]:
    """Advisory rows for contact entries left existential over instanced
    outputs (DESIGN 5.2): the entry's ``unchecked.instanced_copies``
    already names them, but a key inside a green entry is read only by
    whoever reads every entry, and the weakening it marks arrives with
    no file diff — an output arrayed after the entry was written turns
    "touches its support" into "some copy touches". The report's
    warnings say so at the top level. An entry whose ``every`` judged
    one side copy by copy chose its quantifier, the other side's
    existential reading included, and draws no row (an ``every`` over
    one plain output judged the union and nothing else, so its row
    stays); clearance's unchecked copy pairs
    get no row, since its cross-output meaning does not weaken (the
    union minimum is the per-copy minimum)."""
    out: list[dict] = []
    for e in entries:
        if e.get("kind") != "contact":
            continue
        names = (e.get("unchecked") or {}).get("instanced_copies")
        if not names or _quantified_a_side(e):
            continue
        w: dict = {
            "warning": "contact_unquantified",
            "assert": e["assert"],
            "outputs": list(names),
        }
        if "asset" in e:
            w["asset"] = e["asset"]
        w.update(where(e))
        w["note"] = (
            f"contact '{e['assert']}' is existential over instanced "
            f"{list(names)}: one near copy satisfies it for all, and an output "
            f"arrayed after the entry was written weakened it with no file "
            f"diff; write \"every\": {json.dumps(list(names))} if every copy "
            f"must touch"
        )
        out.append(w)
    return out
