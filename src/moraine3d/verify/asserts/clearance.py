"""``clearance`` (DESIGN 5.2): the minimum surface distance between two
subjects — or between one output and every other geometry output the
asset has at validate time (``others``) — is at least ``min``,
inclusive; any touch or pierce fails.
Already universal over an instanced subject (the union minimum is the
per-copy minimum); ``self`` adds every copy pair of each instanced
output in scope (``self_pairs``), and without it the entry names those
outputs under ``unchecked.instanced_self_pairs``. Over one output alone
``self`` is the copy form: that output's copy pairs and nothing else,
so the entry carries no ``distance``. The exact closest
approach rides every entry as ``distance`` (searched once on a green
entry, nearest partner first), with ``nearest`` in the one-vs-rest
form and the closest copy pair as ``self_min``. A fully contained
output is invisible to surface distance — material separation is the
overlap guard's domain.
"""

from __future__ import annotations

from ..distance import closest_copy_pair, self_clearance_pairs
from . import common


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill a clearance entry: see the module docstring; failing partners
    ride ``detail.failing_partners``, failing copy pairs
    ``detail.failing_pairs``."""
    s = common.PairScope(ctx, entry, a, None, key="min")
    gap_min = s.tol
    if s.why is not None:
        common.refuse(entry, s.why)
        return
    fix: list[str] = []
    detail: dict = {}
    nearest: tuple[str, float] | None = None
    failing_pairs: list[tuple[str, float]] = []
    green: list[tuple[float, int, str, dict]] = []
    for order, nb in enumerate(s.partners):
        broad = s.pair_broad(s.name_a, nb)
        best = s.pair_best(broad)
        if best is not None and common.misses(entry, gap_min - best):
            failing_pairs.append((nb, best))  # exact: within the inflation
            if nearest is None or best < nearest[1]:
                nearest = (nb, best)
        else:
            gap = s.pair_gap(s.name_a, nb)
            green.append((gap, order, nb, broad))
    # Green — and the number still rides the entry (DESIGN 5.2): the
    # broad phase only proved "at least min", so the exact search runs
    # here rather than in the reader's tolerance probe — once, on the
    # nearest partner: s.partners in box-gap order (a sound lower bound),
    # stopping when the best exact distance is under the next gap, and
    # not at all when a failing partner is already nearer than min.
    if nearest is None:
        for gap, _order, nb, broad in sorted(green, key=lambda g: g[:2]):
            if nearest is not None and gap >= nearest[1]:
                break
            exact = s.pair_exact(broad)
            if nearest is None or exact < nearest[1]:
                nearest = (nb, exact)
    if nearest is not None:
        entry["distance"] = round(nearest[1], 12)
        if s.others_mode:
            entry["nearest"] = nearest[0]

    def partner_miss(pair) -> str:
        nb, best = pair
        if best <= 0.0:
            return (
                f"outputs '{s.name_a}' and '{nb}' touch or interpenetrate "
                f"(clearance min {gap_min:.6g} m); separate them by at least "
                f"that"
            )
        advice = common.owner_advice(
            entry, "move them apart", "lower 'min' to the intended gap",
            knob="'min'",
        )
        return (
            f"outputs '{s.name_a}' and '{nb}' come within {best:.4g} m "
            f"(clearance min {gap_min:.6g} m); {advice}"
        )

    if failing_pairs:
        fix.append(common.listed(failing_pairs, partner_miss))
    if s.others_mode and failing_pairs:
        # Its own key: the `self` block below fills `failing_pairs`
        # with copy-pair rows of a different arity.
        detail["failing_partners"] = [[nb, round(d, 12)] for nb, d in failing_pairs]
    if a.get("self"):
        checked = 0
        failing: list[tuple[str, str, float]] = []
        closest: tuple[str, str, float] | None = None
        for n in s.inst:
            pairs_n, fail_n, known = self_clearance_pairs(s.srcs[n], gap_min)
            checked += pairs_n
            fail_n = [(i, j, d) for i, j, d in fail_n if common.misses(entry, gap_min - d)]
            failing.extend((f"{n}[{i}]", f"{n}[{j}]", d) for i, j, d in fail_n)
            # The tightest gap rides the entry pass or fail: a failing
            # pair is the closest by construction, a green scope is
            # searched once (DESIGN 5.2), seeded with the pairs the
            # clearance pass already priced.
            near = (
                min(fail_n, key=lambda t: t[2]) if fail_n
                else closest_copy_pair(s.srcs[n], gap_min, known=known)
            )
            if near is not None and (closest is None or near[2] < closest[2]):
                closest = (f"{n}[{near[0]}]", f"{n}[{near[1]}]", near[2])
        entry["self_pairs"] = checked
        if closest is not None:
            entry["self_min"] = [closest[0], closest[1], round(closest[2], 12)]
        if failing:
            listed = common.listed(
                failing, lambda t: f"'{t[0]}' and '{t[1]}' come within {t[2]:.4g} m"
            )
            fix.append(
                f"{listed} (clearance min {gap_min:.6g} m); increase the array "
                f"step, reseed the scatter, or shrink the prototype"
            )
            detail["failing_pairs"] = [[x, y, round(d, 12)] for x, y, d in failing]
    else:
        common.unchecked_instanced(entry, "instanced_self_pairs", s.inst)
    common.conclude(entry, fix, detail)
