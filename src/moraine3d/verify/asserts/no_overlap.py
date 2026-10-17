"""``no_overlap``, the built-in overlap guard (DESIGN 5.2): never
authored (the parse-side registry's built-in entries are injected),
it checks every geometry-output pair and every copy pair of each
instanced output for material interpenetration: A ∩ B (manifold3d
kernel — exact, never a bounds approximation) must not be deeper than
the depth bar, where depth is the intersection solid's mean thickness
over its load-bearing faces (``depth.overlap_depth``) — flush contact of
any area passes. An AABB broad phase culls provably disjoint pairs;
failing pairs only are listed. Image outputs are outside its domain; a
non-watertight output fails it with the reason unless AABB
disjointness already proves every one of its pairs clear, and a
non-watertight prototype fails the copy pass only when AABB overlap
leaves copy pairs unresolved. Declared embeds cede their pairs, and the
entry reports what it ceded under ``ceded``.
"""

from __future__ import annotations

from ...core.asserts.common import DEPTH_BAR
from ...core.mesh import Instances
from .. import depth
from . import common

_OVERLAP_CRITERION = f"depth tol {DEPTH_BAR:.6g} m"


def _overlap_item(pair) -> str:
    x, y, v, d = pair
    return f"'{x}' and '{y}' overlap ~{d:.4g} m deep ({v:.6g} m^3)"


def _self_pair_fix(self_failing) -> tuple[str, list[list]]:
    """Fix sentence + detail rows for the guard's failing copy pairs."""
    listed = common.listed(self_failing, _overlap_item)
    fix = (
        f"{listed} ({_OVERLAP_CRITERION}); increase "
        f"the array step, reseed the scatter, or shrink the prototype"
    )
    rows = [[x, y, round(v, 12), round(d, 12)] for x, y, v, d in self_failing]
    return fix, rows


def _guard_self(ctx: common.Context, entry: dict, embed_self: set[str]):
    """The guard's copy pass: per-copy pairs of each instanced
    output, minus those a declared embedded entry polices instead
    (reported under the entry's ``ceded.self``)."""
    inst = [n for n in ctx.geometry if isinstance(ctx.values[n], Instances)]
    ceded_here = sorted(n for n in inst if n in embed_self)
    if ceded_here:
        inst = [n for n in inst if n not in embed_self]
        entry.setdefault("ceded", {})["self"] = ceded_here
    checked = 0
    failing: list[tuple[str, str, float, float]] = []
    reasons: list[str] = []
    for n in inst:
        pairs_n, over_n, why = depth.copy_pair_depths(ctx.values[n], n, ctx.proto_solid(n))
        checked += pairs_n
        if why is not None and why not in reasons:
            reasons.append(why)
        failing.extend(
            (f"{n}[{i}]", f"{n}[{j}]", vol, dep)
            for i, j, vol, dep in over_n
            if dep > DEPTH_BAR
        )
    entry["self_pairs"] = checked
    return failing, reasons


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill the guard's entry: see the module docstring; ``pairs`` counts
    the cross pairs checked, ``self_pairs`` the copy pairs, and failing
    pairs ride ``detail.failing_pairs`` as ``[a, b, volume, depth]``."""
    embed_pairs, embed_self = ctx.ceded
    all_pairs = [(x, y) for i, x in enumerate(ctx.geometry) for y in ctx.geometry[i + 1:]]
    pairs = [p for p in all_pairs if frozenset(p) not in embed_pairs]
    if len(pairs) < len(all_pairs):
        entry.setdefault("ceded", {})["pairs"] = [
            list(p) for p in all_pairs if frozenset(p) in embed_pairs
        ]
    failing: list[tuple[str, str, float, float]] = []
    reasons: list[str] = []
    for x, y in pairs:
        if ctx.under_bar(x, y):
            continue
        (mx, why_x), (my, why_y) = ctx.solid(x), ctx.solid(y)
        why = why_x or why_y
        if why is not None:
            if why not in reasons:
                reasons.append(why)
            continue
        vol, dep, bad = depth.overlap_exceeds(mx ^ my)
        if bad:
            failing.append((x, y, vol, dep))
    self_failing, self_reasons = _guard_self(ctx, entry, embed_self)
    entry["pairs"] = len(pairs)
    entry["ok"] = not (failing or reasons or self_failing or self_reasons)
    if not entry["ok"]:
        fix: list[str] = []
        rows = [
            [x, y, round(v, 12), round(d, 12)] for x, y, v, d in failing
        ]
        if failing:
            listed = common.listed(failing, _overlap_item)
            fix.append(
                f"{listed} ({_OVERLAP_CRITERION}); move or "
                f"shrink a part - anchored placement (place) keeps parts "
                f"separated by construction, and an intended overlap is "
                f"a declared 'embedded' assert"
            )
        if self_failing:
            self_fix, self_rows = _self_pair_fix(self_failing)
            fix.append(self_fix)
            rows.extend(self_rows)
        fix.extend(reasons)
        fix.extend(r for r in self_reasons if r not in reasons)
        entry["fix"] = "; ".join(fix)
        if rows:
            entry["detail"] = {"failing_pairs": rows}
