"""``embedded`` (DESIGN 5.2): the positive form of the penetration-depth
axis — a declared intent that material interpenetrates, judged in a
``min``/``max`` band on the same load-bearing depth measure as the
overlap guard, over a pair of subjects or over one instanced output's
copy pairs (``self``, ``connected``). A declared embed cedes its pairs
from the guard (``cedes``), which reports what it ceded — never a
silent gap.
"""

from __future__ import annotations

from ...core.asserts import embedded as spec
from ...core.asserts.common import DEPTH_BAR
from ...core.mesh import Image, Instances
from .. import depth
from . import common


def cedes(a: dict, ctx: common.Context) -> tuple[set[frozenset], set[str]]:
    """What one embedded entry polices instead of the guard (DESIGN
    5.2): the pair form cedes its cross-output pair — a group pair every
    member cross pair, since the entry polices the union's intersection,
    which those pairs form — and the copy forms cede that output's copy
    pairs."""
    if len(a["outputs"]) == 2:
        x, y = a["outputs"]
        return (
            {frozenset((p, q)) for p in ctx.members(x) for q in ctx.members(y) if p != q},
            set(),
        )
    return set(), {a["outputs"][0]}


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill an embedded entry (DESIGN 5.2): the declared intent that
    material interpenetrates, judged on the same load-bearing depth
    measure as the overlap guard. Pair form: the two named outputs'
    intersection depth must lie in the declared band (a written min 0
    is the free-overlap spelling: any depth, no overlap included,
    passes, and the pair is still ceded from the guard). Copy forms (one
    output): every copy pair overlapping above the defect floor must
    lie in the band (``self``; its min defaults to 0, so both bounds
    omitted is the pure free-overlap declaration, while a written
    positive min is a positive claim — copies that have all separated
    fail it), and ``connected`` additionally requires the copies to
    form one body through embeds >= min. The entry always carries the
    resolved band — ``max: null`` is the visible spelling of a
    declared-unbounded embed — and the measured depth(s), except the
    pure free-overlap declarations (pair ``min`` 0 without ``max``,
    ``self`` without bounds), which measure nothing: their band admits
    every depth, so the exact intersection would price a number no
    verdict reads; they report ``measured: false`` (the copy form with
    the copy pairs the kernel would have measured, ``candidate_pairs``)
    and need no solid. An unevaluatable entry fails with the reason.
    """
    outs = a["outputs"]
    mn, mx = spec.band(a)
    entry["min"] = mn
    entry["max"] = mx  # None -> JSON null: declared unbounded, visibly
    band = f"[{mn:.6g}, {'unbounded' if mx is None else format(mx, '.6g')}] m"
    if len(outs) == 2:
        name_a, name_b = outs
        if mn == 0.0 and mx is None:
            # The pair form's pure free-overlap declaration: a band
            # admitting every depth reads no verdict, so the intersection
            # (and its erosion pass) is skipped as the copy form's is, and
            # no solid is needed for a claim that polices nothing.
            entry["measured"] = False
            entry["ok"] = True
            return
        (ma, why_a), (mb, why_b) = ctx.solid(name_a), ctx.solid(name_b)
        why = why_a or why_b
        if why is not None:
            common.refuse(entry, why)
            return
        vol, dep = depth.intersection_depth(ma ^ mb)
        entry["depth"] = round(dep, 12)
        low = common.misses(entry, mn - dep)
        high = mx is not None and common.misses(entry, dep - mx)
        entry["ok"] = not (low or high)
        if not entry["ok"]:
            if low:
                advice = common.owner_advice(
                    entry, "deepen the embed",
                    "write 'min': 0 if free overlap is the intent (a depth "
                    "with no design value, e.g. a scatter sunk into displaced "
                    "terrain), or drop the embedded declaration if the parts "
                    "no longer meet",
                    knob="band",
                )
                side = f"under the declared embed min {mn:.6g} m; {advice}"
            else:
                advice = common.owner_advice(
                    entry, "back the part out",
                    "raise the band if the deeper embed is the design",
                    knob="band",
                )
                side = f"over the declared embed max {mx:.6g} m; {advice}"
            entry["fix"] = (
                f"outputs '{name_a}' and '{name_b}' overlap ~{dep:.4g} m "
                f"deep ({vol:.6g} m^3), {side}"
            )
        return
    name = outs[0]
    value = ctx.values[name]
    if isinstance(value, Image):
        common.refuse(entry, f"output '{name}' is an image; assertions apply to geometry")
        return
    if not isinstance(value, Instances):
        # Legal before the first array/scatter node exists, mirroring a
        # blanket 'self': one body, no copy pairs — vacuous.
        entry["self_pairs"] = 0
        entry["ok"] = True
        return
    k = len(value.transforms)
    if mn == 0.0 and mx is None and not a.get("connected"):
        # The pure free-overlap declaration: a band admitting every depth
        # reads no verdict from the kernel, so the exact intersection of
        # every overlapping copy pair would only price a number nobody
        # gates on (the comparison's largest single validate cost). The
        # entry states what it did not measure instead of a measured
        # range, and needs no solid — a non-watertight prototype cannot
        # fail a claim that polices nothing.
        pairs_n, ii, _jj, _xf = depth.copy_pair_candidates(value)
        entry["self_pairs"] = pairs_n
        entry["candidate_pairs"] = int(len(ii))
        entry["measured"] = False
        entry["ok"] = True
        return
    pairs_n, overlapping, why = depth.copy_pair_depths(value, name, ctx.proto_solid(name))
    entry["self_pairs"] = pairs_n
    fix: list[str] = []
    if why is not None:
        fix.append(why)
    # The defect floor keeps flush-contact noise out of the band the
    # same way it bounds the overlap guard's bar.
    material = [t for t in overlapping if t[3] > DEPTH_BAR]
    entry["overlapping"] = len(material)
    if material:
        ds = [d for _, _, _, d in material]
        entry["depth_range"] = [round(min(ds), 12), round(max(ds), 12)]
    elif why is None and mn > 0.0 and k >= 2 and not a.get("connected"):
        # A written floor on `self` is a positive claim: with no copy
        # pair embedding at all it is unmet, not vacuously met (the
        # connected form reports the same situation as k bodies).
        advice = common.owner_advice(
            entry, "close the step", "drop 'min' if free overlap is the intent",
            knob="band",
        )
        fix.append(
            f"no copy pair of '{name}' embeds; expected pairs in the declared "
            f"band {band} - the copies have separated; {advice}"
        )
    band_bad = [
        (i, j, v, d)
        for i, j, v, d in material
        if common.misses(entry, mn - d) or (mx is not None and common.misses(entry, d - mx))
    ]
    if band_bad:
        listed = common.listed(
            band_bad,
            lambda t: f"'{name}[{t[0]}]' and '{name}[{t[1]}]' overlap ~{t[3]:.4g} m deep",
        )
        advice = common.owner_advice(
            entry, "adjust the placement", "adjust the band", knob="band"
        )
        fix.append(f"{listed}, outside the declared band {band}; {advice}")
    comps: list[list[int]] = []
    if a.get("connected") and why is None and k >= 2:
        comps = depth.copy_components(
            k, [(i, j) for i, j, _, d in material if not common.misses(entry, mn - d)]
        )
        if len(comps) > 1:
            entry["components"] = len(comps)
            fix.append(
                f"copies of '{name}' form {len(comps)} bodies through embeds "
                f">= {mn:.6g} m; expected one - bridge the gaps, or split "
                f"the output if separate bodies are the design"
            )
    entry["ok"] = not fix
    if fix:
        entry["fix"] = "; ".join(fix)
        detail: dict = {}
        if band_bad:
            detail["failing_pairs"] = [
                [f"{name}[{i}]", f"{name}[{j}]", round(v, 12), round(d, 12)]
                for i, j, v, d in band_bad
            ]
        if len(comps) > 1:
            detail["components"] = [[len(g), g[0]] for g in comps[:10]]
        if detail:
            entry["detail"] = detail
