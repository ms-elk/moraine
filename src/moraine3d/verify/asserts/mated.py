"""``mated`` (DESIGN 5.2): two tagged faces, read from evaluated
world-space geometry, register in the entry's mode — ``coincident``
(anti-parallel normals within ``angle_tol_deg``, mating origins within
``tol``) or ``contained`` (coplanar within ``tol``, the part contour
inside the anchor contour); the ``mate`` op's independent check.
``every`` judges each copy of the outputs it names, pairing two named
outputs copy by copy.
"""

from __future__ import annotations

import numpy as np

from ...core import mating
from ...core.asserts import mated as spec
from ...core.mesh import Instances
from . import common

# The mode rides the entry for the same reason owner does: a switch to
# the weaker in-plane registration (contained) is visible in the
# report, not just the file diff.
ECHOED = {"mode": "coincident"}

# The measures a pair is judged on, per mode; every one reads "smaller
# is closer", so the `every` form reports each one's maximum.
_MEASURES = {
    "coincident": ("angle_deg", "distance"),
    "contained": ("angle_deg", "plane_gap", "outside", "crossings"),
}
_VERB = {
    "coincident": "rotate/move the part so the faces coincide - run it through mate",
    "contained": "move the part within the anchor face",
}


def _side(face, every, ctx: common.Context):
    """One face of the entry as the copies it is judged on: ``(specs,
    why)`` — the spec as written, or under ``every`` one indexed spec
    per copy of its output (a plain mesh: its one copy), each with its
    world triangles and mating frame. The first copy without a frame is
    the side's reason."""
    name = face[0]
    value = ctx.values[name]
    instanced = isinstance(value, Instances)
    if name in every and instanced:
        specs = [[name, face[1], i] for i in range(len(value.transforms))]
    else:
        specs = [face]
    out = []
    for s in specs:
        tri, err = common.face_triangles(ctx.values, s)
        if err is not None:
            why = err.message
            if err.code == "missing_instance":
                why += (
                    f', or write "every": ["{name}"] to judge the face on '
                    f"every copy"
                )
            return out, why
        try:
            out.append((s, tri, mating.group_frame(tri)))
        except mating.FrameError as e:
            return out, f"face {common.face_label(s)}: {e.message}; {mating.REMEDY[e.code]}"
    return out, None


def _judge(entry: dict, anchor, part, mode: str, tol: float, ang_tol: float):
    """One pair of faces against the entry's mode: ``(problems, detail,
    registers)`` — the misses as sentences, the measured numbers, and
    whether the mating origins also register within ``tol``."""
    _, tri1, f1 = anchor
    _, tri2, f2 = part
    cosang = float(np.clip(f1.normal @ (-f2.normal), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosang)))
    problems: list[str] = []
    if angle > ang_tol:
        problems.append(
            f"face normals are {angle:.4g} deg from anti-parallel "
            f"(angle_tol_deg {ang_tol:.6g})"
        )
    offset = f2.origin - f1.origin
    distance = float(np.linalg.norm(offset))
    if mode == "contained":
        gap = float(abs(offset @ f1.normal))
        outside, crossings = mating.contained_excursion(tri1, tri2, f1, tol)
        if common.misses(entry, gap - tol):
            problems.append(
                f"the faces are {gap:.4g} m apart along the mating normal "
                f"(tol {tol:.6g} m)"
            )
        if common.misses(entry, outside - tol):
            problems.append(
                f"the part contour leaves the anchor contour by "
                f"{outside:.4g} m (tol {tol:.6g} m)"
            )
        if crossings:
            problems.append(
                f"{crossings} part boundary edges cross the anchor contour "
                f"(the part region leaves the anchor face between interior "
                f"points)"
            )
        detail = {
            "angle_deg": round(angle, 12) + 0.0,
            "plane_gap": round(gap, 12) + 0.0,
            "outside": round(outside, 12) + 0.0,
            "crossings": crossings,
        }
    else:
        if common.misses(entry, distance - tol):
            problems.append(
                f"mating origins are {distance:.4g} m apart (offset "
                f"[{', '.join(f'{x:.4g}' for x in offset)}]; tol {tol:.6g} m)"
            )
        detail = {
            "angle_deg": round(angle, 12) + 0.0,
            "offset": [round(float(x), 12) + 0.0 for x in offset],
            "distance": round(distance, 12) + 0.0,
        }
    return problems, detail, distance <= tol


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill a mated entry (DESIGN 5.2): the two tagged faces, read from
    evaluated world-space geometry (never from an op's transform), must
    satisfy the entry's mode. ``coincident`` (default) — stage (a)
    anti-parallel normals within ``angle_tol_deg``, stage (b)
    mating-frame origins within ``tol``. ``contained`` — the same stage
    (a), then the faces coplanar within ``tol`` along the mating normal
    and ``faces[1]``'s contour inside ``faces[0]``'s outer contour
    (rim slack ``tol``, honored by the crossing test too — a part edge
    seated flush with the rim is containment; a boundary crossing
    beyond the allowance fails outright): the joint
    where a smaller part butts a larger blank face, whose face centroids
    legitimately differ so coincidence would fail correct geometry.
    The frame definition is ``mating.group_frame``, shared with the
    ``mate`` op; a face set without a frame (missing tag, unindexed
    instanced output, non-planar or multi-opening set) fails with the
    reason. ``detail`` carries the measured numbers whenever both frames
    exist — numbers first, pass or fail — and ``origin``, each face's
    frame provenance (``opening`` / ``centroid``): a joint meant to
    register on a bore whose rim lost triangles to its predicate falls
    back to the area centroid silently at execution, so the word is the
    only place the fallback shows. A green ``contained`` entry whose
    origins also register within ``tol`` carries ``stronger:
    "coincident"`` — the entry under-claims, and nothing else reports it.
    ``every`` is ``_every``.
    """
    tol = a.get("tol")
    tol = spec.DEFAULT_TOL if tol is None else tol
    ang_tol = a.get("angle_tol_deg")
    ang_tol = spec.DEFAULT_ANGLE_TOL_DEG if ang_tol is None else ang_tol
    every = a.get("every") or []
    sides = []
    reasons: list[str] = []
    for face in a["faces"]:
        copies, why = _side(face, every, ctx)
        if why is not None:
            reasons.append(why)
        sides.append(copies)
    if every:
        entry["every"] = {
            n: len(v.transforms) if isinstance(v, Instances) else 1
            for n in every for v in [ctx.values[n]]
        }
    if reasons:
        entry["ok"] = False
        entry["fix"] = "; ".join(reasons)
        return
    mode = a.get("mode", "coincident")
    if every:
        _every(entry, a, sides, mode, tol, ang_tol)
        return
    anchor, part = sides[0][0], sides[1][0]
    problems, detail, registers = _judge(entry, anchor, part, mode, tol, ang_tol)
    detail["origin"] = [mating.origin_word(anchor[2]), mating.origin_word(part[2])]
    entry["detail"] = detail
    if mode == "contained" and not problems and registers:
        entry["stronger"] = "coincident"  # a claim, not a verdict: no floor
    entry["ok"] = not problems
    if problems:
        la, lb = (common.face_label(s) for s in a["faces"])
        advice = common.owner_advice(
            entry, _VERB[mode], "raise angle_tol_deg / tol to the intended allowance"
        )
        entry["fix"] = (
            f"faces {la} and {lb} are not mated: " + "; ".join(problems) + f"; {advice}"
        )


def _every(entry: dict, a: dict, sides, mode: str, tol: float, ang_tol: float) -> None:
    """The ``every`` form (DESIGN 5.2): each copy of the named outputs
    judged as the indexed entry would judge it. Two named outputs pair
    copy by copy, so their counts must agree; one named output puts
    every copy against the other face as written. The entry reports the
    copy counts as ``every`` and, under ``detail``, the pair count and
    each measure's maximum over the pairs (``*_max``: the loosest
    copy); failing pairs ride the fix as face labels, all of them under
    ``detail.failing_pairs`` with their own numbers.
    """
    anchors, parts = sides
    faces = a["faces"]
    if all(f[0] in a["every"] for f in faces):
        if len(anchors) != len(parts):
            common.refuse(
                entry,
                f'"every" pairs the copies of its two outputs by index, and '
                f"'{faces[0][0]}' has {len(anchors)} while '{faces[1][0]}' has "
                f"{len(parts)}; make the counts agree, or name one output "
                f'under "every" and index the other',
            )
            return
        pairs = list(zip(anchors, parts))
    else:
        pairs = [(x, y) for x in anchors for y in parts]
    detail: dict = {"pairs": len(pairs)}
    failing: list[tuple[str, str, list[str], dict]] = []
    registers = bool(pairs)
    judged = []
    for anchor, part in pairs:
        problems, numbers, reg = _judge(entry, anchor, part, mode, tol, ang_tol)
        judged.append(numbers)
        registers = registers and reg
        if problems:
            failing.append(
                (common.face_label(anchor[0]), common.face_label(part[0]), problems, numbers)
            )
    if judged:
        for key in _MEASURES[mode]:
            detail[f"{key}_max"] = max(n[key] for n in judged)
        detail["origin"] = [
            "/".join(sorted({mating.origin_word(c[2]) for c in side}))
            for side in (anchors, parts)
        ]
    entry["detail"] = detail
    if mode == "contained" and not failing and registers:
        entry["stronger"] = "coincident"
    entry["ok"] = not failing
    if failing:
        listed = common.listed(
            failing, lambda f: f"{f[0]} and {f[1]}: " + ", ".join(f[2])
        )
        advice = common.owner_advice(
            entry, _VERB[mode], "raise angle_tol_deg / tol to the intended allowance"
        )
        entry["fix"] = (
            f"{len(failing)} of {len(pairs)} copy pairs are not mated: "
            f"{listed}; {advice}"
        )
        detail["failing_pairs"] = [
            {"faces": [la, lb], **{k: n[k] for k in _MEASURES[mode]}}
            for la, lb, _, n in failing
        ]
