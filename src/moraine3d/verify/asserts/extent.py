"""``extent`` (DESIGN 5.2): ``query``'s own measurement made gating —
one ``measure`` of a subject's world-space vertex bounds (an output
with its copies united, a group, or one tagged face set) against a
stated ``value`` within ``tol`` or a ``min``/``max`` band; ``measured``
rides every evaluable entry. ``every`` reads each copy of the subject
on its own instead — a dimension of each part, where the union form
states a dimension of the set — and reports the copy counts and the
measurements' span.
"""

from __future__ import annotations

import numpy as np

from ...core import mating
from ...core.asserts import extent as spec
from ...core.mesh import Instances
from ...core.predicates import FaceGroupError, resolve_face_group
from .. import bounds
from . import common

# The bound read rides the entry, like mated's `mode`.
ECHOED = {"measure": common.REQUIRED}

# The fix advice both forms give (common.owner_advice's repair / weaken).
_REPAIR = "resize or move the geometry"
_WEAKEN = "state the intended dimension (value/tol or the band)"


def _read(lo: np.ndarray, hi: np.ndarray, what: str, ax: int) -> np.ndarray:
    """One ``measure`` along axis ``ax`` of boxes ``lo``/``hi`` — ``(k, 3)``
    arrays give ``(k,)`` numbers, one box gives one."""
    if what == "size":
        return hi[..., ax] - lo[..., ax]
    if what == "min":
        return lo[..., ax]
    if what == "max":
        return hi[..., ax]
    return (lo[..., ax] + hi[..., ax]) / 2.0


def _face_group(value, name: str, tag: str, instance):
    """The tagged face set's local triangles ``(g, 3, 3)`` and, when a
    copy is named, its world matrix — or the resolver's reason."""
    try:
        tf = resolve_face_group(value, tag, instance, subject=f"output '{name}'")
    except FaceGroupError as e:
        return None, None, e.message
    return mating.group_triangles(tf.mesh, tf.faces), tf.matrix, None


def _subject_boxes(a: dict, ctx: common.Context, every: bool):
    """Resolve the entry's subject once for both forms: ``(subject,
    counts, boxes, why)`` — the subject's label, the copy count per
    name, and per name ``(labels, lo, hi)`` with ``lo``/``hi`` of shape
    ``(k, 3)``: one box per copy under ``every`` (a plain mesh: its one
    copy, labeled by its name; a group: every member's copies), else
    one box per name (the copies united, or the one indexed copy of the
    face form). The whole-output refusals (image, empty, non-finite)
    fire first on every form, so one defect has one sentence.
    """
    face = a.get("faces")
    if face is not None:
        subject = f"face set '{common.face_label(face)}'"
        names = [face[0]]
    else:
        name = a["outputs"][0]
        subject = f"output '{name}'"
        names = ctx.members(name)
    counts: dict[str, int] = {}
    boxes: list[tuple[list[str], np.ndarray, np.ndarray]] = []
    for m in names:
        why = ctx.refusal(m)
        if why is not None:
            return subject, counts, boxes, why
        value = ctx.values[m]
        instanced = isinstance(value, Instances)
        k = len(value.transforms) if instanced else 1
        counts[m] = k
        if face is not None:
            # The group is resolved once and gathered once; under `every`
            # its local points are then placed per copy, axis by axis,
            # exactly as the output form's copy boxes are.
            index = (0 if instanced else None) if every else (face[2] if len(face) > 2 else None)
            tri, matrix, why = _face_group(value, m, face[1], index)
            if why is not None:
                return subject, counts, boxes, why
            pts = tri.reshape(-1, 3)
            if every and instanced:
                lo, hi = bounds.transformed_bounds(pts, value.transforms)
            else:
                if matrix is not None:
                    pts = pts @ matrix[:3, :3].T + matrix[:3, 3]
                lo, hi = pts.min(axis=0)[None], pts.max(axis=0)[None]
        elif every and instanced:
            lo, hi = ctx.copy_bounds(m)
        else:
            blo, bhi = ctx.world_bounds(m)
            lo, hi = blo[None], bhi[None]
        labels = [f"{m}[{i}]" for i in range(len(lo))] if instanced and every else [m]
        boxes.append((labels, lo, hi))
    return subject, counts, boxes, None


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill an extent entry (DESIGN 5.2): one ``measure`` of the subject's
    world-space vertex bounds — an output (its copies united, measured
    on actual transformed vertices, never on transformed box corners),
    an output group (its members united the same way) or one tagged
    face set (a copy index required on an instanced
    output, as ``mated`` requires it: a copy is never guessed) — must
    meet the stated ``value`` within ``tol`` or lie in the ``min``/``max``
    band, both inclusive. ``measured`` rides every evaluable entry, pass
    or fail: the number the author would otherwise read with ``query``
    and write into prose. Image, empty, and non-finite subjects are
    unevaluatable and fail with the reason. ``every`` is ``_every``.
    """
    common.dimension_requirement(entry, a, spec.DEFAULT_TOL)
    what, axis = a["measure"].split("_")
    ax = "xyz".index(axis)
    if a.get("every"):
        _every(entry, a, ctx, what, ax)
        return
    subject, _counts, boxes, why = _subject_boxes(a, ctx, every=False)
    if why is not None:
        entry["ok"] = False
        entry["fix"] = why
        return
    lo = np.minimum.reduce([b[1][0] for b in boxes])
    hi = np.maximum.reduce([b[2][0] for b in boxes])
    measured = float(_read(lo, hi, what, ax))
    entry["measured"] = round(measured, 12) + 0.0
    miss = common.dimension_verdict(entry, measured, " m")
    entry["ok"] = miss is None
    if miss is not None:
        advice = common.owner_advice(entry, _REPAIR, _WEAKEN, knob="number")
        entry["fix"] = (
            f"{subject} {a['measure']} measures {measured:.6g} m {miss}; {advice}"
        )


def _every(entry: dict, a: dict, ctx: common.Context, what: str, ax: int) -> None:
    """The ``every`` form (DESIGN 5.2): each copy of the subject measured
    on its own — every copy's exact bounds in the output form (a group:
    every member's copies; a plain mesh: its one copy, vacuous and
    legal), the face set resolved on every copy in the face form — and
    each judged against the entry's requirement. The entry reports the
    copy counts per name as ``every`` and the span of the measurements
    as ``measured_range`` in place of ``measured`` (the union's number
    is not what was judged); failing copies ride the fix as ``name[i]``
    (a plain member by its name) with their number, all of them under
    ``detail.failing_copies``.
    """
    subject, counts, boxes, why = _subject_boxes(a, ctx, every=True)
    if why is not None:
        entry["ok"] = False
        entry["fix"] = why
        return
    entry["every"] = counts
    labels = [label for b in boxes for label in b[0]]
    if not labels:
        entry["ok"] = True  # no copy to judge: vacuous, like a plain mesh's one
        return
    numbers = np.concatenate([_read(lo, hi, what, ax) for _, lo, hi in boxes])
    entry["measured_range"] = [
        round(float(numbers.min()), 12) + 0.0, round(float(numbers.max()), 12) + 0.0,
    ]
    failing: list[tuple[str, float, str]] = []
    for label, measured in zip(labels, numbers.tolist()):
        miss = common.dimension_verdict(entry, measured, " m")
        if miss is not None:
            failing.append((label, measured, miss))
    entry["ok"] = not failing
    if failing:
        listed = common.listed(failing, lambda f: f"'{f[0]}' measures {f[1]:.6g} m {f[2]}")
        advice = common.owner_advice(entry, _REPAIR, _WEAKEN, knob="number")
        n = len(failing)
        entry["fix"] = (
            f"{n} {'copy' if n == 1 else 'copies'} of {subject} miss "
            f"{a['measure']}: {listed}; {advice}"
        )
        entry["detail"] = {
            "failing_copies": [[c, round(d, 12) + 0.0] for c, d, _ in failing]
        }
