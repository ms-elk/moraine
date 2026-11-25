"""``count`` (DESIGN 5.2): how many of one subject there are, against
``value`` or a ``min``/``max`` band; ``measured`` rides every evaluable
entry. The output form reads an output's copies (an instanced output's
transforms, a plain mesh counting one), the independent check on
``array`` and ``scatter``. The face form reads the patches of one
tagged face set, counted once on the prototype.
"""

from __future__ import annotations

from ...core import mating
from ...core.mesh import Image, Instances, proto_of
from ...core.predicates import FaceGroupError, resolve_face_group
from . import common


def measures_no_copies(a: dict) -> bool:
    """Whether this entry has a measure for an output that is declared
    and has no copies (DESIGN 5.2): the output form counts 0, where the
    face form, like every other kind, has nothing to read and the entry
    fails as absent."""
    return "outputs" in a


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill a count entry (DESIGN 5.2): the subject's count must equal
    ``value`` or lie in the ``min``/``max`` band (inclusive).
    ``measured`` rides every evaluable entry; an image output is
    unevaluatable."""
    common.dimension_requirement(entry, a, None)
    if "faces" in a:
        _patches(entry, a, ctx)
        return
    name = a["outputs"][0]
    if name in ctx.empty:
        measured = 0
    else:
        value = ctx.values[name]
        if isinstance(value, Image):
            entry["ok"] = False
            entry["fix"] = f"output '{name}' is an image; assertions apply to geometry"
            return
        measured = len(value.transforms) if isinstance(value, Instances) else 1
    entry["measured"] = measured
    miss = common.dimension_verdict(entry, measured)
    entry["ok"] = miss is None
    if miss is not None:
        advice = common.owner_advice(
            entry, "change the array/scatter count", "state the intended count",
            knob="number",
        )
        entry["fix"] = (
            f"output '{name}' has {measured} "
            f"{'copy' if measured == 1 else 'copies'} {miss}; {advice}"
        )


def _patches(entry: dict, a: dict, ctx: common.Context) -> None:
    """The face form (DESIGN 5.2): the patches of one tagged face set,
    its faces joined through shared edges. Tags ride the prototype, so
    the set is resolved there and counted once, in its local frame: one
    copy's number on an instanced output, the whole mesh's on a baked
    one. An image, an empty, or a non-finite output is unevaluatable,
    as is a set with no face that has area, so the form never measures
    0."""
    face = a["faces"]
    name, tag = face
    why = ctx.refusal(name)
    if why is not None:
        common.refuse(entry, why)
        return
    value = ctx.values[name]
    try:
        tf = resolve_face_group(proto_of(value), tag, None, subject=f"output '{name}'")
    except FaceGroupError as e:
        common.refuse(entry, common.face_reason(entry, value, face, e))
        return
    label = common.face_label(face)
    measured = mating.patch_count(mating.group_triangles(tf.mesh, tf.faces))
    if measured == 0:
        common.refuse(
            entry,
            f"face set '{label}' holds no face with area ({len(tf.faces)} "
            f"degenerate), so there is no patch to count; fix the geometry "
            f"the tag selects, or the tag's predicate",
        )
        return
    entry["measured"] = measured
    miss = common.dimension_verdict(entry, measured)
    entry["ok"] = miss is None
    if miss is not None:
        advice = common.owner_advice(
            entry, "change the geometry the tag selects", "state the intended count",
            knob="number",
        )
        entry["fix"] = (
            f"face set '{label}' has {measured} "
            f"{'patch' if measured == 1 else 'patches'} {miss}; {advice}"
        )
