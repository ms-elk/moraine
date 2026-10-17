"""``tag`` operator (DESIGN 5.1).

Semantics: fixes a face-predicate selection into the graph as a named
face group on its single geometry input — the write side of predicate
addressing (the read side is ``moraine query``, same vocabulary, see
predicates.py). An instanced input is tagged on its prototype and the
copies pass through untouched (tags ride the prototype), so a consumer
can tag a received instanced output at the point of use; the predicate
then reads the prototype in its local frame (``normal_*`` / ``center_*``
before the copy transforms), and a copy's world-space face is addressed
downstream by copy index (``instance`` / ``face_instance``). Geometry,
attributes, and material pass through
unchanged; the output gains ``face_groups[name]`` = the ascending face
indices matched by ``select``, re-evaluated on every upstream change so
the selection never goes stale the way an index list would. The group
rides ``result_hash``/cache/``stats`` (``groups: {name: count}`` —
numerically verifiable) but never the exported GLB.

Params: ``name`` (required, ``[A-Za-z][A-Za-z0-9_-]*``; colliding with
an existing group on the input is an error — a silent overwrite would
hide a stale tag); ``select`` (required predicate object, e.g.
``{"normal_y": {"gt": 0.9}}``); ``expect`` — ``"one_or_more"``
(default) / ``"exactly_one"`` / ``"any"``: empty and ambiguous
selections are structured errors, never silent picks. Downstream,
topology-preserving ops carry groups; topology rebuilds drop them with
a ``groups_dropped`` warning. Deterministic and pure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core import numbers
from ..core import predicates
from ..core.mesh import Instances, FACE_GROUP_NAME_RE, Mesh, proto_of
from ..errors import OpError
from . import common

# 2: degenerate faces (zero normal) are masked out of normal_* tests,
# making the documented "never match a normal test" contract true.
OP_IMPL_VERSION = 2
KERNELS = ("core.predicates",)
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "name": numbers.OTHER,
    "select": numbers.FLOAT,
    "expect": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"
PLACEMENT = "identity"  # groups only; positions untouched


def evaluate(inputs: list, params: dict, ctx) -> Mesh | Instances:
    op = "tag"
    value = common.geometry_input(op, inputs, 0)
    # Tags ride the prototype (DESIGN 5.1): an instanced input is tagged
    # on its prototype and the set passes through untouched, so a consumer
    # can tag a received instanced output at the point of use and address
    # one copy's faces through `instance` / `face_instance` downstream.
    src = proto_of(value)

    name = params.get("name")
    if not isinstance(name, str) or not FACE_GROUP_NAME_RE.fullmatch(name):
        raise OpError(
            "bad_param",
            f"{op} 'name' is required and must match [A-Za-z][A-Za-z0-9_-]* - "
            f"it names the face group downstream nodes reference",
            param="name",
        )
    if src.face_groups and name in src.face_groups:
        raise OpError(
            "group_exists",
            f"{op} input already carries a face group '{name}'; pick another "
            f"name (overwriting would silently hide the earlier tag)",
            param="name",
        )

    select = params.get("select")
    predicates.validate_predicate(select, where=f"{op} 'select'")
    expect = common.choice(op, params, "expect", "one_or_more", predicates.CARDINALITIES)

    mask = predicates.evaluate_predicate(src, select)
    predicates.check_cardinality(int(mask.sum()), expect, select, where=f"{op} 'select'")

    groups = dict(src.face_groups) if src.face_groups else {}
    groups[name] = np.flatnonzero(mask).astype(np.int32)
    tagged = replace(
        src,
        face_groups=groups,
        attributes=None if not src.attributes else dict(src.attributes),
    )
    if isinstance(value, Instances):
        return Instances(prototype=tagged, transforms=value.transforms)
    return tagged
