"""``place`` operator — anchored placement (DESIGN 5.1).

Semantics: positions its part input by semantic relation to a tagged
anchor instead of agent-derived coordinates — attacking the coordinate-
arithmetic error class from the generation side (the ground checks
attack it from the detection side). Inputs: 0 = the anchor base (a mesh
carrying the tag), 1 = the part (a mesh). The output is the *translated
part only* — the base flows on through its own graph edge, so part
separation and named outputs are preserved.

Anchor frame (canonical, fixed at design time): origin = area-weighted
centroid of the anchor group's faces; resting plane = the maximum vertex
y over those faces. The group must form a single horizontal surface:
vertex y spread above ``COPLANAR_TOL`` is rejected — a mixed-height
anchor would put the centroid between surfaces and rest the part over
empty space. v1 anchors translate only — no rotation.

Params: ``anchor`` (required) — a face group name on input 0 (from a
``tag`` node; unknown names error listing the available groups);
``mode`` — ``"on_top"`` (v1's single mode): translate the part so its
bounds-min-y sits on the resting plane and its x/z bounds-center sits
on the anchor centroid; ``offset`` [x, y, z] meters added after
(default [0, 0, 0]). The anchor's summed face normal must point upward
(y > 0) — resting a part on a downward or vertical face set is a sign
error, reported, never guessed around. Contract (verifiable from
stats): result ``bounds.min.y`` = plane y + offset y; x/z bounds-center
= anchor centroid + offset. Normals, UVs, attributes, material, and
face groups of the part are preserved (pure translation).

Per-instance anchoring (DESIGN §5.1): the
anchor base may be Instances — tags ride the prototype, so pass
``instance: <int>`` (0-based copy index, required in that case — never
guessed) to select the copy whose transformed group frame anchors the
part. The frame is computed in world space (the copy's matrix applied
first), so the contract above holds unchanged. Copy indices are
structural (``array`` copy i gets step^i); an out-of-range index is an
error, not a stale reference. The part input stays mesh-only: an
instanced part is an error with the standard guidance (place the
prototype before the array/scatter node, or transform the instance
set). Deterministic and pure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core import numbers
from ..core import mating
from ..core import predicates
from ..core.mesh import Image, Instances, Mesh, tri_cross
from ..errors import OpError
from . import common

# 2: non-coplanar anchors (faces at different heights) are rejected —
# they let a part rest over empty space.
OP_IMPL_VERSION = 2
KERNELS = ("core.mating", "core.mesh", "core.predicates")
INPUT_ARITY = (2, 2)
INPUT_NAMES = ("base", "part")
CARRIED_INPUTS = (1,)  # output is the translated part; the base only anchors
PARAM_KINDS = {
    "anchor": numbers.OTHER,
    "mode": numbers.OTHER,
    "offset": numbers.FLOAT,
    "instance": numbers.INT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # pure translation of the part input
PLACEMENT = "rigid"  # a translation computed from the anchor and the part

# Height spread allowed inside one anchor group, meters: above float32
# vertex noise at asset scale, below any intentional step between surfaces.
COPLANAR_TOL = 1e-4


def _mesh(op: str, inputs: list, index: int, role: str, hint: str) -> Mesh:
    v = inputs[index]
    if isinstance(v, (Instances, Image)):
        kind = "instanced" if isinstance(v, Instances) else "an image (texture)"
        raise OpError("bad_input", f"{op} input {index} ({role}) is {kind}; {hint}")
    return v


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    op = "place"
    anchor = common.face_group(
        op, inputs[0], params.get("anchor"), params.get("instance"),
        index=0, role="anchor", param="anchor", inst_param="instance",
    )
    part = _mesh(
        op, inputs, 1, "the part",
        f"place needs a mesh - {common.BAKE_HINT}, place the prototype before "
        f"the array/scatter node, or move the whole instance set with transform",
    )
    common.choice(op, params, "mode", "on_top", ("on_top",))
    offset = common.vec3(op, params, "offset", [0.0, 0.0, 0.0])
    # World-space frame for the selected copy. Instance matrices have
    # positive determinant by construction (array/scatter/transform
    # reject mirroring), so winding and the upward test survive.
    tri = mating.group_triangles(anchor.mesh, anchor.faces, anchor.matrix)
    cross = tri_cross(tri)
    normal_sum = cross.sum(axis=0)  # area-weighted (cross norm = 2*area)
    if normal_sum[1] <= 0:
        raise OpError(
            "anchor_not_upward",
            f"{op} anchor '{params['anchor']}' does not face upward (summed normal y = "
            f"{normal_sum[1]:.6g}); on_top rests the part on an upward-facing "
            f"surface - tag an upward face set instead",
            param="anchor",
        )
    plane_y = float(tri[..., 1].max())
    span = plane_y - float(tri[..., 1].min())
    if span > COPLANAR_TOL:
        raise OpError(
            "anchor_not_coplanar",
            f"{op} anchor '{params['anchor']}' spans {span:.6g} m in y - its faces sit at "
            f"different heights, so the centroid would place the part over empty "
            f"space; tag a single horizontal face set (split the tag per surface)",
            param="anchor",
        )
    # The upward gate above guarantees nonzero total area (an all-degenerate
    # selection sums to a zero normal and is rejected there).
    centroid = predicates.area_centroid(tri)

    lo, hi = part.bounds()
    t = np.array(
        [
            centroid[0] - (float(lo[0]) + float(hi[0])) / 2.0,
            plane_y - float(lo[1]),
            centroid[2] - (float(lo[2]) + float(hi[2])) / 2.0,
        ]
    ) + offset
    return replace(
        part,
        vertices=(part.vertices.astype(np.float64) + t).astype(np.float32),
        attributes=None if not part.attributes else dict(part.attributes),
        face_groups=None if not part.face_groups else dict(part.face_groups),
    )
