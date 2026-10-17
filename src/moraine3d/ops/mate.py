"""``mate`` operator (anchored placement by face coincidence, DESIGN 5.1).

Semantics: positions its part input by bringing two tagged faces into
coincidence — the any-orientation generalization of ``place``, absorbing
the hand-computed translation and rotation every assembly seam otherwise
needs. Inputs: 0 = the anchor base (geometry carrying the ``anchor``
tag), 1 = the part (a mesh or an instance set), optional 2 = the leader
(geometry carrying the ``face`` tag when the part itself does not — a
rider: the part receives the rigid motion that mates the leader, so
sibling outputs of one assembly move together without repeating
arithmetic). Without a leader the ``face`` tag lives on the part. The
output is the *moved part only* — the base (and the leader) flow on
through their own graph edges, so part separation and named outputs are
preserved.

Mating frame (``mating.py``, shared with the ``mated`` assertion):
origin = the centroid of the face set's single opening when it has one
(a port, a tube end, a flange bore — the feature a joint registers on),
else its area-weighted centroid; normal = area-weighted face normal;
secondary axis = world +Y projected into the face plane (+Z for a
face within 1 deg of level). The part is rotated so its face normal opposes the
anchor normal and its secondary axis aligns with the anchor's, rolled by
``roll_deg`` about the anchor normal, then translated origin onto
origin. Mode ``coincident`` (v1's single mode). Each face set must be a
single planar surface (normals within 1 deg of the mean, vertices within
1e-4 m of the plane) with at most one opening — mixed, opposite-facing,
disconnected, or multi-opening sets are structured errors, never a
guessed frame.

Params: ``anchor`` (required; a face group name on input 0), ``face``
(required; a face group name on the leader, or on the part when there
is no leader), ``instance`` (0-based copy index, required when input 0
is instanced, rejected otherwise), ``face_instance`` (the same for the
face-carrying input; on an instanced input the tag's predicate was
evaluated on the prototype in its local frame — the copy index selects
which copy's world-space face mates), ``mode`` = ``"coincident"``, ``roll_deg`` (default
0; explicit roll about the mated normal, right-hand rule). Contract,
verifiable from stats/query: the output's own ``face`` group (when the
part carries it) lies on the anchor plane with centroid = anchor origin
and normal = −anchor normal. Normals, UVs, attributes (TANGENT rotated
as a surface direction), material, and the part's face groups are
preserved — a rigid motion; an instanced part moves as a set (its
transforms are pre-multiplied, prototype untouched). Deterministic and
pure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core import numbers
from ..core import mating
from ..core.mesh import Instances, Mesh
from ..errors import OpError
from . import common
from .transform import apply_to_mesh

OP_IMPL_VERSION = 1
KERNELS = ("core.mating", "core.predicates")
INPUT_ARITY = (2, 3)
INPUT_NAMES = ("base", "part", "leader")
CARRIED_INPUTS = (1,)  # output is the moved part; base and leader only anchor
PARAM_KINDS = {
    "anchor": numbers.OTHER,
    "instance": numbers.INT,
    "face": numbers.OTHER,
    "face_instance": numbers.INT,
    "mode": numbers.OTHER,
    "roll_deg": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # rigid motion of the part input
PLACEMENT = "rigid"  # a motion computed from the two mating frames


def _frame(op: str, tri: np.ndarray, role: str, param: str, name: str) -> mating.Frame:
    try:
        return mating.group_frame(tri)
    except mating.FrameError as e:
        code = "ambiguous_opening" if e.code == "ambiguous_opening" else f"{role}_{e.code}"
        raise OpError(
            code,
            f"{op} {role} '{name}': {e.message}; {mating.REMEDY[e.code]}",
            param=param,
        ) from None


def _anchor_faces(inputs: list, params: dict):
    return common.face_group(
        "mate", inputs[0], params.get("anchor"), params.get("instance"),
        index=0, role="anchor", param="anchor", inst_param="instance",
    )


def _face_faces(inputs: list, params: dict):
    face_index = 2 if len(inputs) > 2 else 1
    return common.face_group(
        "mate", inputs[face_index], params.get("face"), params.get("face_instance"),
        index=face_index, role="face", param="face", inst_param="face_instance",
    )


def _frames(anchor_tf, face_tf, params: dict) -> tuple[mating.Frame, mating.Frame]:
    """The two mating frames of an evaluation: ``(anchor, face)``."""
    anchor = _frame(
        "mate", mating.group_triangles(anchor_tf.mesh, anchor_tf.faces, anchor_tf.matrix),
        "anchor", "anchor", params["anchor"],
    )
    face = _frame(
        "mate", mating.group_triangles(face_tf.mesh, face_tf.faces, face_tf.matrix),
        "face", "face", params["face"],
    )
    return anchor, face


def frame_provenance(inputs: list, params: dict) -> dict:
    """The origin provenance of the frames a successful evaluation used,
    as report words (``mating.origin_word``): ``{"anchor": ..., "face":
    ...}``. The op keeps no state, so the frames are re-derived from the
    same inputs — the words are exactly the execution's. The fallback to
    the area centroid is silent at execution (a bore rim that lost
    triangles to its predicate still mates, off the bore), and the
    paired ``mated`` shares the frame and passes, so this is the one
    report that can say it; ``validate`` carries it under ``mates``."""
    anchor, face = _frames(_anchor_faces(inputs, params), _face_faces(inputs, params), params)
    return {"anchor": mating.origin_word(anchor), "face": mating.origin_word(face)}


def evaluate(inputs: list, params: dict, ctx) -> Mesh | Instances:
    op = "mate"
    anchor_tf = _anchor_faces(inputs, params)
    part = common.geometry_input(op, inputs, 1)
    face_tf = _face_faces(inputs, params)
    common.choice(op, params, "mode", "coincident", ("coincident",))
    roll = common.num(op, params, "roll_deg", 0.0)
    anchor, face = _frames(anchor_tf, face_tf, params)
    m = mating.solve(face, anchor, roll)

    if isinstance(part, Instances):
        # Instancing stays instancing: the whole set moves rigidly, the
        # prototype is untouched (det(m) = +1, so winding never flips).
        return Instances(prototype=part.prototype, transforms=m[None] @ part.transforms)
    # apply_to_mesh moves vertices, normals, and TANGENT but builds a bare
    # Mesh; a two-input op gets no central material/group carry, so the
    # part's identity is re-attached here.
    moved = apply_to_mesh(part, m)
    return replace(
        moved,
        material=part.material,
        attributes=None if not moved.attributes else dict(moved.attributes),
        face_groups=None if not part.face_groups else dict(part.face_groups),
    )
