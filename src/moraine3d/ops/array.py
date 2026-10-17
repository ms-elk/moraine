"""``array`` operator.

Semantics: linear array of its single input as instances. Params:
``count`` (int, 1..1000000, default 2) copies; per-step TRS
``step_translate`` [x, y, z] meters (default [0, 0, 0]),
``step_rotate_deg`` [rx, ry, rz] (default [0, 0, 0]), ``step_scale``
scalar or [sx, sy, sz] (default 1, components > 0 — a mirroring step
would alternate instance handedness
against the shared prototype's winding), composed as the ``transform``
matrix M; copy i gets M^i (copy 0 is
identity). Mesh input -> Instances of that prototype with ``count``
transforms. Instances input -> composed set: each step matrix is applied
to every existing instance (count x k transforms, same prototype), so
nested arrays form grids without merging (Day-One Invariant 6). Emits
prototype + transforms, never baked geometry. Deterministic.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Instances, Mesh
from . import common, transform as transform_op

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "count": numbers.INT,
    "step_translate": numbers.FLOAT,
    "step_rotate_deg": numbers.FLOAT,
    "step_scale": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
# Documented ceiling on copies: above it the transform stack alone is
# tens of MB and the evaluation time is unbounded — a typo, not a design.
COUNT_MAX = 1_000_000
ATTRIBUTE_POLICY = "preserved"  # the prototype passes through untouched


def evaluate(inputs: list, params: dict, ctx) -> Instances:
    count = common.integer("array", params, "count", 2, minimum=1, maximum=COUNT_MAX)
    # Validated under their own names first: transform.matrix reports its
    # params as 'translate' / 'rotate_deg' / 'scale', and an error must
    # name the param the author actually wrote.
    step_translate = common.vec3("array", params, "step_translate", [0.0, 0.0, 0.0])
    step_rotate = common.vec3("array", params, "step_rotate_deg", [0.0, 0.0, 0.0])
    # Positive components only (not just non-zero as in transform): a
    # mirroring step would give copies alternating handedness on one
    # shared prototype, whose winding cannot flip per instance.
    step_scale = common.vec3(
        "array", params, "step_scale", [1.0, 1.0, 1.0], positive=True, scalar_ok=True
    )
    step = transform_op.matrix(
        {
            "translate": step_translate.tolist(),
            "rotate_deg": step_rotate.tolist(),
            "scale": step_scale.tolist(),
        },
        op="array",
    )

    steps = np.empty((count, 4, 4))
    steps[0] = np.eye(4)
    for i in range(1, count):
        steps[i] = steps[i - 1] @ step

    src = common.geometry_input("array", inputs, 0)
    if isinstance(src, Instances):
        # (count, 1, 4, 4) @ (1, k, 4, 4) -> flattened count*k transforms.
        combined = (steps[:, None] @ src.transforms[None]).reshape(-1, 4, 4)
        return Instances(prototype=src.prototype, transforms=combined)
    return Instances(prototype=src, transforms=steps)
