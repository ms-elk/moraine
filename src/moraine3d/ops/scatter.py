"""``scatter`` operator.

Semantics: scatters its second input (the prototype) as instances at
random points on the surface of its first input (the target mesh), the
order ``place`` and ``mate`` take: where, then what.
Params: ``count`` (int, 1..1000000, default 10); ``seed`` (int, default 0);
``align_normal`` (bool, default false) — when true, instance +Y aligns
to the surface normal at the sample point, and a level target leaves the
copy's yaw unchanged (the frame is the identity where the normal is +Y,
so the prototype keeps its +Z front); ``rand_yaw`` (bool, default
false) — adds a uniform random rotation about the instance's up axis;
``scale_jitter`` [lo, hi] uniform random uniform-scale range (default
[1, 1]). Sampling is uniform by triangle area. Stochastic but
deterministic: streams derive as hash(seed, node_id, index) (DESIGN
4.4), so a given node reproduces exactly and two nodes with the same
seed decorrelate. A copy's draws depend on its own index alone, never
on ``count``: raising ``count`` appends copies and leaves the earlier
ones where they were. Output is prototype + transforms (Day-One Invariant
6). The target surface itself is not emitted — reference it as its own
output or input elsewhere.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core import rand
from ..core.mesh import Instances, tri_cross
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 3
KERNELS = ("core.mesh", "core.rand")
INPUT_ARITY = (2, 2)
INPUT_NAMES = ("target", "prototype")
CARRIED_INPUTS = (1,)  # the target (input 0) is sampled, never emitted
USES_NODE_ID = True  # streams are salted with the node id (see rand.py)
PARAM_KINDS = {
    "count": numbers.INT,
    "seed": numbers.INT,
    "align_normal": numbers.OTHER,
    "rand_yaw": numbers.OTHER,
    "scale_jitter": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
SEED_PARAMS = frozenset({"seed"})  # reroll-with-history tier (DESIGN 7.2)
# Documented ceiling on copies (array's): above it the transform stack
# alone is tens of MB and evaluation time unbounded — a typo, not a design.
COUNT_MAX = 1_000_000
ATTRIBUTE_POLICY = "preserved"  # the prototype passes through untouched

# Copy i draws stream k at index i * _STRIDE + k, so no draw depends on
# count. The stride and the stream numbers are part of the output
# semantics: a new stream takes a free number below the stride, and
# changing either one moves every scatter ever seeded.
_STRIDE = 8
_FACE, _BARY_1, _BARY_2, _YAW, _SCALE = range(5)


def evaluate(inputs: list, params: dict, ctx) -> Instances:
    target = common.mesh_input("scatter", inputs, 0, hint=common.BAKE_HINT)
    proto = common.geometry_input("scatter", inputs, 1)
    count = common.integer("scatter", params, "count", 10, minimum=1, maximum=COUNT_MAX)
    seed = common.seed("scatter", params)
    align = common.boolean("scatter", params, "align_normal", False)
    yaw = common.boolean("scatter", params, "rand_yaw", False)
    jit = params.get("scale_jitter", [1.0, 1.0])
    if (
        not common.num_pair(jit)
        or not all(c > 0 for c in jit)
        or jit[0] > jit[1]
    ):
        raise OpError(
            "bad_param", "scatter 'scale_jitter' must be [lo, hi] with 0 < lo <= hi",
            param="scale_jitter",
        )
    if isinstance(proto, Instances):
        raise OpError(
            "bad_input",
            "scatter input 1 (prototype) is already instanced; scatter a mesh prototype",
        )

    v = target.vertices.astype(np.float64)
    tri = v[target.faces.astype(np.int64)]  # (m, 3, 3)
    cross = tri_cross(tri)
    area = np.linalg.norm(cross, axis=1)
    total = area.sum()
    if total <= 0:
        raise OpError("bad_input", "scatter target has zero surface area")
    cdf = np.cumsum(area) / total

    idx = np.arange(count) * _STRIDE
    u_face = rand.uniform(seed, ctx.node_id, idx + _FACE)
    face_i = np.searchsorted(cdf, u_face, side="right").clip(0, len(area) - 1)
    r1 = rand.uniform(seed, ctx.node_id, idx + _BARY_1)
    r2 = rand.uniform(seed, ctx.node_id, idx + _BARY_2)
    su = np.sqrt(r1)
    b0, b1, b2 = 1.0 - su, su * (1.0 - r2), su * r2  # uniform in triangle
    pts = (
        tri[face_i, 0] * b0[:, None] + tri[face_i, 1] * b1[:, None] + tri[face_i, 2] * b2[:, None]
    )

    xf = np.tile(np.eye(4), (count, 1, 1))
    xf[:, :3, 3] = pts

    if align:
        n = cross[face_i] / area[face_i][:, None]
        # Basis: y = n; z = ref x n, x = n x z (right-handed). With ref =
        # world X this is the identity on a level target (n = +Y), so the
        # prototype keeps its +Z front; ref switches to world Z near the
        # +-X poles where X is nearly parallel to n.
        ref = np.where(np.abs(n[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
        z = np.cross(ref, n)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        x = np.cross(n, z)
        xf[:, :3, 0], xf[:, :3, 1], xf[:, :3, 2] = x, n, z
    if yaw:
        ang = rand.uniform(seed, ctx.node_id, idx + _YAW, 0.0, 2.0 * np.pi)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.tile(np.eye(4), (count, 1, 1))
        rot[:, 0, 0], rot[:, 0, 2] = c, s
        rot[:, 2, 0], rot[:, 2, 2] = -s, c
        xf = xf @ rot  # local yaw about the instance's up axis
    if jit != [1.0, 1.0]:
        sc = rand.uniform(seed, ctx.node_id, idx + _SCALE, float(jit[0]), float(jit[1]))
        scale = np.tile(np.eye(4), (count, 1, 1))
        scale[:, 0, 0] = scale[:, 1, 1] = scale[:, 2, 2] = sc
        xf = xf @ scale

    return Instances(prototype=proto, transforms=xf)
