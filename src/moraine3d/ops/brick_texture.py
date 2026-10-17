"""``brick_texture`` operator.

Semantics: brick pattern in running bond by default, seamlessly
tileable (with odd ``rows`` the top course meets course 0 across the v
seam in the same phase — an odd cycle cannot alternate). ``map``
selects what the node emits — ``"color"`` (3-channel: per-brick-varied
``brick_color`` on brick faces, ``mortar_color`` in the joints) or
``"height"`` (1-channel: 1 on brick faces falling to 0 inside the
joints, for ``normal_map``). Both outputs of one param set are pixel-
aligned, so color and height nodes with identical params always match.

Params: ``seed`` (int, default 0) — per-brick value variation stream;
``size`` — int or [w, h] pixels, 4..2048 (default 256); ``rows`` —
brick courses per tile (default 6, >= 1); ``columns`` — bricks per
course (default 3, >= 1); ``mortar`` — joint thickness as a fraction of
a course height (default 0.08, 0..0.9); ``offset`` — shift of each
course relative to the one below, in brick widths, taken modulo one
brick (default 0.5 = running bond for every ``columns``; 0 = stack
bond); ``variation`` — per-brick value
multiplier range (default 0.15, 0..1: each brick's color is scaled by
1 - variation * U01(brick)); ``chamfer`` — height ramp width from joint
floor to brick face, as a fraction of the joint thickness (default 0.5;
0 = hard step); ``brick_color`` (linear RGB, default [0.45, 0.18,
0.12]); ``mortar_color`` (default [0.78, 0.76, 0.72]).

Deterministic and pure: brick variation hashes (seed, brick row, brick
column) — no node-id salt, so identical params share one cache record.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from ..core.rand import lattice_values
from . import common

# 2: offset is taken modulo 1 (the pattern is periodic in it), fixing
# platform-dependent output for out-of-range offsets.
# 3: brick variation moved to the reserved iz=3 lattice plane; on iz=0 it
# read the very corner values a same-seed noise_texture octave
# interpolates through, so brick brightness tracked the noise field.
# 4: offset is in brick widths, not tile fractions; the old unit shifted
# each course by offset*columns bricks, so the default 0.5 was a running
# bond only for odd columns (even columns gave a stack bond).
OP_IMPL_VERSION = 4
KERNELS = ("core.rand",)
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "seed": numbers.INT,
    "size": numbers.INT,
    "rows": numbers.INT,
    "columns": numbers.INT,
    "mortar": numbers.FLOAT,
    "offset": numbers.FLOAT,
    "variation": numbers.FLOAT,
    "chamfer": numbers.FLOAT,
    "brick_color": numbers.FLOAT,
    "mortar_color": numbers.FLOAT,
    "map": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
SEED_PARAMS = frozenset({"seed"})  # reroll-with-history tier (DESIGN 7.2)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes

# Brick variation hashes on the iz=3 lattice plane — off noise_texture's
# iz=0 octave plane and streaks' iz=1/iz=2, so same-seed maps stay
# independent (the plane registry in rand.lattice_values). The plane is
# part of the output semantics; changing it is an OP_IMPL_VERSION bump.
_VARIATION_PLANE = 3


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "brick_texture"
    seed = common.seed(op, params)
    w, h = common.texture_size(op, params)
    rows = common.integer(op, params, "rows", 6, minimum=1, maximum=256)
    columns = common.integer(op, params, "columns", 3, minimum=1, maximum=256)
    mortar = common.num(op, params, "mortar", 0.08, minimum=0.0, maximum=0.9)
    # offset is in brick widths, so the pattern is periodic in it with
    # period 1 (an integer offset shifts each course by whole bricks);
    # reduce so any finite offset is safe for the int64 lattice math below.
    offset = common.num(op, params, "offset", 0.5) % 1.0
    variation = common.num(op, params, "variation", 0.15, minimum=0.0, maximum=1.0)
    chamfer = common.num(op, params, "chamfer", 0.5, minimum=0.0, maximum=1.0)
    brick_color = common.rgb(op, params, "brick_color", [0.45, 0.18, 0.12])
    mortar_color = common.rgb(op, params, "mortar_color", [0.78, 0.76, 0.72])
    which = common.choice(op, params, "map", "color", ("color", "height"))

    u = ((np.arange(w, dtype=np.float64) + 0.5) / w)[None, :]
    v = ((np.arange(h, dtype=np.float64) + 0.5) / h)[:, None]

    # Course index and in-course fraction; joints sit at course boundaries.
    rv = np.floor(v * rows)
    fv = v * rows - rv
    rvi = rv.astype(np.int64) % rows  # wrapped for tiling / hashing

    # Per-course shift of offset bricks, in brick units; the joint pattern
    # has period one brick in u, so any shift keeps the u seam tileable.
    # Course rows-1 meets course 0 across the v seam with a relative shift
    # of (rows-1)*offset bricks, which alternates only for even rows.
    cu = u * columns + offset * rvi
    ci = np.floor(cu).astype(np.int64) % columns
    fu = cu - np.floor(cu)

    # Distances to the nearest joint centerline, in tile-fraction units,
    # so joints are equally thick in both directions on a square tile.
    dv = np.minimum(fv, 1.0 - fv) / rows
    du = np.minimum(fu, 1.0 - fu) / columns
    joint = mortar / rows  # full joint thickness, tile-fraction units
    edge = np.minimum(np.broadcast_to(dv, (h, w)), np.broadcast_to(du, (h, w))) - joint / 2.0

    ramp = chamfer * joint
    if ramp > 0:
        height = np.clip(edge / ramp, 0.0, 1.0)
    else:
        height = (edge > 0).astype(np.float64)

    if which == "height":
        return Image(pixels=height.astype(np.float32)[..., None])

    u01 = (
        lattice_values(seed, ci, np.broadcast_to(rvi, (h, w)), _VARIATION_PLANE)
        * 0.5 + 0.5
    )
    value = 1.0 - variation * u01
    face = brick_color * value[..., None]
    out = mortar_color + (face - mortar_color) * height[..., None]
    return Image(pixels=out.astype(np.float32))
