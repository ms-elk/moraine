"""``streaks`` operator.

Semantics: emits a 1-channel image of vertical weathering streaks (rust
wash, drip stains), seamlessly tileable in both u and v. ``count``
streaks (default 12, 1..64), each at a hashed u position, run down +v
from a hashed head with a short fade-in and a linear fade-out over the
streak length; direction is fixed vertical — orient via the part's UVs.
Background pixels are exactly ``low``; streak cores approach ``high``;
overlaps saturate at exactly ``high`` (defaults 0 / 1 — swap them for
dark streaks on a light map). All pixels lie in
[min(low, high), max(low, high)]. ``length`` — maximum streak length as
a fraction of the tile (default 0.5, in (0, 1]; per-streak jitter
0.5..1x); ``width`` — maximum streak width as a fraction of the tile
(default 0.02, in (0, 0.5]; per-streak jitter 0.7..1x); ``wiggle`` —
horizontal meander amplitude in tile fractions (default 0.01, 0..0.25;
0 = straight);
``seed`` / ``size`` as in ``noise_texture``. Deterministic and pure:
no node-id salt, so two nodes with identical params share one cache
record (DESIGN 4.5).
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from ..core.rand import fade, lattice_values
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ("core.rand",)
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "seed": numbers.INT,
    "size": numbers.INT,
    "count": numbers.INT,
    "length": numbers.FLOAT,
    "width": numbers.FLOAT,
    "wiggle": numbers.FLOAT,
    "low": numbers.FLOAT,
    "high": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
SEED_PARAMS = frozenset({"seed"})  # reroll-with-history tier (DESIGN 7.2)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes

# Streak attributes hash on the iz=1 lattice plane and meander on iz=2,
# keeping both off noise_texture's iz=0 plane for the same seed. The
# planes and the meander period are part of the output semantics;
# changing any is an OP_IMPL_VERSION bump.
_ATTR_PLANE = 1
_DRIFT_PLANE = 2
_DRIFT_PERIOD = 4  # meander noise cells per tile (fixed: v-periodicity)


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "streaks"
    seed = common.seed(op, params)
    w, h = common.texture_size(op, params)
    count = common.integer(op, params, "count", 12, minimum=1)
    if count > 64:
        raise OpError(
            "bad_param",
            "streaks 'count' must be <= 64 (evaluation is per-streak; "
            "tile more material instead)",
            param="count",
        )
    length = common.num(op, params, "length", 0.5, positive=True)
    if length > 1.0:
        raise OpError(
            "bad_param",
            "streaks 'length' must be in 0..1 - a streak longer than the "
            "tile cannot fade out and still wrap seamlessly",
            param="length",
        )
    width = common.num(op, params, "width", 0.02, positive=True)
    if width > 0.5:
        raise OpError(
            "bad_param",
            "streaks 'width' must be in 0..0.5 (wider merges into bands - "
            "raise 'count' instead)",
            param="width",
        )
    wiggle = common.num(op, params, "wiggle", 0.01, minimum=0.0, maximum=0.25)
    low = common.num(op, params, "low", 0.0)
    high = common.num(op, params, "high", 1.0)

    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h

    # Per-streak attributes from the shared lattice hash.
    k = np.arange(count, dtype=np.int64)

    def u01(field: int) -> np.ndarray:
        vals = lattice_values(seed, k, np.full(count, field, np.int64), _ATTR_PLANE)
        return vals * 0.5 + 0.5

    u_k = u01(0)  # center column
    v0_k = u01(1)  # head position
    len_k = length * (0.5 + 0.5 * u01(2))  # length jitter, in [length/2, length)
    amp_k = 0.5 + 0.5 * u01(3)  # peak intensity
    wid_k = width * (0.7 + 0.3 * u01(4))  # width jitter, in [0.7*width, width)

    # Per-streak per-row meander: 1D value noise with a fixed integer
    # period in v, so the u offset is identical at v = 0 and v = 1 and
    # v-tiling survives. Shape (count, h) by broadcasting.
    y = v * _DRIFT_PERIOD
    y0 = np.floor(y).astype(np.int64)
    t = fade(y - y0)[None, :]
    g0 = lattice_values(seed, k[:, None], (y0 % _DRIFT_PERIOD)[None, :], _DRIFT_PLANE)
    g1 = lattice_values(
        seed, k[:, None], ((y0 + 1) % _DRIFT_PERIOD)[None, :], _DRIFT_PLANE
    )
    drift = wiggle * (g0 + (g1 - g0) * t)  # (count, h), in [-wiggle, wiggle]

    # Vertical profile (count, h): fade in over 10% of the length from
    # the head, fade out linearly to the tail. Zero at dv = 0 and for
    # dv >= length, so both sides of the v seam meet at zero — this is
    # why length <= 1 is enforced.
    dv = (v[None, :] - v0_k[:, None]) % 1.0
    pv = np.clip(dv / (0.1 * len_k[:, None]), 0.0, 1.0) * np.clip(
        1.0 - dv / len_k[:, None], 0.0, 1.0
    )

    # Accumulate streak by streak; each iteration must stay vectorized
    # over the full pixel grid (count is capped at 64).
    centered = u[None, :] + 0.5
    shift = u_k[:, None] + drift  # (count, h)
    acc = np.zeros((h, w))
    for i in range(count):
        du = np.abs((centered - shift[i][:, None]) % 1.0 - 0.5)
        ph = np.clip(1.0 - 2.0 * du / wid_k[i], 0.0, 1.0)
        acc += amp_k[i] * pv[i][:, None] * ph

    out = low + np.clip(acc, 0.0, 1.0) * (high - low)
    return Image(pixels=out.astype(np.float32)[..., None])
