"""``noise_texture`` operator.

Semantics: procedural fBm value-noise texture. Output: a 1-channel
Image, values mapped linearly from the normalized fBm range [-1, 1] to
``[low, high]``, seamlessly tileable in both u and v. Params: ``seed``
(int, default 0); ``size`` — int or [w, h] pixels, 4..2048 (default
256); ``scale`` — integer lattice cells across the tile (default 4,
>= 1); ``octaves`` (default 4, 1..10); ``gain`` — per-octave amplitude
falloff (default 0.5, > 0); ``low`` / ``high`` — output value range
(defaults 0 / 1); ``mode`` (default ``"fbm"``) — per-octave shaping
applied to each octave value ``n`` in [-1, 1] before summation:
``"fbm"`` keeps ``n``, ``"ridged"`` is ``1 - 2|n|`` (sharp bright
creases at lattice zero-crossings), ``"billow"`` is ``2|n| - 1``
(rounded bright lumps). Every shaped octave stays in [-1, 1], so the
normalized sum and the exact ``[low, high]`` bounds hold in all modes
(no Musgrave-style spectral weighting — it would break the exact-bounds
contract), and the pointwise transform leaves tiling unchanged.
Lacunarity is fixed at 2 so every octave's lattice stays
integer-periodic — that is what makes the tile edge seamless.
Deterministic and pure: no node-id salt, so two nodes with identical
params share one cache record (DESIGN 4.5).
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
# "mode" is additive: cache keys hash the authored params and the "fbm"
# default path is bit-identical to the op before "mode" existed, so its
# introduction is not an output-semantics change (no impl-version bump).
PARAM_KINDS = {
    "seed": numbers.INT,
    "size": numbers.INT,
    "scale": numbers.INT,
    "octaves": numbers.INT,
    "gain": numbers.FLOAT,
    "low": numbers.FLOAT,
    "high": numbers.FLOAT,
    "mode": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
SEED_PARAMS = frozenset({"seed"})  # reroll-with-history tier (DESIGN 7.2)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes

# Fixed per-octave seed derivation (part of the output semantics; changing
# it is an OP_IMPL_VERSION bump).
_OCTAVE_SEED_STRIDE = 1000003


def fbm_grid(
    seed: int, w: int, h: int, scale: int, octaves: int, gain: float,
    mode: str = "fbm",
) -> np.ndarray:
    """Normalized fBm value noise on the (h, w) pixel grid, in [-1, 1]."""
    if mode not in ("fbm", "ridged", "billow"):
        raise ValueError(f"fbm_grid: unknown mode '{mode}'")
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    acc = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        freq = scale << o
        x = u * freq
        y = v * freq
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        tx = fade(x - x0)[None, :]
        ty = fade(y - y0)[:, None]
        # Wrap lattice indices modulo the octave period: tileability.
        x0m, x1m = (x0 % freq)[None, :], ((x0 + 1) % freq)[None, :]
        y0m, y1m = (y0 % freq)[:, None], ((y0 + 1) % freq)[:, None]
        so = seed + _OCTAVE_SEED_STRIDE * o
        if freq * freq < h * w:
            # An octave has only freq^2 distinct lattice values, but the
            # direct path hashes all four full pixel grids (4*h*w mixes
            # per octave). Hash the octave's table once and gather —
            # bit-identical, lattice_values is a pure elementwise hash of
            # already-wrapped indices; an evaluation strategy, not an
            # output-semantics change (pinned by the frozen-hash test).
            table = lattice_values(
                so, np.arange(freq)[None, :], np.arange(freq)[:, None], 0
            )
            v00 = table[y0m, x0m]
            v10 = table[y0m, x1m]
            v01 = table[y1m, x0m]
            v11 = table[y1m, x1m]
        else:
            v00 = lattice_values(so, x0m, y0m, 0)
            v10 = lattice_values(so, x1m, y0m, 0)
            v01 = lattice_values(so, x0m, y1m, 0)
            v11 = lattice_values(so, x1m, y1m, 0)
        top = v00 + (v10 - v00) * tx
        bot = v01 + (v11 - v01) * tx
        n_o = top + (bot - top) * ty
        # Shaping must keep each octave in [-1, 1], and the "fbm" branch
        # must stay the identity (the default output is pinned
        # bit-identical — see the PARAMS note).
        if mode == "ridged":
            n_o = 1.0 - 2.0 * np.abs(n_o)
        elif mode == "billow":
            n_o = 2.0 * np.abs(n_o) - 1.0
        acc += amp * n_o
        total += amp
        amp *= gain
    return acc / total


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "noise_texture"
    seed = common.seed(op, params)
    w, h = common.texture_size(op, params)
    # Capped at the size cap: lattice cells beyond one per pixel are
    # sub-pixel, and unbounded scale would overflow the int64 lattice
    # (raw OverflowError / platform-dependent casts).
    scale = common.integer(op, params, "scale", 4, minimum=1, maximum=2048)
    octaves = common.integer(op, params, "octaves", 4, minimum=1)
    if octaves > 10:
        raise OpError(
            "bad_param",
            "noise_texture 'octaves' must be <= 10 (beyond that every octave "
            "is sub-pixel at supported sizes)",
            param="octaves",
        )
    gain = common.num(op, params, "gain", 0.5, positive=True)
    low = common.num(op, params, "low", 0.0)
    high = common.num(op, params, "high", 1.0)
    mode = common.choice(op, params, "mode", "fbm", ("fbm", "ridged", "billow"))

    n = fbm_grid(seed, w, h, scale, octaves, gain, mode)  # [-1, 1]
    out = low + (n * 0.5 + 0.5) * (high - low)
    return Image(pixels=out.astype(np.float32)[..., None])
