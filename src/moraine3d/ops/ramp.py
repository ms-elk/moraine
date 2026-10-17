"""``ramp`` operator.

Semantics: maps a 1-channel image through a multi-stop piecewise-linear
gradient — the N-stop generalization of ``colorize``. ``stops`` — two
or more positions in [0, 1], strictly increasing (default [0, 1]);
``values`` — one entry per stop, either all finite numbers (emits a
1-channel Image — grayscale contrast shaping for height / roughness
maps) or all ``[r, g, b]`` linear colors >= 0 (emits a 3-channel
Image); default [0, 1] (the identity grayscale ramp). Input values
clamp to [0, 1]; input at or below the first stop takes the first
value, at or above the last stop the last value; between stops, linear
interpolation. Output is the input's size; a pointwise map, so a
tileable input yields a tileable output. Deterministic and pure.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core.mesh import Image
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "stops": numbers.FLOAT,
    "values": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # no mesh flows through texture nodes


_is_num = common.is_num  # the shared finite-non-bool judgment


def _stops(op: str, params: dict) -> list[float]:
    v = params.get("stops", [0.0, 1.0])
    if (
        not isinstance(v, (list, tuple))
        or len(v) < 2
        or not all(_is_num(s) and 0.0 <= s <= 1.0 for s in v)
    ):
        raise OpError(
            "bad_param",
            f"{op} 'stops' must be a list of >= 2 finite numbers in 0..1",
            param="stops",
        )
    if not all(a < b for a, b in zip(v, v[1:])):
        raise OpError(
            "bad_param",
            f"{op} 'stops' must be strictly increasing - for a hard step, "
            f"separate two stops by ~1/size instead of repeating one",
            param="stops",
        )
    return [float(s) for s in v]


def _values(op: str, params: dict, n_stops: int) -> np.ndarray:
    v = params.get("values", [0.0, 1.0])
    if not isinstance(v, (list, tuple)) or len(v) != n_stops:
        raise OpError(
            "bad_param",
            f"{op} 'values' needs one entry per stop "
            f"({n_stops} stop(s), {len(v) if isinstance(v, (list, tuple)) else 1} value(s))",
            param="values",
        )
    if all(_is_num(e) for e in v):
        return np.asarray(v, dtype=np.float64)  # scalar mode -> (n,)
    if all(
        isinstance(e, (list, tuple))
        and len(e) == 3
        and all(_is_num(c) and c >= 0 for c in e)
        for e in v
    ):
        return np.asarray(v, dtype=np.float64)  # RGB mode -> (n, 3)
    raise OpError(
        "bad_param",
        f"{op} 'values' entries must be all numbers (1-channel output) or "
        f"all [r, g, b] colors >= 0 (3-channel output), not mixed",
        param="values",
    )


def evaluate(inputs: list, params: dict, ctx) -> Image:
    op = "ramp"
    src = common.image_input(op, inputs, 0, channels=1)
    stops = _stops(op, params)
    values = _values(op, params, len(stops))

    # No explicit input clamp needed: np.interp end-clamps to the first /
    # last value outside [stops[0], stops[-1]], which subsumes it.
    p = src.pixels[..., 0].astype(np.float64)
    if values.ndim == 1:
        out = np.interp(p, stops, values)[..., None]
    else:
        out = np.stack(
            [np.interp(p, stops, values[:, c]) for c in range(3)], axis=-1
        )
    return Image(pixels=out.astype(np.float32))
