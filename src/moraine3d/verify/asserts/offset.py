"""``offset`` (DESIGN 5.2): the position of the second named output
less the first's along one axis, each read from the output's
world-space vertex bounds with its copies united (a group: its members
united), against a stated ``value`` within ``tol`` or a ``min``/``max``
band; ``measured`` rides every evaluable entry.
"""

from __future__ import annotations

import numpy as np

from ...core.asserts import offset as spec
from ...core.asserts.common import AXES
from . import common

ECHOED = {"measure": common.REQUIRED}

_REPAIR = "move either part"
_WEAKEN = "state the intended offset (value/tol or the band)"


def _position(ctx: common.Context, name: str, token: str):
    """One side's reading: the ``token`` position of the united bounds
    of ``name`` (an output's copies, a group's members), or the reason
    a member cannot be measured."""
    what, axis = token.split("_")
    los, his = [], []
    for m in ctx.members(name):
        why = ctx.refusal(m)
        if why is not None:
            return None, why
        lo, hi = ctx.world_bounds(m)
        los.append(lo)
        his.append(hi)
    at = common.read_bound(
        np.minimum.reduce(los), np.maximum.reduce(his), what, AXES.index(axis)
    )
    return float(at), None


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill an offset entry (DESIGN 5.2): the position of ``outputs[1]``
    less the position of ``outputs[0]``, the reference, each read at
    its own ``measure`` token from actual transformed vertices with the
    copies united, must meet the stated ``value`` within ``tol`` or lie
    in the ``min``/``max`` band, both inclusive. ``measured`` rides
    every evaluable entry, pass or fail. Image, empty, and non-finite
    subjects are unevaluatable and fail with the reason.
    """
    common.dimension_requirement(entry, a, spec.DEFAULT_TOL)
    measure = a["measure"]
    tokens = measure if isinstance(measure, list) else [measure, measure]
    readings = []
    for name, token in zip(a["outputs"], tokens):
        at, why = _position(ctx, name, token)
        if why is not None:
            common.refuse(entry, why)
            return
        readings.append(at)
    measured = readings[1] - readings[0]
    entry["measured"] = round(measured, 12) + 0.0
    miss = common.dimension_verdict(entry, measured, " m")
    entry["ok"] = miss is None
    if miss is not None:
        reference, part = a["outputs"]
        advice = common.owner_advice(entry, _REPAIR, _WEAKEN, knob="number")
        entry["fix"] = (
            f"output '{part}' {tokens[1]} {readings[1]:.6g} m is "
            f"{measured:+.6g} m from output '{reference}' {tokens[0]} "
            f"{readings[0]:.6g} m, {miss}; {advice}"
        )
