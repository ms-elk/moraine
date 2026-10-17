"""``count`` (DESIGN 5.2): an output's copy count — an instanced
output's transforms, a plain mesh counting one — equals ``value`` or
lies in a ``min``/``max`` band; ``measured`` rides every evaluable
entry. The independent check on ``array`` and ``scatter``.
"""

from __future__ import annotations

from ...core.mesh import Image, Instances
from . import common


def check(entry: dict, a: dict, ctx: common.Context) -> None:
    """Fill a count entry (DESIGN 5.2): the output's copy count — an
    instanced output's transforms, a plain mesh counting one — must equal
    ``value`` or lie in the ``min``/``max`` band (inclusive). ``measured``
    rides every evaluable entry; an image output is unevaluatable."""
    common.dimension_requirement(entry, a, None)
    name = a["outputs"][0]
    value = ctx.values[name]
    if isinstance(value, Image):
        entry["ok"] = False
        entry["fix"] = f"output '{name}' is an image; assertions apply to geometry"
        return
    measured = len(value.transforms) if isinstance(value, Instances) else 1
    entry["measured"] = measured
    miss = common.dimension_verdict(entry, measured)
    entry["ok"] = miss is None
    if miss is not None:
        advice = common.owner_advice(
            entry, "change the array/scatter count", "state the intended count",
            knob="number",
        )
        entry["fix"] = (
            f"output '{name}' has {measured} "
            f"{'copy' if measured == 1 else 'copies'} {miss}; {advice}"
        )
