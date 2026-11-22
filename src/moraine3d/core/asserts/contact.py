"""``contact`` (DESIGN 5.2): the minimum surface distance between two
subjects is at most ``tol``; ``min_area`` (the bearing form) requires
the face area they share to be at least that many square meters;
``every`` quantifies both over each copy of the named instanced
outputs, and an ``[output, copy]`` side names one copy alone.
"""

from __future__ import annotations

from .. import numbers
from . import common

KEY_KINDS = {
    "outputs": numbers.INT,  # a side's copy index
    "every": numbers.OTHER,
    "tol": numbers.FLOAT,
    "min_area": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# contact default, meters: minimum surface distance <= tol counts as
# touching. A numerics floor owned by the tool, not a design quantity:
# vertices are stored float32, so a correctly flush contact reads as a gap
# of about one ulp of its largest coordinate (~5e-7 m at 6 m, ~4e-6 m at
# 50 m). 1e-5 keeps every asset within ~160 m of the origin safe while
# staying 100x below the 0.1-1 mm band where real modeling errors start
# (DESIGN 5.2).
DEFAULT_TOL = 1e-5
TOL_UNIT = "surface-distance tolerance, m"
AREA_UNIT = "least shared face area, m^2"
KEY_MEANING = {
    "outputs": "the two subjects it measures between",
    "every": "every copy of the named instanced outputs must touch the other output",
    "tol": TOL_UNIT,
    "min_area": AREA_UNIT,
}
SUBJECT = "measures between its two named outputs"


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a contact entry (DESIGN 5.2): ``outputs``
    names 2 distinct subjects (either may be a group, or one copy of an
    instanced output as ``[output, copy]``), ``every``, when present, a
    non-empty subset of the names, none of them a side that names one
    copy, and ``min_area``, when present, a finite number > 0 (a
    written 0 would be the plain form spelled twice)."""
    common.check_pair(aid, a, "contact", raw_outputs, groups, copies=True)
    common.check_names(
        aid, a, "every", "outputs", [common.side_name(o) for o in a["outputs"]],
        "the outputs whose every copy must touch the other",
    )
    for o in a["outputs"]:
        if not isinstance(o, str) and o[0] in (a.get("every") or ()):
            raise common.AssetError(
                "bad_assert_outputs",
                f"assert '{aid}': \"every\" names '{o[0]}', so each of its "
                f"copies is judged; drop the copy index, or take '{o[0]}' out "
                f"of \"every\" to name one copy",
            )
    common.check_tol(aid, a, TOL_UNIT)
    area = a.get("min_area")
    if area is not None and (not common.is_number(area) or area <= 0):
        raise common.AssetError(
            "bad_assert",
            f"assert '{aid}' 'min_area' must be a finite number > 0 ({AREA_UNIT}); "
            f"leave it out to require touch alone",
        )
