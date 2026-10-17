"""``clearance`` (DESIGN 5.2): the minimum surface distance between two
subjects — or between one output and every other geometry output
(``others``) — is at least ``min``; ``self`` adds each instanced
output's own copy pairs, and over one output alone it is the spacing
between that output's copies.
"""

from __future__ import annotations

from .. import numbers
from ...errors import AssetError
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "others": numbers.OTHER,
    "self": numbers.OTHER,
    "min": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# The gap is a design value, so it has no default (DESIGN 5.2): an
# entry states the least distance it requires.
MIN_UNIT = "the least surface distance, m"
KEY_MEANING = {
    "outputs": "the two subjects it measures between",
    "others": "one named output kept clear of every other geometry output",
    "self": "per-copy pairs of instanced outputs",
    "min": MIN_UNIT,
}
SUBJECT = "measures between its two named outputs"
# `every` on clearance has a specific answer: the union already holds
# per copy, and copy spacing is `self`. `tol` and `max` name what the
# kind does not have: an allowance, and an upper bound.
REFUSALS = {
    "tol": (
        "'tol' is an allowance on the kinds that measure against a "
        "target; clearance states the least gap, in meters, as 'min'"
    ),
    "max": (
        "'max' does not apply to clearance - it has no upper bound: 'min' "
        "is the least gap; parts that must stay within a distance are "
        "'contact' with a 'tol'"
    ),
    "every": (
        "'every' does not apply to clearance - clearance against another "
        "output already holds for every copy (the union minimum is the "
        'per-copy minimum); for spacing between copies use "self": true'
    ),
}


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a clearance entry (DESIGN 5.2): the pair form
    names 2 distinct subjects; the one-vs-rest form (``others``) and the
    copy form (``self`` alone) name one; ``self`` refuses a group (copy
    pairs belong to one output); ``min`` is stated and positive."""
    common.check_flag(aid, a, "others", "for the two-output form")
    outs = a.get("outputs")
    if not a.get("others") and isinstance(outs, list) and len(outs) == 1:
        # The copy form (DESIGN 5.2): one output, measured between its
        # own copies under its own min, no partner to borrow one from.
        common.check_one_output(
            aid, a, raw_outputs, "clearance",
            "the spacing between one output's copies", groups,
        )
        if a.get("self") is None:
            raise AssetError(
                "bad_assert_outputs",
                f"assert '{aid}' names one output and nothing to measure it "
                f"against: write \"self\": true for the spacing between its "
                f"copies, \"others\": true to keep it clear of every other "
                f"output, or name a second output",
            )
    elif a.get("others"):
        # The one-vs-rest form (DESIGN 5.2): one named output, kept
        # clear of every other geometry output the asset has at
        # validate time.
        common.check_one_output(
            aid, a, raw_outputs, "clearance",
            "the one output kept clear of every other; drop "
            "'others' for a two-output pair",
            groups,
        )
    else:
        common.check_pair(aid, a, "clearance", raw_outputs, groups)
    common.check_flag(aid, a, "self", "to skip per-copy pairs")
    if a.get("self"):
        for o in a["outputs"]:
            if o in groups:
                raise common.group_refusal(
                    aid, "clearance", o, groups,
                    "'self' judges one output's own copy pairs",
                )
    gap = a.get("min")
    if gap is None:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': clearance needs 'min', the least gap in "
            f"meters, as in \"min\": 0.01; the gap is a design value and "
            f"has no default",
        )
    # Distance >= 0 always holds, so a zero gap asserts nothing.
    if not common.is_number(gap) or gap <= 0:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'min' must be a finite number > 0 ({MIN_UNIT})",
        )
