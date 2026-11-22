"""``clearance`` (DESIGN 5.2): the minimum surface distance between two
subjects — or between one output and every other geometry output
(``others``) — is at least ``min``, and on the pair form at most
``max`` when the entry states one; ``self`` adds each instanced
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
    "max": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# The gap is a design value, so it has no default (DESIGN 5.2): an
# entry states the least distance it requires.
MIN_UNIT = "the least surface distance, m"
MAX_UNIT = "the greatest surface distance, m"
KEY_MEANING = {
    "outputs": "the two subjects it measures between",
    "others": "one named output kept clear of every other geometry output",
    "self": "per-copy pairs of instanced outputs",
    "min": MIN_UNIT,
    "max": MAX_UNIT,
}
SUBJECT = "measures between its two named outputs"
# `every` on clearance has a specific answer: the least gap already
# holds per copy, copy spacing is `self`, and the upper bound has no
# per-copy form here. `tol` names what the kind does not have, an
# allowance.
REFUSALS = {
    "tol": (
        "'tol' is an allowance on the kinds that measure against a "
        "target; clearance states the least gap, in meters, as 'min'"
    ),
    "every": (
        "'every' does not apply to clearance - the least gap against "
        "another output already holds for every copy (the union minimum is "
        'the per-copy minimum); for spacing between copies use "self": '
        "true, and for every copy within a distance of the other output "
        "use contact's \"every\" with that 'tol'"
    ),
}
# The forms that take no `max` (DESIGN 5.2), each with what its upper
# bound would have to quantify over.
_NO_MAX = {
    "others": (
        "'max' bounds the gap between two named outputs; with \"others\" "
        "it would hold every other output within reach - write the pair "
        "that must stay close as its own entry"
    ),
    "self": (
        "'max' bounds the gap between two named outputs; an upper bound "
        "on the spacing between copies has no spelling - \"self\" holds "
        "copy pairs to 'min' alone, so write the pair's band and the copy "
        "spacing as two entries"
    ),
}


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a clearance entry (DESIGN 5.2): the pair form
    names 2 distinct subjects; the one-vs-rest form (``others``) and the
    copy form (``self`` alone) name one; ``self`` refuses a group (copy
    pairs belong to one output); ``min`` is stated and positive;
    ``max``, the pair form's alone and never beside ``self``, exceeds
    it."""
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
    top = a.get("max")
    if top is None:
        return
    for key, why in _NO_MAX.items():
        if a.get(key):
            raise AssetError("bad_assert", f"assert '{aid}': {why}")
    if not common.is_number(top) or top <= gap:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'max' must be a finite number above 'min' "
            f"({gap:g}): the band [min, max] holds the closest approach "
            f"({MAX_UNIT})",
        )
