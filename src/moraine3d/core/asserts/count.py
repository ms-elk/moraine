"""``count`` (DESIGN 5.2): an output's copy count — an instanced
output's transforms, a plain mesh counting one — equals ``value`` or
lies in a ``min``/``max`` band.
"""

from __future__ import annotations

from .. import numbers
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "value": numbers.INT,
    "min": numbers.INT,
    "max": numbers.INT,
}
KEYS = frozenset(KEY_KINDS)
KEY_MEANING = {
    "outputs": "one output, its copies counted",
    "value": "the stated copy count",
    "min": "copies",
    "max": "copies",
}
SUBJECT = "measures one subject, copies united"
# A count is exact or banded; a refused `tol` says so.
REFUSALS = {
    "tol": (
        "count is exact ('value') or a 'min'/'max' band of copies; "
        "there is no tolerance on a count"
    ),
}


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a count entry (DESIGN 5.2): one output, and
    ``value`` (the exact copy count) or a ``min``/``max`` band — all
    non-negative integers of copies (a plain mesh counts one; a group
    is refused — a sum of copy counts is no output's count)."""
    common.check_one_output(
        aid, a, raw_outputs, "count", "one output's copy count", groups,
        "counts one output's copies",
    )

    def count_number(val) -> bool:
        return isinstance(val, int) and not isinstance(val, bool) and val >= 0

    common.check_judgment(
        aid, a, "count", "a non-negative integer (copies)", count_number, "copy count"
    )
