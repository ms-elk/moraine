"""``offset`` (DESIGN 5.2): the position of the second named output
less the first's along one axis, each read from the output's
world-space vertex bounds with its copies united, meets a stated
``value`` within ``tol`` or lies in a ``min``/``max`` band.
"""

from __future__ import annotations

from .. import numbers
from ...errors import AssetError
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "measure": numbers.OTHER,
    "value": numbers.FLOAT,
    "tol": numbers.FLOAT,
    "min": numbers.FLOAT,
    "max": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# The `value` form's default allowance, meters: extent's, an offset
# being a stated dimension met at the same scale (DESIGN 5.2).
DEFAULT_TOL = 1e-3
TOL_UNIT = "offset allowance around 'value', m"
KEY_MEANING = {
    "outputs": "the reference output, then the one measured from it",
    "measure": (
        "which position to read on both: min_/max_/center_ x|y|z, or a "
        "pair in the order of the outputs"
    ),
    "value": "the stated offset, m",
    "tol": TOL_UNIT,
    "min": "m",
    "max": "m",
}
SUBJECT = "measures from its first named output to its second"
MEASURES = tuple(
    f"{what}_{axis}" for what in common.POSITIONS for axis in common.AXES
)
REFUSALS = {
    "every": (
        "'every' does not apply to offset - each side is read with its "
        "copies united, and a copy's own offset has no spelling; extent's "
        '"every" holds each copy to a position of its own'
    ),
    "faces": (
        "'faces' does not apply to offset - it names two outputs under "
        "'outputs'; one tagged face set's position is extent's \"faces\" form"
    ),
}


def _check_measure(aid: str, measure) -> None:
    """``measure`` is one position token, read on both outputs, or a
    pair of two different ones along one axis, in the order of
    ``outputs``. A size is refused by name: it is no position."""
    pair = isinstance(measure, list)
    tokens = measure if pair else [measure]
    for t in tokens:
        if isinstance(t, str) and t.startswith("size_"):
            raise AssetError(
                "bad_assert",
                f"assert '{aid}': offset reads positions, and '{t}' is a "
                f"size - extent measures one output's own size, and no kind "
                f"compares the sizes of two outputs",
            )
    if (pair and len(tokens) != 2) or any(t not in MEASURES for t in tokens):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' needs 'measure', one of {list(MEASURES)} - the "
            f"position to read on both outputs, in meters along that axis - "
            f"or a pair of them in the order of 'outputs', where the two "
            f"sides read different positions",
        )
    if not pair:
        return
    first, second = tokens
    if first == second:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'measure' pairs '{first}' with itself; write "
            f"it once (\"measure\": \"{first}\") to read it on both outputs",
        )
    if first[-1] != second[-1]:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'measure' pairs '{first}' with '{second}': an "
            f"offset lies along one axis, so both tokens name the same one",
        )


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of an offset entry (DESIGN 5.2): ``outputs``
    names two disjoint subjects, the reference first (an output or a
    group, its bounds united); ``measure`` one position token or a pair
    along one axis; and one judgment, ``value`` (+ ``tol``, m) or a
    ``min``/``max`` band (m; any edge may be negative, a position
    before the reference)."""
    outs = a.get("outputs")
    if (
        not isinstance(outs, list) or len(outs) != 2
        or not all(isinstance(o, str) for o in outs) or outs[0] == outs[1]
    ):
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}' 'outputs' must list 2 distinct output names, "
            f"the reference first (offset reads the second output's position "
            f"less the first's; one output's own position or size is extent)",
        )
    for name in outs:
        common.known_name(aid, name, raw_outputs, groups)
    common.disjoint_subjects(
        aid, "offset", outs, groups,
        why="an output on both sides is measured from itself",
    )
    _check_measure(aid, a.get("measure"))
    common.check_judgment(
        aid, a, "offset", "a finite length in meters", common.is_number, "offset"
    )
    common.check_tol(aid, a, TOL_UNIT)
