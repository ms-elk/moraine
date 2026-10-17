"""``embedded`` (DESIGN 5.2): the declared intent that material
interpenetrates — the positive form of the penetration-depth axis the
overlap guard polices — judged in a ``min``/``max`` depth band over a
pair of subjects or over one instanced output's copy pairs (``self``,
``connected``).
"""

from __future__ import annotations

from .. import numbers
from ...errors import AssetError
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "self": numbers.OTHER,
    "connected": numbers.OTHER,
    "min": numbers.FLOAT,
    "max": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
KEY_MEANING = {
    "outputs": "the two subjects, or the one output whose copies embed",
    "self": "per-copy pairs of instanced outputs",
    "connected": "the copies of one instanced output form one body through in-band embeds",
    "min": "penetration depth, m",
    "max": "penetration depth, m",
}
SUBJECT = "measures between its two named outputs"
# The band, not a tolerance: a refused `tol` names the right keys.
REFUSALS = {
    "tol": "embedded takes a 'min'/'max' penetration-depth band (m), not 'tol'",
}
# embedded (DESIGN 5.2): the positive form of the penetration-depth
# axis — the declared intent that material interpenetrates. `min`
# default, meters: the "robustly embedded" floor, the repository's own
# separation quantum (the guide's >= 1 mm CSG embed rule) read from the
# inside. A
# stated design embed is always an explicit author-owned band; a written
# pair `min` 0 is the free-overlap spelling (a depth with no design
# value — a scatter sunk into displaced terrain — where any positive
# floor would be a claim about one seed), the `self` form's default
# made explicit; `max` is optional (a parameter-dependent depth has no
# statable ceiling) and its omission is visible in the report as
# `max: null`.
DEFAULT_MIN = 1e-3


def band(a: dict) -> tuple[float, float | None]:
    """The resolved ``[min, max]`` band of an embedded entry (DESIGN
    5.2): ``min`` defaults to 0 on the ``self`` form (the free-overlap
    declaration) and to DEFAULT_MIN elsewhere; ``max`` None is
    declared-unbounded. The one resolver the parser and the checker
    share, so what parse validates and what evaluation judges is the
    same band."""
    mn = a.get("min")
    if mn is None:
        mn = 0.0 if a.get("self") else DEFAULT_MIN
    return mn, a.get("max")


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of an embedded entry (DESIGN 5.2): the pair form
    names 2 distinct outputs (either may be a group — the union's
    intersection depth); the copy forms name 1 output and carry
    ``self`` or ``connected`` (a group is refused: copy pairs belong to
    one output). ``min``/``max`` are the depth band in
    meters — a written ``min`` 0 is the free-overlap spelling on the pair
    form and the ``self`` form's default, while ``connected`` needs a
    positive ``min`` (a link is a positive embed); ``max`` optional but
    above ``min``."""
    outs = a.get("outputs")
    shape = (
        f"assert '{aid}' 'outputs' must list 2 distinct output names (a "
        f"declared embed between them) or 1 name with \"self\" or "
        f"\"connected\": true (its copies overlap by design); embedded "
        f"declares a specific intent, never \"all\""
    )
    if (
        not isinstance(outs, list) or len(outs) not in (1, 2)
        or not all(isinstance(o, str) for o in outs)
        or len(set(outs)) != len(outs)
    ):
        raise AssetError("bad_assert_outputs", shape)
    for o in outs:
        common.known_name(aid, o, raw_outputs, groups)
    copy_form = bool(a.get("self")) or bool(a.get("connected"))
    if len(outs) == 1 and not copy_form:
        raise AssetError("bad_assert_outputs", shape)
    if copy_form and outs[0] in groups:
        raise common.group_refusal(
            aid, "embedded", outs[0], groups,
            "'self'/'connected' judge one output's own copy pairs",
        )
    if len(outs) == 2 and not copy_form:
        common.disjoint_subjects(aid, "embedded", outs, groups)
    if len(outs) == 2 and copy_form:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': \"self\"/\"connected\" take 1 output (its "
            f"own copy pairs); a declared embed between two outputs is the "
            f"pair form without them",
        )
    if a.get("self") and a.get("connected"):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': \"connected\" already includes \"self\"'s "
            f"band over overlapping copy pairs; write one of the two",
        )
    for key in ("min", "max"):
        val = a.get(key)
        if val is not None and (not common.is_number(val) or val < 0):
            raise AssetError(
                "bad_assert",
                f"assert '{aid}' '{key}' must be a finite number >= 0 "
                f"(penetration-depth band, m)",
            )
        if val is not None and 0 < val <= common.DEPTH_BAR:
            raise AssetError(
                "bad_assert",
                f"assert '{aid}' '{key}' ({val:g}) is at or below the defect "
                f"floor ({common.DEPTH_BAR:g} m) the overlap guard treats "
                f"as flush contact, so the band could never be policed; "
                f"state a band edge above the floor, or omit the key",
            )
    mn = a.get("min")
    if mn == 0 and a.get("connected"):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': 'min' 0 asserts no floor - connectivity needs "
            f"a positive embed to link copies (default {DEFAULT_MIN:g})",
        )
    mx = a.get("max")
    if mx is not None:
        floor, _ = band(a)
        if mx <= floor:
            raise AssetError(
                "bad_assert",
                f"assert '{aid}' 'max' ({mx:g}) must exceed the band's "
                f"'min' ({floor:g}) - the band [min, max] is in meters of "
                f"penetration depth",
            )
    common.check_flag(aid, a, "self", "to skip per-copy pairs")
    common.check_flag(aid, a, "connected", "for the pair or self form")
