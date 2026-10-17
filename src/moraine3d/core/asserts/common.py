"""What every kind's parse-time shape shares: the name vocabulary
(declared outputs and root output groups), the literal checks, the
numerics floors (the storage-noise floor every length comparison is
judged under; the depth bar the guard and ``embedded`` share), and the
face-tag spelling.
"""

from __future__ import annotations

import math
import re

from ...errors import AssetError
from ..paths import PORTABLE_NAME_FRAGMENT

# The storage-noise floor on every length comparison an assert makes
# against an authored `tol` or band edge, meters (DESIGN 5.2): vertices
# are stored float32 (~1e-7 relative), so a gap built exactly 0.15 wide
# reads 0.149999973 and would miss a clearance `min` 0.15 by 2.7e-8 —
# the shortfall an author otherwise dodges by writing every tolerance as
# 0.1499. A miss under the floor passes and the entry says so
# (`noise_floor`). 1e-6 is one ulp at ~16 m: beyond that a stated
# tolerance is the author's to widen, as contact's default already
# assumes. The overlap guard's bar is not a comparison against an
# authored number and takes no floor.
NOISE_FLOOR = 1e-6

# The overlap guard's depth bar and embedded's defect floor, meters: an
# intersection whose mean penetration depth (2V over its load-bearing
# area) sits at or below it is flush contact, not interpenetration. A
# volume bar was rejected: flush-contact float32 slabs grow in volume
# with the contact area while their thickness stays a few ulps of the
# contact coordinate — micron-scale once coordinates reach ~10 m (a
# 2-ulp slab is ~1.9e-6 m at y = 8) — so a 1 um bar would fail correct
# flush contact at asset scale. The bar is contact's float32 floor, an
# order of magnitude below the 0.1 mm band where real modeling errors
# start (DESIGN 5.2). An embedded band edge written inside (0, bar] is
# refused at parse: the guard ignores overlap that shallow, so such a
# band could never be policed.
DEPTH_BAR = 1e-5

# Face-group names in `faces`: the same portable identifier form as output
# names (mesh.FACE_GROUP_NAME_RE is the same pattern; asset loading stays
# free of the geometry import).
FACE_TAG_RE = re.compile(rf"^{PORTABLE_NAME_FRAGMENT}$")


def is_number(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def known_name(aid: str, name: str, raw_outputs, groups: dict) -> None:
    """``name`` is a declared output or group, else the unknown-name
    refusal that lists both vocabularies."""
    if name in raw_outputs or name in groups:
        return
    listed = f"asset outputs: {sorted(raw_outputs)}"
    if groups:
        listed += f"; groups: {sorted(groups)}"
    raise AssetError(
        "bad_assert_outputs",
        f"assert '{aid}' references unknown output '{name}'; {listed}",
    )


def group_refusal(aid: str, kind: str, name: str, groups: dict, how: str) -> AssetError:
    """The refusal every kind shares where a group cannot stand for its
    members (DESIGN 5.2): the name is a declared group, ``how`` says
    what to write instead — always one entry per member, the members
    listed so the fix is a copy."""
    return AssetError(
        "bad_assert_outputs",
        f"assert '{aid}': '{name}' is an output group (root 'groups'); "
        f"{kind} {how} - write one entry per member: {groups[name]}",
    )


def disjoint_subjects(aid: str, kind: str, outs, groups: dict) -> None:
    """The two names a relating kind takes must stand for disjoint sets
    of outputs (DESIGN 5.2). Distinct names stopped being enough once a
    name can be a group: an output on both sides is measured against
    itself, so the entry's verdict is fixed before the geometry."""
    x, y = outs
    shared = sorted(set(groups.get(x, [x])) & set(groups.get(y, [y])))
    if shared:
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}': '{x}' and '{y}' both stand for {shared}; "
            f"{kind} relates two disjoint subjects - an output measured "
            f"against itself is always touching. Name the outputs outside "
            f"the overlap, or write the relation over member names",
        )


def check_face_spec(aid: str, f, raw_outputs, shape: str, groups: dict, kind: str) -> None:
    """One ``[output, tag]`` / ``[output, tag, instance]`` spec — the
    shape mated's two faces and extent's one face set share; ``shape``
    is the caller's whole-entry message. Tag presence and instance range
    are evaluation facts, judged at validate time."""
    if (
        not isinstance(f, list) or len(f) not in (2, 3)
        or not isinstance(f[0], str) or not isinstance(f[1], str)
        or not FACE_TAG_RE.fullmatch(f[1])
        or (
            len(f) == 3
            and (not isinstance(f[2], int) or isinstance(f[2], bool) or f[2] < 0)
        )
    ):
        raise AssetError("bad_assert_faces", shape)
    if f[0] in groups:
        raise group_refusal(
            aid, kind, f[0], groups, "names a tagged face set of one output"
        )
    if f[0] not in raw_outputs:
        raise AssetError(
            "bad_assert_faces",
            f"assert '{aid}' references unknown output '{f[0]}'; "
            f"asset outputs: {sorted(raw_outputs)}",
        )


def check_one_output(
    aid: str, a: dict, raw_outputs, kind: str, what: str, groups: dict,
    group_how: str | None = None,
) -> None:
    """``outputs`` names exactly one declared output (extent's output
    form, count, clearance's one-vs-rest form); ``what`` says what that
    output is to the kind. A group stands in where the kind unites its
    members; ``group_how`` names why a kind refuses one."""
    outs = a.get("outputs")
    if not isinstance(outs, list) or len(outs) != 1 or not isinstance(outs[0], str):
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}' 'outputs' must name exactly one output ({kind} "
            f"measures {what})",
        )
    known_name(aid, outs[0], raw_outputs, groups)
    if group_how is not None and outs[0] in groups:
        raise group_refusal(aid, kind, outs[0], groups, group_how)


def check_judgment(aid: str, a: dict, kind: str, unit: str, number, band_word: str) -> None:
    """The judgment half the dimension kinds share (DESIGN 5.2): exactly
    one of ``value`` (with an optional ``tol``, extent only — the key table
    keeps it off count) or a ``min``/``max`` band with at least one edge and
    ``max`` above ``min``. ``number`` validates one literal (a finite
    length, or a non-negative integer count) and ``unit`` names it."""
    has_value = "value" in a
    has_band = "min" in a or "max" in a
    if has_value and has_band:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': write 'value' (with 'tol') or a 'min'/'max' "
            f"band, not both - one requirement sentence, one judgment",
        )
    if not has_value and not has_band:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' ({kind}) states no requirement: write 'value' "
            f"(the stated {band_word}) or a 'min'/'max' band ({unit})",
        )
    if "tol" in a and not has_value:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}': 'tol' is the allowance around 'value'; a band "
            f"states its own edges in 'min'/'max'",
        )
    for key in ("value", "min", "max"):
        if key in a and not number(a[key]):
            raise AssetError(
                "bad_assert",
                f"assert '{aid}' '{key}' must be {unit}",
            )
    if "min" in a and "max" in a and a["max"] <= a["min"]:
        same = a["max"] == a["min"]
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'max' ({a['max']:g}) must exceed 'min' "
            f"({a['min']:g})"
            + ("; a band of one number is 'value'" if same else ""),
        )


def check_pair(aid: str, a: dict, kind: str, raw_outputs, groups: dict) -> None:
    """``outputs`` names exactly two distinct known names (an output or
    a group) standing for disjoint subjects — the pair form the
    distance kinds share."""
    outs = a.get("outputs")
    if (
        not isinstance(outs, list) or len(outs) != 2
        or not all(isinstance(o, str) for o in outs)
        or len(set(outs)) != 2
    ):
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}' 'outputs' must list 2 distinct output "
            f"names ({kind} measures between exactly two; "
            f'"others": true keeps one output clear of every other '
            f"- clearance only; a piece of several outputs is one "
            f"name under root 'groups')",
        )
    for o in outs:
        known_name(aid, o, raw_outputs, groups)
    disjoint_subjects(aid, kind, outs, groups)


def check_names(aid: str, a: dict, key: str, source: str, names: list, what: str) -> None:
    """A quantifier key that names whose copies it ranges over (DESIGN
    5.2): absent, or a non-empty list of distinct names drawn from
    ``names``, the outputs the entry states under ``source``."""
    val = a.get(key)
    if val is None:
        return
    if (
        not isinstance(val, list) or not val
        or not all(isinstance(n, str) for n in val)
        or len(set(val)) != len(val)
        or not set(val) <= set(names)
    ):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' '{key}' must be a non-empty list of "
            f"distinct names drawn from its '{source}' "
            f"{names} - {what}",
        )


def check_flag(aid: str, a: dict, key: str, omit: str) -> None:
    """A boolean key is JSON ``true`` or absent. One spelling per
    meaning: absence is the off state, a written false would be a
    second spelling of it; ``omit`` says what omitting the key means."""
    val = a.get(key)
    if val is not None and val is not True:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' '{key}' must be JSON true when present "
            f"(omit the key {omit})",
        )


def check_tol(aid: str, a: dict, unit: str) -> None:
    """``tol``, when present, is a finite number >= 0 in the kind's
    ``unit`` (the key table already bound the key to the kinds where
    it is a length, so the unit is always meters)."""
    tol = a.get("tol")
    if tol is not None and (not is_number(tol) or tol < 0):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'tol' must be a finite number >= 0 ({unit})",
        )
