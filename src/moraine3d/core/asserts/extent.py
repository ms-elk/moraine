"""``extent`` (DESIGN 5.2): one ``measure`` of a subject's world-space
vertex bounds — an output with its copies united, a group, or one
tagged face set — meets a stated ``value`` within ``tol`` or lies in a
``min``/``max`` band; ``every`` reads each copy on its own.
"""

from __future__ import annotations

from .. import numbers
from ...errors import AssetError
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "faces": numbers.INT,
    "measure": numbers.OTHER,
    "every": numbers.OTHER,
    "value": numbers.FLOAT,
    "tol": numbers.FLOAT,
    "min": numbers.FLOAT,
    "max": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# extent `value` form default, meters: |measured - value| <= tol. A
# design allowance owned by the author; its default is
# the repository's own separation quantum (the guide's >= 1 mm rule) — the
# scale at which a stated dimension is met, not float noise (a 2.1 m rise
# built from twelve 0.175 m steps lands within 1e-12). A stated
# manufacturing or design tolerance is always an explicit `tol`
# (DESIGN 5.2).
DEFAULT_TOL = 1e-3
TOL_UNIT = "dimension allowance around 'value', m"
KEY_MEANING = {
    "outputs": "one output or group, its copies united",
    "faces": "one tagged face set",
    "measure": "which bound of the subject to read: size_/min_/max_/center_ x|y|z",
    "every": "each copy measured on its own",
    "value": "the stated dimension, m",
    "tol": TOL_UNIT,
    "min": "m",
    "max": "m",
}
SUBJECT = "measures one subject, copies united"
# extent's `measure` vocabulary: one token, the predicate fields' spelling
# read along an axis (`center_y` is a face-predicate field already) —
# one key, and no band key called `min` beside a measure called `min`.
MEASURES = tuple(
    f"{what}_{axis}" for what in ("size", "min", "max", "center") for axis in "xyz"
)


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of an extent entry (DESIGN 5.2): one subject —
    ``outputs`` naming one output (its world-space vertex bounds, copies
    united) or ``faces`` naming one tagged face set ``[output, tag]`` /
    ``[output, tag, instance]`` — one ``measure`` token from
    MEASURES, and one judgment: ``value`` (+ ``tol``, m) or a
    ``min``/``max`` band (m; a band edge may be negative — the low edge of
    a part below ground is a legitimate statement). ``every`` is JSON
    true or absent, and names every copy — so the face form drops its
    copy index under it."""
    has_outputs = "outputs" in a
    has_faces = "faces" in a
    if has_outputs == has_faces:
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}' (extent) takes one subject: \"outputs\": [name] "
            f"(the output's bounds, copies united) or \"faces\": [output, "
            f"tag] / [output, tag, copy] (one tagged face set) - not both, "
            f"not neither",
        )
    if has_outputs:
        common.check_one_output(aid, a, raw_outputs, "extent", "one subject's bounds", groups)
    else:
        common.check_face_spec(
            aid, a["faces"], raw_outputs,
            f"assert '{aid}' 'faces' must be [output, tag] or [output, tag, "
            f"instance] (a declared output name, a face-group name, and an "
            f"optional 0-based copy index for an instanced output) - one "
            f"face set; mated is the kind that relates two",
            groups, "extent",
        )
    measure = a.get("measure")
    if measure not in MEASURES:
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' needs 'measure', one of {list(MEASURES)} "
            f"- the bound of the subject to read, in meters along that axis",
        )
    common.check_judgment(
        aid, a, "extent", "a finite length in meters", common.is_number, "dimension"
    )
    common.check_flag(
        aid, a, "every",
        "to measure the copies united (outputs) or one indexed copy (faces)",
    )
    if a.get("every") and has_faces and len(a["faces"]) == 3:
        raise AssetError(
            "bad_assert_faces",
            f"assert '{aid}': \"every\" evaluates the face set on each copy; "
            f"drop the copy index, or name one copy without \"every\"",
        )
    common.check_tol(aid, a, TOL_UNIT)
