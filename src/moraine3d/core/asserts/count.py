"""``count`` (DESIGN 5.2): how many of one subject there are — an
output's copies (an instanced output's transforms, a plain mesh
counting one) or the patches of one tagged face set — against
``value`` or a ``min``/``max`` band.
"""

from __future__ import annotations

from ...errors import AssetError
from .. import numbers
from . import common

KEY_KINDS = {
    "outputs": numbers.OTHER,
    "faces": numbers.OTHER,
    "value": numbers.INT,
    "min": numbers.INT,
    "max": numbers.INT,
}
KEYS = frozenset(KEY_KINDS)
KEY_MEANING = {
    "outputs": "one output, its copies counted",
    "faces": "one tagged face set, its patches counted",
    "value": "the stated count",
    "min": "copies or patches",
    "max": "copies or patches",
}
SUBJECT = "counts one subject, an output's copies or a face set's patches"
# A count is exact or banded; a refused `tol` says so.
REFUSALS = {
    "tol": (
        "count is exact ('value') or a 'min'/'max' band; there is no "
        "tolerance on a count"
    ),
}


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a count entry (DESIGN 5.2): one subject —
    ``outputs`` naming one output (its copy count; a plain mesh counts
    one; a group is refused — a sum of copy counts is no output's
    count) or ``faces`` naming one tagged face set ``[output, tag]``
    (its patches; no copy index, since tags ride the prototype and the
    set is counted once) — and ``value`` or a ``min``/``max`` band, all
    non-negative integers."""
    has_outputs = "outputs" in a
    has_faces = "faces" in a
    if has_outputs == has_faces:
        raise AssetError(
            "bad_assert_outputs",
            f"assert '{aid}' (count) takes one subject: \"outputs\": [name] "
            f"(the output's copy count) or \"faces\": [output, tag] (the "
            f"patches of one tagged face set) - not both, not neither",
        )
    if has_outputs:
        common.check_one_output(
            aid, a, raw_outputs, "count", "one output's copy count", groups,
            "counts one output's copies",
        )
        unit = "copies"
    else:
        faces = a["faces"]
        if isinstance(faces, list) and len(faces) == 3:
            raise AssetError(
                "bad_assert_faces",
                f"assert '{aid}': a count over a face set takes no copy "
                f"index - tags ride the prototype, so the set is counted "
                f"once and is the same on every copy; write \"faces\": "
                f"[output, tag]",
            )
        common.check_face_spec(
            aid, faces, raw_outputs,
            f"assert '{aid}' 'faces' must be [output, tag] (a declared "
            f"output name and a face-group name) - one face set, its "
            f"patches counted",
            groups, "count",
        )
        unit = "patches"

    def count_number(val) -> bool:
        return isinstance(val, int) and not isinstance(val, bool) and val >= 0

    common.check_judgment(
        aid, a, "count", f"a non-negative integer ({unit})", count_number,
        "copy count" if has_outputs else "patch count",
    )
