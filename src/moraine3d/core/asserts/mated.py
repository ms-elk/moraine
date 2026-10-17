"""``mated`` (DESIGN 5.2): two tagged faces, read from evaluated
world-space geometry, register in the entry's mode — ``coincident``
(anti-parallel normals, mating origins within ``tol``) or
``contained`` (coplanar, faces[1]'s contour inside faces[0]'s).
"""

from __future__ import annotations

from .. import numbers
from ...errors import AssetError
from . import common

KEY_KINDS = {
    "faces": numbers.INT,
    "every": numbers.OTHER,
    "mode": numbers.OTHER,
    "tol": numbers.FLOAT,
    "angle_tol_deg": numbers.FLOAT,
}
KEYS = frozenset(KEY_KINDS)
# mated (DESIGN 5.2): two tagged faces — `faces: [[output, tag], [output,
# tag, instance?]]` — must satisfy the mode. `coincident` (default):
# anti-parallel normals within `angle_tol_deg`, mating-frame origins
# within `tol` (meters). `contained`: the same normal stage, then the
# faces coplanar within `tol` along the mating normal and faces[1]'s
# contour inside faces[0]'s outer contour with rim slack `tol`. Both
# defaults are tool-owned numerics floors, not design quantities: the
# origin floor is contact's float32 bar; 0.01 deg of tilt is 0.17 mm
# across a 1 m face — below the modeling-error band, four orders above
# the normal noise of float32 vertices.
DEFAULT_TOL = 1e-5
DEFAULT_ANGLE_TOL_DEG = 0.01
TOL_UNIT = "mating-origin distance tolerance, m"
KEY_MEANING = {
    "faces": "two tagged faces",
    "every": "every copy of the named instanced outputs, paired by copy index",
    "mode": "coincident or contained registration",
    "tol": TOL_UNIT,
    "angle_tol_deg": "normal anti-parallelism tolerance, degrees",
}
SUBJECT = "relates two tagged faces"


def validate_entry(aid: str, a: dict, raw_outputs, groups: dict) -> None:
    """Parse-time shape of a mated entry: ``faces`` = exactly two
    ``[output, tag]`` / ``[output, tag, instance]`` lists over declared
    outputs; ``every`` a list of names drawn from them, whose faces then
    carry no copy index; ``mode`` coincident or contained;
    ``angle_tol_deg`` finite >= 0 (the key table ``KEYS`` has already
    rejected foreign keys). Tag presence and instance range are
    evaluation facts and are judged at validate time, as
    unevaluatable-with-reason entries."""
    faces = a.get("faces")
    shape = (
        f"assert '{aid}' 'faces' must be 2 entries, each [output, tag] or "
        f"[output, tag, instance] (a declared output name, a face-group name, "
        f"and an optional 0-based copy index for an instanced output)"
    )
    if not isinstance(faces, list) or len(faces) != 2:
        raise AssetError("bad_assert_faces", shape)
    for f in faces:
        common.check_face_spec(aid, f, raw_outputs, shape, groups, "mated")
    if faces[0] == faces[1]:
        raise AssetError(
            "bad_assert_faces",
            f"assert '{aid}' names the same face twice; mated relates two "
            f"different tagged faces",
        )
    common.check_names(
        aid, a, "every", "faces", list(dict.fromkeys(f[0] for f in faces)),
        "the outputs whose every copy is mated",
    )
    for f in faces:
        if f[0] in (a.get("every") or ()) and len(f) == 3:
            raise AssetError(
                "bad_assert_faces",
                f"assert '{aid}': \"every\" names '{f[0]}', so its face is "
                f"judged on each copy; drop the copy index, or take '{f[0]}' "
                f"out of \"every\" to name one copy",
            )
    mode = a.get("mode")
    if mode is not None and mode not in ("coincident", "contained"):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'mode' must be \"coincident\" (the faces' "
            f'mating origins register) or "contained" (faces[0] contains '
            f"faces[1])",
        )
    ang = a.get("angle_tol_deg")
    if ang is not None and (not common.is_number(ang) or ang < 0):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'angle_tol_deg' must be a finite number >= 0 "
            f"(normal anti-parallelism tolerance, degrees)",
        )
    common.check_tol(aid, a, TOL_UNIT)
