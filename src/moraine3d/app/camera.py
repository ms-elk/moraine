"""Screenshot camera resolution (DESIGN 7.2).

The dedicated render session's camera is resolved server-side into a
renderer-neutral pose — cartesian position/target/up plus vertical fov in
degrees — so every client (three.js viewer, a native viewer, a path
tracer) consumes the same numbers and the math is testable without a
renderer.

Axis convention (DESIGN 4.3): right-handed, +Y up, +Z forward — an
asset's front faces +Z. Azimuth 0 looks from +Z (front), 90 from +X
(right), measured toward +X; elevation 0 is horizontal, +90 straight
down from the top.

    dir = (sin az * cos el,  sin el,  cos az * cos el)    # target -> camera
    up  = (-sin az * sin el, cos el, -cos az * sin el)    # d(dir)/d(el)
    position = target + distance * dir

`up` is the elevation derivative of `dir`: always unit length, always
orthogonal to the view direction, equal to +Y on the horizon and to the
viewer's historical top/bottom special cases at el = +-90 — one formula,
no singularities inside the accepted range.
"""

from __future__ import annotations

import math

from ..core import mesh as mesh_mod
from ..errors import MoraineError
from ..ops import common

FOV_DEG = 45.0

# Auto-fit contract (kept identical to the viewer's historical
# frameCamera): frame the bounds' bounding sphere with 1.35 slack, and
# never fit tighter than a 0.1 m radius so degenerate/flat bounds keep a
# usable working distance.
_FIT_SLACK = 1.35
_MIN_RADIUS = 0.1

# Named views are presets over the free camera — the single source of
# the view roster (CLI choices and server validation both read it).
# iso's elevation is exactly the historical (1, 0.8, 1) direction.
VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
    "iso": (45.0, math.degrees(math.atan(0.8 / math.sqrt(2.0)))),
}


# Finite only (is_num): inf reaches math.sin as a ValueError, NaN as a
# NaN pose that no JSON output can carry; stat_num is the stats rounding
# convention — both shared spellings, so a hardening lands everywhere.
_is_num = common.is_num
_r6 = mesh_mod.stat_num


def asset_bounds(stats: dict) -> tuple[list[float], list[float]]:
    """Union AABB over the outputs that have one (image outputs carry no
    bounds); no mesh output falls back to the unit cube at the origin —
    the viewer's historical empty-asset framing."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for s in stats.values():
        b = s.get("bounds")
        if not b or any(v is None for v in (*b["min"], *b["max"])):
            continue  # image output, or non-finite geometry (null bounds)
        lo = [min(a, v) for a, v in zip(lo, b["min"])]
        hi = [max(a, v) for a, v in zip(hi, b["max"])]
    if math.isinf(lo[0]):
        return [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]
    return lo, hi


def resolve(
    view=None,
    azimuth=None,
    elevation=None,
    distance=None,
    target=None,
    *,
    bounds: tuple[list[float], list[float]],
    fov: float = FOV_DEG,
) -> dict:
    """Resolve a named view or free-camera parameters into a camera pose.

    Returns ``{position, target, up, fov, azimuth, elevation, distance}``
    — the cartesian half feeds the renderer, the spherical half feeds the
    screenshot report. Raises MoraineError (``bad_view``, ``bad_camera``,
    ``camera_conflict``) on invalid input; the server is the authority on
    this contract, CLI-side checks are fast-path UX only.
    """
    if view is not None and (azimuth is not None or elevation is not None):
        raise MoraineError(
            "camera_conflict",
            "view and azimuth/elevation are exclusive; drop the named view "
            "to place the camera freely",
        )
    if view is not None:
        if not isinstance(view, str) or view not in VIEWS:
            raise MoraineError(
                "bad_view", f"view must be one of {list(VIEWS)}"
            )
        az, el = VIEWS[view]
    else:
        for name, v in (("azimuth", azimuth), ("elevation", elevation)):
            if v is not None and not _is_num(v):
                raise MoraineError("bad_camera", f"{name} must be a finite number (degrees)")
        az = float(azimuth) if azimuth is not None else 0.0
        el = float(elevation) if elevation is not None else 0.0
    if not -90.0 <= el <= 90.0:
        raise MoraineError("bad_camera", "elevation must be in -90..90 degrees")

    if target is None:
        lo, hi = bounds
        tgt = [(a + b) / 2.0 for a, b in zip(lo, hi)]
    else:
        if not (
            isinstance(target, (list, tuple))
            and len(target) == 3
            and all(_is_num(v) for v in target)
        ):
            raise MoraineError(
                "bad_camera", "target must look like 0,0.5,0 (three finite numbers)"
            )
        tgt = [float(v) for v in target]

    if distance is None:
        lo, hi = bounds
        radius = math.dist(lo, hi) / 2.0
        dist = max(radius, _MIN_RADIUS) / math.tan(math.radians(fov / 2.0)) * _FIT_SLACK
    else:
        if not _is_num(distance) or not distance > 0:
            raise MoraineError("bad_camera", "distance must be a finite number > 0 (meters)")
        dist = float(distance)

    az_r, el_r = math.radians(az), math.radians(el)
    d = (
        math.sin(az_r) * math.cos(el_r),
        math.sin(el_r),
        math.cos(az_r) * math.cos(el_r),
    )
    up = (
        -math.sin(az_r) * math.sin(el_r),
        math.cos(el_r),
        -math.cos(az_r) * math.sin(el_r),
    )
    return {
        "position": [_r6(t + dist * c) for t, c in zip(tgt, d)],
        "target": [_r6(v) for v in tgt],
        "up": [_r6(v) for v in up],
        "fov": fov,
        "azimuth": _r6(az),
        "elevation": _r6(el),
        "distance": _r6(dist),
    }
