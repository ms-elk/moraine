"""Screenshot camera resolution (DESIGN 7.2): named views are presets
over the free camera, poses are renderer-neutral, and the auto-fit
reproduces the viewer's historical framing (fov 45, bounding-sphere fit
with 1.35 slack)."""

import math

import pytest

from moraine3d.app import camera
from moraine3d.errors import MoraineError

# Unit cube centered at the origin: radius = half diagonal = sqrt(3)/2.
_UNIT = ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
_UNIT_DIST = (math.sqrt(3.0) / 2.0) / math.tan(math.radians(22.5)) * 1.35


def _close(a, b, tol=1e-6):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


def test_named_views_positions_and_ups():
    """The seven named views reproduce the historical VIEW_DIRS poses:
    axis directions per DESIGN 4.3 (+Z forward -> front looks from +Z),
    up = +Y on the horizon and the top/bottom special cases at el=+-90."""
    d = _UNIT_DIST
    expected = {
        "front": ([0, 0, d], [0, 1, 0]),
        "back": ([0, 0, -d], [0, 1, 0]),
        "right": ([d, 0, 0], [0, 1, 0]),
        "left": ([-d, 0, 0], [0, 1, 0]),
        "top": ([0, d, 0], [0, 0, -1]),
        "bottom": ([0, -d, 0], [0, 0, 1]),
    }
    for view, (pos, up) in expected.items():
        cam = camera.resolve(view=view, bounds=_UNIT)
        assert _close(cam["position"], pos), view
        assert _close(cam["up"], up), view
        assert cam["fov"] == 45.0
        assert math.isclose(cam["distance"], d, abs_tol=1e-6)


def test_iso_matches_historical_direction():
    """iso is exactly the historical (1, 0.8, 1) viewer direction."""
    cam = camera.resolve(view="iso", bounds=_UNIT)
    n = math.sqrt(1 + 0.8**2 + 1)
    want = [_UNIT_DIST * c / n for c in (1.0, 0.8, 1.0)]
    assert _close(cam["position"], want)
    assert math.isclose(cam["azimuth"], 45.0)


def test_free_camera_up_is_unit_and_orthogonal():
    """At arbitrary angles the derivative up-vector stays unit length and
    orthogonal to the view direction — one formula, no singularities."""
    for az, el in ((0, 0), (30, 45), (-120, 80), (200, -35), (45, -90)):
        cam = camera.resolve(azimuth=az, elevation=el, bounds=_UNIT)
        d = [p - t for p, t in zip(cam["position"], cam["target"])]
        up = cam["up"]
        assert math.isclose(math.hypot(*up), 1.0, abs_tol=1e-6), (az, el)
        dot = sum(a * b for a, b in zip(d, up))
        assert abs(dot) < 1e-5, (az, el)


def test_explicit_distance_and_target_respected():
    cam = camera.resolve(
        view="front", distance=2.0, target=[1.0, 0.5, -1.0], bounds=_UNIT
    )
    assert _close(cam["position"], [1.0, 0.5, 1.0])
    assert cam["target"] == [1.0, 0.5, -1.0]
    assert cam["distance"] == 2.0


def test_degenerate_bounds_keep_min_working_distance():
    """A flat/degenerate AABB never fits tighter than the 0.1 m radius
    floor — the historical frameCamera clamp."""
    flat = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cam = camera.resolve(view="front", bounds=flat)
    want = 0.1 / math.tan(math.radians(22.5)) * 1.35
    assert math.isclose(cam["distance"], want, abs_tol=1e-6)


def test_lone_azimuth_or_elevation_defaults_other_to_zero():
    cam = camera.resolve(azimuth=90.0, bounds=_UNIT)
    assert cam["elevation"] == 0.0
    assert _close(cam["position"], [_UNIT_DIST, 0, 0])


def test_validation_errors():
    """bad_view / bad_camera / camera_conflict, with actionable messages
    (DESIGN 2 P2)."""
    with pytest.raises(MoraineError) as e:
        camera.resolve(view="sideways", bounds=_UNIT)
    assert e.value.code == "bad_view"
    assert "front" in e.value.message

    with pytest.raises(MoraineError) as e:
        camera.resolve(view="front", azimuth=30.0, bounds=_UNIT)
    assert e.value.code == "camera_conflict"

    for kwargs in (
        {"elevation": 91.0},
        {"elevation": -90.5},
        {"azimuth": "north"},
        {"distance": 0.0},
        {"distance": -1.0},
        {"distance": True},
        {"target": [1.0, 2.0]},
        {"target": "0,0,0"},
        {"target": [1.0, 2.0, "x"]},
        # Non-finite numbers: inf reaches math.sin as a ValueError, NaN
        # would leave as a pose no JSON output can carry.
        {"azimuth": math.inf},
        {"azimuth": math.nan},
        {"elevation": math.nan},
        {"distance": math.inf},
        {"target": [0.0, math.nan, 0.0]},
    ):
        with pytest.raises(MoraineError) as e:
            camera.resolve(bounds=_UNIT, **kwargs)
        assert e.value.code == "bad_camera", kwargs


def test_asset_bounds_unions_meshes_and_skips_images():
    """Image outputs carry no bounds and must not poison the union; no
    mesh output falls back to the unit cube (the viewer's historical
    empty-asset framing)."""
    stats = {
        "a": {"bounds": {"min": [-1.0, 0.0, 0.0], "max": [0.0, 1.0, 1.0]}},
        "b": {"bounds": {"min": [0.0, 0.0, -2.0], "max": [3.0, 0.5, 0.0]}},
        "tex": {"kind": "image", "size": [64, 64]},
    }
    lo, hi = camera.asset_bounds(stats)
    assert lo == [-1.0, 0.0, -2.0] and hi == [3.0, 1.0, 1.0]
    # Null bound components (non-finite geometry, JSON-safe stats) carry
    # no bounds either.
    stats["c"] = {"bounds": {"min": [None, 0.0, 0.0], "max": [None, 9.0, 9.0]}}
    assert camera.asset_bounds(stats) == (lo, hi)
    assert camera.asset_bounds({"tex": {"kind": "image"}}) == (
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
    )
