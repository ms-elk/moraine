"""Golden-mesh tests for the ``revolve`` operator (DESIGN 5, 10).

Counts, bounds, and volume within tolerance pin the semantics contract:
solid of revolution of an inline [radius, y] profile around +Y,
auto-closed ends (always watertight), crease-threshold profile shading.
Volumes are computed via trimesh signed volume, so they also verify
CCW-outward winding.
"""

import math

import numpy as np
import pytest
import trimesh

from moraine3d.errors import OpError
from moraine3d.ops import cylinder, revolve


def _tm(m) -> trimesh.Trimesh:
    t = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
    t.merge_vertices()
    return t


def volume(m) -> float:
    return _tm(m).volume


def watertight(m) -> bool:
    return _tm(m).is_watertight


# Every cross-section of the tessellated solid is the inscribed S-gon,
# so the exact volume is the polygon-area factor times the profile's
# integral of r^2 dy (each linear band is a frustum of the inscribed
# pyramid - quads are planar because r is linear in y).
def _polygon_factor(seg: int) -> float:
    return 0.5 * seg * math.sin(2.0 * math.pi / seg)


def test_revolve_flange_golden():
    """Semantics: pole ends close with fans, 90-degree steps are hard
    joints (duplicated rings) at the default angle_max_deg, and the
    solid is watertight with the exact inscribed-polygon volume."""
    seg = 32
    prof = [[0, 0], [0.5, 0], [0.5, 0.2], [0.2, 0.2], [0.2, 0.6], [0, 0.6]]
    m = revolve.evaluate([], {"profile": prof, "segments": seg}, None)
    # P=6 points, E0=2 poles, H=4 hard joints: rings = P-2+H = 8.
    assert len(m.vertices) == 8 * seg + 2
    assert len(m.faces) == 8 * seg
    s = m.stats()
    assert s["bounds"]["min"] == [-0.5, 0.0, -0.5]
    assert s["bounds"]["max"] == [0.5, 0.6, 0.5]
    expected = _polygon_factor(seg) * (0.5**2 * 0.2 + 0.2**2 * 0.4)
    assert volume(m) == pytest.approx(expected, rel=1e-5)
    assert watertight(m)


def test_revolve_matches_cylinder():
    """Semantics: a two-point vertical profile with capped ends is
    exactly the cylinder contract - same counts, bounds, and inscribed
    prism volume as the cylinder golden."""
    seg = 32
    m = revolve.evaluate([], {"profile": [[0.5, -1.0], [0.5, 1.0]], "segments": seg}, None)
    c = cylinder.evaluate([], {"radius": 0.5, "height": 2.0, "segments": seg}, None)
    assert len(m.vertices) == len(c.vertices) == 4 * seg + 2
    assert len(m.faces) == len(c.faces) == 4 * seg
    assert m.stats()["bounds"] == c.stats()["bounds"]
    expected = _polygon_factor(seg) * 0.5**2 * 2.0
    assert volume(m) == pytest.approx(expected, rel=1e-5)
    assert watertight(m)


def test_revolve_dome_golden():
    """Semantics: gentle joints (11.25-degree bends, under the default
    threshold) share rings with the half-plane bisector normal - on a
    circular arc that bisector is exactly radial, so normals equal the
    normalized position (the sphere contract) and the pole closes at
    (0, 1, 0)."""
    seg = 32
    prof = [
        [0.5 * math.cos(math.radians(i * 11.25)), 0.5 * math.sin(math.radians(i * 11.25))]
        for i in range(8)
    ] + [[0.0, 0.5]]
    m = revolve.evaluate([], {"profile": prof, "segments": seg}, None)
    # P=9, E0=1 pole, E1=1 cap, H=0: rings = P-2+1 = 8, plus cap ring.
    assert len(m.vertices) == 8 * seg + 1 + (seg + 1)
    assert len(m.faces) == 2 * seg * 8 - seg + seg
    # Exact: the frustum bands between consecutive profile points (the
    # last band is the pole cone; the flat cap at y=0 adds nothing).
    bands = sum(
        abs(y1 - y0) / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)
        for (r0, y0), (r1, y1) in zip(prof, prof[1:])
    )
    assert volume(m) == pytest.approx(_polygon_factor(seg) * bands, rel=1e-5)
    assert watertight(m)
    # Interior shared rings sit at indices [seg, 8*seg): bisector == radial.
    pos = m.vertices[seg : 8 * seg]
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert np.allclose(pos, m.normals[seg : 8 * seg], atol=1e-6)
    assert np.allclose(m.normals[8 * seg], [0.0, 1.0, 0.0])


def test_revolve_crease_threshold():
    """Semantics: a joint bending more than angle_max_deg duplicates its
    ring; one bending less shares it - the counts differ by exactly
    ``segments``."""
    seg = 16
    prof = [[0.3, 0.0], [0.3, 0.5], [0.5, 1.0]]  # one joint bending ~21.8 deg
    soft = revolve.evaluate(
        [], {"profile": prof, "segments": seg, "angle_max_deg": 25.0}, None
    )
    hard = revolve.evaluate(
        [], {"profile": prof, "segments": seg, "angle_max_deg": 20.0}, None
    )
    assert len(hard.vertices) == len(soft.vertices) + seg
    assert len(hard.faces) == len(soft.faces)
    assert watertight(soft) and watertight(hard)


@pytest.mark.parametrize(
    "params,fragment",
    [
        ({}, "requires 'profile'"),
        ({"profile": "x"}, "list of 2"),
        ({"profile": [[0.5, 0.0]]}, "list of 2"),
        ({"profile": [[0.5, 0.0, 1.0], [0.5, 1.0]]}, "point 0"),
        ({"profile": [[0.5, float("nan")], [0.5, 1.0]]}, "point 0"),
        ({"profile": [[-0.1, 0.0], [0.5, 1.0]]}, "radius < 0"),
        ({"profile": [[0.5, 0.0], [0.0, 0.5], [0.5, 1.0]]}, "only at the ends"),
        ({"profile": [[0.5, 0.0], [0.5, 0.0], [0.5, 1.0]]}, "coincide"),
        ({"profile": [[0.0, 0.0], [0.5, 0.5], [0.0, 0.0]]}, "first and last"),
        ({"profile": [[0.5, 0.0], [0.5, 1.0], [0.5, 0.5]]}, "doubles back"),
        ({"profile": [[0.0, 0.0], [0.0, 1.0]]}, "encloses no area"),
        # Side wall pierces the end cap disc (closure edge).
        ({"profile": [[0.5, 0.0], [0.1, 0.6], [0.5, 0.3]]}, "self-intersects"),
        # Two profile segments cross each other.
        ({"profile": [[0.4, 0.0], [0.4, 0.6], [0.1, 0.1], [0.6, 0.5]]}, "self-intersects"),
        ({"profile": [[0.0, 0.6], [0.2, 0.6], [0.2, 0.0], [0.0, 0.0]]}, "reverse"),
        ({"profile": [[0.5, 0.0], [0.5, 1.0]], "segments": 2}, ">= 3"),
        ({"profile": [[0.5, 0.0], [0.5, 1.0]], "segments": 4097}, "<= 4096"),
        ({"profile": [[0.5, 0.0], [0.5, 1.0]], "angle_max_deg": 0}, "> 0"),
        ({"profile": [[0.5, 0.0], [0.5, 1.0]], "angle_max_deg": 181}, "<= 180"),
    ],
)
def test_revolve_refusals(params, fragment):
    """Semantics: degenerate or inward-facing profiles are structured
    refusals naming the fix (DESIGN 8.2), never silent geometry."""
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], params, None)
    assert exc.value.code == "bad_param"
    assert fragment in exc.value.message


def test_revolve_cell_budget_refused_before_allocation():
    """Semantics: a rings x segments product beyond the shared
    4096 x 4096 cell budget is refused (hard joints duplicate rings, so
    the per-param ceilings alone do not bound the product) - a
    structured refusal, never a raw allocation failure (DESIGN 5)."""
    # 4096-point zigzag, every joint a hard ~178-degree bend, strictly
    # rising y (simple polygon): rings = 2*(P-2)+2 = 8190.
    prof = []
    for k in range(2048):
        prof.append([0.5, k * 0.001])
        prof.append([0.6, k * 0.001 + 0.0005])
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": prof, "segments": 4096}, None)
    assert exc.value.code == "bad_param"
    assert "cell budget" in exc.value.message


def test_revolve_profile_count_ceiling():
    """Semantics: the profile point count is capped like every
    tessellation input - beyond it the count is a typo, refused rather
    than surfacing as a raw allocation failure."""
    prof = [[0.5, i * 1e-3] for i in range(4097)]
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": prof}, None)
    assert "4096" in exc.value.message


def test_revolve_reads_a_rounding_sized_negative_radius_as_the_axis():
    """Semantics: a radius within 1e-9 below zero, as an interpolation
    leaves at the axis, is read as 0, so the solid is bit for bit the
    one the exact 0 builds; a radius further below zero is still
    refused, and a small positive one is still a cap, not a pole."""
    exact = [[0.5, 0.0], [0.3, 0.4], [0.0, 0.5]]
    onto_axis = [[0.5, 0.0], [0.3, 0.4], [-1.4e-17, 0.5]]
    want = revolve.evaluate([], {"profile": exact, "segments": 16}, None)
    got = revolve.evaluate([], {"profile": onto_axis, "segments": 16}, None)
    assert np.array_equal(got.vertices, want.vertices)
    assert np.array_equal(got.faces, want.faces)
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": [[0.5, 0.0], [0.3, 0.4], [-1e-6, 0.5]]}, None)
    assert "radius < 0" in str(exc.value)
    capped = revolve.evaluate(
        [], {"profile": [[0.5, 0.0], [0.3, 0.4], [1e-12, 0.5]], "segments": 16}, None,
    )
    assert len(capped.vertices) > len(want.vertices)
