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
def _polygon_factor(seg: int, arc_deg: float = 360.0) -> float:
    return 0.5 * seg * math.sin(math.radians(arc_deg) / seg)


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


@pytest.mark.parametrize(
    "params,faces,frozen",
    [
        # Capped ends, hard and smooth joints (the turned_column example).
        (
            {
                "profile": [
                    [0.3, 0.0], [0.3, 0.05], [0.24, 0.09], [0.2, 0.13], [0.16, 0.18],
                    [0.13, 2.2], [0.18, 2.26], [0.24, 2.3], [0.24, 2.36],
                ],
                "segments": 48,
            },
            864,
            "69d96d32756430a353c39a9ba948a10805e1f8492a2138ab7fdc6213d00da1a6",
        ),
        # Pole ends.
        (
            {
                "profile": [[0, 0], [0.5, 0], [0.5, 0.2], [0.2, 0.2], [0.2, 0.6], [0, 0.6]],
                "segments": 32,
            },
            256,
            "cbb3f17ea4741836eedaec83e293ad7fa55ba5d0670b70b40bda62bb62eb9542",
        ),
        # A profile too thin to triangulate, which a full turn never does.
        (
            {"profile": [[2, 0], [2, 1e-14], [1, 5e-15], [0, 1e-14]]},
            192,
            "ab4f2755cf6561a6e028b288a0e1f34cec4394ef69644628b88f925faee25291",
        ),
    ],
)
def test_revolve_full_turn_result_hash_frozen(params, faces, frozen):
    """DESIGN 5: extending an operator changes nothing an asset already
    builds, so the full `revolve` of an open profile keeps its result,
    op_impl_version stays 1 (DESIGN 4.5) and these hashes pin that
    identity. The ring angles pass through sin and cos in float64
    before the float32 cast, so a platform whose last bit differs
    reaches the hash only at a rounding boundary."""
    assert revolve.OP_IMPL_VERSION == 1
    mesh = revolve.evaluate([], params, None)
    assert len(mesh.faces) == faces
    assert mesh.result_hash() == frozen
    turned = revolve.evaluate([], {**params, "arc_deg": 360.0}, None)
    assert turned.result_hash() == frozen


FLANGE = [[0, 0], [0.5, 0], [0.5, 0.2], [0.2, 0.2], [0.2, 0.6], [0, 0.6]]
FLANGE_BANDS = 0.5**2 * 0.2 + 0.2**2 * 0.4  # the profile's integral of r^2 dy
DRUM = [[0.5, -1.0], [0.5, 1.0]]


def test_revolve_part_turn_golden():
    """Semantics: below 360 both cut faces are closed with flat caps -
    a quarter drum in 8 steps is 9 columns of its 2 rings, two end fans
    of 9 points about a center, and the 4-point generating polygon on
    each cut face, watertight."""
    seg = 8
    m = revolve.evaluate([], {"profile": DRUM, "segments": seg, "arc_deg": 90.0}, None)
    assert len(m.vertices) == 2 * (seg + 1) + 2 * (seg + 2) + 2 * 4
    assert len(m.faces) == 2 * seg + 2 * seg + 2 * 2
    s = m.stats()
    assert s["bounds"]["min"] == [0.0, -1.0, -0.5]
    assert s["bounds"]["max"] == [0.5, 1.0, 0.0]
    assert volume(m) == pytest.approx(_polygon_factor(seg, 90.0) * 0.5**2 * 2.0, rel=1e-5)
    assert watertight(m)


@pytest.mark.parametrize("arc_deg", [30.0, 90.0, 200.0, 270.0])
@pytest.mark.parametrize(
    "prof,bands", [(FLANGE, FLANGE_BANDS), (DRUM, 0.5**2 * 2.0)], ids=["poles", "caps"]
)
def test_revolve_part_turn_volume(arc_deg, prof, bands):
    """Semantics: `segments` counts the steps across the arc, so every
    cross-section is the sector of an inscribed polygon of that many
    sides - the exact volume, whether the ends close at poles or with
    flat caps."""
    seg = 7
    m = revolve.evaluate([], {"profile": prof, "segments": seg, "arc_deg": arc_deg}, None)
    assert volume(m) == pytest.approx(_polygon_factor(seg, arc_deg) * bands, rel=1e-5)
    assert watertight(m)


def test_revolve_half_turn_is_half_the_full_turn_at_the_same_step():
    """Semantics: a half turn of 16 has the facets of a full turn of
    32 - half its volume exactly, half its bounds across the cut, and
    a volume no longer proportional to the arc at equal `segments`."""
    full = revolve.evaluate([], {"profile": FLANGE, "segments": 32}, None)
    half = revolve.evaluate([], {"profile": FLANGE, "segments": 16, "arc_deg": 180.0}, None)
    assert volume(half) == pytest.approx(volume(full) / 2.0, rel=1e-6)
    assert half.stats()["bounds"]["min"] == [-0.5, 0.0, -0.5]
    assert half.stats()["bounds"]["max"] == [0.5, 0.6, 0.0]
    coarse = revolve.evaluate([], {"profile": FLANGE, "segments": 4, "arc_deg": 180.0}, None)
    square = revolve.evaluate([], {"profile": FLANGE, "segments": 4}, None)
    assert volume(coarse) / volume(square) == pytest.approx(math.sqrt(0.5), rel=1e-5)


def test_revolve_turn_starts_at_x_and_runs_toward_minus_z():
    """Semantics: the turn starts at angle 0 (+X) and runs toward -Z,
    and each cut face looks back out of the solid - +Z at the start,
    and -X where a quarter turn ends."""
    m = revolve.evaluate([], {"profile": DRUM, "segments": 8, "arc_deg": 90.0}, None)
    start, end = m.normals[-8:-4], m.normals[-4:]
    assert np.allclose(start, [0.0, 0.0, 1.0])
    assert np.allclose(end, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(m.vertices[-8:-4, 2], 0.0)
    assert np.allclose(m.vertices[-4:, 0], 0.0, atol=1e-6)


def test_revolve_thin_profile_builds_full_and_never_caps_wrongly():
    """Semantics: a turn of less than 360 degrees is refused where the
    profile is too thin to cap; the full turn takes neither check - so
    one profile builds turned all the way, and cut short is refused or
    capped soundly, whichever the installed triangulator allows."""
    thin = [[2, 0], [2, 1e-14], [1, 5e-15], [0, 1e-14]]
    assert len(revolve.evaluate([], {"profile": thin}, None).faces) == 192
    try:
        cut = revolve.evaluate([], {"profile": thin, "arc_deg": 180.0}, None)
    except OpError as exc:
        assert exc.code == "bad_param"
    else:
        assert watertight(cut)


def _grooved(mouth: float) -> list[list[float]]:
    """A ring between radius 33 and 35, from 32 m up to 34 m, with a
    groove in its rim at 33 m."""
    return [[33.0, 32.0], [35.0, 32.0], [35.0, 33.0 - mouth / 2], [34.5, 33.0],
            [35.0, 33.0 + mouth / 2], [35.0, 34.0], [33.0, 34.0]]


def test_revolve_part_turn_refuses_a_profile_float32_closes():
    """Semantics: a turn of less than 360 degrees is refused where the
    profile does not survive its vertices rounding to float32; the
    full turn takes neither check - 33 m up, the lips of a groove
    2e-7 m wide land on one coordinate, and a millimeter builds."""
    assert np.float32(33 - 1e-7) == np.float32(33 + 1e-7)
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": _grooved(2e-7), "segments": 6, "arc_deg": 90.0}, None)
    assert exc.value.code == "bad_param" and "float32" in exc.value.message
    assert watertight(
        revolve.evaluate([], {"profile": _grooved(1e-3), "segments": 6, "arc_deg": 90.0}, None)
    )
    assert len(revolve.evaluate([], {"profile": _grooved(2e-7), "segments": 6}, None).faces) > 0


def _notched(mouth: float) -> list[list[float]]:
    """A flange 40 m out, a notch in its top face `mouth` wide where
    it opens."""
    r = 40.000149
    return [[r + 1, 0], [r + 1, 0.2], [r + mouth, 0.2], [r + mouth, 0.12], [r + 0.1, 0.1],
            [r - 0.1, 0.1], [r, 0.12], [r, 0.2], [r - 1, 0.2]]


def test_revolve_coarse_part_turn_refuses_walls_that_cross_between_columns():
    """Semantics: a turn of less than 360 degrees is refused where two
    walls round onto each other - across a step of 119 degrees the
    chord shows the profile at half its width, so a notch float32
    keeps open at every column closes between two of them, and a
    millimeter builds."""
    mouth = 4.32e-6
    assert np.float32(40.000149) != np.float32(40.000149 + mouth)
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": _notched(mouth), "segments": 3, "arc_deg": 357.0}, None)
    assert exc.value.code == "bad_param"
    assert "two walls round onto each other" in exc.value.message
    assert watertight(
        revolve.evaluate([], {"profile": _notched(1e-3), "segments": 3, "arc_deg": 357.0}, None)
    )


@pytest.mark.parametrize(
    "arc_deg,fragment", [(0, "> 0"), (-90, "> 0"), (360.5, "<= 360"), ("half", "finite number")]
)
def test_revolve_arc_refusals(arc_deg, fragment):
    """Semantics: `arc_deg` is above 0 and at most 360."""
    with pytest.raises(OpError) as exc:
        revolve.evaluate([], {"profile": DRUM, "arc_deg": arc_deg}, None)
    assert exc.value.code == "bad_param" and exc.value.detail["param"] == "arc_deg"
    assert fragment in exc.value.message
