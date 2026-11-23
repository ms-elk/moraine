"""Golden-mesh tests for the ``sweep`` operator (DESIGN 5, 10).

Counts, bounds, and volume within tolerance pin the semantics contract:
a closed inline profile carried along an inline polyline, mitered at
each bend and never twisting about the path, always watertight.
Volumes are computed via trimesh signed volume, so they also verify
CCW-outward winding.
"""

import math

import numpy as np
import pytest
import trimesh

from moraine3d.core.mesh import triangles_meet
from moraine3d.errors import OpError
from moraine3d.ops import sweep

# A 0.1 m square about the origin, and its 0.01 m2 area.
SQUARE = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
ELL = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 3.0]]


def _tm(m) -> trimesh.Trimesh:
    t = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
    t.merge_vertices()
    return t


def volume(m) -> float:
    return _tm(m).volume


def watertight(m) -> bool:
    return _tm(m).is_watertight


def build(profile, path, **params):
    return sweep.evaluate([], {"profile": profile, "path": path, **params}, None)


def bounds(m) -> tuple[list, list]:
    b = m.stats()["bounds"]
    return b["min"], b["max"]


def test_sweep_two_point_path_is_the_prism_golden():
    """Semantics: an open path is capped flat at both ends, so a 2-point
    path extrudes the profile - a square's four hard corners duplicate
    its vertices on both rings, and each cap carries its own four."""
    m = build(SQUARE, [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert len(m.vertices) == 2 * 8 + 2 * 4
    assert len(m.faces) == 4 * 2 + 2 * 2
    assert bounds(m) == ([-0.05, -0.05, 0.0], [0.05, 0.05, 2.0])
    assert volume(m) == pytest.approx(0.01 * 2.0, rel=1e-6)
    assert watertight(m)


def test_sweep_mitered_bend_volume_golden():
    """Semantics: bends are mitered and every segment's cross-section is
    the profile itself, so the volume is the profile's area times the
    path's length when the centroid sits on the origin."""
    m = build(SQUARE, ELL)
    # The 90-degree bend is hard at the default threshold: 4 ring copies.
    assert len(m.vertices) == 4 * 8 + 2 * 4
    assert len(m.faces) == 2 * 8 + 2 * 2
    assert bounds(m) == ([0.0, -0.05, -0.05], [2.05, 0.05, 3.0])
    assert volume(m) == pytest.approx(0.01 * 5.0, rel=1e-6)
    assert watertight(m)


def test_sweep_origin_rides_the_path():
    """Semantics: the profile's origin rides the path, and the volume is
    the profile's area times the length its centroid travels - a
    profile held 0.2 m to the inside of the bend stays there, and its
    centroid cuts the corner by 0.2 m on each leg."""
    inside = [[-0.3, -0.05], [-0.1, -0.05], [-0.1, 0.05], [-0.3, 0.05]]
    m = build(inside, ELL)
    assert bounds(m) == ([0.0, -0.05, 0.1], [1.9, 0.05, 3.0])
    assert volume(m) == pytest.approx(0.2 * 0.1 * (1.8 + 2.8), rel=1e-6)
    assert watertight(m)


def _normals_at(m, point) -> list[tuple]:
    here = np.all(np.isclose(m.vertices, point, atol=1e-6), axis=1)
    return sorted(tuple(np.round(n.astype(np.float64), 6).tolist()) for n in m.normals[here])


def test_sweep_bend_is_hard_or_smooth_by_the_threshold():
    """Semantics: a profile joint or a path bend turning by more than
    `angle_max_deg` is hard (duplicated vertices), a gentler one smooth
    - at the outer corner of a 90-degree bend, the default leaves each
    wall its own normal and a threshold above 90 blends them."""
    corner = [2.05, 0.05, -0.05]
    hard = build(SQUARE, ELL)
    # Two ring copies, each with the profile corner's two normals.
    assert _normals_at(hard, corner) == sorted(
        [(0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    )
    soft = build(SQUARE, ELL, angle_max_deg=100.0)
    assert len(soft.vertices) == 3 * 4 + 2 * 4
    want = np.array([1.0, 2.0, -1.0]) / math.sqrt(6.0)
    assert _normals_at(soft, corner) == [tuple(np.round(want, 6).tolist())]
    assert watertight(hard) and watertight(soft)


@pytest.mark.parametrize("tilt_deg,upright", [(0.0, False), (0.5, False), (2.0, True)])
def test_sweep_first_section_stands_upright(tilt_deg, upright):
    """Semantics: the first section's +y is world +Y projected into the
    plane across the first segment, +Z where that segment runs within 1
    degree of vertical - a 1 m blade along the profile's +y reaches
    +Z on a path within that degree, and leans back up the slope of
    one 2 degrees off vertical."""
    blade = [[-0.01, 0.0], [0.01, 0.0], [0.01, 1.0], [-0.01, 1.0]]
    lean = math.radians(tilt_deg)
    m = build(blade, [[0.0, 0.0, 0.0], [2.0 * math.sin(lean), 2.0 * math.cos(lean), 0.0]])
    lo, hi = bounds(m)
    if upright:
        assert lo[0] == pytest.approx(-math.cos(lean), abs=1e-6)
        assert (lo[2], hi[2]) == (-0.01, 0.01)
    else:
        assert hi[2] == pytest.approx(1.0, abs=1e-6)
        assert lo[0] == pytest.approx(-0.01, abs=1e-6)


def test_sweep_section_right_is_y_cross_the_direction():
    """Semantics: the section's +x is +y cross the segment's direction
    (looking back down the path, +x is right and +y up) - along +Z,
    +x is world +X."""
    right = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.1], [0.0, 0.1]]
    m = build(right, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert bounds(m) == ([0.0, 0.0, 0.0], [0.3, 0.1, 1.0])


def test_sweep_never_twists_about_the_path():
    """Semantics: each bend carries the section by the smallest rotation
    that turns one segment onto the next, so the section never twists
    about the path - on a level path its +y stays world +Y at every
    ring, whatever the bends."""
    tall = [[-0.01, -0.2], [0.01, -0.2], [0.01, 0.2], [-0.01, 0.2]]
    path = [[0, 0, 0], [1, 0, 0.4], [1.6, 0, 1.5], [0.9, 0, 2.4], [1.2, 0, 3.5]]
    m = build(tall, [[float(c) for c in p] for p in path])
    assert np.allclose(np.abs(m.vertices[:, 1]), 0.2, atol=1e-6)
    assert watertight(m)


def test_sweep_closed_path_is_a_loop_without_caps():
    """Semantics: `closed` joins the last path point to the first and
    makes a loop without caps - a square frame is four mitered legs,
    eight ring copies and no cap vertex."""
    frame = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    m = build(SQUARE, frame, closed=True)
    assert len(m.vertices) == 8 * 8
    assert len(m.faces) == 4 * 8
    assert bounds(m) == ([-0.05, -0.05, -0.05], [2.05, 0.05, 2.05])
    assert volume(m) == pytest.approx(0.01 * 8.0, rel=1e-6)
    assert watertight(m)


def test_sweep_closed_path_need_not_be_planar():
    """Semantics: a closed path is accepted where the section carried
    once around stands within 0.01 degrees of its start - the six unit
    legs +X, +Y, +Z, -X, -Y, -Z leave no plane and bring it back
    unturned, and a loop that brings it back turned is an error."""
    legs = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]]
    m = build(SQUARE, [[float(c) for c in p] for p in legs], closed=True)
    assert volume(m) == pytest.approx(0.01 * 6.0, rel=1e-6)
    assert watertight(m)
    with pytest.raises(OpError) as exc:
        build(SQUARE, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 2.0, 0.3], [0.2, 1.7, 2.0]],
              closed=True)
    assert exc.value.code == "bad_param" and "back turned" in exc.value.message


def test_sweep_refuses_a_section_float32_collapses():
    """Semantics: a triangle that collapses when its vertices round to
    float32 is an error - 32 m out, the two sides of a profile 1e-6 m
    wide round to the same coordinate."""
    sliver = [[-5e-7, -0.1], [5e-7, -0.1], [5e-7, 0.1], [-5e-7, 0.1]]
    assert np.float32(32 - 5e-7) == np.float32(32 + 5e-7)
    with pytest.raises(OpError) as exc:
        build(sliver, [[32.0, 0.0, 0.0], [32.0, 0.0, 1.0]])
    assert exc.value.code == "bad_param" and "float32" in exc.value.message


def test_sweep_refuses_a_section_float32_turns_over():
    """Semantics: a triangle that turns over when its vertices round to
    float32 is an error - this profile's area changes sign on the
    float32 grid, which would leave a watertight solid of negative
    volume with no collapsed triangle to show for it."""
    s = 2.0**-18  # the float32 step between 32 and 64
    turned = [[0.0, 0.25 * s], [2 * s, 0.625 * s], [4 * s, 1.375 * s]]
    rounded = np.array(turned, dtype=np.float64) + 32.0
    rounded = rounded.astype(np.float32).astype(np.float64)
    a, b, c = rounded
    assert (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0
    with pytest.raises(OpError) as exc:
        build(turned, [[32.0, 32.0, 0.0], [32.0, 32.0, 1.0]])
    assert exc.value.code == "bad_param" and "float32" in exc.value.message


def _notched(mouth: float) -> list[list[float]]:
    return [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0 - mouth / 2], [2.5, 1.0],
            [3.0, 1.0 + mouth / 2], [3.0, 2.0], [0.0, 2.0]]


def test_sweep_refuses_a_section_float32_closes_on_itself():
    """Semantics: a section does not survive where its vertices weld
    into an open surface - 32 m out, the two lips of a notch 2e-7 m
    wide land on one coordinate, each triangle keeping its area and
    its facing while four faces come to share an edge - and a notch a
    millimeter wide builds there."""
    out = [[32.0, 32.0, 0.0], [32.0, 32.0, 1.0]]
    assert np.float32(33 - 1e-7) == np.float32(33 + 1e-7)
    with pytest.raises(OpError) as exc:
        build(_notched(2e-7), out)
    assert exc.value.code == "bad_param" and "weld into an open" in exc.value.message
    # Two float32 steps apart there, the lips stay apart: what the
    # vertices round to decides, not how fine the notch is drawn.
    assert watertight(build(_notched(6e-6), out))
    assert watertight(build(_notched(1e-3), out))


def _walls_cross(m) -> bool:
    """Do any two triangles of the mesh that share no corner meet?"""
    v = m.vertices.astype(np.float64)
    f = m.faces
    i, j = np.triu_indices(len(f), k=1)
    apart = np.array([
        len(np.unique(np.vstack([v[f[a]], v[f[b]]]), axis=0)) == 6 for a, b in zip(i, j)
    ])
    return bool(triangles_meet(v[f[i[apart]]], v[f[j[apart]]]).any())


def _slotted(mouth: float) -> list[list[float]]:
    """A slot 2 m deep, turned 30 degrees so no wall lies along an axis."""
    outline = [(0, 0), (4, 0), (4, 1), (2, 1), (1, 0.8), (1, 1.2),
               (2, 1 + mouth), (4, 1 + mouth), (4, 2), (0, 2)]
    c, s = math.cos(math.radians(-30)), math.sin(math.radians(-30))
    return [[x * c - y * s, x * s + y * c] for x, y in outline]


def test_sweep_refuses_walls_that_cross_between_sound_rings():
    """Semantics: a section does not survive where two walls of the
    profile round onto each other - a slot 1e-6 m wide, swept 40 m out
    on a slant, keeps its two walls in order at both rings while the
    triangles between them cross, and a slot 2e-5 m wide builds there
    with no triangle touching another."""
    slant = [[35.7, 44.8, 36.3], [35.7, 44.3, 36.2]]
    with pytest.raises(OpError) as exc:
        build(_slotted(1e-6), slant)
    assert exc.value.code == "bad_param" and "round onto each other" in exc.value.message
    wide = build(_slotted(2e-5), slant)
    assert watertight(wide) and not _walls_cross(wide)


def test_sweep_wall_test_reads_a_gap_at_its_own_scale():
    """Semantics: two walls of the profile that round onto each other
    are an error, and walls that stay apart are none however long they
    run - 30 nm apart at the origin, the two faces of this plate are
    told apart by float32 over 500 m of path in one segment or in two."""
    plate = [[0.0, 0.0], [3e-8, 0.0], [3e-8, 0.1], [0.0, 0.1]]
    whole = build(plate, [[0.0, 32.0, 0.0], [0.0, 32.0, 500.0]])
    halves = build(plate, [[0.0, 32.0, 0.0], [0.0, 32.0, 250.0], [0.0, 32.0, 500.0]])
    for m in (whole, halves):
        assert volume(m) == pytest.approx(3e-8 * 0.1 * 500.0, rel=1e-4)
        assert watertight(m)


def test_sweep_rounding_along_the_path_closes_no_wall():
    """Semantics: a section does not survive where two walls of the
    profile round onto each other, and only there - 800 m along +Z the
    float32 step is 6e-5 m, wider than this blade is thick, but it
    falls along the path, where no wall of the blade can meet another."""
    blade = [[-5e-5, -0.05], [5e-5, -0.05], [5e-5, 0.05], [-5e-5, 0.05]]
    assert float(np.spacing(np.float32(800.0))) > 1e-4 / 2
    m = build(blade, [[0.0, 0.0, 800.0], [0.0, 0.0, 801.0]])
    assert volume(m) == pytest.approx(1e-4 * 0.1 * 1.0, rel=1e-4)
    assert watertight(m)


def test_sweep_refuses_a_section_the_weld_opens():
    """Semantics: a section does not survive where its vertices weld
    into an open surface - at the origin float32 tells apart the lips
    of a notch 5e-9 m wide, and the weld `validate` judges a surface by
    does not."""
    tiny = [[c * 1e-3 for c in p] for p in _notched(5e-6)]
    with pytest.raises(OpError) as exc:
        build(tiny, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
    assert exc.value.code == "bad_param" and "weld" in exc.value.message


def test_sweep_profile_may_sit_far_from_its_origin():
    """Semantics: the profile's origin rides the path, wherever the
    profile sits from it - a 1e-5 m square drawn 1000 m out, on a path
    1000 m the other way, is a prism at the world origin."""
    side = 1e-5
    far = [[1000.0, 1000.0], [1000.0 + side, 1000.0], [1000.0 + side, 1000.0 + side],
           [1000.0, 1000.0 + side]]
    m = build(far, [[-1000.0, -1000.0, 0.0], [-1000.0, -1000.0, 1.0]])
    assert bounds(m) == ([0.0, 0.0, 0.0], [1e-05, 1e-05, 1.0])
    assert volume(m) == pytest.approx(side * side, rel=1e-3)
    assert watertight(m)


def test_sweep_refuses_a_segment_its_miters_turn_inside_out():
    """Semantics: a segment too short for the miters at its ends is an
    error - a profile 1.1 m to the inside of a bend 1 m along ends
    before it starts, though the end faces cross outside the profile."""
    far_inside = [[0.0, 1.1], [0.1, 1.1], [0.1, 1.2], [0.0, 1.2]]
    with pytest.raises(OpError) as exc:
        build(far_inside, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 5.0, 0.0]])
    assert exc.value.code == "bad_param"
    assert "segment from point 0 is too short" in exc.value.message


def _returning(gap: float) -> list[list[float]]:
    """A path whose last leg runs back past the mitered corner of its
    first bend, `gap` clear of the corner's tip."""
    z = -0.1 - gap
    return [ELL[0], ELL[1], ELL[2], [5.0, 0.0, 3.0], [5.0, 0.0, z], [2.0, 0.0, z]]


def test_sweep_refuses_segments_that_overlap_or_touch():
    """Semantics: two segments not adjacent whose solids overlap or
    touch are an error - a leg passing the tip of a miter builds with a
    millimeter to spare, and is refused where it touches the tip or
    runs a millimeter into it."""
    assert watertight(build(SQUARE, _returning(1e-3)))
    for gap in (0.0, -1e-3):
        with pytest.raises(OpError) as exc:
            build(SQUARE, _returning(gap))
        assert exc.value.code == "bad_param"
        assert "overlap or touch" in exc.value.message


def test_sweep_accepts_short_segments_in_a_line(monkeypatch):
    """Semantics: only segments whose solids overlap or touch are
    refused, so a straight run cut into pieces far shorter than the
    profile is wide builds as the one prism - along an axis or across
    all three, and without one pair of its segments reaching the
    kernel, since each lies past the end of every other."""
    monkeypatch.setattr(sweep, "_segment_solid", None)
    for end in ([0.0, 0.0, 0.49], [0.03, 0.03, 0.03]):
        path = np.linspace([0.0, 0.0, 0.0], end, 1024).tolist()
        m = build(SQUARE, path)
        assert volume(m) == pytest.approx(0.01 * math.dist(path[0], path[-1]), rel=1e-5)
        assert watertight(m)


CLOCKWISE = SQUARE[::-1]
LINE = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.mark.parametrize(
    "params,fragment",
    [
        ({"path": LINE}, "requires 'profile'"),
        ({"profile": SQUARE[:2], "path": LINE}, "list of 3"),
        ({"profile": [[0.0, 0.0, 0.0], *SQUARE[1:]], "path": LINE}, "point 0"),
        ({"profile": [SQUARE[0], *SQUARE], "path": LINE}, "points 0 and 1 coincide"),
        ({"profile": [*SQUARE, SQUARE[0]], "path": LINE}, "drop the last point"),
        ({"profile": [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.0]], "path": LINE},
         "doubles back"),
        ({"profile": [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], "path": LINE},
         "self-intersects"),
        ({"profile": CLOCKWISE, "path": LINE}, "reverse the point order"),
        ({"profile": SQUARE}, "requires 'path'"),
        ({"profile": SQUARE, "path": LINE[:1]}, "list of 2"),
        ({"profile": SQUARE, "path": LINE, "closed": True}, "list of 3"),
        ({"profile": SQUARE, "path": [[0.0, 0.0], [0.0, 0.0, 1.0]]}, "point 0"),
        ({"profile": SQUARE, "path": [LINE[0], LINE[0], LINE[1]]}, "points 0 and 1 coincide"),
        ({"profile": SQUARE, "path": [LINE[0], LINE[1], [1.0, 0.0, 1.0], LINE[0]]},
         "set closed = True"),
        ({"profile": SQUARE, "path": [LINE[0], LINE[1], [0.2, 0.0, 0.0]]}, "beyond 150"),
        ({"profile": SQUARE, "path": LINE, "closed": 1}, "boolean"),
        ({"profile": SQUARE, "path": LINE, "angle_max_deg": 0}, "> 0"),
        ({"profile": SQUARE, "path": LINE, "angle_max_deg": 181}, "<= 180"),
    ],
)
def test_sweep_refusals(params, fragment):
    """Semantics: a degenerate or clockwise profile and a degenerate or
    too sharply bent path are structured refusals naming the fix
    (DESIGN 8.2), never silent geometry."""
    with pytest.raises(OpError) as exc:
        sweep.evaluate([], params, None)
    assert exc.value.code == "bad_param"
    assert fragment in exc.value.message


def test_sweep_bend_limit_is_inclusive():
    """Semantics: a bend beyond 150 degrees is an error, so one a hair
    under it builds."""
    def bent(deg: float) -> list[list[float]]:
        a = math.radians(deg)
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [4.0 * math.sin(a), 0.0, 4.0 + 4.0 * math.cos(a)]]

    thin = [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]]
    assert watertight(build(thin, bent(149.9)))
    with pytest.raises(OpError) as exc:
        build(thin, bent(150.1))
    assert "beyond 150" in exc.value.message


def test_sweep_cell_budget_refused_before_allocation():
    """Semantics: a rings x profile vertices product beyond the shared
    4096 x 4096 cell budget is refused (hard joints duplicate both, so
    the per-param ceilings alone do not bound the product) - a
    structured refusal, never a raw allocation failure (DESIGN 5)."""
    # A 4096-tooth outline and a 4096-point staircase, every joint hard.
    gear = [
        [r * math.cos(a), r * math.sin(a)]
        for k in range(4096)
        for r, a in [(1.0 if k % 2 else 0.9, k * 2.0 * math.pi / 4096)]
    ]
    stairs = [[float(k // 2 + k % 2), 0.0, float(k // 2)] for k in range(4096)]
    with pytest.raises(OpError) as exc:
        build(gear, stairs)
    assert exc.value.code == "bad_param"
    assert "cell budget" in exc.value.message


def test_sweep_point_count_ceilings():
    """Semantics: the profile and the path each hold at most 4096
    points - beyond that the count is a typo, refused rather than
    surfacing as a raw allocation failure."""
    ring = [[math.cos(k * 1e-3), math.sin(k * 1e-3)] for k in range(4097)]
    with pytest.raises(OpError) as exc:
        build(ring, LINE)
    assert "4096" in exc.value.message
    with pytest.raises(OpError) as exc:
        build(SQUARE, [[0.0, 0.0, float(k)] for k in range(4097)])
    assert "4096" in exc.value.message


def test_sweep_result_hash_frozen():
    """DESIGN 4.5: the result of the shipped example's sweep, frozen so
    a change to what the operator returns shows here before it reaches
    a cache record. Level legs along +X and +Z, so no trigonometric
    value reaches a vertex."""
    assert sweep.OP_IMPL_VERSION == 1
    rail = [[-0.03, 0.0], [0.03, 0.0], [0.03, 0.03], [0.015, 0.04], [-0.015, 0.04], [-0.03, 0.03]]
    m = build(rail, [[0.0, 0.9, 0.0], [2.0, 0.9, 0.0], [2.0, 0.9, 1.2]])
    assert len(m.faces) == 32
    assert m.result_hash() == (
        "86d8c5977e56fab1e4ce63967f6c6b8cc188ad53fcd6077af09b999383d8046f"
    )
