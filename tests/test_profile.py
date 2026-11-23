"""The shared profile kernel (DESIGN 5): the facts the operators that
take an inline profile refuse on, the normals a joint carries, and the
triangles that cap a profile."""

import math

import numpy as np
import pytest

from moraine3d.core import profile
from moraine3d.core.mesh import triangles_meet

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


def test_facts_read_an_open_outline_and_a_loop():
    """DESIGN 5: one contract covers `revolve`'s open profile and a
    closed one, so each fact reads the closing segment only when the
    outline is a loop."""
    back = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert profile.duplicate(SQUARE) is None
    assert profile.duplicate(back) is None
    assert profile.duplicate(back, closed=True) == 3
    assert profile.duplicate([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]) == 0
    spike = [(0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert profile.double_back(spike) == 1
    wrap = [(2.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert profile.double_back(wrap) is None
    assert profile.double_back(wrap, closed=True) == 0
    assert profile.area2(SQUARE) == 8.0
    assert profile.area2(SQUARE[::-1]) == -8.0
    # Clear of both axes, so no term vanishes, and from every start.
    tri = [(2.0, 1.0), (2.0, 0.0), (3.0, 0.0)]
    assert [profile.area2(tri[k:] + tri[:k]) for k in range(3)] == [1.0, 1.0, 1.0]
    # A 1e-5 m square 1000 m out keeps its area and its tiling.
    far = [(1000.0, 1000.0), (1000.00001, 1000.0), (1000.00001, 1000.00001), (1000.0, 1000.00001)]
    assert profile.area2(far) == pytest.approx(2e-10, rel=1e-3)
    assert profile.triangulate(far) is not None
    bowtie = [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)]
    assert profile.self_intersection(SQUARE) is None
    assert profile.self_intersection(bowtie) == (0, 2)


def test_corners_carry_one_normal_when_smooth_and_two_when_hard():
    """DESIGN 5: a joint bending by more than `angle_max_deg` is hard
    and a gentler one smooth, on an open outline and around a loop."""
    hard = profile.corners(SQUARE, 30.0, closed=True)
    assert [len(c) for c in hard] == [2, 2, 2, 2]
    # Counter-clockwise: the incoming segment's normal, then the outgoing one's.
    assert hard[1] == [(0.0, -1.0), (1.0, 0.0)]
    soft = profile.corners(SQUARE, 90.0, closed=True)
    assert [len(c) for c in soft] == [1, 1, 1, 1]
    assert soft[1][0] == pytest.approx((2**-0.5, -(2**-0.5)))
    ends = profile.corners(SQUARE[:3], 30.0)
    assert [len(c) for c in ends] == [1, 2, 1]
    assert ends[0] == [(0.0, -1.0)] and ends[2] == [(1.0, 0.0)]


def test_clearance_is_how_near_the_outline_comes_to_itself():
    """DESIGN 5: a section that closes on itself when its vertices
    round is an error, and what closes is two edges that share no
    point - a square's opposite sides, the mouth of a notch."""
    assert profile.clearance(SQUARE) == 2.0
    assert profile.clearance(SQUARE[:3]) == float("inf")
    notch = [(0.0, 0.0), (3.0, 0.0), (3.0, 0.9), (2.5, 1.0), (3.0, 1.1), (3.0, 2.0), (0.0, 2.0)]
    # From one lip to the far wall of the notch, square to that wall.
    mouth = 0.2 * 0.5 / math.hypot(0.5, 0.1)
    assert profile.clearance(notch) == pytest.approx(mouth)
    assert profile.clearance(notch[3:] + notch[:3]) == pytest.approx(mouth)
    # The lip's edge and the far wall of the notch, and no pair beyond.
    assert [b.tolist() for b in profile.close_edges(notch, 0.2)] == [[[1, 3]], [[2, 4]]]
    assert list(profile.close_edges(notch, 0.1)) == []


@pytest.mark.parametrize(
    "poly",
    [
        SQUARE,
        # Collinear points on every side.
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)],
        # Concave, with collinear points.
        [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 1.5)],
    ],
)
def test_triangulate_tiles_the_polygon(poly):
    """DESIGN 5: a cap uses every point of its profile, so it welds to
    the walls around it, and covers the profile's area once."""
    poly = [(float(a), float(b)) for a, b in poly]
    tris = profile.triangulate(poly)
    assert tris is not None and len(tris) == len(poly) - 2
    assert set(tris.reshape(-1).tolist()) == set(range(len(poly)))
    p = np.array(poly)[tris]
    areas = profile.cross2(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (areas > 0).all()
    assert areas.sum() == pytest.approx(profile.area2(poly))


def test_valid_cap_refuses_a_triangle_turned_over():
    """DESIGN 5: a profile that cannot be triangulated into a valid cap
    is an error, and a cap with one triangle wound the other way is not
    valid though it uses every point and keeps every polygon edge."""
    t = 1e-14
    thin = [(0.0, 0.0), (2.0, 0.0), (2.0, t), (1.0, t / 2), (0.0, t)]
    turned = np.array([[0, 1, 2], [4, 0, 2], [3, 4, 2]])
    assert not profile.valid_cap(thin, turned)
    assert profile.valid_cap(thin, np.array([[0, 1, 3], [1, 2, 3], [0, 3, 4]]))


def test_valid_cap_refuses_inner_edges_left_unpaired():
    """DESIGN 5: a valid cap covers its profile once, so triangles that
    are each counter-clockwise and keep every polygon edge are still
    refused when an inner edge has no triangle on its far side."""
    hexagon = [(2.0, 0.0), (1.0, 2.0), (-1.0, 2.0), (-2.0, 0.0), (-1.0, -2.0), (1.0, -2.0)]
    overlapping = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0], [1, 3, 5]])
    assert not profile.valid_cap(hexagon, overlapping)
    assert profile.valid_cap(hexagon, np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0], [0, 2, 4]]))


def test_triangulate_never_returns_an_invalid_cap():
    """DESIGN 5: where the kernel cannot tile a profile, the answer is
    no cap, never a wrong one - whichever way the installed
    triangulator handles a profile thinner than it resolves."""
    t = 1e-14
    thin = [(0.0, 0.0), (2.0, 0.0), (2.0, t), (1.0, t / 2), (0.0, t)]
    tris = profile.triangulate(thin)
    assert tris is None or profile.valid_cap(thin, tris)


def test_triangles_meet_where_they_share_a_point():
    """DESIGN 5: walls that close on each other are an error, and two
    triangles close on each other where they share a point - crossing,
    touching at a corner or along an edge, or overlapping in one
    plane - and not where any axis holds them apart."""
    flat = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

    def meet(other) -> bool:
        return bool(triangles_meet(np.array([flat], float), np.array([other], float))[0])

    assert meet([[0.5, 0.5, -1], [0.5, 0.5, 1], [3, 3, 1]])      # pierces
    assert meet([[0, 0, 0], [0, 0, 1], [-1, 0, 1]])              # shares a corner
    assert meet([[1, 0, 0], [1, 0, 1], [1, -1, 1]])              # touches an edge
    assert meet([[1, 1, 0], [3, 1, 0], [1, 3, 0]])               # overlaps in the plane
    assert not meet([[0.5, 0.5, 1e-6], [0.5, 0.5, 1], [3, 3, 1]])   # ends just above
    assert not meet([[0, 0, 1e-6], [2, 0, 1e-6], [0, 2, 1e-6]])     # parallel, just above
    assert not meet([[2.1, 0, 0], [4, 0, 0], [2.1, 2, 0]])          # beside it in the plane
    assert not meet([[1.01, 1.01, -1], [1.01, 1.01, 1], [3, 3, 0]])  # passes its long edge


def test_wall_test_sets_a_bounded_number_of_walls_at_once(monkeypatch):
    """DESIGN 5: tessellation ceilings are absurdity guards, so an
    outline at its ceiling is judged in passes of bounded size - one
    span and twenty thousand pairs of edges reach the triangle test a
    pass at a time, never all at once."""
    seen = []

    def count(a, b):
        seen.append(len(a))
        return np.zeros(len(a), dtype=bool)

    monkeypatch.setattr(profile, "triangles_meet", count)
    q = 400
    angle = np.arange(q) * 2.0 * math.pi / q
    start = np.stack([np.cos(angle), np.sin(angle), np.zeros(q)], axis=1)[None]
    end = start + [0.0, 0.0, 1.0]
    k = np.arange(20000)
    pairs = np.stack([k % 100, 150 + k % 200], axis=1)
    assert not profile._walls_meet(start, end, pairs)
    assert sum(seen) == 4 * len(pairs)
    assert max(seen) <= profile._WALL_PAIRS < len(pairs)


def test_wall_test_leaves_out_a_wall_collapsed_onto_a_line():
    """DESIGN 5: walls that close on each other are an error, and a
    wall whose edge sits on an axis has only the one triangle - the
    collapsed one carries no face, so two such walls that stand apart
    are not taken to meet along the line they collapse onto."""
    # Outline points 0 and 3 sit on the axis the stations turn about.
    outline = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    turned = outline[:, [2, 1, 0]] * [-1.0, 1.0, 1.0]
    pairs = np.array([[0, 2]])  # the two rims that run out from the axis
    assert not profile._walls_meet(outline[None], turned[None], pairs)


def test_face_check_reads_a_bounded_number_of_faces_at_once(monkeypatch):
    """DESIGN 5: tessellation ceilings are absurdity guards, and a
    section that collapses or turns over when its vertices round to
    float32 is an error - so the faces are read in passes of bounded
    size, and one lost in a late pass is still counted."""
    seen = []
    inner = profile.tri_cross

    def count(tri):
        seen.append(len(tri))
        return inner(tri)

    monkeypatch.setattr(profile, "tri_cross", count)
    monkeypatch.setattr(profile, "_FACES", 8)
    k = np.arange(40.0)[:, None, None]
    sound = k * [1.0, 0.0, 0.0] + [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    faces = np.arange(120).reshape(40, 3)
    assert profile.rounded_faces_lost(sound.reshape(-1, 3), faces) == 0
    assert max(seen) <= 8 < len(faces)

    # 32 m out one float32 step is 2^-18 m: these corners round to
    # (0, 0), (2, 1), (4, 1) steps, the other way around.
    step = 2.0**-18
    turned = sound.copy()
    turned[30] = 32.0 + step * np.array([[0.0, 0.25, 0.0], [2.0, 0.625, 0.0], [4.0, 1.375, 0.0]])
    turned[30, :, 2] = 0.0
    assert profile.rounded_faces_lost(turned.reshape(-1, 3), faces) == 1

    # A sliver float32 keeps the right way around, thinner than the
    # degeneracy bar.
    thin = sound.copy()
    thin[37] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-10, 0.0]]
    assert profile.rounded_faces_lost(thin.reshape(-1, 3), faces) == 1
