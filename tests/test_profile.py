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
