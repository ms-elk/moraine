"""Golden-mesh tests for operator set v1 (DESIGN 5, 10).

Counts, bounds, and volume within tolerance pin each operator's
semantics contract; implementation swaps must keep these green.
Volumes are computed via trimesh signed volume, so they also verify
CCW-outward winding.
"""

import numpy as np
import pytest
import trimesh

from moraine3d.core.mesh import Instances, Mesh, degenerate_tri_count, tri_cross
from moraine3d.engine.evaluator import OpContext
from moraine3d.errors import OpError
from moraine3d.ops import (
    array,
    bevel,
    boolean,
    box,
    cylinder,
    extrude,
    grid,
    noise_displace,
    scatter,
    smooth_normals,
    sphere,
    transform,
    unwrap,
)


def volume(mesh) -> float:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.merge_vertices()
    return float(tm.volume)


def watertight(mesh) -> bool:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.merge_vertices()
    return bool(tm.is_watertight)


# -- primitives -------------------------------------------------------------


def test_cylinder_golden():
    """``cylinder``: a Y-axis cylinder centered at the origin, smooth
    shared-ring sides and hard fan-closed caps, so the vertex and face
    counts, the bounds, and the exact prism volume of the inscribed
    32-gon follow from the semantics."""
    m = cylinder.evaluate([], {"radius": 0.5, "height": 2.0, "segments": 32}, None)
    assert len(m.vertices) == 2 * 32 + 2 * (32 + 1)
    assert len(m.faces) == 2 * 32 + 2 * 32
    s = m.stats()
    assert s["bounds"]["min"] == [-0.5, -1.0, -0.5]
    assert s["bounds"]["max"] == [0.5, 1.0, 0.5]
    # Exact prism volume of the inscribed 32-gon.
    expected = 0.5 * 32 * 0.5**2 * np.sin(2 * np.pi / 32) * 2.0
    assert volume(m) == pytest.approx(expected, rel=1e-5)
    assert watertight(m)


def test_sphere_golden():
    """Semantics: a UV sphere of ``segments`` longitude and ``rings``
    latitude divisions with pole fans and quad bands — every cross
    section is the inscribed polygon and every band a frustum of the
    inscribed pyramid, so the volume is exact, not an approximation of
    4/3 pi r^3."""
    radius, seg, rings = 0.5, 32, 16
    m = sphere.evaluate([], {"radius": radius, "segments": seg, "rings": rings}, None)
    assert len(m.vertices) == (rings - 1) * seg + 2
    assert len(m.faces) == 2 * seg + (rings - 2) * 2 * seg
    lat = np.arange(rings + 1) * (np.pi / rings)
    r, y = radius * np.sin(lat), radius * np.cos(lat)
    h = y[:-1] - y[1:]
    frusta = np.sum(h / 3.0 * (r[:-1] ** 2 + r[:-1] * r[1:] + r[1:] ** 2))
    polygon_factor = 0.5 * seg * np.sin(2.0 * np.pi / seg)
    assert volume(m) == pytest.approx(polygon_factor * frusta, rel=1e-5)
    assert watertight(m)
    # Smooth shading: normals equal normalized positions.
    n = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    assert np.allclose(n, m.normals, atol=1e-6)


def test_grid_golden():
    """``grid``: a flat XZ rectangle at y = 0 facing +Y, vertices shared,
    normals all +Y, open by design as the base for ``extrude``."""
    m = grid.evaluate([], {"size": [2.0, 3.0], "segments": [2, 3]}, None)
    assert len(m.vertices) == 3 * 4
    assert len(m.faces) == 2 * 2 * 3
    s = m.stats()
    assert s["bounds"]["min"] == [-1.0, 0.0, -1.5]
    assert s["bounds"]["max"] == [1.0, 0.0, 1.5]
    assert np.all(m.normals == [0.0, 1.0, 0.0])
    assert not watertight(m)  # open sheet by design


# -- transform --------------------------------------------------------------


def test_transform_trs():
    """``transform``: a mesh maps through ``M = T @ Rz @ Ry @ Rx @ S``, so
    a 90-degree yaw of a [1, 2, 3] box swaps its X and Z extents around
    the translation and keeps its volume."""
    b = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    m = transform.evaluate(
        [b], {"translate": [10.0, 0.0, 0.0], "rotate_deg": [0.0, 90.0, 0.0]}, None
    )
    s = m.stats()
    # 90 deg about Y maps [1,2,3] extents to [3,2,1] around x=10.
    assert s["bounds"]["min"] == pytest.approx([8.5, -1.0, -0.5], abs=1e-5)
    assert s["bounds"]["max"] == pytest.approx([11.5, 1.0, 0.5], abs=1e-5)
    assert volume(m) == pytest.approx(6.0, rel=1e-6)


def test_transform_rotates_about_x_then_y_then_z():
    """transform semantics: ``rotate_deg`` is applied extrinsically in
    X, Y, Z order, ``M = T @ Rz @ Ry @ Rx @ S``. A quarter turn about X
    then one about Y carries +Y to +Z and on to +X; the reverse order
    would leave it at +Z. One about Y then one about Z carries +X to
    -Z and leaves it there; the reverse order would end at +Y.
    ``array``'s step is the same matrix."""
    m = transform.matrix({"rotate_deg": [90.0, 90.0, 0.0]})
    assert np.allclose(m[:3, :3] @ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], atol=1e-12)
    m = transform.matrix({"rotate_deg": [0.0, 90.0, 90.0]})
    assert np.allclose(m[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)
    row = array.evaluate(
        [box.evaluate([], {}, None)],
        {"count": 2, "step_rotate_deg": [90.0, 90.0, 0.0]}, None,
    )
    assert np.allclose(row.transforms[1][:3, :3] @ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], atol=1e-12)


def test_transform_mirror_keeps_outward_winding():
    """``transform``: a negative-determinant scale flips winding to keep
    faces CCW-outward (DESIGN 4.3), so a mirrored box keeps a positive
    volume."""
    b = box.evaluate([], {}, None)
    m = transform.evaluate([b], {"scale": [-1.0, 1.0, 1.0]}, None)
    assert volume(m) == pytest.approx(1.0, rel=1e-6)  # positive => still CCW outward


def test_transform_on_instances_moves_the_set():
    """``transform`` on Instances: M pre-multiplies every instance
    transform, so the whole set moves rigidly and stays instanced
    (Day-One Invariant 6)."""
    b = box.evaluate([], {}, None)
    inst = array.evaluate([b], {"count": 2, "step_translate": [2.0, 0.0, 0.0]}, None)
    moved = transform.evaluate([inst], {"translate": [0.0, 5.0, 0.0]}, None)
    assert isinstance(moved, Instances)
    lo, hi = moved.bounds()
    assert lo[1] == pytest.approx(4.5)
    assert hi[0] == pytest.approx(2.5)


# -- extrude ----------------------------------------------------------------


def test_extrude_grid_to_slab():
    """``extrude``: the output is the offset front, the flipped back, and
    hard-edged side quads along every boundary edge, so a unit grid at
    distance 0.5 is a watertight slab of volume 0.5."""
    g = grid.evaluate([], {"size": [1.0, 1.0]}, None)
    m = extrude.evaluate([g], {"distance": 0.5}, None)
    # front 4 + back 4 + 4 boundary edges * 4 = 24 verts; 2+2+8 tris.
    assert len(m.vertices) == 24
    assert len(m.faces) == 12
    assert volume(m) == pytest.approx(0.5, rel=1e-6)
    assert watertight(m)


def test_extrude_vector_direction():
    """``extrude``: a fixed ``direction`` vector is normalized and scaled
    by ``distance``, so its own length never reaches the sweep."""
    g = grid.evaluate([], {"size": [2.0, 2.0]}, None)
    m = extrude.evaluate([g], {"distance": 1.0, "direction": [0.0, 2.0, 0.0]}, None)
    s = m.stats()
    assert s["bounds"]["max"][1] == pytest.approx(1.0)  # direction is normalized
    assert volume(m) == pytest.approx(4.0, rel=1e-6)


def test_extrude_rejects_zero_distance():
    """``extrude``: ``distance`` must be > 0; a sweep of no length is
    refused, not emitted as a degenerate solid."""
    g = grid.evaluate([], {}, None)
    with pytest.raises(OpError):
        extrude.evaluate([g], {"distance": 0}, None)


def test_extrude_opposing_direction_builds_outward():
    """A vector that opposes every face normal sweeps the flipped sheet
    (extrude semantics): the output is the intended outward solid, not
    an inside-out one that passes watertight and fails downstream."""
    g = grid.evaluate([], {"size": [1.0, 1.0]}, None)
    m = extrude.evaluate([g], {"distance": 0.5, "direction": [0.0, -1.0, 0.0]}, None)
    assert watertight(m)
    assert volume(m) == pytest.approx(0.5, rel=1e-6)  # positive => outward
    s = m.stats()
    assert s["bounds"]["min"][1] == pytest.approx(-0.5)  # below the sheet
    assert s["bounds"]["max"][1] == pytest.approx(0.0)


def test_extrude_tangent_direction_is_loud():
    """A vector tangent to any face is an actionable error (extrude
    semantics): the sweep would produce degenerate side slabs."""
    g = grid.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        extrude.evaluate([g], {"direction": [1.0, 0.0, 0.0]}, None)
    assert e.value.code == "bad_param"
    assert e.value.detail["param"] == "direction"


def test_extrude_mixed_orientation_direction_is_loud():
    """A vector opposing some faces while aligning with others is an
    actionable error (extrude semantics): the sweep would
    self-intersect."""
    # Two triangles sharing an edge, wound to face +Y and -Y.
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 2, 1], [1, 3, 2]], dtype=np.int32)
    with pytest.raises(OpError) as e:
        extrude.evaluate(
            [Mesh(vertices=verts, faces=faces)],
            {"direction": [0.0, 1.0, 0.0]},
            None,
        )
    assert e.value.code == "bad_param"


def test_extrude_flat_shaded_closed_input_is_loud():
    """A flat-shaded closed surface (duplicated vertices with divergent
    normals) is refused (extrude semantics): it would extrude into
    per-face slabs with open wedges, not a thickened shell."""
    b = box.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        extrude.evaluate([b], {"distance": 0.1}, None)
    assert e.value.code == "bad_input"
    assert "angle_max_deg" in e.value.message  # the terminating remedy


def test_extrude_refusal_remedy_terminates():
    """The refusal's named remedy works (extrude semantics: weld it
    smooth first with angle_max_deg above the hardest edge): the welded
    closed surface has no boundary edges and yields a two-shell solid."""
    b = box.evaluate([], {}, None)
    welded = smooth_normals.evaluate([b], {"angle_max_deg": 91.0}, None)
    m = extrude.evaluate([welded], {"distance": 0.05}, None)
    assert watertight(m)
    assert volume(m) > 0  # outer shell minus inner shell


def test_extrude_seam_split_closed_smooth_input_passes():
    """Seam-split closed surfaces whose duplicates carry agreeing
    normals (UV seams) are smooth-shaded, not flat-shaded, and extrude
    as their boundary edges imply (extrude semantics)."""
    s = sphere.evaluate([], {"segments": 16, "rings": 8}, None)
    seamed = unwrap.evaluate([s], {"mode": "box"}, None)
    m = extrude.evaluate([seamed], {"distance": 0.05}, None)
    assert len(m.faces) >= 2 * len(seamed.faces)  # front + back shells built


# -- boolean ----------------------------------------------------------------


def _two_boxes():
    a = box.evaluate([], {}, None)
    b = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.5, 0.0, 0.0]}, None
    )
    return a, b


@pytest.mark.parametrize(
    "mode,expected", [("union", 1.5), ("difference", 0.5), ("intersection", 0.5)]
)
def test_boolean_volumes(mode, expected):
    """``boolean``: union, difference (input 0 minus the rest), and
    intersection of watertight inputs, each a watertight solid of the
    expected volume."""
    a, b = _two_boxes()
    m = boolean.evaluate([a, b], {"mode": mode}, None)
    assert volume(m) == pytest.approx(expected, rel=1e-5)
    assert watertight(m)


def test_boolean_rejects_open_input():
    """``boolean``: a non-watertight input is an actionable error
    (``not_watertight``), not a silent repair."""
    a = box.evaluate([], {}, None)
    g = grid.evaluate([], {}, None)
    with pytest.raises(OpError) as exc:
        boolean.evaluate([a, g], {"mode": "union"}, None)
    assert exc.value.code == "not_watertight"


# -- array / scatter (instancing) ------------------------------------------


def test_array_linear():
    """``array``: a mesh input becomes Instances of that prototype with
    ``count`` transforms, copy i at M^i, emitted as prototype +
    transforms and never as baked geometry."""
    b = box.evaluate([], {}, None)
    inst = array.evaluate([b], {"count": 3, "step_translate": [2.0, 0.0, 0.0]}, None)
    assert isinstance(inst, Instances)
    assert inst.transforms.shape == (3, 4, 4)
    xs = inst.transforms[:, 0, 3]
    assert xs == pytest.approx([0.0, 2.0, 4.0])
    assert inst.stats()["instances"] == 3


def test_array_nested_makes_grid():
    """``array`` on Instances: each step matrix is applied to every
    existing instance, so nested arrays form grids without merging
    (Day-One Invariant 6)."""
    b = box.evaluate([], {}, None)
    row = array.evaluate([b], {"count": 3, "step_translate": [1.0, 0.0, 0.0]}, None)
    grid_inst = array.evaluate([row], {"count": 2, "step_translate": [0.0, 1.0, 0.0]}, None)
    assert grid_inst.transforms.shape == (6, 4, 4)
    pts = {(round(float(t[0, 3]), 6), round(float(t[1, 3]), 6)) for t in grid_inst.transforms}
    assert pts == {(x, y) for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0)}


def test_array_nested_numbers_copies_outer_first():
    """Over an input of k copies, copy i is this array's step i // k of
    the input's copy i % k (array semantics): the index a report prints
    as name[i] is computable from the two counts."""
    b = box.evaluate([], {}, None)
    row = array.evaluate([b], {"count": 3, "step_translate": [1.0, 0.0, 0.0]}, None)
    grid_inst = array.evaluate([row], {"count": 2, "step_translate": [0.0, 1.0, 0.0]}, None)
    i = np.arange(6)
    assert grid_inst.transforms[:, 0, 3] == pytest.approx(i % 3)
    assert grid_inst.transforms[:, 1, 3] == pytest.approx(i // 3)


def test_scatter_deterministic_and_on_surface():
    """``scatter``: points lie on the target surface, and streams derive
    as hash(seed, node_id, index) (DESIGN 4.4), so a node reproduces
    exactly, two nodes with equal params decorrelate, and the seed
    selects the stream."""
    proto = box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None)
    target = grid.evaluate([], {"size": [2.0, 2.0]}, None)
    i1 = scatter.evaluate([target, proto], {"count": 20, "seed": 1}, OpContext("sc01"))
    i2 = scatter.evaluate([target, proto], {"count": 20, "seed": 1}, OpContext("sc01"))
    assert np.array_equal(i1.transforms, i2.transforms)
    pos = i1.transforms[:, :3, 3]
    assert np.all(np.abs(pos[:, 0]) <= 1.0) and np.all(np.abs(pos[:, 2]) <= 1.0)
    assert np.all(pos[:, 1] == 0.0)
    # Same params, different node id -> decorrelated stream (DESIGN 4.4).
    i3 = scatter.evaluate([target, proto], {"count": 20, "seed": 1}, OpContext("sc02"))
    assert not np.array_equal(i1.transforms, i3.transforms)
    # Same node id, different seed -> a different placement (DESIGN 4.4:
    # every stochastic op takes an explicit seed that selects the stream).
    i4 = scatter.evaluate([target, proto], {"count": 20, "seed": 2}, OpContext("sc01"))
    assert i1.result_hash() != i4.result_hash()


def test_scatter_count_appends_copies():
    """scatter semantics: a copy's draws depend on its own index alone,
    never on ``count``, so raising ``count`` appends copies and leaves
    the earlier ones where they were, position, yaw, and scale alike."""
    proto = box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None)
    target = sphere.evaluate([], {}, None)
    params = {"seed": 4, "align_normal": True, "rand_yaw": True, "scale_jitter": [0.5, 2.0]}
    ten = scatter.evaluate([target, proto], {**params, "count": 10}, OpContext("sc01"))
    eleven = scatter.evaluate([target, proto], {**params, "count": 11}, OpContext("sc01"))
    assert np.array_equal(eleven.transforms[:10], ten.transforms)
    assert not np.array_equal(eleven.transforms[10], ten.transforms[9])


def test_scatter_align_normal_level_target_keeps_yaw():
    """Semantics: with align_normal, instance +Y aligns to the surface
    normal and a level target leaves the copy's yaw unchanged — the
    frame is the identity where the normal is +Y, so a prototype whose
    front faces +Z (DESIGN 4.3) is scattered still facing +Z."""
    proto = box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None)
    target = grid.evaluate([], {"size": [2.0, 2.0]}, None)
    inst = scatter.evaluate(
        [target, proto],
        {"count": 12, "seed": 5, "align_normal": True, "rand_yaw": False},
        OpContext("sc01"),
    )
    lin = inst.transforms[:, :3, :3]
    assert np.allclose(lin, np.eye(3), atol=1e-12)
    # On a curved target the frame is still an aligned, right-handed
    # rotation: column y is the face normal, columns orthonormal.
    curved = scatter.evaluate(
        [sphere.evaluate([], {}, None), proto],
        {"count": 12, "seed": 5, "align_normal": True},
        OpContext("sc01"),
    )
    lin = curved.transforms[:, :3, :3]
    assert np.allclose(lin @ np.transpose(lin, (0, 2, 1)), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(lin), 1.0, atol=1e-9)
    up = lin[:, :, 1]
    pos = curved.transforms[:, :3, 3]
    # A UV sphere's face normals point outward, within the facet angle.
    assert np.all(np.einsum("ij,ij->i", up, pos / np.linalg.norm(pos, axis=1, keepdims=True)) > 0.9)


def test_scatter_align_and_jitter_stay_deterministic():
    """``scatter``: ``align_normal``, ``rand_yaw``, and ``scale_jitter``
    draw from the same node-salted streams (DESIGN 4.4), so the
    placement reproduces and every frame stays right-handed."""
    proto = box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None)
    target = sphere.evaluate([], {}, None)
    params = {
        "count": 8, "seed": 3, "align_normal": True, "rand_yaw": True,
        "scale_jitter": [0.5, 1.5],
    }
    i1 = scatter.evaluate([target, proto], params, OpContext("sc01"))
    i2 = scatter.evaluate([target, proto], params, OpContext("sc01"))
    assert np.array_equal(i1.transforms, i2.transforms)
    dets = np.linalg.det(i1.transforms[:, :3, :3])
    assert np.all(dets > 0)  # right-handed frames, no mirroring


# -- noise_displace ---------------------------------------------------------


def test_noise_displace_flat_shaded_input_is_loud():
    """A flat-shaded input (coincident duplicated vertices with
    divergent normals) is refused (noise_displace semantics): displacing
    the duplicates apart tears the surface at every hard edge."""
    b = box.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        noise_displace.evaluate([b], {"amplitude": 0.05}, None)
    assert e.value.code == "bad_input"
    assert "angle_max_deg" in e.value.message  # the terminating remedy


def test_noise_displace_refusal_remedy_terminates():
    """The refusal's named remedy works (noise_displace semantics: weld
    the shading first with angle_max_deg above the hardest edge): the
    fully welded solid displaces without tearing."""
    b = box.evaluate([], {}, None)
    welded = smooth_normals.evaluate([b], {"angle_max_deg": 91.0}, None)
    m = noise_displace.evaluate([welded], {"amplitude": 0.02, "seed": 5}, None)
    assert np.isfinite(m.vertices).all()
    assert watertight(m)  # welded topology cannot tear


def test_noise_displace_nan_normals_are_loud():
    """Non-finite normals on duplicated vertices count as divergent
    (noise_displace semantics: the guard fails loud, never passes on a
    NaN comparison and emits NaN geometry)."""
    b = box.evaluate([], {}, None)
    n = b.normals.copy()
    n[0] = np.nan
    poisoned = Mesh(vertices=b.vertices, faces=b.faces, normals=n)
    with pytest.raises(OpError) as e:
        noise_displace.evaluate([poisoned], {"amplitude": 0.05}, None)
    assert e.value.code == "bad_input"


def test_noise_displace_uv_seam_duplicates_pass():
    """Coincident duplicates whose normals agree (UV seams) displace
    identically and pass (noise_displace semantics): the field is a
    function of position, so agreement in normals keeps them welded."""
    verts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],  # sheet A
            [1, 0, 0], [2, 0, 0], [1, 0, 1], [2, 0, 1],  # sheet B (seam copies)
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 2, 1], [1, 2, 3], [4, 6, 5], [5, 6, 7]], dtype=np.int32)
    normals = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (8, 1))
    m = noise_displace.evaluate(
        [Mesh(vertices=verts, faces=faces, normals=normals)],
        {"amplitude": 0.1, "seed": 3},
        None,
    )
    assert np.array_equal(m.vertices[1], m.vertices[4])
    assert np.array_equal(m.vertices[3], m.vertices[6])


def test_noise_displace_bounded_and_deterministic():
    """``noise_displace``: displacement along the normal peaks at about
    +-amplitude, and the field is a function of position and ``seed``
    only, so equal params reproduce and another seed decorrelates
    (DESIGN 4.4)."""
    g = grid.evaluate([], {"size": [4.0, 4.0], "segments": [32, 32]}, None)
    params = {"amplitude": 0.1, "frequency": 1.0, "seed": 7}
    m1 = noise_displace.evaluate([g], params, None)
    m2 = noise_displace.evaluate([g], params, None)
    assert m1.result_hash() == m2.result_hash()
    y = m1.vertices[:, 1]
    assert np.all(np.abs(y) <= 0.1 + 1e-6)
    assert y.std() > 0.005  # actually displaced
    assert not np.array_equal(
        m1.vertices,
        noise_displace.evaluate([g], {**params, "seed": 8}, None).vertices,
    )


# -- bevel ----------------------------------------------------------------


def test_bevel_chamfers_box():
    """Semantics: width is the distance cut back along each adjacent
    face, and corners resolve as the intersection of their edges'
    chamfer planes. On the unit box at width 0.1 that fixes every
    vertex: the 24 face vertices keep one coordinate at +-0.5 and sit at
    +-(0.5 - width) on the other two; the 8 apexes sit at
    +-(0.5 - width / 2) on all three. The removed volume is the twelve
    edge prisms less what the corner apexes give back."""
    b = box.evaluate([], {}, None)
    width = 0.1
    m = bevel.evaluate([b], {"width": width}, None)
    tm = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
    tm.merge_vertices()
    # Edge-plane chamfer: 3 face verts + 1 plane-intersection apex per
    # corner (see bevel docstring) -> 8 * 4 = 32 welded vertices.
    assert len(tm.vertices) == 32
    assert tm.is_watertight
    v = np.asarray(tm.vertices)
    on_face = np.isclose(np.abs(v), 0.5).sum(axis=1)
    assert np.bincount(on_face, minlength=2).tolist() == [8, 24]
    face_verts = np.sort(np.abs(v[on_face == 1]), axis=1)
    assert np.allclose(face_verts, [0.5 - width, 0.5 - width, 0.5], atol=1e-6)
    assert np.allclose(np.abs(v[on_face == 0]), 0.5 - width / 2, atol=1e-6)
    assert tm.volume == pytest.approx(0.946, rel=1e-3)
    s = m.stats()
    assert s["bounds"]["min"] == pytest.approx([-0.5, -0.5, -0.5])


def test_bevel_rejects_multi_segment():
    """``bevel``: ``segments > 1`` is refused as ``not_implemented`` until
    a multi-segment kernel lands behind the same signature (DESIGN 5)."""
    b = box.evaluate([], {}, None)
    with pytest.raises(OpError) as exc:
        bevel.evaluate([b], {"width": 0.05, "segments": 3}, None)
    assert exc.value.code == "not_implemented"


def test_bevel_no_sharp_edge_is_loud():
    """A bevel matching no edge is an actionable error, never a silent
    pass-through (bevel semantics): a wrong angle_min_deg must not read
    as success. The message carries the sharpest convex edge angle — the
    shortest path to the fix."""
    b = box.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        bevel.evaluate([b], {"width": 0.05, "angle_min_deg": 120.0}, None)
    assert e.value.code == "no_effect"
    assert "90.0" in e.value.message  # box edges meet at 90 deg


def test_bevel_nonconvex_broken_result_is_loud():
    """Post-conditions gate the result (bevel semantics): on non-convex
    input a global half-space trim slices unrelated geometry — the
    volume budget catches it and bevel fails actionably instead of
    emitting a silently broken mesh."""
    # Tall column + long slab in an L: the slab's front-top chamfer
    # plane, extended globally, shears off almost the whole column.
    column = transform.evaluate(
        [box.evaluate([], {"size": [0.3, 10.0, 0.3]}, None)],
        {"translate": [0.15, 5.0, 0.15]},
        None,
    )
    slab = transform.evaluate(
        [box.evaluate([], {"size": [3.0, 0.3, 0.3]}, None)],
        {"translate": [1.5, 0.15, 0.15]},
        None,
    )
    l_shape = boolean.evaluate([column, slab], {"mode": "union"}, None)
    with pytest.raises(OpError) as e:
        bevel.evaluate([l_shape], {"width": 0.03}, None)
    assert e.value.code == "broken_result"
    assert "convex" in e.value.message


# -- unwrap ---------------------------------------------------------------


def test_unwrap_box_projection():
    """``unwrap`` box mode: each face projects along the dominant axis of
    its normal onto the other two, so a unit box's UVs span
    [-0.5, 0.5], its already-split faces gain no vertices, and the
    geometry is untouched."""
    b = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    m = unwrap.evaluate([b], {"mode": "box"}, None)
    assert m.uv is not None and m.uv.shape == (len(m.vertices), 2)
    assert len(m.vertices) == 24  # box faces already split; no growth
    # +Z face reads (x, y): uv spans [-0.5, 0.5]^2.
    assert m.uv.min() == pytest.approx(-0.5)
    assert m.uv.max() == pytest.approx(0.5)
    assert volume(m) == pytest.approx(1.0, rel=1e-6)  # geometry untouched


def test_unwrap_projections_put_v_down_the_image():
    """A side face maps world +Y to smaller v, and the +Y projection maps
    world +Z to larger v (unwrap semantics): glTF puts (0, 0) at the
    image's top-left texel (DESIGN 4.3), so a map with an intrinsic up
    reads upright from outside instead of inverted."""
    b = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    m = unwrap.evaluate([b], {"mode": "box"}, None)
    face_z = np.abs(m.vertices[m.faces][..., 2].mean(axis=1) - 0.5) < 1e-6
    vi = np.unique(m.faces[face_z])
    assert np.allclose(m.uv[vi, 0], m.vertices[vi, 0])  # u right, un-mirrored
    assert np.allclose(m.uv[vi, 1], -m.vertices[vi, 1])  # v down

    s = sphere.evaluate([], {"segments": 16, "rings": 8}, None)
    top = unwrap.evaluate([s], {"mode": "planar", "axis": "y"}, None)
    assert np.allclose(top.uv[:, 1], top.vertices[:, 2])
    side = unwrap.evaluate([s], {"mode": "planar", "axis": "z"}, None)
    assert np.allclose(side.uv[:, 1], -side.vertices[:, 1])


def test_unwrap_planar_keeps_sharing():
    """``unwrap`` planar mode: every vertex projects along one axis with
    no splits, scaled by ``uv_scale`` UV units per meter."""
    s = sphere.evaluate([], {"segments": 16, "rings": 8}, None)
    m = unwrap.evaluate([s], {"mode": "planar", "axis": "y", "uv_scale": 2.0}, None)
    assert len(m.vertices) == len(s.vertices)
    assert m.uv is not None
    assert m.uv.max() == pytest.approx(1.0, abs=1e-5)  # radius 0.5 * scale 2


def test_noise_displace_adjacent_seeds_decorrelate():
    """The field decorrelates via ``seed`` (noise_displace semantics):
    with a stride-1 octave derivation, seed s octave o+1 and seed s+1
    octave o would sample the identical lattice field, leaving
    adjacent-seed surfaces mostly correlated."""
    g = grid.evaluate([], {"size": [4.0, 4.0], "segments": [32, 32]}, None)
    # frequency 4 -> a 16x16-cell field: enough independent lattice
    # samples that two decorrelated fields measure near-zero correlation,
    # while stride-1 sharing would measure ~0.75.
    params = {
        "amplitude": 0.1, "frequency": 4.0, "octaves": 4,
        "lacunarity": 1.0, "gain": 1.0,
    }
    d0 = noise_displace.evaluate([g], {**params, "seed": 0}, None).vertices[:, 1]
    d1 = noise_displace.evaluate([g], {**params, "seed": 1}, None).vertices[:, 1]
    corr = np.corrcoef(d0, d1)[0, 1]
    assert abs(corr) < 0.3


def test_noise_displace_lattice_overflow_is_loud():
    """Deterministic across runs and platforms (noise_displace
    semantics): params that would overflow the int64 lattice must fail
    actionably, not cast platform-dependently."""
    b = box.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        noise_displace.evaluate([b], {"octaves": 70}, None)
    assert e.value.code == "bad_param"


def test_bevel_knife_edge_is_loud():
    """A knife edge (adjacent faces meeting at ~180 deg) has no defined
    chamfer plane; bevel must fail actionably instead of cutting with a
    NaN/garbage plane that guts the solid."""
    # Flat tetrahedron: closed, combinatorially manifold (passes the
    # watertight gate), zero volume — every inner edge folds back 180 deg.
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0.3, 0, 0.3]], dtype=np.float32
    )
    faces = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]], dtype=np.int32)
    with pytest.raises(OpError) as e:
        bevel.evaluate([Mesh(vertices=verts, faces=faces)], {"width": 0.01}, None)
    assert e.value.code == "degenerate_edge"


# -- parameter domains (shared rules) ---------------------------------------


def test_vec3_scalar_only_for_scale_like_params():
    """Only scale-like params accept a scalar (transform / box / array
    semantics): a bare number for translate, rotate_deg, offset, or
    direction is rejected as bad_param naming the param, never broadcast
    into a wrong-but-valid evaluation."""
    b = box.evaluate([], {}, None)
    for op, params, name in (
        (transform, {"translate": 1}, "translate"),
        (transform, {"rotate_deg": 90}, "rotate_deg"),
        (array, {"step_translate": 1.0}, "step_translate"),
        (array, {"step_rotate_deg": 45}, "step_rotate_deg"),
    ):
        with pytest.raises(OpError) as e:
            op.evaluate([b], params, None)
        assert e.value.code == "bad_param" and e.value.detail["param"] == name
        assert "3 finite numbers" in e.value.message
    # extrude direction and place offset share the rule.
    g = grid.evaluate([], {}, None)
    with pytest.raises(OpError) as e:
        extrude.evaluate([g], {"direction": 1}, None)
    assert e.value.code == "bad_param" and e.value.detail["param"] == "direction"
    # The three scale-like params broadcast.
    cube = box.evaluate([], {"size": 2.0}, None)
    assert cube.stats()["bounds"]["max"] == pytest.approx([1.0, 1.0, 1.0])
    scaled = transform.evaluate([b], {"scale": 2}, None)
    assert scaled.stats()["bounds"]["max"] == pytest.approx([1.0, 1.0, 1.0])
    inst = array.evaluate([b], {"count": 2, "step_scale": 2}, None)
    assert np.allclose(np.diag(inst.transforms[1])[:3], [2.0, 2.0, 2.0])


def test_array_step_params_validated_under_their_own_names():
    """An array step-param error names the param the author wrote
    (step_translate / step_rotate_deg / step_scale), never the transform
    op's internal names."""
    b = box.evaluate([], {}, None)
    for params, name in (
        ({"step_translate": [1, 2]}, "step_translate"),
        ({"step_rotate_deg": [0, "x", 0]}, "step_rotate_deg"),
        ({"step_scale": [1, 0, 1]}, "step_scale"),
    ):
        with pytest.raises(OpError) as e:
            array.evaluate([b], params, None)
        assert e.value.code == "bad_param"
        assert e.value.detail["param"] == name
        assert f"'{name}'" in e.value.message


@pytest.mark.parametrize(
    "op,inputs,params,needle",
    [
        (cylinder, [], {"segments": 4097}, "<= 4096"),
        (sphere, [], {"segments": 4097}, "<= 4096"),
        (sphere, [], {"rings": 4097}, "<= 4096"),
        (grid, [], {"segments": [1, 4097]}, "1..4096"),
        (array, ["box"], {"count": 1_000_001}, "<= 1000000"),
        (scatter, ["box", "grid"], {"count": 1_000_001}, "<= 1000000"),
        (noise_displace, ["grid"], {"octaves": 11}, "<= 10"),
    ],
)
def test_count_params_have_documented_ceilings(op, inputs, params, needle):
    """Count-like integer params have a documented ceiling (each op's
    semantics ranges): beyond it the op refuses with bad_param naming the
    ceiling, instead of surfacing a raw allocation failure or unbounded
    time."""
    made = {
        "box": lambda: box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None),
        "grid": lambda: grid.evaluate([], {}, None),
    }
    with pytest.raises(OpError) as e:
        op.evaluate([made[k]() for k in inputs], params, OpContext("n01"))
    assert e.value.code == "bad_param"
    assert needle in e.value.message


def test_instanced_input_remedy_names_bake_where_baking_is_the_path():
    """A boolean cutter, a scatter target, or a placed part that arrives
    instanced is refused with the bake_instances remedy (the inputs that
    consume the whole instance set), while base/prototype inputs keep the
    apply-to-the-prototype advice."""
    from moraine3d.ops import place, tag

    b = box.evaluate([], {}, None)
    inst = array.evaluate([b], {"count": 2, "step_translate": [3.0, 0.0, 0.0]}, None)
    bake = "insert a bake_instances node between the array/scatter node and this input"
    with pytest.raises(OpError) as e:
        boolean.evaluate([b, inst], {"mode": "difference"}, None)
    assert e.value.code == "bad_input" and bake in e.value.message
    with pytest.raises(OpError) as e:
        scatter.evaluate([inst, b], {"count": 3}, OpContext("sc01"))
    assert e.value.code == "bad_input" and bake in e.value.message
    base = tag.evaluate([b], {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None)
    with pytest.raises(OpError) as e:
        place.evaluate([base, inst], {"anchor": "top"}, None)
    assert e.value.code == "bad_input" and bake in e.value.message
    with pytest.raises(OpError) as e:
        extrude.evaluate([inst], {}, None)
    assert "apply extrude to the prototype" in e.value.message


# -- bevel: rim facets ------------------------------------------------------


def test_bevel_cylinder_rim_with_side_edges_chamfered():
    """Golden cylinder rim (bevel semantics): with angle_min_deg below the
    facet angle 360/segments the side edges are chamfered too and the
    rim chamfers close — watertight, the expected face count, and the
    removed volume inside the chamfer budget."""
    c = cylinder.evaluate([], {"segments": 32}, None)
    width = 0.02
    m = bevel.evaluate([c], {"width": width, "angle_min_deg": 10.0}, None)
    assert len(m.faces) == 548
    assert watertight(m)
    vol_in = 0.5 * 32 * 0.5**2 * np.sin(2 * np.pi / 32) * 1.0
    removed = vol_in - volume(m)
    rim_len = 2 * 32 * 2 * 0.5 * np.sin(np.pi / 32)
    assert removed > width**2 / 2 * rim_len  # at least the two rim chamfers
    assert removed < 4.0 * width**2 * (rim_len + 32 * 1.0)  # the op's budget


def test_bevel_cylinder_rim_at_default_angle_chamfers_rims_only():
    """Golden cylinder rim (bevel semantics): at the default
    angle_min_deg a cylinder keeps its side edges and only the rims are
    chamfered — watertight, the expected face count, the bounds
    unchanged, and the removed volume that of the two rim wedges."""
    c = cylinder.evaluate([], {"segments": 32}, None)
    width = 0.02
    m = bevel.evaluate([c], {"width": width}, None)
    assert len(m.faces) == 334
    assert watertight(m)
    assert degenerate_tri_count(m.vertices, m.faces) == 0
    bounds = m.stats()["bounds"]
    assert bounds["min"] == pytest.approx([-0.5, -0.5, -0.5])
    assert bounds["max"] == pytest.approx([0.5, 0.5, 0.5])
    vol_in = 0.5 * 32 * 0.5**2 * np.sin(2 * np.pi / 32) * 1.0
    rim_len = 2 * 32 * 2 * 0.5 * np.sin(np.pi / 32)
    assert vol_in - volume(m) == pytest.approx(width**2 / 2 * rim_len, rel=0.05)


@pytest.mark.parametrize("width", [0.01, 0.015, 0.02, 0.03])
def test_bevel_chamfers_flat_box_at_any_small_width(width):
    """Exact for convex solids and small widths (bevel semantics): a
    box chamfers at every width below half its shortest edge. Where
    trims meet, edges of no length are collapsed, which moves no
    surface: the result is closed, holds no degenerate triangle, lies
    on the 6 faces and 12 chamfer planes, and removes the twelve edge
    prisms less what the corner apexes give back."""
    size = np.array([1.22, 0.44, 0.25])
    m = bevel.evaluate([box.evaluate([], {"size": size.tolist()}, None)], {"width": width}, None)
    assert watertight(m)
    assert degenerate_tri_count(m.vertices, m.faces) == 0
    normals = tri_cross(m.vertices[m.faces].astype(np.float64))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    assert len(np.unique(np.round(normals, 3), axis=0)) == 18
    removed = width**2 / 2 * 4 * size.sum() - 6 * width**3
    assert volume(m) == pytest.approx(size.prod() - removed, rel=1e-5)
    bounds = m.stats()["bounds"]
    assert bounds["min"] == pytest.approx((-size / 2).tolist())
    assert bounds["max"] == pytest.approx((size / 2).tolist())


# -- frozen result hashes ---------------------------------------------------


def test_scatter_result_hash_frozen():
    """Frozen at scatter op_impl_version 3 — a change here means bump
    (DESIGN 4.5): the node-salted streams (hash(seed, node_id, index))
    and the transform construction are pinned across runs by the result
    hash, not only in-process. Captured on one x86-64 platform; the output is IEEE float arithmetic
    only, so any conformant platform must reproduce it bit-for-bit — a
    mismatch is a semantics change or a determinism gap (DESIGN 4.4),
    never noise."""
    assert scatter.OP_IMPL_VERSION == 3
    proto = box.evaluate([], {"size": [0.1, 0.1, 0.1]}, None)
    target = grid.evaluate([], {"size": [2.0, 2.0]}, None)
    inst = scatter.evaluate([target, proto], {"count": 20, "seed": 1}, OpContext("sc01"))
    assert inst.result_hash() == (
        "af729edf598b7c8431ef656190ac41358d9c843cae6a8072e9cea8fd6dec9edc"
    )


def test_bevel_result_hash_frozen():
    """Frozen at bevel op_impl_version 4 — a change here means bump
    (DESIGN 4.5). Version 4 collapses edges of no length and returns a
    mesh that held none as it was, so these two, which passed the
    post-conditions under version 3, carry their hashes over. Captured on
    one x86-64 platform, and unlike the scatter hash it also pins the
    vertex and triangle order the manifold3d kernel emits: a mismatch
    on another platform or kernel release, with the golden bevel tests
    green, is that order alone, and is answered by freezing again."""
    assert bevel.OP_IMPL_VERSION == 4
    b = bevel.evaluate([box.evaluate([], {}, None)], {"width": 0.1}, None)
    assert b.result_hash() == (
        "30998d5cea535fa8558f5c53d574b16431ee1a970f6e7a0e40598abcdd5ecb09"
    )
    c = cylinder.evaluate([], {"segments": 32}, None)
    r = bevel.evaluate([c], {"width": 0.02, "angle_min_deg": 10.0}, None)
    assert r.result_hash() == (
        "ea481e4e35221edb7e6cf3ead33f9e49714ede976f04c1ecf9fb0058d00a0dfd"
    )


def test_noise_displace_result_hash_frozen():
    """Frozen at noise_displace op_impl_version 3 — a change here means
    bump (DESIGN 4.5): the value-noise field and the octave seed stride
    are pinned across runs by the result hash. Same platform caveat as
    the scatter hash: IEEE float arithmetic only, reproduced bit-for-bit
    on any conformant platform."""
    assert noise_displace.OP_IMPL_VERSION == 3
    g = grid.evaluate([], {"size": [4.0, 4.0], "segments": [32, 32]}, None)
    m = noise_displace.evaluate([g], {"amplitude": 0.1, "frequency": 1.0, "seed": 7}, None)
    assert m.result_hash() == (
        "b6ce47819c4b1350242f72e4d45b0913922c533a68fc141a40a4f3c0e1f5347d"
    )
