"""The mating slice: the ``mate`` op, the ``mated`` assertion, and the
per-copy quantifiers on the distance kinds (DESIGN 5.1, 5.2).

One relation, two sides: ``mate`` constructs a joint from two tagged
faces; ``mated`` declares it and checks it from evaluated world-space
geometry, never from the op's transform; ``every`` / ``self`` make the
distance guards over instanced outputs say what they check.
"""

import json

import numpy as np
import pytest

from moraine3d import ops
from moraine3d.app.cli import main
from moraine3d.core import mating
from moraine3d.core import asserts as vocab
from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import FACE_GROUP_NAME_RE, Instances, Mesh
from moraine3d.errors import OpError, AssetError
from moraine3d.ops import array as array_op
from moraine3d.ops import boolean as boolean_op
from moraine3d.ops import box as box_op
from moraine3d.ops import mate as mate_op
from moraine3d.ops import tag as tag_op
from moraine3d.ops import transform as transform_op
from moraine3d.verify.validate import check_asserts, validate_outputs
from support import flatten
from support.engine import evaluate_records


def _box(size):
    return box_op.evaluate([], {"size": size}, None)


def _moved(mesh, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)):
    return transform_op.evaluate(
        [mesh], {"translate": list(translate), "rotate_deg": list(rotate_deg)}, None
    )


def _tag(mesh, name, select):
    return tag_op.evaluate([mesh], {"name": name, "select": select}, None)


def _holed_wall():
    """A +Z-facing wall face with an off-center port: y 0..3, x -2..2,
    hole 0.5 x 0.7 centered at (0.8, 1.8), front face at z = 0.1."""
    wall = _moved(_box([4.0, 3.0, 0.2]), (0.0, 1.5, 0.0))
    port = _moved(_box([0.5, 0.7, 0.6]), (0.8, 1.8, 0.0))
    cut = boolean_op.evaluate([wall, port], {"mode": "difference"}, None)
    return _tag(cut, "port", {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}})


def _duct():
    """A rectangular tube along z (outer 0.6 x 0.8, bore 0.5 x 0.7, length
    1) whose -Z mouth ring is tagged ``dock``."""
    tube = boolean_op.evaluate(
        [_box([0.6, 0.8, 1.0]), _box([0.5, 0.7, 1.4])], {"mode": "difference"}, None
    )
    return _tag(tube, "dock", {"normal_z": {"lt": -0.9}, "center_z": {"lt": -0.49}})


def _frame_of(mesh, name):
    return mating.group_frame(mating.group_triangles(mesh, mesh.face_groups[name]))


def _two_heights():
    """Two unit boxes side by side, the second raised 0.5 m: their tops are
    coplanar patches at different heights."""
    lo = _box([1.0, 1.0, 1.0])
    hi = _moved(_box([1.0, 1.0, 1.0]), (2.0, 0.5, 0.0))
    return Mesh(
        vertices=np.concatenate([lo.vertices, hi.vertices]),
        faces=np.concatenate([lo.faces, hi.faces + len(lo.vertices)]),
    )


# -- mating frames (mating.py) ----------------------------------------------


def test_frame_origin_is_the_opening_not_the_area_centroid():
    """Origin = the centroid of the face set's single opening when it has
    one (DESIGN 5.1): a wall face with an off-center port registers on the
    port, where the area-weighted centroid would sit elsewhere."""
    f = _frame_of(_holed_wall(), "port")
    assert np.allclose(f.origin, [0.8, 1.8, 0.1], atol=1e-6)
    assert np.allclose(f.normal, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(f.secondary, [0.0, 1.0, 0.0], atol=1e-9)
    assert f.openings == 1
    # The duct's mouth ring — one opening too, the bore.
    d = _frame_of(_duct(), "dock")
    assert np.allclose(d.origin, [0.0, 0.0, -0.5], atol=1e-6)
    assert np.allclose(d.normal, [0.0, 0.0, -1.0], atol=1e-9)
    assert d.openings == 1


def test_frame_without_opening_uses_area_centroid_and_horizontal_fallback():
    """No opening → area-weighted centroid; a horizontal face takes +Z
    (asset front) as its secondary reference (DESIGN 5.1)."""
    top = _tag(_box([2.0, 1.0, 4.0]), "top", {"normal_y": {"gt": 0.9}})
    f = _frame_of(top, "top")
    assert np.allclose(f.origin, [0.0, 0.5, 0.0], atol=1e-9)
    assert np.allclose(f.secondary, [0.0, 0.0, 1.0], atol=1e-9)
    assert f.openings == 0
    side = _tag(_box([2.0, 1.0, 4.0]), "side", {"normal_x": {"gt": 0.9}})
    s = _frame_of(side, "side")
    assert np.allclose(s.secondary, [0.0, 1.0, 0.0], atol=1e-9)


def test_frame_refusals_name_the_defect():
    """Mixed-height, opposite-facing, multi-opening, and disconnected face
    sets have no frame — structured FrameError codes, never a guessed
    frame (DESIGN 5.1)."""
    # Two heights in one group.
    both = _two_heights()
    tops = _tag(both, "tops", {"normal_y": {"gt": 0.9}})
    with pytest.raises(mating.FrameError) as exc:
        _frame_of(tops, "tops")
    assert exc.value.code == "not_planar"
    # Opposite-facing faces cancel.
    ends = _tag(_box([1.0, 1.0, 1.0]), "ends", {"normal_x": {"gt": -2.0, "lt": 2.0}, "area": {"gt": 0.49}})
    with pytest.raises(mating.FrameError):
        _frame_of(ends, "ends")
    # Two openings in one face.
    wall = _moved(_box([4.0, 3.0, 0.2]), (0.0, 1.5, 0.0))
    h1 = _moved(_box([0.5, 0.5, 0.6]), (-1.0, 1.5, 0.0))
    h2 = _moved(_box([0.5, 0.5, 0.6]), (1.0, 1.5, 0.0))
    cut = boolean_op.evaluate([wall, h1, h2], {"mode": "difference"}, None)
    two = _tag(cut, "front", {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}})
    with pytest.raises(mating.FrameError) as exc:
        _frame_of(two, "front")
    assert exc.value.code == "ambiguous_opening"
    # Two coplanar pads in one tag.
    pads = _tag(both, "pads", {"normal_z": {"gt": 0.9}})
    with pytest.raises(mating.FrameError) as exc:
        _frame_of(pads, "pads")
    assert exc.value.code == "disconnected"


def test_frame_loops_touching_at_a_point_stay_distinct():
    """Loops are followed edge to edge and split at pinch vertices, so two
    openings or two patches that merely touch at a point stay two — never
    one merged loop whose origin lands on solid material (DESIGN 5.1)."""
    wall = _moved(_box([4.0, 4.0, 0.2]))
    c1 = _moved(_box([1.0, 1.0, 0.6]), (-0.5, -0.5, 0.0))
    c2 = _moved(_box([1.0, 1.0, 0.6]), (0.5, 0.5, 0.0))
    cut = boolean_op.evaluate([wall, c1, c2], {"mode": "difference"}, None)
    two = _tag(cut, "front", {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}})
    with pytest.raises(mating.FrameError) as exc:
        _frame_of(two, "front")
    assert exc.value.code == "ambiguous_opening"
    # Two coplanar pads sharing one corner vertex.
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tri = np.concatenate([
        np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]),
        np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]) + [1.0, 1.0, 0.0],
    ])
    with pytest.raises(mating.FrameError) as exc:
        mating.group_frame(tri)
    assert exc.value.code == "disconnected"


def test_frame_ignores_sliver_normals_and_defines_horizontal_by_angle():
    """Slivers below 1e-6 of the set's area carry no normal direction and
    are not tested for planarity; a face within 1 deg of level is
    horizontal (+Z secondary), beyond it the upright projection applies
    (DESIGN 5.1)."""
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    sliver = np.array([[1, 0, 0], [1 + 1e-9, 0.5, 0], [1, 1, 0]])  # reversed winding: -Z
    tri = np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]], sliver])
    f = mating.group_frame(tri)
    assert np.allclose(f.normal, [0.0, 0.0, 1.0], atol=1e-9)
    for deg, expect in ((0.0, [0, 0, 1]), (0.5, [0, 0, 1])):
        b = _moved(_box([2.0, 1.0, 2.0]), rotate_deg=(0.0, 0.0, deg))
        t = _tag(b, "top", {"normal_y": {"gt": 0.9}})
        assert np.allclose(_frame_of(t, "top").secondary, expect, atol=1e-9), deg
    b = _moved(_box([2.0, 1.0, 2.0]), rotate_deg=(0.0, 0.0, 2.0))
    t = _tag(b, "top", {"normal_y": {"gt": 0.9}})
    sec = _frame_of(t, "top").secondary
    assert abs(sec[2]) < 1e-9 and sec[0] > 0.99  # +Y projected into the tilted plane


def test_solve_reproduces_the_assembly_docking_numbers():
    """The frame solve: part normal → −anchor normal, secondary → secondary
    (upright), origin onto origin (DESIGN 5.1). A +Z port at z=3.4 with an
    off-center hole and a +Z tube end at z=9 give Ry(180) + [0, 0, 12.4];
    the same port facing +X gives Ry(−90) + [12.4, 0, 0]; horizontal mating
    is a pure translation; roll_deg turns about the anchor normal."""
    facet = _moved(_box([2.817, 3.0, 0.3]), (0.0, 1.5, 3.25))
    hole = _moved(_box([2.0, 2.4, 1.2]), (0.0, 1.35, 3.4))
    cut = boolean_op.evaluate([facet, hole], {"mode": "difference"}, None)
    port = _tag(cut, "port", {"normal_z": {"gt": 0.9}, "center_z": {"gt": 3.39}})
    fa = _frame_of(port, "port")
    assert np.allclose(fa.origin, [0.0, 1.35, 3.4], atol=1e-6)  # the hole, not y=1.5+
    outer = _moved(_box([2.3, 2.7, 2.0]), (0.0, 1.35, 8.0))
    bore = _moved(_box([2.0, 2.4, 2.4]), (0.0, 1.35, 8.0))
    tube = boolean_op.evaluate([outer, bore], {"mode": "difference"}, None)
    dock = _tag(tube, "dock", {"normal_z": {"gt": 0.9}, "center_z": {"gt": 8.99}})
    fp = _frame_of(dock, "dock")
    m = mating.solve(fp, fa)
    assert np.allclose(m[:3, :3], np.diag([-1.0, 1.0, -1.0]), atol=1e-9)
    assert np.allclose(m[:3, 3], [0.0, 0.0, 12.4], atol=1e-6)
    port_x = _tag(
        _moved(cut, rotate_deg=(0.0, 90.0, 0.0)),
        "port", {"normal_x": {"gt": 0.9}, "center_x": {"gt": 3.39}},
    )
    mx = mating.solve(fp, _frame_of(port_x, "port"))
    assert np.allclose(mx[:3, :3], [[0, 0, -1], [0, 1, 0], [1, 0, 0]], atol=1e-9)
    assert np.allclose(mx[:3, 3], [12.4, 0.0, 0.0], atol=1e-6)
    top = _frame_of(_tag(_box([2.0, 1.0, 2.0]), "top", {"normal_y": {"gt": 0.9}}), "top")
    bottom = _frame_of(
        _tag(_moved(_box([0.5, 0.5, 0.5]), (3.0, 3.0, 3.0)), "b", {"normal_y": {"lt": -0.9}}),
        "b",
    )
    mh = mating.solve(bottom, top)
    assert np.allclose(mh[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(mh[:3, 3], [-3.0, -2.25, -3.0], atol=1e-9)
    mr = mating.solve(bottom, top, 90.0)
    assert np.allclose(mr[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)


# -- the mate op --------------------------------------------------------------


def test_mate_contract_moves_the_face_onto_the_anchor():
    """mate: the output's own face group lies on the anchor plane with
    centroid = anchor origin and normal = −anchor normal; geometry,
    groups, and material ride the rigid motion (the reference's contract)."""
    wall, duct = _holed_wall(), _duct()
    out = mate_op.evaluate([wall, duct], {"anchor": "port", "face": "dock"}, None)
    assert isinstance(out, Mesh)
    assert "dock" in out.face_groups
    f = _frame_of(out, "dock")
    a = _frame_of(wall, "port")
    assert np.allclose(f.origin, a.origin, atol=1e-6)
    assert np.allclose(f.normal, -a.normal, atol=1e-9)
    lo, hi = out.bounds()
    assert np.allclose(lo, [0.5, 1.4, 0.1], atol=1e-6)
    assert np.allclose(hi, [1.1, 2.2, 1.1], atol=1e-6)
    assert np.array_equal(out.faces, duct.faces)


def test_mate_rotates_attributes_and_keeps_identity():
    """A rigid motion: normals and TANGENT rotate as directions, UVs and
    material are untouched, face groups are kept (DESIGN 5.1)."""
    from dataclasses import replace

    from moraine3d.core.mesh import Material

    wall = _holed_wall()
    port_x = _tag(
        _moved(boolean_op.evaluate(
            [_moved(_box([4.0, 3.0, 0.2]), (0.0, 1.5, 0.0)),
             _moved(_box([0.5, 0.7, 0.6]), (0.8, 1.8, 0.0))],
            {"mode": "difference"}, None,
        ), rotate_deg=(0.0, 90.0, 0.0)),
        "port", {"normal_x": {"gt": 0.9}, "center_x": {"gt": 0.09}},
    )
    duct = _duct()
    n = len(duct.vertices)
    tangent = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    duct = replace(
        duct,
        normals=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (n, 1)),
        uv=np.zeros((n, 2), dtype=np.float32),
        attributes={"TANGENT": tangent},
        material=Material(name="duct_paint"),
        face_groups=dict(duct.face_groups),
    )
    out = mate_op.evaluate([port_x, duct], {"anchor": "port", "face": "dock"}, None)
    # -Z mouth now faces -X (opposing the +X anchor): normals rotated.
    assert np.allclose(out.normals, [-1.0, 0.0, 0.0], atol=1e-6)
    # TANGENT +X rotates with the part (Ry(+90) maps +X to -Z), w kept.
    assert np.allclose(out.attributes["TANGENT"][:, :3], [0.0, 0.0, -1.0], atol=1e-6)
    assert np.allclose(out.attributes["TANGENT"][:, 3], 1.0)
    assert np.array_equal(out.uv, duct.uv)
    assert out.material is duct.material
    assert set(out.face_groups) == {"dock"}
    assert wall is not out


def test_mate_instanced_part_moves_as_a_set():
    """An instanced part moves rigidly — transforms pre-multiplied,
    prototype untouched (instancing stays instancing); ``face_instance``
    picks the copy whose tagged face mates (DESIGN 5.1)."""
    wall = _holed_wall()
    row = array_op.evaluate(
        [_duct()], {"count": 3, "step_translate": [0.0, 0.0, 1.0]}, None
    )
    with pytest.raises(OpError) as exc:
        mate_op.evaluate([wall, row], {"anchor": "port", "face": "dock"}, None)
    assert exc.value.code == "missing_face_instance"
    assert "0..2" in exc.value.message
    out = mate_op.evaluate(
        [wall, row], {"anchor": "port", "face": "dock", "face_instance": 2}, None
    )
    assert isinstance(out, Instances)
    assert out.prototype is row.prototype
    # Copy 2's mouth (z = 1.5 in the row) lands on the port plane z = 0.1.
    f = mating.group_frame(
        mating.group_triangles(out.prototype, out.prototype.face_groups["dock"], out.transforms[2])
    )
    assert np.allclose(f.origin, [0.8, 1.8, 0.1], atol=1e-6)
    lo, _ = out.bounds()
    assert abs(float(lo[2]) - (0.1 - 2.0)) < 1e-6  # copies 0 and 1 sit behind the wall


def test_tag_on_instanced_input_tags_the_prototype():
    """tag on an instanced input tags the prototype and passes the copies
    through — a consumer tags a received instanced output at the point of
    use, then mates one copy's face (DESIGN 5.1)."""
    tube = boolean_op.evaluate(
        [_box([0.6, 0.8, 1.0]), _box([0.5, 0.7, 1.4])], {"mode": "difference"}, None
    )
    row = array_op.evaluate([tube], {"count": 3, "step_translate": [0.0, 0.0, 1.0]}, None)
    tagged = tag_op.evaluate(
        [row], {"name": "dock", "select": {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.49}}}, None
    )
    assert isinstance(tagged, Instances)
    assert "dock" in tagged.prototype.face_groups
    assert tagged.transforms is row.transforms
    assert tagged.prototype.face_groups["dock"].size == 8
    out = mate_op.evaluate(
        [_holed_wall(), tagged], {"anchor": "port", "face": "dock", "face_instance": 2}, None
    )
    f = mating.group_frame(mating.group_triangles(
        out.prototype, out.prototype.face_groups["dock"], out.transforms[2]))
    assert np.allclose(f.origin, [0.8, 1.8, 0.1], atol=1e-6)
    assert np.allclose(f.normal, [0.0, 0.0, -1.0], atol=1e-9)


def test_mate_rider_follows_the_leader():
    """A third input is the leader: the part receives the rigid motion that
    mates the leader's tagged face, for Mesh and Instances parts alike
    (DESIGN 5.1)."""
    wall, duct = _holed_wall(), _duct()
    flange = _moved(_box([0.8, 1.0, 0.04]), (0.0, 0.0, -0.48))
    out = mate_op.evaluate([wall, flange, duct], {"anchor": "port", "face": "dock"}, None)
    lo, hi = out.bounds()
    assert np.allclose(lo, [0.4, 1.3, 0.1], atol=1e-6)
    assert np.allclose(hi, [1.2, 2.3, 0.14], atol=1e-6)
    ring = array_op.evaluate([flange], {"count": 2, "step_translate": [0.0, 0.0, 0.5]}, None)
    outi = mate_op.evaluate([wall, ring, duct], {"anchor": "port", "face": "dock"}, None)
    assert isinstance(outi, Instances)
    lo, hi = outi.bounds()
    assert abs(float(lo[2]) - 0.1) < 1e-6 and abs(float(hi[2]) - 0.64) < 1e-6


def test_mate_instanced_anchor_and_roll():
    """``instance`` selects the anchor copy (world-space frame, required
    for an instanced base); ``roll_deg`` turns the part about the mated
    normal (DESIGN 5.1)."""
    walls = array_op.evaluate(
        [_holed_wall()], {"count": 2, "step_translate": [0.0, 0.0, -5.0]}, None
    )
    with pytest.raises(OpError) as exc:
        mate_op.evaluate([walls, _duct()], {"anchor": "port", "face": "dock"}, None)
    assert exc.value.code == "missing_instance"
    out = mate_op.evaluate(
        [walls, _duct()], {"anchor": "port", "face": "dock", "instance": 1}, None
    )
    lo, hi = out.bounds()
    assert abs(float(lo[2]) - (0.1 - 5.0)) < 1e-6
    rolled = mate_op.evaluate(
        [_holed_wall(), _duct()],
        {"anchor": "port", "face": "dock", "roll_deg": 90.0}, None,
    )
    lo, hi = rolled.bounds()
    # 0.6 x 0.8 section rolled a quarter turn about z: 0.8 wide, 0.6 tall.
    assert abs(float(hi[0] - lo[0]) - 0.8) < 1e-6
    assert abs(float(hi[1] - lo[1]) - 0.6) < 1e-6


@pytest.mark.parametrize(
    "inputs, params, code, needle",
    [
        (lambda: [_box([1, 1, 1]), _duct()], {"anchor": "port", "face": "dock"},
         "unknown_anchor", "carries no tags"),
        (lambda: [_holed_wall(), _duct()], {"anchor": "roof", "face": "dock"},
         "unknown_anchor", "port"),
        (lambda: [_holed_wall(), _duct()], {"anchor": "port", "face": "mouth"},
         "unknown_face", "dock"),
        (lambda: [_holed_wall(), _duct()], {"anchor": "port", "face": "dock", "instance": 0},
         "bad_param", "drop the copy index"),
        (lambda: [_holed_wall(), _duct()], {"anchor": "port", "face": "dock", "mode": "flush"},
         "bad_param", "coincident"),
        (lambda: [_holed_wall(), _duct()], {"anchor": "port", "face": "dock", "roll_deg": "x"},
         "bad_param", "finite"),
        (lambda: [_holed_wall(), _tag(_box([1, 1, 1]), "dock", {"normal_y": {"gt": -2}})],
         {"anchor": "port", "face": "dock"}, "face_degenerate", "one facing side"),
        (lambda: [_tag(_two_heights(), "port", {"normal_y": {"gt": 0.9}}), _duct()],
         {"anchor": "port", "face": "dock"}, "anchor_not_planar", "split the tag"),
    ],
)
def test_mate_errors_are_structured_and_name_a_fix(inputs, params, code, needle):
    """Every mate refusal is a structured code whose message names a
    terminating remedy — first-class batch scope, since a union-consumed
    joint has no assert behind it (DESIGN 5.1)."""
    with pytest.raises(OpError) as exc:
        mate_op.evaluate(inputs(), params, None)
    assert exc.value.code == code
    assert needle in exc.value.message


def test_face_group_resolver_is_shared():
    """place, mate, and mated address a tagged face the same way — one
    resolver (predicates.resolve_face_group): tags ride the prototype, a
    copy index is required on an instanced value and refused on a mesh,
    unknown tags list what is available (DESIGN 5.1)."""
    from moraine3d.core.predicates import FaceGroupError, resolve_face_group

    wall = _holed_wall()
    tf = resolve_face_group(wall, "port", None, subject="x")
    assert tf.matrix is None and tf.copies is None and len(tf.faces) == 8
    row = array_op.evaluate([wall], {"count": 2, "step_translate": [0.0, 0.0, 1.0]}, None)
    tf = resolve_face_group(row, "port", 1, subject="x")
    assert tf.copies == 2 and np.allclose(tf.matrix[2, 3], 1.0)
    for value, name, inst, code, needle in [
        (wall, "roof", None, "unknown_group", "port"),
        (_box([1, 1, 1]), "roof", None, "unknown_group", "carries no tags"),
        (row, "port", None, "missing_instance", "0..1"),
        (row, "port", 5, "bad_instance", "out of range"),
        (row, "port", True, "bad_instance", "integer"),
        (wall, "port", 0, "bad_instance", "drop the copy index"),
    ]:
        with pytest.raises(FaceGroupError) as exc:
            resolve_face_group(value, name, inst, subject="x")
        assert exc.value.code == code and needle in exc.value.message


def test_mate_traits_and_arity():
    """mate declares its contract traits: arity 2-3, carried input 1 only,
    params frozen (ops registry contract)."""
    mod = ops.get("mate")
    assert mod.INPUT_ARITY == (2, 3)
    assert mod.CARRIED_INPUTS == (1,)
    assert mod.OP_IMPL_VERSION == 1
    with pytest.raises(OpError):
        ops.check_arity("mate", 1)
    with pytest.raises(OpError):
        ops.check_arity("mate", 4)


# -- the mated assertion ---------------------------------------------------


def _asset(nodes: list[dict], outputs: dict, asserts: list[dict], checks=None) -> str:
    doc = {"build_version": "fixture", "nodes": nodes, "outputs": outputs}
    if checks:
        doc["checks"] = checks
    doc["asserts"] = asserts
    return json.dumps(doc)


_WALL_NODES = [
    {"id": "wl01", "op": "box", "params": {"size": [4.0, 3.0, 0.2]}},
    {"id": "wl02", "op": "transform", "inputs": ["wl01"], "params": {"translate": [0.0, 1.5, 0.0]}},
    {"id": "pt01", "op": "box", "params": {"size": [0.5, 0.7, 0.6]}},
    {"id": "pt02", "op": "transform", "inputs": ["pt01"], "params": {"translate": [0.8, 1.8, 0.0]}},
    {"id": "wl03", "op": "boolean", "inputs": ["wl02", "pt02"], "params": {"mode": "difference"}},
    {"id": "wl04", "op": "tag", "inputs": ["wl03"],
     "params": {"name": "port", "select": {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}}}},
    {"id": "dt01", "op": "box", "params": {"size": [0.6, 0.8, 1.0]}},
    {"id": "dt02", "op": "box", "params": {"size": [0.5, 0.7, 1.4]}},
    {"id": "dt03", "op": "boolean", "inputs": ["dt01", "dt02"], "params": {"mode": "difference"}},
    {"id": "dt04", "op": "tag", "inputs": ["dt03"],
     "params": {"name": "dock", "select": {"normal_z": {"lt": -0.9}, "center_z": {"lt": -0.49}}}},
]
_MATE_NODE = {"id": "dt05", "op": "mate", "inputs": ["wl04", "dt04"],
              "params": {"anchor": "port", "face": "dock"}}
# The built-in overlap guard covers interpenetration; contact is the
# authored other half of the resting-contact idiom.
_GUARDS = [
    {"id": "duct_meets_wall", "kind": "contact", "outputs": ["wall", "duct"], "owner": "agent"},
]
_MATED = {"id": "duct_mated", "kind": "mated", "faces": [["wall", "port"], ["duct", "dock"]],
          "owner": "human"}


def _duct_asset(placer: dict, asserts: list[dict]) -> str:
    return _asset(
        _WALL_NODES + [placer], {"wall": "wl04", "duct": "dt05"}, asserts,
    )


def _run_asserts(asset_json: str) -> dict:
    sc = asset_mod.loads(asset_json)
    recs = evaluate_records(sc)
    return validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts,
    )


def test_mated_passes_a_mated_joint_and_reports_the_numbers(tmp_path, monkeypatch):
    """mated reads evaluated world-space faces: a joint built by mate
    passes both stages and the entry carries angle/offset/distance even
    when green (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_duct_asset(_MATE_NODE, [_MATED] + _GUARDS))
    assert report["ok"] is True
    entry = next(e for e in report["asserts"] if e["assert"] == "duct_mated")
    assert entry["kind"] == "mated"
    assert entry["faces"] == [["wall", "port"], ["duct", "dock"]]
    assert entry["mode"] == "coincident"  # the default, echoed like owner
    assert entry["ok"] is True
    assert entry["detail"]["angle_deg"] < 1e-9
    assert entry["detail"]["distance"] < 1e-6
    assert "fix" not in entry


def test_mated_catches_the_slid_duct_that_contact_and_no_overlap_pass(tmp_path, monkeypatch):
    """The field repro (DESIGN 5.2): a duct slid 0.3 m across its port still
    touches (contact 0) and does not interpenetrate (no_overlap 0), but its
    mating origins are 0.3 m apart — mated fails at stage (b) with the
    offset, and the human-owned fix never steers toward the tol."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    slid = {"id": "dt05", "op": "transform", "inputs": ["dt04"],
            "params": {"translate": [1.1, 1.8, 0.6]}}
    report = _run_asserts(_duct_asset(slid, [_MATED] + _GUARDS))
    by_id = {e["assert"]: e for e in report["asserts"]}
    assert by_id["no_overlap"]["ok"] is True  # the built-in guard
    assert by_id["duct_meets_wall"]["ok"] is True
    m = by_id["duct_mated"]
    assert m["ok"] is False
    assert abs(m["detail"]["distance"] - 0.3) < 1e-5
    assert abs(m["detail"]["offset"][0] - 0.3) < 1e-5
    assert "0.3 m apart" in m["fix"]
    assert "fix the geometry, not the tol" in m["fix"]
    assert "raise" not in m["fix"]
    assert report["ok"] is False


def test_mated_tilt_fails_stage_a_and_agent_owner_may_weaken(tmp_path, monkeypatch):
    """Stage (a): normals not anti-parallel within angle_tol_deg fail with
    the measured angle; an agent-owned entry's fix may mention raising the
    tolerance (DESIGN 5.2 ownership)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    tilted = {"id": "dt05", "op": "transform", "inputs": ["dt04"],
              "params": {"translate": [0.8, 1.8, 0.6], "rotate_deg": [5.0, 0.0, 0.0]}}
    mated = dict(_MATED, owner="agent")
    report = _run_asserts(_duct_asset(tilted, [mated]))
    m = next(e for e in report["asserts"] if e["assert"] == "duct_mated")
    assert m["ok"] is False
    assert abs(m["detail"]["angle_deg"] - 5.0) < 1e-4
    assert "5 deg from anti-parallel" in m["fix"]
    assert "raise angle_tol_deg" in m["fix"]
    # A generous explicit tolerance accepts the tilt.
    loose = dict(mated, angle_tol_deg=10.0, tol=0.2)
    report = _run_asserts(_duct_asset(tilted, [loose]))
    assert next(e for e in report["asserts"]
                if e["assert"] == "duct_mated")["ok"] is True


def test_mated_unevaluatable_reasons(tmp_path, monkeypatch):
    """A face without a frame fails with the reason, never ok and never a
    crash: missing tag (lists available), instanced output without a copy
    index, out-of-range index, index on a plain mesh (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    row = {"id": "dt06", "op": "array", "inputs": ["dt05"],
           "params": {"count": 2, "step_translate": [0.0, 0.0, 2.0]}}
    asset = _asset(
        _WALL_NODES + [_MATE_NODE, row], {"wall": "wl04", "ducts": "dt06"},
        [
            {"id": "a", "kind": "mated", "faces": [["wall", "roof"], ["ducts", "dock", 0]], "owner": "agent"},
            {"id": "b", "kind": "mated", "faces": [["wall", "port"], ["ducts", "dock"]], "owner": "agent"},
            {"id": "c", "kind": "mated", "faces": [["wall", "port"], ["ducts", "dock", 5]], "owner": "agent"},
            {"id": "d", "kind": "mated", "faces": [["wall", "port", 0], ["ducts", "dock", 0]], "owner": "agent"},
            {"id": "e", "kind": "mated", "faces": [["wall", "port"], ["ducts", "dock", 0]], "owner": "agent"},
            {"id": "f", "kind": "mated", "faces": [["wall", "port"], ["ducts", "dock", 1]], "owner": "agent"},
        ],
    )
    by_id = {e["assert"]: e for e in _run_asserts(asset)["asserts"]}
    assert by_id["a"]["ok"] is False and "no face group 'roof'" in by_id["a"]["fix"]
    assert "port" in by_id["a"]["fix"]
    assert by_id["b"]["ok"] is False and "add the copy index" in by_id["b"]["fix"]
    assert by_id["c"]["ok"] is False and "out of range" in by_id["c"]["fix"]
    assert by_id["d"]["ok"] is False and "drop the copy index" in by_id["d"]["fix"]
    assert by_id["e"]["ok"] is True  # copy 0 is the mated duct
    assert by_id["f"]["ok"] is False and abs(by_id["f"]["detail"]["distance"] - 2.0) < 1e-5


# -- mated mode "contained" -------------------------------------------------


_PLATE_NODES = [
    {"id": "bg01", "op": "box", "params": {"size": [2.5, 2.9, 0.12]}},
    {"id": "bg02", "op": "transform", "inputs": ["bg01"], "params": {"translate": [0.0, 1.45, 0.0]}},
    {"id": "bg03", "op": "tag", "inputs": ["bg02"],
     "params": {"name": "front", "select": {"normal_z": {"gt": 0.9}}}},
    {"id": "sm01", "op": "box", "params": {"size": [0.8, 0.8, 0.1]}},
    {"id": "sm02", "op": "transform", "inputs": ["sm01"], "params": {"translate": [0.4, 1.0, 0.11]}},
    {"id": "sm03", "op": "tag", "inputs": ["sm02"],
     "params": {"name": "back", "select": {"normal_z": {"lt": -0.9}}}},
]
_PLATE_OUTS = {"big": "bg03", "small": "sm03"}


def _contained(faces, **kw):
    kw.setdefault("owner", "agent")
    return {"id": "seated", "kind": "mated", "mode": "contained", "faces": faces, **kw}


def test_mated_contained_passes_the_offset_butt_joint_coincident_fails(tmp_path, monkeypatch):
    """The contained mode covers the joint coincidence cannot (DESIGN 5.2):
    a smaller part butted against a larger blank face, whose face
    centroids legitimately differ — coincident fails the correct geometry
    with the centroid offset, contained passes it."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    faces = [["big", "front"], ["small", "back"]]
    report = _run_asserts(_asset(
        _PLATE_NODES, _PLATE_OUTS,
        [_contained(faces),
         {"id": "co", "kind": "mated", "faces": faces, "owner": "agent"}],
    ))
    by_id = {e["assert"]: e for e in report["asserts"]}
    seated = by_id["seated"]
    assert seated["mode"] == "contained"  # the weaker registration is visible
    assert seated["ok"] is True
    assert seated["detail"]["plane_gap"] < 1e-6
    assert seated["detail"]["outside"] == 0.0
    assert seated["detail"]["crossings"] == 0
    co = by_id["co"]
    assert co["ok"] is False and co["detail"]["distance"] > 0.5


def test_mated_contained_fails_beyond_the_rim_and_on_a_plane_gap(tmp_path, monkeypatch):
    """Contained failures carry the measured numbers (DESIGN 5.2): a part
    overhanging the anchor fails with the rim excursion; a lifted part
    fails with the gap along the mating normal."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    faces = [["big", "front"], ["small", "back"]]
    over = [dict(n, params={"translate": [1.3, 1.0, 0.11]}) if n["id"] == "sm02" else n
            for n in _PLATE_NODES]
    report = _run_asserts(_asset(over, _PLATE_OUTS, [_contained(faces)]))
    entry = next(e for e in report["asserts"] if e["assert"] == "seated")
    assert entry["ok"] is False
    assert abs(entry["detail"]["outside"] - 0.45) < 1e-6
    assert "leaves the anchor contour" in entry["fix"]
    assert "raise angle_tol_deg" in entry["fix"]  # agent-owned may weaken
    lifted = [dict(n, params={"translate": [0.4, 1.0, 0.12]}) if n["id"] == "sm02" else n
              for n in _PLATE_NODES]
    report = _run_asserts(
        _asset(lifted, _PLATE_OUTS, [_contained(faces, owner="human")])
    )
    entry = next(e for e in report["asserts"] if e["assert"] == "seated")
    assert entry["ok"] is False
    assert abs(entry["detail"]["plane_gap"] - 0.01) < 1e-6
    assert "along the mating normal" in entry["fix"]
    assert "fix the geometry, not the tol" in entry["fix"]


def test_mated_contained_ignores_anchor_holes_and_order_carries_roles(tmp_path, monkeypatch):
    """Containment is against the anchor's outer extent — holes are
    ignored, so a blank plate may cover a bore — and faces[0] is the
    containing face: the reversed entry fails (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    nodes = _WALL_NODES + [
        {"id": "pl01", "op": "box", "params": {"size": [1.0, 1.2, 0.1]}},
        {"id": "pl02", "op": "transform", "inputs": ["pl01"], "params": {"translate": [0.8, 1.8, 0.15]}},
        {"id": "pl03", "op": "tag", "inputs": ["pl02"],
         "params": {"name": "back", "select": {"normal_z": {"lt": -0.9}}}},
    ]
    outs = {"wall": "wl04", "plate": "pl03"}
    report = _run_asserts(_asset(
        nodes, outs,
        [_contained([["wall", "port"], ["plate", "back"]]),
         dict(_contained([["plate", "back"], ["wall", "port"]]), id="reversed")],
    ))
    by_id = {e["assert"]: e for e in report["asserts"]}
    assert by_id["seated"]["ok"] is True
    assert by_id["reversed"]["ok"] is False
    assert by_id["reversed"]["detail"]["outside"] > 1.0


def test_mated_contained_crossing_catches_the_spanned_notch(tmp_path, monkeypatch):
    """A part region can leave a non-convex anchor between two interior
    vertices; the boundary-crossing test catches it where the vertex
    distances alone read 0 (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    nodes = [
        {"id": "wu01", "op": "box", "params": {"size": [4.0, 3.0, 0.2]}},
        {"id": "wu02", "op": "transform", "inputs": ["wu01"], "params": {"translate": [0.0, 1.5, 0.0]}},
        {"id": "nc01", "op": "box", "params": {"size": [1.0, 2.0, 0.6]}},
        {"id": "nc02", "op": "transform", "inputs": ["nc01"], "params": {"translate": [0.0, 2.5, 0.0]}},
        {"id": "wu03", "op": "boolean", "inputs": ["wu02", "nc02"], "params": {"mode": "difference"}},
        {"id": "wu04", "op": "tag", "inputs": ["wu03"],
         "params": {"name": "front", "select": {"normal_z": {"gt": 0.9}, "center_z": {"gt": 0.09}}}},
        {"id": "br01", "op": "box", "params": {"size": [3.0, 0.5, 0.1]}},
        {"id": "br02", "op": "transform", "inputs": ["br01"], "params": {"translate": [0.0, 2.0, 0.15]}},
        {"id": "br03", "op": "tag", "inputs": ["br02"],
         "params": {"name": "back", "select": {"normal_z": {"lt": -0.9}}}},
    ]
    report = _run_asserts(_asset(
        nodes, {"wall": "wu04", "bridge": "br03"},
        [_contained([["wall", "front"], ["bridge", "back"]])],
    ))
    entry = next(e for e in report["asserts"] if e["assert"] == "seated")
    assert entry["ok"] is False
    assert entry["detail"]["outside"] == 0.0  # every vertex is inside
    assert entry["detail"]["crossings"] == 2  # top and bottom edges span the notch
    assert "cross the anchor contour" in entry["fix"]


def test_mated_contained_rim_seat_within_tol_passes(tmp_path, monkeypatch):
    """The rim allowance holds for edges as it does for vertices
    (DESIGN 5.2: a part edge seated flush with the anchor's edge —
    collinear within `tol` — is containment, not a crossing): a part
    seated 1e-7 m below the anchor's bottom rim, inside the default
    tol but beyond collinear float noise, passes with zero crossings."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    nodes = [
        {"id": "an01", "op": "box", "params": {"size": [2.8167, 3.0, 0.12]}},
        {"id": "an02", "op": "transform", "inputs": ["an01"], "params": {"translate": [0.0, 1.5, 0.0]}},
        {"id": "an03", "op": "tag", "inputs": ["an02"],
         "params": {"name": "front", "select": {"normal_z": {"gt": 0.9}}}},
        {"id": "pt01", "op": "box", "params": {"size": [2.3, 2.7, 0.1]}},
        {"id": "pt02", "op": "transform", "inputs": ["pt01"], "params": {"translate": [0.0, 1.35 - 1e-7, 0.11]}},
        {"id": "pt03", "op": "tag", "inputs": ["pt02"],
         "params": {"name": "back", "select": {"normal_z": {"lt": -0.9}}}},
    ]
    report = _run_asserts(_asset(
        nodes, {"anchor": "an03", "part": "pt03"},
        [_contained([["anchor", "front"], ["part", "back"]])],
    ))
    entry = next(e for e in report["asserts"] if e["assert"] == "seated")
    assert entry["ok"] is True
    assert entry["detail"]["crossings"] == 0
    assert entry["detail"]["outside"] <= 1e-6


def test_mated_contained_flush_rim_is_contact_not_crossing():
    """A part boundary flush with the anchor contour rim is containment,
    not a crossing — grazing and collinear overlap are excluded from the
    strict-crossing count under arbitrary rigid placement, where the
    collinear orientation determinants are float noise (DESIGN 5.2)."""
    from moraine3d.core import mating

    def quad(x0, z0, x1, z1, flip=False):
        # CCW seen from +Y; flip for the mated face looking back at it.
        p00, p10 = [x0, 0.0, z0], [x1, 0.0, z0]
        p11, p01 = [x1, 0.0, z1], [x0, 0.0, z1]
        tris = np.array([[p00, p11, p10], [p00, p01, p11]], dtype=np.float64)
        return tris[:, ::-1, :] if flip else tris

    anchor = quad(0.0, 0.0, 2.0, 2.0)
    flush = quad(0.0, 0.0, 1.0, 1.0, flip=True)  # two edges on the rim
    rng = np.random.default_rng(12345)
    for _ in range(50):
        v = rng.normal(size=3)
        rot = mating._rotation_about(v / np.linalg.norm(v), rng.uniform(0, 360))
        t = rng.uniform(-5, 5, size=3)
        a, p = anchor @ rot.T + t, flush @ rot.T + t
        frame = mating.group_frame(a)
        outside, crossings = mating.contained_excursion(a, p, frame)
        assert crossings == 0
        assert outside <= 1e-4
    # The strict-crossing test must still catch a real excursion.
    over = quad(1.5, 0.5, 2.5, 1.5, flip=True)  # straddles the rim
    frame = mating.group_frame(anchor)
    outside, crossings = mating.contained_excursion(anchor, over, frame)
    assert crossings > 0 and outside > 0.4


def test_mated_parse_rejections_and_round_trip(tmp_path):
    """Parse-time shape of mated (core/asserts): faces required and well
    formed, outputs/self/every rejected, mated keys rejected elsewhere;
    entries round-trip through dumps in KEY_ORDER."""
    base = json.loads(_duct_asset(_MATE_NODE, [_MATED]))

    def with_assert(**kw):
        doc = json.loads(json.dumps(base))
        doc["asserts"] = [{"id": "x", "kind": "mated", **kw}]
        return json.dumps(doc)

    cases = [
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], outputs=["wall", "duct"]), "bad_assert"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], self=True), "bad_assert"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], every=["duct"]), "bad_assert"),
        (with_assert(), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["duct", "1bad"]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock", -1]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock", True]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["ghost", "dock"]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["wall", "port"]]), "bad_assert_faces"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], mode="equal"), "bad_assert"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], angle_tol_deg=-1), "bad_assert"),
        (with_assert(faces=[["wall", "port"], ["duct", "dock"]], tol=-1), "bad_assert"),
    ]
    for text, code in cases:
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(text)
        assert exc.value.code == code, text
    # mated-only keys on other kinds.
    for key, val in (("faces", [["wall", "port"], ["duct", "dock"]]), ("mode", "coincident"),
                     ("angle_tol_deg", 1)):
        doc = json.loads(json.dumps(base))
        doc["asserts"] = [{"id": "x", "kind": "contact", "outputs": ["wall", "duct"], key: val}]
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(json.dumps(doc))
        assert exc.value.code == "bad_assert"
    # Round trip keeps the entry and the key order.
    doc = json.loads(json.dumps(base))
    doc["asserts"] = [{"owner": "human", "angle_tol_deg": 0.5, "tol": 0.002, "mode": "coincident",
                       "faces": [["wall", "port"], ["duct", "dock", 0]], "kind": "mated", "id": "rt"}]
    sc = asset_mod.loads(json.dumps(doc))
    line = [ln for ln in asset_mod.dumps(sc).splitlines() if '"rt"' in ln][0]
    assert line.index('"kind"') < line.index('"faces"') < line.index('"mode"')
    assert line.index('"mode"') < line.index('"tol"') < line.index('"angle_tol_deg"') < line.index('"owner"')
    assert asset_mod.loads(asset_mod.dumps(sc)).asserts == sc.asserts
    assert vocab.output_names(sc.asserts[0]) == {"wall", "duct"}
    assert vocab.output_names({"kind": "contact", "outputs": ["a", "b"]}) == {"a", "b"}
    assert vocab.output_names({"kind": "no_overlap", "outputs": "all"}) == "all"
    assert vocab.common.FACE_TAG_RE.pattern == FACE_GROUP_NAME_RE.pattern
    assert vocab.mated.DEFAULT_TOL == 1e-5
    assert vocab.mated.DEFAULT_ANGLE_TOL_DEG == 0.01


def test_mated_in_a_sub_asset_gates_the_consumer(tmp_path, monkeypatch):
    """A mated entry authored inside a sub-asset gates the consuming root's
    validate like any sub-asset assert (DESIGN 4.6) — the evaluator
    resolves the faces' outputs, not an `outputs` key."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "proj"
    (project / "subassets").mkdir(parents=True)
    (project / "subassets" / "joint.mora").write_text(
        _duct_asset(_MATE_NODE, [_MATED]), encoding="utf-8"
    )
    root = json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "r1", "op": "asset_ref",
                   "params": {"asset": "subassets/joint.mora", "output": "duct"}}],
        "outputs": {"duct": "r1"},
        "asset_checks": {"grounded": False},
    })
    (project / "proj.mora").write_text(root, encoding="utf-8")
    flatten.write_project(project, project)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["validate", str(project)])
    report = json.loads(buf.getvalue())
    assert rc == 0
    # Green sub-asset entries fold to the per-identity row (DESIGN 4.6):
    # the guard and the mated entry both counted, both green.
    assert report["ref_assets"] == [
        {"asset": "subassets/joint.star", "asserts_total": 2, "asserts_ok": 2}
    ]
    assert not [e for e in report["asserts"] if e.get("asset")]


def test_check_asserts_unknown_kind_is_still_loud():
    """A parsed-but-unhandled kind must never fall through to another
    kind's semantics (the silent-pick class)."""
    with pytest.raises(ValueError):
        check_asserts({}, [{"id": "x", "kind": "mystery", "outputs": ["a", "b"],
                            "owner": "agent"}])


# -- quantifiers: contact `every`, clearance `self` ---------------------------


def _row_asset(entries: list[dict], probe_x: float = 1.0, count: int = 3) -> str:
    """A row of `count` unit boxes (x = 0, 1, 2 ...) on the ground and a
    slab under the row; a probe box touching only the copy at probe_x."""
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "a1", "op": "array", "inputs": ["g1"],
             "params": {"count": count, "step_translate": [1.0, 0.0, 0.0]}},
            {"id": "p1", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
            {"id": "p2", "op": "transform", "inputs": ["p1"],
             "params": {"translate": [probe_x, 1.25, 0.0]}},
        ],
        "outputs": {"row": "a1", "probe": "p2"},
        "asserts": entries,
    })


def test_contact_every_makes_the_guard_universal(tmp_path, monkeypatch):
    """contact over an instanced output is existential (one near copy passes
    for all); `every` checks each named copy and names the ones that do not
    touch (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    plain = {"id": "c", "kind": "contact", "outputs": ["row", "probe"],
             "owner": "agent"}
    report = _run_asserts(_row_asset([plain]))
    e = next(x for x in report["asserts"] if x["assert"] == "c")
    assert e["ok"] is True
    assert e["unchecked"] == {"instanced_copies": ["row"]}
    quantified = dict(plain, every=["row"], owner="human")
    report = _run_asserts(_row_asset([quantified]))
    e = next(x for x in report["asserts"] if x["assert"] == "c")
    assert e["ok"] is False
    assert e["every"] == {"row": 3}
    assert "unchecked" not in e

    rows = e["detail"]["failing_copies"]
    assert [r[0] for r in rows] == ["row[0]", "row[2]"]
    assert all(abs(r[1] - 0.25) < 1e-9 for r in rows)  # probe is 0.25 m inboard of them
    assert "row[0]" in e["fix"] and "fix the geometry, not the tol" in e["fix"]
    # Naming a plain mesh is vacuous (one copy); both names are legal.
    both = dict(plain, every=["probe", "row"])
    report = _run_asserts(_row_asset([both]))
    e = next(x for x in report["asserts"] if x["assert"] == "c")
    assert e["every"] == {"probe": 1, "row": 3}
    assert e["ok"] is False


def test_contact_unquantified_is_a_top_level_warning(tmp_path, monkeypatch):
    """A contact left existential over an instanced output is named in
    the report's warnings, not only inside its green entry (DESIGN 5.2):
    an output arrayed after the entry was written weakens the entry with
    no file diff, and the note names the `every` fix. Quantifying it
    removes the row; `--failures-only` keeps warnings."""
    from moraine3d.verify.validate import failures_only

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    plain = {"id": "c", "kind": "contact", "outputs": ["row", "probe"],
             "owner": "agent"}
    report = _run_asserts(_row_asset([plain]))
    rows = [w for w in report.get("warnings", []) if w["warning"] == "contact_unquantified"]
    assert len(rows) == 1
    assert rows[0]["assert"] == "c" and rows[0]["outputs"] == ["row"]
    assert '"every": ["row"]' in rows[0]["note"]
    assert report["ok"] is True  # advisory: the entry itself stays green
    slim = failures_only(report)
    assert any(w["warning"] == "contact_unquantified" for w in slim["warnings"])

    quantified = dict(plain, every=["row"])
    report = _run_asserts(_row_asset([quantified]))
    assert not any(
        w["warning"] == "contact_unquantified" for w in report.get("warnings", [])
    )


def test_contact_unquantified_carries_the_site_of_its_assert(tmp_path, monkeypatch):
    """Every entry that names an assert carries its recipe site (DESIGN
    6): the warning's row is where the `every` fix is read, so it names
    the line to write it at."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    plain = {"id": "c", "kind": "contact", "outputs": ["row", "probe"],
             "owner": "agent", "site": "row/row.star:12"}
    report = _run_asserts(_row_asset([plain]))
    [row] = [w for w in report["warnings"] if w["warning"] == "contact_unquantified"]
    assert row["site"] == "row/row.star:12"


def test_contact_every_passes_when_every_copy_touches(tmp_path, monkeypatch):
    """Every copy touching the other output passes and still reports the
    checked count (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.loads(_row_asset([{"id": "c", "kind": "contact", "outputs": ["row", "slab"],
                                    "every": ["row"], "owner": "agent"}]))
    asset["nodes"] += [
        {"id": "s1", "op": "box", "params": {"size": [4.0, 0.2, 2.0]}},
        {"id": "s2", "op": "transform", "inputs": ["s1"], "params": {"translate": [1.0, -0.1, 0.0]}},
    ]
    asset["outputs"] = {"row": "a1", "slab": "s2"}
    asset["checks"] = {"slab": {"not_sunk": False}}
    report = _run_asserts(json.dumps(asset))
    e = next(x for x in report["asserts"] if x["assert"] == "c")
    assert e["ok"] is True and e["every"] == {"row": 3} and "fix" not in e
    # The loosest copy rides the green entry (DESIGN 5.2): every copy is
    # flush here, so the largest per-copy distance is at the float floor.
    assert e["every_max"][0].startswith("row[") and e["every_max"][1] <= 1e-5


def test_quantifier_parse_rules(tmp_path):
    """`every`: contact only, a non-empty distinct subset of `outputs`;
    rejected on clearance with the union-minimum reason; `self`:
    clearance and embedded, never contact or mated (asset.py gating)."""
    def load(entry):
        return asset_mod.loads(_row_asset([entry]))

    load({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": ["row"],
          "owner": "agent"})
    load({"id": "c", "kind": "clearance", "outputs": ["row", "probe"], "self": True,
          "owner": "agent", "min": 0.001})
    for entry, needle in [
        ({"id": "c", "kind": "clearance", "outputs": ["row", "probe"], "every": ["row"], "min": 0.001},
         "already holds for every copy"),
        ({"id": "c", "kind": "embedded", "outputs": ["row", "probe"], "every": ["row"]},
         "only for contact"),
        ({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": []}, "non-empty"),
        ({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": ["ghost"]}, "drawn from"),
        ({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": ["row", "row"]}, "distinct"),
        ({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": "row"}, "list"),
        ({"id": "c", "kind": "contact", "outputs": ["row", "probe"], "self": True},
         "clearance and embedded"),
        ({"id": "c", "kind": "clearance", "outputs": ["row", "probe"], "self": False, "min": 0.001}, "JSON true"),
    ]:
        with pytest.raises(AssetError) as exc:
            load(entry)
        assert exc.value.code == "bad_assert"
        assert needle in exc.value.message


def _spaced_row(step: float, entries: list[dict], count: int = 3) -> str:
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "a1", "op": "array", "inputs": ["g1"],
             "params": {"count": count, "step_translate": [step, 0.0, 0.0]}},
            {"id": "w1", "op": "box", "params": {"size": [0.2, 1.0, 1.0]}},
            {"id": "w2", "op": "transform", "inputs": ["w1"], "params": {"translate": [-2.0, 0.5, 0.0]}},
        ],
        "outputs": {"row": "a1", "wall": "w2"},
        "asserts": entries,
    })


def test_clearance_self_checks_copy_spacing(tmp_path, monkeypatch):
    """clearance against another output already holds for every copy; the
    copy-pair gap is what `self` adds: pairs closer than tol fail with
    name[i] labels, the count rides the entry, and without the opt-in the
    instanced output is named unchecked (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    plain = {"id": "gap", "kind": "clearance", "outputs": ["row", "wall"], "min": 0.05,
             "owner": "agent"}

    def gap_entry(report):
        return next(x for x in report["asserts"] if x["assert"] == "gap")

    e = gap_entry(_run_asserts(_spaced_row(1.02, [plain])))
    assert e["ok"] is True and e["unchecked"] == {"instanced_self_pairs": ["row"]}
    opt = dict(plain, self=True)

    e = gap_entry(_run_asserts(_spaced_row(1.02, [opt])))
    assert e["ok"] is False
    assert e["self_pairs"] == 3
    rows = e["detail"]["failing_pairs"]
    assert [r[:2] for r in rows] == [["row[0]", "row[1]"], ["row[1]", "row[2]"]]
    assert all(abs(r[2] - 0.02) < 1e-9 for r in rows)
    assert "increase the array step" in e["fix"]
    assert "unchecked" not in e
    assert e["self_min"][:2] == ["row[0]", "row[1]"] and abs(e["self_min"][2] - 0.02) < 1e-9
    e = gap_entry(_run_asserts(_spaced_row(1.1, [opt])))
    assert e["ok"] is True and e["self_pairs"] == 3 and "fix" not in e
    # The tightest copy gap rides a green entry too (DESIGN 5.2), found
    # by the doubling box search rather than measuring every pair.
    assert e["self_min"][:2] in (["row[0]", "row[1]"], ["row[1]", "row[2]"])  # equal gaps
    assert abs(e["self_min"][2] - 0.1) < 1e-9
    # Plain-mesh scope: vacuous.
    e = gap_entry(_run_asserts(_spaced_row(1.1, [dict(opt, outputs=["wall", "row"])])))
    assert e["self_pairs"] == 3


def test_clearance_self_min_is_the_true_closest_pair_on_a_green_entry(tmp_path, monkeypatch):
    """The closest copy pair a green `self` clearance names is the true
    closest approach (DESIGN 5.2), on a prototype whose clearance pass
    finds a farther triangle pair inside the copy boxes than the pair
    that sets the minimum: a large diagonal triangle and a small one
    inside its box but well off its surface, the second copy turned
    180 degrees about Y so the small triangles face each other 0.08 m
    apart while every pair the pass prices sits above 1 m."""
    from moraine3d.verify.distance import tri_pair_dist

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    verts = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 2.0, 0.0],
        [1.5, 1.0, 0.2], [1.6, 1.0, 0.2], [1.5, 1.1, 0.2],
    ], dtype=np.float32)
    proto = Mesh(vertices=verts, faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    xf = np.zeros((2, 4, 4))
    xf[0] = np.eye(4)
    xf[1] = np.diag([-1.0, 1.0, -1.0, 1.0])
    xf[1, :3, 3] = [3.28, 0.0, 0.4]
    row = Instances(prototype=proto, transforms=xf)
    wall = _moved(_box([0.2, 1.0, 1.0]), (-5.0, 0.5, 0.0))
    rep = validate_outputs({"row": row, "wall": wall}, asserts=[
        {"id": "gap", "kind": "clearance", "outputs": ["row", "wall"], "min": 0.05,
         "self": True, "owner": "agent"},
    ])
    e = next(x for x in rep["asserts"] if x["assert"] == "gap")
    assert e["ok"] is True and e["self_pairs"] == 1
    tris = verts.astype(np.float64)[proto.faces]
    world = [np.einsum("ij,mtj->mti", x[:3, :3], tris) + x[:3, 3] for x in xf]
    brute = tri_pair_dist(
        np.repeat(world[0], len(world[1]), axis=0), np.tile(world[1], (len(world[0]), 1, 1))
    ).min()
    assert abs(brute - 0.08) < 1e-6
    assert e["self_min"][:2] == ["row[0]", "row[1]"]
    assert abs(e["self_min"][2] - brute) < 1e-6


def test_overlapping_copy_pairs_matches_brute_force():
    """The chunked copy-pair broad phase returns every unordered pair whose
    boxes come within `inflate` on all axes, i < j, in row-major order —
    identical to the dense formulation it replaces (DESIGN 5.2)."""
    from moraine3d.verify.pairs import overlapping_copy_pairs

    rng = np.random.default_rng(3)
    k = 60
    lo = rng.uniform(0.0, 10.0, size=(k, 3))
    hi = lo + rng.uniform(0.1, 1.5, size=(k, 3))
    for inflate in (0.0, 0.3):
        total, ii, jj = overlapping_copy_pairs(lo, hi, inflate=inflate)
        assert total == k * (k - 1) // 2
        dense = np.all(
            (lo[:, None, :] - inflate <= hi[None, :, :]) & (lo[None, :, :] <= hi[:, None, :] + inflate),
            axis=2,
        )
        ei, ej = np.nonzero(np.triu(dense, k=1))
        assert np.array_equal(ii, ei) and np.array_equal(jj, ej)


def test_every_fix_lists_a_bounded_number_of_copies(tmp_path, monkeypatch):
    """A quantified contact names at most a few failing copies in its fix
    sentence and carries all of them in detail (token economy)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.loads(_row_asset(
        [{"id": "c", "kind": "contact", "outputs": ["row", "probe"], "every": ["row"],
          "owner": "agent"}],
        probe_x=0.0, count=14,
    ))
    report = _run_asserts(json.dumps(asset))
    e = next(x for x in report["asserts"] if x["assert"] == "c")
    assert e["ok"] is False
    assert len(e["detail"]["failing_copies"]) == 13
    assert "13 copies not touching" in e["fix"]
    assert "and 3 more" in e["fix"]
    assert e["fix"].count("row[") == 10


def test_clearance_self_fix_lists_a_bounded_number_of_pairs(tmp_path, monkeypatch):
    """A `self` clearance names at most a few failing copy pairs in its
    fix sentence and carries all of them in detail (token economy)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    opt = {"id": "gap", "kind": "clearance", "outputs": ["row", "wall"], "min": 0.05,
           "self": True, "owner": "agent"}
    rep = _run_asserts(_spaced_row(1.02, [opt], count=14))
    e = next(x for x in rep["asserts"] if x["assert"] == "gap")
    assert e["ok"] is False and e["self_pairs"] == 91
    assert len(e["detail"]["failing_pairs"]) == 13
    assert "and 3 more (see detail)" in e["fix"]
    assert e["fix"].count("row[") == 20
    assert "increase the array step" in e["fix"]


def test_guard_copy_pair_fix_lists_a_bounded_number_of_pairs(tmp_path, monkeypatch):
    """The overlap guard names at most a few failing copy pairs in its
    fix sentence and carries all of them in detail (token economy)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    row = array_op.evaluate(
        [_moved(_box([1.0, 1.0, 1.0]), (0.0, 0.5, 0.0))],
        {"count": 14, "step_translate": [0.5, 0.0, 0.0]}, None,
    )
    rep = validate_outputs({"row": row})
    e = next(x for x in rep["asserts"] if x["assert"] == "no_overlap")
    assert e["ok"] is False and e["self_pairs"] == 91
    assert len(e["detail"]["failing_pairs"]) == 13
    assert "and 3 more (see detail)" in e["fix"]
    assert e["fix"].count("row[") == 20
    assert "increase the array step" in e["fix"]


def test_query_reads_instanced_outputs_on_the_prototype(tmp_path, monkeypatch, capsys):
    """`moraine query` on an instanced output evaluates the predicate on the
    prototype in its local frame — the same frame `tag` uses — and says so
    (Face predicates contract)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "q"
    project.mkdir()
    (project / "q.mora").write_text(json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "a1", "op": "array", "inputs": ["b1"],
             "params": {"count": 3, "step_rotate_deg": [0.0, 0.0, 90.0]}},
        ],
        "outputs": {"row": "a1"},
        "checks": {"row": {"not_sunk": False}},
    }), encoding="utf-8")
    flatten.write_project(project, project)
    rc = main(["query", str(project), "--output", "row", "--select",
               json.dumps({"normal_y": {"gt": 0.9}})])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    row = out["outputs"]["row"]
    assert row["instanced"] == 3
    assert row["frame"].startswith("prototype")
    assert row["matched"] == 2


def test_docs_slice_mate(capsys):
    """`moraine docs mate` returns the mate section (docs CLI contract)."""
    assert main(["docs", "mate"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### mate")
    assert "face_instance" in out and "roll_deg" in out
    assert "### place" not in out and "### Asset assertions" not in out


def _cell_quads(n: int, removed: set) -> np.ndarray:
    """An n x n planar quad sheet in the XY plane (+Z facing, CCW) with the
    listed (column, row) cells removed, as (m, 3, 3) triangles."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = np.array([(int(a), int(b)) not in removed for a, b in zip(i.ravel(), j.ravel())])
    i, j = i.ravel()[keep].astype(np.float64), j.ravel()[keep].astype(np.float64)
    z = np.zeros_like(i)
    a = np.stack([i, j, z], axis=1)
    b = np.stack([i + 1, j, z], axis=1)
    c = np.stack([i + 1, j + 1, z], axis=1)
    d = np.stack([i, j + 1, z], axis=1)
    return np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])


def test_frame_refuses_openings_pinched_at_two_vertices():
    """A boundary chain that leaves one pinch vertex and ends at another
    does not close, so the openings cannot be told apart — the set is
    refused as ambiguous_opening (never a guessed count or a wrong
    disconnected verdict), while a plain ring and a single-hole face keep
    their frames (DESIGN 5.1)."""
    with pytest.raises(mating.FrameError) as exc:
        mating.group_frame(_cell_quads(7, {(1, 3), (2, 2), (3, 1)}))
    assert exc.value.code == "ambiguous_opening"
    assert "pinch" in exc.value.message
    assert mating.REMEDY["ambiguous_opening"] == "tag the faces around one opening"
    one_hole = mating.group_frame(_cell_quads(7, {(3, 3)}))
    assert one_hole.openings == 1
    assert np.allclose(one_hole.origin, [3.5, 3.5, 0.0], atol=1e-9)
    ring = mating.group_frame(_cell_quads(3, {(1, 1)}))
    assert ring.openings == 1 and np.allclose(ring.origin, [1.5, 1.5, 0.0], atol=1e-9)
    assert mating.group_frame(_cell_quads(4, set())).openings == 0
    assert mate_op.OP_IMPL_VERSION == 1  # refusal only: no legal output changed


def test_frame_planarity_test_always_keeps_the_largest_faces(monkeypatch):
    """The sliver floor never empties the planarity test set: the largest
    face(s) are always tested, so a set of many equal faces (each under
    the relative floor) yields a frame instead of a raw reduction error
    (DESIGN 5.1)."""
    monkeypatch.setattr(mating, "SLIVER_AREA_FRACTION", 0.5)  # every quarter is "a sliver"
    f = mating.group_frame(_cell_quads(2, set()))
    assert f.openings == 0 and np.allclose(f.normal, [0.0, 0.0, 1.0], atol=1e-9)
    flipped = _cell_quads(2, set())
    flipped[-1] = flipped[-1, ::-1]  # one equal-area face wound the other way: still tested
    with pytest.raises(mating.FrameError) as exc:
        mating.group_frame(flipped)
    assert exc.value.code == "not_planar"


# -- frame provenance and the stronger-mode word (DESIGN 5.1, 5.2) ----------


def test_mated_entry_echoes_the_origin_provenance(tmp_path, monkeypatch):
    """A mated entry's detail carries each face's frame provenance
    (DESIGN 5.2): `opening` when the origin is the single opening's
    centroid — the duct docks on the wall's port and its own mouth."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_duct_asset(_MATE_NODE, [_MATED] + _GUARDS))
    entry = next(e for e in report["asserts"] if e["assert"] == "duct_mated")
    assert entry["detail"]["origin"] == ["opening", "opening"]
    assert "stronger" not in entry  # coincident already is the stronger mode


_BLANK_NODES = [
    {"id": "bs01", "op": "box", "params": {}},
    {"id": "bs02", "op": "transform", "inputs": ["bs01"], "params": {"translate": [0.0, 0.5, 0.0]}},
    {"id": "bs03", "op": "tag", "inputs": ["bs02"],
     "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
    {"id": "pt01", "op": "box", "params": {"size": [0.4, 0.4, 0.4]}},
    {"id": "pt02", "op": "tag", "inputs": ["pt01"],
     "params": {"name": "bottom", "select": {"normal_y": {"lt": -0.9}}}},
    {"id": "pt03", "op": "mate", "inputs": ["bs03", "pt02"],
     "params": {"anchor": "top", "face": "bottom"}},
]
_SEATED = {"id": "seat", "kind": "mated", "mode": "contained",
           "faces": [["base", "top"], ["part", "bottom"]], "owner": "agent"}


def _validate_project(project, *flags) -> dict:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["validate", str(project), *flags])
    assert rc == 0, buf.getvalue()
    return json.loads(buf.getvalue())


def test_mate_report_carries_the_centroid_fallback_and_the_stronger_mode(tmp_path, monkeypatch):
    """The mate op's own report (DESIGN 5.1): `validate` lists every root
    mate node under `mates` with the origin word per role — `centroid`
    is the silent fallback the paired mated shares and passes through,
    so the word is the only place it shows. A green `contained` entry
    whose origins also register carries `stronger: "coincident"` (DESIGN
    5.2) — the entry under-claims and nothing else reports it — and both
    survive `--failures-only`."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "proj.mora").write_text(
        _asset(_BLANK_NODES, {"base": "bs03", "part": "pt03"}, [_SEATED]),
        encoding="utf-8",
    )
    flatten.write_project(project, project)
    report = _validate_project(project)
    assert report["mates"] == [
        {"node": "pt03", "anchor": "centroid", "face": "centroid", "site": "proj.star:6"}
    ]
    seat = next(e for e in report["asserts"] if e["assert"] == "seat")
    assert seat["ok"] is True and seat["mode"] == "contained"
    assert seat["stronger"] == "coincident"
    assert seat["detail"]["origin"] == ["centroid", "centroid"]

    slim = _validate_project(project, "--failures-only")
    assert slim["stronger"] == {"entries": 1, "coincident": 1}
    assert slim["mates"] == report["mates"]
    assert "asserts" not in slim  # the green entry itself is still compressed

    # The duct joint registers on openings: the words say so.
    duct = tmp_path / "duct"
    duct.mkdir()
    (duct / "duct.mora").write_text(
        _duct_asset(_MATE_NODE, [_MATED] + _GUARDS), encoding="utf-8"
    )
    flatten.write_project(duct, duct)
    [mate] = _validate_project(duct)["mates"]
    assert mate.pop("site").startswith("duct.star:")
    assert mate == {"node": "dt05", "anchor": "opening", "face": "opening"}


def _others_asset(entries: list[dict], *, extra: list[tuple[str, float]] = (), image: bool = False) -> str:
    """A unit drum at the origin with unit boxes a (gap 1.0) and b (gap
    1.5), plus caller-added boxes ``(name, x)`` and optionally an image
    output, for the one-vs-rest clearance form."""
    nodes = [
        {"id": "d", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "dt", "op": "transform", "inputs": ["d"], "params": {"translate": [0.0, 0.5, 0.0]}},
    ]
    outputs = {"drum": "dt"}
    for name, x in (("a", 2.0), ("b", -2.5), *extra):
        nodes.append({"id": f"{name}_b", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}})
        nodes.append({"id": f"{name}_t", "op": "transform", "inputs": [f"{name}_b"],
                      "params": {"translate": [x, 0.5, 0.0]}})
        outputs[name] = f"{name}_t"
    if image:
        nodes.append({"id": "tex", "op": "checker_texture", "params": {"tiles": 2, "size": [8, 8]}})
        outputs["tex"] = "tex"
    return json.dumps({"build_version": "fixture", "nodes": nodes, "outputs": outputs, "asserts": entries})


def test_clearance_others_covers_every_other_output(tmp_path, monkeypatch):
    """The one-vs-rest clearance (DESIGN 5.2): `"outputs": [a], "others":
    true` keeps `a` at least tol from every other geometry output the
    asset has at validate time — an output added later is covered
    automatically, images are not partners, and the entry reports the
    partner count, the nearest partner with its distance, and failing
    partners by name."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = {"id": "clear", "kind": "clearance", "outputs": ["drum"], "others": True,
             "min": 0.5, "owner": "human"}

    def clear(report):
        return next(x for x in report["asserts"] if x["assert"] == "clear")

    e = clear(_run_asserts(_others_asset([entry], image=True)))
    assert e["ok"] is True and e["others"] is True and e["outputs"] == ["drum"]
    assert e["pairs"] == 2 and e["nearest"] == "a" and abs(e["distance"] - 1.0) < 1e-6
    # A part added later is a partner nobody had to write.
    report = _run_asserts(_others_asset([entry], extra=[("c", 1.3)]))
    e = clear(report)
    assert e["ok"] is False and report["ok"] is False
    assert e["pairs"] == 3 and e["nearest"] == "c" and abs(e["distance"] - 0.3) < 1e-6
    assert e["detail"]["failing_partners"] == [["c", e["distance"]]]
    assert "'drum' and 'c' come within 0.3 m" in e["fix"]
    assert "fix the geometry, not the 'min'" in e["fix"]  # human-owned: no weakening advice


def test_clearance_others_is_vacuous_alone_and_composes_with_self(tmp_path, monkeypatch):
    """With no other geometry output the one-vs-rest entry is vacuous and
    legal (pairs 0, no distance); `self` composes over the named output's
    copies alone (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.loads(_others_asset([
        {"id": "clear", "kind": "clearance", "outputs": ["drum"], "others": True,
         "min": 0.5, "owner": "agent"},
    ], image=True))
    for name in ("a", "b"):
        del asset["outputs"][name]
    e = next(x for x in _run_asserts(json.dumps(asset))["asserts"] if x["assert"] == "clear")
    assert e["ok"] is True and e["pairs"] == 0 and "distance" not in e and "nearest" not in e

    entry = {"id": "gap", "kind": "clearance", "outputs": ["row"], "others": True,
             "self": True, "min": 0.05, "owner": "agent"}
    e = next(x for x in _run_asserts(_spaced_row(1.02, [entry]))["asserts"] if x["assert"] == "gap")
    assert e["ok"] is False and e["pairs"] == 1 and e["nearest"] == "wall"
    assert e["self_pairs"] == 3 and [r[:2] for r in e["detail"]["failing_pairs"]] == [
        ["row[0]", "row[1]"], ["row[1]", "row[2]"]]
    assert "unchecked" not in e
    # The two failure lists keep their own keys and row shapes: copy pairs
    # `[a, b, distance]` under `failing_pairs`, partners `[other,
    # distance]` under `failing_partners` — never one overwriting the other.
    assert all(len(r) == 3 for r in e["detail"]["failing_pairs"])
    assert all(len(r) == 2 for r in e["detail"].get("failing_partners", []))
    assert ("failing_partners" in e["detail"]) == (e["distance"] < 0.05)


def test_clearance_others_parse_rules(tmp_path):
    """`others`: clearance only, JSON true, exactly one output named
    (asset.py gating); the two-output form stays as it was."""
    def load(entry):
        return asset_mod.loads(_row_asset([entry]))

    load({"id": "c", "kind": "clearance", "outputs": ["row"], "others": True, "owner": "agent", "min": 0.001})
    for entry, code, needle in [
        ({"id": "c", "kind": "contact", "outputs": ["row"], "others": True},
         "bad_assert", "only for clearance"),
        ({"id": "c", "kind": "clearance", "outputs": ["row", "probe"], "others": True, "min": 0.001},
         "bad_assert_outputs", "exactly one"),
        ({"id": "c", "kind": "clearance", "outputs": ["row"], "others": False, "min": 0.001},
         "bad_assert", "JSON true"),
        ({"id": "c", "kind": "clearance", "outputs": ["row"], "min": 0.001},
         "bad_assert_outputs", "nothing to measure it against"),
        ({"id": "c", "kind": "clearance", "outputs": ["ghost"], "others": True, "min": 0.001},
         "bad_assert_outputs", "unknown output"),
    ]:
        with pytest.raises(AssetError) as exc:
            load(dict(entry, owner="agent"))
        assert exc.value.code == code, entry
        assert needle in exc.value.message, entry
