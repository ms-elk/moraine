"""P6 vertical slice: predicates, tag, place, asserts, query.

One vocabulary end to end (DESIGN 5.1): a predicate selects faces, tag
fixes the selection into the graph, place anchors a part to it, an
assertion guards the result, and `moraine query` reads with the same
expressions.
"""

import json

import numpy as np
import pytest

from moraine3d.app.cli import main
from moraine3d.core import predicates
from moraine3d.core import asserts as vocab
from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Mesh
from moraine3d.errors import MoraineError, OpError, AssetError
from moraine3d.ops import bake_instances, box
from moraine3d.ops import place as place_op
from moraine3d.ops import tag as tag_op
from moraine3d.verify.validate import validate_outputs
from support import flatten
from support.engine import evaluate_records


def _box(size):
    return box.evaluate([], {"size": size}, None)


def _tag(mesh, name="top", select=None, **params):
    select = select if select is not None else {"normal_y": {"gt": 0.9}}
    return tag_op.evaluate([mesh], {"name": name, "select": select, **params}, None)


# -- predicate vocabulary ----------------------------------------------------


def test_predicate_selects_box_top():
    m = _box([1.0, 2.0, 3.0])
    mask = predicates.evaluate_predicate(m, {"normal_y": {"gt": 0.9}})
    assert mask.sum() == 2  # box top = 2 triangles
    tri = m.vertices[m.faces[mask]]
    assert np.allclose(tri[..., 1], 1.0)  # all corners at y = +sy/2


def test_predicate_fields_and_conjunction():
    m = _box([1.0, 1.0, 1.0])
    # +Y faces AND centered above 0.4: still the top.
    both = predicates.evaluate_predicate(
        m, {"normal_y": {"gt": 0.9}, "center_y": {"ge": 0.4}}
    )
    assert both.sum() == 2
    # Conjunction with an unsatisfiable bound is empty.
    none = predicates.evaluate_predicate(
        m, {"normal_y": {"gt": 0.9}, "center_y": {"lt": -0.4}}
    )
    assert none.sum() == 0
    # area: every unit-cube face triangle is 0.5 m^2.
    assert predicates.evaluate_predicate(m, {"area": {"eq": 0.5, "tol": 1e-9}}).all()


@pytest.mark.parametrize(
    "pred",
    [
        None,
        {},
        {"normal_w": {"gt": 0}},
        {"normal_y": 0.9},
        {"normal_y": {}},
        {"normal_y": {"gt": True}},
        {"normal_y": {"between": [0, 1]}},
        {"normal_y": {"tol": 0.1}},  # tol without eq
        {"normal_y": {"eq": 1.0, "tol": -1}},
    ],
)
def test_predicate_validation_rejects(pred):
    with pytest.raises(MoraineError) as exc:
        predicates.validate_predicate(pred)
    assert exc.value.code == "bad_predicate"


def test_cardinality_errors_are_structured():
    with pytest.raises(MoraineError) as exc:
        predicates.check_cardinality(0, "one_or_more", {"area": {"lt": 0}})
    assert exc.value.code == "empty_selection"
    with pytest.raises(MoraineError) as exc:
        predicates.check_cardinality(2, "exactly_one", {})
    assert exc.value.code == "ambiguous_selection"
    predicates.check_cardinality(0, "any", {})  # never raises


# -- output_query / resolve_pin (value-kind dispatch, DESIGN 6 / 7.2) --------


def test_output_query_dispatches_per_kind():
    """One output's answer to a predicate dispatches on the value's kind:
    an image is skipped (face predicates apply to meshes), an instanced
    output is read on its prototype in its local frame — the same frame
    `tag` uses — and says so, a plain mesh is summarized directly (Face
    predicates contract)."""
    from moraine3d.core.mesh import Image, Instances

    pred = {"normal_y": {"gt": 0.9}}
    m = _box([1.0, 2.0, 3.0])
    plain = predicates.output_query(m, pred)
    assert plain["matched"] == 2 and plain["faces"] == len(m.faces)
    inst = predicates.output_query(
        Instances(prototype=m, transforms=np.stack([np.eye(4)] * 3)), pred
    )
    assert inst["matched"] == 2
    assert inst["instanced"] == 3
    assert inst["frame"].startswith("prototype")
    img = predicates.output_query(
        Image(pixels=np.zeros((2, 2, 3), dtype=np.float32)), pred
    )
    assert "skipped" in img and "matched" not in img


def test_resolve_pin_reads_groups_and_fields():
    """A pin resolves a pointed-at face into structural vocabulary: its
    tag-group membership plus the predicate-vocabulary face fields
    (DESIGN 7.2) — the gesture hands the agent the makings of a
    predicate, never just a bare index."""
    t = _tag(_box([1.0, 2.0, 3.0]))  # tags faces 4, 5 as "top"
    pin = predicates.resolve_pin(t, 4, None, subject="output 'part'")
    assert pin["groups"] == ["top"]
    assert pin["face"] == 4 and pin["instance"] is None
    assert pin["face_center"][1] == 1.0  # top plane of the 2 m box
    assert pin["face_normal"] == [0.0, 1.0, 0.0]
    assert pin["area"] == 1.5  # half of the 1 x 3 top rectangle
    # Plain mesh: world position is the local face center.
    assert pin["position"] == pin["face_center"]
    # An untagged face reports no groups, not an error.
    assert predicates.resolve_pin(t, 0, None, subject="s")["groups"] == []


def test_resolve_pin_instanced_uses_copy_frame():
    """On an instanced value the pin's face fields stay in the prototype's
    local frame while ``position`` is the face centroid placed in world
    space by the chosen copy's transform; the copy index is required,
    never guessed (DESIGN 7.2)."""
    from moraine3d.core.mesh import Instances
    from moraine3d.core.predicates import FaceGroupError

    xf = np.stack([np.eye(4), np.eye(4)])
    xf[1, :3, 3] = [10.0, 0.0, 0.0]
    inst = Instances(prototype=_box([1.0, 2.0, 3.0]), transforms=xf)
    pin = predicates.resolve_pin(inst, 4, 1, subject="output 'row'")
    assert pin["instance"] == 1
    assert pin["face_center"][1] == 1.0
    assert pin["position"][0] == pin["face_center"][0] + 10.0
    with pytest.raises(FaceGroupError) as exc:
        predicates.resolve_pin(inst, 4, None, subject="s")
    assert exc.value.code == "missing_instance"
    with pytest.raises(FaceGroupError) as exc:
        predicates.resolve_pin(inst, 4, 5, subject="s")
    assert exc.value.code == "bad_instance"
    with pytest.raises(FaceGroupError) as exc:
        predicates.resolve_pin(_box([1.0, 1.0, 1.0]), 0, 0, subject="s")
    assert exc.value.code == "bad_instance"  # plain mesh refuses a copy index


def test_predicate_pin_lists_matched_faces_capped(monkeypatch):
    """A predicate pin carries the matched face indices so display
    clients can highlight exactly the selected polygons, and omits the
    list past the display cap — marker only, never a truncated set
    passed off as complete (DESIGN 7.2)."""
    m = _box([1.0, 2.0, 3.0])
    pin = predicates.resolve_predicate_pin(
        m, {"normal_y": {"gt": 0.9}}, None, subject="s"
    )
    assert pin["faces"] == [4, 5]
    monkeypatch.setattr(predicates, "PIN_FACES_MAX", 1)
    pin = predicates.resolve_predicate_pin(
        m, {"normal_y": {"gt": 0.9}}, None, subject="s"
    )
    assert "faces" not in pin and pin["summary"]["matched"] == 2


def test_resolve_pin_barycentric_names_the_pointed_spot():
    """A click position yields ``bary`` — the click's barycentric
    coordinates in the face, clamped into the triangle, computed in the
    prototype's frame even when the click arrives in a copy's world
    space — so display clients re-derive the exact pointed spot on
    whatever shape the face currently has (DESIGN 7.2). Without a
    position there is no bary."""
    from moraine3d.core.mesh import Instances

    m = _box([1.0, 2.0, 3.0])
    tri = m.vertices[m.faces[4]].astype(float)
    pin = predicates.resolve_pin(
        m, 4, None, subject="s", position=[float(v) for v in tri[1]]
    )
    assert pin["bary"] == [0.0, 1.0, 0.0]
    # The reported position is the bary point on the file-state face —
    # a click cast against a preview shape cannot skew the record.
    assert pin["position"] == [round(float(v), 6) for v in tri[1]]
    # A hit a hair outside the face still names a point on it.
    outside = tri[0] + 2.0 * (tri[0] - (tri.sum(axis=0) / 3.0))
    bary = predicates.resolve_pin(
        m, 4, None, subject="s", position=[float(v) for v in outside]
    )["bary"]
    assert all(b >= 0.0 for b in bary) and abs(sum(bary) - 1.0) < 1e-6
    assert "bary" not in predicates.resolve_pin(m, 4, None, subject="s")
    xf = np.stack([np.eye(4), np.eye(4)])
    xf[1, :3, 3] = [10.0, 0.0, 0.0]
    inst = Instances(prototype=m, transforms=xf)
    world = [float(tri[2][0]) + 10.0, float(tri[2][1]), float(tri[2][2])]
    pin = predicates.resolve_pin(inst, 4, 1, subject="s", position=world)
    assert pin["bary"] == [0.0, 0.0, 1.0]


def test_resolve_pin_rejects_images_and_bad_faces():
    """Pins live on geometry: an image output and an out-of-range or
    non-integer face index are structured errors, never silent picks
    (DESIGN 7.2)."""
    from moraine3d.core.mesh import Image
    from moraine3d.core.predicates import FaceGroupError

    with pytest.raises(FaceGroupError) as exc:
        predicates.resolve_pin(
            Image(pixels=np.zeros((2, 2, 3), dtype=np.float32)), 0, None, subject="s"
        )
    assert exc.value.code == "image"
    m = _box([1.0, 1.0, 1.0])
    for face in (-1, len(m.faces), True, "4"):
        with pytest.raises(FaceGroupError) as exc:
            predicates.resolve_pin(m, face, None, subject="s")
        assert exc.value.code == "bad_face"


# -- tag ---------------------------------------------------------------------


def test_tag_fixes_selection_and_preserves_geometry():
    m = _box([1.0, 2.0, 3.0])
    t = _tag(m)
    assert np.array_equal(t.vertices, m.vertices)
    assert np.array_equal(t.faces, m.faces)
    assert list(t.face_groups) == ["top"]
    assert t.face_groups["top"].tolist() == [4, 5]  # +Y face triangles, ascending
    assert t.stats()["groups"] == {"top": 2}
    # The tag is a semantic change downstream must see.
    assert t.result_hash() != m.result_hash()


def test_tag_stacks_and_rejects_collision():
    t = _tag(_box([1.0, 1.0, 1.0]))
    t2 = tag_op.evaluate(
        [t], {"name": "bottom", "select": {"normal_y": {"lt": -0.9}}}, None
    )
    assert sorted(t2.face_groups) == ["bottom", "top"]
    with pytest.raises(OpError) as exc:
        _tag(t2)  # "top" again
    assert exc.value.code == "group_exists"


def test_tag_cardinality_gates():
    m = _box([1.0, 1.0, 1.0])
    with pytest.raises(MoraineError) as exc:
        _tag(m, select={"center_y": {"gt": 99.0}})
    assert exc.value.code == "empty_selection"
    with pytest.raises(MoraineError) as exc:
        _tag(m, expect="exactly_one")
    assert exc.value.code == "ambiguous_selection"
    # expect: "any" admits an empty selection.
    t = _tag(m, select={"center_y": {"gt": 99.0}}, expect="any")
    assert t.face_groups["top"].tolist() == []


# -- place -------------------------------------------------------------------


def test_place_on_top_contract():
    base = _tag(_box([2.0, 1.0, 2.0]))
    part = _box([0.5, 0.4, 0.5])
    placed = place_op.evaluate([base, part], {"anchor": "top"}, None)
    lo, hi = placed.bounds()
    # Rests on the anchor plane (base top at y = 0.5), centered on x/z.
    assert abs(float(lo[1]) - 0.5) < 1e-6
    assert abs(float(lo[0] + hi[0])) < 1e-6
    assert abs(float(lo[2] + hi[2])) < 1e-6
    assert np.array_equal(placed.faces, part.faces)
    assert np.array_equal(placed.normals, part.normals)  # pure translation


def test_place_offset_and_errors():
    base = _tag(_box([2.0, 1.0, 2.0]))
    part = _box([0.5, 0.5, 0.5])
    shifted = place_op.evaluate(
        [base, part], {"anchor": "top", "offset": [0.3, 0.1, 0.0]}, None
    )
    lo, hi = shifted.bounds()
    assert abs(float(lo[1]) - 0.6) < 1e-6
    assert abs(float(lo[0] + hi[0]) / 2 - 0.3) < 1e-6

    with pytest.raises(OpError) as exc:
        place_op.evaluate([base, part], {"anchor": "roof"}, None)
    assert exc.value.code == "unknown_anchor"
    assert "top" in exc.value.message  # lists what IS available

    with pytest.raises(OpError) as exc:
        place_op.evaluate([_box([1, 1, 1]), part], {"anchor": "top"}, None)
    assert exc.value.code == "unknown_anchor"  # no tags at all -> hint

    down = tag_op.evaluate(
        [_box([1, 1, 1])], {"name": "under", "select": {"normal_y": {"lt": -0.9}}}, None
    )
    with pytest.raises(OpError) as exc:
        place_op.evaluate([down, part], {"anchor": "under"}, None)
    assert exc.value.code == "anchor_not_upward"

    hollow = _tag(_box([1, 1, 1]), name="none",
                  select={"center_y": {"gt": 99.0}}, expect="any")
    with pytest.raises(OpError) as exc:
        place_op.evaluate([hollow, part], {"anchor": "none"}, None)
    assert exc.value.code == "empty_anchor"


def test_place_non_coplanar_anchor_rejected():
    # Two upward faces at different heights in one tag: the centroid would
    # sit between the surfaces and rest the part over empty space —
    # rejected, never guessed around.
    v = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [2.0, 0.5, 0.0], [3.0, 0.5, 0.0], [3.0, 0.5, 1.0], [2.0, 0.5, 1.0],
        ],
        np.float32,
    )
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], np.int32)
    part = _box([0.5, 0.5, 0.5])

    base = Mesh(vertices=v, faces=f, face_groups={"pads": np.arange(4, dtype=np.int64)})
    with pytest.raises(OpError) as exc:
        place_op.evaluate([base, part], {"anchor": "pads"}, None)
    assert exc.value.code == "anchor_not_coplanar"
    assert "different heights" in exc.value.message

    # The same tag restricted to one surface is a valid anchor.
    one = Mesh(vertices=v, faces=f, face_groups={"pads": np.array([2, 3], np.int64)})
    placed = place_op.evaluate([one, part], {"anchor": "pads"}, None)
    assert np.isclose(placed.bounds()[0][1], 0.5)


def test_place_instance_anchor():
    """Per-instance anchoring (DESIGN §5.1): an instanced anchor plus
    'instance' selects one copy's world-space frame; the on_top contract
    holds unchanged against that frame."""
    from moraine3d.ops import array

    base = _tag(_box([2.0, 1.0, 2.0]))
    row = array.evaluate(
        [base], {"count": 3, "step_translate": [3.0, 0.0, 0.0]}, None
    )
    part = _box([0.5, 0.4, 0.5])
    placed = place_op.evaluate([row, part], {"anchor": "top", "instance": 2}, None)
    lo, hi = placed.bounds()
    # Copy 2 sits at x = 6; its top plane is still y = 0.5.
    assert abs(float(lo[1]) - 0.5) < 1e-6
    assert abs(float(lo[0] + hi[0]) / 2 - 6.0) < 1e-6
    assert abs(float(lo[2] + hi[2])) < 1e-6
    assert np.array_equal(placed.faces, part.faces)  # pure translation


def test_place_instance_rotated_copy():
    """The instance frame is world-space: a rotated copy's anchor centroid
    moves with the rotation (docs contract)."""
    from moraine3d.ops import array
    from moraine3d.ops import transform as transform_op

    # Anchor centroid off the rotation axis: box shifted to x = 1.
    shifted = transform_op.evaluate(
        [_box([2.0, 1.0, 2.0])], {"translate": [1.0, 0.5, 0.0]}, None
    )
    row = array.evaluate(
        [_tag(shifted)], {"count": 2, "step_rotate_deg": [0.0, 90.0, 0.0]}, None
    )
    part = _box([0.4, 0.4, 0.4])
    placed = place_op.evaluate([row, part], {"anchor": "top", "instance": 1}, None)
    lo, hi = placed.bounds()
    # Ry(90) maps the top centroid (1, 1, 0) to (0, 1, -1); plane y stays 1.
    assert abs(float(lo[1]) - 1.0) < 1e-6
    assert abs(float(lo[0] + hi[0]) / 2 - 0.0) < 1e-6
    assert abs(float(lo[2] + hi[2]) / 2 - (-1.0)) < 1e-6


def test_guard_ignores_image_outputs(tmp_path, monkeypatch):
    """The overlap guard quantifies over parts: an image output is
    outside its domain and neither fails the entry nor counts a pair."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [1.2, 0.5, 0.0]}},
                {"id": "tx", "op": "noise_texture",
                 "params": {"seed": 1, "size": 16}},
            ],
            "outputs": {"a": "g1", "b": "g2", "tex": "tx"},
        }
    )
    sc = asset_mod.loads(asset)
    recs = evaluate_records(sc)
    report = validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts,
    )
    entry = next(e for e in report["asserts"] if e.get("builtin"))
    assert entry["ok"] and entry["pairs"] == 1  # a-b only; the image is no part


def test_place_instance_errors():
    from moraine3d.ops import array

    base = _tag(_box([2.0, 1.0, 2.0]))
    row = array.evaluate(
        [base], {"count": 2, "step_translate": [3.0, 0.0, 0.0]}, None
    )
    part = _box([0.5, 0.5, 0.5])
    # Instanced anchor without 'instance': the copy is never guessed.
    with pytest.raises(OpError) as exc:
        place_op.evaluate([row, part], {"anchor": "top"}, None)
    assert exc.value.code == "missing_instance"
    assert "0..1" in exc.value.message
    # Out of range errors instead of going stale.
    with pytest.raises(OpError) as exc:
        place_op.evaluate([row, part], {"anchor": "top", "instance": 5}, None)
    assert exc.value.code == "bad_param"
    # 'instance' on a plain-mesh anchor is a mistake, not a no-op.
    with pytest.raises(OpError) as exc:
        place_op.evaluate([base, part], {"anchor": "top", "instance": 0}, None)
    assert exc.value.code == "bad_param"
    # The part input stays mesh-only.
    inst_part = array.evaluate([part], {"count": 2}, None)
    with pytest.raises(OpError) as exc:
        place_op.evaluate([base, inst_part], {"anchor": "top"}, None)
    assert exc.value.code == "bad_input"


# -- group plumbing (propagation, cache, warnings, baking) -------------------

TAGGED_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [2.0, 1.0, 2.0]}},
{"id": "g1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "t1", "op": "tag", "inputs": ["g1"], "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
{"id": "m1", "op": "transform", "inputs": ["t1"], "params": {"translate": [1.0, 0.0, 0.0]}},
{"id": "b2", "op": "box", "params": {"size": [0.5, 0.4, 0.5]}},
{"id": "p1", "op": "place", "inputs": ["m1", "b2"], "params": {"anchor": "top"}}
],
"outputs": {"base": "m1", "crate": "p1"}
}
"""


def test_groups_ride_cache_and_survive_transform(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(TAGGED_ASSET)
    recs = evaluate_records(sc)
    # transform after tag: group propagated by the evaluator.
    assert recs["base"].stats["groups"] == {"top": 2}
    # placed crate rests on the moved base's top plane, centered at x=1.
    lo, hi = recs["crate"].value().bounds()
    assert abs(float(lo[1]) - 1.0) < 1e-6
    assert abs(float(lo[0] + hi[0]) / 2 - 1.0) < 1e-6

    # Cold reload from cache: groups round-trip through npz.
    recs2 = evaluate_records(asset_mod.loads(TAGGED_ASSET))
    assert recs2["base"].key == recs["base"].key
    v = recs2["base"].value()
    assert v.face_groups["top"].tolist() == [4, 5]


def test_topology_rebuild_drops_groups_with_warning(tmp_path, monkeypatch):
    from support.engine import Evaluation

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(
        """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "t1", "op": "tag", "inputs": ["b1"], "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
{"id": "s1", "op": "smooth_normals", "inputs": ["t1"], "params": {}}
],
"outputs": {"main": "s1"}
}
"""
    )
    ev = Evaluation(sc)
    recs = ev.records()
    assert "groups" not in recs["main"].stats
    warnings = ev.attribute_drop_warnings()
    drops = [w for w in warnings if w["warning"] == "groups_dropped"]
    assert len(drops) == 1 and drops[0]["groups"] == ["top"]


def test_bake_instances_tiles_groups():
    from moraine3d.ops import array

    proto = _tag(_box([1.0, 1.0, 1.0]))
    inst = array.evaluate(
        [proto], {"count": 3, "step_translate": [2.0, 0.0, 0.0]}, None
    )
    baked = bake_instances.evaluate([inst], {}, None)
    # 12 faces per copy; the +Y pair repeats at each copy's offset.
    assert baked.face_groups["top"].tolist() == [4, 5, 16, 17, 28, 29]


# -- asserts -----------------------------------------------------------------


def _assert_asset(dy: float) -> str:
    # The stacked-box geometry of `_dist_asset` with no authored asserts:
    # the built-in overlap guard covers the pair (single source for the
    # dy=1.5-is-flush contract).
    return _dist_asset(dy, [])


def _guard(report: dict) -> dict:
    return next(e for e in report["asserts"] if e.get("builtin"))


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def test_overlap_guard_pass_and_fail(tmp_path, monkeypatch):
    """The built-in overlap guard (DESIGN 5.2): every validate checks all
    output pairs with no authored entry — touching passes, material
    interpenetration fails — and the entry carries ``builtin: true``
    plus the spec-assigned human owner."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    # Stacked exactly: touching, no overlap.
    report = _run_asserts(_assert_asset(1.5))
    assert report["ok"] is True
    entry = _guard(report)
    assert entry["assert"] == "no_overlap" and entry["builtin"] is True
    assert entry["owner"] == "human" and entry["ok"] and entry["pairs"] == 1

    # Sunk 0.2 into the lower box: overlap volume 1*1*0.2.
    report = _run_asserts(_assert_asset(1.3))
    assert report["ok"] is False
    entry = _guard(report)
    assert not entry["ok"]
    pair = entry["detail"]["failing_pairs"][0]
    assert pair[:2] == ["lower", "upper"] and abs(pair[2] - 0.2) < 1e-6
    assert "overlap" in entry["fix"] and "embedded" in entry["fix"]

    from moraine3d.verify.validate import failures_only

    compact = failures_only(report)
    assert compact["asserts_total"] == 1 and compact["asserts_ok"] == 0
    assert compact["asserts"][0]["assert"] == "no_overlap"


def _flush_wall_asset(embed: float) -> str:
    """Two 4 m x 1 m slabs stacked with `embed` meters of interpenetration
    over the full 4 m^2 contact face — the interpenetrating wall-section
    class, guarded by the built-in overlap guard alone."""
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [4.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [4.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [0.0, 1.5 - embed, 0.0]}},
            ],
            "outputs": {"lower": "g1", "upper": "g2"},
        }
    )


def _run_asserts(asset_json: str) -> dict:
    sc = asset_mod.loads(asset_json)
    recs = evaluate_records(sc)
    return validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts,
    )


def test_overlap_guard_depth_passes_large_area_fp_slab(tmp_path, monkeypatch):
    """A large-area false positive (DESIGN §5.2): a sub-um slab over
    a large contact area carries a volume far above the old 1e-9 m^3 bar,
    but its penetration depth is below the depth criterion — it must pass.
    (2^-22 m: the smallest flush-contact slab float32 vertices can carry
    at this coordinate scale, ~2 ulps at y=1.5.)"""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_flush_wall_asset(2**-22))  # vol ~9.5e-7 m^3, depth ~2.4e-7 m
    assert _guard(report)["ok"] is True
    assert report["ok"] is True


def test_overlap_guard_depth_fails_real_interpenetration(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_flush_wall_asset(1e-3))  # a real 1 mm sink
    entry = _guard(report)
    assert entry["ok"] is False
    assert abs(entry["detail"]["failing_pairs"][0][3] - 1e-3) < 1e-4
    assert "deep" in entry["fix"]


def test_no_overlap_depth_default_is_float32_floor(tmp_path, monkeypatch):
    """The depth default is contact's float32 floor (DESIGN 5.2): a
    flush-contact fp slab a few ulps thick is micron-scale at
    tens-of-meters coordinates, so a sub-1e-5 depth must pass."""
    from moraine3d.core.asserts.common import DEPTH_BAR

    assert DEPTH_BAR == 1e-5
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_flush_wall_asset(4e-6))  # depth ~4e-6 m
    assert _guard(report)["ok"] is True


def _seated_plate_pair(i: int, dip: float):
    """Module box + a plate seated flush on its top face whose tab dips
    ``dip`` meters in: the intersection is the tab column plus, at some
    transforms, kernel-exact coplanar sheets along the seat plane."""
    import trimesh

    from moraine3d.verify.solids import to_manifold

    def M(tm):
        return to_manifold(tm.vertices, tm.faces)

    module = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
    plate = trimesh.creation.box(extents=(0.6, 0.05, 0.4))
    plate.apply_translation([0, 0.4 + 0.025, 0])
    tab = trimesh.creation.box(extents=(0.05, 0.02, 0.05))
    tab.apply_translation([0.1, 0.4 - dip + 0.01, 0.05])
    joined = (M(plate) + M(tab)).to_mesh()
    attach = trimesh.Trimesh(
        vertices=np.asarray(joined.vert_properties)[:, :3],
        faces=np.asarray(joined.tri_verts),
        process=False,
    )
    xf = trimesh.transformations.rotation_matrix(np.deg2rad(10 * i), [0, 1, 0])
    xf[:3, 3] = [i * 1.213, i * 0.00071, i * 0.577]
    a = module.copy()
    a.apply_transform(xf)
    b = attach.copy()
    b.apply_transform(xf)
    return M(a), M(b)


def test_no_overlap_depth_ignores_coplanar_residue():
    """The depth criterion measures the load-bearing surface (DESIGN 5.2):
    kernel-exact coplanar residue leaves the denominator, so identical
    intersection volumes read identical depths regardless of transform.
    Transform 12 fuses a ~0.3 m^2 seat sheet into the intersection's
    triangulation (40x depth dilution under a plain 2V/A)."""
    from moraine3d.verify.depth import overlap_exceeds

    depths = []
    for i in (0, 10, 12):
        ma, mb = _seated_plate_pair(i, dip=0.005)
        vol, depth, bad = overlap_exceeds(ma ^ mb)
        assert bad, "a 5 mm tab sink is a real interpenetration"
        assert abs(vol - 0.05 * 0.005 * 0.05) < 1e-8
        depths.append(depth)
    assert max(depths) / min(depths) < 1.01
    assert abs(depths[0] - 0.00417) < 2e-4  # 2V/A of the 5 mm tab column


def test_no_overlap_pure_coplanar_residue_reads_flush():
    """An intersection that is nothing but coplanar residue is flush
    contact (DESIGN 5.2): a cutter seated in the pocket it cut passes,
    whatever zero-volume sheets the kernel returns for it."""
    import trimesh

    from moraine3d.verify.depth import overlap_exceeds
    from moraine3d.verify.solids import to_manifold

    def M(tm):
        return to_manifold(tm.vertices, tm.faces)

    base = trimesh.creation.box(extents=(0.3, 0.1, 0.3))
    cutter = trimesh.creation.box(extents=(0.05, 0.04, 0.05))
    xf = trimesh.transformations.rotation_matrix(np.deg2rad(84), [0, 1, 0])
    xf[:3, 3] = [3.756, 0.0384, 1.32]
    b = base.copy()
    b.apply_transform(xf)
    c = cutter.copy()
    c.apply_translation([0.0, 0.03, 0.0])
    c.apply_transform(xf)
    vol, depth, bad = overlap_exceeds(M(c) ^ (M(b) - M(c)))
    assert not bad and depth == 0.0


def test_no_overlap_residue_cannot_hide_a_defect():
    """Residue dilutes a plain 2V/A below the depth bar, so a bar-scale
    defect next to a flush seat would pass silently; with the residue out
    of the denominator it fails (DESIGN 5.2)."""
    from moraine3d.core.asserts.common import DEPTH_BAR
    from moraine3d.verify.depth import overlap_exceeds

    ma, mb = _seated_plate_pair(12, dip=5e-5)
    vol, depth, bad = overlap_exceeds(ma ^ mb)
    assert vol < 2e-7  # tab column only: the sheets add no volume
    assert 2.0 * vol / 0.3 < DEPTH_BAR, "diluted mean sat under the bar"
    assert bad and depth > DEPTH_BAR


def test_overlap_guard_instanced_union(tmp_path, monkeypatch):
    """For cross-output pairs an instanced output counts as the union of
    its copies (guard semantics): a probe dipped into one copy fails the
    pair."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "a1", "op": "array", "inputs": ["g1"],
                 "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
                # Dips 0.25 into copy 2 of the row (x = 4, top at y = 1).
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [4.0, 1.0, 0.0]}},
            ],
            "outputs": {"row": "a1", "probe": "g2"},
        }
    )
    report = _run_asserts(asset)
    entry = _guard(report)
    assert not entry["ok"]
    pair = entry["detail"]["failing_pairs"][0]
    # 0.5^3 / 2 in the top half of the third copy: 0.5*0.5*0.25.
    assert pair[:2] == ["probe", "row"] and abs(pair[2] - 0.0625) < 1e-6


def _row_asset(step: float) -> str:
    """One instanced row of unit boxes, no authored asserts; step < 1.0
    interpenetrates copies."""
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "a1", "op": "array", "inputs": ["g1"],
                 "params": {"count": 3, "step_translate": [step, 0.0, 0.0]}},
            ],
            "outputs": {"row": "a1"},
        }
    )


def test_overlap_guard_checks_copy_pairs_by_default(tmp_path, monkeypatch):
    """The guard checks every copy pair of each instanced output with no
    opt-in (guard semantics): copy interpenetration the union view makes
    structurally invisible — an array step below the prototype width —
    fails loud, and a separated row passes with the checked count."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_row_asset(0.8))
    entry = _guard(report)
    assert entry["ok"] is False and report["ok"] is False
    assert entry["self_pairs"] == 3  # every copy pair counts as checked
    pairs = entry["detail"]["failing_pairs"]
    assert [p[:2] for p in pairs] == [["row[0]", "row[1]"], ["row[1]", "row[2]"]]
    assert abs(pairs[0][2] - 0.2) < 1e-6  # 0.2 m overlap of unit boxes
    assert "array step" in entry["fix"]
    assert "unchecked" not in entry

    report = _run_asserts(_row_asset(2.0))
    entry = _guard(report)
    assert entry["ok"] is True and report["ok"] is True
    assert entry["self_pairs"] == 3
    assert "unchecked" not in entry


def test_guard_self_pairs_disjoint_open_copies_pass(tmp_path, monkeypatch):
    """The copy-pass watertight policy mirrors cross pairs (guard
    semantics): a non-watertight prototype fails only when AABB overlap
    leaves copy pairs unresolved — disjoint copies pass without a
    solid."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "g1", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "a1", "op": "array", "inputs": ["g1"],
                 "params": {"count": 3, "step_translate": [5.0, 0.0, 0.0]}},
            ],
            "outputs": {"row": "a1"},
        }
    )
    report = _run_asserts(asset)
    entry = _guard(report)
    assert entry["ok"] is True and entry["self_pairs"] == 3


def test_guard_watertight_reason_reported_once(tmp_path, monkeypatch):
    """The cross-pair and copy-pair passes report the not-watertight
    defect with one sentence, deduped (guard semantics): the same
    reason must not repeat inside one entry's fix."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "g1", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "s1", "op": "transform", "inputs": ["g1"],
                 "params": {"rotate_deg": [20.0, 0.0, 0.0]}},  # a thick box
                {"id": "a1", "op": "array", "inputs": ["s1"],
                 "params": {"count": 3, "step_translate": [0.5, 0.0, 0.0]}},
                {"id": "b1", "op": "box", "params": {}},
                {"id": "t1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.5, 0.4, 0.0]}},
            ],
            "outputs": {"row": "a1", "cube": "t1"},
        }
    )
    report = _run_asserts(asset)
    entry = _guard(report)
    assert entry["ok"] is False
    assert entry["fix"].count("cannot be built as a solid") == 1


def test_guard_self_pairs_vacuous_on_mesh_outputs(tmp_path, monkeypatch):
    """The guard's copy pass over a scope with no instanced output is
    vacuous with self_pairs 0 (guard semantics)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_all_assert_asset(2.4))
    entry = _guard(report)
    assert entry["ok"] is True and entry["self_pairs"] == 0


def _all_assert_asset(x3: float) -> str:
    # Three unit boxes on the ground: a at x 0, b at x 1.2, c at x3
    # (2.4 clears everything; 1.7 sinks 0.5 into b). The guard covers
    # every pair.
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [1.2, 0.5, 0.0]}},
                {"id": "b3", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g3", "op": "transform", "inputs": ["b3"],
                 "params": {"translate": [x3, 0.5, 0.0]}},
            ],
            "outputs": {"a": "g1", "b": "g2", "c": "g3"},
        }
    )


# -- embedded: the positive form of the depth axis ---------------------------


def _wall_asset_with(embed: float, asserts: list[dict]) -> str:
    doc = json.loads(_flush_wall_asset(embed))
    doc["asserts"] = asserts
    return json.dumps(doc)


def _row_asset_with(step: float, asserts: list[dict]) -> str:
    doc = json.loads(_row_asset(step))
    doc["asserts"] = asserts
    return json.dumps(doc)


def test_embedded_pair_band_pass_and_report(tmp_path, monkeypatch):
    """embedded pair form (embedded semantics): the two outputs'
    intersection depth must lie in the declared band, and the entry
    always carries the resolved band and the measured depth — max: null
    is the visible spelling of a declared-unbounded embed."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_wall_asset_with(5e-3, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "min": 4e-3, "max": 6e-3, "owner": "human"},
    ]))
    entry = _entry(report, "seam")
    assert entry["ok"] is True and report["ok"] is True
    assert entry["min"] == 4e-3 and entry["max"] == 6e-3
    assert abs(entry["depth"] - 5e-3) < 5e-4
    assert entry["owner"] == "human"

    # Default min (the 1 mm separation quantum read from the inside),
    # unbounded max: a parameter-dependent embed needs no ceiling.
    report = _run_asserts(_wall_asset_with(5e-3, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "agent"},
    ]))
    entry = _entry(report, "seam")
    assert entry["ok"] is True
    assert entry["min"] == 1e-3 and entry["max"] is None


def test_embedded_pair_band_failures(tmp_path, monkeypatch):
    """embedded fails on both sides of the band, and on no overlap at
    all — the away direction no_overlap cannot see (embedded
    semantics)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = _entry(_run_asserts(_wall_asset_with(2e-4, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "human"},
    ])), "seam")
    assert entry["ok"] is False and "under the declared embed min" in entry["fix"]

    entry = _entry(_run_asserts(_wall_asset_with(1e-2, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "min": 4e-3, "max": 6e-3, "owner": "human"},
    ])), "seam")
    assert entry["ok"] is False and "over the declared embed max" in entry["fix"]

    # A gap: the parts separate entirely; asset stays red, not silent.
    entry = _entry(_run_asserts(_wall_asset_with(-0.05, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "human"},
    ])), "seam")
    assert entry["ok"] is False
    assert entry["depth"] == 0.0 and "under the declared embed min" in entry["fix"]


def test_embedded_pair_written_min_zero_is_the_free_spelling(tmp_path, monkeypatch):
    """A written pair `min` 0 is the explicit free-overlap spelling
    (DESIGN 5.2): a depth with no design value passes at any depth, no
    overlap included, measuring nothing (``measured: false``, as the
    ``self`` form's free declaration), the pair is still ceded from the
    guard and reported, a written ceiling still polices, while the
    omitted min keeps the 1 mm floor and its refusal names the free
    spelling."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    free = [{"id": "sink", "kind": "embedded", "outputs": ["lower", "upper"],
             "min": 0, "owner": "agent"}]
    # A seed-sized overlap below the default floor: green under min 0.
    report = _run_asserts(_wall_asset_with(2e-4, free))
    entry = _entry(report, "sink")
    assert entry["ok"] is True and report["ok"] is True
    assert entry["min"] == 0 and entry["max"] is None
    assert entry["measured"] is False and "depth" not in entry
    assert _guard(report)["ceded"] == {"pairs": [["lower", "upper"]]}
    # No overlap at all: free means free, not "must meet".
    entry = _entry(_run_asserts(_wall_asset_with(-0.05, free)), "sink")
    assert entry["ok"] is True and entry["measured"] is False
    # A ceiling still polices under the free floor.
    entry = _entry(_run_asserts(_wall_asset_with(1e-2, [
        dict(free[0], max=5e-3),
    ])), "sink")
    assert entry["ok"] is False and "over the declared embed max" in entry["fix"]
    # The default floor is unchanged, and its refusal names the spelling.
    entry = _entry(_run_asserts(_wall_asset_with(2e-4, [
        {"id": "sink", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "agent"},
    ])), "sink")
    assert entry["ok"] is False and entry["min"] == 1e-3
    assert "'min': 0" in entry["fix"]


def test_embedded_cedes_pair_from_guard(tmp_path, monkeypatch):
    """A declared embed cedes its pair from the built-in overlap guard:
    the embedded entry polices the pair under its own band and the guard
    reports what it ceded (guard semantics: ceded.pairs); without the
    declaration the same geometry fails the guard."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_wall_asset_with(5e-3, []))
    assert report["ok"] is False  # control: undeclared 5 mm overlap is red

    report = _run_asserts(_wall_asset_with(5e-3, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "min": 4e-3, "max": 6e-3, "owner": "agent"},
    ]))
    assert report["ok"] is True
    guard = _guard(report)
    assert guard["ok"] is True
    assert guard["ceded"] == {"pairs": [["lower", "upper"]]}
    assert guard["pairs"] == 0  # the asset's only pair went to the embed


def test_embedded_self_is_the_free_overlap_declaration(tmp_path, monkeypatch):
    """embedded `self` with both bounds omitted is the pure free-overlap
    declaration (embedded semantics): copies may interpenetrate, the
    guard cedes that output's copy pairs and reports it (ceded.self),
    and the entry measures nothing — a band admitting every depth reads
    no verdict from the kernel — reporting the copy pairs it would have
    measured under `measured: false` instead of a depth range."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_row_asset_with(0.8, [
        {"id": "weave", "kind": "embedded", "outputs": ["row"], "self": True,
         "owner": "agent"},
    ]))
    assert report["ok"] is True
    guard = _guard(report)
    weave = _entry(report, "weave")
    assert guard["ok"] is True and guard["ceded"] == {"self": ["row"]}
    assert guard["self_pairs"] == 0
    assert weave["ok"] is True
    assert weave["min"] == 0.0 and weave["max"] is None
    assert weave["self_pairs"] == 3 and weave["candidate_pairs"] == 2
    assert weave["measured"] is False
    assert "depth_range" not in weave and "overlapping" not in weave


def test_embedded_free_overlap_declaration_skips_the_kernel(tmp_path, monkeypatch):
    """The pure free-overlap declaration never builds a copy solid or
    measures an intersection (embedded semantics: the exact depth of
    every overlapping copy pair was the comparison's largest single
    validate cost, paid for a band that polices nothing); a written
    bound brings the kernel back."""
    from moraine3d.verify import depth

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    calls = []
    real = depth.intersection_depth
    monkeypatch.setattr(
        depth, "intersection_depth",
        lambda inter: calls.append(1) or real(inter),
    )
    free = {"id": "weave", "kind": "embedded", "outputs": ["row"], "self": True,
            "owner": "agent"}
    entry = _entry(_run_asserts(_row_asset_with(0.8, [dict(free)])), "weave")
    assert entry["ok"] is True and entry["candidate_pairs"] == 2
    assert calls == []
    entry = _entry(
        _run_asserts(_row_asset_with(0.8, [dict(free, max=0.5)])), "weave"
    )
    assert entry["ok"] is True and entry["overlapping"] == 2
    assert len(calls) == 2


def test_embedded_free_overlap_declaration_needs_no_solid(tmp_path, monkeypatch):
    """A free-overlap `self` declaration over a non-watertight prototype
    passes (embedded semantics): a claim that polices nothing cannot fail
    for want of a solid, where a written bound over the same overlapping
    open copies fails with the watertight reason."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def sheets(asserts):
        return json.dumps({
            "build_version": "fixture",
            "nodes": [
                {"id": "g1", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "s1", "op": "transform", "inputs": ["g1"],
                 "params": {"rotate_deg": [20.0, 0.0, 0.0]}},
                {"id": "a1", "op": "array", "inputs": ["s1"],
                 "params": {"count": 3, "step_translate": [0.5, 0.0, 0.0]}},
            ],
            "outputs": {"row": "a1"},
            "asserts": asserts,
        })

    free = {"id": "w", "kind": "embedded", "outputs": ["row"], "self": True,
            "owner": "agent"}
    entry = _entry(_run_asserts(sheets([dict(free)])), "w")
    assert entry["ok"] is True and entry["measured"] is False
    assert entry["candidate_pairs"] == 2
    entry = _entry(_run_asserts(sheets([dict(free, max=0.5)])), "w")
    assert entry["ok"] is False and "solid" in entry["fix"]


def test_embedded_pair_free_overlap_needs_no_solid(tmp_path, monkeypatch):
    """The pair form's free-overlap spelling (`min` 0, no `max`) measures
    nothing and needs no solid, as the `self` form's does (DESIGN 5.2):
    over two overlapping open sheets it passes under `measured: false`,
    where the same pair with a written ceiling fails with the
    watertight reason."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def sheets(asserts):
        return json.dumps({
            "build_version": "fixture",
            "nodes": [
                {"id": "g1", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "s1", "op": "transform", "inputs": ["g1"],
                 "params": {"rotate_deg": [20.0, 0.0, 0.0]}},
                {"id": "s2", "op": "transform", "inputs": ["s1"],
                 "params": {"translate": [0.5, 0.0, 0.0]}},
            ],
            "outputs": {"a": "s1", "b": "s2"},
            "asserts": asserts,
        })

    free = {"id": "w", "kind": "embedded", "outputs": ["a", "b"], "min": 0,
            "owner": "agent"}
    entry = _entry(_run_asserts(sheets([dict(free)])), "w")
    assert entry["ok"] is True and entry["measured"] is False
    assert "depth" not in entry
    entry = _entry(_run_asserts(sheets([dict(free, max=0.5)])), "w")
    assert entry["ok"] is False and "solid" in entry["fix"]


def test_embedded_self_band_max_fails_deep_copies(tmp_path, monkeypatch):
    """embedded `self` with a ceiling still catches copies overlapping
    beyond the declared band (embedded semantics: the declaration that
    punches the hole carries its own guard)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = _entry(_run_asserts(_row_asset_with(0.8, [
        {"id": "weave", "kind": "embedded", "outputs": ["row"], "self": True,
         "max": 0.1, "owner": "agent"},
    ])), "weave")
    assert entry["ok"] is False
    rows = entry["detail"]["failing_pairs"]
    assert [p[:2] for p in rows] == [["row[0]", "row[1]"], ["row[1]", "row[2]"]]
    assert "outside the declared band [0, 0.1] m" in entry["fix"]


def test_embedded_connected_chain_pass_and_split(tmp_path, monkeypatch):
    """embedded `connected` requires the copies to form one body through
    embeds >= min (embedded semantics): an embedding row passes; a
    separated row is as many bodies as copies — the away direction that
    was invisible to every overlap check."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    chain = {"id": "rail", "kind": "embedded", "outputs": ["row"],
             "connected": True, "min": 0.1, "max": 0.3, "owner": "human"}
    report = _run_asserts(_row_asset_with(0.8, [dict(chain)]))
    entry = _entry(report, "rail")
    assert entry["ok"] is True and "components" not in entry

    entry = _entry(_run_asserts(_row_asset_with(2.0, [dict(chain)])), "rail")
    assert entry["ok"] is False and entry["components"] == 3
    assert "form 3 bodies" in entry["fix"]
    assert entry["detail"]["components"] == [[1, 0], [1, 1], [1, 2]]


def test_embedded_connected_requires_embeds_not_contact(tmp_path, monkeypatch):
    """`connected` links copies through material embeds, never mere
    contact (embedded semantics): a flush-touching row is not one body —
    the guide's >= 1 mm embed rule is the floor for a reason."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = _entry(_run_asserts(_row_asset_with(1.0, [
        {"id": "rail", "kind": "embedded", "outputs": ["row"],
         "connected": True, "owner": "human"},
    ])), "rail")
    assert entry["ok"] is False and entry["components"] == 3


def test_embedded_connected_split_halves_fail():
    """Connectivity is strictly stronger than "no isolated copy" (DESIGN
    5.2): a chain broken into two halves leaves every copy with a
    neighbor and still fails."""
    from moraine3d.verify.depth import copy_components

    comps = copy_components(4, [(0, 1), (2, 3)])
    assert [g for g in comps] == [[0, 1], [2, 3]]
    assert len(copy_components(4, [(0, 1), (1, 2), (2, 3)])) == 1


def test_embedded_copy_forms_vacuous_on_mesh_output(tmp_path, monkeypatch):
    """embedded's copy forms are vacuous on a plain-mesh output — one
    body, no copy pairs — so the declaration is legal before the first
    array/scatter node exists (embedded semantics), mirroring blanket
    `self`."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    for quant in ({"self": True}, {"connected": True}):
        report = _run_asserts(_wall_asset_with(5e-3, [
            {"id": "w", "kind": "embedded", "outputs": ["lower"],
             **quant, "owner": "agent"},
        ]))
        entry = _entry(report, "w")
        assert entry["ok"] is True and entry["self_pairs"] == 0


def test_embedded_self_written_min_fails_separated_copies(tmp_path, monkeypatch):
    """A written positive `min` on embedded `self` is a positive claim
    (embedded semantics): copies that have all separated fail it — the
    away direction the guard cannot see — while both bounds omitted
    stays the vacuous free-overlap declaration."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    floored = {"id": "weave", "kind": "embedded", "outputs": ["row"],
               "self": True, "min": 0.1, "owner": "agent"}
    entry = _entry(_run_asserts(_row_asset_with(2.0, [dict(floored)])), "weave")
    assert entry["ok"] is False and entry["overlapping"] == 0
    assert "have separated" in entry["fix"]

    entry = _entry(_run_asserts(_row_asset_with(0.8, [dict(floored)])), "weave")
    assert entry["ok"] is True and entry["overlapping"] == 2

    free = {"id": "weave", "kind": "embedded", "outputs": ["row"],
            "self": True, "owner": "agent"}
    entry = _entry(_run_asserts(_row_asset_with(2.0, [free])), "weave")
    assert entry["ok"] is True and entry["candidate_pairs"] == 0
    assert entry["measured"] is False


def test_overlap_depth_is_the_plain_measure_on_a_clean_solid():
    """The depth measure (DESIGN 5.2) reads a residue-free solid's plain
    2V/A: no pass between the geometry and the number, so a band edge
    holds at equality."""
    import manifold3d

    from moraine3d.verify.depth import overlap_depth
    from moraine3d.verify.solids import manifold_measures

    slab = manifold3d.Manifold.cube([1.0, 1.0, 0.01])
    vol, area = manifold_measures(slab)
    assert abs(overlap_depth(slab, vol, area) - 2.0 * vol / area) < 1e-12


def test_overlap_depth_does_not_depend_on_where_or_how_large_the_solid_is():
    """The depth measure (DESIGN 5.2) is the same number for a solid at
    the origin and a hundred meters out, at half a meter and at a
    kilometer: no length in it is near the kernel's resolution, so no
    position or size reaches it."""
    import manifold3d

    from moraine3d.verify.depth import overlap_depth
    from moraine3d.verify.solids import manifold_measures

    box = manifold3d.Manifold.cube([0.4, 0.1, 0.1])
    ell = box + box.rotate([0, 0, 90]).translate([0.1, 0.0, 0.0])
    for shift, scale in [
        ([0.0, 0.0, 0.0], 1.0), ([106.8, 20.1, 123.5], 1.0),
        ([0.0, 0.0, 0.0], 100.0), ([0.0, 0.0, 0.0], 2000.0), ([400.0, 20.0, 450.0], 2000.0),
    ]:
        solid = ell.scale([scale, scale, scale]).translate(shift)
        vol, area = manifold_measures(solid)
        assert abs(overlap_depth(solid, vol, area) / (2.0 * vol / area) - 1.0) < 1e-9, (shift, scale)


def test_guard_thin_box_overlap_needs_no_solid(tmp_path, monkeypatch):
    """A pair whose box overlap is thinner than the depth bar is under
    the bar before any solid is built (guard semantics): a flush row of
    open sheets passes, its pairs counted, with no watertight demand."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "g1", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "a1", "op": "array", "inputs": ["g1"],
                 "params": {"count": 3, "step_translate": [1.0, 0.0, 0.0]}},
            ],
            "outputs": {"row": "a1"},
        }
    )
    entry = _guard(_run_asserts(asset))
    assert entry["ok"] is True and entry["self_pairs"] == 3


def test_assert_all_outputs_pass_and_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run_asserts(_all_assert_asset(2.4))
    entry = _guard(report)
    assert entry["outputs"] == "all" and entry["pairs"] == 3 and entry["ok"]
    assert report["ok"] is True

    report = _run_asserts(_all_assert_asset(1.7))
    entry = _guard(report)
    assert report["ok"] is False and not entry["ok"]
    (pair,) = entry["detail"]["failing_pairs"]
    assert pair[0] == "b" and pair[1] == "c" and abs(pair[2] - 0.5) < 1e-6
    assert "'b' and 'c' overlap" in entry["fix"]


def test_assert_all_open_surface(tmp_path, monkeypatch):
    """A non-watertight output fails the guard only when its bounds meet
    another output's thicker than the depth bar — a disjoint or
    thinner-than-the-bar box overlap is sound proof the pair is under
    the bar and needs no solid (guard semantics)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def asset(gx: float, tilt: float) -> str:
        return json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                    {"id": "g1", "op": "transform", "inputs": ["b1"],
                     "params": {"translate": [0.0, 0.5, 0.0]}},
                    {"id": "sheet", "op": "grid", "params": {"size": [1.0, 1.0]}},
                    {"id": "gs", "op": "transform", "inputs": ["sheet"],
                     "params": {"translate": [gx, 0.0, 0.0],
                                "rotate_deg": [tilt, 0.0, 0.0]}},
                ],
                "outputs": {"solid": "g1", "open": "gs"},
            }
        )

    for gx, tilt, expect_ok in ((5.0, 20.0, True), (0.2, 20.0, False)):
        report = _run_asserts(asset(gx, tilt))
        entry = _guard(report)
        assert entry["ok"] is expect_ok, gx
        if not expect_ok:
            assert "watertight" in entry["fix"]
    # A flat sheet lying on the cube's bottom plane: the box overlap is
    # zero-thick, so the pair is cleared without a solid.
    assert _guard(_run_asserts(asset(0.2, 0.0)))["ok"] is True


def _parse_asset() -> str:
    # The stacked-box geometry with one legal authored entry to mutate.
    return _dist_asset(1.3, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "human"},
    ])


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: d["asserts"].append(dict(d["asserts"][0])), "duplicate_assert_id"),
        (lambda d: d["asserts"][0].update(kind="no_gap"), "bad_assert_kind"),
        # The overlap guard is built in: an authored no_overlap entry is
        # refused with the redirect to `embedded`, and the guard's report
        # id is reserved.
        (lambda d: d["asserts"][0].update(kind="no_overlap", outputs="all"),
         "bad_assert_kind"),
        (lambda d: d["asserts"][0].update(id="no_overlap"), "bad_assert_id"),
        (lambda d: d["asserts"][0].update(outputs=["lower", "ghost"]), "bad_assert_outputs"),
        (lambda d: d["asserts"][0].update(kind="contact", outputs="some"),
         "bad_assert_outputs"),
        # `volume_tol` left the vocabulary with the authored no_overlap.
        (lambda d: d["asserts"][0].update(volume_tol=1e-9), "bad_assert"),
        # A band edge at or below the defect floor could never be policed.
        (lambda d: d["asserts"][0].update(min=5e-6), "bad_assert"),
        (lambda d: d["asserts"][0].update(max=1e-5), "bad_assert"),
        (lambda d: d["asserts"][0].update(owner="tool"), "bad_assert"),
        (lambda d: d["asserts"][0].pop("owner"), "bad_assert"),
        (lambda d: d["asserts"][0].update(note="hi"), "bad_assert"),
        (lambda d: d["asserts"][0].pop("id"), "bad_assert_id"),
        # Distance kinds take exactly two named outputs, never "all"
        # (DESIGN 5.2), and a zero clearance asserts nothing.
        (lambda d: d["asserts"][0].update(kind="contact", outputs="all"),
         "bad_assert_outputs"),
        (lambda d: d["asserts"][0].update(kind="clearance", outputs="all"),
         "bad_assert_outputs"),
        (lambda d: d["asserts"][0].update(kind="clearance", min=0),
         "bad_assert"),
        # 'self' never applies to contact, and when present must be JSON
        # true (absence is the off state — one spelling per meaning).
        (lambda d: d["asserts"][0].update(kind="contact", self=True),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="clearance", self=False),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="clearance", self="yes"),
         "bad_assert"),
        # The dimension kinds (DESIGN 5.2): one subject, one measure
        # token, one judgment — value (+ tol) or a band, never both or
        # neither; count is integers of copies and takes no tol.
        (lambda d: d["asserts"][0].update(kind="extent", outputs=["lower"], value=1.0),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(
            kind="extent", outputs=["lower"], measure="size_w", value=1.0), "bad_assert"),
        (lambda d: d["asserts"][0].update(
            kind="extent", outputs=["lower"], measure="size_x", value=1.0, min=0.5),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="extent", outputs=["lower"], measure="size_x"),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(
            kind="extent", outputs=["lower"], measure="size_x", tol=0.1, min=0.5),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(
            kind="extent", outputs=["lower"], measure="size_x", value=1.0, tol=-1),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="extent", measure="size_x", value=1.0),
         "bad_assert_outputs"),
        (lambda d: d["asserts"][0].update(
            kind="extent", outputs=["lower"], faces=["lower", "top"], measure="size_x",
            value=1.0), "bad_assert_outputs"),
        (lambda d: (d["asserts"][0].pop("outputs"), d["asserts"][0].update(
            kind="extent", faces=[["lower", "top"], ["upper", "top"]], measure="size_x",
            value=1.0)), "bad_assert_faces"),
        (lambda d: d["asserts"][0].update(kind="count", outputs=["lower"], value=2, tol=1),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="count", outputs=["lower"], value=2.5),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="count", outputs=["lower"], value=True),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="count", outputs=["lower"], min=2, max=2),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="count", outputs=["lower"], min=-1),
         "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="contact", measure="size_x"), "bad_assert"),
        (lambda d: d["asserts"][0].update(kind="clearance", value=1.0), "bad_assert"),
    ],
)
def test_assert_parse_rejects(mutate, code):
    doc = json.loads(_parse_asset())
    mutate(doc)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == code


def test_legacy_no_overlap_entry_gets_the_retirement_message_first():
    """An authored no_overlap entry is refused by the one message that
    states its fix — removal and `embedded` — ahead of the key and id
    gates (DESIGN 8.2: the shortest path to a fix), whatever legacy keys
    or id it carries."""
    doc = json.loads(_parse_asset())
    doc["asserts"][0] = {"id": "no_overlap", "kind": "no_overlap",
                         "outputs": "all", "volume_tol": 1e-6, "owner": "human"}
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_assert_kind"
    assert "built in" in str(exc.value) and "embedded" in str(exc.value)
    # The guard holds the requirement at owner human, so the refusal
    # says the deletion it forces needs no approval round (DESIGN 4.1).
    assert "owner 'human'" in str(exc.value) and "no approval" in str(exc.value)


def test_embedded_band_refusals_name_the_fix():
    """The band refusals point at the guard and the floor, never at a
    retired key (DESIGN 8.2)."""
    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(outputs=["lower"], connected=True, min=0)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    msg = str(exc.value)
    assert "positive embed" in msg and "tol" not in msg

    doc = json.loads(_parse_asset())
    doc["asserts"][0]["min"] = 5e-6
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "defect floor" in str(exc.value) and "1e-05" in str(exc.value)


@pytest.mark.parametrize(
    "keys,fragment",
    [
        # The pair form takes no copy quantifier: two outputs with
        # `self` or `connected` is refused toward the pair form.
        ({"self": True}, "take 1 output"),
        ({"connected": True}, "take 1 output"),
        # `connected` already includes `self`'s band; one of the two.
        ({"outputs": ["lower"], "self": True, "connected": True},
         "already includes"),
    ],
)
def test_embedded_form_refusals_name_the_other_form(keys, fragment):
    """DESIGN 5.2: embedded is a pair of subjects or one output's own
    copy pairs (`self`, `connected` adding one-body connectivity) —
    two outputs with a copy key, or `self` together with `connected`,
    is refused with the form to write instead."""
    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(keys)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_assert"
    assert fragment in str(exc.value)


def test_asserts_serialize_one_per_line():
    sc = asset_mod.loads(_parse_asset())
    text = asset_mod.dumps(sc)
    assert '"asserts": [' in text
    line = next(l for l in text.splitlines() if l.startswith('{"id": "seam"'))
    assert asset_mod.loads(text).asserts == sc.asserts
    assert json.loads(line)["kind"] == "embedded"


# -- distance asserts (contact / clearance) ---------------------------------


def _dist_asset(dy: float, entries: list[dict]) -> str:
    """The stacked-box geometry of `_assert_asset`, with caller-chosen
    assert entries: dy=1.5 is exactly touching, larger is a gap of
    dy - 1.5, smaller an interpenetration."""
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [0.0, dy, 0.0]}},
            ],
            "outputs": {"lower": "g1", "upper": "g2"},
            "asserts": entries,
        }
    )


def test_contact_flush_passes_gap_fails(tmp_path, monkeypatch):
    """contact: minimum surface distance <= tol, default contact.DEFAULT_TOL
    (DESIGN 5.2); a failing entry reports the true closest approach, not a
    bounds approximation."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
             "owner": "human"}
    report = _run_asserts(_dist_asset(1.5, [entry]))
    e = _entry(report, "t")
    assert e["ok"] is True and e["distance"] <= 1e-5  # the number rides a green entry too
    assert report["ok"] is True

    report = _run_asserts(_dist_asset(1.6, [entry]))
    e = _entry(report, "t")
    assert e["ok"] is False and report["ok"] is False
    assert abs(e["distance"] - 0.1) < 1e-4
    assert "0.1" in e["fix"] and "contact tol" in e["fix"]


def test_contact_guard_composition(tmp_path, monkeypatch):
    """Crossing surfaces count as distance 0, so contact plus the
    built-in overlap guard assert resting contact (DESIGN 5.2): flush
    passes both, an interpenetration passes contact but fails the
    guard, a gap the reverse."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entries = [
        {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
         "owner": "agent"},
    ]
    for dy, contact_ok, overlap_ok in ((1.5, True, True),
                                       (1.3, True, False),
                                       (1.7, False, True)):
        report = _run_asserts(_dist_asset(dy, entries))
        assert _entry(report, "t")["ok"] is contact_ok, dy
        assert _guard(report)["ok"] is overlap_ok, dy


def test_clearance_pass_fail_touch(tmp_path, monkeypatch):
    """clearance: minimum surface distance >= tol; touching or crossing
    surfaces are distance 0 and fail (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = {"id": "g", "kind": "clearance", "outputs": ["lower", "upper"],
             "min": 0.05, "owner": "human"}
    e = _entry(_run_asserts(_dist_asset(1.6, [entry])), "g")
    # Green, and the gap is still reported: the broad phase only proved
    # ">= tol", so the exact search runs once here instead of in the
    # reader's raised-tolerance probe (DESIGN 5.2).
    assert e["ok"] is True and abs(e["distance"] - 0.1) < 1e-6

    e = _entry(_run_asserts(_dist_asset(1.52, [entry])), "g")
    assert e["ok"] is False
    assert abs(e["distance"] - 0.02) < 1e-4
    assert "0.02" in e["fix"] and "clearance min" in e["fix"]

    e = _entry(_run_asserts(_dist_asset(1.5, [entry])), "g")
    assert e["ok"] is False and e["distance"] == 0.0
    assert "touch or interpenetrate" in e["fix"]


def test_clearance_states_its_gap():
    """The gap is a design value with no default (DESIGN 5.2): an entry
    without `min` is refused with the key to write, and `tol` and `max`
    are answered with what clearance takes."""
    base = {"id": "g", "kind": "clearance", "outputs": ["lower", "upper"]}
    for extra, needle in (
        ({}, "needs 'min'"),
        ({"min": 0}, "finite number > 0"),
        ({"tol": 0.05}, "as 'min'"),
        ({"min": 0.05, "max": 0.1}, "no upper bound"),
    ):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_dist_asset(1.6, [{**base, **extra}]))
        assert exc.value.code == "bad_assert" and needle in str(exc.value), extra


def test_distance_pierce_clamps_to_zero(tmp_path, monkeypatch):
    """Crossing surfaces count as distance 0 (DESIGN §5.2): a rod through
    a large face keeps every vertex-face and edge-edge feature distance
    large while the true surface distance is 0 — the transversal-pierce
    clamp must make clearance fail and contact pass."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "p1", "op": "box", "params": {"size": [4.0, 0.02, 4.0]}},
                {"id": "pl", "op": "transform", "inputs": ["p1"],
                 "params": {"translate": [0.0, 1.0, 0.0]}},
                {"id": "r1", "op": "box", "params": {"size": [0.02, 3.0, 0.02]}},
                {"id": "rd", "op": "transform", "inputs": ["r1"],
                 "params": {"translate": [0.0, 1.5, 0.0]}},
            ],
            "outputs": {"plate": "pl", "rod": "rd"},
            "asserts": [
                {"id": "gap", "kind": "clearance", "outputs": ["plate", "rod"],
                 "min": 0.05, "owner": "human"},
                {"id": "tch", "kind": "contact", "outputs": ["plate", "rod"],
                 "owner": "agent"},
            ],
        }
    )
    by_id = {e["assert"]: e for e in _run_asserts(asset)["asserts"]}
    assert by_id["gap"]["ok"] is False
    assert by_id["gap"]["distance"] == 0.0
    assert by_id["tch"]["ok"] is True


def test_distance_edge_edge_interior_closest(tmp_path, monkeypatch):
    """Two 45-degree ridges crossing in plan: the minimum is attained
    strictly between edge interiors (no vertex-face pair comes as close),
    pinning the edge-edge narrow-phase term."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    s2 = float(np.sqrt(2.0))
    gap = 0.25
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"rotate_deg": [45.0, 0.0, 0.0],
                            "translate": [0.0, s2 / 2, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"rotate_deg": [0.0, 0.0, 45.0],
                            "translate": [0.0, s2 + gap + s2 / 2, 0.0]}},
            ],
            "outputs": {"ridge_x": "g1", "ridge_z": "g2"},
            "asserts": [
                {"id": "tch", "kind": "contact",
                 "outputs": ["ridge_x", "ridge_z"], "owner": "agent"},
                {"id": "gap", "kind": "clearance",
                 "outputs": ["ridge_x", "ridge_z"], "min": 0.2,
                 "owner": "agent"},
            ],
        }
    )
    by_id = {e["assert"]: e for e in _run_asserts(asset)["asserts"]}
    assert by_id["gap"]["ok"] is True
    assert by_id["tch"]["ok"] is False
    assert abs(by_id["tch"]["distance"] - gap) < 1e-3


def test_contact_failure_reports_true_distance_not_aabb_gap(tmp_path, monkeypatch):
    """A 45-degree-rotated box beside an axis-aligned one: the AABB gap is
    0.1 m but the surfaces are 0.6/sqrt(2) m apart — the failing entry must
    report the true closest approach (the descent-seeded exact search),
    never a bounds figure."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    s2 = float(np.sqrt(2.0))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"rotate_deg": [0.0, 45.0, 0.0],
                            "translate": [0.6 + s2 / 2, 0.5, 1.0]}},
            ],
            "outputs": {"block": "g1", "diamond": "g2"},
            "asserts": [
                {"id": "tch", "kind": "contact", "outputs": ["block", "diamond"],
                 "owner": "agent"}
            ],
        }
    )
    e = _entry(_run_asserts(asset), "tch")
    assert e["ok"] is False
    assert abs(e["distance"] - 0.6 / s2) < 1e-3


def test_contact_flush_at_moderate_scale(tmp_path, monkeypatch):
    """The contact default is a float32-noise floor (DESIGN 5.2, revised
    1e-6 -> 1e-5): a sub-ulp authored gap at y ~ 50 m — indistinguishable
    from flush in float32 storage — must read as touching."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 50.2, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [0.0, 51.2 + 2e-6, 0.0]}},
            ],
            "outputs": {"lower": "g1", "upper": "g2"},
            "checks": {"lower": {"not_sunk": False}},
            "asserts": [
                {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
                 "owner": "agent"}
            ],
        }
    )
    assert _run_asserts(asset)["asserts"][0]["ok"] is True


def test_distance_kinds_work_on_open_surfaces(tmp_path, monkeypatch):
    """Distance kinds read raw world-space triangles — watertightness is
    not required (DESIGN 5.2), unlike no_overlap's solids."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "sheet", "op": "grid", "params": {"size": [1.0, 1.0]}},
                {"id": "gs", "op": "transform", "inputs": ["sheet"],
                 "params": {"translate": [0.0, 1.0, 0.0]}},
            ],
            "outputs": {"solid": "g1", "open": "gs"},
            "asserts": [
                {"id": "tch", "kind": "contact", "outputs": ["solid", "open"],
                 "owner": "agent"},
                {"id": "gap", "kind": "clearance", "outputs": ["solid", "open"],
                 "owner": "agent", "min": 0.001},
            ],
        }
    )
    by_id = {e["assert"]: e for e in _run_asserts(asset)["asserts"]}
    assert by_id["tch"]["ok"] is True
    assert by_id["gap"]["ok"] is False  # they touch


def test_distance_assert_image_unevaluatable(tmp_path, monkeypatch):
    """An unevaluatable assertion fails with the reason (DESIGN 5.2): a
    distance assert naming an image output cannot report ok."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "tx", "op": "noise_texture",
                 "params": {"seed": 1, "size": 16}},
            ],
            "outputs": {"a": "g1", "tex": "tx"},
            "asserts": [
                {"id": "t", "kind": "contact", "outputs": ["a", "tex"],
                 "owner": "agent"}
            ],
        }
    )
    e = _entry(_run_asserts(asset), "t")
    assert e["ok"] is False and "image" in e["fix"]


def test_distance_instanced_nearest_copy_not_first(tmp_path, monkeypatch):
    """An instanced output is measured as placed, all copies included: the
    per-copy culling must keep the copy realizing the minimum even when it
    is neither the first nor the nearest-by-prototype copy."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def asset(probe_y: float) -> str:
        return json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                    {"id": "g1", "op": "transform", "inputs": ["b1"],
                     "params": {"translate": [0.0, 0.5, 0.0]}},
                    {"id": "a1", "op": "array", "inputs": ["g1"],
                     "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
                    {"id": "b2", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
                    {"id": "g2", "op": "transform", "inputs": ["b2"],
                     "params": {"translate": [4.0, probe_y, 0.0]}},
                ],
                "outputs": {"row": "a1", "probe": "g2"},
                    "asserts": [
                    {"id": "t", "kind": "contact", "outputs": ["row", "probe"],
                     "owner": "agent"}
                ],
            }
        )

    # Probe resting on copy 3 (x = 4, top at y = 1): touching.
    assert _entry(_run_asserts(asset(1.25)), "t")["ok"] is True
    # Probe 0.05 above copy 3: the reported distance is to that copy.
    e = _entry(_run_asserts(asset(1.30)), "t")
    assert e["ok"] is False
    assert abs(e["distance"] - 0.05) < 1e-4


def test_clearance_blind_to_full_containment(tmp_path, monkeypatch):
    """A fully contained output is invisible to surface distance — the
    surface-to-surface gap is genuinely positive — so clearance passes and
    material separation stays the overlap guard's domain (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [2.0, 2.0, 2.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 1.0, 0.0]}},
                {"id": "b2", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [0.0, 1.0, 0.0]}},
            ],
            "outputs": {"shell": "g1", "core": "g2"},
            "checks": {"core": {"not_enclosed": False}},
            "asserts": [
                {"id": "gap", "kind": "clearance", "outputs": ["shell", "core"],
                 "min": 0.5, "owner": "agent"},
            ],
        }
    )
    report = _run_asserts(asset)
    assert _entry(report, "gap")["ok"] is True  # surface gap 0.75 >= 0.5
    assert _guard(report)["ok"] is False  # the guard sees the containment


def test_distance_tol_boundary_inclusive(tmp_path, monkeypatch):
    """Both distance criteria are inclusive (DESIGN 5.2): a gap exactly at
    tol satisfies contact's <= and clearance's >= at once. 2^-10 m keeps
    the constructed gap binary-exact through float32 storage."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    gap = 2.0**-10
    entries = [
        {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
         "tol": gap, "owner": "agent"},
        {"id": "g", "kind": "clearance", "outputs": ["lower", "upper"],
         "min": gap, "owner": "agent"},
    ]
    by_id = {e["assert"]: e for e in
             _run_asserts(_dist_asset(1.5 + gap, entries))["asserts"]}
    assert by_id["t"]["ok"] is True
    assert by_id["g"]["ok"] is True


def test_distance_asserts_deterministic(tmp_path, monkeypatch):
    """Determinism (DESIGN 4.4): the same asset evaluates to bit-identical
    assert entries, floats included."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    entry = {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
             "owner": "agent"}
    first = _run_asserts(_dist_asset(1.6, [entry]))["asserts"]
    second = _run_asserts(_dist_asset(1.6, [entry]))["asserts"]
    assert first == second


def test_distance_failures_only_mixed_kinds(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    from moraine3d.verify.validate import failures_only

    entries = [
        {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
         "owner": "agent"},
        {"id": "g", "kind": "clearance", "outputs": ["lower", "upper"],
         "min": 0.2, "owner": "agent"},
    ]
    compact = failures_only(_run_asserts(_dist_asset(1.5, entries)))
    # The built-in guard rides every report: 3 entries, 1 failing.
    assert compact["asserts_total"] == 3 and compact["asserts_ok"] == 2
    assert compact["asserts"][0]["assert"] == "g"


def test_distance_default_constants_pinned():
    """The defaults are decisions, not implementation details: contact is
    the tool-owned float32-noise floor, and clearance has none (DESIGN
    5.2)."""
    from moraine3d.core.asserts import clearance, contact

    assert contact.DEFAULT_TOL == 1e-5
    assert not hasattr(clearance, "DEFAULT_TOL")


def test_tol_keys_are_unit_stable_per_kind():
    """`tol`, `min`, `max`, and `value` carry the kind's own measure —
    meters on every length kind, copies on count — never a unit that
    depends on which other keys are present (DESIGN 5.2): value errors
    name the unit, and a misplaced key's refusal names the right key."""
    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(kind="contact", tol=-1)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "tolerance, m)" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(kind="extent", outputs=["lower"], measure="size_x",
                             value=1.0, tol=-1)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "around 'value', m)" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(kind="count", outputs=["lower"], min=-1)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "(copies)" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(kind="count", outputs=["lower"], value=2, tol=1)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "no tolerance on a count" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(min=-1)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "penetration-depth band, m" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(tol=0.01)  # embedded takes the band, not tol
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "'min'/'max'" in str(exc.value)

    doc = json.loads(_parse_asset())
    doc["asserts"][0].update(kind="contact", min=0.001)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert "bound for clearance" in str(exc.value) and "takes 'tol'" in str(exc.value)


def test_assert_owner_required_never_defaulted():
    """Ownership is explicit (DESIGN 5.2): an entry without `owner` is a
    parse error naming both owners — a default of agent would silently
    make a forgotten human requirement agent-editable."""
    doc = json.loads(_parse_asset())
    del doc["asserts"][0]["owner"]
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_assert"
    msg = str(exc.value)
    assert "owner" in msg and "human" in msg and "agent" in msg


def test_distance_asserts_serialize_round_trip():
    entries = [
        {"id": "t", "kind": "contact", "outputs": ["lower", "upper"],
         "tol": 0.001, "owner": "human"}
    ]
    sc = asset_mod.loads(_dist_asset(1.5, entries))
    text = asset_mod.dumps(sc)
    line = next(l for l in text.splitlines() if l.startswith('{"id": "t"'))
    assert json.loads(line)["kind"] == "contact"
    assert asset_mod.loads(text).asserts == sc.asserts


# -- distance kernel unit tests ----------------------------------------------


def test_embedded_fix_advice_respects_owner(tmp_path, monkeypatch):
    """DESIGN 5.2: a declared embed is the hole in the human-owned overlap
    guard, so a failing human-owned embedded entry says fix the geometry
    and never steers toward widening the band; band advice appears only
    on agent-owned entries — on every embedded fix path."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def pair_fix(embed, owner, **band):
        entry = {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
                 "owner": owner, **band}
        return _entry(_run_asserts(_wall_asset_with(embed, [entry])), "seam")["fix"]

    def row_fix(step, owner, **band):
        entry = {"id": "weave", "kind": "embedded", "outputs": ["row"],
                 "self": True, "owner": owner, **band}
        return _entry(_run_asserts(_row_asset_with(step, [entry])), "weave")["fix"]

    # Pair form, under the band.
    human = pair_fix(2e-4, "human")
    assert "fix the geometry, not the band" in human
    assert "'min': 0" not in human and "drop the embedded" not in human
    assert "'min': 0" in pair_fix(2e-4, "agent")
    # Pair form, over the band.
    human = pair_fix(1e-2, "human", min=4e-3, max=6e-3)
    assert "fix the geometry, not the band" in human and "raise the band" not in human
    assert "raise the band" in pair_fix(1e-2, "agent", min=4e-3, max=6e-3)
    # Copy form, written floor with separated copies.
    human = row_fix(2.0, "human", min=0.1)
    assert "fix the geometry, not the band" in human and "drop 'min'" not in human
    assert "drop 'min'" in row_fix(2.0, "agent", min=0.1)
    # Copy form, a pair outside the band.
    human = row_fix(0.8, "human", max=0.1)
    assert "fix the geometry, not the band" in human and "adjust the band" not in human
    assert "adjust the band" in row_fix(0.8, "agent", max=0.1)


def _tri_pairs_dist(tri_a, tri_b):
    from moraine3d.verify.distance import tri_pair_dist

    A = np.asarray([tri_a], dtype=np.float64)
    B = np.asarray([tri_b], dtype=np.float64)
    return float(tri_pair_dist(A, B)[0])


def test_kernel_degenerate_triangles_finite():
    """A degenerate triangle still yields its true, finite distance — a
    repeated-vertex sliver has edges and vertices even without an
    interior, and a fully collapsed one is a point."""
    sliver = [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    d = _tri_pairs_dist(sliver, tri)
    assert np.isfinite(d) and abs(d - 1.0) < 1e-12

    point = [[2.0, 0.0, 0.0]] * 3
    d = _tri_pairs_dist(point, tri)
    assert np.isfinite(d) and abs(d - 1.0) < 1e-12


def test_kernel_parallel_edges():
    """Exactly parallel and near-parallel edge pairs report the true
    separation; the apexes point away so no other feature comes closer."""
    a = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, -5.0, 0.0]]
    b = [[0.0, 0.5, 0.0], [10.0, 0.5, 0.0], [5.0, 5.5, 0.0]]
    d = _tri_pairs_dist(a, b)
    assert np.isfinite(d) and abs(d - 0.5) < 1e-12
    # And a near-parallel skew pair (1e-9 rad) stays stable.
    b_skew = [[0.0, 0.5, 0.0], [10.0, 0.5 + 1e-8, 0.0], [5.0, 5.5, 0.0]]
    d = _tri_pairs_dist(a, b_skew)
    assert np.isfinite(d) and abs(d - 0.5) < 1e-6


def test_kernel_needle_pierce():
    """The pierce clamp at triangle level: a needle crossing a big face has
    every feature distance large and true distance 0."""
    needle = [[0.05, -1.0, 0.0], [-0.05, 1.0, 0.0], [0.0, 1.0, 0.05]]
    face = [[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [0.0, 0.0, 5.0]]
    assert _tri_pairs_dist(needle, face) == 0.0


def test_kernel_cell_cap_survives_giant_triangle():
    """Broad-phase soundness holds under extreme extent ratios: the
    registration cap must degrade the grid, never silently break it — a
    touching pair beside a ~1e6 m triangle is still found (int64 span
    products would wrap and report the pair as clear)."""
    from moraine3d.verify.distance import min_candidate_distance

    giant = np.array(
        [[[-1e6, 0.0, -1e6], [1e6, 0.0, -1e6], [0.0, 0.0, 1e6]]]
    )
    speck = np.array([[[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 1e-3]]])
    # 51 mm-scale triangles far from everything pin the median extent low.
    chaff = speck + np.array([100.0, 50.0, 100.0]) + np.arange(51)[:, None, None]
    ta = np.concatenate([giant, chaff])
    d = min_candidate_distance(ta, speck, 1e-5)
    assert d is not None and d <= 1e-9


def test_distance_fix_advice_respects_owner(tmp_path, monkeypatch):
    """The error channel must not advise weakening a human-owned assertion
    (DESIGN 5.2: a failing human-owned assertion means fix the geometry,
    not the constraint); tol advice appears only on agent-owned entries."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def fix_for(kind, dy, owner, **extra):
        entry = {"id": "x", "kind": kind, "outputs": ["lower", "upper"],
                 "owner": owner, **extra}
        return _entry(_run_asserts(_dist_asset(dy, [entry])), "x")["fix"]

    human = fix_for("clearance", 1.52, "human", min=0.05)
    assert "lower 'min'" not in human and "fix the geometry, not the 'min'" in human
    agent = fix_for("clearance", 1.52, "agent", min=0.05)
    assert "lower 'min'" in agent

    human = fix_for("contact", 1.6, "human")
    assert "explicit tol" not in human and "fix the geometry" in human
    agent = fix_for("contact", 1.6, "agent")
    assert "explicit tol" in agent


def test_check_asserts_unknown_kind_is_loud():
    """A kind that parses (future vocabulary) but has no evaluation branch
    must raise, never fall through to another kind's semantics — the
    silent-pick class the vocabulary exists to prevent (DESIGN 5.2)."""
    from moraine3d.verify.validate import check_asserts

    values = {"a": _box([1.0, 1.0, 1.0]), "b": _box([1.0, 1.0, 1.0])}
    with pytest.raises(ValueError, match="unhandled assert kind"):
        check_asserts(
            values,
            [{"id": "x", "kind": "symmetry", "outputs": ["a", "b"],
              "owner": "agent"}],
        )


def test_kernel_grid_matches_bruteforce():
    """Broad-phase soundness: on seeded random soups the grid join must
    reproduce the full cross-product minimum whenever that minimum is
    within the inflation — including at a 100 m coordinate offset
    (float64 kernel; DESIGN 5.2)."""
    from moraine3d.verify.distance import min_candidate_distance, tri_pair_dist

    rng = np.random.default_rng(7)
    for offset in (0.0, 100.0):
        base_a = rng.uniform(0.0, 1.0, (40, 3)) + offset
        base_b = rng.uniform(0.3, 1.3, (40, 3)) + offset
        ta = base_a[:, None, :] + rng.uniform(-0.15, 0.15, (40, 3, 3))
        tb = base_b[:, None, :] + rng.uniform(-0.15, 0.15, (40, 3, 3))
        ia, ib = np.repeat(np.arange(40), 40), np.tile(np.arange(40), 40)
        brute = float(tri_pair_dist(ta[ia], tb[ib]).min())
        inflate = max(brute, 0.05)
        got = min_candidate_distance(ta, tb, inflate)
        assert got is not None and abs(got - brute) < 1e-12, offset


# -- query CLI ---------------------------------------------------------------


def test_query_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "stack.mora").write_text(TAGGED_ASSET, encoding="utf-8")
    flatten.write_project(proj, proj)

    rc = main(["query", str(proj), "--select", '{"center_y": {"gt": 1.0}}'])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # The base ends at y=1.0; only the placed crate has faces above it.
    assert out["outputs"]["base"]["matched"] == 0
    crate = out["outputs"]["crate"]
    assert 0 < crate["matched"] <= 12
    assert crate["bounds"]["min"][1] >= 1.0 - 1e-6

    rc = main(["query", str(proj), "--select", "{bad json"])
    assert rc == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "bad_predicate"

    rc = main(
        ["query", str(proj), "--select", '{"height": {"gt": 0}}']
    )
    assert rc == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "bad_predicate"
    assert "known fields" in err["error"]["message"]


def test_degenerate_faces_never_match_normal_tests():
    """Degenerate faces get a zero normal and simply never match a
    normal test (predicates module contract): a zero normal must not
    satisfy le/lt/eq-0 comparators and silently join selections."""
    m = _box([1.0, 1.0, 1.0])
    nv = len(m.vertices)
    v = np.vstack(
        [m.vertices, [[0.0, 2.0, 0.0], [0.5, 2.0, 0.0], [1.0, 2.0, 0.0]]]
    ).astype(np.float32)
    f = np.vstack([m.faces, [[nv, nv + 1, nv + 2]]]).astype(np.int32)
    m2 = Mesh(vertices=v, faces=f)  # box + one zero-area sliver
    down = predicates.evaluate_predicate(m2, {"normal_y": {"le": 0.0}})
    assert not down[-1]  # the sliver joins no normal selection
    assert down.sum() == 10  # bottom 2 + sides 8, sliver excluded
    # Non-normal fields still see it: area 0 is a real value.
    assert predicates.evaluate_predicate(m2, {"area": {"eq": 0.0}})[-1]


# -- dimension asserts (extent / count) ---------------------------------------


def _dim(entries: list[dict]) -> dict:
    # The stacked boxes at dy=2.0: lower spans y 0..1, upper y 1.5..2.5,
    # both 1 m wide in x and z.
    return _run_asserts(_dist_asset(2.0, entries))


def test_extent_value_pass_and_fail_report_measured(tmp_path, monkeypatch):
    """extent: one measure of the subject's world-space bounds must meet
    `value` within `tol`, and `measured` rides the entry pass or fail
    beside the requirement it echoes (DESIGN 5.2); the fix text follows
    the owner rule."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    rep = _dim([
        {"id": "top", "kind": "extent", "outputs": ["upper"], "measure": "max_y",
         "value": 2.5, "owner": "human"},
        {"id": "wide", "kind": "extent", "outputs": ["upper"], "measure": "size_x",
         "value": 1.2, "owner": "agent"},
        {"id": "deep", "kind": "extent", "outputs": ["lower"], "measure": "size_z",
         "value": 1.2, "owner": "human"},
    ])
    top = _entry(rep, "top")
    assert top["ok"] is True and top["measure"] == "max_y"
    assert top["value"] == 2.5 and top["tol"] == vocab.extent.DEFAULT_TOL
    assert top["measured"] == pytest.approx(2.5, abs=1e-6)
    wide = _entry(rep, "wide")
    assert wide["ok"] is False and wide["measured"] == pytest.approx(1.0, abs=1e-6)
    assert "size_x measures 1 m against value 1.2 m +/- 0.001 m (-0.2 m off)" in wide["fix"]
    assert "state the intended dimension" in wide["fix"]
    deep = _entry(rep, "deep")
    assert deep["ok"] is False and "not the number" in deep["fix"]
    assert "state the intended" not in deep["fix"]
    assert rep["ok"] is False


def test_extent_band_edges_inclusive_and_omitted_edge_visible(tmp_path, monkeypatch):
    """A `min`/`max` band needs one edge, both edges are inclusive, and
    the omitted edge is reported as null — declared-unbounded made
    visible, as embedded's `max: null` is (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    rep = _dim([
        {"id": "lo", "kind": "extent", "outputs": ["upper"], "measure": "min_y",
         "min": 1.5, "owner": "agent"},
        {"id": "hi", "kind": "extent", "outputs": ["upper"], "measure": "center_y",
         "max": 2.0, "owner": "agent"},
        {"id": "band", "kind": "extent", "outputs": ["lower"], "measure": "size_z",
         "min": 0.9, "max": 0.99, "owner": "human"},
    ])
    lo = _entry(rep, "lo")
    assert lo["ok"] is True and lo["min"] == 1.5 and lo["max"] is None
    hi = _entry(rep, "hi")
    assert hi["ok"] is True and hi["min"] is None and hi["max"] == 2.0
    assert hi["measured"] == pytest.approx(2.0, abs=1e-6)
    band = _entry(rep, "band")
    assert band["ok"] is False
    assert "outside the band [0.9, 0.99] m" in band["fix"] and "not the number" in band["fix"]


def test_extent_instanced_union_measures_actual_vertices():
    """The output form measures the union of the copies on actual
    transformed vertices, never on transformed box corners (DESIGN 5.2):
    a tilted curved prototype reads true where AABB-corner bounds dip."""
    from moraine3d.ops import array as array_op
    from moraine3d.ops import cylinder, transform

    proto = cylinder.evaluate([], {"radius": 0.5, "height": 2.0, "segments": 64}, None)
    lifted = transform.evaluate([proto], {"translate": [0.0, 1.0, 0.0]}, None)
    inst = array_op.evaluate([lifted], {"count": 2, "step_translate": [3.0, 0.0, 0.0]}, None)
    tilted = transform.evaluate([inst], {"rotate_deg": [8.0, 0.0, 8.0]}, None)
    v = np.c_[lifted.vertices.astype(np.float64), np.ones(len(lifted.vertices))]
    world = np.concatenate([v @ m.T for m in tilted.transforms])[:, :3]
    true_min = float(world[:, 1].min())
    span = float(world[:, 0].max() - world[:, 0].min())
    assert float(tilted.bounds()[0][1]) < true_min - 0.01  # the corner gap is real
    rep = validate_outputs({"t": tilted}, asserts=[
        {"id": "low", "kind": "extent", "outputs": ["t"], "measure": "min_y",
         "value": true_min, "tol": 1e-6, "owner": "agent"},
        {"id": "span", "kind": "extent", "outputs": ["t"], "measure": "size_x",
         "value": span, "tol": 1e-6, "owner": "agent"},
    ])
    assert _entry(rep, "low")["ok"] is True and _entry(rep, "span")["ok"] is True


def test_extent_face_set_form_and_refusals():
    """The face form reads one tagged face set on one copy (DESIGN 5.2):
    a copy index is required on an instanced output and refused on a
    plain mesh — a copy is never guessed, mated's rule — and an unknown
    tag fails naming the tags available; no `measured` on a refusal."""
    from moraine3d.ops import array as array_op
    from moraine3d.ops import transform

    top = _tag(_box([1.0, 1.0, 1.0]), "top")
    lid = _tag(transform.evaluate([top], {"translate": [0.0, 3.0, 0.0]}, None), "top")
    row = array_op.evaluate([top], {"count": 2, "step_translate": [2.0, 0.0, 0.0]}, None)
    rep = validate_outputs({"lid": lid, "row": row}, asserts=[
        {"id": "plain", "kind": "extent", "faces": ["lid", "top"], "measure": "min_y",
         "value": 3.5, "owner": "agent"},
        {"id": "copy1", "kind": "extent", "faces": ["row", "top", 1],
         "measure": "center_x", "value": 2.0, "owner": "agent"},
        {"id": "noidx", "kind": "extent", "faces": ["row", "top"], "measure": "size_x",
         "value": 1.0, "owner": "agent"},
        {"id": "notag", "kind": "extent", "faces": ["lid", "lip"], "measure": "size_x",
         "value": 1.0, "owner": "agent"},
        {"id": "badidx", "kind": "extent", "faces": ["lid", "top", 0],
         "measure": "size_x", "value": 1.0, "owner": "agent"},
    ])
    plain = _entry(rep, "plain")
    assert plain["ok"] is True and plain["faces"] == ["lid", "top"]
    assert plain["measured"] == pytest.approx(3.5, abs=1e-6)
    copy1 = _entry(rep, "copy1")
    assert copy1["ok"] is True and copy1["measured"] == pytest.approx(2.0, abs=1e-6)
    for aid, words in (
        ("noidx", "add the copy index"),
        ("notag", "available tags: ['top']"),
        ("badidx", "drop the copy index"),
    ):
        e = _entry(rep, aid)
        assert e["ok"] is False and words in e["fix"] and "measured" not in e


def test_count_value_band_and_plain_mesh(tmp_path, monkeypatch):
    """count: an output's copy count against `value` or an integer
    `min`/`max` band, a plain mesh counting one (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    rep = _run_asserts(_row_asset_with(1.5, [
        {"id": "three", "kind": "count", "outputs": ["row"], "value": 3, "owner": "human"},
        {"id": "two", "kind": "count", "outputs": ["row"], "value": 2, "owner": "human"},
        {"id": "least", "kind": "count", "outputs": ["row"], "min": 3, "owner": "agent"},
        {"id": "most", "kind": "count", "outputs": ["row"], "max": 2, "owner": "agent"},
    ]))
    three = _entry(rep, "three")
    assert three["ok"] is True and three["measured"] == 3 and three["value"] == 3
    two = _entry(rep, "two")
    assert two["ok"] is False
    assert "has 3 copies against value 2 (+1 off)" in two["fix"] and "not the number" in two["fix"]
    least = _entry(rep, "least")
    assert least["ok"] is True and least["min"] == 3 and least["max"] is None
    most = _entry(rep, "most")
    assert most["ok"] is False and "above max 2" in most["fix"]
    assert "state the intended count" in most["fix"]
    one = _entry(_dim([
        {"id": "one", "kind": "count", "outputs": ["lower"], "value": 1, "owner": "agent"},
    ]), "one")
    assert one["ok"] is True and one["measured"] == 1


def test_dimension_asserts_image_unevaluatable(tmp_path, monkeypatch):
    """An image subject is unevaluatable for both dimension kinds and
    fails with the reason — never a silent pass (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
                {"id": "g1", "op": "transform", "inputs": ["b1"],
                 "params": {"translate": [0.0, 0.5, 0.0]}},
                {"id": "tx", "op": "noise_texture",
                 "params": {"seed": 1, "size": 16}},
            ],
            "outputs": {"a": "g1", "tex": "tx"},
            "asserts": [
                {"id": "e", "kind": "extent", "outputs": ["tex"], "measure": "size_x",
                 "value": 1.0, "owner": "agent"},
                {"id": "c", "kind": "count", "outputs": ["tex"], "value": 1,
                 "owner": "agent"},
            ],
        }
    )
    rep = _run_asserts(asset)
    for aid in ("e", "c"):
        e = _entry(rep, aid)
        assert e["ok"] is False and "image" in e["fix"] and "measured" not in e


def test_dimension_asserts_failures_only(tmp_path, monkeypatch):
    """--failures-only drops a green dimension entry and keeps a failing
    one with its `measured` (DESIGN 8.2 compression, 5.2 numbers first)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    from moraine3d.verify.validate import failures_only

    compact = failures_only(_dim([
        {"id": "top", "kind": "extent", "outputs": ["upper"], "measure": "max_y",
         "value": 2.5, "owner": "human"},
        {"id": "n", "kind": "count", "outputs": ["upper"], "value": 2, "owner": "agent"},
    ]))
    assert compact["asserts_total"] == 3 and compact["asserts_ok"] == 2
    (entry,) = compact["asserts"]
    assert entry["assert"] == "n" and entry["measured"] == 1


def test_dimension_asserts_serialize_and_leave_older_entries_intact():
    """`measure` follows the subject and `value` precedes the band in the
    canonical entry line (DESIGN 4.1 fixed key order), and an entry
    written before the dimension kinds serializes byte-identically — the
    key-order extension is not a schema change."""
    sc = asset_mod.loads(_dist_asset(2.0, [
        {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
         "owner": "human"},
        {"id": "top", "kind": "extent", "faces": ["upper", "cap", 0],
         "measure": "max_y", "value": 2.5, "tol": 0.01, "owner": "human"},
        {"id": "n", "kind": "count", "outputs": ["upper"], "min": 1, "max": 4,
         "owner": "agent"},
    ]))
    lines = asset_mod.dumps(sc).splitlines()
    assert '{"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"], "owner": "human"},' in lines
    assert (
        '{"id": "top", "kind": "extent", "faces": ["upper", "cap", 0], "measure": "max_y", '
        '"value": 2.5, "tol": 0.01, "owner": "human"},'
    ) in lines
    assert '{"id": "n", "kind": "count", "outputs": ["upper"], "min": 1, "max": 4, "owner": "agent"}' in lines
    assert asset_mod.loads("\n".join(lines)).asserts == sc.asserts
    assert vocab.output_names(sc.asserts[1]) == {"upper"}
    assert vocab.output_names(sc.asserts[2]) == {"upper"}


def test_extent_vocabulary_pinned():
    """extent's `value` allowance defaults to the 1 mm separation quantum
    — an author-owned design allowance, not a numerics floor — and its
    `measure` tokens are the predicate-field spelling read along an axis:
    size/min/max/center × x/y/z (DESIGN 5.2)."""
    assert vocab.extent.DEFAULT_TOL == 1e-3
    assert {"extent", "count"} <= vocab.AUTHORABLE_KINDS
    assert list(vocab.extent.MEASURES) == [
        f"{what}_{axis}"
        for what in ("size", "min", "max", "center")
        for axis in "xyz"
    ]
    order = vocab.KEY_ORDER
    assert order.index("measure") == order.index("faces") + 1
    assert order.index("value") == order.index("min") - 1


def test_query_reads_a_tags_faces_and_their_centroid(tmp_path, monkeypatch, capsys):
    """DESIGN 6: `query --group` answers for the faces a tag caught,
    alone or narrowed by `--select`, and every answer holds the
    matched faces' area-weighted `centroid`, the origin `place`
    anchors on (DESIGN 5.1). An output without the tag is skipped with
    what it carries, and the others still answer."""
    from support.recipes import write_files

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = (
        'slab = mo.transform(mo.box(id = "slab_box", size = [2.0, 0.2, 1.0]),'
        ' id = "slab_up", translate = [0.5, 0.1, 0])\n'
        'slab = mo.tag(slab, id = "slab", name = "top", select = {"normal_y": {"gt": 0.9}})\n'
        'mo.output("slab", slab)\n'
        'mo.output("post", mo.transform(mo.box(id = "post_box", size = [0.1, 1, 0.1]),'
        ' id = "post", translate = [3, 0.5, 0]))\n'
    )
    project = write_files(tmp_path / "proj", {"asset.star": asset})
    assert main(["query", str(project), "--tag", "top"]) == 0
    out = json.loads(capsys.readouterr().out)["outputs"]
    assert out["slab"]["matched"] == 2 and out["slab"]["area"] == 2.0
    assert out["slab"]["centroid"] == [0.5, 0.2, 0.0]
    assert out["post"]["skipped"].startswith("no tag 'top'; it carries no tags")

    assert main(["query", str(project), "--tag", "top",
                     "--select", '{"center_x": {"gt": 100}}']) == 0
    narrowed = json.loads(capsys.readouterr().out)["outputs"]["slab"]
    assert narrowed["matched"] == 0 and "centroid" not in narrowed

    assert main(["query", str(project), "--select", '{"normal_y": {"gt": 0.9}}',
                     "--output", "post"]) == 0
    post = json.loads(capsys.readouterr().out)["outputs"]["post"]
    assert post["centroid"] == [3.0, 1.0, 0.0]
    assert main(["query", str(project)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_predicate"


def test_the_queried_centroid_is_where_place_anchors(tmp_path, monkeypatch, capsys):
    """DESIGN 5.1, 6: the `centroid` `query --tag` reports is the origin
    `place` anchors on, so a part placed with no offset stands on it.
    On an arrayed anchor `--instance` reads the copy in the world, the
    frame `place` takes it in, where the default answer is the
    prototype's own. A tag no output carries is an error."""
    from support.recipes import write_files

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = (
        'step = mo.transform(mo.box(id = "step_box", size = [1.0, 0.2, 0.4]),'
        ' id = "step_up", translate = [0.3, 0.1, 0])\n'
        'step = mo.tag(step, id = "step", name = "top", select = {"normal_y": {"gt": 0.9}})\n'
        'steps = mo.array(step, id = "steps", count = 3, step_translate = [2.0, 0.5, 0])\n'
        'mo.output("steps", steps)\n'
        'post = mo.place(steps, mo.box(id = "post_box", size = [0.1, 1, 0.1]),'
        ' id = "post", anchor = "top", instance = 2)\n'
        'mo.output("post", post)\n'
    )
    project = write_files(tmp_path / "proj", {"asset.star": asset})
    assert main(["query", str(project), "--tag", "top", "--output", "steps"]) == 0
    local = json.loads(capsys.readouterr().out)["outputs"]["steps"]
    assert local["centroid"] == [0.3, 0.2, 0.0] and local["frame"].startswith("prototype")
    assert main(["query", str(project), "--tag", "top", "--instance", "2",
                 "--output", "steps"]) == 0
    copy = json.loads(capsys.readouterr().out)["outputs"]["steps"]
    assert copy["centroid"] == [4.3, 1.2, 0.0] and copy["frame"] == "copy 2 (world)"
    assert copy["bounds"]["min"] == [3.8, 1.2, -0.2]

    assert main(["stats", str(project)]) == 0
    post = json.loads(capsys.readouterr().out)["outputs"]["post"]["bounds"]
    stands_at = [(post["min"][0] + post["max"][0]) / 2, post["min"][1],
                 (post["min"][2] + post["max"][2]) / 2]
    assert stands_at == pytest.approx(copy["centroid"], abs=1e-6)

    assert main(["query", str(project), "--tag", "top", "--instance", "9",
                 "--output", "steps"]) == 0
    assert "skipped" in json.loads(capsys.readouterr().out)["outputs"]["steps"]
    assert main(["query", str(project), "--tag", "top",
                 "--select", '{"center_x": {"gt": 0.3}}', "--output", "steps"]) == 0
    half = json.loads(capsys.readouterr().out)["outputs"]["steps"]
    assert half["matched"] == 1
    assert main(["query", str(project), "--tag", "tpo"]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "unknown_tag" and err["detail"]["tags"] == ["top"]
