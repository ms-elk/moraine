"""asset_ref ``material_rename`` (DESIGN 4.6): the consumer's
explicit per-use material identity — applied at the reference boundary,
depth-agnostic, outside the override identity, a derived cache record;
and the end-to-end path it exists for: overrides reaching a material no
longer break name-as-identity, the uniqueness gate's fix routes to it."""

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import MoraineError, OpError
from moraine3d.ops import asset_ref as asset_ref_mod
from moraine3d.verify.validate import validate_outputs
from support.engine import evaluate_asset, Evaluation, export_asset
from tests.test_export import parse_glb
from tests.test_asset_ref import (  # noqa: F401
    build_project, cache, multi_ref_asset, write_project,
)

# A kit piece whose declared knob reaches its material: the field-test
# shape (texture-class overrides) in its smallest form.
MAT_PART = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
{"id": "tr01", "op": "transform", "inputs": ["bx01"], "params": {"translate": [0.0, 0.25, 0.0]}},
{"id": "mt01", "op": "material", "inputs": ["tr01"], "params": {"name": "paint", "base_color": [0.2, 0.3, 0.4], "roughness": 0.5}},
{"id": "tr02", "op": "transform", "inputs": ["mt01"], "params": {"translate": [3.0, 0.0, 0.0]}},
{"id": "ar01", "op": "array", "inputs": ["tr02"], "params": {"count": 2, "step_translate": [1.0, 0.0, 0.0]}}
],
"params": [
{"id": "rough", "value": 0.5, "node": "mt01", "param": "roughness", "min": 0.0, "max": 1.0}
],
"outputs": {"panel": "mt01", "row": "ar01"},
"asserts": [
{"id": "apart", "kind": "clearance", "outputs": ["panel", "row"], "owner": "agent", "min": 0.001}
]
}
"""


def _ref(output="panel", **extra):
    return {"asset": "subassets/part.mora", "output": output, **extra}


def test_material_rename_param_contract():
    """DESIGN 4.6: the mapping is validated like a material name — a
    portable identifier, never the reserved neutral name, distinct
    targets, and never a self-mapping (one spelling per meaning)."""
    for rename in (
        ["paint"],
        {"": "x"},
        {"paint": 1},
        {"paint": "bad name"},
        {"paint": "moraine_default"},
        {"paint": "paint"},
        {"a": "same", "b": "same"},
    ):
        with pytest.raises(OpError) as exc:
            asset_ref_mod.ref_params(_ref(material_rename=rename))
        assert exc.value.code == "bad_param"
        assert exc.value.detail["param"] == "material_rename"
    spec = asset_ref_mod.ref_params(_ref(material_rename={"paint": "paint_b"}))
    assert spec.material_rename == {"paint": "paint_b"}
    assert asset_ref_mod.ref_params(_ref()).material_rename == {}


def test_material_rename_applies_at_boundary(cache):
    """DESIGN 4.6: the renamed ref's value carries the new name over the
    same content (factors and textures untouched), as its own record —
    the un-renamed sibling keeps aliasing the referenced output."""
    proj = cache / "proj"
    root = write_project(
        proj,
        root_text=multi_ref_asset({
            "a": _ref(),
            "b": _ref(material_rename={"paint": "paint_b"}),
            "c": _ref("row", material_rename={"paint": "paint_c"}),
        }),
        part_text=MAT_PART,
    )
    ev = Evaluation(asset_mod.load(root))
    recs = ev.records()
    sub = Evaluation(asset_mod.load(proj / "subassets" / "part.mora")).records()
    assert recs["a"].key == sub["panel"].key
    assert recs["b"].key != sub["panel"].key
    assert recs["a"].stats["material"] == "paint"
    assert recs["b"].stats["material"] == "paint_b"
    assert recs["c"].stats["material"] == "paint_c"
    assert recs["c"].stats["instances"] == 2  # instancing passes through
    ma, mb = recs["a"].value().material, recs["b"].value().material
    assert mb.name == "paint_b" and ma.name == "paint"
    assert {k: v for k, v in mb.factors().items() if k != "name"} == \
        {k: v for k, v in ma.factors().items() if k != "name"}
    assert mb.content_hash() != ma.content_hash()
    # The referenced record itself is untouched by the rename.
    assert recs["a"].value().material.name == "paint"


def test_rename_materials_is_total_over_its_inputs():
    """rename_materials as a plain function: an empty mapping is the
    identity (the evaluator keeps it off that path, a direct caller must
    not hit a bare KeyError), and a renamed value never writes into the
    source value's dicts (face groups canonicalized on a copy)."""
    import numpy as np

    from moraine3d.ops import box, material, tag

    src = material.evaluate([box.evaluate([], {}, None)], {"name": "paint"}, None)
    src = tag.evaluate([src], {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None)
    assert asset_ref_mod.rename_materials(src, {}) is src
    groups_before = {k: v for k, v in src.face_groups.items()}
    out = asset_ref_mod.rename_materials(src, {"paint": "paint_b"})
    assert out.material.name == "paint_b" and src.material.name == "paint"
    assert out.face_groups is not src.face_groups
    assert all(src.face_groups[k] is v for k, v in groups_before.items())
    assert np.array_equal(out.face_groups["top"], src.face_groups["top"])


def test_material_rename_is_depth_agnostic(cache):
    """DESIGN 4.6: materials arriving through a nested reference are
    renamed too — a material name is the output's public export
    identity wherever it was authored."""
    proj = cache / "proj"
    wrap = json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "in01", "op": "asset_ref",
                   "params": {"asset": "subassets/part.mora", "output": "panel"}},
                  {"id": "tr01", "op": "transform", "inputs": ["in01"],
                   "params": {"translate": [2.0, 0.0, 0.0]}}],
        "outputs": {"moved": "tr01"},
    })
    root = write_project(
        proj,
        root_text=multi_ref_asset({
            "deep": {"asset": "subassets/wrap.mora", "output": "moved",
                     "material_rename": {"paint": "paint_deep"}},
        }),
        part_text=MAT_PART,
    )
    (proj / "subassets" / "wrap.mora").write_text(wrap, encoding="utf-8")
    recs = Evaluation(asset_mod.load(root)).records()
    assert recs["deep"].stats["material"] == "paint_deep"


def test_material_rename_unknown_name_fails_loud(cache):
    """DESIGN 4.6: every key must name a material present on the received
    output — an absent name, an unmaterialed output, or an image output
    fails with what is actually there, never a silent no-op."""
    proj = cache / "proj"
    image_part = json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "tx01", "op": "checker_texture", "params": {"size": 8}}],
        "outputs": {"tex": "tx01"},
    })
    cases = [
        (MAT_PART, _ref(material_rename={"nope": "x"}), ["paint"], ["nope"]),
        (MAT_PART, _ref(material_rename={"paint": "p2", "nope": "x"}), ["paint"], ["nope"]),
        (asset_mod.dumps(asset_mod.loads(MAT_PART)).replace(
            '"panel": "mt01"', '"panel": "tr01"'), _ref(material_rename={"paint": "x"}),
         [], ["paint"]),
        (image_part, _ref("tex", material_rename={"paint": "x"}), [], ["paint"]),
    ]
    for part_text, ref, present, unknown in cases:
        root = write_project(
            proj, root_text=multi_ref_asset({"main": ref}), part_text=part_text,
        )
        with pytest.raises(MoraineError) as exc:
            evaluate_asset(asset_mod.load(root))
        assert exc.value.code == "material_rename_unknown"
        assert exc.value.detail["present"] == present
        assert exc.value.detail["unknown"] == unknown
        assert exc.value.detail["node"] == "rf01"
        assert exc.value.detail["ref"] == "subassets/part.mora"


def test_empty_material_rename_is_plain_identity(cache):
    """Rule 3 pin: ``"material_rename": {}`` and no key are one identity —
    both alias the referenced output's record outright, so the default
    path is bit-identical to the pre-param behavior (no op_impl_version
    bump)."""
    proj = cache / "proj"
    root = write_project(
        proj,
        root_text=multi_ref_asset({"a": _ref(), "b": _ref(material_rename={})}),
        part_text=MAT_PART,
    )
    recs = Evaluation(asset_mod.load(root)).records()
    assert recs["a"] is recs["b"]
    # And the frozen default-path hash (the referenced output's own):
    # a drift here means the un-renamed path's bits moved.
    assert recs["a"].result_hash == (
        "8096114b88498d5b2cc5aeb84ae4900972d6455d127d25932e9a60fca5b55f5a"
    )


def test_material_rename_stays_outside_override_identity(cache):
    """DESIGN 4.6: a rename does not create a sub-asset instance — two uses
    with the same overrides share one instance (and one assert bundle)
    whether or not one of them renames; only the record keys differ."""
    proj = cache / "proj"
    root = build_project(
        proj,
        root_text=multi_ref_asset({
            "a": _ref(overrides={"rough": 0.9}),
            "b": _ref(overrides={"rough": 0.9}, material_rename={"paint": "paint_b"}),
        }),
        part_text=MAT_PART,
    )
    ev = Evaluation(asset_mod.load(root))
    recs = ev.records()
    assert recs["a"].key != recs["b"].key
    assert len(ev._assets) == 1
    assert len(ev.ref_asserts()) == 1
    assert recs["b"].value().material.roughness_factor == 0.9  # overrides landed first


def test_material_rename_is_a_cache_citizen(cache, monkeypatch):
    """DESIGN 4.5: a renamed ref re-evaluates as a pure warm pass;
    changing the target re-keys only that use; a clean cache regenerates
    identical keys (determinism)."""
    proj = cache / "proj"

    def root_text(new):
        return multi_ref_asset({
            "plain": _ref(),
            "named": _ref(material_rename={"paint": new}),
        })

    root = write_project(proj, root_text=root_text("paint_b"), part_text=MAT_PART)
    r1 = evaluate_asset(asset_mod.load(root))
    ev = Evaluation(asset_mod.load(root))
    ev.records()
    assert ev.profile()["misses"] == 0 and ev.profile()["hits"] > 0
    write_project(proj, root_text=root_text("paint_c"), part_text=MAT_PART)
    r2 = evaluate_asset(asset_mod.load(root))
    assert r2["named"]["key"] != r1["named"]["key"]
    assert r2["plain"]["key"] == r1["plain"]["key"]
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache / "cache2"))
    r3 = evaluate_asset(asset_mod.load(root))
    assert r3["named"]["key"] == r2["named"]["key"]


def test_material_rename_result_hash_frozen(cache):
    """Rule 3 pin (asset_ref op_impl_version 2 unchanged): the renamed
    value's content contract — only the name changes — is frozen by its
    result hash; a change here means the rename semantics moved and the
    version must bump."""
    proj = cache / "proj"
    root = write_project(
        proj,
        root_text=multi_ref_asset({"named": _ref(material_rename={"paint": "paint_b"})}),
        part_text=MAT_PART,
    )
    recs = Evaluation(asset_mod.load(root)).records()
    assert recs["named"].result_hash == (
        "252636ef52e7e6f201949a8f7209944cb695c99cc1de0d6aaf48ef911b778a30"
    )


def test_material_rename_repairs_override_collision_end_to_end(cache):
    """The rename-repair line (DESIGN 9.1/11): overrides reaching a material
    fail plain validate's uniqueness gate with a fix naming the ref node
    and material_rename; adding the rename turns validate green, exports
    under the ``gltf`` profile, and ships two glTF materials."""
    proj = cache / "proj"

    def root_text(rename):
        b = _ref(overrides={"rough": 0.9})
        if rename:
            b["material_rename"] = {"paint": "paint_rough"}
        doc = json.loads(multi_ref_asset({"a": _ref(), "b": b}))
        # The two uses coincide on purpose (only the material differs):
        # a declared embed keeps the overlap guard out of the way.
        doc["asserts"] = [{"id": "stacked", "kind": "embedded",
                          "outputs": ["a", "b"], "owner": "agent"}]
        return json.dumps(doc)

    root = build_project(proj, root_text=root_text(False), part_text=MAT_PART)
    sc = asset_mod.load(root)
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    report = validate_outputs(values, asserts=sc.asserts,
                              origins=ev.material_origins())
    assert report["ok"] is False
    for name, other, ref in (("a", "b", "rf01"), ("b", "a", "rf02")):
        checks = {c["check"]: c for c in report["outputs"][name]["checks"]}
        entry = checks["material_name_unique"]
        assert entry["ok"] is False
        assert entry["detail"] == {"material": "paint", "outputs": [other]}
        assert f"asset_ref ['{ref}']" in entry["fix"]
        assert '"material_rename": {"paint": "<new name>"}' in entry["fix"]

    build_project(proj, root_text=root_text(True), part_text=MAT_PART)
    sc = asset_mod.load(root)
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    assert validate_outputs(
        values, asserts=sc.asserts, origins=ev.material_origins()
    )["ok"] is True
    result = export_asset(sc, profile="gltf", out_dir=cache / "exports")
    assert result["ok"] is True
    doc, _ = parse_glb((cache / "exports" / "root.glb").read_bytes())
    names = sorted(m["name"] for m in doc["materials"])
    assert names == ["paint", "paint_rough"]
    assert ev.material_origins() == {
        "a": {"refs": ["rf01"]},
        "b": {"refs": []},
    }
