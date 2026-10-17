"""Material/texture half of export: materials, textures, and
images in the GLB; encoding conventions (sRGB base color, linear data,
packed metallicRoughness); dedup and name-identity rules; profile
checks for textured outputs."""

import json

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.glb import build_glb
from moraine3d.core.mesh import Image, Material
from moraine3d.engine.export import profile_checks
from moraine3d.errors import MoraineError
from moraine3d.ops import box, material, transform, unwrap
from moraine3d.verify.validate import validate_outputs
from support.engine import export_asset
from tests.test_export import parse_glb
from tests.test_png import decode_png


_MESH_SLOT = [0]


def _mesh(uv=True):
    # Each mesh gets its own x slot: coincident outputs would trip the
    # built-in overlap guard, which is not what these tests measure.
    _MESH_SLOT[0] += 1
    m = box.evaluate([], {}, None)
    m = transform.evaluate(
        [m], {"translate": [2.0 * _MESH_SLOT[0], 0.5, 0.0]}, None
    )  # grounded, spread apart
    return unwrap.evaluate([m], {}, None) if uv else m


def _img_view(doc, binary, tex_index):
    src = doc["textures"][tex_index]["source"]
    view = doc["bufferViews"][doc["images"][src]["bufferView"]]
    start = view.get("byteOffset", 0)
    return decode_png(binary[start : start + view["byteLength"]])


def test_alpha_below_one_exports_blend_mode():
    # glTF defaults to OPAQUE; without alphaMode the exported look would
    # contradict the validate-side transparency semantics (_opaque).
    glass = material.evaluate(
        [_mesh(uv=False)], {"name": "glass", "base_color": [1.0, 1.0, 1.0, 0.5]}, None
    )
    doc, _ = parse_glb(build_glb({"part": glass}, "test")[0])
    assert doc["materials"][0]["alphaMode"] == "BLEND"

    solid = material.evaluate([_mesh(uv=False)], {"name": "solid"}, None)
    doc, _ = parse_glb(build_glb({"part": solid}, "test")[0])
    assert "alphaMode" not in doc["materials"][0]  # opaque keeps the default


def test_neutral_material_when_unbound():
    glb, _ = build_glb({"part": _mesh(uv=False)}, "test")
    doc, _ = parse_glb(glb)
    assert len(doc["materials"]) == 1
    assert doc["materials"][0]["name"] == "moraine_default"
    assert "textures" not in doc and "images" not in doc


def test_material_factors_and_textures_in_glb():
    base = Image(pixels=np.full((4, 4, 3), 0.5, np.float32))
    rough = Image(pixels=np.full((4, 4, 1), 0.25, np.float32))
    m = material.evaluate(
        [_mesh(), base, rough],
        {
            "name": "worn",
            "slots": ["base_color", "roughness"],
            "base_color": [1.0, 0.5, 0.25],
            "metallic": 0.1,
            "roughness": 1.0,
            "emissive": [0.0, 1.5, 0.0],
        },
        None,
    )
    glb, _ = build_glb({"part": m}, "test")
    doc, binary = parse_glb(glb)
    (mat,) = doc["materials"]
    assert mat["name"] == "worn"
    pbr = mat["pbrMetallicRoughness"]
    assert pbr["baseColorFactor"] == [1.0, 0.5, 0.25, 1.0]
    assert pbr["metallicFactor"] == 0.1
    # glTF caps emissiveFactor at 1: the intensity rides the extension
    # (used, never required), the factor is the unit-range direction.
    assert mat["emissiveFactor"] == [0.0, 1.0, 0.0]
    assert mat["extensions"] == {
        "KHR_materials_emissive_strength": {"emissiveStrength": 1.5}
    }
    assert doc["extensionsUsed"] == ["KHR_materials_emissive_strength"]
    assert "extensionsRequired" not in doc
    assert doc["samplers"] == [
        {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
    ]

    # base color: linear 0.5 -> sRGB byte 188.
    px = _img_view(doc, binary, pbr["baseColorTexture"]["index"])
    assert px.shape == (4, 4, 3)
    assert int(px[0, 0, 0]) == 188

    # metallicRoughness pack: R=255 (unused), G=roughness linear byte,
    # B=255 (no metallic map -> white so the factor rules).
    mr = _img_view(doc, binary, pbr["metallicRoughnessTexture"]["index"])
    assert int(mr[0, 0, 0]) == 255
    assert int(mr[0, 0, 1]) == 64  # 0.25 linear -> 64
    assert int(mr[0, 0, 2]) == 255


def test_normal_texture_linear_with_scale():
    n = Image(pixels=np.full((4, 4, 3), 0.5, np.float32))
    m = material.evaluate(
        [_mesh(), n], {"name": "bumpy", "slots": ["normal"], "normal_scale": 0.7}, None
    )
    glb, _ = build_glb({"part": m}, "test")
    doc, binary = parse_glb(glb)
    nt = doc["materials"][0]["normalTexture"]
    assert nt["scale"] == 0.7
    px = _img_view(doc, binary, nt["index"])
    assert int(px[0, 0, 0]) == 128  # linear encode, no sRGB transfer


def test_shared_material_dedups():
    params = {"name": "kit", "roughness": 0.5}
    a = material.evaluate([_mesh()], params, None)
    b = material.evaluate([_mesh()], params, None)
    glb, _ = build_glb({"a": a, "b": b}, "test")
    doc, _ = parse_glb(glb)
    assert len(doc["materials"]) == 1
    mats = {doc["meshes"][i]["primitives"][0]["material"] for i in range(2)}
    assert mats == {0}


def test_shared_texture_image_dedups():
    img = Image(pixels=np.full((4, 4, 3), 0.3, np.float32))
    a = material.evaluate([_mesh(), img], {"name": "ma", "slots": ["base_color"]}, None)
    b = material.evaluate([_mesh(), img], {"name": "mb", "slots": ["base_color"]}, None)
    glb, _ = build_glb({"a": a, "b": b}, "test")
    doc, _ = parse_glb(glb)
    assert len(doc["materials"]) == 2
    assert len(doc["images"]) == 1  # same pixels + same encoding -> one image


def test_material_name_collision_errors():
    a = material.evaluate([_mesh()], {"name": "kit", "roughness": 0.5}, None)
    b = material.evaluate([_mesh()], {"name": "kit", "roughness": 0.9}, None)
    with pytest.raises(MoraineError) as e:
        build_glb({"a": a, "b": b}, "test")
    assert e.value.code == "material_name_collision"


def test_glb_deterministic_with_textures():
    img = Image(pixels=np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3))
    m = material.evaluate([_mesh(), img], {"name": "m", "slots": ["base_color"]}, None)
    assert build_glb({"p": m}, "t")[0] == build_glb({"p": m}, "t")[0]


# -- profile checks ----------------------------------------------------------


def test_textured_output_without_uv_blocks():
    img = Image(pixels=np.zeros((4, 4, 3), np.float32))
    m = material.evaluate([_mesh(uv=False), img], {"name": "m", "slots": ["base_color"]}, None)
    report = profile_checks("gltf", {"part": m}, validate_outputs({"part": m}))
    checks = {c["check"]: c for c in report["outputs"]["part"]["checks"]}
    assert checks["uv_present"]["ok"] is False
    assert "level" not in checks["uv_present"]  # blocking, not advisory
    assert report["ok"] is False


def test_untextured_output_without_uv_still_warns_only():
    m = material.evaluate([_mesh(uv=False)], {"name": "flat"}, None)
    report = profile_checks("gltf", {"part": m}, validate_outputs({"part": m}))
    checks = {c["check"]: c for c in report["outputs"]["part"]["checks"]}
    assert checks["uv_present"]["level"] == "warn"
    assert report["ok"] is True


def test_image_output_blocks_export(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {}},
{"id": "nz", "op": "noise_texture", "params": {"size": 16}}
],
"outputs": {"part": "bx", "preview": "nz"}
}
"""
    p = tmp_path / "s.mora"
    p.write_text(asset, encoding="utf-8")
    result = export_asset(asset_mod.load(p), profile="gltf", out_dir=tmp_path / "exports")
    assert result["ok"] is False
    checks = {
        c["check"]: c for c in result["validation"]["outputs"]["preview"]["checks"]
    }
    assert checks["mesh_output"]["ok"] is False
    assert not (tmp_path / "exports").exists()


def test_name_collision_caught_by_plain_validate():
    """DESIGN 4.1/9.1: one material name with two contents fails
    plain validate — a gate, no profile needed — so green never outruns
    the GLB writer's hard error; the fix names the colliding outputs."""
    a = material.evaluate([_mesh()], {"name": "kit", "roughness": 0.5}, None)
    b = material.evaluate([_mesh()], {"name": "kit", "roughness": 0.9}, None)
    c = material.evaluate([_mesh()], {"name": "kit", "roughness": 0.5}, None)
    report = validate_outputs({"a": a, "b": b, "c": c})
    assert report["ok"] is False
    for name, others in (("a", ["b", "c"]), ("b", ["a", "c"]), ("c", ["a", "b"])):
        checks = {ch["check"]: ch for ch in report["outputs"][name]["checks"]}
        entry = checks["material_name_unique"]
        assert entry["ok"] is False
        assert entry["detail"] == {"material": "kit", "outputs": others}
        # Either side is offered — the author decides which use deviates.
        assert "'a' (authored here), 'b' (authored here), 'c' (authored here)" in entry["fix"]
        assert "rename the material node of the locally authored side" in entry["fix"]
        assert "or make the params identical" in entry["fix"]
        assert "asset_ref" not in entry["fix"]  # no reference-borne side here
        assert "material_rename" not in entry["fix"]  # no reference to route to
    # Not duplicated by the profile pass.
    report = profile_checks("gltf", {"a": a, "b": b}, validate_outputs({"a": a, "b": b}))
    names = [ch["check"] for ch in report["outputs"]["a"]["checks"]]
    assert names.count("material_name_unique") == 1
    # Same name, same content: one identity, no collision.
    assert validate_outputs({"a": a, "c": c})["ok"] is True


def test_material_budget_declared_gates():
    """DESIGN 9.1: a declared materials budget
    gates plain validate, reports every identity with its outputs and
    structural flags, and is absent when undeclared."""
    values = {}
    for i in range(4):
        values[f"p{i}"] = material.evaluate(
            [_mesh()], {"name": f"m{i}", "roughness": i / 10}, None
        )
    values["glass"] = material.evaluate(
        [_mesh()], {"name": "glass", "base_color": [0.5, 0.6, 0.7, 0.2]}, None
    )
    values["lamp"] = material.evaluate(
        [_mesh()], {"name": "lamp", "emissive": [2.0, 2.0, 2.0]}, None
    )
    values["m0_again"] = material.evaluate(
        [_mesh()], {"name": "m0", "roughness": 0.0}, None
    )  # same name+params as p0: one identity
    plain = validate_outputs(values)
    assert plain["ok"] is True and "budgets" not in plain
    report = validate_outputs(values, budgets={"materials": 6})
    assert report["ok"] is True
    (entry,) = report["budgets"]
    assert entry["ok"] is True and entry["count"] == 6 and entry["limit"] == 6
    report = validate_outputs(values, budgets={"materials": 4})
    assert report["ok"] is False
    (entry,) = report["budgets"]
    assert entry["ok"] is False and entry["count"] == 6 and "fix" in entry
    by_name = {i["name"]: i for i in entry["identities"]}
    assert by_name["m0"]["outputs"] == ["m0_again", "p0"]
    assert by_name["glass"]["transparent"] is True and by_name["glass"]["emissive"] is False
    assert by_name["lamp"]["emissive"] is True and by_name["lamp"]["transparent"] is False
    assert "glass" in entry["fix"] and "lamp" in entry["fix"]


def test_textured_asset_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "up", "op": "transform", "inputs": ["bx"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "uw", "op": "unwrap", "inputs": ["up"], "params": {}},
{"id": "col", "op": "brick_texture", "params": {"size": 32}},
{"id": "hgt", "op": "brick_texture", "params": {"size": 32, "map": "height"}},
{"id": "nrm", "op": "normal_map", "inputs": ["hgt"], "params": {"strength": 0.4}},
{"id": "mt", "op": "material", "inputs": ["uw", "col", "nrm"], "params": {"name": "brick", "slots": ["base_color", "normal"]}}
],
"outputs": {"wall": "mt"}
}
"""
    p = tmp_path / "wall.mora"
    p.write_text(asset, encoding="utf-8")
    r1 = export_asset(asset_mod.load(p), profile="gltf", out_dir=tmp_path / "exports")
    assert r1["ok"] is True
    data = (tmp_path / "exports" / "wall.glb").read_bytes()
    doc, _ = parse_glb(data)
    assert doc["materials"][0]["name"] == "brick"
    assert len(doc["images"]) == 2  # base color + normal
    assert r1["outputs"]["wall"]["material"] == "brick"
    manifest = json.loads((tmp_path / "exports" / "wall.manifest.json").read_text())
    assert "zlib" in manifest["kernel_versions"]
    # Byte-identical re-export with textures (9.1 naming determinism).
    r2 = export_asset(asset_mod.load(p), profile="gltf", out_dir=tmp_path / "exports")
    assert (tmp_path / "exports" / "wall.glb").read_bytes() == data
    assert r2["ok"] is True


def test_glb_bytes_identical_after_cache_roundtrip(tmp_path, monkeypatch):
    # Factors must normalize on construction so a
    # cold-cache and a warm-cache evaluation serialize identical GLB bytes.
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {}},
{"id": "up", "op": "transform", "inputs": ["bx"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "mt", "op": "material", "inputs": ["up"], "params": {"name": "m", "roughness": 0.1234567891}}
],
"outputs": {"part": "mt"}
}
"""
    p = tmp_path / "s.mora"
    p.write_text(asset, encoding="utf-8")
    from support.engine import Evaluation

    fresh = Evaluation(asset_mod.load(p)).records()["part"].value()
    loaded = Evaluation(asset_mod.load(p)).records()["part"].value()  # cache hit
    assert fresh.material.roughness_factor == loaded.material.roughness_factor
    assert build_glb({"part": fresh}, "t")[0] == build_glb({"part": loaded}, "t")[0]


def test_nan_instance_transforms_rejected():
    import numpy as np
    from moraine3d.core.mesh import Instances

    xf = np.eye(4)[None].copy()
    xf[0, 1, 3] = np.nan
    inst = Instances(prototype=_mesh(uv=False), transforms=xf)
    with pytest.raises(MoraineError) as e:
        build_glb({"part": inst}, "t")
    assert e.value.code == "nonfinite_instance"
    report = validate_outputs({"part": inst})
    finite = [c for c in report["outputs"]["part"]["checks"] if c["check"] == "finite"]
    assert finite[0]["ok"] is False


def test_nan_material_texture_fails_validate():
    import numpy as np

    img = Image(pixels=np.full((4, 4, 1), np.nan, np.float32))
    m = material.evaluate([_mesh()], {"name": "m"}, None)
    m.material.textures["roughness"] = img
    report = validate_outputs({"part": m})
    finite = [c for c in report["outputs"]["part"]["checks"] if c["check"] == "finite"]
    assert finite[0]["ok"] is False
    assert "roughness" in finite[0]["fix"]


def test_material_budget_counts_neutral():
    """DESIGN 9.1: an unmaterialed part ships the neutral default, a real
    glTF material — it counts as one identity under its reserved name."""
    values = {}
    for i in range(4):
        values[f"p{i}"] = material.evaluate(
            [_mesh()], {"name": f"m{i}", "roughness": i / 10}, None
        )
    values["plain"] = _mesh()  # unmaterialed -> neutral joins the count
    report = validate_outputs(values, budgets={"materials": 4})
    (entry,) = report["budgets"]
    assert entry["ok"] is False and entry["count"] == 5
    neutral = [i for i in entry["identities"] if i["name"] == "moraine_default"]
    assert neutral == [{"name": "moraine_default", "outputs": ["plain"],
                        "transparent": False, "emissive": False}]
