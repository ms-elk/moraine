"""Export validation tests (DESIGN 10): the GLB is re-parsed and checked —
hierarchy, instancing, naming stability, and an asymmetric golden asset
that pins orientation (front faces +Z) through the ``gltf`` profile."""

import json
import struct

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Instances
from moraine3d.engine.export import build_glb, decompose_trs, profile_checks
from moraine3d.errors import MoraineError
from moraine3d.ops import array, box, material, transform
from moraine3d.verify.validate import validate_outputs
from support.engine import Evaluation, export_asset


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A
    doc = json.loads(data[20 : 20 + json_len])
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + json_len)
    assert bin_type == 0x004E4942
    binary = data[28 + json_len : 28 + json_len + bin_len]
    return doc, binary


def read_accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    start = view.get("byteOffset", 0)
    raw = binary[start : start + view["byteLength"]]
    arr = np.frombuffer(raw, dtype=dtype)[: acc["count"] * width]
    return arr.reshape(acc["count"], width) if width > 1 else arr


# The golden asset is asymmetric on every axis: a body with a nose block
# shifted +Z and up-shifted cap, so any axis flip or handedness error
# moves recognizable geometry.
_GOLDEN = """{
"build_version": "fixture",
"nodes": [
{"id": "body", "op": "box", "params": {"size": [0.4, 0.2, 0.6]}},
{"id": "body2", "op": "transform", "inputs": ["body"], "params": {"translate": [0.0, 0.1, 0.0]}},
{"id": "nose", "op": "box", "params": {"size": [0.1, 0.1, 0.2]}},
{"id": "nose2", "op": "transform", "inputs": ["nose"], "params": {"translate": [0.0, 0.15, 0.35]}},
{"id": "u1", "op": "boolean", "inputs": ["body2", "nose2"], "params": {"mode": "union"}}
],
"outputs": {"probe": "u1"}
}"""


def _export(tmp_path, monkeypatch, text, name="asset", **kw):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / f"{name}.mora"
    p.write_text(text, encoding="utf-8")
    sc = asset_mod.load(p)
    return export_asset(sc, profile="gltf", out_dir=tmp_path / "exports", **kw)


_UNREACHABLE_UNKNOWN = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "t1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "x1", "op": "warp9000", "params": {}}
],
"outputs": {"part": "t1"}
}"""


def test_unreachable_unknown_op_never_breaks_export(tmp_path, monkeypatch):
    """DESIGN 9: the manifest records the op versions of the nodes the
    artifact was evaluated from; an unevaluated node (here one of an
    unknown op) contributes nothing and cannot fail the manifest after
    the GLB is written."""
    result = _export(tmp_path, monkeypatch, _UNREACHABLE_UNKNOWN)
    assert result["ok"] is True
    assert (tmp_path / "exports" / "asset.glb").is_file()
    manifest = json.loads(
        (tmp_path / "exports" / "asset.manifest.json").read_text(encoding="utf-8")
    )
    assert sorted(manifest["op_impl_versions"]) == ["box", "transform"]


def test_golden_asset_pins_orientation(tmp_path, monkeypatch):
    """DESIGN 9.1: the ``gltf`` profile converts nothing, so geometry
    passes through in the asset's own frame (+Y up, +Z forward, DESIGN
    4.3), and each named output is a root node of that name with no
    transform (Day-One Invariants 7 and 8)."""
    result = _export(tmp_path, monkeypatch, _GOLDEN)
    assert result["ok"] is True
    doc, binary = parse_glb((tmp_path / "exports" / "asset.glb").read_bytes())

    assert doc["asset"]["version"] == "2.0"
    assert doc["nodes"][0]["name"] == "probe"  # Day-One Invariant 7
    assert "matrix" not in doc["nodes"][0]  # pivot = asset origin (Invariant 8)

    pos_idx = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    pos = read_accessor(doc, binary, pos_idx)
    acc = doc["accessors"][pos_idx]
    # Geometry passes through unchanged: the nose sticks out to +Z (0.45),
    # the body ends at -Z 0.3, and the asset rests on the ground (y in 0..0.2).
    assert acc["max"][2] == pytest.approx(0.45)
    assert acc["min"][2] == pytest.approx(-0.3)
    assert acc["min"][1] == pytest.approx(0.0, abs=1e-6)
    assert acc["max"][1] == pytest.approx(0.2)
    # The +Z extreme is the nose: small XY cross-section, upper half.
    nose_pts = pos[pos[:, 2] > 0.36]
    assert np.abs(nose_pts[:, 0]).max() == pytest.approx(0.05)
    assert doc["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] is not None


def test_reexport_is_byte_identical(tmp_path, monkeypatch):
    """DESIGN 9.1: the same content exports the same bytes, so a
    byte-identical re-export is the evidence that nothing changed."""
    _export(tmp_path, monkeypatch, _GOLDEN)
    first = (tmp_path / "exports" / "asset.glb").read_bytes()
    sc = asset_mod.load(tmp_path / "asset.mora")
    export_asset(sc, profile="gltf", out_dir=tmp_path / "exports")
    assert (tmp_path / "exports" / "asset.glb").read_bytes() == first


_INSTANCED = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "b2", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "a1", "op": "array", "inputs": ["b2"], "params": {"count": 4, "step_translate": [2.0, 0.0, 0.0], "step_rotate_deg": [0.0, 90.0, 0.0]}}
],
"outputs": {"row": "a1"}
}"""


def test_instancing_stays_instancing(tmp_path, monkeypatch):
    """Day-One Invariant 6 (DESIGN 9.1): an instanced output exports as
    one shared mesh and one node carrying ``EXT_mesh_gpu_instancing``,
    the only required extension, never merged."""
    result = _export(tmp_path, monkeypatch, _INSTANCED, name="inst")
    assert result["ok"] is True
    assert result["outputs"]["row"]["instances"] == 4
    doc, binary = parse_glb((tmp_path / "exports" / "inst.glb").read_bytes())

    assert doc["extensionsRequired"] == ["EXT_mesh_gpu_instancing"]
    ext = doc["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]
    t = read_accessor(doc, binary, ext["attributes"]["TRANSLATION"])
    q = read_accessor(doc, binary, ext["attributes"]["ROTATION"])
    s = read_accessor(doc, binary, ext["attributes"]["SCALE"])
    assert len(t) == len(q) == len(s) == 4
    # Prototype geometry is shared, one mesh; 24 verts stays 24.
    assert doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"] == 24
    # Instance 1: translated one step and yawed 90 deg about +Y.
    assert t[1] == pytest.approx([2.0, 0.0, 0.0], abs=1e-6)
    assert s[1] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    half = np.sqrt(0.5)
    assert np.abs(q[1]) == pytest.approx([0.0, half, 0.0, half], abs=1e-6)


def test_bake_instances_projects_at_export_edge(tmp_path, monkeypatch):
    """DESIGN 9: ``--bake-instances`` projects every Instances output to a
    plain mesh at the export edge — transforms applied to tiled prototype
    copies, no instancing extension — and the manifest records the flag."""
    result = _export(tmp_path, monkeypatch, _INSTANCED, name="inst",
                     bake_instances=True)
    assert result["ok"] is True
    # Info reports what ships: merged counts, no live "instances" key.
    row = result["outputs"]["row"]
    assert row["baked"] is True
    assert row["tris"] == 48 and row["verts"] == 96
    assert "instances" not in row
    doc, binary = parse_glb((tmp_path / "exports" / "inst.glb").read_bytes())
    assert doc.get("extensionsRequired", []) == []
    assert "extensions" not in doc["nodes"][0]
    pos_idx = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    pos = read_accessor(doc, binary, pos_idx)
    assert len(pos) == 96  # 24 prototype verts x 4 copies
    # Copy 1 actually moved: its step translation (+2 in x, existing
    # instanced test pins it) lands in the baked vertex positions.
    assert doc["accessors"][pos_idx]["max"][0] == pytest.approx(2.5)
    manifest = json.loads(
        (tmp_path / "exports" / "inst.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bake_instances"] is True
    # The projection op's semantics shape the bytes, so its version joins
    # the recorded determinism boundary (DESIGN 4.4) despite no graph node.
    assert manifest["op_impl_versions"]["bake_instances"] == 1


_INSTANCED_MATERIAL = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "b2", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "m1", "op": "material", "inputs": ["b2"], "params": {"name": "steel", "base_color": [0.3, 0.4, 0.5]}},
{"id": "a1", "op": "array", "inputs": ["m1"], "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}}
],
"outputs": {"row": "a1"}
}"""


def test_bake_preserves_authored_material(tmp_path, monkeypatch):
    """DESIGN 9: materials are carried through the export-edge projection
    per bake_instances semantics — the evaluator's preserved-material
    contract applies at the edge too, never the neutral fallback."""
    result = _export(tmp_path, monkeypatch, _INSTANCED_MATERIAL, name="mat",
                     bake_instances=True)
    assert result["ok"] is True
    doc, _ = parse_glb((tmp_path / "exports" / "mat.glb").read_bytes())
    assert [m["name"] for m in doc["materials"]] == ["steel"]
    base = doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]
    assert base[:3] != [1.0, 1.0, 1.0]  # not the neutral default
    assert doc["meshes"][0]["primitives"][0]["material"] == 0


_FLUSH = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "b2", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "a1", "op": "array", "inputs": ["b2"], "params": {"count": 3, "step_translate": [1.0, 0.0, 0.0]}}
],
"outputs": {"wall": "a1"}
}"""


def test_bake_never_changes_exportability(tmp_path, monkeypatch):
    """DESIGN 9: the projection changes delivery format, never
    exportability — validation runs on the authored values, so a flush
    instanced layout (welds to non-manifold if the baked mesh were
    validated) exports green under the flag exactly as by default."""
    kept = _export(tmp_path, monkeypatch, _FLUSH, name="flush")
    assert kept["ok"] is True
    baked = _export(tmp_path, monkeypatch, _FLUSH, name="flush2",
                    bake_instances=True)
    assert baked["ok"] is True
    assert (tmp_path / "exports" / "flush2.glb").is_file()


_SHEARED = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "b2", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "a1", "op": "array", "inputs": ["b2"], "params": {"count": 2, "step_translate": [3.0, 0.0, 0.0], "step_rotate_deg": [0.0, 45.0, 0.0]}},
{"id": "a2", "op": "array", "inputs": ["a1"], "params": {"count": 2, "step_translate": [0.0, 0.0, 8.0], "step_scale": [1.0, 1.0, 2.0]}}
],
"outputs": {"grid": "a2"}
}"""


def test_bake_is_the_fix_for_sheared_instances(tmp_path, monkeypatch):
    """DESIGN 9: the flag is the documented fix the sheared-instance
    export error points to — TRS-representability is skipped when baking
    projects the transforms into vertices."""
    kept = _export(tmp_path, monkeypatch, _SHEARED, name="shear")
    assert kept["ok"] is False
    checks = {c["check"]: c
              for c in kept["validation"]["outputs"]["grid"]["checks"]}
    assert checks["instances_trs"]["ok"] is False
    baked = _export(tmp_path, monkeypatch, _SHEARED, name="shear2",
                    bake_instances=True)
    assert baked["ok"] is True
    assert (tmp_path / "exports" / "shear2.glb").is_file()


def test_baked_reexport_is_byte_identical(tmp_path, monkeypatch):
    """DESIGN 9: byte-identical re-export holds per flag value."""
    _export(tmp_path, monkeypatch, _INSTANCED, name="inst", bake_instances=True)
    first = (tmp_path / "exports" / "inst.glb").read_bytes()
    sc = asset_mod.load(tmp_path / "inst.mora")
    export_asset(sc, profile="gltf", out_dir=tmp_path / "exports",
                 bake_instances=True)
    assert (tmp_path / "exports" / "inst.glb").read_bytes() == first


def test_bake_draws_no_built_in_triangle_advisory(tmp_path, monkeypatch):
    """DESIGN 4.1: an undeclared budget is not checked at all and there is
    no built-in default — so the bake path, which multiplies the
    prototype's triangles by its copy count, reports no threshold warning
    of its own (the multiplied count is what `stats` reports)."""
    result = _export(tmp_path, monkeypatch, _INSTANCED, name="inst",
                     bake_instances=True)
    assert result["ok"] is True
    assert result["outputs"]["row"]["baked"] is True and result["outputs"]["row"]["tris"] == 48
    assert not [w for w in result.get("warnings", [])
                if "tri" in w.get("warning", "")]


def test_sheared_instances_error_actionably():
    """DESIGN 9.1: instance transforms must be TRS-representable; a
    sheared one is refused as ``sheared_instance`` with the fix named
    (``--bake-instances``)."""
    shear = np.eye(4)
    shear[0, 1] = 0.5
    with pytest.raises(MoraineError) as e:
        decompose_trs(np.stack([np.eye(4), shear]), "row")
    assert e.value.code == "sheared_instance"
    assert "bake_instances" in e.value.message


def test_trs_round_trips_through_decomposition():
    """DESIGN 9.1: an instance transform composed of translation, rotation,
    and scale decomposes into the TRS the extension carries and
    recomposes to the same matrix."""
    inst = array.evaluate(
        [box.evaluate([], {}, None)],
        {"count": 5, "step_translate": [1.0, 0.2, 0.0],
         "step_rotate_deg": [10.0, 20.0, 30.0], "step_scale": 1.1},
        None,
    )
    t, q, s = decompose_trs(inst.transforms, "x")
    x, y, z, w = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    rebuilt = np.zeros((len(q), 4, 4))
    rebuilt[:, :3, :3] = r * s[:, None, :]
    rebuilt[:, :3, 3] = t
    rebuilt[:, 3, 3] = 1.0
    assert np.allclose(rebuilt, inst.transforms, atol=1e-9)


def test_export_blocked_by_failed_validation_writes_nothing(tmp_path, monkeypatch):
    """DESIGN 9: a failed validation blocks the export, which then writes
    nothing. The default box sinks below ground, so ``not_sunk`` fails."""
    text = """{
"build_version": "fixture",
"nodes": [{"id": "b1", "op": "box", "params": {}}],
"outputs": {"main": "b1"}
}"""
    result = _export(tmp_path, monkeypatch, text, name="sunk")
    assert result["ok"] is False
    checks = {c["check"]: c for c in result["validation"]["outputs"]["main"]["checks"]}
    assert checks["not_sunk"]["ok"] is False
    assert not (tmp_path / "exports").exists()


def test_export_leaves_no_temp_files(tmp_path, monkeypatch):
    """DESIGN 9: the GLB and the manifest are written atomically (temp +
    rename), so a finished export directory holds exactly the two
    deliverables."""
    _export(tmp_path, monkeypatch, _GOLDEN)
    names = sorted(p.name for p in (tmp_path / "exports").iterdir())
    assert names == ["asset.glb", "asset.manifest.json"]


def test_manifest_records_determinism_boundary(tmp_path, monkeypatch):
    """DESIGN 9: the manifest names the profile, the GLB, the
    ``--bake-instances`` flag, and the op and kernel versions the
    artifact was evaluated under: the boundary a re-export is compared
    against."""
    result = _export(tmp_path, monkeypatch, _GOLDEN)
    manifest = json.loads((tmp_path / "exports" / "asset.manifest.json").read_text())
    assert manifest["profile"] == "gltf"
    assert manifest["glb"] == "asset.glb"
    assert manifest["bake_instances"] is False  # DESIGN 9: flag joins the manifest
    assert manifest["op_impl_versions"] == {"boolean": 1, "box": 1, "transform": 1}
    assert set(manifest["kernel_versions"]) == {"numpy", "trimesh", "manifold3d", "zlib"}
    assert result["manifest"].endswith("asset.manifest.json")


def test_manifest_glb_hash_binds_the_pair(tmp_path, monkeypatch):
    """DESIGN 9 artifact pair: the manifest records `glb_hash`, the
    24-hex truncated sha256 of the exact GLB bytes beside it, so a
    mismatched pair is detectable."""
    import hashlib

    result = _export(tmp_path, monkeypatch, _GOLDEN)
    assert result["ok"] is True
    glb = (tmp_path / "exports" / "asset.glb").read_bytes()
    manifest = json.loads(
        (tmp_path / "exports" / "asset.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["glb_hash"] == hashlib.sha256(glb).hexdigest()[:24]


def test_manifest_write_failure_leaves_detectably_stale_pair(tmp_path, monkeypatch):
    """DESIGN 9 artifact pair: the GLB commits first and the manifest is
    the commit marker — a manifest write failure surfaces through the
    error channel naming the stale pair, and what remains on disk (new
    GLB, old manifest) is detectable by glb_hash mismatch, never a
    silently inconsistent pair."""
    import hashlib

    from moraine3d.core import atomic as atomic_mod
    from moraine3d.engine import export as export_mod

    result = _export(tmp_path, monkeypatch, _GOLDEN)
    assert result["ok"] is True
    old_manifest = (tmp_path / "exports" / "asset.manifest.json").read_bytes()
    old_glb = (tmp_path / "exports" / "asset.glb").read_bytes()

    real = export_mod.atomic_replace

    def fail_manifest(path, data):
        if path.name.endswith(".manifest.json"):
            raise atomic_mod.ReplaceError(
                "replace", atomic_mod.tmp_name(path), OSError("held open")
            )
        real(path, data)

    monkeypatch.setattr(export_mod, "atomic_replace", fail_manifest)
    changed = _GOLDEN.replace("[0.4, 0.2, 0.6]", "[0.5, 0.2, 0.6]")
    with pytest.raises(MoraineError) as e:
        _export(tmp_path, monkeypatch, changed)
    assert e.value.code == "export_locked"
    assert "stale" in e.value.message
    assert "re-run" in e.value.message

    new_glb = (tmp_path / "exports" / "asset.glb").read_bytes()
    assert new_glb != old_glb  # the GLB replace had already committed
    assert (
        tmp_path / "exports" / "asset.manifest.json"
    ).read_bytes() == old_manifest
    stale = json.loads(old_manifest)
    assert stale["glb_hash"] != hashlib.sha256(new_glb).hexdigest()[:24]


def test_unknown_profile_is_refused_at_both_entry_points(tmp_path, monkeypatch):
    """A profile is a measured delta (DESIGN 9), and the roster is what makes
    ``--profile`` mean anything: the checks are identical for every accepted
    name, so a name outside it must be refused rather than collect the
    container-general checks under its own label. Both entry points state the
    roster — the error is its single source of truth for the CLI."""
    b = transform.evaluate([box.evaluate([], {}, None)], {"translate": [0.0, 0.5, 0.0]}, None)
    with pytest.raises(MoraineError) as e:
        profile_checks("unreal", {"m": b}, validate_outputs({"m": b}))
    assert e.value.code == "unknown_profile"
    assert "gltf" in e.value.message

    p = tmp_path / "asset.mora"
    p.write_text(_GOLDEN, encoding="utf-8")
    with pytest.raises(MoraineError) as e:
        export_asset(asset_mod.load(p), profile="unreal", out_dir=tmp_path / "exports")
    assert e.value.code == "unknown_profile"
    assert "gltf" in e.value.message
    assert not (tmp_path / "exports").exists()  # refused before any write


def test_profile_checks_warn_on_missing_uv_without_blocking():
    """DESIGN 9.1: a missing UV set on an untextured output is advisory
    (``level: warn``) and does not block the export."""
    b = transform.evaluate([box.evaluate([], {}, None)], {"translate": [0.0, 0.5, 0.0]}, None)
    report = validate_outputs({"m": b})
    report = profile_checks("gltf", {"m": b}, report)
    checks = {c["check"]: c for c in report["outputs"]["m"]["checks"]}
    assert checks["uv_present"]["ok"] is False
    assert checks["uv_present"]["level"] == "warn"
    assert report["ok"] is True


def _materialed_box(i: int):
    b = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [3.0 * i, 0.5, 0.0]}, None
    )
    return material.evaluate(
        [b], {"name": f"m{i}", "base_color": [0.1 + 0.1 * i, 0.2, 0.3]}, None
    )


def test_export_bake_matches_in_graph_bake(tmp_path, monkeypatch):
    """DESIGN 9: the export edge's ``--bake-instances`` projection is the
    same evaluate-and-carry step as an in-graph ``bake_instances`` node —
    identical shipped stats (material re-bound, counts, bounds) and GLB
    material, so the two paths cannot drift."""
    edge = _export(tmp_path, monkeypatch, _INSTANCED_MATERIAL, name="edge",
                   bake_instances=True)
    in_graph_asset = _INSTANCED_MATERIAL.replace(
        '],\n"outputs": {"row": "a1"}',
        ',\n{"id": "k1", "op": "bake_instances", "inputs": ["a1"], "params": {}}\n'
        '],\n"outputs": {"row": "k1"}',
    )
    assert in_graph_asset != _INSTANCED_MATERIAL
    graph = _export(tmp_path, monkeypatch, in_graph_asset, name="graph")
    assert edge["ok"] is True and graph["ok"] is True
    e, g = edge["outputs"]["row"], graph["outputs"]["row"]
    assert e["baked"] is True and "baked" not in g
    for key in ("verts", "tris", "material", "bounds"):
        assert e[key] == g[key], key
    docs = [parse_glb((tmp_path / "exports" / f"{n}.glb").read_bytes())[0]
            for n in ("edge", "graph")]
    assert docs[0]["materials"] == docs[1]["materials"]
    assert docs[0]["meshes"][0]["primitives"][0]["attributes"] == \
        docs[1]["meshes"][0]["primitives"][0]["attributes"]


def test_material_budget_counts_shipped_outputs():
    """DESIGN 4.1/9.1: the material budget counts every identity the
    asset ships — reference-borne identities included, so the origins
    walk never exempts an output from the count."""
    values = {f"p{i}": _materialed_box(i) for i in range(5)}
    ref_borne = {n: {"refs": [f"r{i}"]} for i, n in enumerate(values)}
    report = validate_outputs(values, budgets={"materials": 4}, origins=ref_borne)
    assert report["ok"] is False
    entry = report["budgets"][0]
    assert entry["count"] == 5 and entry["limit"] == 4 and entry["ok"] is False
    assert "budgets" in entry["fix"]
    report = validate_outputs(values, budgets={"materials": 5}, origins=ref_borne)
    assert report["ok"] is True and report["budgets"][0]["count"] == 5


def test_material_origins_is_trait_driven(tmp_path):
    """DESIGN 9.1: an output's material origin is decided by the op-declared
    traits (IS_REFERENCE / authors_material), not module identity — a plain
    ref names its ref node for the uniqueness fix, a ref with
    material_rename authors the identity locally (empty refs), a local
    material over a ref likewise; and the walk follows CARRIED_INPUTS
    only, so a place anchor base or scatter target contributes neither
    its material nor its references."""
    (tmp_path / "subassets").mkdir()
    (tmp_path / "subassets" / "piece.mora").write_text(
        '{"build_version": "fixture", "nodes": [\n'
        '{"id": "b", "op": "box", "params": {}},\n'
        '{"id": "m", "op": "material", "inputs": ["b"], "params": {"name": "paint"}}\n'
        '], "outputs": {"p": "m"}}', encoding="utf-8")
    root = tmp_path / "root.mora"
    root.write_text(
        '{"build_version": "fixture", "nodes": [\n'
        '{"id": "r1", "op": "asset_ref", "params": {"asset": "subassets/piece.mora", "output": "p"}},\n'
        '{"id": "r2", "op": "asset_ref", "params": {"asset": "subassets/piece.mora", "output": "p", "material_rename": {"paint": "paint_b"}}},\n'
        '{"id": "r3", "op": "asset_ref", "params": {"asset": "subassets/piece.mora", "output": "p"}},\n'
        '{"id": "s3", "op": "material", "inputs": ["r3"], "params": {"name": "skin"}},\n'
        '{"id": "b4", "op": "box", "params": {}},\n'
        '{"id": "base", "op": "box", "params": {"size": [4.0, 0.2, 4.0]}},\n'
        '{"id": "base_m", "op": "material", "inputs": ["base"], "params": {"name": "floor"}},\n'
        '{"id": "base_t", "op": "tag", "inputs": ["base_m"], "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},\n'
        '{"id": "r5", "op": "asset_ref", "params": {"asset": "subassets/piece.mora", "output": "p"}},\n'
        '{"id": "pl5", "op": "place", "inputs": ["base_t", "r5"], "params": {"anchor": "top", "mode": "on_top"}},\n'
        '{"id": "r6", "op": "asset_ref", "params": {"asset": "subassets/piece.mora", "output": "p"}},\n'
        '{"id": "t6", "op": "tag", "inputs": ["r6"], "params": {"name": "bottom", "select": {"normal_y": {"lt": -0.9}}}},\n'
        '{"id": "mt6", "op": "mate", "inputs": ["base_t", "t6"], "params": {"anchor": "top", "face": "bottom"}},\n'
        '{"id": "b7", "op": "box", "params": {}},\n'
        '{"id": "mt7", "op": "mate", "inputs": ["base_t", "b7", "t6"], "params": {"anchor": "top", "face": "bottom"}}\n'
        '], "outputs": {"a": "r1", "b": "r2", "c": "s3", "d": "b4", "e": "pl5", "f": "mt6", "g": "mt7"}}', encoding="utf-8")
    origins = Evaluation(asset_mod.load(root)).material_origins()
    assert origins == {
        "a": {"refs": ["r1"]},
        "b": {"refs": []},
        "c": {"refs": []},
        "d": {"refs": []},
        # place carries the part only: the locally-materialed anchor base
        # neither authors the output's identity nor hides the ref it
        # actually arrives through.
        "e": {"refs": ["r5"]},
        # mate likewise: the anchor base and the leader are reference-only
        # inputs — a mated ref stays a ref, a local rider stays local.
        "f": {"refs": ["r6"]},
        "g": {"refs": []},
    }


def test_assembly_budget_counts_shipped_identities(tmp_path, monkeypatch):
    """DESIGN 4.1/9.1, end to end: an asset's declared budget counts what
    its outputs ship — a kit assembly's reference-borne identities count
    against the assembly's own budget (it bounds the exported glb), while
    identities shared across referenced pieces count once."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    piece_nodes, piece_outputs = [], {}
    for i in range(5):
        piece_nodes += [
            f'{{"id": "b{i}", "op": "box", "params": {{}}}}',
            f'{{"id": "t{i}", "op": "transform", "inputs": ["b{i}"], '
            f'"params": {{"translate": [{3.0 * i}, 0.5, 0.0]}}}}',
            f'{{"id": "m{i}", "op": "material", "inputs": ["t{i}"], '
            f'"params": {{"name": "m{i}", "base_color": [{0.1 + 0.1 * i}, 0.2, 0.3]}}}}',
        ]
        piece_outputs[f"p{i}"] = f"m{i}"
    piece = ('{"build_version": "fixture", "nodes": [\n'
             + ",\n".join(piece_nodes)
             + '\n], "outputs": ' + json.dumps(piece_outputs)
             + ', "budgets": {"materials": 4}}')
    (proj / "subassets" / "piece.mora").write_text(piece, encoding="utf-8")
    refs = ",\n".join(
        f'{{"id": "r{i}", "op": "asset_ref", "params": '
        f'{{"asset": "subassets/piece.mora", "output": "p{i}"}}}}'
        for i in range(5)
    )
    ref_outputs = json.dumps({f"a{i}": f"r{i}" for i in range(5)})
    root = ('{"build_version": "fixture", "nodes": [\n' + refs
            + '\n], "outputs": ' + ref_outputs + ', "budgets": {"materials": 4}}')
    (proj / "assembly.mora").write_text(root, encoding="utf-8")

    result = export_asset(
        asset_mod.load(proj / "assembly.mora"), profile="gltf",
        out_dir=tmp_path / "exports",
    )
    assert result["ok"] is False
    entry = result["validation"]["budgets"][0]
    assert entry["ok"] is False and entry["count"] == 5 and entry["limit"] == 4
    assert [i["name"] for i in entry["identities"]] == [f"m{i}" for i in range(5)]

    piece_result = export_asset(
        asset_mod.load(proj / "subassets" / "piece.mora"), profile="gltf",
        out_dir=tmp_path / "exports-piece",
    )
    assert piece_result["ok"] is False
    entry = piece_result["validation"]["budgets"][0]
    assert entry["ok"] is False and entry["count"] == 5 and entry["limit"] == 4
    assert [i["name"] for i in entry["identities"]] == [f"m{i}" for i in range(5)]

    shared_refs = ",\n".join(
        f'{{"id": "r{i}", "op": "asset_ref", "params": '
        f'{{"asset": "subassets/piece.mora", "output": "p0"}}}},\n'
        f'{{"id": "t{i}", "op": "transform", "inputs": ["r{i}"], '
        f'"params": {{"translate": [{3.0 * i}, 0.0, 0.0]}}}}'
        for i in range(5)
    )
    shared_outputs = json.dumps({f"a{i}": f"t{i}" for i in range(5)})
    shared = ('{"build_version": "fixture", "nodes": [\n' + shared_refs
              + '\n], "outputs": ' + shared_outputs
              + ', "budgets": {"materials": 1}}')
    (proj / "shared.mora").write_text(shared, encoding="utf-8")
    # budgets: {"materials": 1} — five refs of one kit material pass only
    # because the shared identity counts once.
    shared_result = export_asset(
        asset_mod.load(proj / "shared.mora"), profile="gltf",
        out_dir=tmp_path / "exports-shared",
    )
    assert shared_result["ok"] is True


def test_assembly_authored_materials_still_count(tmp_path, monkeypatch):
    """DESIGN 4.1/9.1: materials bound in the referencing asset count
    against its declared budget even over referenced geometry — they
    ship in this asset's glb like any other identity."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    piece_nodes = ",\n".join(
        f'{{"id": "b{i}", "op": "box", "params": {{}}}},\n'
        f'{{"id": "t{i}", "op": "transform", "inputs": ["b{i}"], '
        f'"params": {{"translate": [{3.0 * i}, 0.5, 0.0]}}}}'
        for i in range(5)
    )
    piece_outputs = json.dumps({f"p{i}": f"t{i}" for i in range(5)})
    piece = ('{"build_version": "fixture", "nodes": [\n' + piece_nodes
             + '\n], "outputs": ' + piece_outputs + "}")
    (proj / "subassets" / "piece.mora").write_text(piece, encoding="utf-8")
    nodes = ",\n".join(
        f'{{"id": "r{i}", "op": "asset_ref", "params": '
        f'{{"asset": "subassets/piece.mora", "output": "p{i}"}}}},\n'
        f'{{"id": "s{i}", "op": "material", "inputs": ["r{i}"], '
        f'"params": {{"name": "skin{i}", "base_color": [0.2, {0.1 + 0.1 * i}, 0.4]}}}}'
        for i in range(5)
    )
    outputs = json.dumps({f"k{i}": f"s{i}" for i in range(5)})
    root = ('{"build_version": "fixture", "nodes": [\n' + nodes
            + '\n], "outputs": ' + outputs + ', "budgets": {"materials": 4}}')
    (proj / "skinned.mora").write_text(root, encoding="utf-8")
    result = export_asset(
        asset_mod.load(proj / "skinned.mora"), profile="gltf",
        out_dir=tmp_path / "exports",
    )
    assert result["ok"] is False
    entry = result["validation"]["budgets"][0]
    assert entry["ok"] is False and entry["count"] == 5


def test_instanced_output_with_waiver_exports(tmp_path, monkeypatch):
    """DESIGN 4.1: a waiver (``checks: {output: {kind: false}}``) turns a
    failing builtin check into a reported warning, so a deliberately
    below-ground instanced output (its base copy centered) still
    exports."""
    text = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "a1", "op": "array", "inputs": ["b1"], "params": {"count": 2, "step_translate": [0.0, 2.0, 0.0]}}
],
"outputs": {"stack": "a1"},
"checks": {"stack": {"not_sunk": false}}
}"""
    result = _export(tmp_path, monkeypatch, text, name="waived")
    assert result["ok"] is True
    assert any(c.get("waived") for c in result.get("warnings", []))


_CLASHING_ASSERT = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "g1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "g2", "op": "transform", "inputs": ["b2"], "params": {"translate": [0.5, 0.5, 0.0]}}
],
"outputs": {"a": "g1", "b": "g2"}
}"""


def test_export_gates_on_asserts(tmp_path, monkeypatch):
    """Asserts gate the profile/export path: every per-output check here is
    green, only the built-in overlap guard fails — export must refuse and
    write nothing (profile_checks recomputing report["ok"] from outputs
    alone would silently drop the gate)."""
    result = _export(tmp_path, monkeypatch, _CLASHING_ASSERT, name="clash")
    assert result["ok"] is False
    report = result["validation"]
    assert report["ok"] is False
    assert all(o["ok"] for o in report["outputs"].values())  # outputs green
    guard = next(e for e in report["asserts"] if e.get("builtin"))
    assert not guard["ok"]  # the guard is what failed
    assert not (tmp_path / "exports" / "clash.glb").exists()


def test_glb_refuses_empty_and_nonfinite_meshes():
    """build_glb raises a structured error for a mesh with no triangles
    (empty_mesh) or with NaN/inf positions (nonfinite_mesh, mirroring the
    instance-transform gate) instead of a raw numpy error or a GLB whose
    JSON carries an 'Infinity' token no reader parses."""
    from moraine3d.core.mesh import Mesh

    empty = Mesh(
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
    )
    with pytest.raises(MoraineError) as exc:
        build_glb({"hole": empty}, generator="t")
    assert exc.value.code == "empty_mesh" and "hole" in exc.value.message

    bad = box.evaluate([], {}, None)
    bad.vertices = bad.vertices.copy()
    bad.vertices[0, 0] = np.inf
    with pytest.raises(MoraineError) as exc:
        build_glb({"part": bad}, generator="t")
    assert exc.value.code == "nonfinite_mesh" and "part" in exc.value.message


def test_tangents_are_not_written():
    """DESIGN 9.1: tangents are not written. A TANGENT slot a mesh
    carries stays out of the GLB, so the consumer generates MikkTSpace
    tangents as the glTF spec requires (DESIGN 4.3)."""
    part = box.evaluate([], {}, None)
    tan = np.zeros((len(part.vertices), 4), dtype=np.float32)
    tan[:, 0] = 1.0
    tan[:, 3] = 1.0
    part.attributes = {"TANGENT": tan}
    data, _ = build_glb({"part": part}, generator="t")
    doc, _bin = parse_glb(data)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "TANGENT" not in attrs and "NORMAL" in attrs


def test_alpha_mode_follows_the_factor_only():
    """Transparency in the GLB comes from the base-color factor's alpha
    alone: Image buffers are 1- or 3-channel, so no texture can carry
    alpha and no textured material is BLEND without a factor < 1."""
    from moraine3d.core.mesh import Image, Material

    tex = Image(pixels=np.full((4, 4, 3), 0.5, dtype=np.float32))
    opaque_textured = box.evaluate([], {}, None)
    opaque_textured.material = Material(name="painted", textures={"base_color": tex})
    translucent = box.evaluate([], {}, None)
    translucent.material = Material(name="glass", base_color_factor=(1.0, 1.0, 1.0, 0.5))
    data, _ = build_glb({"a": opaque_textured, "b": translucent}, generator="t")
    doc, _bin = parse_glb(data)
    modes = {m["name"]: m.get("alphaMode") for m in doc["materials"]}
    assert modes == {"painted": None, "glass": "BLEND"}


def test_export_surfaces_the_asset_level_waiver(tmp_path, monkeypatch):
    """Export warnings carry waived asset checks like waived per-output
    ones (DESIGN 9: export gates on validate and must not be quieter
    than it): an `asset_checks` grounded waiver rides `warnings`."""
    import json

    from moraine3d.core import asset as asset_mod
    from support.engine import export_asset

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "proj.mora").write_text(json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "b2", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 2.0, 0.0]}},
        ],
        "outputs": {"main": "b2"},
        "asset_checks": {"grounded": False},
    }), encoding="utf-8")
    result = export_asset(asset_mod.load(proj / "proj.mora"), "gltf", tmp_path / "out")
    assert result["ok"] is True
    waived = [w for w in result["warnings"] if w.get("waived")]
    assert waived == [{"check": "grounded", "ok": True, "waived": True}]


_TAGGED = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "t1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "tag1", "op": "tag", "inputs": ["t1"], "params": {"name": "lid_faces", "select": {"normal_y": {"gt": 0.9}}}}
],
"outputs": {"part": "tag1"}
}"""


def test_face_groups_ride_stats_but_never_the_glb(tmp_path, monkeypatch):
    """DESIGN 5.1: a tag's face group rides result_hash/cache/stats and is
    never exported — the export's per-output info carries the group under
    `groups`, while the GLB's JSON chunk names neither the group nor any
    groups key."""
    result = _export(tmp_path, monkeypatch, _TAGGED)
    assert result["ok"] is True
    assert result["outputs"]["part"]["groups"] == {"lid_faces": 2}
    doc, _binary = parse_glb((tmp_path / "exports" / "asset.glb").read_bytes())
    text = json.dumps(doc)
    assert "lid_faces" not in text
    assert "face_groups" not in text and "groups" not in text


_TWO_PARTS = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "base", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}},
{"id": "red", "op": "material", "inputs": ["base"], "params": {"name": "red", "base_color": [1.0, 0.0, 0.0, 1.0]}},
{"id": "b2", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
{"id": "cap", "op": "transform", "inputs": ["b2"], "params": {"translate": [0.0, 1.25, 0.0]}},
{"id": "blue", "op": "material", "inputs": ["cap"], "params": {"name": "blue", "base_color": [0.0, 0.0, 1.0, 1.0]}}
],
"outputs": {%s}
}"""


def test_the_order_outputs_are_declared_in_moves_no_byte(tmp_path, monkeypatch):
    """DESIGN 9.1: an export writes the parts in the order of their
    names, so the same content declared in another order exports the
    same bytes."""
    declared = ['"Tower_base": "red"', '"cap": "blue"']
    got = []
    for i, outputs in enumerate((declared, declared[::-1])):
        (tmp_path / str(i)).mkdir()
        result = _export(tmp_path / str(i), monkeypatch, _TWO_PARTS % ", ".join(outputs), "kit")
        assert result["ok"] is True
        got.append((tmp_path / str(i) / "exports" / "kit.glb").read_bytes())
    assert got[0] == got[1]
    doc, _ = parse_glb(got[0])
    assert [n["name"] for n in doc["nodes"]] == ["cap", "Tower_base"]
    assert [m["name"] for m in doc["materials"]] == ["blue", "red"]
