"""The material op and the material model: binding, hash/stats/cache
participation, evaluator propagation (MATERIAL_POLICY), drop warnings
(mirrors tests/test_attributes.py for the
attribute model)."""

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Image, Instances
from moraine3d.errors import OpError
from moraine3d.ops import box, material, noise_texture
from support.engine import Evaluation


def _box():
    return box.evaluate([], {}, None)


def _gray(v=0.5, size=8):
    return Image(pixels=np.full((size, size, 1), v, np.float32))


def _rgb(size=8):
    return Image(pixels=np.zeros((size, size, 3), np.float32))


# -- op ----------------------------------------------------------------------


def test_material_binds_and_geometry_untouched():
    src = _box()
    out = material.evaluate([src], {"name": "steel", "metallic": 1.0, "roughness": 0.3}, None)
    assert out.material is not None
    assert out.material.name == "steel"
    assert out.material.metallic_factor == 1.0
    assert np.array_equal(out.vertices, src.vertices)
    assert src.material is None  # input object untouched


def test_material_requires_name():
    with pytest.raises(OpError):
        material.evaluate([_box()], {}, None)
    with pytest.raises(OpError):
        material.evaluate([_box()], {"name": "moraine_default"}, None)


def test_material_slots_positional():
    out = material.evaluate(
        [_box(), _rgb(), _gray()], {"name": "brick", "slots": ["base_color", "roughness"]}, None
    )
    assert set(out.material.textures) == {"base_color", "roughness"}


def test_material_slot_arity_mismatch():
    with pytest.raises(OpError):
        material.evaluate([_box(), _rgb()], {"name": "m", "slots": []}, None)


def test_material_slot_channel_check():
    with pytest.raises(OpError):
        material.evaluate([_box(), _gray()], {"name": "m", "slots": ["base_color"]}, None)


def test_material_rough_metal_size_mismatch():
    with pytest.raises(OpError):
        material.evaluate(
            [_box(), _gray(size=8), _gray(size=16)],
            {"name": "m", "slots": ["roughness", "metallic"]},
            None,
        )


def test_material_on_instances_copies_prototype():
    proto = _box()
    inst = Instances(prototype=proto, transforms=np.eye(4)[None])
    out = material.evaluate([inst], {"name": "kit"}, None)
    assert out.prototype.material.name == "kit"
    assert proto.material is None  # shared prototype object not mutated


def test_material_changes_hash_and_stats():
    plain = _box()
    bound = material.evaluate([plain], {"name": "steel"}, None)
    assert plain.result_hash() != bound.result_hash()
    assert bound.stats()["material"] == "steel"
    # Different factors -> different hash (cache correctness).
    other = material.evaluate([plain], {"name": "steel", "roughness": 0.2}, None)
    assert bound.result_hash() != other.result_hash()


# -- evaluator: propagation, cache round-trip, drop warnings -----------------

_ASSET_PRESERVE = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {}},
{"id": "mt", "op": "material", "inputs": ["bx"], "params": {"name": "steel"}},
{"id": "tr", "op": "transform", "inputs": ["mt"], "params": {"translate": [0.0, 0.5, 0.0]}}
],
"outputs": {"main": "tr"}
}
"""

_ASSET_DROP = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {}},
{"id": "mt", "op": "material", "inputs": ["bx"], "params": {"name": "steel"}},
{"id": "b2", "op": "box", "params": {"size": [0.4, 2.0, 0.4]}},
{"id": "bo", "op": "boolean", "inputs": ["mt", "b2"], "params": {"mode": "union"}}
],
"outputs": {"main": "bo"}
}
"""

_ASSET_TEXTURED = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {}},
{"id": "uw", "op": "unwrap", "inputs": ["bx"], "params": {}},
{"id": "nz", "op": "noise_texture", "params": {"size": 16}},
{"id": "mt", "op": "material", "inputs": ["uw", "nz"], "params": {"name": "worn", "slots": ["roughness"]}}
],
"outputs": {"main": "mt"}
}
"""


def test_material_propagates_through_transform(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    recs = Evaluation(asset_mod.loads(_ASSET_PRESERVE)).records()
    assert recs["main"].value().material.name == "steel"
    assert recs["main"].stats["material"] == "steel"


def test_material_dropped_by_boolean_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ev = Evaluation(asset_mod.loads(_ASSET_DROP))
    recs = ev.records()
    assert recs["main"].value().material is None
    warns = [w for w in ev.attribute_drop_warnings() if w["warning"] == "material_dropped"]
    assert len(warns) == 1
    assert warns[0]["node"] == "bo"
    assert warns[0]["materials"] == ["steel"]


def test_textured_material_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(_ASSET_TEXTURED)
    r1 = Evaluation(sc).records()
    h1 = r1["main"].result_hash
    # Second evaluation loads from cache; the material (factors + texture
    # pixels) must round-trip bit-exactly, including the result hash.
    r2 = Evaluation(asset_mod.loads(_ASSET_TEXTURED)).records()
    v2 = r2["main"].value()
    assert r2["main"].key == r1["main"].key
    assert v2.material.name == "worn"
    assert v2.material.textures["roughness"].pixels.shape == (16, 16, 1)
    assert v2.result_hash() == h1
