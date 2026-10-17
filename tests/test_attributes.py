"""Mesh attribute model (DESIGN 4.1): glTF vocabulary,
cache round-trip, hash participation, op preserve/drop contracts."""

import numpy as np
import pytest

from moraine3d.cache.records import load_value, save_value
from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Mesh
from moraine3d.ops import boolean, box, noise_displace, sphere, transform, unwrap
from support.engine import Evaluation


def _colored_box():
    m = box.evaluate([], {}, None)
    m.attributes = {"COLOR_0": np.linspace(0, 1, len(m.vertices) * 4, dtype=np.float32).reshape(-1, 4)}
    return m


def test_vocabulary_is_gltf_verbatim():
    with pytest.raises(ValueError):
        Mesh(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.array([[0, 1, 2]], np.int32),
            attributes={"vertex_color": np.zeros((3, 4), np.float32)},
        )
    with pytest.raises(ValueError):  # TEXCOORD_0 lives in the uv field
        Mesh(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.array([[0, 1, 2]], np.int32),
            attributes={"TEXCOORD_0": np.zeros((3, 2), np.float32)},
        )
    with pytest.raises(ValueError):  # row count must match vertices
        Mesh(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.array([[0, 1, 2]], np.int32),
            attributes={"COLOR_0": np.zeros((2, 4), np.float32)},
        )


def test_attributes_join_hash_stats_and_gltf_view():
    plain = box.evaluate([], {}, None)
    colored = _colored_box()
    assert plain.result_hash() != colored.result_hash()
    assert colored.stats()["attributes"] == ["COLOR_0"]
    view = colored.gltf_attributes()
    assert set(view) == {"NORMAL", "COLOR_0"}
    uvd = unwrap.evaluate([colored], {"mode": "box"}, None)
    assert uvd.stats()["attributes"] == ["COLOR_0", "TEXCOORD_0"]


def test_attributes_cache_round_trip(tmp_path):
    colored = _colored_box()
    p = tmp_path / "rec.npz"
    save_value(colored, p)
    loaded = load_value(p)
    assert np.array_equal(loaded.attributes["COLOR_0"], colored.attributes["COLOR_0"])
    assert loaded.result_hash() == colored.result_hash()


def test_preserving_ops_carry_attributes():
    colored = _colored_box()
    moved = transform.evaluate([colored], {"translate": [1.0, 2.0, 3.0]}, None)
    assert np.array_equal(moved.attributes["COLOR_0"], colored.attributes["COLOR_0"])
    # noise_displace refuses flat-shaded inputs (its semantics), so its
    # preservation leg runs on a smooth-shaded sphere.
    smooth = sphere.evaluate([], {}, None)
    smooth.attributes = {
        "COLOR_0": np.linspace(
            0, 1, len(smooth.vertices) * 4, dtype=np.float32
        ).reshape(-1, 4)
    }
    bumpy = noise_displace.evaluate([smooth], {"amplitude": 0.01}, None)
    assert np.array_equal(bumpy.attributes["COLOR_0"], smooth.attributes["COLOR_0"])
    # unwrap box mode re-indexes split vertices, keeping per-vertex values.
    uvd = unwrap.evaluate([colored], {"mode": "box"}, None)
    assert len(uvd.attributes["COLOR_0"]) == len(uvd.vertices)


def test_dropping_op_declares_and_validate_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    assert boolean.ATTRIBUTE_POLICY == "dropped"
    asset_text = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {}},
{"id": "u1", "op": "unwrap", "inputs": ["b1"], "params": {}},
{"id": "b2", "op": "box", "params": {"size": [0.4, 2.0, 0.4]}},
{"id": "d1", "op": "boolean", "inputs": ["u1", "b2"], "params": {"mode": "difference"}}
],
"outputs": {"main": "d1"}
}"""
    path = tmp_path / "warn.mora"
    path.write_text(asset_text, encoding="utf-8")
    sc = asset_mod.load(path)
    ev = Evaluation(sc)
    ev.records()
    warnings = ev.attribute_drop_warnings()
    assert len(warnings) == 1
    w = warnings[0]
    assert w["node"] == "d1" and w["op"] == "boolean"
    assert w["dropped"] == ["TEXCOORD_0"]
    assert "unwrap" in w["note"]


def test_transform_rotates_tangents():
    """TANGENT directions map through the linear part (renormalized) and
    a mirroring transform flips the handedness sign w (transform
    semantics) — carried verbatim, a rotated mesh would keep its
    old-world tangent frame and skew every normal map sampled through
    it."""
    m = box.evaluate([], {}, None)
    n = len(m.vertices)
    tan = np.zeros((n, 4), np.float32)
    tan[:, 0] = 1.0  # +X tangents
    tan[:, 3] = 1.0
    m.attributes = {"TANGENT": tan}
    r = transform.evaluate([m], {"rotate_deg": [0.0, 90.0, 0.0]}, None)
    got = r.attributes["TANGENT"]
    assert np.allclose(got[:, :3], [0.0, 0.0, -1.0], atol=1e-6)  # +X -> -Z
    assert np.allclose(got[:, 3], 1.0)
    mirrored = transform.evaluate([m], {"scale": [-1.0, 1.0, 1.0]}, None)
    got = mirrored.attributes["TANGENT"]
    assert np.allclose(got[:, :3], [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(got[:, 3], -1.0)  # handedness flips under mirror


def test_drop_warnings_only_for_losses_that_reach_an_output(tmp_path, monkeypatch):
    """attributes_dropped is restricted to losses that reach an output
    (DESIGN 4.1): a boolean's UV loss re-authored by an unwrap before
    the output never ships and is not reported; one that reaches the
    output is, naming the outputs; and a boolean that only anchors a
    place (never itself an output) reaches no output and is silent."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    common = [
        {"id": "b1", "op": "box", "params": {}},
        {"id": "u1", "op": "unwrap", "inputs": ["b1"], "params": {}},
        {"id": "b2", "op": "box", "params": {"size": [0.4, 2.0, 0.4]}},
        {"id": "d1", "op": "boolean", "inputs": ["u1", "b2"], "params": {"mode": "difference"}},
    ]

    def warnings_of(extra, outputs):
        doc = {"build_version": "fixture", "nodes": common + extra, "outputs": outputs}
        path = tmp_path / "w.mora"
        path.write_text(json.dumps(doc), encoding="utf-8")
        ev = Evaluation(asset_mod.load(path))
        ev.records()
        return ev.attribute_drop_warnings()

    import json

    # Re-authored downstream: the loss never ships.
    reauthored = [{"id": "u2", "op": "unwrap", "inputs": ["d1"], "params": {}}]
    assert warnings_of(reauthored, {"main": "u2"}) == []
    # Reaching the output bare: reported, outputs named.
    (w,) = warnings_of([], {"main": "d1"})
    assert w["dropped"] == ["TEXCOORD_0"] and w["outputs"] == ["main"]
    # Anchor base only: the boolean's value never ships.
    anchoring = [
        {"id": "t1", "op": "tag", "inputs": ["d1"],
         "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
        {"id": "p1", "op": "box", "params": {"size": [0.2, 0.2, 0.2]}},
        {"id": "pl", "op": "place", "inputs": ["t1", "p1"],
         "params": {"anchor": "top", "mode": "on_top"}},
    ]
    assert warnings_of(anchoring, {"main": "pl"}) == []
