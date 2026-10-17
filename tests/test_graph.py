"""Graph evaluation: edges, cycle rejection, dirty-subgraph caching,
instancing preserved through GLB export (DESIGN 4.5, 10, Invariant 6)."""

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from support.engine import evaluate_asset, Evaluation

CHAIN = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "tr01", "op": "transform", "inputs": ["bx01"], "params": {"translate": [2.0, 0.0, 0.0]}}
],
"outputs": {"main": "tr01"}
}
"""

ARRAYED = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
{"id": "ar01", "op": "array", "inputs": ["bx01"], "params": {"count": 4, "step_translate": [1.0, 0.0, 0.0]}}
],
"outputs": {"kit": "ar01"}
}
"""


def test_edges_evaluate_through(tmp_path, monkeypatch):
    """DESIGN 4.1: ``inputs`` are edges by node id, so a chain evaluates
    through its upstream node (the translated box's bounds)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(CHAIN)
    r = evaluate_asset(sc)
    assert r["main"]["stats"]["bounds"]["min"] == [1.5, -0.5, -0.5]


def test_input_ref_and_cycle_rejected():
    """DESIGN 4.1: the IR is a DAG; an input naming no node is refused at
    load as ``bad_input_ref`` and a cycle as ``cycle``."""
    doc = json.loads(CHAIN)
    doc["nodes"][1]["inputs"] = ["nope"]
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_input_ref"

    doc = json.loads(CHAIN)
    doc["nodes"][0]["inputs"] = ["tr01"]  # bx01 <-> tr01
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "cycle"


def test_dirty_subgraph_caching(tmp_path, monkeypatch):
    """A downstream param change must not re-key the upstream node."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc1 = asset_mod.loads(CHAIN)
    evaluate_asset(sc1)
    npz_before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "cache").glob("*.npz")}

    doc = json.loads(CHAIN)
    doc["nodes"][1]["params"]["translate"] = [5.0, 0.0, 0.0]
    sc2 = asset_mod.loads(json.dumps(doc))
    r2 = evaluate_asset(sc2)
    npz_after = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "cache").glob("*.npz")}

    # Exactly one new entry (the re-keyed transform); box entry untouched.
    assert len(npz_after) == len(npz_before) + 1
    for name, mtime in npz_before.items():
        assert npz_after[name] == mtime
    assert r2["main"]["stats"]["bounds"]["min"] == [4.5, -0.5, -0.5]


def test_full_asset_determinism(tmp_path, monkeypatch):
    """Determinism (DESIGN 4.4): re-evaluating the same asset reaches
    the same cache keys and, recomputed against the records those keys
    name, the same result hash at every node — not merely a warm hit."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(ARRAYED)
    r1 = evaluate_asset(sc)
    r2 = evaluate_asset(asset_mod.loads(ARRAYED))
    assert r1["kit"]["key"] == r2["kit"]["key"]
    ev = Evaluation(asset_mod.loads(ARRAYED), recompute=True)
    ev.records()
    assert ev.recompute_report() == {"nodes": 2, "matched": 2, "unrecorded": 0}


def test_instancing_survives_glb(tmp_path, monkeypatch):
    """Day-One Invariant 6: the viewer GLB comes from the shared writer,
    so an arrayed output stays one mesh and one node carrying
    ``EXT_mesh_gpu_instancing`` with every instance, never merged."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(ARRAYED)
    r = evaluate_asset(sc)
    assert r["kit"]["stats"]["instances"] == 4
    raw = open(r["kit"]["glb"], "rb").read()
    json_len = int.from_bytes(raw[12:16], "little")
    doc = json.loads(raw[20 : 20 + json_len])
    assert len(doc["meshes"]) == 1
    ext = doc["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]
    translation_acc = doc["accessors"][ext["attributes"]["TRANSLATION"]]
    assert translation_acc["count"] == 4


def test_inputs_roundtrip_canonical():
    """DESIGN 4.1: ``inputs`` serialize as a list of node ids in the
    canonical form, and a dump reloads to the same edges."""
    sc = asset_mod.loads(CHAIN)
    text = asset_mod.dumps(sc)
    assert '"inputs": ["bx01"]' in text
    sc2 = asset_mod.loads(text)
    assert sc2.nodes["tr01"].inputs == ["bx01"]


def test_deep_chain_loads_and_evaluates(tmp_path, monkeypatch):
    """A deep-but-valid linear chain is a legal authoring pattern: it
    must load and evaluate, not die at the interpreter recursion limit
    with a raw RecursionError instead of a structured error."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    n = 1200  # past the default 1000-frame recursion limit
    nodes = [{"id": "n0", "op": "box", "params": {}}]
    nodes += [
        {
            "id": f"n{i}", "op": "transform", "inputs": [f"n{i - 1}"],
            "params": {"translate": [0.0, 0.0, 0.0]},
        }
        for i in range(1, n)
    ]
    doc = {
        "build_version": "fixture", "nodes": nodes,
        "outputs": {"main": f"n{n - 1}"},
    }
    sc = asset_mod.loads(json.dumps(doc))
    out = evaluate_asset(sc)
    assert out["main"]["stats"]["tris"] == 12
