"""The storage-noise floor on assert length comparisons (DESIGN 5.2).

Vertices are stored float32, so a gap built exactly 0.15 wide reads
0.149999973 and used to miss a clearance `min` 0.15 — the shortfall
authors dodged by writing every tolerance as 0.1499. A miss under
1e-6 m passes and the entry says so (`noise_floor`); a miss beyond it
is real; `count` compares integers and takes no floor.
"""

import json

import numpy as np

from moraine3d.core import asset as asset_mod
from moraine3d.core.asserts.common import NOISE_FLOOR
from moraine3d.verify.validate import validate_outputs
from support.engine import evaluate_records


def _pair(dx: float, entries: list[dict], *, size_b=(1.0, 1.0, 1.0)) -> str:
    """Two boxes on the ground: a unit box at the origin and box B moved
    ``dx`` along x — the gap between their facing faces is dx - 0.5 -
    size_b/2, as float32 stores it."""
    nodes = [
        {"id": "a0", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "a", "op": "transform", "inputs": ["a0"], "params": {"translate": [0.0, 0.5, 0.0]}},
        {"id": "b0", "op": "box", "params": {"size": list(size_b)}},
        {"id": "b", "op": "transform", "inputs": ["b0"], "params": {"translate": [dx, size_b[1] / 2, 0.0]}},
    ]
    return json.dumps({"build_version": "fixture", "nodes": nodes, "outputs": {"a": "a", "b": "b"}, "asserts": entries})


def _entry(asset_json: str, aid: str) -> dict:
    sc = asset_mod.loads(asset_json)
    recs = evaluate_records(sc)
    report = validate_outputs({n: r.value() for n, r in recs.items()}, asserts=sc.asserts)
    return next(e for e in report["asserts"] if e["assert"] == aid)


def test_float32_shortfall_under_the_floor_passes_and_is_marked():
    """A clearance built exactly at its `min` reads a few 1e-8 short in
    float32; the entry passes and carries `noise_floor`, the measured
    distance still the exact number."""
    assert float(np.float32(0.65)) < 0.65  # the shortfall the asset below builds
    e = _entry(_pair(1.15, [
        {"id": "gap", "kind": "clearance", "outputs": ["a", "b"], "min": 0.15, "owner": "human"},
    ]), "gap")
    assert e["ok"] is True and e["noise_floor"] == NOISE_FLOOR
    assert 0.15 - 1e-6 < e["distance"] < 0.15


def test_a_miss_beyond_the_floor_is_real():
    """A gap 1e-4 short of its tol fails as before, with no floor key."""
    e = _entry(_pair(1.1499, [
        {"id": "gap", "kind": "clearance", "outputs": ["a", "b"], "min": 0.15, "owner": "human"},
    ]), "gap")
    assert e["ok"] is False and "noise_floor" not in e
    assert "come within 0.1499 m" in e["fix"]
    # A clean pass carries no key either: the floor is reported only
    # when it decided the verdict.
    e = _entry(_pair(1.2, [
        {"id": "gap", "kind": "clearance", "outputs": ["a", "b"], "min": 0.15, "owner": "human"},
    ]), "gap")
    assert e["ok"] is True and "noise_floor" not in e


def test_contact_and_every_take_the_floor_on_the_far_side():
    """Contact's bound is an upper one: a gap a few 1e-9 over an explicit
    `tol` passes under the floor, for the union check and for the
    per-copy quantifier alike."""
    assert float(np.float32(0.55)) > 0.55
    entries = [
        {"id": "near", "kind": "contact", "outputs": ["a", "b"], "tol": 0.05, "owner": "human"},
        {"id": "near_every", "kind": "contact", "outputs": ["a", "b"], "every": ["b"],
         "tol": 0.05, "owner": "human"},
    ]
    asset = _pair(1.05, entries)
    for aid in ("near", "near_every"):
        e = _entry(asset, aid)
        assert e["ok"] is True and e["noise_floor"] == NOISE_FLOOR, aid
        assert 0.05 < e["distance"] < 0.05 + 1e-6


def test_extent_takes_the_floor_and_count_does_not():
    """An exact-sum dimension with `tol` 0 survives its float32 storage;
    a copy count is an integer comparison and never carries the key."""
    asset = _pair(2.0, [
        {"id": "width", "kind": "extent", "outputs": ["b"], "measure": "size_x",
         "value": 0.82, "tol": 0, "owner": "human"},
        {"id": "at_least", "kind": "extent", "outputs": ["b"], "measure": "size_x",
         "min": 0.82, "owner": "human"},
        {"id": "at_most", "kind": "extent", "outputs": ["b"], "measure": "size_x",
         "max": 0.82, "owner": "human"},
        {"id": "one", "kind": "count", "outputs": ["b"], "value": 1, "owner": "agent"},
    ], size_b=(0.82, 1.0, 1.0))
    width = _entry(asset, "width")
    assert width["ok"] is True and width["noise_floor"] == NOISE_FLOOR
    assert width["measured"] != 0.82 and abs(width["measured"] - 0.82) < 1e-6
    # The storage error lands on one side: that band edge is the one the
    # floor decides, the other passes clean.
    edges = [_entry(asset, aid) for aid in ("at_least", "at_most")]
    assert all(e["ok"] for e in edges)
    assert sorted("noise_floor" in e for e in edges) == [False, True]
    one = _entry(asset, "one")
    assert one["ok"] is True and "noise_floor" not in one


def test_a_failing_entry_never_carries_the_floor_key():
    """`noise_floor` marks the green verdict the floor decided. An entry
    that makes many comparisons can waive one and still fail on
    another — a one-vs-rest clearance over several partners — and the
    key must not then read a real violation as a rounding artifact."""
    nodes = [
        {"id": "a0", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "a", "op": "transform", "inputs": ["a0"], "params": {"translate": [0.0, 0.5, 0.0]}},
        # b sits exactly `tol` away (float32-short: the floor waives it)
        {"id": "b0", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "b", "op": "transform", "inputs": ["b0"], "params": {"translate": [1.15, 0.5, 0.0]}},
        # c is 0.1 m inside the tol: a real miss, five orders over the floor
        {"id": "c0", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "c", "op": "transform", "inputs": ["c0"], "params": {"translate": [-1.05, 0.5, 0.0]}},
    ]
    entry = {"id": "clear", "kind": "clearance", "outputs": ["a"], "others": True,
             "min": 0.15, "owner": "human"}
    asset = json.dumps({
        "build_version": "fixture", "nodes": nodes,
        "outputs": {"a": "a", "b": "b", "c": "c"}, "asserts": [entry],
    })
    e = _entry(asset, "clear")
    assert e["ok"] is False and "noise_floor" not in e
    assert "'a' and 'c' come within 0.05 m" in e["fix"]
    # Drop the real violation and the same waived partner marks the pass.
    doc = json.loads(asset)
    doc["nodes"] = nodes[:4]
    doc["outputs"] = {"a": "a", "b": "b"}
    e = _entry(json.dumps(doc), "clear")
    assert e["ok"] is True and e["noise_floor"] == NOISE_FLOOR


def test_embedded_band_edges_take_the_floor():
    """An embedded band edge a hair past the measured depth passes under
    the floor; one beyond it fails."""
    # A ceiling keeps the probe measuring (min 0 alone measures nothing).
    probe = _entry(_pair(0.98, [
        {"id": "emb", "kind": "embedded", "outputs": ["a", "b"], "min": 0, "max": 1.0,
         "owner": "agent"},
    ]), "emb")
    depth = probe["depth"]
    within = _entry(_pair(0.98, [
        {"id": "emb", "kind": "embedded", "outputs": ["a", "b"], "min": depth + 5e-7, "owner": "agent"},
    ]), "emb")
    assert within["ok"] is True and within["noise_floor"] == NOISE_FLOOR
    beyond = _entry(_pair(0.98, [
        {"id": "emb", "kind": "embedded", "outputs": ["a", "b"], "min": depth + 5e-6, "owner": "agent"},
    ]), "emb")
    assert beyond["ok"] is False and "noise_floor" not in beyond
