"""``clearance``'s copy form (DESIGN 5.2): ``"outputs": [a], "self":
true`` — the spacing between one output's own copies, under its own
``tol`` and with no partner output.
"""

from __future__ import annotations

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from moraine3d.ops import array as array_op
from moraine3d.ops import box, transform
from moraine3d.verify.validate import validate_outputs


def _row(count: int, step: float, z: float = 0.0):
    """``count`` unit boxes on the ground along x, ``step`` apart
    center to center: the gap between neighbors is ``step - 1``."""
    proto = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    placed = transform.evaluate([proto], {"translate": [0.0, 0.5, z]}, None)
    return array_op.evaluate(
        [placed], {"count": count, "step_translate": [step, 0.0, 0.0]}, None
    )


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def _clearance(aid: str, **keys) -> dict:
    return {"id": aid, "kind": "clearance", "owner": "agent", **keys}


def test_copy_spacing_is_stated_without_a_partner():
    """The copy form measures one output's copy pairs and nothing else:
    an output that overlaps itself by design no longer decides the entry
    that states its neighbor's spacing (DESIGN 5.2)."""
    values = {
        "braces": _row(4, 1.5),            # neighbors 0.5 m apart
        "struts": _row(3, 0.5, z=3.0),     # copies overlap each other
    }
    rep = validate_outputs(values, asserts=[
        _clearance("pair", outputs=["braces", "struts"], self=True, min=0.25),
        _clearance("copies", outputs=["braces"], self=True, min=0.25),
    ])
    pair = _entry(rep, "pair")
    assert pair["ok"] is False
    assert {p[0].split("[")[0] for p in pair["detail"]["failing_pairs"]} == {"struts"}
    copies = _entry(rep, "copies")
    assert copies["ok"] is True
    assert copies["self_pairs"] == 6  # every pair of the four copies
    assert copies["self_min"][:2] == ["braces[0]", "braces[1]"]
    assert copies["self_min"][2] == pytest.approx(0.5, abs=1e-6)
    assert "distance" not in copies and "unchecked" not in copies


def test_an_entry_without_its_gap_fails_as_one_entry():
    """DESIGN 8.2: the parser requires clearance's `min`, and an entry
    that reaches evaluation without it (a list handed to the checker
    directly) still fails as one entry with the key named, never as a
    traceback."""
    rep = validate_outputs({"braces": _row(3, 1.5)}, asserts=[
        _clearance("bare", outputs=["braces"], self=True),
    ])
    bare = _entry(rep, "bare")
    assert bare["ok"] is False and "'min' is missing" in bare["fix"]


def test_copy_spacing_has_its_own_min():
    """`min` is the copy spacing's alone, so it can state a gap no
    partner output keeps (DESIGN 5.2)."""
    rep = validate_outputs({"braces": _row(3, 1.5)}, asserts=[
        _clearance("copies", outputs=["braces"], self=True, min=0.75),
    ])
    copies = _entry(rep, "copies")
    assert copies["ok"] is False
    assert "'braces[0]' and 'braces[1]' come within 0.5 m" in copies["fix"]
    assert [p[:2] for p in copies["detail"]["failing_pairs"]] == [
        ["braces[0]", "braces[1]"], ["braces[1]", "braces[2]"],
    ]


def test_a_plain_mesh_has_no_copy_pairs():
    """A plain mesh is one copy: no pair to judge, vacuous and legal, as
    embedded's copy form counts it (DESIGN 5.2)."""
    plain = transform.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)],
        {"translate": [0.0, 0.5, 0.0]}, None,
    )
    rep = validate_outputs({"p": plain}, asserts=[
        _clearance("copies", outputs=["p"], self=True, min=0.25),
    ])
    copies = _entry(rep, "copies")
    assert copies["ok"] is True and copies["self_pairs"] == 0


def _asset(entry: dict) -> str:
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "a1", "op": "array", "inputs": ["b1"],
             "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
            {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        ],
        "outputs": {"row": "a1", "post": "b1", "cap": "b2"},
        "groups": {"frame": ["post", "cap"]},
        "asserts": [dict(entry, owner="agent")],
    })


def test_copy_form_parse_rules():
    """One output states a copy spacing only with `self`; `self` stays
    JSON true on every form, and a group has no copies of its own
    (DESIGN 5.2)."""
    asset_mod.loads(_asset(
        {"id": "c", "kind": "clearance", "outputs": ["row"], "self": True, "min": 0.001}
    ))
    for entry, code, needle in [
        ({"outputs": ["row"]}, "bad_assert_outputs", 'write "self": true'),
        ({"outputs": ["row"], "self": ["row"]}, "bad_assert", "JSON true"),
        ({"outputs": ["frame"], "self": True}, "bad_assert_outputs", "output group"),
        ({"outputs": ["row"], "self": True, "min": 0}, "bad_assert", "finite number > 0"),
        ({"outputs": ["row"], "self": True}, "bad_assert", "needs 'min'"),
        ({"outputs": ["ghost"], "self": True}, "bad_assert_outputs", "unknown output"),
    ]:
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_asset({"id": "c", "kind": "clearance", **entry}))
        assert exc.value.code == code, entry
        assert needle in exc.value.message, entry
