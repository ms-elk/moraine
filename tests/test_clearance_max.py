"""``clearance``'s ``max`` (DESIGN 5.2): the pair form's upper edge, a
gap that must exist and stay bounded — the closest approach between
two subjects lies in ``[min, max]``."""

from __future__ import annotations

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError, BuildError
from moraine3d.ops import array as array_op
from moraine3d.ops import box, transform
from moraine3d.verify.validate import compact, validate_outputs
from support.recipes import build_recipe, ir


def _box_at(x: float, z: float = 0.0):
    proto = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    return transform.evaluate([proto], {"translate": [x, 0.5, z]}, None)


def _values(gap: float) -> dict:
    """Two unit boxes on the ground, ``gap`` apart along x."""
    return {"post": _box_at(0.0), "lamp": _box_at(1.0 + gap)}


def _clearance(**keys) -> dict:
    return {"id": "g", "kind": "clearance", "outputs": ["post", "lamp"],
            "owner": "agent", **keys}


def _entry(values: dict, entry: dict) -> dict:
    rep = validate_outputs(values, asserts=[entry])
    return next(e for e in rep["asserts"] if e["assert"] == "g")


def test_the_closest_approach_lies_in_the_band():
    """``min`` <= distance <= ``max``, both inclusive (DESIGN 5.2): a gap
    inside the band passes with its ``distance``, one past ``max`` fails
    by that number, and one under ``min`` fails as it does without
    ``max``."""
    band = _clearance(min=0.3, max=0.5)
    e = _entry(_values(0.4), band)
    assert e["ok"] is True and e["distance"] == pytest.approx(0.4, abs=1e-6)
    assert "unchecked" not in e
    e = _entry(_values(0.7), band)
    assert e["ok"] is False and e["distance"] == pytest.approx(0.7, abs=1e-6)
    assert "are 0.7 m apart (clearance max 0.5 m)" in e["fix"]
    e = _entry(_values(0.2), band)
    assert e["ok"] is False and "clearance min 0.3 m" in e["fix"]
    assert "clearance max" not in e["fix"]


def test_the_upper_edge_takes_the_noise_floor():
    """A gap built exactly ``max`` wide reads a float32 hair past it
    (0.35 stores as 0.350000023842) and passes under the storage-noise
    floor, the entry saying so (DESIGN 5.2); a millimeter past fails."""
    e = _entry(_values(0.35), _clearance(min=0.3, max=0.35))
    assert e["distance"] > 0.35
    assert e["ok"] is True and e["noise_floor"] == 1e-6
    e = _entry(_values(0.351), _clearance(min=0.3, max=0.35))
    assert e["ok"] is False and "noise_floor" not in e


def test_the_advice_on_max_follows_the_owner():
    """A failing human-owned band means move the parts, never the bound
    (DESIGN 5.2)."""
    agent = _entry(_values(0.7), _clearance(min=0.3, max=0.5))["fix"]
    assert "bring them closer or raise 'max'" in agent
    human = _entry(_values(0.7), _clearance(min=0.3, max=0.5, owner="human"))["fix"]
    assert "bring them closer" in human and "raise" not in human
    assert "not the 'max'" in human


def test_the_upper_edge_reads_the_nearest_copy_and_says_so():
    """The least gap holds for every copy; ``max`` reads the nearest
    copy alone, so the entry names its instanced outputs under
    ``unchecked.instanced_copies`` beside the copy pairs it did not
    judge, and the default report counts both (DESIGN 5.2)."""
    lamps = array_op.evaluate(
        [_box_at(1.4)], {"count": 3, "step_translate": [3.0, 0.0, 0.0]}, None
    )
    values = {"post": _box_at(0.0), "lamp": lamps}
    rep = validate_outputs(values, asserts=[_clearance(min=0.3, max=0.5)])
    e = next(e for e in rep["asserts"] if e["assert"] == "g")
    assert e["ok"] is True and e["distance"] == pytest.approx(0.4, abs=1e-6)
    assert e["unchecked"] == {
        "instanced_copies": ["lamp"], "instanced_self_pairs": ["lamp"],
    }
    assert compact(rep)["unchecked"] == {
        "entries": 1, "instanced_copies": 1, "instanced_self_pairs": 1,
    }
    # Without `max` nothing about the pair is existential.
    e = _entry(values, _clearance(min=0.3))
    assert e["unchecked"] == {"instanced_self_pairs": ["lamp"]}


def _doc(entry: dict) -> str:
    import json

    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "a", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "b", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "c", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        ],
        "outputs": {"post": "a", "lamp": "b", "rail": "c"},
        "asserts": [entry],
    })


def test_max_is_the_pair_forms_alone():
    """Parse-time shape (DESIGN 5.2): ``max`` is a number above ``min``;
    the one-vs-rest form, the copy form, and ``self`` beside a pair
    refuse it, each with what to write, and ``every`` is answered with
    contact's."""
    asset_mod.loads(_doc(_clearance(min=0.3, max=0.5)))
    asset_mod.loads(_doc(_clearance(min=0.3, max=1)))  # an int is a number
    for keys, needle in (
        ({"min": 0.3, "max": 0.3}, "above 'min' (0.3)"),
        ({"min": 0.3, "max": 0.1}, "above 'min' (0.3)"),
        ({"min": 0.3, "max": "far"}, "above 'min' (0.3)"),
        ({"max": 0.5}, "needs 'min'"),
        ({"min": 0.3, "max": 0.5, "self": True}, "write the pair's band and the copy"),
        ({"min": 0.3, "max": 0.5, "every": ["lamp"]}, "contact's \"every\" with that 'tol'"),
    ):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_doc(_clearance(**keys)))
        assert exc.value.code == "bad_assert" and needle in str(exc.value), keys
    for keys, needle in (
        ({"outputs": ["post"], "others": True}, "write the pair that must stay close"),
        ({"outputs": ["post"], "self": True}, "has no spelling"),
    ):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_doc(_clearance(min=0.3, max=0.5, **keys)))
        assert exc.value.code == "bad_assert" and needle in str(exc.value), keys


_BEYOND_FLOAT = "1" + "0" * 400


def test_a_bound_no_float_holds_is_refused_by_its_key(tmp_path):
    """An error states what failed (DESIGN 8.2): an integer beyond a
    float's range is no finite length, and the entry is refused naming
    the key, from the IR and from the recipe alike."""
    for key, written in (("max", '"max": 0.5'), ("min", '"min": 0.3')):
        doc = _doc(_clearance(min=0.3, max=0.5)).replace(
            written, f'"{key}": {_BEYOND_FLOAT}'
        )
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(doc)
        assert exc.value.code == "bad_assert" and f"'{key}'" in str(exc.value)
    asset = (
        'mo.output("post", mo.box(id = "post", size = [1, 1, 1]))\n'
        'mo.output("lamp", mo.box(id = "lamp", size = [1, 1, 1]))\n'
        'mo.assert_clearance(id = "g", outputs = ["post", "lamp"], min = 0.3, '
        f'max = {_BEYOND_FLOAT})\n'
    )
    with pytest.raises(BuildError) as exc:
        build_recipe(tmp_path / "a", asset)
    assert exc.value.code == "bad_assert"
    assert "'max' must be a finite number" in str(exc.value)


def test_max_builds_from_the_recipe_and_follows_a_knob(tmp_path):
    """The recipe spelling reaches the IR after ``min`` (DESIGN 4.8), an
    integer widened, and a requirements file may take ``max`` whole from
    a knob."""
    asset = (
        'reach = mo.param("reach", 0.5, min = 0.4, max = 2.0)\n'
        'mo.output("post", mo.box(id = "post", size = [1, 1, 1]))\n'
        'mo.output("lamp", mo.box(id = "lamp", size = [1, 1, 1]))\n'
    )
    doc = ir(build_recipe(
        tmp_path / "a",
        asset + 'mo.assert_clearance(id = "g", outputs = ["post", "lamp"], min = 0.3, max = 1)\n',
    ))
    [entry] = doc["asserts"]
    assert entry["max"] == 1.0 and isinstance(entry["max"], float)
    assert list(entry)[:5] == ["id", "kind", "outputs", "min", "max"]
    req = (
        'mo.assert_clearance(id = "g", outputs = ["post", "lamp"], min = 0.3, '
        'max = mo.param_value("reach"))\n'
    )
    doc = ir(build_recipe(tmp_path / "b", asset, files={"asset.requirements.star": req}))
    [entry] = doc["asserts"]
    assert entry["max"] == 0.5 and entry["from_param"] == {"max": "reach"}
