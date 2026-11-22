"""A number no float holds (DESIGN 8.2): an integer literal beyond a
float's range is refused by whoever reads it, with the error that
names what failed, where ``math.isfinite`` would raise past every
refusal."""

from __future__ import annotations

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core import numbers
from moraine3d.errors import AssetError, BuildError, MoraineError
from support.engine import evaluate_records
from support.recipes import build_recipe

_BEYOND = "1" + "0" * 400
_MARK = 7777  # replaced by the literal once the document is JSON text


def _doc(**extra) -> str:
    doc = {
        "build_version": "fixture",
        "nodes": [{"id": "a", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}],
        "outputs": {"post": "a"},
        **extra,
    }
    return json.dumps(doc).replace(str(_MARK), _BEYOND)


def test_is_finite_answers_for_every_number():
    """One reading of "a finite number" for every caller: a bool is no
    number, an infinity and an integer beyond a float are not finite,
    and the answer is never an exception."""
    assert numbers.is_finite(0) and numbers.is_finite(-2.5) and numbers.is_finite(10**300)
    for value in (True, None, "1", float("inf"), float("nan"), 10**400, -(10**400)):
        assert numbers.is_finite(value) is False


@pytest.mark.parametrize("decl, code, needle", [
    ({"value": _MARK, "min": 0.5, "max": 2.0}, "bad_param_value", "a knob is a finite number"),
    ({"value": [1.0, _MARK], "min": 0.5, "max": 2.0}, "bad_param_value", "a knob is a finite number"),
    ({"value": 1.0, "min": _MARK, "max": 2.0}, "bad_param_decl", "'min' must be a finite number"),
    ({"value": 1.0, "min": 0.5, "max": _MARK}, "bad_param_decl", "'max' must be a finite number"),
    ({"value": 1.0, "min": 0.5, "max": 2.0, "step": _MARK}, "bad_param_decl",
     "'step' must be a finite number"),
])
def test_a_knob_refuses_a_number_no_float_holds(decl, code, needle):
    """A knob's value and range are finite numbers (DESIGN 4.1): the IR
    is refused naming the knob and the key."""
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc(params=[{"id": "w", **decl}]))
    assert exc.value.code == code and needle in str(exc.value)
    assert "param 'w'" in str(exc.value)


@pytest.mark.parametrize("call, code", [
    (f'mo.param("w", {_BEYOND}, min = 0, max = 2)', "bad_param_value"),
    (f'mo.param("w", 1.0, min = 0, max = {_BEYOND})', "bad_param_decl"),
])
def test_the_recipe_refuses_the_same_knob(tmp_path, call, code):
    """The build answers with the same refusal as the IR's loader
    (DESIGN 4.8), at the call that declared the knob."""
    recipe = f'w = {call}\nmo.output("post", mo.box(id = "post", size = [1, 1, 1]))\n'
    with pytest.raises(BuildError) as exc:
        build_recipe(tmp_path / "a", recipe)
    assert exc.value.code == code and "param 'w'" in str(exc.value)


@pytest.mark.parametrize("comparator, needle", [
    ({"gt": _MARK}, "'gt' must be a finite number"),
    ({"eq": 1.0, "tol": _MARK}, "'tol' must be a finite number >= 0"),
])
def test_a_predicate_refuses_a_number_no_float_holds(
    tmp_path, monkeypatch, comparator, needle,
):
    """A face predicate compares against finite numbers (DESIGN 5.1):
    the op that reads it fails with ``bad_predicate`` naming the field
    and the comparator."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    tag = {"id": "t", "op": "tag", "inputs": ["a"],
           "params": {"name": "top", "select": {"normal_y": comparator}}}
    doc = json.loads(_doc())
    doc["nodes"].append(tag)
    doc["outputs"] = {"post": "t"}
    sc = asset_mod.loads(json.dumps(doc).replace(str(_MARK), _BEYOND))
    with pytest.raises(MoraineError) as exc:
        evaluate_records(sc)
    assert exc.value.code == "bad_predicate" and needle in str(exc.value)
