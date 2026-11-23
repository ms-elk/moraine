"""``offset`` (DESIGN 5.2): the position of the second named output less
the first's along one axis, each side read from its world-space vertex
bounds with its copies united — a requirement stated from another part,
which an absolute ``extent`` loses when the reference moves."""

from __future__ import annotations

import json

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Image
from moraine3d.errors import AssetError
from moraine3d.ops import array as array_op
from moraine3d.ops import box, transform
from moraine3d.verify.validate import compact, validate_outputs
from support.recipes import build_recipe, ir


def _box(center: list[float], size: list[float]):
    proto = box.evaluate([], {"size": [float(s) for s in size]}, None)
    return transform.evaluate([proto], {"translate": [float(c) for c in center]}, None)


def _pier(deck_top: float = 1.0, rail_top: float = 2.1) -> dict:
    """A deck slab and a railing above it, each top where asked."""
    return {
        "deck": _box([0.0, deck_top - 0.5, 0.0], [4.0, 1.0, 2.0]),
        "railing": _box([0.0, rail_top - 0.5, 0.9], [4.0, 1.0, 0.1]),
    }


def _offset(**keys) -> dict:
    return {"id": "o", "kind": "offset", "outputs": ["deck", "railing"],
            "measure": "max_y", "owner": "agent", **keys}


def _entry(values: dict, entry: dict, **kw) -> dict:
    rep = validate_outputs(values, asserts=[entry], **kw)
    return next(e for e in rep["asserts"] if e["assert"] == entry["id"])


# -- the measurement -----------------------------------------------------------


def test_the_number_is_the_second_output_less_the_reference():
    """`"outputs": [a, b]`, the reference first: the position of `b`
    less the position of `a`, and a negative number is a position
    before the reference (DESIGN 5.2). The entry echoes its `measure`
    and its requirement and carries `measured`, pass or fail."""
    e = _entry(_pier(), _offset(value=1.1))
    assert e["ok"] is True and e["measured"] == pytest.approx(1.1, abs=1e-6)
    assert (e["measure"], e["value"], e["tol"]) == ("max_y", 1.1, 1e-3)
    back = _entry(_pier(), _offset(outputs=["railing", "deck"], value=-1.1))
    assert back["ok"] is True and back["measured"] == pytest.approx(-1.1, abs=1e-6)
    miss = _entry(_pier(), _offset(value=1.0))
    assert miss["ok"] is False and miss["measured"] == pytest.approx(1.1, abs=1e-6)


def test_the_offset_follows_the_reference_where_a_height_does_not():
    """An offset moves when either part moves (DESIGN 5.2): the deck
    raised under a railing that stayed fails the offset while the
    railing's absolute height still reads true, and both raised
    together keep it."""
    height = {"id": "h", "kind": "extent", "outputs": ["railing"],
              "measure": "max_y", "value": 2.1, "owner": "agent"}
    raised_deck = _pier(deck_top=1.3)
    assert _entry(raised_deck, height)["ok"] is True
    e = _entry(raised_deck, _offset(value=1.1))
    assert e["ok"] is False and e["measured"] == pytest.approx(0.8, abs=1e-6)
    together = _pier(deck_top=1.3, rail_top=2.4)
    assert _entry(together, _offset(value=1.1))["ok"] is True
    assert _entry(together, height)["ok"] is False


def test_a_pair_reads_each_side_at_its_own_position():
    """`measure` is one position token read on both sides, or a pair in
    the order of `outputs` where the sides differ: `["center_x",
    "max_x"]` is an arm's reach from its pole's axis (DESIGN 5.2)."""
    values = {
        "pole": _box([0.0, 1.5, 0.0], [0.2, 3.0, 0.2]),
        "arm": _box([0.6, 2.8, 0.0], [1.2, 0.1, 0.1]),
    }
    reach = {"id": "o", "kind": "offset", "outputs": ["pole", "arm"],
             "measure": ["center_x", "max_x"], "value": 1.2, "owner": "agent"}
    e = _entry(values, reach)
    assert e["ok"] is True and e["measured"] == pytest.approx(1.2, abs=1e-6)
    assert e["measure"] == ["center_x", "max_x"]
    single = _entry(values, {**reach, "measure": "max_x"})
    assert single["measured"] == pytest.approx(1.1, abs=1e-6)


def test_the_judgment_is_a_value_or_a_band():
    """The judgment is `extent`'s (DESIGN 5.2): `value` within `tol`, or
    a `min`/`max` band, both inclusive, the omitted edge echoed as
    `null`."""
    assert _entry(_pier(), _offset(value=1.0, tol=0.11))["ok"] is True
    band = _entry(_pier(), _offset(min=1.0, max=1.2))
    assert band["ok"] is True and (band["min"], band["max"]) == (1.0, 1.2)
    low = _entry(_pier(), _offset(min=1.2))
    assert low["ok"] is False and low["max"] is None
    assert "below min 1.2 m" in low["fix"]
    assert _entry(_pier(), _offset(max=1.0))["ok"] is False


def test_a_failing_entry_says_where_each_side_stands():
    """A red entry states what failed and the shortest path to a fix
    (DESIGN 8.2): both readings, the offset between them, and the miss;
    tolerance-weakening advice only on an agent-owned entry (DESIGN
    5.2)."""
    agent = _entry(_pier(), _offset(value=1.0))["fix"]
    assert agent.startswith(
        "output 'railing' max_y 2.1 m is +1.1 m from output 'deck' max_y 1 m, "
        "against value 1 m +/- 0.001 m (+0.1 m off); move either part or "
    )
    assert "state the intended offset" in agent
    human = _entry(_pier(), _offset(value=1.0, owner="human"))["fix"]
    assert "move either part - a failing human-owned assertion" in human
    assert "state the intended" not in human


def test_a_stated_offset_takes_the_noise_floor():
    """Every length comparison against an authored `tol` carries the
    storage-noise floor: an offset built exactly as stated reads a
    float32 hair off it and passes, the entry saying `noise_floor`
    (DESIGN 5.2)."""
    e = _entry(_pier(), _offset(value=1.1, tol=0))
    assert e["measured"] != 1.1
    assert e["ok"] is True and e["noise_floor"] == 1e-6
    off = _entry(_pier(rail_top=2.101), _offset(value=1.1, tol=0))
    assert off["ok"] is False and "noise_floor" not in off


# -- what a name stands for ----------------------------------------------------


def test_an_instanced_side_is_read_with_its_copies_united():
    """Each side is read with its copies united, as `extent`'s output
    form reads them, and the union is the requirement, not a weakened
    per-copy one: no `unchecked` (DESIGN 5.2)."""
    post = _box([0.0, 1.6, 0.9], [0.1, 1.2, 0.1])
    posts = array_op.evaluate([post], {"count": 3, "step_translate": [1.5, 0.1, 0.0]}, None)
    values = {"deck": _pier()["deck"], "posts": posts}
    e = _entry(values, _offset(outputs=["deck", "posts"], value=1.4))
    assert e["ok"] is True and e["measured"] == pytest.approx(1.4, abs=1e-6)
    assert "unchecked" not in e
    far = _entry(values, _offset(outputs=["deck", "posts"], measure="max_x", value=1.05))
    assert far["measured"] == pytest.approx(1.05, abs=1e-6)


def test_a_group_is_read_as_its_members_united():
    """A root `groups` name resolves to the union of its members on
    either side, and the entry echoes the resolution under `members`
    (DESIGN 5.2)."""
    values = {
        "slab": _box([0.0, 0.5, 0.0], [4.0, 1.0, 2.0]),
        "kerb": _box([0.0, 1.1, -0.9], [4.0, 0.2, 0.2]),
        "railing": _pier()["railing"],
    }
    e = _entry(
        values, _offset(value=0.9), groups={"deck": ["slab", "kerb"]},
    )
    assert e["ok"] is True and e["measured"] == pytest.approx(0.9, abs=1e-6)
    assert e["members"] == {"deck": ["slab", "kerb"]}


def test_an_image_side_is_unevaluatable():
    """An unevaluatable entry fails with the reason (DESIGN 5.2), its
    requirement still echoed."""
    values = {**_pier(), "railing": Image(pixels=np.full((4, 4, 3), 0.5, np.float32))}
    e = _entry(values, _offset(value=1.1))
    assert e["ok"] is False and "measured" not in e
    assert "railing" in e["fix"] and e["value"] == 1.1


# -- outputs the asset does not have -------------------------------------------


def test_an_absent_output_fails_the_entry_as_itself():
    """An entry that names an absent output fails with the name, and
    one that judged nothing has no measurement to carry (DESIGN 5.2):
    the entry keeps its `measure` and carries neither requirement nor
    `measured`."""
    values = {"deck": _pier()["deck"]}
    e = _entry(values, _offset(value=1.1, owner="human"), absent=["railing"])
    assert e["ok"] is False and e["absent"] == ["railing"]
    assert e["measure"] == "max_y"
    assert not {"value", "tol", "measured"} & set(e)


def test_an_optional_output_is_excused_only_while_it_is_absent():
    """An entry whose absent outputs are all optional under a knob that
    is off judges nothing: `ok` is true and `unchecked` names what was
    not there. An optional output that is present is judged whatever
    its knob holds (DESIGN 5.2)."""
    values = {"deck": _pier()["deck"]}
    entry = _offset(value=1.1, owner="human")
    off = validate_outputs(
        values, asserts=[entry], absent=["railing"], excused=["railing"],
    )
    e = next(x for x in off["asserts"] if x["assert"] == "o")
    assert e["ok"] is True and e["unchecked"] == {"optional": ["railing"]}
    assert e["measure"] == "max_y" and "measured" not in e
    assert compact(off)["unchecked"] == {"entries": 1, "optional": 1}
    present = _entry(_pier(rail_top=2.5), entry, excused=["railing"])
    assert present["ok"] is False and present["measured"] == pytest.approx(1.5, abs=1e-6)


# -- parse-time shape ----------------------------------------------------------


def _doc(entry: dict, groups: dict | None = None) -> str:
    doc = {
        "build_version": "fixture",
        "nodes": [
            {"id": n, "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
            for n in ("a", "b", "c")
        ],
        "outputs": {"deck": "a", "railing": "b", "kerb": "c"},
        "asserts": [entry],
    }
    if groups:
        doc["groups"] = groups
    return json.dumps(doc)


def _refused(entry: dict, groups: dict | None = None) -> AssetError:
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc(entry, groups))
    return exc.value


def test_a_well_formed_entry_parses_in_both_measure_forms():
    """`measure` is one position token, `min|max|center` x `x|y|z`, or
    a pair along one axis; a band edge may be negative (DESIGN 5.2)."""
    asset_mod.loads(_doc(_offset(value=1.1)))
    asset_mod.loads(_doc(_offset(measure=["min_x", "max_x"], min=-0.2, max=0.0)))


def test_a_size_is_refused_as_no_position():
    """A `size_*` is refused: an offset moves when either part moves and
    a difference of sizes does not (DESIGN 5.2). The refusal sends one
    output's size to `extent` and says no kind compares two."""
    for measure in ("size_y", ["size_x", "max_x"]):
        err = _refused(_offset(measure=measure, value=0))
        assert err.code == "bad_assert"
        msg = str(err)
        assert "offset reads positions" in msg
        assert "extent measures one output's own size" in msg
        assert "no kind compares the sizes of two outputs" in msg


def test_a_pair_names_two_positions_along_one_axis():
    """A pair of one token is refused, being the single form, and both
    tokens of a pair lie along one axis (DESIGN 5.2)."""
    same = _refused(_offset(measure=["max_y", "max_y"], value=1.1))
    assert 'write it once ("measure": "max_y")' in str(same)
    crossed = _refused(_offset(measure=["max_x", "max_y"], value=1.1))
    assert "an offset lies along one axis" in str(crossed)
    for measure in (["max_y"], ["min_y", "max_y", "center_y"], "top", None, 3):
        err = _refused(_offset(measure=measure, value=1.1))
        assert err.code == "bad_assert" and "needs 'measure'" in str(err)


def test_the_two_names_stand_for_disjoint_outputs():
    """The two names a relating kind takes must stand for disjoint sets
    of outputs, a group resolved to its members (DESIGN 5.2)."""
    for outs in (["deck", "deck"], ["deck"], ["deck", "railing", "kerb"], "deck"):
        err = _refused(_offset(outputs=outs, value=0))
        assert err.code == "bad_assert_outputs"
        assert "2 distinct output names, the reference first" in str(err)
    shared = _refused(
        _offset(outputs=["deck", "floor"], value=0), {"floor": ["deck", "kerb"]},
    )
    assert shared.code == "bad_assert_outputs"
    assert "both stand for ['deck']" in str(shared)
    assert "measured from itself" in str(shared)
    unknown = _refused(_offset(outputs=["deck", "lamp"], value=0))
    assert "unknown output 'lamp'" in str(unknown)


def test_every_and_faces_are_refused_with_what_to_write():
    """The entry takes outputs and groups alone and no `every` (DESIGN
    5.2); each refusal is the kind's own and names the form that does
    exist (DESIGN 8.2)."""
    every = str(_refused(_offset(value=1.1, every=["railing"])))
    assert "'every' does not apply to offset" in every and "copies united" in every
    faces = str(_refused({
        "id": "o", "kind": "offset", "measure": "max_y", "value": 1.1,
        "faces": [["deck", "top"], ["railing", "top"]], "owner": "agent",
    }))
    assert "'faces' does not apply to offset" in faces and "extent" in faces


def test_one_entry_states_one_judgment():
    """`value` within `tol` or a `min`/`max` band, never both, and a
    `tol` is a finite number >= 0 beside `value` (DESIGN 5.2)."""
    assert "not both" in str(_refused(_offset(value=1.1, min=1.0)))
    assert "states no requirement" in str(_refused(_offset()))
    assert "'tol' is the allowance around 'value'" in str(_refused(_offset(min=1.0, tol=0.01)))
    assert "finite number >= 0" in str(_refused(_offset(value=1.1, tol=-1)))


# -- the recipe ----------------------------------------------------------------

ASSET = (
    'rise = mo.param("rail_height", 1.1, min = 0.9, max = 1.3)\n'
    'deck = mo.transform(mo.box(id = "dk", size = [4, 1, 2]), id = "dk2", translate = [0, 0.5, 0])\n'
    'rail = mo.box(id = "rl", size = [4, rise, 0.1])\n'
    'mo.output("deck", deck)\n'
    'mo.output("railing", mo.place(base = mo.tag(deck, id = "tp", name = "top", '
    'select = {"normal_y": {"gt": 0.9}}), part = rail, id = "pl", anchor = "top"))\n'
)


def test_the_recipe_call_builds_the_entry(tmp_path):
    """Each entry is one recipe call, `mo.assert_offset(...)`, written
    to the IR with the keys in their fixed order (DESIGN 4.1), and
    human-owned when a requirements file declares it (DESIGN 5.2)."""
    result = build_recipe(tmp_path, ASSET, files={"asset.requirements.star": (
        'mo.assert_offset(id = "rail_height", outputs = ["deck", "railing"], '
        'measure = "max_y", min = 1.0, max = 1.2)\n'
        'mo.assert_offset(id = "rail_flush", outputs = ["deck", "railing"], '
        'measure = ["min_x", "max_x"], value = 4)\n'
    )})
    height, flush = ir(result)["asserts"]
    assert list(height) == ["id", "kind", "outputs", "measure", "min", "max", "owner", "site"]
    assert height["kind"] == "offset" and height["owner"] == "human"
    assert flush["measure"] == ["min_x", "max_x"] and flush["value"] == 4.0


def test_a_threshold_may_follow_a_knob(tmp_path):
    """A human-owned threshold may take a knob's value whole
    (`mo.param_value`, DESIGN 4.8), and `from_param` records which: the
    stated offset then moves with the knob the human named."""
    result = build_recipe(tmp_path, ASSET, files={"asset.requirements.star": (
        'mo.assert_offset(id = "rail_height", outputs = ["deck", "railing"], '
        'measure = "max_y", value = mo.param_value("rail_height"))\n'
    )})
    (entry,) = ir(result)["asserts"]
    assert entry["value"] == 1.1 and entry["from_param"] == {"value": "rail_height"}
