"""The `triangles` and `parts` budgets (DESIGN 4.1): each declared key
gates `validate` on its own, over what the asset's outputs ship, and
counts what `stats` rolls up."""

from __future__ import annotations

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.ops import array, box, material, transform
from moraine3d.recipe.build import build
from moraine3d.recipe.version import CONTRACT, build_version
from moraine3d.verify.validate import compact, validate_outputs
from support.recipes import build_recipe, ir, write_files


def _box(x: float):
    return transform.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)],
        {"translate": [x, 0.5, 0.0]}, None,
    )


def _values() -> dict:
    """A slab, a row of three boxes, and a lone box: 12 + 36 + 12
    rendered triangles in three parts."""
    row = array.evaluate([_box(3.0)], {"count": 3, "step_translate": [2.0, 0.0, 0.0]}, None)
    return {"slab": _box(0.0), "row": row, "lone": _box(-3.0)}


def _budgets(**budgets) -> list[dict]:
    return validate_outputs(_values(), budgets=budgets).get("budgets", [])


# -- each key on its own -------------------------------------------------------


def test_a_triangle_budget_gates_alone():
    """`triangles` is the rendered triangle total, instance counts
    multiplied in, and `validate` checks each declared key on its own
    (DESIGN 4.1): the one key declared gates, and its failing entry's
    `fix` names the outputs that carry the most (DESIGN 6)."""
    (at,) = _budgets(triangles=60)
    assert at == {"budget": "triangles", "ok": True, "count": 60, "limit": 60}
    report = validate_outputs(_values(), budgets={"triangles": 59})
    (over,) = report["budgets"]
    assert report["ok"] is False
    assert (over["ok"], over["count"], over["limit"]) == (False, 60, 59)
    assert "60 rendered triangles exceed the declared budget of 59" in over["fix"]
    assert "'row' (36), 'lone' (12), 'slab' (12)" in over["fix"]
    assert "ask the human to raise mo.budget(triangles = 60) in the requirements file" in over["fix"]


def test_a_part_budget_gates_alone():
    """`parts` is the exportable parts, an instanced output one part
    (DESIGN 4.1), and the one key declared gates on its own."""
    (at,) = _budgets(parts=3)
    assert at == {"budget": "parts", "ok": True, "count": 3, "limit": 3}
    report = validate_outputs(_values(), budgets={"parts": 2})
    (over,) = report["budgets"]
    assert report["ok"] is False
    assert (over["ok"], over["count"], over["limit"]) == (False, 3, 2)
    assert "3 parts exceed the declared budget of 2" in over["fix"]
    assert "ask the human to raise mo.budget(parts = 3) in the requirements file" in over["fix"]


def test_the_report_lists_one_entry_per_declared_key_in_the_vocabulary_order():
    """DESIGN 6: the `budgets` list holds one entry per declared key in
    the vocabulary's order, and the `materials` entry alone adds
    `identities`."""
    entries = _budgets(parts=3, triangles=1, materials=1)
    assert [e["budget"] for e in entries] == ["materials", "triangles", "parts"]
    assert [e["ok"] for e in entries] == [True, False, True]
    assert [i["outputs"] for i in entries[0]["identities"]] == [["lone", "row", "slab"]]
    assert all("identities" not in e for e in entries[1:])
    # A failing entry's fix spells the declaration as the recipe does.
    (over,) = validate_outputs(
        {"a": _box(0.0), "b": material.evaluate([_box(3.0)], {"name": "steel"}, None)},
        budgets={"materials": 1},
    )["budgets"]
    assert "ask the human to raise mo.budget(materials = 2) in the requirements file" in over["fix"]
    report = validate_outputs(_values(), budgets={"parts": 3, "triangles": 1, "materials": 1})
    small = compact(report)
    assert (small["budgets_total"], small["budgets_ok"]) == (3, 2)
    assert [e["budget"] for e in small["budgets"]] == ["triangles"]


# -- the counts are the ones stats rolls up --------------------------------------

PART = 'mo.output("p", mo.transform(mo.box(id = "b", size = [1, 1, 1]), id = "p", translate = [0, 0.5, 0]))\n'
ASSET = '''n = mo.param("lamps", 0, min = 0, max = 4, label = "Lamps")
mo.output("deck", mo.transform(mo.box(id = "deck_box", size = [1, 1, 1]), id = "deck", translate = [0, 0.5, 0]))
post = mo.transform(mo.box(id = "post_box", size = [0.2, 1, 0.2]), id = "post", translate = [2, 0.5, 0])
mo.output("posts", mo.array(post, id = "posts", count = 3, step_translate = [0.5, 0, 0]))
lamp = mo.transform(mo.box(id = "lamp_box", size = [0.1, 1, 0.1]), id = "lamp", translate = [-2, 0.5, 0])
mo.output("lamps", mo.array(lamp, id = "lamps", count = n, step_translate = [-0.4, 0, 0]))
mo.output("tex", mo.noise_texture(id = "tex", seed = 1, size = 16))
mo.output("crate", mo.transform(mo.asset_ref("subassets/part.star", "p", id = "ref"), id = "crate", translate = [0, 0, 3]))
'''


def _run(argv, capsys) -> dict:
    main(argv)
    return json.loads(capsys.readouterr().out)


def test_a_budget_counts_what_stats_rolls_up(tmp_path, monkeypatch, capsys):
    """DESIGN 4.1: a budget counts what the asset's outputs ship,
    reference-borne content included, and `stats` reports the same
    counts over the same scope: an instanced output's triangles times
    its copies and one part, an image output none, and an absent
    output no part of the asset."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET, "subassets/part.star": PART,
        "yard.requirements.star": "mo.budget(triangles = 60, parts = 3)\n",
    })
    rollup = _run(["stats", str(project)], capsys)["asset"]
    assert (rollup["tris"], rollup["parts"]) == (60, 3)
    report = _run(["validate", str(project), "--full"], capsys)
    assert report["absent"] == ["lamps"]
    counts = {e["budget"]: (e["count"], e["ok"]) for e in report["budgets"]}
    assert counts == {"triangles": (60, True), "parts": (3, True)}
    # One more lamp is one more part and its twelve triangles.
    report = _run(["validate", str(project), "--param", "lamps=1"], capsys)
    assert report["ok"] is False and (report["budgets_total"], report["budgets_ok"]) == (2, 0)
    assert {e["budget"]: e["count"] for e in report["budgets"]} == {"triangles": 72, "parts": 4}


# -- the declaration -----------------------------------------------------------


def test_the_ir_lists_the_budgets_in_the_vocabulary_order(tmp_path):
    """DESIGN 4.1: root `budgets` is one inline object, its keys in the
    vocabulary's order whatever order the file's keywords came in."""
    result = build_recipe(tmp_path, PART, files={
        "asset.requirements.star": "mo.budget(parts = 1, materials = 1)\nmo.budget(triangles = 12)\n",
    })
    assert list(ir(result)["budgets"].items()) == [
        ("materials", 1), ("triangles", 12), ("parts", 1),
    ]
    line = '"budgets": {"materials": 1, "triangles": 12, "parts": 1}'
    assert line in result.target.text
    # An IR that holds them in another order is written back in this one.
    shuffled = result.target.text.replace(
        line, '"budgets": {"parts": 1, "triangles": 12, "materials": 1}',
    )
    assert asset_mod.dumps(asset_mod.loads(shuffled)) == result.target.text


@pytest.mark.real_stamp
def test_an_ir_of_the_build_before_the_two_budget_keys_is_refused(tmp_path, monkeypatch, capsys):
    """DESIGN 4.4: `build_version` covers the IR's shape, so an IR that
    carries the stamp of the contract before `triangles` and `parts`
    were checked is stale, and the loader refuses it
    (`version_mismatch`)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": PART, "yard.requirements.star": "mo.budget(triangles = 12, parts = 1)\n",
    })
    path = build(project / "yard.star").target.ir_path
    assert main(["stats", str(path)]) == 0
    capsys.readouterr()
    now = build_version()
    assert CONTRACT >= 12 and now.startswith(f"{CONTRACT}+")
    before = "11" + now[len(str(CONTRACT)):]
    text = path.read_text(encoding="utf-8")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(now, before))
    assert main(["stats", str(path)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "version_mismatch"
    assert (err["detail"]["found"], err["detail"]["expected"]) == (before, now)
