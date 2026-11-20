"""An output without copies (DESIGN 4.1, 5.2, 9.1): an `array` of
count 0 evaluates, its output is absent, and every reader of the
outputs leaves it out and names it."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.cache import records
from moraine3d.core.mesh import Instances
from moraine3d.errors import OpError
from moraine3d.ops import array, bake_instances, box, material, place, tag, transform
from support.recipes import workspace, write_files

DECK = 'mo.transform(mo.box(id = "deck_box", size = [2, 0.1, 2]), id = "deck", translate = [0, 0.05, 0])'
ASSET = f'''n = mo.param("lamps", 2, min = 0, max = 4, label = "Lamps")
mo.output("deck", {DECK})
lamp = mo.transform(mo.box(id = "lamp_box", size = [0.1, 1, 0.1]), id = "lamp", translate = [-0.6, 0.6, 0])
mo.output("lamps", mo.array(lamp, id = "lamps", count = n, step_translate = [0.4, 0, 0]))
'''
REQUIREMENTS = '''mo.assert_count(id = "lamp_count", outputs = ["lamps"], value = mo.param_value("lamps"))
mo.assert_contact(id = "lamps_stand", outputs = ["lamps", "deck"], every = ["lamps"])
'''
OPTIONAL = 'mo.optional(["lamps"], "lamps")\n'


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _project(base, requirements=REQUIREMENTS, asset=ASSET):
    return write_files(base / "pier", {
        "pier.star": asset, "pier.requirements.star": requirements,
    })


def _run(argv, capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


# -- the operators -------------------------------------------------------------


def test_an_array_of_count_0_emits_no_copy():
    """``array``: a count of 0 emits the prototype with no copy, over a
    mesh and over instances alike, and the value's stats hold no
    bounds."""
    b = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    none = array.evaluate([b], {"count": 0, "step_translate": [2.0, 0.0, 0.0]}, None)
    assert isinstance(none, Instances) and none.transforms.shape == (0, 4, 4)
    assert none.prototype is b
    assert none.stats() == {"verts": 24, "tris": 12, "instances": 0}
    row = array.evaluate([b], {"count": 3, "step_translate": [2.0, 0.0, 0.0]}, None)
    for nested in (
        array.evaluate([row], {"count": 0}, None),
        array.evaluate([none], {"count": 3}, None),
    ):
        assert nested.transforms.shape == (0, 4, 4)


def test_the_array_result_hash_is_frozen():
    """DESIGN 4.5: a count of 0 is a value the operator refused, so the
    same params keep returning the same result and `op_impl_version`
    stays 1; this hash pins that identity. IEEE float arithmetic only
    (no rotation), so any conformant platform reproduces it."""
    assert array.OP_IMPL_VERSION == 1
    b = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    row = array.evaluate(
        [b], {"count": 3, "step_translate": [2.0, 0.0, 0.5], "step_scale": 1.25}, None,
    )
    assert row.result_hash() == (
        "85748cca4c3f08bf23ea0a95a1e04fc02650cf8666c90de18ff1581c159e51e6"
    )
    grid = array.evaluate([row], {"count": 2, "step_translate": [0.0, 1.0, 0.0]}, None)
    assert grid.result_hash() == (
        "b9199eb8ff280cea05db1fa1a94828b3a0be1443a10a31684277175c5a7bbd8b"
    )


def test_an_operator_that_carries_instances_carries_none():
    """DESIGN 4.1: an operator that carries instances carries none the
    same way."""
    b = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    none = array.evaluate([b], {"count": 0}, None)
    moved = transform.evaluate([none], {"translate": [1.0, 0.0, 0.0]}, None)
    painted = material.evaluate([moved], {"name": "paint", "base_color": [1, 0, 0]}, None)
    tagged = tag.evaluate(
        [painted], {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None,
    )
    for value in (moved, painted, tagged):
        assert isinstance(value, Instances) and len(value.transforms) == 0
    assert tagged.prototype.face_groups["top"].size == 2


def test_an_operator_that_reads_a_copy_refuses_none():
    """DESIGN 4.1: an operator that reads a copy or merges them (the
    anchor of `place` and `mate`, `bake_instances`) refuses an input
    without copies (`no_copies`)."""
    b = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    top = tag.evaluate([b], {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None)
    none = array.evaluate([top], {"count": 0}, None)
    with pytest.raises(OpError) as exc:
        bake_instances.evaluate([none], {}, None)
    assert exc.value.code == "no_copies"
    with pytest.raises(OpError) as exc:
        place.evaluate([none, b], {"anchor": "top", "mode": "on_top", "instance": 0}, None)
    assert exc.value.code == "no_copies"


# -- the readers of the outputs ------------------------------------------------


def test_validate_leaves_the_output_out_and_names_it(cache, capsys):
    """DESIGN 4.1: an output whose value has no copies is absent, the
    checks leave it out, and the report names it under `absent`.
    DESIGN 5.2: an entry over it fails, and `count` alone measures an
    output that is declared and has no copies, as 0."""
    project = _project(cache)
    code, on = _run(["validate", str(project)], capsys)
    assert code == 0 and "absent" not in on and on["outputs_total"] == 2
    code, report = _run(["validate", str(project), "--param", "lamps=0", "--full"], capsys)
    assert code == 1
    assert list(report["outputs"]) == ["deck"] and report["outputs_total"] == 1
    assert report["absent"] == ["lamps"]
    entries = {a["assert"]: a for a in report["asserts"]}
    assert entries["lamp_count"]["ok"] is True and entries["lamp_count"]["measured"] == 0
    assert entries["lamps_stand"]["ok"] is False
    assert entries["lamps_stand"]["absent"] == ["lamps"]
    assert "no copies" in entries["lamps_stand"]["fix"]
    assert entries["no_overlap"]["ok"] is True


def test_count_still_measures_an_optional_output_without_copies(cache, capsys):
    """DESIGN 5.2: `count` still measures an optional output that is
    declared and has no copies, as 0, while the other entries over it
    judge nothing."""
    project = _project(cache, OPTIONAL + REQUIREMENTS)
    code, report = _run(["validate", str(project), "--param", "lamps=0", "--full"], capsys)
    assert code == 0
    entries = {a["assert"]: a for a in report["asserts"]}
    assert entries["lamp_count"]["measured"] == 0 and "unchecked" not in entries["lamp_count"]
    assert entries["lamps_stand"]["unchecked"] == {"optional": ["lamps"]}
    wrong = REQUIREMENTS.replace('mo.param_value("lamps")', "2")
    project = _project(cache / "b", OPTIONAL + wrong)
    code, report = _run(["validate", str(project), "--param", "lamps=0"], capsys)
    assert code == 1 and [a["assert"] for a in report["asserts"]] == ["lamp_count"]


def test_stats_eval_and_query_carry_the_same_key(cache, capsys):
    """DESIGN 6: `stats`, `eval`, and `query` leave an absent output
    out of their outputs and carry `absent`."""
    project = _project(cache, "", ASSET.replace('"lamps", 2', '"lamps", 0'))
    for argv in (
        ["stats", str(project)],
        ["eval", str(project)],
        ["eval", str(project), "--output", "lamps"],
        ["query", str(project), "--select", '{"normal_y": {"gt": 0.9}}'],
    ):
        code, report = _run(argv, capsys)
        assert code == 0, argv
        assert report["absent"] == ["lamps"], argv
        assert "lamps" not in report["outputs"], argv
    code, stats = _run(["stats", str(project)], capsys)
    assert (stats["asset"]["outputs"], stats["asset"]["instances"]) == (1, 0)


def test_export_writes_the_parts_the_asset_has(cache, capsys):
    """DESIGN 9.1: each named output the asset has is one part, and an
    absent one is not written."""
    project = _project(cache, OPTIONAL + REQUIREMENTS, ASSET.replace('"lamps", 2', '"lamps", 0'))
    code, report = _run(["export", str(project), "-o", str(cache / "out")], capsys)
    assert code == 0 and report["ok"] is True
    assert list(report["outputs"]) == ["deck"]
    assert report["validation"]["absent"] == ["lamps"]
    manifest = json.loads((cache / "out" / "pier.manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["outputs"]) == ["deck"]


def test_an_asset_with_no_part_is_not_exported(cache, capsys):
    """DESIGN 9.1: an export writes parts; an asset whose outputs are
    all absent has none, and the command says so in place of an empty
    file."""
    only = '\n'.join(ASSET.splitlines()[:1] + ASSET.splitlines()[2:]) + "\n"
    project = _project(cache, "", only.replace('"lamps", 2', '"lamps", 0'))
    code = main(["export", str(project), "-o", str(cache / "out")])
    error = json.loads(capsys.readouterr().out)["error"]
    assert code != 0 and error["code"] == "nothing_to_export"
    assert not (cache / "out" / "pier.glb").exists()


def test_a_reference_to_an_output_without_copies_is_absent_too(cache, capsys):
    """DESIGN 4.6: `asset_ref`'s result is the referenced output's
    value, instancing untouched, so a use that sets the count to 0 gets
    no copy, and the kit's own count is judged per that use."""
    root = workspace(cache / "ws", {
        "pier": {"pier.star": ASSET, "pier.requirements.star": OPTIONAL + REQUIREMENTS},
        "walk": {"walk.star": (
            'for part in ["deck", "lamps"]:\n'
            '    mo.output(part, mo.asset_ref(project = "pier", output = part, '
            'id = "pier/" + part, overrides = {"lamps": 0}))\n'
        )},
    })
    code, report = _run(["validate", str(root / "walk"), "--full"], capsys)
    assert code == 0
    assert list(report["outputs"]) == ["deck"] and report["absent"] == ["lamps"]
    [row] = report["ref_assets"]
    assert row["asserts_total"] == row["asserts_ok"] == 3
    assert report["unchecked"] == {"entries": 1, "optional": 1}


# -- what else reads an output ---------------------------------------------------


def test_a_record_of_no_copies_is_served_from_the_cache(cache, capsys):
    """DESIGN 4.5: a record is the node's result, and one of no copies
    is intact without bounds, so the next run reads it and the
    determinism check compares against it."""
    assert records.meta_intact({
        "result_hash": "0" * 64, "stats": {"verts": 24, "tris": 12, "instances": 0},
    })
    assert not records.meta_intact({
        "result_hash": "0" * 64, "stats": {"verts": 24, "tris": 12, "instances": 2},
    })
    project = _project(cache, "", ASSET.replace('"lamps", 2', '"lamps", 0'))
    _run(["stats", str(project)], capsys)
    _, again = _run(["stats", str(project)], capsys)
    assert again["profile"]["misses"] == 0
    code, checked = _run(["eval", str(project), "--recompute"], capsys)
    assert code == 0
    assert checked["recompute"]["matched"] == checked["recompute"]["nodes"]


def test_an_embed_left_unjudged_in_a_reference_cedes_nothing_to_its_consumer(cache, capsys):
    """DESIGN 5.2: an entry left unjudged cedes nothing, and the cede
    a consumer inherits through a reference is the one that use's
    values leave standing."""
    kit = (
        'n = mo.param("pins", 1, min = 0, max = 2, label = "Pins")\n'
        'mo.output("a", mo.box(id = "a", size = [1, 1, 1]))\n'
        'mo.output("b", mo.transform(mo.box(id = "b_box", size = [0.5, 0.5, 0.5]), '
        'id = "b", translate = [0.3, 0.2, 0]))\n'
        'pin = mo.transform(mo.box(id = "pin_box", size = [0.1, 0.1, 0.1]), '
        'id = "pin", translate = [0.3, 0.2, 0])\n'
        'mo.output("c", mo.array(pin, id = "c", count = n, step_translate = [0.2, 0, 0]))\n'
    )
    root = workspace(cache / "ws", {
        "kit": {"kit.star": kit, "kit.requirements.star": (
            'mo.group("body", ["a", "c"])\n'
            'mo.optional(["c"], "pins")\n'
            'mo.assert_embedded(id = "set", outputs = ["body", "b"], min = 0)\n'
        )},
        "use": {"use.star": (
            'for part in ["a", "b"]:\n'
            '    mo.output(part, mo.asset_ref(project = "kit", output = part, '
            'id = "kit/" + part, overrides = {"pins": 0}))\n'
        )},
    })
    _, report = _run(["validate", str(root / "use"), "--full"], capsys)
    [guard] = [
        a for a in report["asserts"] if a["assert"] == "no_overlap" and "asset" not in a
    ]
    assert guard["ok"] is False and "ceded" not in guard


def test_a_loss_reaches_no_output_without_copies(cache, capsys):
    """DESIGN 4.1: the drop warnings fire where the slot is still
    absent at an output the value flows into, and an absent output is
    no part of the asset."""
    asset = (
        'n = mo.param("posts", 1, min = 0, max = 2, label = "Posts")\n'
        f'mo.output("deck", {DECK})\n'
        'stock = mo.tag(mo.box(id = "stock", size = [0.2, 1, 0.2]), id = "stock_top", '
        'name = "top", select = {"normal_y": {"gt": 0.9}})\n'
        'hole = mo.transform(mo.box(id = "hole", size = [0.1, 0.1, 0.4]), id = "hole_at", '
        'translate = [0, 0.3, 0])\n'
        'post = mo.transform(mo.boolean(stock, hole, id = "post_cut", mode = "difference"), '
        'id = "post", translate = [0.5, 0.6, 0])\n'
        'mo.output("posts", mo.array(post, id = "posts", count = n, step_translate = [0.4, 0, 0]))\n'
    )
    project = _project(cache, "", asset)
    _, on = _run(["validate", str(project)], capsys)
    assert "groups_dropped" in {w["warning"] for w in on.get("warnings", [])}
    _, off = _run(["validate", str(project), "--param", "posts=0"], capsys)
    assert off["absent"] == ["posts"]
    assert "groups_dropped" not in {w["warning"] for w in off.get("warnings", [])}


def test_the_refusal_to_export_no_part_carries_the_recompute_verdict(cache, capsys):
    """DESIGN 6: a verdict reached before an export failed rides the
    error, so a sweep's aggregate counts the project."""
    only = "\n".join(ASSET.splitlines()[:1] + ASSET.splitlines()[2:]) + "\n"
    project = _project(cache, "", only.replace('"lamps", 2', '"lamps", 0'))
    code = main(["export", str(project), "-o", str(cache / "out"), "--recompute"])
    error = json.loads(capsys.readouterr().out)["error"]
    assert code != 0 and error["code"] == "nothing_to_export"
    assert error["detail"]["recompute"]["nodes"] >= 1
