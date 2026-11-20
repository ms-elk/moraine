"""Absent outputs (DESIGN 4.1, 4.8, 5.2): a name a requirements file
takes that no output declares builds, and fails the entries that reach
it in the report every other entry is judged in."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.core import irdiff
from moraine3d.errors import AssetError
from support.recipes import build_recipe, ir, refusal, workspace, write_files

SLAB = 'mo.transform(mo.box(id = "slab_box", size = [2, 0.1, 2]), id = "slab", translate = [0, 0.05, 0])'
POST = 'mo.transform(mo.box(id = "post_box", size = [0.1, 1, 0.1]), id = "post", translate = [0, 0.6, 0])'
ASSET = f'''lamp_on = mo.param("lamp", True, label = "Lamp")
mo.output("slab", {SLAB})
mo.output("post", {POST})
if lamp_on:
    lamp = mo.transform(mo.box(id = "lamp_box", size = [0.2, 0.2, 0.2]), id = "lamp", translate = [0.5, 0.2, 0])
    mo.output("lamp", lamp)
mo.assert_contact(id = "post_stands", outputs = ["post", "slab"])
'''
REQUIREMENTS = '''mo.assert_extent(id = "slab_width", outputs = ["slab"], measure = "size_x", value = 2.0)
mo.assert_contact(id = "lamp_stands", outputs = ["lamp", "slab"])
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return write_files(tmp_path / "yard", {
        "yard.star": ASSET, "yard.requirements.star": REQUIREMENTS,
    })


def _run(argv, capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_a_requirement_over_an_undeclared_output_builds(tmp_path):
    """DESIGN 4.8: in a requirements file a name nothing declares does
    not fail the build, which records it as absent. DESIGN 4.1: the IR
    lists the names under root `absent`, sorted."""
    result = build_recipe(tmp_path, f'mo.output("slab", {SLAB})\n', files={
        "asset.requirements.star": (
            'mo.assert_contact(id = "stands", outputs = ["post", "slab"])\n'
            'mo.assert_count(id = "lamps", outputs = ["lamp"], value = 2)\n'
        ),
    })
    doc = ir(result)
    assert doc["absent"] == ["lamp", "post"]
    assert list(doc) == [k for k in asset_mod.ROOT_KEY_ORDER if k in doc]
    [row] = result.report()["assets"].values()
    assert row["absent"] == ["lamp", "post"]


def test_the_entry_fails_beside_every_other(project, capsys):
    """DESIGN 5.2: an entry that names an absent output fails with the
    name, as one entry beside every other. DESIGN 4.1: the report names
    the output under `absent`."""
    code, green = _run(["validate", str(project)], capsys)
    assert code == 0 and "absent" not in green
    code, report = _run(["validate", str(project), "--param", "lamp=false", "--full"], capsys)
    assert code == 1
    verdicts = {a["assert"]: a["ok"] for a in report["asserts"]}
    assert verdicts == {
        "no_overlap": True, "post_stands": True, "slab_width": True,
        "lamp_stands": False,
    }
    [entry] = [a for a in report["asserts"] if not a["ok"]]
    assert entry["absent"] == ["lamp"] and "'lamp'" in entry["fix"]
    assert report["absent"] == ["lamp"]
    assert report["outputs_total"] == report["outputs_ok"] == 2


def test_a_human_owned_entry_is_not_steered_toward_its_removal(project, capsys):
    """DESIGN 5.2: editing or deleting a line of a requirements file is
    the change that needs human approval, so the fix of a human-owned
    entry asks, where an agent-owned one may be removed."""
    _, report = _run(["validate", str(project), "--param", "lamp=false"], capsys)
    [entry] = report["asserts"]
    assert entry["owner"] == "human"
    assert "approval" in entry["fix"] and "remove the entry" not in entry["fix"]


def test_a_group_is_absent_with_any_one_member(tmp_path, monkeypatch, capsys):
    """DESIGN 5.2: a group with an absent member is absent and never
    narrows to the members left, since its members are the entry's
    reach."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET,
        "yard.requirements.star": (
            'mo.group("fittings", ["post", "lamp"])\n'
            'mo.assert_extent(id = "fittings_low", outputs = ["fittings"], measure = "max_y", max = 1.2)\n'
        ),
    })
    code, report = _run(["validate", str(project), "--param", "lamp=false"], capsys)
    assert code == 1
    [entry] = report["asserts"]
    assert entry["assert"] == "fittings_low" and entry["absent"] == ["lamp"]
    assert entry["members"] == {"fittings": ["post", "lamp"]}
    assert "measured" not in entry


def test_an_entry_over_an_absent_output_cedes_nothing(tmp_path, monkeypatch, capsys):
    """DESIGN 5.2: an entry that names an absent output judges nothing,
    so the pairs a declared embed would take over stay with the
    guard."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sunk = 'mo.transform(mo.box(id = "peg_box", size = [0.2, 0.4, 0.2]), id = "peg", translate = [0, 0.1, 0])'
    project = write_files(tmp_path / "yard", {
        "yard.star": f'mo.output("slab", {SLAB})\nmo.output("peg", {sunk})\n',
        "yard.requirements.star": (
            'mo.group("pegs", ["peg", "spare_peg"])\n'
            'mo.assert_embedded(id = "pegs_set", outputs = ["pegs", "slab"])\n'
        ),
    })
    code, report = _run(["validate", str(project)], capsys)
    assert code == 1
    failed = {a["assert"] for a in report["asserts"]}
    assert failed == {"no_overlap", "pegs_set"}
    assert "ceded" not in report


@pytest.mark.parametrize("asset_line,requirement,code", [
    # the asset file's own entry
    ('mo.assert_contact(id = "stands", outputs = ["post", "slab"])', "", "unknown_output"),
    # a name the requirements file takes too is no more declared for it
    ('mo.assert_contact(id = "stands", outputs = ["post", "slab"])',
     'mo.assert_count(id = "posts", outputs = ["post"], value = 4)', "unknown_output"),
    # the asset file's own group
    ('mo.group("legs", ["slab", "post"])', "", "bad_group"),
    # a name that is no output's form can never be declared
    ("", 'mo.assert_count(id = "posts", outputs = ["corner post"], value = 4)', "unknown_output"),
    # a group holds outputs, never another group
    ("", 'mo.group("a", ["slab", "b"])\nmo.group("b", ["slab", "lid"])', "bad_group"),
])
def test_a_name_nothing_declares_fails_the_build_elsewhere(tmp_path, asset_line, requirement, code):
    """DESIGN 4.8: a name nothing declares fails the build in an asset
    file, whose author declares the outputs too; the requirements file
    is the one place it is taken."""
    files = {"asset.requirements.star": requirement + "\n"} if requirement else None
    err = refusal(tmp_path, f'mo.output("slab", {SLAB})\n{asset_line}\n', files=files)
    assert err.code == code


def test_a_group_the_asset_declares_through_a_function_is_the_assets(tmp_path):
    """DESIGN 5.2: the file is the one whose top level is running, so a
    function the asset calls declares the asset's group wherever the
    function was written, and its unknown member fails the build."""
    err = refusal(
        tmp_path,
        'load("asset.requirements.star", "pair")\n'
        f'mo.output("slab", {SLAB})\nmo.output("post", {POST})\n'
        'pair("legs", ["slab", "ghost"])\n',
        files={"asset.requirements.star": "def pair(name, members):\n    mo.group(name, members)\n"},
    )
    assert err.code == "bad_group" and "'ghost'" in err.message


def test_an_asset_with_no_output_yet_fails_the_build(tmp_path):
    """DESIGN 8.1: an asset with no output yet fails the build
    (`no_outputs`); its requirements fail as entries from its first
    output on."""
    requirements = {"asset.requirements.star": 'mo.assert_count(id = "posts", outputs = ["post"], value = 4)\n'}
    assert refusal(tmp_path / "a", "x = 1\n", files=requirements).code == "no_outputs"
    result = build_recipe(tmp_path / "b", f'mo.output("slab", {SLAB})\n', files=requirements)
    assert ir(result)["absent"] == ["post"]


def test_a_referenced_assets_requirement_fails_in_the_consumer(tmp_path, monkeypatch, capsys):
    """DESIGN 4.6: a sub-asset's asserts gate every consuming root per
    override identity, so a requirement of the kit over an output this
    use's overrides removed fails in the consumer's report."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = workspace(tmp_path / "ws", {
        "yard": {"yard.star": ASSET, "yard.requirements.star": REQUIREMENTS},
        "town": {"town.star": (
            'mo.output("slab", mo.asset_ref(project = "yard", output = "slab", '
            'id = "yard_slab", overrides = {"lamp": False}))\n'
        )},
    })
    code, report = _run(["validate", str(root / "town")], capsys)
    assert code == 1
    [entry] = report["asserts"]
    assert entry["assert"] == "lamp_stands" and entry["absent"] == ["lamp"]
    assert entry["asset"] == "yard/yard.star"
    [row] = report["ref_assets"]
    assert (row["asserts_total"], row["asserts_ok"]) == (4, 3)


def test_the_loader_judges_root_absent(tmp_path):
    """DESIGN 4.1: root `absent` is a sorted list of names no output
    declares; an assert or a group takes one as it takes an output."""
    result = build_recipe(tmp_path, f'mo.output("slab", {SLAB})\nmo.output("lid", {POST})\n', files={
        "asset.requirements.star": (
            'mo.group("pair", ["slab", "post"])\n'
            'mo.assert_count(id = "lamps", outputs = ["lamp"], value = 2)\n'
        ),
    })
    doc = ir(result)
    assert doc["absent"] == ["lamp", "post"]
    assert asset_mod.loads(json.dumps(doc)).absent == ["lamp", "post"]
    for bad in (["post", "lamp"], ["lamp", "lamp"], ["slab"], ["pair"], [], ["9a"], "lamp"):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(json.dumps({**doc, "absent": bad}))
        assert exc.value.code in ("bad_absent", "bad_group", "bad_assert_outputs"), bad


def test_the_loader_takes_an_absent_name_from_a_human_owned_entry_alone(tmp_path):
    """DESIGN 4.8: a name nothing declares fails the build in an asset
    file, and an IR read by path is held to the same: an agent-owned
    entry names declared outputs."""
    doc = ir(build_recipe(tmp_path, f'mo.output("slab", {SLAB})\n', files={
        "asset.requirements.star": 'mo.assert_count(id = "lamps", outputs = ["lamp"], value = 2)\n',
    }))
    [entry] = doc["asserts"]
    assert asset_mod.loads(json.dumps(doc)).absent == ["lamp"]
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps({**doc, "asserts": [{**entry, "owner": "agent"}]}))
    assert exc.value.code == "bad_assert_outputs" and "'lamp'" in exc.value.message


def test_the_build_diff_names_a_change_of_what_is_absent(tmp_path):
    """DESIGN 6: `build --diff` names the declarations that changed,
    and an output a requirement lost is one."""
    requirements = {"asset.requirements.star": 'mo.assert_count(id = "lamps", outputs = ["lamp"], value = 1)\n'}
    old = build_recipe(tmp_path / "a", f'mo.output("slab", {SLAB})\nmo.output("lamp", {POST})\n', files=requirements)
    new = build_recipe(tmp_path / "b", f'mo.output("slab", {SLAB})\n', files=requirements)
    diff = irdiff.asset_diff(old.target.asset, new.target.asset)
    assert "absent" in diff.declarations
