"""Optional outputs (DESIGN 4.1, 4.8, 5.2): a part the human made
optional is declared once in the requirements file, and a requirement
over it judges nothing while the part is absent and its knob is off."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.core import irdiff
from moraine3d.errors import AssetError
from support.recipes import build_recipe, ir, refusal, workspace, write_files

SLAB = 'mo.transform(mo.box(id = "slab_box", size = [2, 0.1, 2]), id = "slab", translate = [0, 0.05, 0])'
ASSET = f'''lamp_on = mo.param("lamp", True, label = "Lamp")
posts = mo.param("posts", 2, min = 0, max = 4, label = "Posts")
finish = mo.param("finish", "oak", label = "Finish")
mo.output("slab", {SLAB})
if lamp_on:
    lamp = mo.transform(mo.box(id = "lamp_box", size = [0.2, 0.2, 0.2]), id = "lamp", translate = [0.5, 0.2, 0])
    mo.output("lamp", lamp)
if posts:
    post = mo.transform(mo.box(id = "post_box", size = [0.1, 1, 0.1]), id = "post", translate = [-0.8, 0.6, -0.8])
    mo.output("posts", mo.array(post, id = "posts", count = posts, step_translate = [0.4, 0, 0]))
'''
REQUIREMENTS = '''mo.optional(["lamp"], "lamp")
mo.optional(["posts"], "posts")
mo.assert_contact(id = "lamp_stands", outputs = ["lamp", "slab"])
mo.assert_contact(id = "posts_stand", outputs = ["posts", "slab"], every = ["posts"])
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


def test_the_declaration_is_written_per_output(tmp_path):
    """DESIGN 4.1: root `optional` holds the outputs the human made
    optional, each with the knob that excuses it. DESIGN 5.2: the
    declaration names outputs, or groups, and a group stands for its
    members."""
    result = build_recipe(tmp_path, ASSET, files={"asset.requirements.star": (
        'mo.group("fittings", ["lamp", "posts"])\n'
        'mo.optional(["fittings"], "lamp")\n'
    )})
    doc = ir(result)
    assert doc["optional"] == {"lamp": "lamp", "posts": "lamp"}
    assert list(doc) == [k for k in asset_mod.ROOT_KEY_ORDER if k in doc]
    assert asset_mod.loads(json.dumps(doc)).optional == doc["optional"]


def test_an_entry_over_a_part_switched_off_judges_nothing_and_says_so(project, capsys):
    """DESIGN 5.2: an entry whose absent outputs are all optional under
    a knob that is off judges nothing; its `ok` is true and
    `unchecked: {"optional": [...]}` names what was not there to
    measure. The knob is a boolean or a number, off while it is false
    or 0. DESIGN 6: a green entry carrying `unchecked` is counted under
    that key at the top level."""
    code, on = _run(["validate", str(project)], capsys)
    assert code == 0 and "unchecked" not in on
    args = ["validate", str(project), "--param", "lamp=false", "--param", "posts=0"]
    code, report = _run([*args, "--full"], capsys)
    assert code == 0
    entries = {a["assert"]: a for a in report["asserts"]}
    assert entries["lamp_stands"]["ok"] is True
    assert entries["lamp_stands"]["unchecked"] == {"optional": ["lamp"]}
    assert entries["posts_stand"]["unchecked"] == {"optional": ["posts"]}
    assert "distance" not in entries["lamp_stands"]
    assert report["absent"] == ["lamp", "posts"]
    code, compact = _run(args, capsys)
    assert code == 0 and compact["unchecked"] == {"entries": 2, "optional": 2}


def test_an_optional_part_absent_while_its_knob_is_on_fails(tmp_path, monkeypatch, capsys):
    """DESIGN 5.2: an optional output absent while its knob is on fails
    its entries."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET.replace("if lamp_on:", "if False:"),
        "yard.requirements.star": REQUIREMENTS,
    })
    code, report = _run(["validate", str(project), "--full"], capsys)
    assert code == 1
    entries = {a["assert"]: a for a in report["asserts"]}
    assert entries["lamp_stands"]["ok"] is False
    assert entries["lamp_stands"]["absent"] == ["lamp"]
    assert "unchecked" not in entries["lamp_stands"]
    assert entries["posts_stand"]["ok"] is True and "distance" in entries["posts_stand"]


def test_an_optional_part_that_is_present_is_judged_whatever_the_knob_holds(
    tmp_path, monkeypatch, capsys,
):
    """DESIGN 5.2: an optional output that is present is judged
    whatever the knob holds."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET.replace("if lamp_on:", "if True:").replace(
            "translate = [0.5, 0.2, 0]", "translate = [0.5, 0.6, 0]"),
        "yard.requirements.star": REQUIREMENTS,
    })
    code, report = _run(["validate", str(project), "--param", "lamp=false"], capsys)
    assert code == 1
    assert [a["assert"] for a in report["asserts"]] == ["lamp_stands"]
    assert "distance" in report["asserts"][0]


def test_an_entry_over_an_absent_part_that_is_not_optional_still_fails(
    tmp_path, monkeypatch, capsys,
):
    """DESIGN 5.2: only an entry whose absent outputs are all optional
    under a knob that is off is left unjudged; one that also reaches an
    output the asset lost fails. DESIGN 8.2: the fix is the shortest
    path, so it names the output to build and not the excused one."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET,
        "yard.requirements.star": (
            'mo.optional(["lamp"], "lamp")\n'
            'mo.assert_clearance(id = "lamp_clear", outputs = ["lamp", "rail"], min = 0.1)\n'
        ),
    })
    code, report = _run(["validate", str(project), "--param", "lamp=false"], capsys)
    assert code == 1
    [entry] = report["asserts"]
    assert entry["absent"] == ["lamp", "rail"] and "unchecked" not in entry
    assert "'rail'" in entry["fix"] and "'lamp'" not in entry["fix"]


def test_an_unjudged_entry_cedes_nothing(tmp_path, monkeypatch, capsys):
    """DESIGN 5.2: an entry left unjudged cedes nothing, so the pairs a
    declared embed would take over stay with the guard."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sunk = 'mo.transform(mo.box(id = "peg_box", size = [0.2, 0.4, 0.2]), id = "peg", translate = [0, 0.25, 0])'
    project = write_files(tmp_path / "yard", {
        "yard.star": (
            'spare = mo.param("spare", False, label = "Spare peg")\n'
            f'mo.output("slab", {SLAB})\nmo.output("peg", {sunk})\n'
        ),
        "yard.requirements.star": (
            'mo.group("pegs", ["peg", "spare_peg"])\n'
            'mo.optional(["spare_peg"], "spare")\n'
            'mo.assert_embedded(id = "pegs_set", outputs = ["pegs", "slab"])\n'
        ),
    })
    code, report = _run(["validate", str(project)], capsys)
    assert code == 1
    assert [a["assert"] for a in report["asserts"]] == ["no_overlap"]
    assert report["unchecked"] == {"entries": 1, "optional": 1}


@pytest.mark.parametrize("requirement,asset_line,code,says", [
    # which parts may be absent is the human's to say
    ("", 'mo.optional(["lamp"], "lamp")', "optional_outside_requirements", "if"),
    ('mo.optional(["lamp"], "finish")', "", "bad_optional", "boolean"),
    ('mo.optional(["lamp"], "nope")', "", "param_unknown", "nope"),
    ('mo.optional(["lamp"], "lamp")\nmo.optional(["lamp"], "posts")', "", "bad_optional", "twice"),
    ('mo.optional(["corner post"], "lamp")', "", "unknown_output", "corner post"),
    ('mo.optional("lamp", "lamp")', "", "bad_declaration", "mo.optional(["),
    # the knob is named by its id, as an output is
    ('mo.optional(["lamp"], mo.param_value("lamp"))', "", "bad_declaration", "id of the knob"),
    # a condition the language evaluates tells the file nothing
    ('if mo.param_value("lamp"):\n    mo.assert_count(id = "n", outputs = ["slab"], value = 1)',
     "", "param_value_unused", "mo.optional("),
])
def test_the_declaration_is_the_requirements_files(tmp_path, requirement, asset_line, code, says):
    """DESIGN 5.2: the declaration is written in the requirements file,
    over a boolean or a number knob, and an asset file takes none.
    DESIGN 4.8: a requirements file cannot branch on a knob's value."""
    files = {"asset.requirements.star": requirement + "\n"} if requirement else None
    err = refusal(tmp_path, ASSET + asset_line + "\n", files=files)
    assert err.code == code and says in err.message


def test_a_default_that_switches_a_part_off_shows_in_the_build_diff(tmp_path):
    """DESIGN 5.2: the file owns which knob excuses the part, not the
    knob's value; a default that switches the part off moves no line of
    that file. DESIGN 6: `build --diff` names the outputs removed and
    the declarations that changed."""
    files = {"asset.requirements.star": REQUIREMENTS}
    old = build_recipe(tmp_path / "a", ASSET, files=files)
    new = build_recipe(
        tmp_path / "b", ASSET.replace('"lamp", True', '"lamp", False'), files=files,
    )
    diff = irdiff.asset_diff(old.target.asset, new.target.asset)
    assert diff.outputs_removed == ["lamp"]
    assert {"absent", "params"} <= set(diff.declarations)
    assert diff.asserts_changed == []


def test_a_referenced_assets_optional_part_is_excused_per_use(tmp_path, monkeypatch, capsys):
    """DESIGN 4.6: a sub-asset's asserts gate every consuming root per
    override identity, so a use that switches the part off leaves the
    kit's requirement over it unjudged there and counts it."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = workspace(tmp_path / "ws", {
        "yard": {"yard.star": ASSET, "yard.requirements.star": REQUIREMENTS},
        "town": {"town.star": (
            'mo.output("slab", mo.asset_ref(project = "yard", output = "slab", '
            'id = "yard_slab", overrides = {"lamp": False}))\n'
        )},
    })
    code, report = _run(["validate", str(root / "town")], capsys)
    assert code == 0
    assert report["unchecked"]["optional"] == 1
    [row] = report["ref_assets"]
    assert row["asserts_total"] == row["asserts_ok"] == 3


def test_the_loader_judges_root_optional(tmp_path):
    """DESIGN 4.1: root `optional` names outputs, declared or absent,
    each with a declared boolean or number knob."""
    doc = ir(build_recipe(tmp_path, ASSET, files={"asset.requirements.star": REQUIREMENTS}))
    for bad in ({}, {"rail": "lamp"}, {"lamp": "nope"}, {"lamp": "finish"}, {"lamp": 1}, ["lamp"]):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(json.dumps({**doc, "optional": bad}))
        assert exc.value.code == "bad_optional", bad
