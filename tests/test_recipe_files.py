"""Recipe files (DESIGN 4.2, 4.8, 5.2): what an asset loads, what a
`lib/` file may hold, the requirements file and the owner it gives,
and the knob's resolution."""

import pytest

from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.errors import BuildError
from moraine3d.recipe.build import build
from support.links import link_dir as _link_dir
from support.recipes import build_recipe, ir, refusal, write_files

BOX = 'mo.box(id = "a", size = [1, 1, 1])'
LEG = 'def leg(id):\n    return mo.box(id = id, size = [0.1, 1, 0.1])\n'


# -- load() --------------------------------------------------------------------


def test_a_asset_loads_its_projects_lib_files(tmp_path):
    """DESIGN 4.8: helpers shared by the root and its sub-assets live in
    `lib/*.star`, and an asset loads `lib/` files of its own project."""
    result = build_recipe(
        tmp_path, 'load("lib/parts.star", "leg", "H")\nmo.output("a", leg("a"))\nx = H\n',
        files={"lib/parts.star": LEG + "H = 2\n"},
    )
    assert ir(result)["nodes"][0]["site"] == "lib/parts.star:2"
    assert {p.name for p in result.sources} == {"asset.star", "parts.star"}


def test_a_lib_file_may_load_another(tmp_path):
    """DESIGN 4.8: a `lib/` file is definition-only, and definitions
    build on definitions."""
    build_recipe(
        tmp_path, 'load("lib/b.star", "pair")\nmo.output("a", pair("a"))\n',
        files={
            "lib/a.star": LEG,
            "lib/b.star": 'load("lib/a.star", "leg")\ndef pair(id):\n    return leg(id)\n',
        },
    )


@pytest.mark.parametrize("top", [
    BOX,
    'mo.assert_contact(id = "x", outputs = ["a", "b"])',
    'mo.param("w", 1.0, min = 0.0, max = 2.0)',
    'mo.output("a", None)',
])
def test_a_lib_file_that_declares_at_its_top_level_fails(tmp_path, top):
    """DESIGN 4.8: a `lib/` file that creates a node, declares an
    output, param, or assert at its top level fails the build."""
    err = refusal(
        tmp_path, 'load("lib/parts.star", "K")\n',
        files={"lib/parts.star": f"K = 1\n{top}\n"},
    )
    assert err.code == "lib_not_definitions"
    assert err.detail["site"] == "lib/parts.star:2"


@pytest.mark.parametrize("path,why", [
    ("subassets/part.star", "referenced, never loaded"),
    ("other.star", "referenced, never loaded"),
    ("lib/deep/parts.star", "referenced, never loaded"),
    ("../sibling/lib/parts.star", "project-relative"),
    ("/abs/parts.star", "project-relative"),
    ("lib\\\\parts.star", "project-relative"),
    ("lib/parts.bzl", "ends in .star"),
    ("other.requirements.star", "its own requirements file"),
    ("@moraine", "in scope already"),
])
def test_a_asset_loads_nothing_else(tmp_path, path, why):
    """DESIGN 4.8: an asset loads `lib/` files of its own project and
    its own requirements file, and nothing else; an asset file is never
    loaded, only referenced."""
    err = refusal(tmp_path, f'load("{path}", "x")\n')
    assert err.code == "load_refused"
    assert why in err.message
    assert err.detail["site"] == "asset.star"


@pytest.mark.parametrize("held", ["../shared", "subassets"])
def test_a_load_reads_no_file_through_a_link(tmp_path, held):
    """DESIGN 4.8: a recipe file reached through a link is refused,
    since the rules judge the path a recipe spells; a `lib/` that is a
    link holds another project's files, or this project's assets."""
    write_files(tmp_path / "shared", {"parts.star": LEG})
    project = write_files(tmp_path / "yard", {
        "yard.star": 'load("lib/parts.star", "leg")\nmo.output("a", leg("a"))\n',
        "subassets/parts.star": LEG + 'mo.output("a", leg("a"))\n',
    })
    try:
        _link_dir((project / held).resolve(), project / "lib")
    except OSError as e:
        pytest.skip(f"cannot create a directory link here: {e}")
    with pytest.raises(BuildError) as e:
        build(project / "yard.star", write=False)
    assert e.value.code == "load_refused"
    assert "link" in e.value.message
    assert e.value.detail["site"] == "yard.star"


def test_a_load_may_spell_a_name_in_another_case(tmp_path):
    """DESIGN 4.2: extensions and names compare as the filesystem
    folds them, so a load that spells a file's name in another case
    reads that file where the filesystem does."""
    project = write_files(tmp_path / "yard", {
        "yard.star": 'load("lib/Parts.star", "leg")\nmo.output("a", leg("a"))\n',
        "lib/parts.star": LEG,
    })
    if not (project / "lib" / "Parts.star").is_file():
        pytest.skip("this filesystem tells the two names apart")
    assert build(project / "yard.star", write=False).target.counts["nodes"] == 1


def test_a_requirements_file_is_no_link(tmp_path):
    """DESIGN 4.8: a recipe file reached through a link is refused;
    the requirements file the build reads beside an asset is one."""
    outside = write_files(tmp_path / "shared", {"yard.requirements.star": "K = 1\n"})
    project = write_files(tmp_path / "yard", {"yard.star": f'mo.output("a", {BOX})\n'})
    try:
        (project / "yard.requirements.star").symlink_to(outside / "yard.requirements.star")
    except OSError as e:
        pytest.skip(f"cannot create a file link here: {e}")
    with pytest.raises(BuildError) as e:
        build(project / "yard.star", write=False)
    assert e.value.code == "recipe_is_link"
    assert e.value.detail["file"] == "yard.requirements.star"


def test_a_lib_file_loads_no_requirements_file(tmp_path):
    """DESIGN 4.8: a `lib/` file loads no requirements file; a helper
    that needs a requirement constant takes it as an argument."""
    err = refusal(
        tmp_path, 'load("lib/parts.star", "K")\n',
        files={
            "lib/parts.star": 'load("asset.requirements.star", "W")\nK = W\n',
            "asset.requirements.star": "W = 2\n",
        },
    )
    assert err.code == "load_refused"
    assert "takes it as an argument" in err.message
    assert err.detail["site"] == "lib/parts.star"


def test_a_load_cycle_names_the_chain(tmp_path):
    """DESIGN 4.8: every loop iterates a finite sequence, so evaluation
    always terminates; a file that loads itself through another does
    not."""
    err = refusal(
        tmp_path, 'load("lib/a.star", "x")\n',
        files={
            "lib/a.star": 'load("lib/b.star", "y")\nx = 1\n',
            "lib/b.star": 'load("lib/a.star", "x")\ny = 1\n',
        },
    )
    assert err.code == "load_refused"
    assert err.detail["chain"] == ["lib/a.star", "lib/b.star", "lib/a.star"]


def test_a_missing_file_is_named_with_the_line_that_loads_it(tmp_path):
    """DESIGN 8.2: errors state what failed and the shortest path to a
    fix."""
    err = refusal(tmp_path, 'load("lib/nope.star", "x")\n')
    assert err.code == "recipe_not_found"
    assert err.detail == {"file": "lib/nope.star", "site": "asset.star"}


def test_a_loaded_name_is_not_passed_on(tmp_path):
    """DESIGN 4.8: an asset loads what it names; `mo` reaches a file as
    a builtin and never through another file."""
    err = refusal(
        tmp_path, 'load("lib/parts.star", "mo")\n',
        files={"lib/parts.star": "K = 1\n"},
    )
    assert err.code == "recipe_error"
    assert "not exported" in err.message


# -- the build's targets -----------------------------------------------------


@pytest.mark.parametrize("name", ["asset.requirements.star", "lib/parts.star", "notes.txt"])
def test_only_a_asset_file_is_a_target(tmp_path, name):
    """DESIGN 4.8: a file named `*.requirements.star` is a requirements
    file and never an asset, so it is never a command's target."""
    write_files(tmp_path, {name: "K = 1\n"})
    with pytest.raises(BuildError) as e:
        build(tmp_path / name)
    assert e.value.code == "not_an_asset"


def test_a_asset_file_is_not_named_like_a_variant(tmp_path):
    """DESIGN 4.6: a variant IR is named by its asset and its override
    identity, so an asset file's own name never ends like one."""
    err = refusal(tmp_path, f'mo.output("a", {BOX})\n', name="panel." + "0" * 24)
    assert err.code == "bad_asset_name"


# -- requirements files ------------------------------------------------------


REQUIREMENTS = '''WIDTH = 2.0
mo.assert_extent(id = "slab_width", outputs = ["slab"], measure = "size_x", value = WIDTH)
mo.budget(materials = 4)
'''
ASSET = '''mo.output("slab", mo.box(id = "slab", size = [2.0, 0.1, 2.0]))
mo.output("lid", mo.box(id = "lid", size = [1, 1, 1]))
mo.assert_contact(id = "lid_on_slab", outputs = ["lid", "slab"])
'''


def test_the_file_decides_the_owner(tmp_path):
    """DESIGN 5.2: an entry declared in a requirements file is
    human-owned, an entry declared in an asset file is agent-owned."""
    result = build_recipe(tmp_path, ASSET, files={"asset.requirements.star": REQUIREMENTS})
    doc = ir(result)
    assert {a["id"]: a["owner"] for a in doc["asserts"]} == {
        "lid_on_slab": "agent", "slab_width": "human",
    }
    assert doc["asserts"][1]["site"] == "asset.requirements.star:2"
    assert doc["budgets"] == {"materials": 4}


def test_the_requirements_file_is_built_whether_or_not_it_is_loaded(tmp_path):
    """DESIGN 4.8: an asset's requirements file runs exactly once per
    build whether a `load()` reached it first or the build runs it
    after the recipe."""
    loaded = 'load("asset.requirements.star", "WIDTH")\n' + ASSET.replace("2.0, 0.1", "WIDTH, 0.1")
    for text, first in ((ASSET, "lid_on_slab"), (loaded, "slab_width")):
        base = tmp_path / first
        result = build_recipe(base, text, files={"asset.requirements.star": REQUIREMENTS})
        ids = [a["id"] for a in ir(result)["asserts"]]
        assert sorted(ids) == ["lid_on_slab", "slab_width"] and ids[0] == first
        assert {p.name for p in result.sources} == {"asset.star", "asset.requirements.star"}


def test_a_asset_builds_against_its_requirement_constants(tmp_path):
    """DESIGN 5.2: an asset may `load()` the requirement constants and
    build against them, so a requirement changed by the human moves the
    geometry and the check together."""
    result = build_recipe(
        tmp_path,
        'load("asset.requirements.star", "WIDTH")\n'
        'mo.output("slab", mo.box(id = "slab", size = [WIDTH, 0.1, WIDTH]))\n',
        files={"asset.requirements.star": REQUIREMENTS.replace("2.0", "3.5")},
    )
    doc = ir(result)
    assert doc["nodes"][0]["params"]["size"] == [3.5, 0.1, 3.5]
    assert doc["asserts"][0]["value"] == 3.5


def test_a_requirement_is_the_humans_enumeration(tmp_path):
    """DESIGN 5.2: a list of six roof widths written in the
    requirements file and iterated into six asserts is the human's
    enumeration, not a measurement."""
    widths = [1.0, 3.5, 3.5, 3.5, 3.5, 1.0]
    asset = "".join(
        f'mo.output("bay{i}", mo.box(id = "bay{i}", size = [{w}, 0.1, 1]))\n'
        for i, w in enumerate(widths)
    )
    requirements = f'''BAYS = {widths}
for i, w in enumerate(BAYS):
    mo.assert_extent(id = "bay%d_width" % i, outputs = ["bay%d" % i], measure = "size_x", value = w)
'''
    result = build_recipe(tmp_path, asset, files={"asset.requirements.star": requirements})
    asserts = ir(result)["asserts"]
    assert [a["value"] for a in asserts] == widths
    assert {a["owner"] for a in asserts} == {"human"}
    assert {a["site"] for a in asserts} == {"asset.requirements.star:3"}


@pytest.mark.parametrize("line,code", [
    (BOX, "requirements_builtin"),
    ('mo.output("slab", None)', "requirements_builtin"),
    ('mo.param("w", 1.0, min = 0.0, max = 2.0)', "requirements_builtin"),
    ('mo.waive("slab", "not_sunk")', "requirements_builtin"),
    ('load("lib/parts.star", "K")', "requirements_load"),
])
def test_a_requirements_file_builds_nothing_and_loads_nothing(tmp_path, line, code):
    """DESIGN 5.2: a requirements file loads nothing and no builtin in
    it returns a recipe value. DESIGN 4.8: no operator, no declaration
    that creates a node, and no `load()`."""
    err = refusal(
        tmp_path, ASSET,
        files={"asset.requirements.star": f"K = 1\n{line}\n", "lib/parts.star": "K = 1\n"},
    )
    assert err.code == code
    assert err.detail["site"] == "asset.requirements.star:2"


def test_a_requirement_is_named(tmp_path):
    """DESIGN 4.8: an assert declared in a requirements file carries an
    `id`, so adding a requirement never renames another."""
    err = refusal(tmp_path, ASSET, files={
        "asset.requirements.star": 'mo.assert_count(outputs = ["slab"], value = 1)\n',
    })
    assert err.code == "id_required"
    assert err.detail["site"] == "asset.requirements.star:1"


_PIECES = (
    'mo.output("board", mo.box(id = "board", size = [1, 0.05, 0.3]))\n'
    'mo.output("posts", mo.transform(mo.box(id = "post", size = [0.05, 0.8, 0.3]), id = "posts", translate = [0, -0.4, 0]))\n'
    'mo.output("wall", mo.transform(mo.box(id = "w", size = [0.1, 1, 1]), id = "wall", translate = [2, 0, 0]))\n'
)


def test_a_group_a_requirement_names_lives_in_the_requirements_file(tmp_path):
    """DESIGN 5.2: a group a human-owned entry names is declared in the
    requirements file, since its members are the entry's reach; the IR
    holds it beside the asset's own groups, unchanged in shape."""
    result = build_recipe(tmp_path, _PIECES, files={"asset.requirements.star": (
        'mo.group("shelf", ["board", "posts"])\n'
        'mo.assert_clearance(id = "shelf_clear", outputs = ["shelf", "wall"], min = 0.5)\n'
    )})
    doc = ir(result)
    assert doc["groups"] == {"shelf": ["board", "posts"]}
    assert doc["asserts"][0]["owner"] == "human"
    assert result.warnings == []


def test_a_group_the_asset_declares_and_a_requirement_names_is_warned(tmp_path):
    """DESIGN 5.2: a group declared in the asset file and named by a
    human-owned entry is the agent's to narrow with no diff in the
    human's file, so the build warns (`group_not_owned`) with the move
    as the fix; the agent's own entry over the same group draws none."""
    asset = _PIECES + (
        'mo.group("shelf", ["board", "posts"])\n'
        'mo.assert_clearance(id = "mine", outputs = ["shelf", "wall"], min = 0.1)\n'
    )
    quiet = build_recipe(tmp_path / "a", asset)
    assert quiet.warnings == []
    loud = build_recipe(tmp_path / "b", asset, files={"asset.requirements.star": (
        'mo.assert_clearance(id = "theirs", outputs = ["shelf", "wall"], min = 0.5)\n'
        'mo.assert_extent(id = "wide", outputs = ["shelf"], measure = "size_x", min = 0.9)\n'
    )})
    [row] = loud.warnings
    assert row["warning"] == "group_not_owned" and row["group"] == "shelf"
    assert row["site"] == "asset.star:4" and row["asserts"] == ["theirs", "wide"]
    assert "asset.requirements.star" in row["fix"]
    assert ir(loud)["groups"] == {"shelf": ["board", "posts"]}


def test_a_group_is_declared_once_across_the_two_files(tmp_path):
    """DESIGN 4.1: groups and outputs share one namespace, so a name
    declared in the asset and again in the requirements file is one
    group declared twice, refused with the first site."""
    err = refusal(tmp_path, _PIECES + 'mo.group("shelf", ["board", "posts"])\n',
                  files={"asset.requirements.star": 'mo.group("shelf", ["board", "wall"])\n'})
    assert err.code == "bad_group" and "asset.star:4" in err.message
    assert err.detail["site"] == "asset.requirements.star:1"


def test_a_budget_lives_in_the_requirements_file(tmp_path):
    """DESIGN 5.2: budgets live in the requirements file, since they
    are human-owned acceptance criteria by definition."""
    err = refusal(tmp_path, ASSET + "mo.budget(materials = 4)\n")
    assert err.code == "budget_outside_requirements"
    assert "'asset.requirements.star'" in err.message
    assert err.detail["site"] == "asset.star:4"


@pytest.mark.parametrize("call,code", [
    ("mo.budget(triangles = 100)", "unknown_budget"),
    ("mo.budget(materials = 0)", "bad_budget_value"),
    ('mo.budget("materials", 4)', "bad_declaration"),
    ("mo.budget(materials = 4)\nmo.budget(materials = 5)", "bad_declaration"),
])
def test_a_budget_is_judged_as_its_ir_entry_is(tmp_path, call, code):
    """DESIGN 4.1: `triangles` and `parts` are refused as unknown
    today, never silently unchecked."""
    err = refusal(tmp_path, ASSET, files={"asset.requirements.star": call + "\n"})
    assert err.code == code


def test_a_function_called_by_the_asset_declares_for_the_asset(tmp_path):
    """DESIGN 5.2: ownership is the file's whose declarations run; a
    function the asset calls with its own values declares the agent's
    entry, wherever the function was written."""
    result = build_recipe(
        tmp_path,
        'load("asset.requirements.star", "sized")\n'
        'mo.output("slab", mo.box(id = "slab", size = [2, 0.1, 2]))\n'
        'sized("slab", 2.0)\n',
        files={"asset.requirements.star": (
            'def sized(name, w):\n'
            '    mo.assert_extent(id = name + "_w", outputs = [name], measure = "size_x", value = w)\n'
        )},
    )
    [entry] = ir(result)["asserts"]
    assert (entry["owner"], entry["site"]) == ("agent", "asset.requirements.star:2")


# -- knobs -------------------------------------------------------------------


KNOB = '''h = mo.param("height", 2.0, min = 1.0, max = 3.0, label = "Height", step = 0.5)
mo.output("post", mo.box(id = "post", size = [0.1, h, 0.1]))
mo.output("cap", mo.box(id = "cap", size = [0.2, h / 4, 0.2]))
'''


def test_a_knob_returns_its_value_and_the_recipe_derives_from_it(tmp_path):
    """DESIGN 4.8: `mo.param` returns the knob's resolved value, and a
    value stated once is derived everywhere it is used (DESIGN 4)."""
    doc = ir(build_recipe(tmp_path, KNOB))
    assert doc["params"] == [{
        "id": "height", "value": 2.0, "label": "Height", "min": 1.0,
        "max": 3.0, "step": 0.5, "site": "asset.star:1",
    }]
    assert [n["params"]["size"][1] for n in doc["nodes"]] == [2.0, 0.5]


def test_an_offered_value_takes_precedence_over_the_default(tmp_path):
    """DESIGN 4.8: values resolve in this order, the recipe's default,
    then a consumer's override; later values take precedence."""
    result = build_recipe(tmp_path, KNOB, offered={"height": 3})
    doc = ir(result)
    assert doc["params"][0]["value"] == 3.0
    assert [n["params"]["size"][1] for n in doc["nodes"]] == [3.0, 0.75]
    assert result.written == [] and not (tmp_path / "build").exists()


@pytest.mark.parametrize("offered,code", [
    ({"height": 3.5}, "param_out_of_range"),
    ({"height": "tall"}, "param_type_mismatch"),
    ({"height": [1.0, 2.0]}, "param_type_mismatch"),
    ({"heigth": 2.0}, "param_unknown"),
])
def test_a_value_a_consumer_offers_is_refused_by_the_declaration(tmp_path, offered, code):
    """DESIGN 4.6: an override outside the range fails with
    `param_out_of_range`, enforced in the build's one param resolution;
    the refusal covers a value a consumer offers."""
    err = refusal(tmp_path, KNOB, offered=offered)
    assert err.code == code


def test_the_assets_own_value_outside_its_range_builds_and_fails_its_entry(tmp_path):
    """DESIGN 4.6: an asset's own value outside its range, a default
    that a later narrowing of the range left behind, is a failing entry
    of the report (DESIGN 6), never a failed build."""
    result = build_recipe(tmp_path, KNOB.replace("min = 1.0", "min = 2.5"))
    assert ir(result)["params"][0]["value"] == 2.0
    [entry] = params_mod.range_entries(result.target.asset)
    assert entry["ok"] is False
    assert "asset.star:1" in entry["fix"]


@pytest.mark.parametrize("call,code", [
    ('mo.param("height", 2.0)', "bad_param_decl"),
    ('mo.param("height", 2.0, min = 3.0, max = 1.0)', "bad_param_decl"),
    ('mo.param("height", 2.0, min = 1.0, max = 3.0, step = 0)', "bad_param_decl"),
    ('mo.param("9h", 2.0, min = 1.0, max = 3.0)', "bad_param_id"),
    ('mo.param("a/b", 2.0, min = 1.0, max = 3.0)', "bad_param_id"),
    ('mo.param("time", 2.0, min = 1.0, max = 3.0)', "reserved_param_id"),
    ('mo.param("finish", "steel", min = 0, max = 1)', "bad_param_decl"),
    ('mo.param("v", [1, 2, 3, 4, 5], min = 0, max = 9)', "bad_param_value"),
    ('mo.param("height", 2.0, min = 1.0, max = 3.0)\n'
     'mo.param("height", 2.0, min = 1.0, max = 3.0)', "duplicate_param_id"),
    ('mo.param("height")', "bad_declaration"),
    ('mo.param("height", 2.0, low = 1.0)', "bad_declaration"),
])
def test_a_declaration_the_panel_cannot_show_is_refused(tmp_path, call, code):
    """DESIGN 7.2: kinds are derived from the declaration's default,
    and number and vector declarations must carry min/max. DESIGN 4.1:
    `time` is refused as a public parameter id."""
    err = refusal(tmp_path, f'{call}\nmo.output("a", {BOX})\n')
    assert err.code == code
    assert err.detail["site"].startswith("asset.star:")


def test_the_kind_is_the_defaults(tmp_path):
    """DESIGN 7.2: kinds are `number | vector | boolean | text`, derived
    from the declaration's default, never from where the value lands."""
    doc = ir(build_recipe(tmp_path, f'''mo.param("n", 3, min = 1, max = 9)
mo.param("v", [1.0, 2.0], min = 0.0, max = 4.0)
mo.param("on", True)
mo.param("finish", "steel")
mo.output("a", {BOX})
'''))
    assert [asset_mod.param_kind(p["value"]) for p in doc["params"]] == [
        "number", "vector", "boolean", "text",
    ]
