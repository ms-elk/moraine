"""`moraine build --diff` (DESIGN 6): the graph change an edit made, so
an edit inside a loop or a function shows which nodes it reached."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import irdiff
from moraine3d.core import asset as asset_mod
from support.recipes import workspace, write_files

pytestmark = pytest.mark.real_stamp

LIB = '''def post(id, x, height):
    shaft = mo.box(id = id + "/shaft", size = [0.1, height, 0.1])
    return mo.transform(shaft, id = id, translate = [x, height / 2, 0])
'''

YARD = '''load("lib/parts.star", "post")
HEIGHT = 2.0
slab = mo.transform(
    mo.box(id = "slab_box", size = [8, 0.1, 2]),
    id = "slab", translate = [0, -0.05, 0],
)
mo.output("slab", slab)
for i in range(3):
    mo.output("post_%d" % i, post("post_%d" % i, 2 * i - 2, HEIGHT))
    mo.assert_contact(id = "post_%d_stands" % i, outputs = ["post_%d" % i, "slab"])
mo.waive("slab", "not_sunk")
'''

REQUIREMENTS = '''mo.assert_extent(id = "post_height", outputs = ["post_0"], measure = "size_y", value = 2.0)
mo.budget(materials = 2)
'''


def _project(base):
    return write_files(base, {
        "yard.star": YARD, "lib/parts.star": LIB,
        "yard.requirements.star": REQUIREMENTS,
    })


def _diff(argv, capsys, code=0) -> dict:
    capsys.readouterr()
    assert main(["build", *argv]) == code
    return json.loads(capsys.readouterr().out)


def test_an_edit_inside_a_function_shows_the_nodes_it_reached(tmp_path, capsys):
    """DESIGN 6: `--diff` reports the graph change the last edit made,
    so an edit inside a loop or a function shows which nodes it
    reached: nodes changed with the changed params, and the outputs
    made dirty."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    write_files(project, {"yard.star": YARD.replace("HEIGHT = 2.0", "HEIGHT = 2.5")})
    out = _diff([str(project), "--diff"], capsys)
    assert out["diff"]["base"] == "disk"
    [(name, asset)] = out["diff"]["assets"].items()
    assert (name, asset["base"]) == ("build/yard.mora", "build/yard.mora")
    assert asset["nodes"]["added"] == {"count": 0}
    assert asset["nodes"]["removed"] == {"count": 0}
    assert asset["nodes"]["changed"]["count"] == 6
    shaft = asset["nodes"]["changed"]["rows"][0]
    assert shaft == {
        "id": "post_0/shaft", "site": "lib/parts.star:2", "via": ["yard.star:9"],
        "params": {"size": [[0.1, 2.0, 0.1], [0.1, 2.5, 0.1]]},
    }
    assert asset["outputs"]["dirty"] == {
        "count": 3, "ids": ["post_0", "post_1", "post_2"],
    }
    assert asset["asserts"]["changed"] == {"count": 0}
    assert out["written"] == ["build/yard.mora"]


def test_the_base_is_the_ir_on_disk_before_the_build(tmp_path, capsys):
    """DESIGN 6: the comparison base is the IR on disk before the
    build. A build that changes nothing reports no asset, and a first
    build, with nothing on disk, reports every entry as added."""
    project = _project(tmp_path)
    first = _diff([str(project), "--diff"], capsys)
    asset = first["diff"]["assets"]["build/yard.mora"]
    assert asset["base"] is None
    assert asset["nodes"]["added"]["count"] == 8
    assert asset["outputs"]["added"]["count"] == 4
    again = _diff([str(project), "--diff"], capsys)
    assert again["diff"] == {"base": "disk", "assets": {}, "unchanged": 1}
    assert again["unchanged"] == 1


def test_a_long_change_is_counted_by_the_line_that_made_it(tmp_path, capsys):
    """DESIGN 8.2: a report's cost follows the answer, not the asset.
    Past ten entries the diff lists ten and counts the rest by the
    recipe line that made them, so one edited expression in a loop
    reads as one row."""
    project = _project(tmp_path)
    many = YARD.replace("range(3)", "range(12)")
    write_files(project, {"yard.star": many})
    _diff([str(project)], capsys)
    write_files(project, {"yard.star": many.replace("HEIGHT = 2.0", "HEIGHT = 2.5")})
    asset = _diff([str(project), "--diff"], capsys)["diff"]["assets"]["build/yard.mora"]
    assert asset["nodes"]["changed"]["count"] == 24
    assert len(asset["nodes"]["changed"]["rows"]) == irdiff.LISTED
    assert asset["nodes"]["by_site"] == [
        {"site": "lib/parts.star:2", "count": 12},
        {"site": "lib/parts.star:3", "count": 12},
    ]
    assert asset["outputs"]["dirty"]["count"] == 12
    assert len(asset["outputs"]["dirty"]["ids"]) == irdiff.LISTED


def test_nodes_and_asserts_added_and_removed_are_named(tmp_path, capsys):
    """DESIGN 6: nodes added, removed, and changed, asserts added,
    removed, and changed."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    write_files(project, {"yard.star": YARD.replace("range(3)", "range(2)").replace(
        'outputs = ["post_%d" % i, "slab"])',
        'outputs = ["post_%d" % i, "slab"], tol = 0.001)',
    ) + 'mo.output("cap", mo.box(id = "cap", size = [1, 1, 1]))\n'})
    asset = _diff([str(project), "--diff"], capsys)["diff"]["assets"]["build/yard.mora"]
    assert asset["nodes"]["added"] == {"count": 1, "ids": ["cap"]}
    assert asset["nodes"]["removed"] == {"count": 2, "ids": ["post_2/shaft", "post_2"]}
    assert asset["outputs"]["added"] == {"count": 1, "ids": ["cap"]}
    assert asset["outputs"]["removed"] == {"count": 1, "ids": ["post_2"]}
    assert asset["outputs"]["dirty"] == {"count": 0}
    assert asset["asserts"]["removed"] == {"count": 1, "ids": ["post_2_stands"]}
    assert asset["asserts"]["changed"]["count"] == 2
    assert asset["asserts"]["changed"]["rows"][0] == {
        "id": "post_0_stands", "site": "yard.star:10", "keys": {"tol": [None, 0.001]},
    }


def test_a_human_owned_entry_is_listed_under_its_own_key(tmp_path, capsys):
    """DESIGN 6: asserts added, removed, and changed with human-owned
    entries listed under their own key. DESIGN 5.2: a requirements file
    changes only by a diff of that file, which `build --diff` lists
    under its own key."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    write_files(project, {"yard.requirements.star": REQUIREMENTS.replace(
        "value = 2.0", "value = 2.2").replace("materials = 2", "materials = 3")})
    asset = _diff([str(project), "--diff"], capsys)["diff"]["assets"]["build/yard.mora"]
    assert asset["asserts"]["changed"] == {"count": 0}
    assert asset["asserts"]["human"]["changed"] == {"count": 1, "rows": [{
        "id": "post_height", "site": "yard.requirements.star:1",
        "keys": {"value": [2.0, 2.2]},
    }]}
    assert asset["declarations"] == ["budgets"]
    assert asset["nodes"]["changed"] == {"count": 0}


def test_where_an_entry_was_written_is_no_change(tmp_path, capsys):
    """DESIGN 4.1: the site is location, not content. A recipe moved
    down by a comment line builds the same graph."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    write_files(project, {"yard.star": "# the yard\n" + YARD})
    out = _diff([str(project), "--diff"], capsys)
    assert out["diff"]["assets"] == {}
    assert out["written"] == ["build/yard.mora"]


def test_a_changed_sub_asset_makes_the_outputs_that_read_it_dirty(tmp_path, capsys):
    """DESIGN 6: the outputs made dirty. A reference's own line did not
    change when the asset it reads did, and the output downstream of it
    is dirty all the same."""
    part = 'mo.output("block", mo.box(id = "b", size = [1, %s, 1]))\nmo.waive_asset("grounded")\n'
    project = write_files(tmp_path, {
        "yard.star": 'p = mo.asset_ref("subassets/part.star", "block", id = "p")\n'
                     'up = mo.transform(p, id = "up", translate = [0, 1, 0])\n'
                     'mo.output("piece", up)\n'
                     'mo.output("other", mo.box(id = "o", size = [1, 1, 1]))\n'
                     'mo.waive_asset("grounded")\n',
        "subassets/part.star": part % "1",
    })
    _diff([str(project)], capsys)
    write_files(project, {"subassets/part.star": part % "2"})
    assets = _diff([str(project), "--diff"], capsys)["diff"]["assets"]
    assert assets["build/subassets/part.mora"]["nodes"]["changed"]["count"] == 1
    root = assets["build/yard.mora"]
    assert root["nodes"]["changed"] == {"count": 0}
    assert root["outputs"]["dirty"] == {"count": 1, "ids": ["piece"]}


def test_a_variant_is_compared_with_the_one_its_reference_read_before(tmp_path, capsys):
    """DESIGN 6: an edit shows which nodes it reached. A changed
    override opens a variant under another name (DESIGN 4.6), and the
    diff pairs it with the variant the same reference read before, so
    one moved value reads as one changed node."""
    part = ('w = mo.param("w", 1.0, min = 0.5, max = 4.0)\n'
            'mo.output("block", mo.box(id = "b", size = [w, 1, 1]))\n'
            'mo.waive_asset("grounded")\n')
    root = ('p = mo.asset_ref("subassets/part.star", "block", id = "p", overrides = {"w": %s})\n'
            'mo.output("piece", p)\nmo.waive_asset("grounded")\n')
    project = write_files(tmp_path, {"yard.star": root % "2.0", "subassets/part.star": part})
    _diff([str(project)], capsys)
    write_files(project, {"yard.star": root % "3.0"})
    assets = _diff([str(project), "--diff"], capsys)["diff"]["assets"]
    [variant] = [v for k, v in assets.items() if "subassets" in k]
    assert variant["base"] is not None and variant["base"] != variant
    assert variant["nodes"]["added"] == {"count": 0}
    [row] = variant["nodes"]["changed"]["rows"]
    assert row["params"] == {"size": [[2.0, 1.0, 1.0], [3.0, 1.0, 1.0]]}
    assert variant["declarations"] == ["params"]
    assert assets["build/yard.mora"]["outputs"]["dirty"]["ids"] == ["piece"]


def test_a_path_names_the_revision_to_compare_against(tmp_path, capsys):
    """DESIGN 6: the base is the IR file or `build/` directory
    `--diff PATH` names, so a review compares two revisions by building
    each, and the tool never reads version control."""
    before = _project(tmp_path / "before" / "yard")
    after = _project(tmp_path / "after" / "yard")
    write_files(after, {"yard.star": YARD.replace("HEIGHT = 2.0", "HEIGHT = 2.5")})
    _diff([str(before)], capsys)
    for base in (before / "build", before / "build" / "yard.mora"):
        out = _diff([str(after), "--diff", str(base)], capsys)
        assert out["diff"]["base"] == str(base)
        [asset] = out["diff"]["assets"].values()
        assert asset["nodes"]["changed"]["count"] == 6
    err = _diff([str(after), "--diff", str(tmp_path / "nowhere")], capsys, code=1)
    assert err["error"]["code"] == "diff_base_not_found"


def test_diff_with_check_writes_nothing(tmp_path, capsys):
    """DESIGN 6: `--check` is never a write, with `--diff` beside it as
    without."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    before = (project / "build" / "yard.mora").read_bytes()
    write_files(project, {"yard.star": YARD.replace("HEIGHT = 2.0", "HEIGHT = 2.5")})
    out = _diff([str(project), "--diff", "--check"], capsys, code=1)
    assert out["check"]["differs"][0]["file"] == "build/yard.mora"
    assert out["diff"]["assets"]["build/yard.mora"]["nodes"]["changed"]["count"] == 6
    assert "written" not in out
    assert (project / "build" / "yard.mora").read_bytes() == before


def test_the_order_entries_are_written_in_is_no_change():
    """DESIGN 6: references use ids, never array indices (DESIGN 4.1),
    so the diff compares by id and the order of the file is not a
    change."""
    doc = {
        "build_version": "x",
        "nodes": [
            {"id": "a", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "b", "op": "box", "params": {"size": [2.0, 1.0, 1.0]}},
        ],
        "outputs": {"a": "a", "b": "b"},
    }
    swapped = {**doc, "nodes": doc["nodes"][::-1], "outputs": {"b": "b", "a": "a"}}
    diff = irdiff.asset_diff(
        asset_mod.loads(json.dumps(doc)), asset_mod.loads(json.dumps(swapped)),
    )
    assert diff.empty


def test_a_node_under_a_new_id_is_named_as_renamed(tmp_path, capsys):
    """DESIGN 6: of the nodes added and removed, the pairs that are one
    subgraph under two ids are named, since the cache, pins, and the
    journal know a node by its id; a node whose streams the id salts
    is marked, a new id there being a new result."""
    project = _project(tmp_path)
    _diff([str(project)], capsys)
    write_files(project, {"lib/parts.star": LIB.replace('id + "/shaft"', 'id + "/body/shaft"')})
    nodes = _diff([str(project), "--diff"], capsys)["diff"]["assets"]["build/yard.mora"]["nodes"]
    assert nodes["added"]["count"] == nodes["removed"]["count"] == 3
    assert nodes["renamed"] == {"count": 3, "rows": [
        {"from": "post_%d/shaft" % i, "to": "post_%d/body/shaft" % i,
         "site": "lib/parts.star:2", "via": ["yard.star:9"]}
        for i in range(3)
    ]}

    ground = (
        'g = mo.grid(id = "g", size = [4, 4], segments = [2, 2])\n'
        'mo.output("rocks", mo.scatter(g, mo.box(id = "rock", size = [0.1, 0.1, 0.1]),'
        ' id = "%s", count = 3, seed = 1))\n'
    )
    field = write_files(tmp_path / "field", {"field.star": ground % "rocks"})
    _diff([str(field)], capsys)
    write_files(field, {"field.star": ground % "stones"})
    nodes = _diff([str(field), "--diff"], capsys)["diff"]["assets"]["build/field.mora"]["nodes"]
    assert nodes["renamed"]["rows"] == [{
        "from": "rocks", "to": "stones", "site": "field.star:2", "stream_changed": True,
    }]


def test_a_asset_the_named_base_does_not_hold_is_not_compared(tmp_path, capsys):
    """DESIGN 6: under `--diff PATH` an asset the named base holds no IR
    of is left out of the comparison and listed, never read as all
    added, and the outputs that read it are not dirty for it. So an
    unedited consumer of another project compares empty against its
    own saved `build/`."""
    import shutil

    ws = workspace(tmp_path / "ws", {
        "kit": {"kit.star": 'mo.output("block", mo.box(id = "b", size = [1, 1, 1]))\n'
                            'mo.waive_asset("grounded")\n'},
        "yard": {"yard.star": 'mo.output("piece", mo.asset_ref(project = "kit", '
                              'output = "block", id = "p"))\n'
                              'mo.waive_asset("grounded")\n'},
    })
    _diff([str(ws / "yard")], capsys)
    saved = tmp_path / "saved" / "build"
    shutil.copytree(ws / "yard" / "build", saved)
    diff = _diff([str(ws / "yard"), "--diff", str(saved)], capsys)["diff"]
    assert diff["assets"] == {} and diff["unchanged"] == 1
    assert diff["not_compared"] == {"count": 1, "assets": ["kit/build/kit.mora"]}
    # Against the IR on disk, an asset with none yet is what it is: new.
    shutil.rmtree(ws / "kit" / "build")
    disk = _diff([str(ws / "yard"), "--diff"], capsys)["diff"]
    assert "not_compared" not in disk
    assert disk["assets"]["kit/build/kit.mora"]["nodes"]["added"]["count"] == 1


def test_a_change_through_a_helper_is_counted_by_its_callers(tmp_path, capsys):
    """DESIGN 6: where the changed nodes are too many to list and came
    through calls, they are counted by the outermost call beside the
    line that made them, so an edit every part reaches through one
    helper still says which parts changed."""
    lib = (
        'def at(node, id, x):\n'
        '    return mo.transform(node, id = id, translate = [x, 0.5, 0])\n'
        'def row(id, x0, n):\n'
        '    return [at(mo.box(id = "%s/b%d" % (id, i), size = [0.1, 1, 0.1]),'
        ' "%s/t%d" % (id, i), x0 + i) for i in range(n)]\n'
    )
    asset = (
        'load("lib/parts.star", "row")\n'
        'for i, part in enumerate(row("left", 0.0, 8) + row("right", 20.0, 4)):\n'
        '    mo.output("p%d" % i, part)\n'
    )
    project = write_files(tmp_path, {"yard.star": asset, "lib/parts.star": lib})
    _diff([str(project)], capsys)
    write_files(project, {"lib/parts.star": lib.replace("[x, 0.5, 0]", "[x, 0.5, 1]")})
    nodes = _diff([str(project), "--diff"], capsys)["diff"]["assets"]["build/yard.mora"]["nodes"]
    assert nodes["changed"]["count"] == 12
    assert nodes["by_site"] == [{"site": "lib/parts.star:2", "count": 12}]
    assert nodes["by_caller"] == [{"site": "yard.star:2", "count": 12}]


def test_a_asset_only_a_new_reference_reads_is_added_under_a_named_base(tmp_path, capsys):
    """DESIGN 6: under a named base an asset only a new reference reads
    is new, and reported as added; only an asset an existing reference
    read, whose IR the base lacks, is left uncompared."""
    import shutil

    root = 'mo.output("o", mo.box(id = "o", size = [1, 1, 1]))\nmo.waive_asset("grounded")\n'
    project = write_files(tmp_path / "yard", {"yard.star": root})
    _diff([str(project)], capsys)
    saved = tmp_path / "saved" / "build"
    shutil.copytree(project / "build", saved)
    write_files(project, {
        "yard.star": root + 'mo.output("piece", mo.asset_ref("subassets/part.star", '
                            '"block", id = "p"))\n',
        "subassets/part.star": 'mo.output("block", mo.box(id = "b", size = [1, 2, 1]))\n'
                               'mo.waive_asset("grounded")\n',
    })
    diff = _diff([str(project), "--diff", str(saved)], capsys)["diff"]
    assert "not_compared" not in diff
    assert diff["assets"]["build/subassets/part.mora"]["nodes"]["added"] == {
        "count": 1, "ids": ["b"],
    }
    assert diff["assets"]["build/yard.mora"]["outputs"]["added"]["ids"] == ["piece"]
