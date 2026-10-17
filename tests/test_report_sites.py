"""Every diagnostic maps to a recipe line (DESIGN 4.1, 6): an entry
that names a node, an assert, or a knob carries the site of the call
that made it, so a report never makes its reader search the recipe."""

import asyncio
import json

import pytest

from moraine3d.app.cli import main
from moraine3d.app.server import MoraineServer
from support.recipes import write_files

LIB = '''def post(id, x, height):
    shaft = mo.box(id = id + "/shaft", size = [0.1, height, 0.1])
    return mo.transform(shaft, id = id, translate = [x, height / 2, 0])
'''

ASSET = '''load("lib/parts.star", "post")
h = mo.param("height", 2.0, min = 1.0, max = 3.0)
slab = mo.transform(
    mo.box(id = "slab_box", size = [2, 0.1, 2]),
    id = "slab",
    translate = [0, -0.05, 0],
)
mo.output("slab", slab)
for i in range(2):
    mo.output("post_%d" % i, post("post_%d" % i, i - 0.5, h))
    mo.assert_contact(id = "post_%d_stands" % i, outputs = ["post_%d" % i, "slab"])
mo.waive("slab", "not_sunk")
'''

REQUIREMENTS = '''mo.assert_extent(id = "post_height", outputs = ["post_0"], measure = "size_y", value = 2.5)
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return write_files(tmp_path / "yard", {
        "yard.star": ASSET, "lib/parts.star": LIB,
        "yard.requirements.star": REQUIREMENTS,
    })


def _run(argv, capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_a_failing_assert_names_the_line_that_declared_it(project, capsys):
    """DESIGN 4.1: each assert entry carries the `site` of the recipe
    call that declared it; one written in a loop names the loop's line,
    one in the requirements file that file's."""
    code, report = _run(["validate", str(project)], capsys)
    assert code == 1
    sites = {a["assert"]: a.get("site") for a in report["asserts"]}
    assert sites == {
        "no_overlap": None,  # the built-in guard is in no file
        "post_0_stands": "yard.star:11", "post_1_stands": "yard.star:11",
        "post_height": "yard.requirements.star:1",
    }
    [red] = [a for a in report["asserts"] if not a["ok"]]
    assert (red["assert"], red["owner"]) == ("post_height", "human")


def test_an_output_names_the_line_that_made_its_node(project, capsys):
    """DESIGN 6: every entry that names a node carries its recipe site;
    a node made inside a helper carries the helper's line (DESIGN
    4.8)."""
    _, report = _run(["validate", str(project)], capsys)
    assert {n: o["site"] for n, o in report["outputs"].items()} == {
        "slab": "yard.star:3", "post_0": "lib/parts.star:3", "post_1": "lib/parts.star:3",
    }
    assert [(p["param"], p["site"]) for p in report["params"]] == [("height", "yard.star:2")]
    for cmd in (["stats", str(project)],
                ["query", str(project), "--select", '{"normal_y": {"gt": 0.9}}'],
                ["eval", str(project)]):
        _, out = _run(cmd, capsys)
        assert {n: o["site"] for n, o in out["outputs"].items()} == {
            "slab": "yard.star:3", "post_0": "lib/parts.star:3", "post_1": "lib/parts.star:3",
        }, cmd


def test_failures_only_keeps_the_sites(project, capsys):
    """DESIGN 6: `--failures-only` compresses every gating list to
    counts plus failures; a failure keeps the line to edit."""
    _, slim = _run(["validate", str(project), "--failures-only"], capsys)
    [red] = slim["asserts"]
    assert red["site"] == "yard.requirements.star:1"


def test_an_evaluation_error_names_the_node_and_its_line(project, capsys):
    """DESIGN 8.2: errors state what failed and the shortest path to a
    fix; an operator's refusal names the node and the recipe line that
    wrote it."""
    write_files(project, {"lib/parts.star": LIB.replace("[0.1, height, 0.1]", "[0.1, -height, 0.1]")})
    code, out = _run(["validate", str(project)], capsys)
    assert code == 1
    detail = out["error"]["detail"]
    assert (detail["node"], detail["site"]) == ("post_0/shaft", "lib/parts.star:2")


def test_a_profile_row_names_the_line_of_a_slow_node(project, capsys):
    """DESIGN 6: the `profile` key names the slowest misses, and a row
    that names a node names its line."""
    _, out = _run(["eval", str(project)], capsys)
    rows = out["profile"]["slowest"]
    assert rows and all(r["site"].split(":")[0] in ("yard.star", "lib/parts.star") for r in rows)


def test_a_pin_names_the_line_of_the_node_it_points_at(project):
    """DESIGN 7.2: a pin resolves against file state and records what
    it points at; the record names the recipe line of that node."""
    server = MoraineServer(project / "yard.star")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    pin = asyncio.run(server._add_pin("post_1", face=0, source="api"))
    assert (pin["node_id"], pin["site"]) == ("post_1", "lib/parts.star:3")


def test_a_referenced_assets_entry_names_its_own_recipe(tmp_path, monkeypatch, capsys):
    """DESIGN 4.6: a referenced asset's failing entry is listed with its
    `asset`, and its site is the line of that asset's own recipe."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": (
            'gate = mo.asset_ref("subassets/gate.star", "bar", id = "gate", overrides = {"gap": 0.05})\n'
            'mo.output("gate", gate)\n'
            'mo.waive_asset("grounded")\n'
        ),
        "subassets/gate.star": (
            'g = mo.param("gap", 0.5, min = 0.01, max = 1.0)\n'
            'a = mo.box(id = "a", size = [1, 0.1, 0.1])\n'
            'b = mo.transform(mo.box(id = "b_box", size = [1, 0.1, 0.1]), id = "b", translate = [0, 0.1 + g, 0])\n'
            'mo.output("bar", a)\n'
            'mo.output("rail", b)\n'
            'mo.waive_asset("grounded")\n'
        ),
        "subassets/gate.requirements.star": (
            'mo.assert_clearance(id = "apart", outputs = ["bar", "rail"], min = 0.2)\n'
        ),
    })
    code, report = _run(["validate", str(project)], capsys)
    assert code == 1
    [red] = [a for a in report["asserts"] if not a["ok"]]
    assert red["asset"] == "subassets/gate.star"
    assert red["overrides"] == {"gap": 0.05}
    assert (red["owner"], red["site"]) == ("human", "subassets/gate.requirements.star:1")


def test_an_error_in_a_referenced_asset_names_its_line(tmp_path, monkeypatch, capsys):
    """DESIGN 8.2, 4.6: an operator's refusal names the node and the
    recipe line that wrote it, in an output of a referenced asset no
    consumer evaluates as in any other."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": 'mo.output("gate", mo.asset_ref("subassets/gate.star", "bar", id = "gate"))\n',
        "subassets/gate.star": (
            'mo.output("bar", mo.box(id = "a", size = [1, 0.1, 0.1]))\n'
            'mo.output("rail", mo.box(id = "r", size = [1, -0.1, 0.1]))\n'
        ),
    })
    code, out = _run(["validate", str(project)], capsys)
    assert code == 1
    detail = out["error"]["detail"]
    assert (detail["asset"], detail["node"], detail["site"]) == (
        "subassets/gate.star", "r", "subassets/gate.star:2",
    )


def test_an_entry_names_the_calls_on_the_way_to_its_line(project, capsys):
    """DESIGN 6: every entry that names a node or an assert carries its
    recipe site, and one made inside a function the calls it was
    reached through, so a line shared by many entries still leads to
    the call that asked for this one."""
    want = {"slab": None, "post_0": ["yard.star:10"], "post_1": ["yard.star:10"]}
    for cmd in (["validate", str(project)], ["stats", str(project)],
                ["query", str(project), "--select", '{"normal_y": {"gt": 0.9}}'],
                ["eval", str(project)]):
        _, out = _run(cmd, capsys)
        assert {n: o.get("via") for n, o in out["outputs"].items()} == want, cmd
    server = MoraineServer(project / "yard.star")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    pin = asyncio.run(server._add_pin("post_1", face=0, source="api"))
    assert pin["via"] == ["yard.star:10"]


def test_an_evaluation_error_names_the_calls_on_the_way(project, capsys):
    """DESIGN 8.2: an operator's refusal names the node, its line, and
    the calls on the way to it."""
    write_files(project, {"lib/parts.star": LIB.replace("[0.1, height, 0.1]", "[0.1, -height, 0.1]")})
    _, out = _run(["validate", str(project)], capsys)
    detail = out["error"]["detail"]
    assert (detail["site"], detail["via"]) == ("lib/parts.star:2", ["yard.star:10"])


# -- the rollup by line --------------------------------------------------------

RING = '''def post(id, x, height):
    shaft = mo.box(id = id + "/shaft", size = [0.1, height, 0.1])
    return mo.transform(shaft, id = id, translate = [x, height / 2 - %s, 0])

def row(id, n, sink):
    for i in range(n):
        mo.output("%%s_%%d" %% (id, i), post("%%s_%%d" %% (id, i), i, 1.0 + sink))
'''


def _rollup_project(tmp_path, monkeypatch, lib_sink, calls):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return write_files(tmp_path / "yard", {
        "yard.star": 'load("lib/parts.star", "row")\n' + calls,
        "lib/parts.star": RING % lib_sink,
    })


def test_failing_entries_that_share_a_line_are_counted_by_it(tmp_path, monkeypatch, capsys):
    """DESIGN 6: when several failing entries share one site the report
    adds `by_site`, a rollup of counts per site, so one wrong
    expression in a loop reads as one line with a count rather than as
    its expansion. The entries stay."""
    project = _rollup_project(tmp_path, monkeypatch, "0.4", 'row("a", 3, 0)\nrow("b", 2, 0)\n')
    code, report = _run(["validate", str(project)], capsys)
    assert code == 1
    assert sum(not o["ok"] for o in report["outputs"].values()) == 5
    assert report["by_site"] == [{
        "site": "lib/parts.star:3", "count": 5,
        "via": [
            {"via": ["yard.star:2", "lib/parts.star:7"], "count": 3},
            {"via": ["yard.star:3", "lib/parts.star:7"], "count": 2},
        ],
    }]
    _, slim = _run(["validate", str(project), "--failures-only"], capsys)
    assert slim["by_site"] == report["by_site"]
    assert len(slim["outputs"]) == 5


def test_the_calls_on_the_way_tell_a_shared_line_from_one_call(tmp_path, monkeypatch, capsys):
    """DESIGN 6: where the entries of a line were reached through more
    than one call, the row counts them by the calls on the way. Here
    one call hands the helper a value that sinks its posts, and every
    failing entry leads to that call."""
    project = _rollup_project(tmp_path, monkeypatch, "0", 'row("a", 3, 0)\nrow("b", 2, -0.5)\n')
    _, report = _run(["validate", str(project)], capsys)
    assert report["by_site"] == [{"site": "lib/parts.star:3", "count": 2}]
    failing = [o for o in report["outputs"].values() if not o["ok"]]
    assert {tuple(o["via"]) for o in failing} == {("yard.star:3", "lib/parts.star:7")}


def test_a_report_with_no_shared_line_has_no_rollup(project, capsys):
    """DESIGN 6: the rollup is for the lines several failing entries
    share; one failing entry is its own line."""
    _, report = _run(["validate", str(project)], capsys)
    assert sum(not a["ok"] for a in report["asserts"]) == 1
    assert "by_site" not in report
