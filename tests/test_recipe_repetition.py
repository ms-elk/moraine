"""Copies written by hand (DESIGN 4.8): the build warns where a loop
places an unchanged part by one step, and names the `array` to write."""

import json

import pytest

from moraine3d.app.cli import main
from support.recipes import build_recipe, write_files

pytestmark = pytest.mark.real_stamp

POST = 'post = mo.box(id = "post", size = [0.1, 1, 0.1])\n'
ROW = POST + '''for i in range(%d):
    copy = mo.transform(post, id = "post_%%d" %% i, translate = [%s, 0.5, 0])
    mo.output("post_%%d" %% i, copy)
'''


def test_a_row_of_copies_draws_the_array_to_write(tmp_path):
    """DESIGN 4.8: a loop whose body emits an unchanged prototype under
    a per-copy transform writes by hand what `array` emits as
    instances. `build` detects the pattern and warns with the `array`
    to write."""
    result = build_recipe(tmp_path, ROW % (4, "1.5 * i - 2"))
    [row] = result.warnings
    assert row["warning"] == "copies_not_instanced"
    assert (row["site"], row["count"]) == ("asset.star:3", 4)
    assert row["nodes"] == ["post_0", "post_1", "post_2", "post_3"]
    assert row["fix"] == (
        "mo.array(<the first copy, 'post_0'>, id = '<name>', count = 4, "
        "step_translate = [1.5, 0, 0])"
    )
    assert "'fix'" in row["note"] and "bake_instances" in row["note"]
    assert row["fix"] not in row["note"]
    assert result.report()["warnings"] == [row]


def test_a_ring_of_copies_draws_the_array_to_write(tmp_path):
    """DESIGN 4.8: the `array` fix covers a placement `array` composes
    the same way, a ring about the origin as a row along a line."""
    result = build_recipe(tmp_path, POST
        + 'out = mo.transform(post, id = "out", translate = [2, 0.5, 0])\n'
        + 'for i in range(6):\n'
        + '    mo.output("p%d" % i, mo.transform(out, id = "p%d" % i, rotate_deg = [0, 60 * i, 0]))\n')
    [row] = result.warnings
    assert row["count"] == 6
    assert "step_rotate_deg = [0, 60, 0]" in row["fix"]


@pytest.mark.parametrize("recipe", [
    ROW % (2, "1.5 * i"),
    ROW % (4, "i * i"),
    POST + 'for i in range(4):\n'
           '    mo.output("p%d" % i, mo.transform(post, id = "p%d" % i, '
           'translate = [i, 0.5, 0], scale = [1, 1 + i, 1]))\n',
    'for i in range(4):\n'
    '    part = mo.box(id = "b%d" % i, size = [0.1, 1 + i, 0.1])\n'
    '    mo.output("p%d" % i, mo.transform(part, id = "p%d" % i, translate = [i, 0.5, 0]))\n',
    'for i in range(4):\n'
    '    part = mo.box(id = "b%d" % i, size = [4, 0.1, 4])\n'
    '    s = mo.scatter(part, mo.box(id = "r%d" % i, size = [0.1, 0.1, 0.1]), '
    'id = "s%d" % i, count = 3, seed = 1)\n'
    '    mo.output("p%d" % i, mo.transform(s, id = "p%d" % i, translate = [5 * i, 0.5, 0]))\n',
], ids=["a pair", "no step", "a scale per copy", "parts that differ", "streams that differ"])
def test_copies_that_differ_or_follow_no_step_draw_no_warning(tmp_path, recipe):
    """DESIGN 4.8: a loop is for copies that differ by rule. A pair,
    copies placed by no step, copies under a scale of their own, and
    parts whose content differs (a node whose id salts a stream
    differs by its id) draw no warning: a warning without a fix trains
    its reader to pass warnings by."""
    assert build_recipe(tmp_path, recipe).warnings == []


def test_parts_of_one_content_are_one_part_under_any_id(tmp_path):
    """DESIGN 4.8: sibling nodes whose content, all but a transform,
    hashes equal. A loop that makes the part anew in every pass names
    it anew, and the copies are of one part all the same."""
    result = build_recipe(tmp_path, 'for i in range(3):\n'
        '    part = mo.box(id = "b%d" % i, size = [0.1, 1, 0.1])\n'
        '    mo.output("p%d" % i, mo.transform(part, id = "p%d" % i, translate = [i, 0.5, 0]))\n')
    [row] = result.warnings
    assert row["nodes"] == ["p0", "p1", "p2"]


def test_the_warning_reaches_validate_and_never_gates(tmp_path, capsys, monkeypatch):
    """DESIGN 4.8: `build` warns in its own report and among
    `validate`'s warnings. A warning is advisory: the report stays
    green and the command succeeds."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {"yard.star": ROW % (4, "1.5 * i - 2")})
    for argv in (["build", str(project)], ["validate", str(project)],
                 ["validate", str(project), "--failures-only"]):
        capsys.readouterr()
        assert main(argv) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is True
        assert [w["warning"] for w in out["warnings"]] == ["copies_not_instanced"], argv


def test_an_export_carries_what_the_build_warned_of(tmp_path, capsys, monkeypatch):
    """DESIGN 9: an export is no quieter than the validate it gates
    on. What the build warned of is in the export's report, written or
    refused."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {"yard.star": ROW % (4, "1.5 * i - 2")})
    capsys.readouterr()
    assert main(["export", str(project), "--profile", "gltf"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "copies_not_instanced" in [w.get("warning") for w in out["warnings"]]
    write_files(project, {"yard.star": (ROW % (4, "1.5 * i - 2")).replace("0.5, 0]", "-3, 0]")})
    assert main(["export", str(project), "--profile", "gltf"]) == 1
    refused = json.loads(capsys.readouterr().out)
    assert refused["ok"] is False
    assert "copies_not_instanced" in [
        w.get("warning") for w in refused["validation"]["warnings"]
    ]


def test_a_referenced_assets_copies_are_its_producers_to_hear_of(tmp_path):
    """DESIGN 4.1: a warning names an edit site, so it is raised for
    root-asset nodes only. A node inside a referenced asset is the
    producer's, reported by the producer's own build."""
    project = write_files(tmp_path, {
        "yard.star": 'r = mo.asset_ref("subassets/row.star", "post_0", id = "r")\n'
                     'mo.output("post", r)\n',
        "subassets/row.star": ROW % (4, "1.5 * i"),
    })
    from moraine3d.recipe.build import build

    assert build(project / "yard.star").warnings == []
    assert len(build(project / "subassets" / "row.star").warnings) == 1
