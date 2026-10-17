"""`moraine build` (DESIGN 6): the command, its report, and its
failure."""

import pytest
import json

from moraine3d import dist_version
from moraine3d.app.cli import main
from moraine3d.recipe.version import build_version
from support.recipes import workspace, write_files

pytestmark = pytest.mark.real_stamp

ASSET = '''slab = mo.box(id = "slab", size = [2, 0.1, 2])
mo.output("slab", slab)
'''


def _out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_build_takes_a_project_directory_or_a_recipe_path(tmp_path, capsys):
    """DESIGN 6: `PROJECT` is a project directory holding its root
    asset file or an explicit recipe path, which may be a sub-asset."""
    write_files(tmp_path / "yard", {
        "yard.star": ASSET, "subassets/part.star": ASSET,
        "yard.requirements.star": "mo.budget(materials = 2)\n",
    })
    assert main(["build", str(tmp_path / "yard")]) == 0
    by_dir = _out(capsys)
    assert by_dir["written"] == ["build/yard.mora"]
    assert main(["build", str(tmp_path / "yard" / "yard.star")]) == 0
    assert _out(capsys)["unchanged"] == 1
    assert main(["build", str(tmp_path / "yard" / "subassets" / "part.star")]) == 0
    assert _out(capsys)["written"] == ["build/subassets/part.mora"]


def test_the_report_leads_with_the_tool_and_names_the_build(tmp_path, capsys):
    """DESIGN 6: every evaluated report leads with `moraine_version`,
    and `build` reports `build_version`, the files written, and per
    asset the node, assert, and output counts."""
    write_files(tmp_path, {"yard.star": ASSET})
    assert main(["build", str(tmp_path)]) == 0
    out = _out(capsys)
    assert list(out)[:3] == ["moraine_version", "ok", "build_version"]
    assert out["moraine_version"] == dist_version()
    assert out["build_version"] == build_version()
    assert out["assets"] == {
        "build/yard.mora": {"recipe": "yard.star", "nodes": 1, "asserts": 0, "outputs": 1},
    }
    assert isinstance(out["ms"], float)


def test_a_build_failure_is_the_commands_failure(tmp_path, capsys):
    """DESIGN 6: a build failure is the command's failure, reported with
    the recipe site and, for a construct Starlark lacks, the spelling
    to use instead."""
    write_files(tmp_path, {"yard.star": "n = 0\nwhile n < 3:\n    n += 1\n"})
    assert main(["build", str(tmp_path)]) == 1
    err = _out(capsys)["error"]
    assert err["code"] == "unsupported_syntax"
    assert err["detail"]["site"] == "yard.star:2"
    assert "range(n)" in err["detail"]["fix"]
    assert not (tmp_path / "build").exists()


def test_build_takes_one_project(tmp_path, capsys):
    """DESIGN 6: write-side commands take one project or asset per
    invocation; pointed at a workspace, they fail with an actionable
    `workspace_target` error listing its projects."""
    root = workspace(tmp_path, {"yard": {"yard.star": ASSET}, "lamp": {"lamp.star": ASSET}})
    assert main(["build", str(root)]) == 1
    err = _out(capsys)["error"]
    assert err["code"] == "workspace_target"
    assert err["detail"]["projects"] == ["lamp", "yard"]


def test_a_directory_without_a_asset_file_says_what_a_project_holds(tmp_path, capsys):
    """DESIGN 4.2: a directory with exactly one top-level asset file is
    a project."""
    write_files(tmp_path, {"yard.requirements.star": "K = 1\n", "lib/parts.star": "K = 1\n"})
    assert main(["build", str(tmp_path)]) == 1
    assert _out(capsys)["error"]["code"] == "asset_not_found"


def test_the_session_log_names_the_revision_a_build_read(tmp_path, monkeypatch, capsys):
    """DESIGN 6: every event carries, where the command reached an
    asset, the root, the project, and `asset_hash`, the content hash
    over every contributing recipe file's bytes."""
    from moraine3d.core import asset as asset_mod

    log = tmp_path / "log"
    monkeypatch.setenv("MORAINE_LOG_DIR", str(log))
    root = workspace(tmp_path / "ws", {"yard": {
        "yard.star": 'load("lib/parts.star", "slab")\nmo.output("slab", slab("slab"))\n',
        "lib/parts.star": 'def slab(id):\n    return mo.box(id = id, size = [2, 0.1, 2])\n',
        "yard.requirements.star": "mo.budget(materials = 2)\n",
    }})
    assert main(["build", str(root / "yard")]) == 0
    capsys.readouterr()
    [day] = log.glob("*.jsonl")
    [event] = [json.loads(l) for l in day.read_text(encoding="utf-8").splitlines()]
    assert (event["event"], event["cmd"], event["ok"]) == ("command", "build", True)
    assert event["project"] == "yard"
    files = {root / "yard" / n for n in ("yard.star", "lib/parts.star", "yard.requirements.star")}
    assert event["asset_hash"] == asset_mod.content_hash(files, root.resolve())
    assert event["summary"]["nodes"] == 1 and event["summary"]["written"] == 1


def test_the_revision_is_the_bytes_the_build_read(tmp_path, monkeypatch, capsys):
    """DESIGN 6: `asset_hash` names the revision a command evaluated; a
    save that lands while the command runs is the next revision."""
    from moraine3d.core import asset as asset_mod
    from moraine3d.engine import evaluator

    log = tmp_path / "log"
    monkeypatch.setenv("MORAINE_LOG_DIR", str(log))
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {"yard.star": ASSET})
    built = asset_mod.content_hash({project / "yard.star"}, project.resolve())
    records = evaluator.Evaluation.records

    def saved_meanwhile(self, only=None):
        write_files(project, {"yard.star": ASSET.replace("0.1", "0.2")})
        return records(self, only)

    monkeypatch.setattr(evaluator.Evaluation, "records", saved_meanwhile)
    assert main(["stats", str(project)]) == 0
    capsys.readouterr()
    [day] = log.glob("*.jsonl")
    [event] = [json.loads(l) for l in day.read_text(encoding="utf-8").splitlines()]
    assert event["asset_hash"] == built
    assert event["asset_hash"] != asset_mod.content_hash(
        {project / "yard.star"}, project.resolve(),
    )


def test_a_build_reads_each_file_once(tmp_path, monkeypatch):
    """DESIGN 6: one revision has one name, so every asset and variant
    of a build runs the same bytes of a file they share."""
    from pathlib import Path

    from moraine3d.recipe.build import build

    project = write_files(tmp_path / "kit", {
        "kit.star": (
            'load("lib/parts.star", "bar")\n'
            'mo.output("a", mo.asset_ref("subassets/part.star", "p", id = "a", overrides = {"w": 1.0}))\n'
            'mo.output("b", mo.asset_ref("subassets/part.star", "p", id = "b", overrides = {"w": 2.0}))\n'
            'mo.output("c", mo.asset_ref("subassets/part.star", "p", id = "c"))\n'
        ),
        "subassets/part.star": (
            'load("lib/parts.star", "bar")\n'
            'w = mo.param("w", 0.5, min = 0.1, max = 3.0)\n'
            'mo.output("p", bar("p", w))\n'
        ),
        "subassets/part.requirements.star": "mo.budget(materials = 2)\n",
        "lib/parts.star": 'def bar(id, w):\n    return mo.box(id = id, size = [w, 1, 1])\n',
    })
    reads: dict[str, int] = {}
    read_bytes = Path.read_bytes

    def counted(self):
        if self.name.endswith(".star"):
            reads[self.name] = reads.get(self.name, 0) + 1
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counted)
    result = build(project / "kit.star", write=False)
    assert len(result.assets) == 4
    assert reads == {
        "kit.star": 1, "part.star": 1, "part.requirements.star": 1, "parts.star": 1,
    }
    found = {p.name: data for p, data in result.contents.items()}
    assert found.pop("kit.requirements.star") is None
    assert found.pop("params.json") is None
    assert found == {p.name: read_bytes(p) for p in result.sources}


# -- check ---------------------------------------------------------------------

CLOSURE = {
    "yard.star": 'p = mo.asset_ref("subassets/part.star", "slab", id = "p")\n'
                 'mo.output("slab", p)\n',
    "subassets/part.star": ASSET,
}


def test_check_is_ok_when_the_disk_holds_what_the_recipe_builds(tmp_path, capsys):
    """DESIGN 6: `--check` builds to memory and compares with the IR on
    disk: `ok`. DESIGN 4.4: the build is a pure function of its inputs,
    and `build --check` is the proof."""
    write_files(tmp_path, CLOSURE)
    assert main(["build", str(tmp_path)]) == 0
    _out(capsys)
    assert main(["build", str(tmp_path), "--check"]) == 0
    out = _out(capsys)
    assert out["ok"] is True
    assert out["check"] == {"same": 2}
    assert "written" not in out and "unchanged" not in out
    assert sorted(out["assets"]) == ["build/subassets/part.mora", "build/yard.mora"]


def test_check_quotes_the_first_differing_line_from_both(tmp_path, capsys):
    """DESIGN 6: `--check` answers `ok`, or the first differing line
    quoted from both, and the difference is the command's failure."""
    write_files(tmp_path, CLOSURE)
    assert main(["build", str(tmp_path)]) == 0
    _out(capsys)
    write_files(tmp_path, {"subassets/part.star": ASSET.replace("[2, 0.1, 2]", "[3, 0.1, 2]")})
    assert main(["build", str(tmp_path), "--check"]) == 1
    out = _out(capsys)
    assert out["ok"] is False
    [row] = out["check"]["differs"]
    assert (row["file"], row["recipe"], row["line"]) == (
        "build/subassets/part.mora", "subassets/part.star", 4,
    )
    assert '"size": [3.0, 0.1, 2.0]' in row["built"]
    assert '"size": [2.0, 0.1, 2.0]' in row["on_disk"]
    assert out["check"]["same"] == 1
    assert "moraine build" in out["check"]["fix"]


def test_check_never_writes(tmp_path, capsys):
    """DESIGN 6: `--check` is never a write: a file that differs keeps
    its bytes, and a file the disk does not hold is named, not made."""
    write_files(tmp_path, CLOSURE)
    assert main(["build", str(tmp_path), "--check"]) == 1
    out = _out(capsys)
    assert out["check"] == {
        "same": 0,
        "missing": ["build/subassets/part.mora", "build/yard.mora"],
        "fix": out["check"]["fix"],
    }
    assert not (tmp_path / "build").exists()
    assert main(["build", str(tmp_path)]) == 0
    _out(capsys)
    ir_path = tmp_path / "build" / "yard.mora"
    before = ir_path.read_bytes()
    write_files(tmp_path, {"yard.star": CLOSURE["yard.star"] + 'mo.waive_asset("grounded")\n'})
    assert main(["build", str(tmp_path), "--check"]) == 1
    _out(capsys)
    assert ir_path.read_bytes() == before


def test_check_quotes_a_long_line_in_part(tmp_path, capsys):
    """DESIGN 8.2: a report states what failed, never a dump. A long
    line is quoted up to where a reader finds the difference."""
    profile = ", ".join(f"[{1 + i / 1000}, {i / 100}]" for i in range(120))
    recipe = (f'v = mo.revolve(id = "v", profile = [{profile}], segments = %d)\n'
              'mo.output("vase", v)\n')
    write_files(tmp_path, {"yard.star": recipe % 8})
    assert main(["build", str(tmp_path)]) == 0
    _out(capsys)
    write_files(tmp_path, {"yard.star": recipe % 9})
    assert main(["build", str(tmp_path), "--check"]) == 1
    [row] = _out(capsys)["check"]["differs"]
    assert row["built"].endswith("more characters)") and len(row["built"]) < 450
