"""What a build leaves in `build/` (DESIGN 4.5, 4.8): the build of a
project's root removes the IR no root of the workspace reaches, deletes
nothing it did not write, and names the projects that read this one
(DESIGN 6)."""

import json
import os
from pathlib import Path

import pytest

from moraine3d.app.cli import main
from moraine3d.recipe import clean
from moraine3d.recipe.build import build
from support.links import link_dir
from support.recipes import workspace, write_files

pytestmark = pytest.mark.real_stamp

PART = ('w = mo.param("w", 1.0, min = 0.5, max = 4.0)\n'
        'b = mo.box(id = "b", size = [w, 1, 1])\n'
        'mo.output("part", mo.transform(b, id = "up", translate = [0, 0.5, 0]))\n')
BLOCK = ('h = mo.param("h", 1.0, min = 0.5, max = 4.0)\n'
         'b = mo.box(id = "b", size = [1, h, 1])\n'
         'mo.output("block", mo.transform(b, id = "up", translate = [0, h / 2, 0]))\n')
# `c` hands the value it is given to its own sub-asset and to `d`.
C = ('width = mo.param("width", 1.0, min = 0.5, max = 4.0)\n'
     'p = mo.asset_ref("subassets/x.star", "part", id = "p", overrides = {"w": width})\n'
     'q = mo.asset_ref(project = "d", output = "block", id = "q", overrides = {"h": width})\n'
     'mo.output("piece", p)\n'
     'mo.output("block", mo.transform(q, id = "far", translate = [5, 0, 0]))\n')
A = ('r = mo.asset_ref(project = "c", output = "piece", id = "r", overrides = {"width": %s})\n'
     'mo.output("thing", r)\n')


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    """The tests build one after another; a file is judged at once,
    and the test of the wait asks for the wait."""
    monkeypatch.setattr(clean, "FRESH_S", 0.0)


def _ws(base: Path) -> Path:
    return workspace(base, {
        "a": {"a.star": A % "2.0"},
        "c": {"c.star": C, "subassets/x.star": PART},
        "d": {"d.star": BLOCK},
    })


def _irs(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.mora"))


def clean_fix() -> str:
    """The consumers' fix where every project could be asked."""
    return clean.Swept(consumers=["x"]).readers["fix"]


def test_a_project_with_two_top_level_assets_loses_no_ir(tmp_path, capsys):
    """DESIGN 4.8: every project's root IR is a root, named by the
    project's root asset. A project that holds two top-level asset
    files has no one root to be read from, so nothing of it is removed
    and the report says why."""
    project = write_files(tmp_path / "yard", {"one.star": BLOCK, "two.star": BLOCK})
    for name in ("one", "two", "one"):
        out = _run(["build", str(project / f"{name}.star")], capsys)
        assert "removed" not in out
        assert out["blocked"]["assets"] == ["one.star", "two.star"]
        assert "lib/" in out["blocked"]["fix"] and "files" not in out["blocked"]
    assert _irs(project) == ["build/one.mora", "build/two.mora"]


def _run(argv, capsys) -> dict:
    capsys.readouterr()
    main(argv)
    return json.loads(capsys.readouterr().out)


# -- what is kept ----------------------------------------------------------------


def test_a_variant_a_sibling_wrote_outlives_the_owners_build(tmp_path):
    """DESIGN 4.8: what no root of the workspace reaches is removed. A
    project that references a sibling under overrides writes the
    variant beside the sibling, and with it the variants of the
    sibling's own references, and the sibling's build, which did not
    write them, keeps them: the root of the project that did still
    reaches them."""
    ws = _ws(tmp_path)
    by_a = build(ws / "a" / "a.star")
    theirs = sorted(
        p.relative_to(ws).as_posix() for p in by_a.written if "a/build" not in p.as_posix()
    )
    assert [t.rsplit("/", 1)[0] for t in theirs] == [
        "c/build", "c/build/subassets", "d/build",
    ]
    by_c = build(ws / "c" / "c.star")
    assert by_c.swept.removed == []
    for rel in theirs:
        assert (ws / rel).is_file()
    assert clean.walk(ws / "c").complete


def test_a_variant_nothing_reaches_any_more_is_removed(tmp_path):
    """DESIGN 4.8: a changed override opens a variant under another
    name, and the variant of the value before is reached by no root
    once its consumer is built again. Each project's build removes
    what is its own to remove."""
    ws = _ws(tmp_path)
    before = {p for p in build(ws / "a" / "a.star").written}
    write_files(ws / "a", {"a.star": A % "3.0"})
    assert build(ws / "a" / "a.star").swept.removed == []
    for name in ("c", "d"):
        stale = sorted(p for p in before if f"{name}/build" in p.as_posix())
        assert stale and all(p.is_file() for p in stale)  # the project's to remove, not `a`'s
        assert sorted(build(ws / name / f"{name}.star").swept.removed) == stale
    assert not any(p.exists() for p in before - {ws.resolve() / "a" / "build" / "a.mora"})


def test_the_ir_of_a_sub_asset_built_on_its_own_goes_with_the_next_root_build(tmp_path):
    """DESIGN 4.8: the build of a project's root removes what no root
    reaches. A sub-asset built on its own leaves its plain IR, which a
    root that reads the sub-asset under overrides never opens; a later
    reader would take it for current."""
    ws = _ws(tmp_path)
    alone = build(ws / "c" / "subassets" / "x.star")
    assert [p.name for p in alone.written] == ["x.mora"]
    assert alone.swept.removed == []  # a sub-asset's build says nothing of the project
    root = build(ws / "c" / "c.star")
    assert [p.relative_to(ws.resolve()).as_posix() for p in root.swept.removed] == [
        "c/build/subassets/x.mora",
    ]


def test_the_ir_of_a_root_asset_since_renamed_is_no_root(tmp_path):
    """DESIGN 4.8: the root is the recipe's to name. The IR of a root
    asset that was renamed keeps nothing alive and is removed."""
    project = write_files(tmp_path, {"old.star": BLOCK})
    build(project / "old.star")
    (project / "old.star").rename(project / "new.star")
    result = build(project / "new.star")
    assert [p.name for p in result.swept.removed] == ["old.mora"]
    assert _irs(project) == ["build/new.mora"]


def test_a_build_that_is_not_the_roots_or_not_written_removes_nothing(tmp_path, capsys):
    """DESIGN 4.8: only a written build of a project's root removes.
    `--check` is never a write, and a preview is never written."""
    ws = _ws(tmp_path)
    build(ws / "c" / "subassets" / "x.star")
    held = _irs(ws)
    out = _run(["build", str(ws / "c"), "--check"], capsys)
    assert "removed" not in out
    assert out["check"]["stale"] == ["c/build/subassets/x.mora"]
    build(ws / "c" / "c.star", offered={"width": 2.0})
    build(ws / "c" / "c.star", write=False)
    assert _irs(ws) == held


# -- what is never removed -------------------------------------------------------


def test_what_the_build_did_not_write_is_named_and_left(tmp_path, capsys):
    """DESIGN 4.5: Moraine must be structurally unable to delete
    anything it did not itself write. A file under `build/` whose name
    is not the writer's, or whose first bytes are not an IR's, is
    reported and left."""
    project = write_files(tmp_path, {
        "yard.star": BLOCK,
        "build/notes.txt": "mine\n",
        "build/other.mora": '{"nodes": []}\n',
        "build/subassets/sketch.json": "{}\n",
        "build/keep/deep.mora": '{\n"build_version": "x"\n}\n',
    })
    out = _run(["build", str(project)], capsys)
    assert "removed" not in out
    assert out["unrecognized"] == {"count": 4, "names": [
        "build/keep", "build/notes.txt", "build/other.mora",
        "build/subassets/sketch.json",
    ]}
    for rel in ("build/notes.txt", "build/other.mora", "build/subassets/sketch.json",
                "build/keep/deep.mora"):
        assert (project / rel).is_file(), rel


def test_every_removal_passes_the_gate(tmp_path, monkeypatch):
    """DESIGN 4.5: every unlink of an IR goes through one function
    behind its own narrow gate: a direct child of the project's own
    two build directories, under the writer's name, beginning as an
    IR begins."""
    ws = _ws(tmp_path)
    build(ws / "a" / "a.star")
    build(ws / "c" / "subassets" / "x.star")
    write_files(ws / "a", {"a.star": A % "3.0"})
    build(ws / "a" / "a.star")
    deleted: list[Path] = []
    real = os.unlink

    def recording(path, *a, **kw):
        deleted.append(Path(path))
        return real(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", recording)
    project = (ws / "c").resolve()
    build(project / "c.star")
    assert len(deleted) == 3
    homes = {project / "build", project / "build" / "subassets"}
    for path in deleted:
        assert path.parent in homes
        assert path.name.endswith(".mora")
    for outside in (ws / "a" / "build" / "a.mora", project / "c.star",
                    project / "build" / "subassets"):
        assert clean.unlink_stale_ir(outside.resolve(), project) is False
        assert outside.exists()


def test_a_build_directory_reached_through_a_link_is_left(tmp_path):
    """DESIGN 4.5: no symlink following. A `subassets/` under `build/`
    that is a link leads somewhere the build did not make, and nothing
    behind it is removed."""
    project = write_files(tmp_path / "yard", {"yard.star": BLOCK})
    elsewhere = write_files(tmp_path / "elsewhere", {
        "stale.mora": '{\n"build_version": "x",\n"nodes": [\n]\n}\n',
    })
    (project / "build").mkdir()
    link_dir(elsewhere, project / "build" / "subassets")
    result = build(project / "yard.star")
    assert result.swept.removed == []
    assert (elsewhere / "stale.mora").is_file()
    assert clean.unlink_stale_ir(
        project.resolve() / "build" / "subassets" / "stale.mora", project.resolve(),
    ) is False


def test_a_walk_that_cannot_read_a_root_removes_nothing_and_says_so(tmp_path, capsys):
    """DESIGN 4.8: what no root reaches is removed, so a root that
    cannot be read stops the removal: the build knows less than the
    disk holds. The report names the file that stopped it and the
    project that could not be asked."""
    ws = _ws(tmp_path)
    build(ws / "a" / "a.star")
    build(ws / "d" / "d.star")
    build(ws / "c" / "subassets" / "x.star")
    root = ws / "a" / "build" / "a.mora"
    root.write_text(root.read_text(encoding="utf-8").replace("\n", " "), encoding="utf-8")
    out = _run(["build", str(ws / "c")], capsys)
    assert "removed" not in out
    assert (ws / "c" / "build" / "subassets" / "x.mora").is_file()
    assert out["blocked"]["files"] == ["a/build/a.mora"]
    assert "moraine build" in out["blocked"]["fix"]
    assert out["consumers"]["unread"] == ["a"]
    assert "'unread'" in out["consumers"]["fix"]


def test_a_file_written_a_moment_ago_is_left_for_a_later_build(tmp_path, monkeypatch):
    """DESIGN 4.8: a file written within the last minute is left. A
    build writes its closure file by file, so a variant can land
    beside this project before the root that names it lands beside
    its own, and in that moment no root reaches it."""
    monkeypatch.setattr(clean, "FRESH_S", 60.0)
    ws = _ws(tmp_path)
    alone = build(ws / "c" / "subassets" / "x.star")
    [plain] = alone.written
    first = build(ws / "c" / "c.star")
    assert first.swept.removed == [] and first.swept.stale == [plain] and plain.is_file()
    old = plain.stat().st_mtime - 61
    os.utime(plain, (old, old))
    assert build(ws / "c" / "c.star").swept.removed == [plain]
    assert not plain.exists()


def test_a_file_is_kept_under_any_spelling_of_its_name(tmp_path):
    """DESIGN 4.8: what no root reaches is removed, and a file a root
    reaches is kept however the path to it is spelled. The supported
    filesystems fold case, and a name written through another spelling
    is the same file."""
    project = write_files(tmp_path / "yard", {"yard.star": BLOCK}).resolve()
    result = build(project / "yard.star")
    [written] = result.written
    spelled = written.with_name(written.name.upper().replace(".MORA", ".mora"))
    swept = clean.sweep(project, {spelled}, remove=True)
    assert swept.removed == [] and swept.stale == []
    assert written.is_file()


# -- who reads the project -------------------------------------------------------


def test_a_report_names_the_projects_that_reference_this_one(tmp_path, capsys):
    """DESIGN 6: `build` and `validate` name the other projects whose
    roots reach an asset of the target's project, so a change here is
    followed by a check there. `--failures-only` passes the key
    through."""
    ws = _ws(tmp_path)
    _run(["build", str(ws / "a")], capsys)
    for argv in (["build", str(ws / "c")], ["validate", str(ws / "c")],
                 ["validate", str(ws / "c"), "--failures-only"],
                 ["build", str(ws / "c"), "--check"]):
        out = _run(argv, capsys)
        assert out["consumers"]["projects"] == ["a"], argv
        assert "workspace root" in out["consumers"]["fix"]
    assert _run(["validate", str(ws / "d")], capsys)["consumers"]["projects"] == ["a", "c"]
    assert "consumers" not in _run(["validate", str(ws / "a")], capsys)


def test_a_project_that_stopped_referencing_is_named_no_more(tmp_path, capsys):
    """DESIGN 6: the consumers are read from the IR on disk, so a
    project is named until its own build says otherwise."""
    ws = _ws(tmp_path)
    for name in ("a", "d"):
        _run(["build", str(ws / name)], capsys)
    write_files(ws / "a", {"a.star": BLOCK})
    assert _run(["build", str(ws / "c")], capsys)["consumers"] == {
        "projects": ["a"], "fix": clean_fix(),
    }
    _run(["build", str(ws / "a")], capsys)
    assert "consumers" not in _run(["build", str(ws / "c")], capsys)


def test_a_project_not_built_is_named_as_not_asked(tmp_path, capsys):
    """DESIGN 6: a report that says less never says less about what it
    did not check. A project with no root IR on disk has no reference
    to read, and the report names it."""
    ws = _ws(tmp_path)
    out = _run(["build", str(ws / "d")], capsys)
    assert out["consumers"] == {
        "projects": [], "not_built": ["a", "c"],
        "fix": "validate the workspace root once: the projects under "
               "'not_built' have no IR on disk that says whether they "
               "reference this one",
    }
    _run(["build", str(ws / "a")], capsys)
    again = _run(["build", str(ws / "d")], capsys)["consumers"]
    assert (again["projects"], again["not_built"]) == (["a"], ["c"])
    _run(["build", str(ws / "c")], capsys)
    assert _run(["build", str(ws / "d")], capsys)["consumers"] == {
        "projects": ["a", "c"], "fix": clean_fix(),
    }


def test_a_project_outside_a_workspace_has_no_consumers(tmp_path, capsys):
    """DESIGN 4.6: references stay workspace-internal, so a project
    outside a workspace is read by none."""
    project = write_files(tmp_path / "yard", {"yard.star": BLOCK})
    write_files(tmp_path / "next", {"next.star": BLOCK})
    assert "consumers" not in _run(["build", str(project)], capsys)


def test_a_writers_temp_file_is_neither_removed_nor_named(tmp_path, capsys):
    """DESIGN 4.5: a build's own in-flight temp beside the IR it writes
    is the writer's, to rename or to remove. Another build leaves it,
    and does not report it as a file that is not the build's."""
    from moraine3d.core import atomic

    project = write_files(tmp_path / "yard", {"yard.star": BLOCK})
    _run(["build", str(project)], capsys)
    temp = atomic.tmp_name(project / "build" / "yard.mora")
    temp.write_bytes(clean.HEADER)
    out = _run(["build", str(project)], capsys)
    assert "unrecognized" not in out and "removed" not in out
    assert temp.is_file()
    assert atomic.in_flight_to(temp.name) == "yard.mora"
    assert atomic.in_flight_to("yard.mora.tmp4242") == "yard.mora"
    for name in ("yard.mora", ".tmp12-3", "yard.mora.tmp", "yard.mora.tmp12-3.bak"):
        assert atomic.in_flight_to(name) is None, name
