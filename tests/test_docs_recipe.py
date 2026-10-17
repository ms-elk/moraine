"""The recipes the reference shows are real (DESIGN 8.2): an agent
copies what the reference prints, so every recipe in it builds, and
the IR it shows is the IR that recipe builds to."""

import json
import re

from moraine3d.app.cli import main, reference_text
from support.recipes import write_files

BLOCK = re.compile(r"```python\n# (\S+\.star)\n(.*?)```", re.S)


def _shown_recipes() -> dict[str, dict[str, str]]:
    """The recipe files the reference shows, by project: a block opens
    with a comment naming its file as `<project>/<path>`, and that line
    stays in the file so the sites the reference shows are the
    block's own line numbers plus one."""
    projects: dict[str, dict[str, str]] = {}
    for name, body in BLOCK.findall(reference_text("recipe.md")):
        project, _, path = name.partition("/")
        projects.setdefault(project, {})[path] = f"# {name}\n{body}"
    return projects


def _run(argv, capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_every_recipe_the_reference_shows_passes_validate(tmp_path, monkeypatch, capsys):
    """DESIGN 8.2: documentation an agent reads is a contract to copy
    from, so each recipe shown is a complete project that validates
    green."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    projects = _shown_recipes()
    assert sorted(projects) == ["dock", "steps"]
    assert sorted(projects["steps"]) == [
        "lib/parts.star", "steps.requirements.star", "steps.star",
    ]
    for name, files in projects.items():
        project = write_files(tmp_path / name, files)
        code, report = _run(["validate", str(project)], capsys)
        assert code == 0 and report["ok"] is True, (name, report)


def test_the_shown_requirements_are_the_humans(tmp_path, monkeypatch, capsys):
    """DESIGN 5.2: the file decides the owner. The asserts of the shown
    requirements file report as the human's, the asset's as the
    agent's."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "steps", _shown_recipes()["steps"])
    _, report = _run(["validate", str(project)], capsys)
    owners = {
        a["assert"]: a["owner"] for a in report["asserts"] if not a.get("builtin")
    }
    assert owners == {
        "top_height": "human", "clear_width": "human",
        "step_1_meets_lower": "agent", "step_2_meets_lower": "agent",
    }


def test_the_shown_ir_is_what_the_shown_recipe_builds(tmp_path, capsys):
    """DESIGN 4: the IR is the recipe's deterministic expansion. The IR
    the asset reference shows is the build of the recipe the recipe
    reference shows, line for line apart from the stamp."""
    project = write_files(tmp_path / "dock", _shown_recipes()["dock"])
    code, _ = _run(["build", str(project)], capsys)
    assert code == 0
    built = (project / "build" / "dock.mora").read_text(encoding="utf-8")
    m = re.search(
        r"## A complete minimal asset.*?```json\n(.*?)```",
        reference_text("asset.md"), re.S,
    )
    assert m, "the IR block is missing from reference/asset.md"

    def unstamped(text: str) -> list[str]:
        lines = text.splitlines()
        assert lines[1].startswith('"build_version": ')
        return lines[:1] + lines[2:]

    assert unstamped(m.group(1)) == unstamped(built)
