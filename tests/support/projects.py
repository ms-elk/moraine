"""Projects on disk for tests written against IR fixtures (DESIGN 10).

A test that states its asset as an IR document reaches the runtime one
of two ways. ``write_project`` translates the fixture into a recipe
project, which every command builds first (DESIGN 6). ``write_ir``
writes the document as an IR file, the form a command evaluates as
given when handed the path; the tests' build stamps its IR as the
fixtures are stamped (conftest).
"""

from __future__ import annotations

import json
from pathlib import Path

from moraine3d.core import asset as asset_mod
from moraine3d.core import sources as sources_mod
from moraine3d.recipe.build import BuildResult, build

from . import flatten
from .ir import with_knob_values


def _doc(fixture: str | dict) -> dict:
    doc = json.loads(fixture) if isinstance(fixture, str) else fixture
    return with_knob_values(doc)


def write_project(project: Path, root: str | dict, *, name: str = "asset",
                  subassets: dict[str, str | dict] | None = None) -> Path:
    """Write a recipe project from IR fixtures: ``root`` as the root
    asset ``<name>.star`` and each of ``subassets`` (stem -> fixture)
    under ``subassets/``. Returns the root asset file."""
    for stem, fixture in (subassets or {}).items():
        flatten.write_doc(_doc(fixture), project, stem, sub=True)
    return flatten.write_doc(_doc(root), project, name)


def write_workspace(root: Path, projects: dict[str, str | dict]) -> Path:
    """A workspace of recipe projects, each from one root fixture and
    named after its directory."""
    (root / ".moraine").mkdir(parents=True, exist_ok=True)
    for name, fixture in projects.items():
        write_project(root / name, fixture, name=name)
    return root


def built(target: Path) -> BuildResult:
    """Build a project directory or an asset file."""
    target = Path(target)
    return build(sources_mod.root_asset(target) if target.is_dir() else target)


def load(target: Path) -> asset_mod.Asset:
    """The IR a project builds to, written under its ``build/``."""
    return built(target).target.asset


def write_ir(path: Path, fixture: str | dict) -> Path:
    """Write an IR file as a command reads one handed over by path,
    LF endings."""
    text = fixture if isinstance(fixture, str) else json.dumps(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path
