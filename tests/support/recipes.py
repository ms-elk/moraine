"""Recipe projects on disk, for the build's tests (DESIGN 10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moraine3d.errors import BuildError
from moraine3d.recipe.build import BuildResult, build


def write_files(base: Path, files: dict[str, str]) -> Path:
    """Write ``files`` (relative POSIX name -> text) under ``base``,
    LF endings, and return ``base``."""
    for name, text in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return base


def workspace(base: Path, projects: dict[str, dict[str, str]]) -> Path:
    """A workspace root holding ``projects`` (name -> files)."""
    (base / ".moraine").mkdir(parents=True, exist_ok=True)
    for name, files in projects.items():
        write_files(base / name, files)
    return base


def build_recipe(base: Path, text: str, *, name: str = "asset",
                 files: dict[str, str] | None = None, **kw) -> BuildResult:
    """Build a project whose root asset is ``text``."""
    write_files(base, {f"{name}.star": text, **(files or {})})
    return build(base / f"{name}.star", **kw)


def ir(result: BuildResult) -> dict:
    """The target's IR as a document."""
    return json.loads(result.target.text)


def refusal(base: Path, text: str, **kw) -> BuildError:
    """The error a recipe's build fails with."""
    with pytest.raises(BuildError) as e:
        build_recipe(base, text, **kw)
    return e.value
