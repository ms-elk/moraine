"""Every shipped example project is a recipe that builds and validates
green, and says exactly what the IR it was written from says.

The example recipes live in the package (src/moraine3d/docs/examples/,
DESIGN 5: shipped and printed with the section that cites them), so a
red example is a broken promise, not just a stale sample. Each was
written by hand from an IR fixture (tests/fixtures/ir/<name>), and the
fixture stays the measure of the rewrite: the same node ids, outputs,
groups, asserts, knobs and their bindings, and the same result hash
for every output (DESIGN 10), so no value moved in the writing.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import IR_FIXTURES, SRC_ROOT
from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.recipe.build import build
from moraine3d.recipe.version import build_version
from support import flatten
from support.engine import Evaluation
from support.fixtures import realize

EXAMPLES_ROOT = SRC_ROOT / "docs" / "examples"
EXAMPLES = sorted(p for p in EXAMPLES_ROOT.iterdir() if p.is_dir())
SUBASSETS = sorted(
    p for p in EXAMPLES_ROOT.glob("*/subassets/*.star")
    if not p.name.endswith(".requirements.star")
)
GOLDEN = Path(__file__).parent / "golden" / "example_sums.json"


def _rel(p: Path) -> str:
    return p.relative_to(EXAMPLES_ROOT).as_posix()


def _copy(project: Path, tmp_path: Path) -> Path:
    """The package tree is a delivery, never a workspace (DESIGN 5): a
    build writes beside the asset, so every test works on a copy."""
    dest = tmp_path / project.name
    shutil.copytree(project, dest)
    return dest


def _fixture_doc(project: Path, rel: str) -> dict:
    return json.loads((IR_FIXTURES / project.name / rel).read_text(encoding="utf-8"))


@pytest.mark.parametrize("project", EXAMPLES, ids=lambda p: p.name)
def test_example_validates_green(project, tmp_path, monkeypatch):
    """DESIGN 10: every shipped example builds and validates green."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["validate", str(_copy(project, tmp_path))]) == 0


@pytest.mark.parametrize("project", EXAMPLES, ids=lambda p: p.name)
def test_example_says_what_its_fixture_says(project, tmp_path):
    """DESIGN 10: the hand-written recipe builds to the IR it was
    written from — the same node ids and contents, outputs, groups,
    asserts, checks, budgets, and knobs — and binds every knob to the
    node param the fixture bound it to, so its tier holds."""
    result = build(_copy(project, tmp_path) / f"{project.name}.star")
    root = (tmp_path / project.name).resolve()
    compared = 0
    for ir_path, built in result.assets.items():
        rel = ir_path.relative_to(root / "build")
        if rel.name.count(".") > 1:
            continue  # a variant: no fixture file holds it
        fixture = _fixture_doc(project, rel.as_posix())
        doc = json.loads(built.text)
        assert flatten.differences(fixture, doc, order=False) == [], rel
        bound = {p["id"]: (p["node"], p["param"]) for p in fixture.get("params") or [] if "node" in p}
        built_bound = {p["id"]: (p.get("node"), p.get("param")) for p in doc.get("params") or []}
        for pid, where in bound.items():
            assert built_bound.get(pid) == where, (rel, pid)
        compared += 1
    assert compared >= 1


@pytest.mark.parametrize("project", EXAMPLES, ids=lambda p: p.name)
def test_example_evaluates_to_what_its_fixture_does(project, tmp_path, monkeypatch):
    """DESIGN 10: no value moved in the rewrite — every output's result
    hash and cache key equal the fixture's."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = realize(IR_FIXTURES / project.name, tmp_path / "fixture" / project.name)
    expected = Evaluation(asset_mod.load(root)).records()
    result = build(_copy(project, tmp_path) / f"{project.name}.star")
    got = Evaluation(
        result.target.asset, irs={p: b.asset for p, b in result.assets.items()},
    ).records()
    assert {n: r.result_hash for n, r in got.items()} == {
        n: r.result_hash for n, r in expected.items()
    }
    assert {n: r.key for n, r in got.items()} == {n: r.key for n, r in expected.items()}


@pytest.mark.parametrize("asset_file", SUBASSETS, ids=_rel)
def test_example_subasset_validates_standalone(asset_file, tmp_path, monkeypatch):
    """DESIGN 4.6: a sub-asset is a complete, individually buildable and
    evaluable asset — pointed at the file directly, `validate` stays
    green."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = asset_file.parent.parent
    dest = _copy(project, tmp_path)
    assert main(["validate", str(dest / "subassets" / asset_file.name)]) == 0


def _sums(tmp_path: Path) -> dict[str, str]:
    """sha256 of every IR the shipped examples build to, by its path
    under the example, the stamp's line left out (its own test pins
    it)."""
    out = {}
    for project in EXAMPLES:
        result = build(_copy(project, tmp_path) / f"{project.name}.star")
        for ir_path, built in result.assets.items():
            lines = built.text.split("\n")
            assert lines[1] == f'"build_version": "{build_version()}",'
            body = "\n".join(lines[:1] + lines[2:])
            rel = ir_path.relative_to(tmp_path.resolve()).as_posix()
            out[rel] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return dict(sorted(out.items()))


@pytest.mark.real_stamp
def test_examples_build_to_the_frozen_bytes(tmp_path):
    """DESIGN 10: the same recipe builds byte-identical twice and across
    the two platforms; the sums are frozen in the repository, so the
    suite passing on Windows and on macOS is the comparison."""
    sums = _sums(tmp_path / "a")
    assert sums == _sums(tmp_path / "b")
    if os.environ.get("MORAINE_REGEN_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        with open(GOLDEN, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(sums, indent=1) + "\n")
    assert sums == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_every_operator_appears_in_an_example():
    """DESIGN 5: every operator ships a minimal example asset, read per
    authoring pattern, so each registered name is called in at least
    one shipped example recipe."""
    from moraine3d import ops

    text = "\n".join(
        p.read_text(encoding="utf-8") for p in EXAMPLES_ROOT.rglob("*.star")
    )
    missing = [name for name in ops.names() if f"mo.{name}(" not in text]
    assert missing == []


def test_every_fixture_has_its_example_and_no_other():
    """The shipped examples and the IR corpus are the same twenty-one
    projects: a fixture without its recipe is an example not yet
    written, a recipe without its fixture an example no measure holds."""
    assert [p.name for p in EXAMPLES] == sorted(
        p.name for p in IR_FIXTURES.iterdir() if p.is_dir()
    )
