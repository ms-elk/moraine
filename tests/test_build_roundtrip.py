"""Build tests, the round trip's evaluated half (DESIGN 10): the build
of a fixture's flat recipe evaluates to what the fixture does."""

import pytest

from conftest import IR_FIXTURES
from moraine3d.recipe.build import build
from support import flatten

EXAMPLES = IR_FIXTURES
PROJECTS = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())


def _built(fixture, tmp_path):
    return build(flatten.write_project(fixture, tmp_path / fixture.name))


@pytest.mark.parametrize("fixture", PROJECTS, ids=lambda p: p.name)
def test_the_flat_recipe_evaluates_to_what_the_fixture_does(fixture, tmp_path, monkeypatch):
    """DESIGN 10: the build of a fixture's flat recipe has the same
    result hash for every output. Id equality keeps every
    `USES_NODE_ID` stream where it was."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation
    from support.fixtures import realize

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = realize(fixture, tmp_path / "fixture" / fixture.name)
    expected = Evaluation(asset_mod.load(root)).records()
    result = _built(fixture, tmp_path)
    got = Evaluation(
        result.target.asset, irs={p: b.asset for p, b in result.assets.items()},
    ).records()
    assert {n: r.result_hash for n, r in got.items()} == {
        n: r.result_hash for n, r in expected.items()
    }
    assert {n: r.key for n, r in got.items()} == {n: r.key for n, r in expected.items()}
