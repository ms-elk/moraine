"""The ``threshold_tautology`` warning (DESIGN 5.2): an agent-owned
entry whose threshold is a constant that also builds the geometry."""

from __future__ import annotations

import pytest

from moraine3d.recipe.build import build

from support.recipes import build_recipe, write_files

HEAD = 'W = 2.0\nT = 0.1\n'
BODY = 'mo.output("main", mo.box(id = "bx", size = [W, 1, T]))\n'


def _rows(result) -> list[dict]:
    return [w for w in result.warnings if w["warning"] == "threshold_tautology"]


def _assert(threshold: str, aid: str = "wide") -> str:
    return (
        f'mo.assert_extent(id = "{aid}", outputs = ["main"], measure = "size_x",'
        f' {threshold})\n'
    )


def test_a_constant_that_builds_and_judges_is_reported(tmp_path):
    """DESIGN 5.2: a constant the agent chose lives in the asset, and
    an agent-owned entry that reuses it is reported as a warning
    (`threshold_tautology`). The row names the constant, the line that
    binds it, and the entries it judges, and never gates."""
    result = build_recipe(tmp_path, HEAD + BODY + _assert("value = W"), write=False)
    (row,) = _rows(result)
    assert (row["constant"], row["site"]) == ("W", "asset.star:1")
    assert (row["asserts"], row["count"]) == (["wide"], 1)
    assert row["fix"] == (
        "write the number the requirement states in the assert, or as a "
        "constant of 'asset.requirements.star' that the asset loads"
    )
    assert result.report(wrote=False)["ok"] is True


def test_a_threshold_computed_from_constants_names_each_of_them(tmp_path):
    """DESIGN 5.2: the number that built the geometry becomes the
    number that judges it, whatever arithmetic lies between."""
    result = build_recipe(tmp_path, (
        HEAD + 'INNER = W - 2 * T\n' + BODY
        + _assert("min = INNER", "inner") + _assert("max = W + T", "outer")
    ), write=False)
    assert {r["constant"]: r["asserts"] for r in _rows(result)} == {
        "W": ["inner", "outer"], "T": ["inner", "outer"],
    }


def test_one_constant_is_one_row(tmp_path):
    """DESIGN 8.2: a repeated shape is stated once. A constant that
    judges many entries is one row with their count."""
    asserts = "".join(_assert("max = W", f"a{i}") for i in range(14))
    (row,) = _rows(build_recipe(tmp_path, HEAD + BODY + asserts, write=False))
    assert row["count"] == 14 and len(row["asserts"]) == 10


def test_an_entry_made_by_a_function_or_a_loop_is_counted(tmp_path):
    """DESIGN 5.2: an entry declared while an asset's top level runs is
    the asset's, wherever the call is written."""
    result = build_recipe(tmp_path, HEAD + BODY + (
        'def wide(id):\n'
        '    mo.assert_extent(id = id, outputs = ["main"], measure = "size_x", max = W)\n'
        'for i in range(3):\n'
        '    wide("w%d" % i)\n'
    ), write=False)
    (row,) = _rows(result)
    assert row["asserts"] == ["w0", "w1", "w2"]


@pytest.mark.parametrize("asset", [
    pytest.param(HEAD + BODY + _assert("value = 2.0"), id="a literal"),
    pytest.param(
        HEAD + 'LIMIT = 2.5\n' + BODY + _assert("max = LIMIT"),
        id="a constant that only judges",
    ),
    pytest.param(
        'load("asset.requirements.star", "SPAN")\nT = 0.1\n'
        'mo.output("main", mo.box(id = "bx", size = [SPAN, 1, T]))\n'
        + _assert("value = SPAN"),
        id="the human's constant",
    ),
    pytest.param(
        'load("asset.requirements.star", "SPAN")\nT = 0.1\nHALF = SPAN / 2\n'
        'mo.output("main", mo.box(id = "bx", size = [HALF, 1, T]))\n'
        + _assert("value = HALF"),
        id="computed from the human's constant",
    ),
    pytest.param(
        'W = mo.param("w", 2.0, min = 1.0, max = 3.0)\nT = 0.1\n' + BODY
        + _assert("value = W"),
        id="a knob's value",
    ),
    pytest.param(
        HEAD + BODY + (
            'def wide(id, limit):\n'
            '    mo.assert_extent(id = id, outputs = ["main"], measure = "size_x",'
            ' max = limit)\n'
            'wide("a", W)\n'
        ),
        id="a function's argument",
    ),
    pytest.param(
        HEAD + BODY + (
            'for W in [2.0, 3.0]:\n'
            '    mo.assert_extent(id = "a%d" % int(W), outputs = ["main"],'
            ' measure = "size_x", max = W)\n'
        ),
        id="a loop's variable",
    ),
    pytest.param(
        HEAD + BODY + (
            'def limit(x):\n    return x + 1.0\n'
            + _assert("max = limit(3.0)")
        ),
        id="a function the threshold calls",
    ),
])
def test_what_is_not_the_agents_constant_draws_no_row(tmp_path, asset):
    """DESIGN 5.2: an asset may load the requirement constants and
    build against them, and a knob's value is human-owned data through
    the viewer. DESIGN 4.8: a warning without a fix trains its reader
    to pass warnings by, so a threshold the text does not trace to a
    constant of the asset draws none."""
    result = build_recipe(
        tmp_path, asset, write=False,
        files={"asset.requirements.star": "SPAN = 2.0\n"},
    )
    assert _rows(result) == []


def test_a_constant_of_lib_is_the_agents(tmp_path):
    """DESIGN 4.8: helpers shared by the root and its sub-assets live
    in `lib/`, the agent's files (DESIGN 4.2), so a constant loaded
    from one is the agent's."""
    result = build_recipe(tmp_path, (
        'load("lib/sizes.star", "W", "T")\n' + BODY + _assert("value = W")
    ), write=False, files={"lib/sizes.star": HEAD})
    (row,) = _rows(result)
    assert (row["constant"], row["site"]) == ("W", "asset.star:1")


def test_the_humans_own_entry_is_never_reported(tmp_path):
    """DESIGN 5.2: a requirements file loads nothing, so its
    thresholds are its own constants. The warning is for the agent's
    side."""
    result = build_recipe(tmp_path, HEAD + BODY, write=False, files={
        "asset.requirements.star": "W = 2.0\n" + _assert("value = W"),
    })
    assert _rows(result) == []


def test_the_warning_is_the_assets_own_and_reaches_validate(tmp_path, monkeypatch):
    """DESIGN 4.1: every such warning names an edit site, so it is
    raised for the asset in hand, and a referenced asset's is its
    producer's. DESIGN 4.8: the warning rides `build`'s report and
    `validate`'s warnings."""
    from moraine3d.app import cli

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    project = write_files(tmp_path / "p", {
        "asset.star": 'mo.output("a", mo.asset_ref("subassets/part.star", "main", id = "a"))\n',
        "subassets/part.star": HEAD + BODY + _assert("value = W"),
    })
    assert _rows(build(project / "asset.star", write=False)) == []
    assert len(_rows(build(project / "subassets" / "part.star", write=False))) == 1
    report, _ = cli._validate_report(project / "subassets" / "part.star", None, True)
    rows = [w for w in report["warnings"] if w["warning"] == "threshold_tautology"]
    assert [r["constant"] for r in rows] == ["W"]
    assert report["asserts_ok"] == report["asserts_total"]
