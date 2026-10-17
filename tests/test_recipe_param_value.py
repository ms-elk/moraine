"""``mo.param_value``: a knob's resolved value as a human-owned
threshold (DESIGN 4.8, 5.2)."""

from __future__ import annotations

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import BuildError
from moraine3d.recipe.build import build

from support.recipes import ir, write_files

ASSET = (
    'h = mo.param("height", 2.0, min = 1.0, max = 3.0)\n'
    'n = mo.param("posts", 3, min = 1, max = 6)\n'
    'c = mo.param("tint", [0.8, 0.2, 0.2, 1.0], min = 0.0, max = 1.0)\n'
    'post = mo.box(id = "bx", size = [0.2, h, 0.2])\n'
    'mo.output("row", mo.array(post, id = "row", count = n, step_translate = [1, 0, 0]))\n'
    'mo.output("main", mo.material(post, id = "mt", name = "paint", base_color = c))\n'
)


def _built(tmp_path, requirements: str, asset: str = ASSET, **files):
    write_files(tmp_path, {
        "asset.star": asset, "asset.requirements.star": requirements, **files,
    })
    return build(tmp_path / "asset.star", write=False)


def _refused(tmp_path, requirements: str, asset: str = ASSET) -> BuildError:
    with pytest.raises(BuildError) as e:
        _built(tmp_path, requirements, asset)
    return e.value


def _entry(doc: dict, aid: str) -> dict:
    return next(a for a in doc["asserts"] if a["id"] == aid)


# -- a threshold taken whole ---------------------------------------------------


def test_a_threshold_takes_the_knobs_resolved_value(tmp_path):
    """DESIGN 4.8: `mo.param_value(id)` is an opaque handle to one of
    that asset's declared knobs, usable whole as a threshold (`value`,
    `tol`, `min`, or `max`). DESIGN 5.2: every assert carries its
    owner, and the entry declared in a requirements file is
    human-owned."""
    doc = ir(_built(tmp_path, (
        'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y",\n'
        '                 value = mo.param_value("height"), tol = 0.01)\n'
        'mo.assert_count(id = "posts", outputs = ["row"], value = mo.param_value("posts"))\n'
    )))
    tall, posts = _entry(doc, "tall"), _entry(doc, "posts")
    assert (tall["value"], tall["from_param"], tall["owner"]) == (
        2.0, {"value": "height"}, "human",
    )
    assert (posts["value"], posts["from_param"]) == (3, {"value": "posts"})
    assert tall["tol"] == 0.01


def test_a_clearance_gap_follows_a_knob_and_is_judged_once_resolved(tmp_path):
    """DESIGN 4.8: a handle stands whole as `min`; DESIGN 5.2: clearance's
    `min` is required and positive, judged where the knob resolves, so a
    knob at zero refuses the requirement at the reference's site."""
    asset = ASSET + 'g = mo.param("gap", 0.02, min = 0.0, max = 0.1)\n'
    requirements = (
        'mo.assert_clearance(id = "apart", outputs = ["row", "main"],\n'
        '                    min = mo.param_value("gap"))\n'
    )
    apart = _entry(ir(_built(tmp_path / "a", requirements, asset)), "apart")
    assert (apart["min"], apart["from_param"]) == (0.02, {"min": "gap"})
    zero = asset.replace('"gap", 0.02', '"gap", 0.0')
    err = _refused(tmp_path / "b", requirements, zero)
    assert err.code == "bad_assert" and "finite number > 0" in err.message
    assert err.detail["site"] == "asset.requirements.star:1"


def test_the_threshold_follows_an_accepted_value(tmp_path):
    """DESIGN 5.2: a threshold is a knob's resolved value, human-owned
    data through the viewer. DESIGN 4.8: values resolve in the order
    default, accepted, override."""
    requirements = (
        'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y",\n'
        '                 max = mo.param_value("height"))\n'
    )
    result = _built(tmp_path, requirements, **{"params.json": '{\n"height": 2.5\n}\n'})
    assert _entry(ir(result), "tall")["max"] == 2.5
    offered = build(tmp_path / "asset.star", offered={"height": 1.5}, write=False)
    assert _entry(ir(offered), "tall")["max"] == 1.5


def test_a_handle_is_resolved_once_for_each_variant(tmp_path):
    """DESIGN 5.2: a sub-asset's requirements file is evaluated with
    the sub-asset's asserts, once per override identity, so a kit's
    own requirements hold for every variant a consumer builds."""
    write_files(tmp_path, {
        "asset.star": (
            'mo.output("a", mo.asset_ref("subassets/part.star", "p", id = "a"))\n'
            'mo.output("b", mo.asset_ref("subassets/part.star", "p", id = "b",'
            ' overrides = {"w": 2.0}))\n'
        ),
        "subassets/part.star": (
            'w = mo.param("w", 0.5, min = 0.1, max = 3.0)\n'
            'mo.output("p", mo.box(id = "p", size = [w, 1, 1]))\n'
        ),
        "subassets/part.requirements.star": (
            'mo.assert_extent(id = "wide", outputs = ["p"], measure = "size_x",\n'
            '                 value = mo.param_value("w"))\n'
        ),
    })
    result = build(tmp_path / "asset.star", write=False)
    found = {
        p.name.count("."): _entry(json.loads(b.text), "wide")["value"]
        for p, b in result.assets.items() if p.name.startswith("part")
    }
    assert found == {1: 0.5, 2: 2.0}


def test_the_handle_is_bound_before_the_asset_declares_its_knob(tmp_path):
    """DESIGN 4.8: a requirements file runs exactly once per build
    whether a `load()` reached it first or the build runs it after the
    recipe, and every `mo.param_value` call it makes is recorded,
    never resolved, until every declaration is in."""
    doc = ir(_built(
        tmp_path,
        (
            'LIMIT = 2.4\n'
            'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y",\n'
            '                 max = mo.param_value("height"))\n'
        ),
        asset='load("asset.requirements.star", "LIMIT")\n' + ASSET,
    ))
    assert _entry(doc, "tall")["max"] == 2.0


def test_an_integer_knob_is_held_as_the_key_holds_a_number(tmp_path):
    """DESIGN 4.8: every assert key declares the number type it reads.
    A float key given an integer holds it as a float, and an integer
    key given a float refuses it."""
    result = _built(tmp_path, (
        'mo.assert_extent(id = "wide", outputs = ["row"], measure = "size_x",\n'
        '                 max = mo.param_value("posts"))\n'
    ))
    assert '"max": 3.0' in result.target.text
    err = _refused(tmp_path / "f", (
        'mo.assert_count(id = "posts", outputs = ["row"],'
        ' value = mo.param_value("height"))\n'
    ))
    assert err.code == "bad_assert" and "integer default" in err.message
    assert err.detail["site"] == "asset.requirements.star:1"


def test_the_record_passes_the_loader_and_the_report(tmp_path):
    """DESIGN 4.8: the entry records the knob each threshold was taken
    from, so a requirement that moved with an accepted value says
    which knob moved it."""
    from moraine3d.engine import validation
    from moraine3d.engine.evaluator import Evaluation
    from moraine3d.recipe import target

    write_files(tmp_path, {
        "asset.star": ASSET,
        "asset.requirements.star": (
            'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y",\n'
            '                 max = mo.param_value("height"))\n'
        ),
    })
    opened = target.open_target(tmp_path / "asset.star", write=False)
    assert asset_mod.dumps(asset_mod.loads(asset_mod.dumps(opened.asset))) == (
        asset_mod.dumps(opened.asset)
    )
    report, _ = validation.report(opened.asset, Evaluation(opened.asset, **opened.inputs))
    tall = next(a for a in report["asserts"] if a["assert"] == "tall")
    assert tall["ok"] and tall["from_param"] == {"max": "height"}
    doc = json.loads(asset_mod.dumps(opened.asset))
    _entry(doc, "tall")["from_param"] = {"min": "height"}
    with pytest.raises(Exception) as e:
        asset_mod.loads(json.dumps(doc))
    assert e.value.code == "bad_assert"


# -- refusals ------------------------------------------------------------------


@pytest.mark.parametrize("expr", [
    'mo.param_value("height") + 0.1',
    '2 * mo.param_value("height")',
    'mo.param_value("height") / 2',
    '-mo.param_value("height")',
    'mo.param_value("height") < 2.0',
    'max(mo.param_value("height"), 2.0)',
    'float(mo.param_value("height"))',
])
def test_arithmetic_on_a_handle_is_refused(tmp_path, expr):
    """DESIGN 4.8: arithmetic on a handle is refused
    (`param_value_arithmetic`) with the fix of writing the derived
    quantity as a constant, since a handle resolves only after the
    asset has declared its knobs."""
    err = _refused(tmp_path, (
        f'limit = {expr}\n'
        'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y", max = limit)\n'
    ))
    assert err.code == "param_value_arithmetic"
    assert "constant" in err.message
    assert err.detail["site"] == "asset.requirements.star:1"


@pytest.mark.parametrize("lines", [
    'if mo.param_value("posts"):\n    {tall}',
    'if mo.param_value("posts") == 3:\n    {tall}',
    'if not mo.param_value("posts"):\n    {tall}',
    'if mo.param_value("posts") and True:\n    {tall}',
    'spare = mo.param_value("posts")\n{tall}',
])
def test_a_handle_no_threshold_took_is_refused(tmp_path, lines):
    """DESIGN 4.8: a handle that became no assert's threshold is
    refused (`param_value_unused`), which is where a condition written
    on one shows: a requirements file cannot branch on a knob's
    value."""
    tall = 'mo.assert_extent(id = "tall", outputs = ["main"], measure = "size_y", max = 2.5)'
    body = lines.format(tall=tall).replace("\n", "\n    ")
    err = _refused(tmp_path, f"def check():\n    {body}\ncheck()\n")
    assert err.code == "param_value_unused"
    assert "cannot" in err.message and "branch" in err.message
    assert err.detail["site"] == "asset.requirements.star:2"
    assert err.detail["param"] == "posts"


@pytest.mark.parametrize("line, code", [
    ('mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y",'
     ' max = [mo.param_value("height")])\n', "param_value_misplaced"),
    ('mo.assert_extent(id = "a", outputs = [mo.param_value("height")],'
     ' measure = "size_y", max = 2.0)\n', "param_value_misplaced"),
    ('mo.budget(materials = mo.param_value("posts"))\n', "param_value_misplaced"),
    ('mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y",'
     ' max = mo.param_value("missing"))\n', "param_unknown"),
    ('mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y",'
     ' max = mo.param_value("tint"))\n', "param_value_kind"),
    ('mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y",'
     ' max = mo.param_value(3))\n', "bad_declaration"),
])
def test_a_handle_anywhere_but_a_threshold_is_refused(tmp_path, line, code):
    """DESIGN 4.8: a handle is usable whole as a threshold and nowhere
    else, and it names one of that asset's declared knobs. A threshold
    is one number."""
    err = _refused(tmp_path, line)
    assert err.code == code
    assert err.detail["site"] == "asset.requirements.star:1"


def test_a_asset_takes_its_knobs_value_from_mo_param(tmp_path):
    """DESIGN 4.8: `mo.param_value` is a requirements file's builtin.
    An asset has the value `mo.param` returned, and a handle an asset
    loads from its requirements file judges nothing of the agent's."""
    err = _refused(tmp_path, "", ASSET + (
        'mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y",'
        ' max = mo.param_value("height"))\n'
    ))
    assert err.code == "param_value_outside_requirements"
    assert "mo.param returned" in err.message and err.detail["site"] == "asset.star:7"
    loaded = 'load("asset.requirements.star", "H")\n' + ASSET
    for use in (
        'mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y", max = H)\n',
        'mo.output("x", mo.box(id = "x", size = [1, H, 1]))\n',
    ):
        err = _refused(tmp_path, 'H = mo.param_value("height")\n', loaded + use)
        assert err.code == "param_value_misplaced" and "'height'" in err.message


@pytest.mark.parametrize("file", ["asset.star", "asset.requirements.star"])
def test_the_record_is_the_builds_to_write(tmp_path, file):
    """DESIGN 4.1: `from_param` is the build's record of the knob a
    threshold was taken from. A recipe that writes it states a knob
    the threshold never followed, so the call refuses the keyword as
    it refuses `owner`."""
    line = (
        'mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y", max = 2.0,'
        ' from_param = {"max": "height"})\n'
    )
    asset, requirements = (ASSET + line, "") if file == "asset.star" else (ASSET, line)
    err = _refused(tmp_path, requirements, asset)
    assert err.code == "bad_assert" and "mo.param_value" in err.message
    assert err.detail["site"].startswith(file)


def test_what_the_interpreter_does_not_refuse_is_stated(tmp_path):
    """DESIGN 4.8: a handle is opaque. Equality and truth are the
    language's own for any value and tell a recipe nothing of the
    knob: a handle equals no number and is true."""
    doc = ir(_built(tmp_path, (
        'h = mo.param_value("height")\n'
        'same = 1.0 if h == 2.0 else 2.0\n'
        'held = 3.0 if h else 4.0\n'
        'mo.assert_extent(id = "a", outputs = ["main"], measure = "size_y", max = h)\n'
        'mo.assert_extent(id = "b", outputs = ["main"], measure = "size_x", max = same)\n'
        'mo.assert_extent(id = "c", outputs = ["main"], measure = "size_z", max = held)\n'
    )))
    assert (_entry(doc, "b")["max"], _entry(doc, "c")["max"]) == (2.0, 3.0)
