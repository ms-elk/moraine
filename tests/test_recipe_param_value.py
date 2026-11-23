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


# -- a threshold that is the knob's value ---------------------------------------------------


def test_a_threshold_takes_the_knobs_resolved_value(tmp_path):
    """DESIGN 4.8: `mo.param_value(id)` is an opaque handle to one of
    that asset's declared knobs, usable as a threshold (`value`,
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
    """DESIGN 4.8: a handle stands as `min`; DESIGN 5.2: clearance's
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
    """DESIGN 5.2: the file owns which knob a threshold follows, not
    the knob's value, which resolves as any knob's does. DESIGN 4.8:
    values resolve in the order default, accepted, override."""
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
    (`param_value_arithmetic`) with the fix of the two keywords, or
    of the derived quantity as a constant, since a handle resolves
    only after the asset has declared its knobs."""
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
    """DESIGN 4.8: a handle is usable as a threshold and nowhere
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


# -- a threshold as scale times the knob plus offset ---------------------------

CART = (
    'load("asset.requirements.star", "POST")\n'
    'w = mo.param("width", 1.2, min = 0.6, max = 2.4)\n'
    'n = mo.param("posts", 3, min = 1, max = 6)\n'
    'mo.output("shelf", mo.box(id = "shelf", size = [w - 2 * POST, 0.02, 0.4]))\n'
    'post = mo.box(id = "post", size = [POST, 0.8, POST])\n'
    'mo.output("row", mo.array(post, id = "row", count = 2 * n + 1,'
    ' step_translate = [1, 0, 0]))\n'
)
SHELF = (
    'POST = 0.04\n'
    'mo.assert_extent(id = "shelf_w", outputs = ["shelf"], measure = "size_x",\n'
    '                 value = mo.param_value("width", offset = -2 * POST))\n'
)


def test_a_threshold_is_the_knob_scaled_and_offset(tmp_path):
    """DESIGN 4.8: `scale` and `offset` are constants of the
    requirements file, and the threshold is `scale` times the knob's
    resolved value plus `offset`. The entry names the knob it follows,
    as one that takes the knob's value does, and such a knob reaches
    the graph through a rebuild (DESIGN 7.2)."""
    doc = ir(_built(tmp_path, SHELF + (
        'mo.assert_extent(id = "half", outputs = ["shelf"], measure = "max_x",\n'
        '                 max = mo.param_value("width", scale = 0.5, offset = 1))\n'
        'mo.assert_extent(id = "twice", outputs = ["shelf"], measure = "size_x",\n'
        '                 max = mo.param_value("width", scale = 2))\n'
        'mo.assert_extent(id = "same", outputs = ["shelf"], measure = "size_x",\n'
        '                 max = mo.param_value("width", scale = 1.0, offset = 0))\n'
    ), CART))
    shelf = _entry(doc, "shelf_w")
    assert shelf["value"] == 1.2 + -0.08 and shelf["from_param"] == {"value": "width"}
    half = _entry(doc, "half")
    assert half["max"] == 0.5 * 1.2 + 1 and half["from_param"] == {"max": "width"}
    assert _entry(doc, "twice")["max"] == 2.4
    assert _entry(doc, "same")["max"] == 1.2
    assert "node" not in next(p for p in doc["params"] if p["id"] == "width")


def test_an_integer_key_follows_an_integer_knob_by_integers(tmp_path):
    """DESIGN 4.8: an integer key takes an integer knob with an integer
    scale and offset, and the result is an integer. A float is refused
    whatever it holds, 1.0 as 2.0 is, since a fraction would pass or
    fail with the knob's parity."""
    line = (
        'POST = 0.04\n'
        'mo.assert_count(id = "posts", outputs = ["row"],'
        ' value = mo.param_value("posts", {call}))\n'
    )
    posts = _entry(ir(_built(tmp_path / "a", line.format(call='scale = 2, offset = 1'), CART)),
                   "posts")
    assert posts["value"] == 7 and isinstance(posts["value"], int)
    assert posts["from_param"] == {"value": "posts"}
    same = _entry(ir(_built(tmp_path / "b", line.format(call='scale = 1, offset = 0'), CART)),
                  "posts")
    assert same["value"] == 3 and isinstance(same["value"], int)
    for i, call in enumerate((
        'scale = 0.5', 'scale = 2.0', 'offset = 1.0', 'scale = 1.0', 'offset = 0.0',
    )):
        err = _refused(tmp_path / str(i), line.format(call=call), CART)
        assert err.code == "bad_assert" and "integer scale and offset" in err.message
        assert err.detail["site"] == "asset.requirements.star:2"


def test_a_float_key_takes_an_integer_knob_by_a_fraction(tmp_path):
    """DESIGN 4.8: only an integer key holds the scale and the offset
    to integers. On a length an integer knob scales by a fraction, and
    the IR the build wrote is one the loader reads."""
    result = _built(tmp_path, (
        'POST = 0.04\n'
        'mo.assert_extent(id = "half", outputs = ["row"], measure = "size_x",\n'
        '                 value = mo.param_value("posts", scale = 0.5))\n'
        'mo.assert_extent(id = "past", outputs = ["row"], measure = "size_x",\n'
        '                 max = mo.param_value("posts", offset = 0.25))\n'
    ), CART)
    doc = ir(result)
    assert _entry(doc, "half")["value"] == 1.5
    assert _entry(doc, "half")["from_param"] == {"value": "posts"}
    assert _entry(doc, "past")["max"] == 3.25
    asset_mod.loads(result.target.text)


@pytest.mark.parametrize("call, needle", [
    ('mo.param_value("width", scale = 0)', "follows no knob"),
    ('mo.param_value("width", scale = "half")', "'scale' is a finite number"),
    ('mo.param_value("width", offset = [0.1])', "'offset' is a finite number"),
    ('mo.param_value("width", offset = True)', "'offset' is a finite number"),
    ('mo.param_value("width", offset = mo.param_value("posts"))', "follows one knob"),
    ('mo.param_value("width", 0.5)', "too many values"),
    ('mo.param_value("width", times = 2)', "unexpected 'times'"),
])
def test_a_scale_and_an_offset_are_constants_of_the_file(tmp_path, call, needle):
    """DESIGN 4.8: `scale` and `offset` go by keyword and are finite
    numbers written in the requirements file. A threshold follows one
    knob, and a scale of 0 follows none."""
    err = _refused(tmp_path, (
        'POST = 0.04\n'
        'mo.assert_extent(id = "w", outputs = ["shelf"], measure = "size_x",'
        f' max = {call})\n'
    ), CART)
    assert err.code == "bad_declaration" and needle in err.message
    assert err.detail["site"] == "asset.requirements.star:2"


def test_a_resolved_threshold_the_kind_refuses_says_how_it_was_reached(tmp_path):
    """DESIGN 4.8: a threshold its kind refuses once resolved fails the
    build at the requirement. DESIGN 8.2: the error states what
    failed, here the knob's value and the transform, since a negative
    offset is no error in itself."""
    requirements = (
        'POST = 0.04\n'
        'mo.assert_clearance(id = "apart", outputs = ["shelf", "row"],\n'
        '                    min = mo.param_value("width", offset = -1.5))\n'
    )
    err = _refused(tmp_path / "a", requirements, CART)
    assert err.code == "bad_assert" and "finite number > 0" in err.message
    assert "'min' is 1 * width + -1.5 = " in err.message
    assert "at width = 1.2" in err.message
    assert err.detail["site"] == "asset.requirements.star:2"
    wide = CART.replace('"width", 1.2', '"width", 2.0')
    assert _entry(ir(_built(tmp_path / "b", requirements, wide)), "apart")["min"] == 0.5
    whole = requirements.replace(', offset = -1.5', '')
    zero = CART.replace('"width", 1.2, min = 0.6', '"width", 0.0, min = 0.0')
    err = _refused(tmp_path / "c", whole, zero.replace("w - 2 * POST", "1"))
    assert "'min' is the knob 'width', which holds 0.0" in err.message


def test_integers_are_computed_exactly_before_the_key_types_them(tmp_path):
    """DESIGN 4.8: the threshold is computed in the numbers as written,
    as the recipe's own arithmetic over the knob is, and the result
    takes the key's type: an integer knob less an integer offset is
    exact where their floats would cancel to nothing."""
    asset = (
        'n = mo.param("big", 9007199254740994, min = 0, max = 9007199254740999)\n'
        'mo.output("box", mo.box(id = "b", size = [n - 9007199254740993, 1, 1]))\n'
    )
    doc = ir(_built(tmp_path, (
        'mo.assert_extent(id = "w", outputs = ["box"], measure = "size_x",\n'
        '                 value = mo.param_value("big", offset = -9007199254740993))\n'
    ), asset))
    wide = _entry(doc, "w")
    assert wide["value"] == 1.0 and isinstance(wide["value"], float)
    assert doc["nodes"][0]["params"]["size"][0] == wide["value"]


@pytest.mark.parametrize("call", ['scale = 1 << 600, offset = 0.5', 'scale = 1 << 600',
                                  'scale = 1.5e300'])
def test_a_threshold_beyond_a_float_is_refused_at_its_requirement(tmp_path, call):
    """DESIGN 4.8: a threshold its kind refuses once resolved fails the
    build at the requirement. One no float holds is such a threshold
    however the arithmetic overflows, and the refusal names the
    transform and the knob's value in a line a reader can take in."""
    asset = (
        'n = mo.param("big", 1 << 600, min = 0, max = 1 << 601)\n'
        'mo.output("box", mo.box(id = "b", size = [1, 1, 1]))\n'
    )
    err = _refused(tmp_path, (
        'mo.assert_extent(id = "w", outputs = ["box"], measure = "size_x",'
        f' max = mo.param_value("big", {call}))\n'
    ), asset)
    assert err.code == "bad_assert" and "'max' must be a finite number" in err.message
    assert "beyond what a float holds" in err.message and len(err.message) < 240
    assert err.detail["site"] == "asset.requirements.star:1"
    assert err.detail["param"] == "big"


def test_moving_the_knob_moves_the_threshold_and_the_report(tmp_path):
    """DESIGN 5.2: the requirements file owns which knob a threshold
    follows and by what transform; the build puts in the value the
    knob resolved to. An accepted width moves the shelf, its threshold
    and the report together, and the report names the knob that moved
    the requirement."""
    from moraine3d.engine import validation
    from moraine3d.engine.evaluator import Evaluation
    from moraine3d.recipe import target

    write_files(tmp_path, {"asset.star": CART, "asset.requirements.star": SHELF})

    def judged() -> dict:
        opened = target.open_target(tmp_path / "asset.star", write=False)
        assert asset_mod.dumps(asset_mod.loads(asset_mod.dumps(opened.asset))) == (
            asset_mod.dumps(opened.asset)
        )
        report, _ = validation.report(
            opened.asset, Evaluation(opened.asset, **opened.inputs),
        )
        return next(a for a in report["asserts"] if a["assert"] == "shelf_w")

    at_default = judged()
    assert at_default["ok"] and at_default["value"] == pytest.approx(1.12)
    assert at_default["measured"] == pytest.approx(1.12)
    write_files(tmp_path, {"params.json": '{\n"width": 2.0\n}\n'})
    moved = judged()
    assert moved["ok"] and moved["value"] == pytest.approx(1.92)
    assert moved["measured"] == pytest.approx(1.92)
    assert moved["from_param"] == {"value": "width"}
    assert moved["site"] == "asset.requirements.star:2"
    # The requirement is no echo of the geometry: a shelf built to the
    # full width fails it at every width.
    write_files(tmp_path, {"asset.star": CART.replace("w - 2 * POST", "w")})
    assert judged()["ok"] is False
