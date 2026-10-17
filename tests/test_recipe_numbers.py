"""The number type a param declares (DESIGN 4.8): a float param holds
an integer as a float, an integer param refuses a float, and one value
written two ways is one IR."""

import json
import re

import pytest

from conftest import IR_FIXTURES, SRC_ROOT
from moraine3d import ops
from moraine3d.core import asserts as vocab
from moraine3d.core import numbers
from moraine3d.core import asset as asset_mod
from moraine3d.errors import MoraineError
from support.engine import Evaluation
from support.fixtures import realize
from support.recipes import build_recipe, ir, refusal

EXAMPLES = IR_FIXTURES
PROJECTS = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())

BOX = 'b = mo.box(id = "b", size = [1, 1, 1])\n'
OUT = 'mo.output("part", %s)\nmo.waive_asset("grounded")\n'


# -- the declarations ----------------------------------------------------------


def test_every_param_declares_its_number_type():
    """DESIGN 4.8: what a param holds is the param's to say. Every
    operator param and every assert key is declared under one of the
    three kinds, and the accepted names are the declared ones."""
    for name in ops.names():
        module = ops.get(name)
        assert module.PARAMS == frozenset(module.PARAM_KINDS), name
        assert set(module.PARAM_KINDS.values()) <= set(numbers.KINDS), name
    for kind in vocab.AUTHORABLE_KINDS:
        module = vocab.KINDS[kind]
        assert module.KEYS == frozenset(module.KEY_KINDS), kind
        assert set(module.KEY_KINDS.values()) <= set(numbers.KINDS), kind


# The shared readers of a number param, by the type each reads.
_READERS = {
    "num": numbers.FLOAT, "vec3": numbers.FLOAT, "rgb": numbers.FLOAT,
    "integer": numbers.INT,
}
_READ = re.compile(
    r"common\.(num|vec3|rgb|integer)\(\s*(?:op|\"\w+\"),\s*params,\s*\"(\w+)\""
)
# The readers that name their param by default.
_DEFAULT_NAMES = {"seed": "seed", "texture_size": "size"}
_READ_DEFAULT = re.compile(r"common\.(seed|texture_size)\(\s*(?:op|\"\w+\"),\s*params\s*\)")


def test_a_declaration_is_the_type_the_operator_reads():
    """DESIGN 4.8: the build writes a number in its param's type, the
    conversion the operator makes when it reads the param. A
    declaration that names another type than the reader's would change
    what the operator computes from."""
    read = 0
    for name in ops.names():
        module = ops.get(name)
        source = (SRC_ROOT / "ops" / f"{name}.py").read_text(encoding="utf-8")
        for reader, param in _READ.findall(source):
            assert module.PARAM_KINDS[param] == _READERS[reader], (name, param)
            read += 1
        for reader in _READ_DEFAULT.findall(source):
            assert module.PARAM_KINDS[_DEFAULT_NAMES[reader]] == numbers.INT, (name, reader)
            read += 1
    assert read > 60  # the scan reads the operators, not an empty match


# -- the corpus ----------------------------------------------------------------


def _respelled(doc: dict, kind: str, spell) -> tuple[dict, set]:
    """``doc`` with every number of a ``kind`` param respelled, and the
    (op, param) pairs a respelling reached."""
    reached = set()

    def walk(value):
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return spell(value)

    doc = json.loads(json.dumps(doc))
    for node in doc["nodes"]:
        kinds = ops.get(node["op"]).PARAM_KINDS
        for param, value in node["params"].items():
            if kinds[param] != kind:
                continue
            new = walk(value)
            if json.dumps(new) != json.dumps(value):
                node["params"][param] = new
                reached.add((node["op"], param))
    return doc, reached


def _as_integer(v):
    whole = isinstance(v, float) and v.is_integer() and (v != 0 or str(v) == "0.0")
    return int(v) if whole else v


def _as_float(v):
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v


def _hashes(root) -> dict:
    return {
        name: rec.result_hash
        for name, rec in Evaluation(asset_mod.load(root)).records().items()
    }


def _rewrite(root, kind, spell) -> set:
    """Respell every IR of a realized project in place."""
    reached = set()
    for path in sorted(root.parent.rglob("*.mora")):
        doc, hit = _respelled(json.loads(path.read_text(encoding="utf-8")), kind, spell)
        reached |= hit
        if hit:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(doc))
    return reached


@pytest.mark.parametrize("fixture", PROJECTS, ids=lambda p: p.name)
def test_an_operator_reads_an_integer_as_the_float_it_is(fixture, tmp_path):
    """DESIGN 4.8: a float param takes an integer and holds it as a
    float, the value the operator computes from either way. Every whole
    number of a float param in the corpus, written as an integer,
    evaluates to the same result."""
    as_written = _hashes(realize(fixture, tmp_path / "written" / fixture.name))
    root = realize(fixture, tmp_path / "respelled" / fixture.name)
    if not _rewrite(root, numbers.FLOAT, _as_integer):
        pytest.skip("no whole number in a float param")
    assert _hashes(root) == as_written


def test_an_operator_refuses_the_float_the_build_refuses(tmp_path):
    """DESIGN 4.8: an integer param refuses a float, whatever its
    value. Every integer param the corpus sets, given the same value
    as a float, fails its operator."""
    seen = set()
    for fixture in PROJECTS:
        source = json.loads(
            sorted(fixture.glob("*.mora"))[0].read_text(encoding="utf-8")
        )
        for index, node in enumerate(source["nodes"]):
            kinds = ops.get(node["op"]).PARAM_KINDS
            for param in node["params"]:
                pair = (node["op"], param)
                if kinds[param] != numbers.INT or pair in seen:
                    continue
                seen.add(pair)
                root = realize(fixture, tmp_path / f"{len(seen)}" / fixture.name)
                doc = json.loads(root.read_text(encoding="utf-8"))
                target = doc["nodes"][index]["params"]
                target[param] = json.loads(
                    json.dumps(target[param]), parse_int=float,
                )
                with open(root, "w", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(doc))
                with pytest.raises(MoraineError) as e:
                    ev = Evaluation(asset_mod.load(root))
                    ev.records()
                assert e.value.code in ("bad_param", "bad_params"), pair
    assert len(seen) > 15


def test_the_reference_lists_each_number_params_type(capsys):
    """DESIGN 4.8: `moraine docs <op>` lists each number param's type
    from the declaration the build reads, so the reference cannot lag
    the build."""
    from moraine3d.app.cli import main

    assert main(["docs", "array"]) == 0
    assert (
        "numbers: count int, step_translate float, step_rotate_deg float, "
        "step_scale float"
    ) in capsys.readouterr().out.splitlines()
    assert main(["docs", "boolean"]) == 0
    assert "numbers: " not in capsys.readouterr().out
    assert main(["docs", "knobs"]) == 0
    assert "numbers: " not in capsys.readouterr().out


# -- the build -----------------------------------------------------------------


def test_one_value_written_two_ways_is_one_ir(tmp_path):
    """DESIGN 4.8: one value written two ways is one IR. A quotient and
    the integer it equals build the same node, so the cache key and
    every diff read them as one."""
    divided = build_recipe(
        tmp_path / "a",
        BOX + 'r = mo.transform(b, id = "r", rotate_deg = [0, 360 / 4, 0])\n' + OUT % "r",
    )
    written = build_recipe(
        tmp_path / "b",
        BOX + 'r = mo.transform(b, id = "r", rotate_deg = [0, 90, 0])\n' + OUT % "r",
    )
    assert divided.target.text == written.target.text
    assert ir(written)["nodes"][1]["params"] == {"rotate_deg": [0.0, 90.0, 0.0]}
    assert '"rotate_deg": [0.0, 90.0, 0.0]' in written.target.text


def test_a_float_param_holds_numbers_inside_a_structure_as_floats(tmp_path):
    """DESIGN 4.8: numbers inside a list or a dict take the param's
    type, and what is not a number is left as it is."""
    result = build_recipe(
        tmp_path,
        BOX
        + 't = mo.tag(b, id = "t", name = "top", select = {"normal_y": {"gt": 1}})\n'
        + 'e = mo.extrude(t, id = "e", distance = 1, direction = "normal")\n'
        + OUT % "e",
    )
    nodes = {n["id"]: n["params"] for n in ir(result)["nodes"]}
    assert json.dumps(nodes["t"]["select"]) == '{"normal_y": {"gt": 1.0}}'
    assert json.dumps(nodes["e"]) == '{"distance": 1.0, "direction": "normal"}'


@pytest.mark.parametrize("count", ["12 / 4", "3.0", "12.0 // 4"])
def test_an_integer_param_refuses_a_float(tmp_path, count):
    """DESIGN 4.8: an integer param refuses a float whatever its value,
    and the refusal names what to write."""
    e = refusal(
        tmp_path,
        BOX + f'r = mo.array(b, id = "r", count = {count})\n' + OUT % "r",
    )
    assert e.code == "bad_param"
    assert e.detail["site"] == "asset.star:2"
    assert e.detail["param"] == "count"
    assert "int(x)" in e.message and "12.0 // 3 is 4.0" in e.message


@pytest.mark.parametrize("count", ["12 // 4", "int(12 / 4)", "3"])
def test_an_integer_param_takes_an_integer(tmp_path, count):
    """DESIGN 4.8: `//` between integers and `int(x)` give the integer
    an integer param takes."""
    result = build_recipe(
        tmp_path,
        BOX + f'r = mo.array(b, id = "r", count = {count})\n' + OUT % "r",
    )
    assert json.dumps(ir(result)["nodes"][1]["params"]) == '{"count": 3}'


def test_an_assert_holds_a_length_as_a_float_and_a_count_as_an_integer(tmp_path):
    """DESIGN 4.8: an assert's keys declare their types as an
    operator's params do. A length is a float, a count of copies an
    integer."""
    recipe = (
        BOX
        + 'r = mo.array(b, id = "r", count = 3, step_translate = [2, 0, 0])\n'
        + OUT % "r"
        + 'mo.assert_extent(id = "wide", outputs = ["part"], measure = "size_x", max = 6)\n'
        + 'mo.assert_count(id = "three", outputs = ["part"], value = 3)\n'
    )
    asserts = {a["id"]: a for a in ir(build_recipe(tmp_path / "a", recipe))["asserts"]}
    assert json.dumps(asserts["wide"]["max"]) == "6.0"
    assert json.dumps(asserts["three"]["value"]) == "3"
    e = refusal(tmp_path / "b", recipe.replace("value = 3", "value = 6 / 2"))
    assert e.code == "bad_assert"
    assert e.detail["site"] == "asset.star:6"
    assert "int(x)" in e.message


def test_a_knob_keeps_the_type_its_default_has(tmp_path):
    """DESIGN 4.6: a knob's number type is its declaration's. The build
    writes the knob as declared and the param it reaches in the param's
    own type, and the recipe reads back the value it declared."""
    result = build_recipe(
        tmp_path,
        'n = mo.param("n", 2, min = 1, max = 4)\n'
        + 'w = mo.param("w", 2.0, min = 1.0, max = 4.0)\n'
        + 'b = mo.box(id = "b", size = [n, w, 1])\n'
        + 'r = mo.array(b, id = "r", count = n)\n'
        + OUT % "r",
    )
    doc = ir(result)
    knobs = {k["id"]: k for k in doc["params"]}
    assert json.dumps(knobs["n"]["value"]) == "2"
    assert json.dumps(knobs["w"]["value"]) == "2.0"
    assert json.dumps(doc["nodes"][0]["params"]) == '{"size": [2.0, 2.0, 1.0]}'
    assert json.dumps(doc["nodes"][1]["params"]) == '{"count": 2}'
