"""The builtins (DESIGN 4.8): `mo` is generated from the registries, so
the recipe's vocabulary and the IR's are one; ids derive from the
recipe's names; declarations write the IR's root keys."""

import re

import pytest

from moraine3d import ops
from moraine3d.core import asserts as vocab
from moraine3d.core import asset as asset_mod
from moraine3d.recipe import dialect, prelude
from moraine3d.recipe.build import build
from moraine3d.recipe.version import build_version
from support.recipes import build_recipe, ir, refusal, write_files

pytestmark = pytest.mark.real_stamp

BOX = 'mo.box(id = "a", size = [1, 1, 1])'


# -- the vocabulary ----------------------------------------------------------


def test_every_operator_and_assert_kind_is_a_builtin_and_nothing_else():
    """DESIGN 4.8: `mo.<op>` for every registered op, bare names
    identical to the IR's `"op"` strings, and `mo.assert_<kind>`, one
    function per registered kind. Beside them the declarations, and in
    a requirements file `mo.param_value`."""
    names = prelude.names()
    assert len(names) == len(set(names))
    expected = (
        set(ops.names())
        | {f"assert_{k}" for k in vocab.AUTHORABLE_KINDS}
        | set(prelude.DECLARATIONS)
        | {"param_value"}
    )
    assert set(names) == expected
    for name in names:
        assert f"{name} = _{name}" in prelude.source()


def test_every_root_key_has_its_declaration():
    """DESIGN 4.8: declarations, one per IR root key."""
    written_by_calls = {"build_version", "nodes", "asserts"}
    assert set(prelude.DECLARATIONS.values()) | written_by_calls == set(
        asset_mod.ROOT_KEY_ORDER
    )


def test_no_operator_declares_a_param_named_id():
    """DESIGN 4.8: the node's id is the keyword `id`, and the registry
    test that keeps the builtins equal to the IR also refuses an
    operator declaring a param called `id`."""
    for name in ops.names():
        assert "id" not in ops.get(name).PARAMS, name


def test_an_input_name_is_no_param_and_covers_the_arity():
    """DESIGN 4.8: an input's name is a keyword of the builtin, so the
    registry test refuses one that is also a param's. The names cover
    every input the op takes, and an op whose optional inputs go by
    keyword declares the param the build writes their slots to."""
    for name in ops.names():
        module = ops.get(name)
        names = getattr(module, "INPUT_NAMES", ())
        slotted = getattr(module, "SLOT_INPUTS", {})
        keywords = list(names) + list(slotted)
        assert len(set(keywords)) == len(keywords), name
        assert not set(keywords) & (set(module.PARAMS) | {"id"}), name
        lo, hi = module.INPUT_ARITY
        if names:
            assert len(names) == hi, name
        if slotted:
            assert hi == lo + len(slotted), name
            assert module.SLOT_PARAM in module.PARAMS, name
    assert ops.get("place").INPUT_NAMES == ("base", "part")
    assert ops.get("mate").INPUT_NAMES == ("base", "part", "leader")
    assert ops.get("scatter").INPUT_NAMES == ("target", "prototype")
    assert list(ops.get("material").SLOT_INPUTS) == [
        "base_color_texture", "roughness_texture", "metallic_texture", "normal_texture",
    ]


def test_no_builtin_name_collides_with_another_family():
    """DESIGN 4.8: there is one spelling; an operator named like a
    declaration would make two."""
    assert not set(ops.names()) & (set(prelude.DECLARATIONS) | set(prelude.HANDLES))
    assert not any(n.startswith(prelude.ASSERT_PREFIX) for n in ops.names())


def test_every_operator_name_is_in_the_tools_grammar():
    """DESIGN 4.8: an operator's name matches `[a-z][a-z0-9_]*`, so
    whatever a workspace comes to define is spelled outside it."""
    grammar = re.compile(r"[a-z][a-z0-9_]*")
    for name in ops.names():
        assert grammar.fullmatch(name), name


def test_the_stamp_signs_the_vocabulary(monkeypatch):
    """DESIGN 4.4: the build's contract is versioned by the builtins'
    signatures, so a builtin added moves `build_version`."""
    before = build_version()
    try:
        with monkeypatch.context() as patched:
            patched.setattr(
                prelude, "DECLARATIONS", {**prelude.DECLARATIONS, "extra": "extra"},
            )
            build_version.cache_clear()
            assert build_version() != before
    finally:
        build_version.cache_clear()
    assert build_version() == before


def test_math_is_correctly_rounded(tmp_path):
    """DESIGN 4.4: the transcendental `math` builtins are computed
    beyond the double and rounded once to the nearest double, so no
    platform's library shows in the IR."""
    import math

    import mpmath

    result = build_recipe(tmp_path, '''vals = [math.tan(0.7), math.atan2(1.0, 3.0), math.pow(1.1, 7.3), math.hypot(3.0, 4.1)]
for i, v in enumerate(vals):
    mo.output("b%d" % i, mo.box(id = "b%d" % i, size = [1, v, 1]))
''')
    built = [n["params"]["size"][1] for n in ir(result)["nodes"] if n["op"] == "box"]
    with mpmath.workdps(60):
        exact = [mpmath.tan(mpmath.mpf(0.7)), mpmath.atan2(1, 3),
                 mpmath.power(mpmath.mpf(1.1), mpmath.mpf(7.3)),
                 mpmath.sqrt(mpmath.mpf(9) + mpmath.mpf(4.1) ** 2)]
        assert built == [float(x) for x in exact]
    assert dialect.TRANSCENDENTAL <= set(dialect.MATH_FUNCTIONS) <= set(dir(math))


def test_math_rounding_is_the_same_under_concurrent_builds(tmp_path):
    """DESIGN 4.4: identical inputs produce byte-identical IR. The
    transcendental builtins are computed beyond the double in a
    context of the thread's own, so neither a build on another thread
    nor the precision another thread sets and restores on the library's
    shared default context reaches the IR."""
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mpmath

    recipe = write_files(tmp_path, {"asset.star": '''for i in range(400):
    v = math.hypot(4.24519189142514, 8.26852124672038)
    mo.output("b%d" % i, mo.box(id = "b%d" % i, size = [1, v, 1]))
'''}) / "asset.star"
    stop = threading.Event()

    def toggle():
        while not stop.is_set():
            with mpmath.workdps(5):
                pass

    def sizes(_):
        nodes = ir(build(recipe, write=False))["nodes"]
        return {n["params"]["size"][1] for n in nodes if n["op"] == "box"}

    interval = sys.getswitchinterval()
    other = threading.Thread(target=toggle)
    try:
        sys.setswitchinterval(1e-6)
        other.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            built = list(pool.map(sizes, range(2)))
    finally:
        stop.set()
        if other.is_alive():
            other.join()
        sys.setswitchinterval(interval)
    assert built == [{9.294627362217817}] * 2


def test_a_math_domain_error_is_refused_at_its_line(tmp_path):
    """DESIGN 4.4: a value the IR cannot hold stops the build at the
    line that made it, a domain error like a non-finite result."""
    err = refusal(tmp_path, "x = math.asin(2.0)\n")
    assert err.detail["site"] == "asset.star:1"


MATH_DERIVED = '''vals = [
    math.sin(math.pi / 6),
    math.cos(1.0),
    math.tan(0.7),
    math.asin(0.3),
    math.acos(0.3),
    math.atan(2.0),
    math.atan2(1.0, 3.0),
    math.sqrt(2.0),
    math.hypot(3.0, 4.1),
    math.radians(37.0),
    math.degrees(1.0),
    math.exp(1.0),
    math.log(10.0),
    math.pow(1.1, 7.3),
    math.fmod(7.5, 2.2),
]
for i, v in enumerate(vals):
    mo.output("b%d" % i, mo.box(id = "b%d" % i, size = [1, v, 1]))
'''

# What the recipe above derived on the platform that froze it, bit for bit.
MATH_DERIVED_VALUES = [
    0.49999999999999994,
    0.5403023058681398,
    0.8422883804630794,
    0.3046926540153975,
    1.2661036727794992,
    1.1071487177940904,
    0.3217505543966422,
    1.4142135623730951,
    5.080354318352215,
    0.6457718232379019,
    57.29577951308232,
    2.718281828459045,
    2.302585092994046,
    2.0052411193823754,
    0.8999999999999995,
]


def test_the_values_a_recipe_derives_through_math_are_frozen(tmp_path):
    """DESIGN 4.4: identical inputs produce byte-identical IR on every
    platform, the transcendental functions computed beyond the double
    and rounded once. The values are frozen here, so the suite passing
    on both platforms is the comparison."""
    nodes = ir(build_recipe(tmp_path, MATH_DERIVED))["nodes"]
    derived = [n["params"]["size"][1] for n in nodes if n["op"] == "box"]
    assert derived == MATH_DERIVED_VALUES


def test_a_math_result_that_is_not_finite_is_refused(tmp_path):
    """DESIGN 4.4: identical inputs produce byte-identical IR; a value
    the IR cannot hold stops the build at the line that made it."""
    err = refusal(tmp_path, "x = math.exp(1000.0)\n")
    assert err.detail["site"] == "asset.star:1"


# -- operator calls ------------------------------------------------------------


def test_a_call_site_reads_as_its_ir_line(tmp_path):
    """DESIGN 4.8: inputs as node handles, params by keyword with the
    IR's names."""
    result = build_recipe(tmp_path, '''a = mo.box(id = "a", size = [1, 2, 3], segments = [1, 1, 1])
b = mo.transform(a, id = "b", translate = [0, 1, 0])
mo.output("b", b)
''')
    assert ir(result)["nodes"] == [
        {"id": "a", "op": "box", "params": {"size": [1, 2, 3], "segments": [1, 1, 1]},
         "site": "asset.star:1"},
        {"id": "b", "op": "transform", "inputs": ["a"],
         "params": {"translate": [0, 1, 0]}, "site": "asset.star:2"},
    ]


def test_a_list_of_inputs_is_expanded_after_the_keywords(tmp_path):
    """DESIGN 4.8: inputs held in a list are expanded after the
    keywords, the language's own order."""
    result = build_recipe(tmp_path, '''treads = [mo.box(id = "t%d" % i, size = [1, 0.1, 0.3]) for i in range(3)]
base = mo.box(id = "base", size = [1, 1, 1])
stair = mo.boolean(base, id = "stair", mode = "union", *treads)
mo.output("stair", stair)
''')
    assert ir(result)["nodes"][-1]["inputs"] == ["base", "t0", "t1", "t2"]


_PLACED = '''slab = mo.tag(mo.box(id = "s", size = [2, 0.1, 2]), id = "slab", name = "top", select = {"normal_y": {"gt": 0.9}})
crate = mo.box(id = "crate", size = [0.4, 0.4, 0.4])
%s
mo.output("p", p)
'''


def test_an_input_is_passed_by_position_or_by_its_name(tmp_path):
    """DESIGN 4.8: where an operator's inputs play different roles the
    operator names them, and a recipe may pass such an input by its
    name. The IR keeps the positions."""
    built = [
        ir(build_recipe(tmp_path / str(i), _PLACED % call))["nodes"][-1]["inputs"]
        for i, call in enumerate((
            'p = mo.place(slab, crate, id = "p", anchor = "top")',
            'p = mo.place(base = slab, part = crate, id = "p", anchor = "top")',
            'p = mo.place(part = crate, base = slab, id = "p", anchor = "top")',
            'p = mo.place(slab, part = crate, id = "p", anchor = "top")',
        ))
    ]
    assert built == [["slab", "crate"]] * 4


def test_a_named_input_is_refused_where_it_cannot_be_placed(tmp_path):
    """DESIGN 4.8, 8.2: an input given twice, an input left out before
    a named one, and a value that is no node each fail the build at
    the call, naming the input by its role."""
    for i, (call, needle) in enumerate((
        ('p = mo.place(slab, crate, base = slab, id = "p", anchor = "top")',
         "'base' by position and by keyword"),
        ('p = mo.place(part = crate, id = "p", anchor = "top")',
         "missing its input 'base'"),
        ('p = mo.place(base = slab, part = 0.4, id = "p", anchor = "top")',
         "input 'part' is a float"),
    )):
        err = refusal(tmp_path / str(i), _PLACED % call)
        assert err.code == "bad_input" and needle in err.message, call
        assert err.detail["site"] == "asset.star:3"


_TEXTURED = '''box = mo.unwrap(mo.box(id = "b", size = [1, 1, 1]), id = "uv")
rough = mo.noise_texture(id = "rough", size = 16)
metal = mo.noise_texture(id = "metal", size = 16)
%s
mo.output("m", m)
'''


def test_a_texture_is_named_by_the_slot_it_fills(tmp_path):
    """DESIGN 4.8: an input an operator may take or leave is passed by
    keyword alone, the keyword naming the slot it fills, and the build
    writes the filled slots beside the inputs, in the operator's order
    whatever order the call wrote."""
    node = ir(build_recipe(tmp_path / "a", _TEXTURED % (
        'm = mo.material(box, metallic_texture = metal, roughness_texture = rough, '
        'id = "m", name = "steel")'
    )))["nodes"][-1]
    assert node["inputs"] == ["uv", "rough", "metal"]
    assert node["params"] == {"name": "steel", "slots": ["roughness", "metallic"]}
    bare = ir(build_recipe(tmp_path / "b", _TEXTURED % (
        'm = mo.material(box, id = "m", name = "steel")'
    )))["nodes"][-1]
    assert bare["inputs"] == ["uv"] and "slots" not in bare["params"]


def test_a_texture_by_position_or_a_written_slots_names_the_keywords(tmp_path):
    """DESIGN 4.8: textures by position beside a list of their slots
    is not the recipe's spelling; the refusal names the keywords."""
    for i, call in enumerate((
        'm = mo.material(box, rough, id = "m", name = "steel", slots = ["roughness"])',
        'm = mo.material(box, rough, id = "m", name = "steel")',
        'm = mo.material(box, id = "m", name = "steel", slots = [])',
    )):
        err = refusal(tmp_path / str(i), _TEXTURED % call)
        assert err.code == "bad_input" and "roughness_texture = <node>" in err.message, call


def test_a_keyword_after_an_expansion_names_the_order_to_write(tmp_path):
    """DESIGN 4.8: a call that names an argument after a `*` expansion
    fails with the order to write, `f(id = ..., *xs)`."""
    err = refusal(tmp_path, '''xs = [mo.box(id = "a", size = [1, 1, 1])]
mo.boolean(*xs, id = "u", mode = "union")
''')
    assert err.code == "argument_order"
    assert "f(id = ..., *xs)" in err.message
    assert err.detail["site"] == "asset.star:2"


def test_no_builtin_takes_a_list_in_place_of_its_inputs(tmp_path):
    """DESIGN 4.8: no builtin takes a list in place of its inputs."""
    err = refusal(tmp_path, '''a = mo.box(id = "a", size = [1, 1, 1])
mo.boolean([a, a], id = "u", mode = "union")
''')
    assert err.code == "bad_input"
    assert "*nodes" in err.message


@pytest.mark.parametrize("call,code", [
    ('mo.box(id = "a", sise = [1, 1, 1])', "unknown_param"),
    ('mo.transform(id = "a", translate = [0, 0, 0])', "bad_inputs"),
    ('mo.box(mo.box(id = "b", size = [1, 1, 1]), id = "a")', "bad_inputs"),
    ('mo.box(id = "a", size = mo.box(id = "b", size = [1, 1, 1]))', "bad_value"),
    ('mo.box(id = "a", size = [1, None, 1])', "bad_value"),
    ('mo.box(id = "a", size = [1, 1e308 * 10, 1])', "bad_value"),
])
def test_a_call_the_ir_cannot_hold_is_refused_at_its_line(tmp_path, call, code):
    """DESIGN 4.8: params by keyword with the IR's names and
    validation; the error names the recipe line (DESIGN 8.2)."""
    err = refusal(tmp_path, f"x = 1\n{call}\n")
    assert err.code == code
    assert err.detail["site"] == "asset.star:2"


def test_a_value_that_lost_its_finiteness_says_so(tmp_path):
    """DESIGN 8.2: errors state what failed and the shortest path to a
    fix; a non-finite number reaches the host as None."""
    err = refusal(tmp_path, 'mo.box(id = "a", size = [1, 1e308 * 10, 1])\n')
    assert "not a finite number" in err.message


# -- ids -----------------------------------------------------------------------


def test_a_named_node_takes_its_name_and_a_function_names_under_its_id(tmp_path):
    """DESIGN 4.8: a function that creates nodes takes `id` as its first
    argument and creates them under `<id>/`."""
    result = build_recipe(tmp_path, '''def pipe(id, length):
    barrel = mo.cylinder(id = id + "/barrel", radius = 0.1, height = length)
    return mo.transform(barrel, id = id, translate = [0, length / 2, 0])

for i in range(2):
    mo.output("pipe_%d" % i, pipe("pipe_left_%d" % i, 1.0 + i))
''')
    assert [n["id"] for n in ir(result)["nodes"]] == [
        "pipe_left_0/barrel", "pipe_left_0", "pipe_left_1/barrel", "pipe_left_1",
    ]


def test_an_unnamed_node_at_the_root_counts_its_op_in_program_order(tmp_path):
    """DESIGN 4.8: a node left unnamed gets its op and ordinal, counting
    that op within its scope in program order; the root scope is empty,
    so the id carries no leading separator."""
    result = build_recipe(tmp_path, '''a = mo.box(size = [1, 1, 1])
b = mo.transform(a, translate = [0, 1, 0])
c = mo.box(size = [2, 2, 2])
d = [mo.transform(c, translate = [i, 0, 0]) for i in range(2)]
mo.output("top", mo.boolean(b, id = "top", mode = "union", *d))
''')
    assert [n["id"] for n in ir(result)["nodes"]] == [
        "box1", "transform1", "box2", "transform2", "transform3", "top",
    ]


def test_an_unnamed_assert_at_the_root_counts_its_kind(tmp_path):
    """DESIGN 4.8: assert ids derive the same way."""
    result = build_recipe(tmp_path, f'''mo.output("a", {BOX})
mo.output("b", mo.box(id = "b", size = [1, 1, 1]))
mo.assert_contact(outputs = ["a", "b"])
mo.assert_clearance(outputs = ["a", "b"], min = 0.1)
mo.assert_contact(outputs = ["a", "b"], tol = 0.001)
''')
    assert [a["id"] for a in ir(result)["asserts"]] == ["contact1", "clearance1", "contact2"]


@pytest.mark.parametrize("body", [
    "    return mo.box(size = [1, 1, 1])",
    "    return (lambda: mo.box(size = [1, 1, 1]))()",
])
def test_inside_a_function_every_node_is_named(tmp_path, body):
    """DESIGN 4.8: inside a function every node and assert is named,
    and a call that leaves one unnamed fails with the spelling to
    write."""
    err = refusal(tmp_path, f"def part(id):\n{body}\npart(\"p\")\n")
    assert err.code == "id_required"
    assert 'id = id + "/<name>"' in err.message
    assert err.detail["site"] == "asset.star:2"


def test_inside_a_function_every_assert_is_named(tmp_path):
    """DESIGN 4.8: inside a function every node and assert is named."""
    err = refusal(tmp_path, f'''mo.output("a", {BOX})
mo.output("b", mo.box(id = "b", size = [1, 1, 1]))
def joined(x, y):
    mo.assert_contact(outputs = [x, y])
joined("a", "b")
''')
    assert err.code == "id_required"
    assert err.detail["site"] == "asset.star:4"


def test_an_op_that_salts_its_streams_with_its_id_is_named(tmp_path):
    """DESIGN 4.8: a node that anything can point at carries an `id`:
    every op that salts its streams with its id."""
    err = refusal(tmp_path, '''proto = mo.box(id = "proto", size = [0.1, 0.1, 0.1])
ground = mo.grid(id = "ground", size = [4, 4])
mo.scatter(ground, proto, count = 5, seed = 1)
''')
    assert err.code == "id_required"
    assert err.detail["site"] == "asset.star:3"


def test_an_outputs_node_is_named(tmp_path):
    """DESIGN 4.8: a node that anything can point at carries an `id`:
    every output's node."""
    err = refusal(tmp_path, 'mo.output("a", mo.box(size = [1, 1, 1]))\n')
    assert err.code == "id_required"
    assert "box1" in err.message


def test_a_duplicate_id_fails_naming_both_sites(tmp_path):
    """DESIGN 4.8: ids are unique within their scope, and a duplicate
    fails the build."""
    err = refusal(tmp_path, f"{BOX}\nx = 1\n{BOX}\n")
    assert err.code == "duplicate_id"
    assert err.detail == {"site": "asset.star:3", "first": "asset.star:1"}


@pytest.mark.parametrize(
    "bad", ['"/a"', '"a/"', '"a//b"', '""', "3", '"9bad"', '"a b"', '"a/b.c"'],
)
def test_an_id_is_a_path_of_names(tmp_path, bad):
    """DESIGN 4.8: the scope separator is `/`, the root scope is empty,
    and each segment is a portable name, as outputs and params are."""
    err = refusal(tmp_path, f"mo.box(id = {bad}, size = [1, 1, 1])\n")
    assert err.code == "bad_id"


def test_ids_are_case_sensitive(tmp_path):
    """DESIGN 4.8: ids are case-sensitive."""
    result = build_recipe(tmp_path, '''a = mo.box(id = "Post", size = [1, 1, 1])
b = mo.box(id = "post", size = [1, 1, 1])
mo.output("a", a)
mo.output("b", b)
''')
    assert [n["id"] for n in ir(result)["nodes"]] == ["Post", "post"]


# -- declarations ------------------------------------------------------------


def test_declarations_write_the_root_keys(tmp_path):
    """DESIGN 4.8: `mo.output`, `mo.group`, `mo.waive`,
    `mo.waive_asset`, and `mo.vendored`, one per IR root key."""
    result = build_recipe(tmp_path, '''a = mo.box(id = "a", size = [1, 1, 1])
b = mo.transform(a, id = "b", translate = [0, -2, 0])
mo.output("frame", a)
mo.output("footing", b)
mo.group("shelf", ["frame", "footing"])
mo.waive("footing", "not_sunk")
mo.waive_asset("grounded")
mo.vendored("kits/shelf.star", "0123456789abcdef01234567")
''')
    doc = ir(result)
    assert doc["outputs"] == {"frame": "a", "footing": "b"}
    assert doc["groups"] == {"shelf": ["frame", "footing"]}
    assert doc["checks"] == {"footing": {"not_sunk": False}}
    assert doc["asset_checks"] == {"grounded": False}
    assert doc["provenance"] == {
        "source": "kits/shelf.star", "content_hash": "0123456789abcdef01234567",
    }
    assert list(doc) == [k for k in asset_mod.ROOT_KEY_ORDER if k in doc]


@pytest.mark.parametrize("line,code", [
    ('mo.output("9a", a)', "bad_output_name"),
    ('mo.output("a/b", a)', "bad_output_name"),
    ('mo.output("a", "a")', "bad_declaration"),
    ('mo.output("a", a)\nmo.output("A", a)', "output_name_collision"),
    ('mo.output("a", a)\nmo.output("a", a)', "duplicate_output"),
    ('mo.output("a", a)\nmo.group("g", ["a"])', "bad_group"),
    ('mo.output("a", a)\nmo.waive("a", "not_flat")', "unknown_check"),
    ('mo.output("a", a)\nmo.waive("b", "not_sunk")', "checks_unknown_output"),
    ('mo.output("a", a)\nmo.waive_asset("level")', "unknown_check"),
    ('mo.output("a", a, 3)', "bad_declaration"),
    ('mo.output(name = "a")', "bad_declaration"),
])
def test_a_declaration_is_judged_as_its_ir_entry_is(tmp_path, line, code):
    """DESIGN 4.8: the declarations carry the IR's validation, and the
    error names the recipe line."""
    err = refusal(tmp_path, f"a = {BOX}\n{line}\n")
    assert err.code == code
    assert err.detail["site"].startswith("asset.star:")


def test_a_asset_without_an_output_fails_the_build(tmp_path):
    """DESIGN 8.1: the finished project declares what ships; a recipe
    that declares no output has nothing to build."""
    err = refusal(tmp_path, f"{BOX}\n")
    assert err.code == "no_outputs"


def test_an_assert_over_an_undeclared_output_fails_the_build(tmp_path):
    """DESIGN 8.1: an assert over an output not yet declared fails the
    build (`unknown_output`): the red state the loop starts in."""
    err = refusal(tmp_path, f'''mo.assert_contact(id = "stands", outputs = ["post", "slab"])
mo.output("slab", {BOX})
''')
    assert err.code == "unknown_output"
    assert err.detail["output"] == "post"
    assert err.detail["site"] == "asset.star:1"


def test_names_resolve_after_every_declaration_is_in(tmp_path):
    """DESIGN 4.8: order inside a file matters to values but not to
    names; an assert's output names resolve after every declaration is
    in."""
    result = build_recipe(tmp_path, f'''mo.assert_extent(id = "wide", outputs = ["slab"], measure = "size_x", value = 1.0)
mo.output("slab", {BOX})
''')
    assert [a["id"] for a in ir(result)["asserts"]] == ["wide"]


@pytest.mark.parametrize("call,code", [
    ('mo.assert_contact(["a", "a"])', "bad_assert"),
    ('mo.assert_contact(id = "x", outputs = ["a", "b"], owner = "human")', "bad_assert"),
    ('mo.assert_contact(id = "x", outputs = ["a", "b"], min = 0.1)', "bad_assert"),
    ('mo.assert_contact(id = "no_overlap", outputs = ["a", "b"])', "bad_assert_id"),
    ('mo.assert_contact(id = "x y", outputs = ["a", "b"])', "bad_id"),
])
def test_an_assert_is_judged_as_its_ir_entry_is(tmp_path, call, code):
    """DESIGN 4.8: asserts name outputs by string, with the IR's
    argument names, and there is no `owner` argument."""
    err = refusal(tmp_path, f'''mo.output("a", {BOX})
mo.output("b", mo.box(id = "b", size = [1, 1, 1]))
{call}
''')
    assert err.code == code
    assert err.detail["site"] == "asset.star:3"


# -- the boundary --------------------------------------------------------------

_STACK = "Traceback (most recent call last):\n  * %s:1, in <module>\n      call()\n"


def _session():
    from moraine3d.recipe import host

    session = host.Session(
        asset_file="asset.star", requirements_file="asset.requirements.star",
    )
    session.roles["asset.requirements.star"] = host.REQUIREMENTS
    return session


def test_no_builtin_returns_an_evaluated_value():
    """DESIGN 4.8, Invariant 10: a recipe never observes an evaluated
    value, and no builtin returns one. An operator returns a handle, an
    assert and a declaration return nothing, and `mo.param` returns the
    value the recipe declared."""
    import starlark as sl

    from moraine3d.recipe import host

    session = _session()
    asset, human = _STACK % "asset.star", _STACK % "asset.requirements.star"
    seeds = [
        session.node("box", asset, id=f"seed_{i}", size=[1, 1, 1]) for i in range(4)
    ]
    for name in ops.names():
        module = ops.get(name)
        if getattr(module, "IS_REFERENCE", False):
            got = session.node(name, asset, "subassets/x.star", "out", id="ref")
        else:
            # The binding hands a host function the handle itself.
            inputs = [host.Handle(f"seed_{i}", True)
                      for i in range(module.INPUT_ARITY[0])]
            got = session.node(name, asset, *inputs, id=f"made_{name}")
        assert isinstance(got, sl.OpaquePythonObject), name
    assert seeds
    for kind in sorted(vocab.AUTHORABLE_KINDS):
        assert session.assertion(kind, human, id=f"a_{kind}", outputs=["x"]) is None, kind
    handle = host.Handle("seed_0", True)
    returned = {
        "output": session.declare("output", asset, "part", handle),
        "group": session.declare("group", asset, "pair", ["part", "other"]),
        "waive": session.declare("waive", asset, "part", "not_sunk"),
        "waive_asset": session.declare("waive_asset", asset, "grounded"),
        "vendored": session.declare("vendored", asset, "kit/part.star", "0" * 24),
        "budget": session.declare("budget", human, materials=2),
        "param": session.declare("param", asset, "w", 2.5, min=1.0, max=3.0),
    }
    assert set(returned) == set(prelude.DECLARATIONS)
    assert returned.pop("param") == 2.5
    assert set(returned.values()) == {None}


def test_a_handle_tells_a_recipe_nothing(tmp_path):
    """DESIGN 4.8: the call returns an immutable node handle, to the
    recipe a value it can pass on and nothing else. It shows no id, no
    attribute, and no text a recipe could compute from."""
    head = ('x = mo.box(id = "a", size = [1, 1, 1])\n'
            'def say(v):\n    fail(v)\n')
    for expr, shown in (("str(x)", "<opaque python object>"),
                        ("repr(x)", "<opaque python object>"),
                        ("dir(x)", "[]")):
        err = refusal(tmp_path / expr[:3], head + f"say({expr})\n")
        assert err.message.endswith(shown), expr
    for expr in ("x.id", "{x: 1}", "x + 1", "x[0]"):
        err = refusal(tmp_path / str(abs(hash(expr))), head + f"y = {expr}\n")
        assert err.code == "recipe_error", expr
