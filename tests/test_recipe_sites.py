"""Call sites (DESIGN 4.8): a builtin learns where it was called from
the interpreter's call stack, whose text the build parses under four
rules. These tests are the second rule: they freeze the shape of that
text at both layers the pin moves together, the interpreter that
writes it and the binding that carries it."""

import importlib.metadata
import tomllib

import pytest
import starlark as sl

from conftest import SRC_ROOT
from moraine3d.recipe import dialect, sites
from moraine3d.recipe.version import build_version
from support.recipes import build_recipe, ir, refusal

pytestmark = pytest.mark.real_stamp

PROBE = '''def _probe(*args, **kw):
    return _host(call_stack(strip_frames = 1))
probe = struct(here = _probe)
'''

RECIPE = '''load("probe", "probe")
probe.here()
def helper():
    probe.here()
helper()
for i in range(2):
    probe.here()
xs = [probe.here() for i in range(1)]
f = lambda: probe.here()
f()
probe.here(
    1,
)
'''


def _stacks() -> list[str]:
    seen: list[str] = []
    module = sl.Module()
    module.add_callable("_host", lambda stack: seen.append(stack))
    sl.eval(module, sl.parse("<prelude>", PROBE, dialect.dialect()), dialect.prelude_globals())
    frozen = module.freeze()
    sl.eval(
        sl.Module(), sl.parse("yard/yard.star", RECIPE, dialect.dialect()),
        dialect.recipe_globals(), sl.FileLoader(lambda name: frozen),
    )
    return seen


def test_the_interpreter_is_pinned_to_one_exact_version():
    """DESIGN 4.8: the interpreter distribution is pinned to one exact
    version, which is part of `build_version`."""
    pyproject = tomllib.loads(
        (SRC_ROOT.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    pins = [
        d for d in pyproject["project"]["dependencies"]
        if d.startswith(dialect.INTERPRETER)
    ]
    installed = importlib.metadata.version(dialect.INTERPRETER)
    assert pins == [f"{dialect.INTERPRETER}=={installed}"]
    assert f"{dialect.INTERPRETER}.{installed}" in build_version()


def test_the_call_stack_text_has_the_frozen_shape():
    """DESIGN 4.8: a test freezes the shape of the text, so a format
    change fails Moraine's own suite when the pin moves."""
    stacks = _stacks()
    assert stacks[0] == (
        "Traceback (most recent call last):\n"
        "  * yard/yard.star:2, in <module>\n"
        "      probe.here()\n"
    )
    assert stacks[1] == (
        "Traceback (most recent call last):\n"
        "  * yard/yard.star:5, in <module>\n"
        "      helper()\n"
        "  * yard/yard.star:4, in helper\n"
        "      probe.here()\n"
    )


def test_the_innermost_frame_is_the_call_in_every_position():
    """DESIGN 4.8: the build takes the innermost recipe frame as the
    site, so a node made inside a helper carries the helper's line."""
    found = [sites.frames(s)[-1] for s in _stacks()]
    assert [(f.line, f.function) for f in found] == [
        (2, "<module>"),   # top level
        (4, "helper"),     # inside a function
        (7, "<module>"),   # a loop body, each pass
        (7, "<module>"),
        (8, "<module>"),   # a comprehension at the top level
        (9, "lambda"),     # a lambda
        (11, "<module>"),  # a call over several lines: its first line
    ]
    assert all(f.file == "yard/yard.star" for f in found)


def test_the_binding_reports_an_error_as_text_with_a_span():
    """DESIGN 4.8: the binding carries an error as text alone, and the
    build reads the span and the call chain from it."""
    with pytest.raises(sl.StarlarkError) as e:
        sl.parse("yard/yard.star", "x = 1\nwhile x:\n    pass\n", dialect.dialect())
    text = str(e.value)
    assert e.value.args == (text,)
    assert text.startswith(
        "error: Parse error: cannot use reserved keyword `while`\n"
        " --> yard/yard.star:2:1\n"
    )
    assert sites.span(text) == "yard/yard.star:2"


def test_a_host_exception_reaches_the_build_as_text():
    """DESIGN 4.8: a host function's failure crosses the interpreter as
    text under the call chain, so the build keeps its own error and
    raises it in place of the text."""
    def boom():
        raise ValueError("kept by the host")

    module = sl.Module()
    module.add_callable("boom", boom)
    with pytest.raises(sl.StarlarkError) as e:
        sl.eval(
            module,
            sl.parse("yard/yard.star", "def f():\n    boom()\nf()\n", dialect.dialect()),
            dialect.recipe_globals(),
        )
    text = str(e.value)
    assert "error: ValueError: kept by the host\n" in text
    assert [f.site for f in sites.frames(text)] == ["yard/yard.star:3", "yard/yard.star:2"]
    assert e.value.__cause__ is None


def test_values_cross_the_binding_as_the_build_expects():
    """DESIGN 4.4: the interpreter's floats are Python floats; a value
    that is not a finite number arrives as None, which the build
    refuses by name."""
    got = []
    module = sl.Module()
    module.add_callable("take", lambda *a, **kw: got.append((a, kw)))
    sl.eval(
        module,
        sl.parse(
            "a.star",
            'take(0.1 + 0.2, -0.0, 1 << 70, (1, 2), {"k": [1.5]}, inf = 1e308 * 10)\n',
            dialect.dialect(),
        ),
        dialect.recipe_globals(),
    )
    (args, kw), = got
    assert args[0] == 0.30000000000000004
    assert str(args[1]) == "-0.0"
    assert args[2] == 1 << 70
    assert args[3] == [1, 2]
    assert args[4] == {"k": [1.5]}
    assert kw == {"inf": None}


def test_a_recipe_cannot_observe_its_call_site(tmp_path):
    """DESIGN 4.8: the extension is loaded for the prelude alone, so a
    recipe cannot observe where it was called from."""
    err = refusal(tmp_path, "x = call_stack()\n")
    assert err.code == "unknown_name"
    assert err.detail["site"] == "asset.star:1"


def test_every_node_and_assert_carries_its_site(tmp_path):
    """DESIGN 4.1: every node carries the recipe file and line that
    created it, and each assert the site of the call that declared
    it."""
    result = build_recipe(tmp_path, '''def leg(id, x):
    return mo.box(id = id, size = [0.1, 1.0, 0.1])

a = leg("a", 0)
b = mo.transform(
    a,
    id = "b",
    translate = [1, 0, 0],
)
mo.output("a", a)
mo.output("b", b)
mo.assert_clearance(id = "apart", outputs = ["a", "b"], min = 0.1)
''')
    doc = ir(result)
    assert {n["id"]: n["site"] for n in doc["nodes"]} == {
        "a": "asset.star:2", "b": "asset.star:5",
    }
    assert doc["asserts"][0]["site"] == "asset.star:12"


def test_sites_are_framed_from_the_workspace_root(tmp_path):
    """DESIGN 4.1: the site is workspace-relative."""
    from support.recipes import workspace
    from moraine3d.recipe.build import build

    root = workspace(tmp_path, {"yard": {
        "yard.star": 'load("lib/parts.star", "slab")\nmo.output("slab", slab("slab"))\n',
        "lib/parts.star": 'def slab(id):\n    return mo.box(id = id, size = [2, 0.1, 2])\n',
    }})
    doc = ir(build(root / "yard" / "yard.star"))
    assert doc["nodes"][0]["site"] == "yard/lib/parts.star:2"


def test_a_site_that_cannot_be_read_keeps_its_own_code(tmp_path, monkeypatch):
    """DESIGN 4.8: a node or an assert whose site cannot be read fails
    the build, and none is written without one. The failure is the
    build's own and says so, whatever the interpreter made of it on the
    way out."""
    from moraine3d.recipe import sites
    from support.recipes import refusal

    monkeypatch.setattr(sites, "frames", lambda stack: [])
    err = refusal(tmp_path, 'mo.output("a", mo.box(id = "a", size = [1, 1, 1]))\n')
    assert err.code == "site_unreadable"
    assert err.detail["builtin"] == "box"
    assert not (tmp_path / "build").exists()


# -- the calls on the way ------------------------------------------------------

RING = '''def post(id, x):
    return mo.box(id = id, size = [0.1, 1.0, x])

def ring(id, n):
    posts = [post("%s/post_%d" % (id, i), 1 + i) for i in range(n)]
    mo.assert_count(id = id + "/count", outputs = [id], value = n)
    return mo.boolean(id = id, mode = "union", *posts)
'''


def test_an_entry_made_in_a_function_carries_the_calls_on_the_way(tmp_path):
    """DESIGN 4.8: the site is the innermost frame, and an entry made
    inside a function also carries `via`, the sites of the outer
    frames, outermost first. An entry made at a file's top level has
    none."""
    result = build_recipe(tmp_path, 'load("lib/frame.star", "ring")\n'
        'legs = ring("legs", 2)\n'
        'rail = ring(\n    "rail", 2)\n'
        'mo.output("legs", legs)\nmo.output("rail", rail)\n'
        'mo.waive_asset("grounded")\n'
        'top = mo.box(id = "top", size = [1, 1, 1])\n',
        files={"lib/frame.star": RING})
    doc = ir(result)
    nodes = {n["id"]: n for n in doc["nodes"]}
    assert (nodes["legs"]["site"], nodes["legs"]["via"]) == (
        "lib/frame.star:7", ["asset.star:2"],
    )
    assert (nodes["rail"]["site"], nodes["rail"]["via"]) == (
        "lib/frame.star:7", ["asset.star:3"],
    )
    assert (nodes["legs/post_1"]["site"], nodes["legs/post_1"]["via"]) == (
        "lib/frame.star:2", ["asset.star:2", "lib/frame.star:5"],
    )
    assert "via" not in nodes["top"]
    [legs, rail] = doc["asserts"]
    assert (legs["site"], legs["via"]) == ("lib/frame.star:6", ["asset.star:2"])
    assert rail["via"] == ["asset.star:3"]
    assert list(nodes["legs"]) == ["id", "op", "inputs", "params", "site", "via"]


def test_a_knob_declared_in_a_function_carries_the_calls_on_the_way(tmp_path):
    """DESIGN 4.1: a knob's entry carries the site of the call that
    declared it, and with it the calls on the way."""
    doc = ir(build_recipe(tmp_path, '''def knob(id):
    return mo.param(id, 1.0, min = 0.5, max = 2.0)

w = knob("w")
mo.output("a", mo.box(id = "a", size = [w, 1, 1]))
mo.waive_asset("grounded")
'''))
    assert (doc["params"][0]["site"], doc["params"][0]["via"]) == (
        "asset.star:2", ["asset.star:4"],
    )


def test_a_long_run_of_calls_is_cut_in_the_middle(tmp_path):
    """DESIGN 4.8: a long chain keeps its two ends, the call in the
    asset and the calls nearest the entry."""
    doc = ir(build_recipe(tmp_path, '''def down(id, n):
    if n == 0:
        return mo.box(id = id, size = [1, 1, 1])
    return down(id, n - 1)

mo.output("a", down("a", 9))
mo.waive_asset("grounded")
'''))
    via = doc["nodes"][0]["via"]
    assert via == (
        ["asset.star:6"] + ["asset.star:4"] * 2 + ["... 4 more"] + ["asset.star:4"] * 3
    )


def test_the_calls_on_the_way_join_no_cache_key(tmp_path, monkeypatch):
    """DESIGN 4.1: location is not content. A node reached through
    another call has the key it had."""
    from support.engine import evaluate_records

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    lib = {"lib/frame.star": RING}
    one = build_recipe(tmp_path / "a", 'load("lib/frame.star", "ring")\n'
        'mo.output("legs", ring("legs", 2))\nmo.waive_asset("grounded")\n', files=lib)
    other = build_recipe(tmp_path / "b", 'load("lib/frame.star", "ring")\n'
        '\n\ndef outer():\n    return ring("legs", 2)\n'
        'mo.output("legs", outer())\nmo.waive_asset("grounded")\n', files=lib)
    assert ir(one)["nodes"][-1]["via"] != ir(other)["nodes"][-1]["via"]
    keys = [evaluate_records(r.target.asset)["legs"].key for r in (one, other)]
    assert keys[0] == keys[1]


def test_a_refusal_inside_a_function_names_the_calls_on_the_way(tmp_path):
    """DESIGN 6: the build's errors carry the recipe site, and for a
    call inside a function the calls it was reached through, so the
    line to fix is in the error whether it is the helper's or the
    caller's."""
    err = refusal(tmp_path, 'load("lib/frame.star", "ring")\n'
        'mo.output("legs", ring("legs", 2.5))\n', files={"lib/frame.star": RING.replace(
            "range(n)", "range(2)")})
    assert err.code == "bad_assert"
    assert (err.detail["site"], err.detail["via"]) == ("lib/frame.star:6", ["asset.star:2"])


def test_the_loader_reads_an_ir_with_or_without_the_calls(tmp_path):
    """DESIGN 4.1: the loader reads an IR with or without a position,
    and refuses calls that name no site to lead to."""
    import json

    from moraine3d.core import asset as asset_mod
    from moraine3d.errors import AssetError

    doc = {
        "build_version": "x",
        "nodes": [{"id": "a", "op": "box", "params": {}, "site": "s.star:2", "via": ["s.star:9"]}],
        "outputs": {"a": "a"},
    }
    loaded = asset_mod.loads(json.dumps(doc))
    assert loaded.nodes["a"].via == ["s.star:9"]
    assert '"site": "s.star:2", "via": ["s.star:9"]}' in asset_mod.dumps(loaded)
    del doc["nodes"][0]["via"]
    assert asset_mod.loads(json.dumps(doc)).nodes["a"].via is None
    for bad in ({"via": ["s.star:9"]}, {"site": "s.star:2", "via": []},
                {"site": "s.star:2", "via": "s.star:9"}):
        doc["nodes"][0] = {"id": "a", "op": "box", "params": {}, **bad}
        with pytest.raises(AssetError) as e:
            asset_mod.loads(json.dumps(doc))
        assert e.value.code == "bad_site"
