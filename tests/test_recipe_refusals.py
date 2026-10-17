"""Refusals (DESIGN 4.8): what an agent's Python prior must unlearn is
small and loud. A construct the language lacks fails the build with the
spelling to use instead, the actionable-error rule (DESIGN 8.2) applied
to a language. Each case freezes the interpreter's message the build
reads and the refusal it answers with."""

import pytest
import starlark as sl

from moraine3d.recipe import dialect, refusals
from support.recipes import build_recipe, refusal


def _interpreter(source: str, *, load: bool = True) -> str:
    """The interpreter's own text for ``source``."""
    with pytest.raises(sl.StarlarkError) as e:
        ast = sl.parse("asset.star", source, dialect.dialect(load=load))
        sl.eval_with(dialect.options(), sl.Module(), ast, dialect.recipe_globals())
    return str(e.value)


# Every bare name a Starlark dialect is known to offer, the ones this
# dialect withholds among them, so a name the pin starts to resolve is
# caught beside one it stops resolving.
_KNOWN_BARE_NAMES = """
abs all any bool breakpoint bytes call_stack chr debug dedupe depset dict dir
enum enumerate eval_type fail field filter float frozenset getattr glob
hasattr hash id input int isinstance iter json len list map max min next
None True False object open ord partial pow pprint prepr print pstr range
record repr reversed round select set slice sorted str struct sum tuple type
typing zip
""".split()
_RESOLVED = {
    "None", "True", "False", "abs", "all", "any", "bool", "chr", "dict", "dir",
    "enumerate", "fail", "float", "getattr", "hasattr", "hash", "int", "len",
    "list", "max", "min", "ord", "range", "repr", "reversed", "sorted", "str",
    "struct", "tuple", "type", "zip",
}


def test_the_bare_names_a_recipe_resolves_are_frozen():
    """DESIGN 4.8: the bare names a recipe resolves are frozen by a
    test, so a name the interpreter gains or drops with its pin fails
    Moraine's own suite, never a user's build."""
    resolved = set()
    for name in _KNOWN_BARE_NAMES:
        ast = sl.parse("asset.star", f"x = {name}\n", dialect.dialect())
        try:
            sl.eval_with(dialect.options(), sl.Module(), ast, dialect.recipe_globals())
        except sl.StarlarkError:
            continue
        resolved.add(name)
    assert resolved == _RESOLVED


KEYWORD_CASES = [
    ("while", "x = 0\nwhile x < 3:\n    x += 1\n", 2, "for i in range(n)"),
    ("class", "class A:\n    pass\n", 1, "struct("),
    ("try", "try:\n    x = 1\nexcept:\n    pass\n", 1, "the build reports"),
    ("import", "import math\n", 1, "load("),
    ("from", "from math import sin\n", 1, "load("),
    ("raise", 'raise ValueError("x")\n', 1, "mo.assert_<kind>"),
    ("assert", "assert 1 == 1\n", 1, "mo.assert_<kind>"),
    ("global", "def f():\n    global x\n", 2, "returns the values"),
    ("del", "x = [1]\ndel x[0]\n", 2, "new list"),
    ("with", 'with open("f") as f:\n    pass\n', 1, "opens nothing"),
    ("yield", "def f():\n    yield 1\n", 2, "return a list"),
    ("is", "x = 1 is 1\n", 1, "=="),
]


@pytest.mark.parametrize("word,source,line,fix", KEYWORD_CASES)
def test_a_construct_the_language_lacks_names_what_to_write(tmp_path, word, source, line, fix):
    """DESIGN 4.8: a construct Starlark lacks fails the build with the
    spelling to use instead (`while` -> `for` over `range()`, or
    `array` for copies of one part; `class` -> a function returning a
    `struct`; `try` -> the build's own error; `import` -> `load()`)."""
    assert f"cannot use reserved keyword `{word}`" in _interpreter(source)
    err = refusal(tmp_path, source)
    assert err.code == "unsupported_syntax"
    assert err.detail["construct"] == word
    assert err.detail["site"] == f"asset.star:{line}"
    assert fix in err.message and err.detail["fix"] == err.message


def test_while_offers_the_loop_and_the_operator(tmp_path):
    """DESIGN 4.8: `while` -> `for` over `range()`, or `array` for
    copies of one part."""
    err = refusal(tmp_path, "while True:\n    pass\n")
    assert "range(n)" in err.message and "'array'" in err.message


DIALECT_CASES = [
    ('x = f"a{1}"\n', "Your Starlark dialect must enable f-strings to use them", ".format("),
    ("def f(a: int):\n    return a\n", "type annotations are not allowed in this dialect", "names alone"),
    ("def f(a, *, b):\n    return a\n", "* keyword-only-arguments is not allowed in this dialect", "by keyword"),
    ("def f(a, /, b):\n    return a\n", "/ positional-only-arguments is not allowed in this dialect", "parameters alone"),
]


@pytest.mark.parametrize("source,said,fix", DIALECT_CASES)
def test_the_dialect_is_fixed_and_says_what_to_write(tmp_path, source, said, fix):
    """DESIGN 4.8: the dialect is one, with no interpreter option
    exposed, and what it leaves out fails with the spelling to use."""
    assert said in _interpreter(source)
    err = refusal(tmp_path, source)
    assert err.code == "unsupported_syntax"
    assert fix in err.message
    assert err.detail["site"] == "asset.star:1"


FORMAT_CASES = [
    ('x = "%02d" % 3\n', "Unsupported format character: '0'", "%d"),
    ('x = "%.3f" % 1.5\n', "Unsupported format character: '.'", "int(x * 100) / 100"),
    ('x = "{:02d}".format(3)\n', "Key `:02d` was not found", "'.format'"),
]


@pytest.mark.parametrize("source,said,fix", FORMAT_CASES)
def test_a_format_spec_names_the_bare_conversion_and_the_slice(tmp_path, source, said, fix):
    """DESIGN 4.8: what the prior must unlearn fails the build with the
    spelling to use instead; a width, flag or precision in '%' or
    '.format' is the interpreter's bare 'Unsupported format character'
    otherwise."""
    assert said in _interpreter(source)
    err = refusal(tmp_path, source)
    assert err.code == "unsupported_format"
    assert fix in err.message and err.detail["fix"] == err.message
    assert "('00' + str(i))[-2:]" in err.message
    assert err.detail["site"] == "asset.star:1"
    assert "`" not in err.message


NAME_CASES = [
    ("round(1.5)", "round", "math.floor"),
    ("sum([1, 2])", "sum", "loop"),
    ("pow(2, 3)", "pow", "math.pow"),
    ('print("x")', "print", "prints nothing"),
    ("set([1])", "set", "dict"),
    ("isinstance(1, int)", "isinstance", "type(x)"),
    ('open("f")', "open", "reads no file"),
]


@pytest.mark.parametrize("expr,name,fix", NAME_CASES)
def test_a_name_the_prior_reaches_for_names_what_to_write(tmp_path, expr, name, fix):
    """DESIGN 4.8: language-level names stay bare, and they are
    `range`, `struct`, `load`, and `math`; a name Python would bind
    fails with what to write."""
    assert f"Variable `{name}` not found" in _interpreter(f"x = {expr}\n")
    err = refusal(tmp_path, f"x = {expr}\n")
    assert err.code == "unknown_name"
    assert err.detail["name"] == name
    assert fix in err.message
    assert err.detail["site"] == "asset.star:1"


def test_an_unbound_name_keeps_the_interpreters_suggestion(tmp_path):
    """DESIGN 8.2: errors state what failed and the shortest path to a
    fix."""
    err = refusal(tmp_path, "heigth = 2\nx = height + 1\n")
    assert err.code == "unknown_name"
    assert "did you mean 'heigth'" in err.message
    assert err.detail["site"] == "asset.star:2"


def test_recursion_is_bounded_not_refused(tmp_path):
    """DESIGN 4.8: recursion is bounded, not refused; every loop
    iterates a finite sequence and the build caps call depth, so
    evaluation always terminates."""
    build_recipe(tmp_path, '''def fact(n):
    return 1 if n == 0 else n * fact(n - 1)
h = fact(10) / 3628800.0
mo.output("a", mo.box(id = "a", size = [1, h, 1]))
''')


def test_a_call_chain_that_reaches_the_cap_names_the_chain_and_the_loop(tmp_path):
    """DESIGN 4.8: a call chain that reaches the cap fails the build
    with the chain and the loop over `range()` to write instead."""
    source = "def down(n):\n    return down(n + 1)\ndown(0)\n"
    assert "error: Starlark call stack overflow" in _interpreter(source)
    err = refusal(tmp_path, source)
    assert err.code == "call_depth"
    assert "range(n)" in err.message
    assert str(dialect.CALL_DEPTH_CAP) in err.message
    chain = err.detail["chain"]
    assert chain[0] == "asset.star:3 in <module>"
    assert chain[-1] == "asset.star:2 in down"
    assert any(row.startswith("...") for row in chain)


@pytest.mark.parametrize("depth,builds", [(40, True), (48, False), (60, False)])
def test_the_cap_is_the_builds_own(tmp_path, depth, builds):
    """DESIGN 4.8: the build caps call depth. The bound is set by the
    build, and a builtin's prelude function takes a frame of it."""
    assert dialect.CALL_DEPTH_CAP == 50
    source = f'''def down(n):
    if n == 0:
        return mo.box(id = "a", size = [1, 1, 1])
    return down(n - 1)
mo.output("a", down({depth}))
'''
    if builds:
        build_recipe(tmp_path, source)
    else:
        assert refusal(tmp_path, source).code == "call_depth"


@pytest.mark.parametrize("source,said", [
    ("x = {1, 2}\n", "Parse error: unexpected symbol"),
    ("y = (x := 1)\n", "Parse error: unexpected symbol"),
    ("x = 1 < 2 < 3\n", "Parse error: unexpected symbol"),
    ('x = 1 + "a"\n', "Operation `+` not supported"),
    ("x = 1 / 0\n", "division by zero"),
    ('for c in "abc":\n    pass\n', "Operation `(iter)` not supported on type `string`"),
])
def test_any_other_error_carries_the_interpreters_words_and_the_line(tmp_path, source, said):
    """DESIGN 6: `build`'s errors carry the recipe `site`."""
    assert said in _interpreter(source)
    err = refusal(tmp_path, source)
    assert err.code == "recipe_error"
    assert err.detail["site"] == "asset.star:1"
    assert "`" not in err.message


def test_an_error_inside_a_helper_names_the_helpers_line(tmp_path):
    """DESIGN 8.1: a build failure names the recipe file and line."""
    err = refusal(tmp_path, 'load("lib/parts.star", "leg")\nleg("a")\n', files={
        "lib/parts.star": 'def leg(id):\n    return mo.box(id = id, size = [1, 1, "x"] + 1)\n',
    })
    assert err.code == "recipe_error"
    assert err.detail["site"] == "lib/parts.star:2"


def test_every_refusal_is_plain_ascii():
    """DESIGN 8.2: tool output is terse machine-readable JSON; the
    refusal texts travel in it."""
    for table in (refusals.KEYWORDS, refusals.DIALECT, refusals.FORMATS, refusals.NAMES):
        for text in table.values():
            assert text.isascii()


# -- fail() ---------------------------------------------------------------------

HELPER = '''def need(n):
    if n < 1:
        fail("n must be >= 1, got", n)
    return n
'''


@pytest.mark.parametrize("name", ["asset.star", "asset.requirements.star", "lib/h.star"])
def test_fail_at_the_top_level_is_refused_in_every_file(tmp_path, name):
    """DESIGN 4.8: Starlark's `fail()` is refused at the top level. It
    would stop the build with one message where a report carries every
    finding. The refusal names the assert to write, in an asset, a
    requirements file, and a `lib/` file alike."""
    asset = 'mo.output("a", mo.box(id = "a", size = [1, 1, 1]))\n'
    files = {"asset.star": asset, "lib/h.star": "H = 1\n"}
    if name == "lib/h.star":
        files["asset.star"] = 'load("lib/h.star", "H")\n' + asset
    files[name] = files.get(name, "") + 'fail("the tank is too tall")\n'
    text = files.pop("asset.star")
    err = refusal(tmp_path, text, files=files)
    assert err.code == "fail_at_top_level"
    assert err.detail["site"].split(":")[0] == name
    assert "mo.assert_<kind>" in err.message and "the tank is too tall" in err.message


def test_fail_inside_a_function_is_the_functions_refusal(tmp_path):
    """DESIGN 4.8: inside a function `fail()` is the function's own
    refusal of what it was handed. The build fails with its message,
    its line, and the calls on the way, the line to fix among them."""
    err = refusal(
        tmp_path, 'load("lib/h.star", "need")\nn = need(3)\nm = need(0)\n',
        files={"lib/h.star": HELPER},
    )
    assert err.code == "recipe_failed"
    assert err.message == "'need' refused what it was called with: n must be >= 1, got 0"
    assert (err.detail["site"], err.detail["via"]) == ("lib/h.star:3", ["asset.star:3"])


@pytest.mark.parametrize("call, said", [
    ('fail(msg = "too wide")', "too wide"),
    ('fail("too", "wide", sep = "-")', "too-wide"),
    ('fail("too wide", attr = "width")', "too wide attr = width"),
    ("fail()", "no message"),
])
def test_fail_keeps_what_it_was_told_however_it_was_passed(tmp_path, call, said):
    """DESIGN 4.8: inside a function `fail()` fails the build with its
    message. The message is what the call passed, by position or by
    keyword."""
    err = refusal(tmp_path, f"def need():\n    {call}\n\nneed()\n")
    assert err.message == f"'need' refused what it was called with: {said}"
