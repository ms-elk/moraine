"""The interpreter's errors as the build's refusals (DESIGN 4.8).

Python syntax is kept so an agent's prior transfers; what the prior
must unlearn fails the build with the spelling to use instead. The
interpreter reports an error as text, so this module reads that text:
the message line, the span it points at, and the call chain above it.
Tests freeze the messages read here, as they freeze the stack's shape.
"""

from __future__ import annotations

import re

from ..errors import BuildError
from . import dialect, sites

_MESSAGE = re.compile(r"^error: (.*)$", re.M)
_KEYWORD = re.compile(r"cannot use reserved keyword `(\w+)`")
_UNKNOWN_NAME = re.compile(r"Variable `(\w+)` not found")

_LOOP = (
    "write 'for i in range(n):' over a finite sequence, or use the 'array' "
    "operator for copies of one part"
)

# A construct the language lacks, by its keyword, with what to write.
KEYWORDS = {
    "while": f"Starlark has no 'while': {_LOOP}",
    "class": (
        "Starlark has no 'class': write a function that returns a "
        "struct(...) of the values"
    ),
    "try": (
        "Starlark has no 'try': a recipe does not catch errors, the build "
        "reports them with the recipe line"
    ),
    "raise": (
        "Starlark has no 'raise': a requirement is an assert "
        "(mo.assert_<kind>), which the report judges with every other entry"
    ),
    "assert": (
        "Starlark has no 'assert' statement: a requirement is a declaration, "
        "mo.assert_<kind>(...), which the report judges with every other "
        "entry"
    ),
    "import": (
        "Starlark has no 'import': 'math' and 'mo' are already in scope, "
        "and a project's own definitions load with load(\"lib/<name>.star\", "
        "\"<symbol>\")"
    ),
    "from": (
        "Starlark has no 'from ... import': a project's own definitions "
        "load with load(\"lib/<name>.star\", \"<symbol>\")"
    ),
    "global": (
        "Starlark has no 'global': a function returns the values it "
        "computes, and a file's top-level names are constants once bound"
    ),
    "nonlocal": (
        "Starlark has no 'nonlocal': a function returns the values it "
        "computes"
    ),
    "del": "Starlark has no 'del': build a new list or dict without the entry",
    "with": "Starlark has no 'with': a recipe opens nothing",
    "yield": (
        "Starlark has no 'yield': return a list, built with a "
        "comprehension or a loop"
    ),
    "async": "Starlark has no 'async': a recipe runs top to bottom",
    "await": "Starlark has no 'await': a recipe runs top to bottom",
    "is": "Starlark has no 'is': compare with '=='",
    "lambda": "write a named function with 'def'",
}

# The dialect's own refusals, by a phrase of the interpreter's message.
DIALECT = {
    "must enable f-strings": (
        "f-strings are not part of the recipe language: format with "
        "'\"%s_%d\" % (name, i)' or '\"{}_{}\".format(name, i)'"
    ),
    "type annotations are not allowed": (
        "type annotations are not part of the recipe language: write the "
        "parameter names alone"
    ),
    "keyword-only-arguments is not allowed": (
        "a bare '*' in a parameter list is not part of the recipe "
        "language: list the parameters and pass them by keyword at the call"
    ),
    "positional-only-arguments is not allowed": (
        "'/' in a parameter list is not part of the recipe language: list "
        "the parameters alone"
    ),
}

_PAD = (
    "pad with a slice, ('00' + str(i))[-2:], and shorten a float with "
    "arithmetic, int(x * 100) / 100"
)

# A format spec the language lacks, by a phrase of the interpreter's message.
FORMATS = {
    "Unsupported format character": (
        "'%' in the recipe language takes a bare conversion only (%d, %s, "
        "%x, %e, %%), with no width, flag or precision: " + _PAD
    ),
    "format spec": (
        "'.format' in the recipe language takes empty braces only ({} or "
        "{0}), with no spec after a ':': " + _PAD
    ),
}
_FORMAT_KEY = re.compile(r"Key [`'][^`']*:[^`']*[`'] was not found")

# Names an agent's prior reaches for that the language does not bind.
NAMES = {
    "round": "write 'math.floor(x + 0.5)', or 'int(x)' to drop the fraction",
    "sum": "add in a loop: 'total = 0' then 'for x in xs: total += x'",
    "pow": "write 'math.pow(x, y)', or 'x * x' for a square",
    "divmod": "write 'x // y' and 'x % y'",
    "isinstance": "compare 'type(x)' with a type name: 'type(x) == \"list\"'",
    "print": (
        "a recipe prints nothing: the build reports what it built, and "
        "'moraine query' reads an evaluated shape"
    ),
    "set": "use a dict's keys, or a list checked with 'in'",
    "open": "a recipe reads no file: values come from its own text and its knobs",
    "json": "a recipe reads no file: write the values as literals",
    "input": "a value the human sets is a knob: mo.param(id, default, ...)",
    "call_stack": (
        "a recipe cannot observe where it was called from; every node "
        "carries its site already"
    ),
}


def _message(text: str) -> str:
    m = _MESSAGE.search(text)
    return m[1].strip() if m else text.strip().split("\n")[0]


def _site(text: str) -> dict:
    """The site an interpreter error names: its span, else the
    innermost frame of its call chain."""
    at = sites.span(text)
    if at is None:
        found = sites.frames(text)
        at = found[-1].site if found else None
    return {"site": at} if at else {}


def from_interpreter(text: str) -> BuildError:
    """The refusal for an interpreter error, given its text."""
    message = _message(text)
    where = _site(text)

    m = _KEYWORD.search(message)
    if m:
        word = m[1]
        fix = KEYWORDS.get(word)
        if fix is None:
            return BuildError(
                "unsupported_syntax",
                f"'{word}' is a reserved word of the recipe language and "
                f"names no construct in it; pick another name",
                construct=word, **where,
            )
        return BuildError("unsupported_syntax", fix, construct=word, fix=fix, **where)

    for phrase, fix in DIALECT.items():
        if phrase in message:
            return BuildError("unsupported_syntax", fix, fix=fix, **where)

    if "Unsupported format character" in message:
        fix = FORMATS["Unsupported format character"]
        return BuildError("unsupported_format", fix, fix=fix, **where)
    if _FORMAT_KEY.search(message):
        fix = FORMATS["format spec"]
        return BuildError("unsupported_format", fix, fix=fix, **where)

    if "named argument after *args or **kwargs" in message:
        fix = (
            "a keyword cannot follow a '*' expansion: write the keywords "
            "first, then the expansion, as in f(id = ..., *xs)"
        )
        return BuildError("argument_order", fix, fix=fix, **where)

    if "`load` is not allowed in this dialect" in message:
        fix = (
            "a requirements file loads nothing: its thresholds are its own "
            "constants, written in the file"
        )
        return BuildError("requirements_load", fix, fix=fix, **where)

    if "call stack overflow" in message:
        found = sites.frames(text)
        fix = (
            f"the call chain reached the depth cap of "
            f"{dialect.CALL_DEPTH_CAP}; a recipe repeats with a loop, not "
            f"with a function that calls itself: {_LOOP}"
        )
        return BuildError(
            "call_depth", fix, fix=fix, chain=sites.chain(found), **where,
        )

    m = _UNKNOWN_NAME.search(message)
    if m:
        name = m[1]
        hint = NAMES.get(name)
        if hint is not None:
            return BuildError(
                "unknown_name",
                f"'{name}' is not part of the recipe language: {hint}",
                name=name, fix=hint, **where,
            )
        return BuildError(
            "unknown_name",
            f"{message.replace('`', chr(39))}; a name is bound before the "
            f"line that uses it, by an assignment, a 'def', or a load()",
            name=name, **where,
        )

    return BuildError("recipe_error", message.replace("`", "'"), **where)
