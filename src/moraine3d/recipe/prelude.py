"""The prelude: ``mo`` and ``math`` as Starlark, generated from the
registries (DESIGN 4.8).

One thin function per builtin reads the call stack and hands it to the
host with the call's arguments. The source is generated, so the
recipe's vocabulary is the registries' by construction, and its text
is what ``build_version`` signs (DESIGN 4.4).
"""

from __future__ import annotations

from .. import ops
from ..core import asserts as vocab
from . import dialect

# One declaration per IR root key (DESIGN 4.8), by the root key it
# writes.
DECLARATIONS = {
    "output": "outputs",
    "group": "groups",
    "param": "params",
    "budget": "budgets",
    "waive": "checks",
    "waive_asset": "asset_checks",
    "vendored": "provenance",
}

# The builtins that hand a recipe a value to pass on and write no
# entry of their own (DESIGN 4.8).
HANDLES = ("param_value",)

ASSERT_PREFIX = "assert_"
# The language-level name the build binds over in every module.
FAIL = "fail"

_STACK = "call_stack(strip_frames = 1)"


def operator_names() -> list[str]:
    return ops.names()


def assert_names() -> list[str]:
    return [ASSERT_PREFIX + kind for kind in sorted(vocab.AUTHORABLE_KINDS)]


def names() -> list[str]:
    """Every name under ``mo``, in the order the prelude defines them."""
    return operator_names() + assert_names() + list(DECLARATIONS) + list(HANDLES)


def source() -> str:
    """The prelude's Starlark text. The host functions it calls
    (``_node``, ``_assert``, ``_declare``, ``_handle``) and the ``math`` members are
    bound by the build before it is evaluated."""
    out: list[str] = []
    for op in operator_names():
        out.append(
            f"def _{op}(*args, **kw):\n"
            f'    return _node("{op}", {_STACK}, *args, **kw)\n'
        )
    for kind in sorted(vocab.AUTHORABLE_KINDS):
        out.append(
            f"def _{ASSERT_PREFIX}{kind}(*args, **kw):\n"
            f'    return _assert("{kind}", {_STACK}, *args, **kw)\n'
        )
    for decl in DECLARATIONS:
        out.append(
            f"def _{decl}(*args, **kw):\n"
            f'    return _declare("{decl}", {_STACK}, *args, **kw)\n'
        )
    for handle in HANDLES:
        out.append(
            f"def _{handle}(*args, **kw):\n"
            f'    return _handle("{handle}", {_STACK}, *args, **kw)\n'
        )
    out.append(
        "mo = struct(" + ", ".join(f"{n} = _{n}" for n in names()) + ")\n"
    )
    # The language's own ``fail``, taken over in every module (DESIGN
    # 4.8): the build answers it, knowing where it was called.
    out.append(
        f"def {FAIL}(*args, **kw):\n"
        f"    return _fail({_STACK}, *args, **kw)\n"
    )
    members = list(dialect.MATH_CONSTANTS) + list(dialect.MATH_FUNCTIONS)
    out.append(
        "math = struct(" + ", ".join(f"{n} = _math_{n}" for n in members) + ")\n"
    )
    return "\n".join(out)
