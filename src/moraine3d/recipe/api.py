"""The ``mo`` namespace as text (DESIGN 4.8): one signature per
builtin, generated from the registries the prelude is generated from,
so the spelling the reference shows beside each contract and the
listing ``moraine docs api`` prints can never lag the build.

A signature lists the inputs, by role where the operator names one
(``base, part``) and by position otherwise (``a, b, ...``; optional
ones in brackets), an input passed by keyword alone as
``name = node``, then ``id`` and every keyword the builtin accepts,
an integer-typed keyword marked ``int`` (a float there is refused,
DESIGN 4.8) and every other one as ``...``. The declarations' shapes
are the host's argument tables.
"""

from __future__ import annotations

from .. import ops
from ..core import asserts as vocab
from ..core import numbers
from . import dialect, prelude

_INPUT_NAMES = "abcdefgh"

# The declarations and handles, as the host reads their arguments:
# positional names first, keyword-only ones as ``name = ...``.
DECLARATIONS = {
    "output": "mo.output(name, node)",
    "group": "mo.group(name, outputs)",
    "param": "mo.param(id, default, min = ..., max = ..., label = ..., step = ...)",
    "budget": "mo.budget(materials = int)",
    "waive": "mo.waive(output, check)",
    "waive_asset": "mo.waive_asset(check)",
    "vendored": "mo.vendored(source, hash)",
    "param_value": "mo.param_value(id)",
}


def _kinded(name: str, kind: str) -> str:
    return f"{name} = int" if kind == numbers.INT else f"{name} = ..."


def _inputs(lo: int, hi: int | None, roles: tuple[str, ...] = ()) -> list[str]:
    names = list(roles) + list(_INPUT_NAMES)[len(roles):]
    required = names[:lo]
    if hi is None:
        return required + ["..."]
    optional = names[lo:hi]
    return required + ([f"[{', '.join(optional)}]"] if optional else [])


def operator(name: str) -> str:
    """``mo.<op>(...)`` as the recipe writes it."""
    module = ops.get(name)
    if getattr(module, "IS_REFERENCE", False):
        keywords = [k for k in module.PARAM_KINDS if k not in ("asset", "output", "project")]
        head = ["asset, output"]
    else:
        slotted = getattr(module, "SLOT_INPUTS", {})
        written = getattr(module, "SLOT_PARAM", None)
        keywords = [k for k in module.PARAM_KINDS if k != written]
        lo, hi = module.INPUT_ARITY
        head = _inputs(lo, lo if slotted else hi, getattr(module, "INPUT_NAMES", ()))
        head += [f"{k} = node" for k in slotted]
    parts = head + ["id = ..."] + [_kinded(k, module.PARAM_KINDS[k]) for k in keywords]
    return f"mo.{name}({', '.join(parts)})"


def assertion(kind: str) -> str:
    """``mo.assert_<kind>(...)`` as the recipe writes it."""
    module = vocab.KINDS[kind]
    parts = ["id = ..."] + [
        _kinded(k, v if k in vocab.THRESHOLDS else numbers.OTHER)
        for k, v in module.KEY_KINDS.items()
    ]
    return f"mo.{prelude.ASSERT_PREFIX}{kind}({', '.join(parts)})"


def signature(name: str) -> str | None:
    """The signature of one name under ``mo``, or None for a name that
    is not one (a section the reference titles otherwise)."""
    if name in DECLARATIONS:
        return DECLARATIONS[name]
    if name.startswith(prelude.ASSERT_PREFIX):
        kind = name[len(prelude.ASSERT_PREFIX):]
        return assertion(kind) if kind in vocab.AUTHORABLE_KINDS else None
    if name in ops.names():
        return operator(name)
    if name in vocab.AUTHORABLE_KINDS:
        return assertion(name)
    return None


def text() -> str:
    """The ``api`` section: every name under ``mo`` and ``math``, one
    signature per line, in the prelude's order."""
    lines = [
        "## api (the `mo` namespace, generated from the registries)",
        "",
        "Every builtin a recipe may call, with its arguments: inputs by",
        "position (an input listed by a name may be passed by it, and one",
        "listed as `name = node` only so), then `id` and the keywords. A",
        "keyword marked `int`",
        "takes an integer (a float there is refused; an integer where a",
        "float is written is widened). The contract behind each name is",
        "its own section (`moraine docs <name>`, whose reply lists the",
        "number type of every param under `numbers`); the declarations",
        "are under *builtins*.",
        "",
        "### operators",
        "",
    ]
    lines += [operator(n) for n in prelude.operator_names()]
    lines += ["", "### asserts", ""]
    lines += [assertion(k) for k in sorted(vocab.AUTHORABLE_KINDS)]
    lines += ["", "### declarations", ""]
    lines += [DECLARATIONS[d] for d in prelude.DECLARATIONS]
    lines += ["", "### handles", ""]
    lines += [DECLARATIONS[h] for h in prelude.HANDLES]
    lines += ["", "### math", ""]
    lines.append(", ".join(f"math.{c}" for c in dialect.MATH_CONSTANTS))
    lines.append(", ".join(f"math.{f}(...)" for f in dialect.MATH_FUNCTIONS))
    return "\n".join(lines)
