"""The recipe's language, fixed (DESIGN 4.8).

One dialect, no option exposed: what a recipe may write is the same in
every project, so a recipe that builds here builds everywhere.
"""

from __future__ import annotations

import math
import os
import threading

# The pure-Python backend, set before mpmath is first imported: the
# transcendental functions are then integer arithmetic on every
# platform, which is what makes their results the same on each.
os.environ.setdefault("MPMATH_NOGMPY", "1")
import mpmath  # noqa: E402
import starlark as sl  # noqa: E402

# The name of the interpreter's distribution; its version is part of
# ``build_version`` (DESIGN 4.4).
INTERPRETER = "starlark-pyo3"

# The call depth cap (DESIGN 4.8): recursion is bounded, not refused.
# The interpreter's own default made explicit, so the bound is the
# build's and not an accident of the pin. A builtin's prelude function
# takes one frame of it.
CALL_DEPTH_CAP = 50

# The name a recipe module loads its builtins under. Not a file: the
# build evaluates the load itself before the recipe runs, and refuses
# the name anywhere a recipe writes it.
BUILTINS = "@moraine"

# ``math`` as the recipe sees it (DESIGN 4.4). The functions IEEE 754
# rounds exactly are the host's; the transcendental ones are computed
# beyond the double and rounded once, so no platform's library shows
# in the IR. No non-finite constant: such a value cannot cross into
# the IR.
MATH_CONSTANTS = ("pi", "tau", "e")
MATH_FUNCTIONS = (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "floor",
    "ceil", "fabs", "hypot", "radians", "degrees", "exp", "log", "pow",
    "fmod",
)


def dialect(*, load: bool = True) -> sl.Dialect:
    """The recipe dialect. ``load`` is off for a requirements file,
    which loads nothing (DESIGN 5.2)."""
    d = sl.Dialect.standard()
    d.enable_def = True
    d.enable_load = load
    d.enable_top_level_stmt = True
    d.enable_f_strings = False
    d.enable_types = sl.DialectTypes.DISABLE
    d.enable_load_reexport = False
    return d


def recipe_globals() -> sl.Globals:
    return sl.Globals.extended_by([sl.LibraryExtension.StructType])


def prelude_globals() -> sl.Globals:
    """The prelude alone sees the call stack (DESIGN 4.8), so a recipe
    cannot observe where it was called from."""
    return sl.Globals.extended_by(
        [sl.LibraryExtension.CallStack, sl.LibraryExtension.StructType]
    )


def options() -> sl.EvalOptions:
    return sl.EvalOptions(max_callstack_size=CALL_DEPTH_CAP)


# The functions whose result the host's library rounds differently
# between platforms (DESIGN 4.4). Each is computed at TRANSCENDENTAL_DPS
# decimal digits and rounded once to the nearest double, the value a
# correctly rounded library would return.
TRANSCENDENTAL = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log",
    "pow", "hypot",
})
TRANSCENDENTAL_DPS = 40

# A context per thread, its precision set once: the library's default
# context is shared by every thread, and a precision set for the span
# of one call would be reset by another thread's call ending inside
# that span. Per thread rather than one of the build's own, so a
# precision the library itself changes inside a call stays with the
# thread that made the call.
_local = threading.local()


def _context() -> mpmath.MPContext:
    ctx = getattr(_local, "context", None)
    if ctx is None:
        ctx = _local.context = mpmath.MPContext()
        ctx.dps = TRANSCENDENTAL_DPS
    return ctx


def _transcendental(name: str):
    def compute(*args):
        ctx = _context()
        values = [ctx.mpf(a) for a in args]
        if name == "hypot":
            result = ctx.sqrt(sum((v * v for v in values), ctx.mpf(0)))
        elif name == "pow":
            result = ctx.power(*values)
        else:
            result = getattr(ctx, name)(*values)
        if not isinstance(result, ctx.mpf):
            raise ValueError("math domain error")
        return float(result)

    return compute


def math_function(name: str):
    """One ``math`` function for the recipe, refusing a result that is
    not a finite number and a domain error alike at the call."""
    fn = _transcendental(name) if name in TRANSCENDENTAL else getattr(math, name)

    def call(*args):
        try:
            result = fn(*args)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"math.{name}: {e}") from None
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError(
                f"math.{name} returned a non-finite number, which a "
                f"recipe cannot hold"
            )
        return result

    call.__name__ = name
    return call
