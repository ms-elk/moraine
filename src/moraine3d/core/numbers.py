"""The number type a param declares (DESIGN 4.8).

A recipe's numbers are the language's: ``/`` gives a float, ``//`` an
integer. What a param holds is the param's to say. A float param takes
an integer and holds it as a float, and an integer param refuses a
float whatever its value, so one value written two ways is one IR.
"""

from __future__ import annotations

FLOAT = "float"
INT = "int"
OTHER = "other"
KINDS = (FLOAT, INT, OTHER)


class FloatForInteger(ValueError):
    """A float reached an integer param; ``path`` says where inside the
    value (``[1]``, ``.gt``), empty for the value itself."""

    def __init__(self, value: float, path: str):
        super().__init__(f"{value!r} at '{path}'")
        self.value = value
        self.path = path


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def typed(value, kind: str, path: str = ""):
    """``value`` as a param of ``kind`` holds it. Numbers inside a list
    or a dict take the same kind, and what is not a number is left as
    it is."""
    if kind == OTHER:
        return value
    if isinstance(value, list):
        return [typed(v, kind, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: typed(v, kind, f"{path}.{k}") for k, v in value.items()}
    if kind == FLOAT and _is_int(value):
        return float(value)
    if kind == INT and isinstance(value, float):
        raise FloatForInteger(value, path)
    return value


def integer_fix(what: str, e: FloatForInteger) -> str:
    """The refusal's text: what to write in the float's place."""
    where = f" (at {e.path})" if e.path else ""
    return (
        f"{what} takes an integer and got the float {e.value!r}{where}; "
        f"write int(x) around the expression - '/' always gives a float, "
        f"and '//' on a float stays a float (12.0 // 3 is 4.0)"
    )
