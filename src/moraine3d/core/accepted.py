"""The human's accepted knob values, ``params.json`` (DESIGN 4.2, 7.2).

One file per project, beside the root asset: a JSON object holding one
``param_id`` per line in id order, the values the human accepted in the
viewer for the root asset's knobs. The build reads it, the viewer's
accept is its one writer, and the order is the ids' own so that a
recipe edit which reorders the declarations moves no line of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import AssetError
from .atomic import ReplaceError, atomic_replace
from .asset import PARAM_ID_RE, param_kind

PARAMS_FILE = "params.json"


def beside(site: str) -> str:
    """The ``params.json`` of the project whose root asset declared a
    knob at ``site``, as reports show it."""
    folder = site.rsplit(":", 1)[0].rpartition("/")[0]
    return f"{folder}/{PARAMS_FILE}" if folder else PARAMS_FILE


def _refuse(shown: str, why: str, line: int | None = None) -> AssetError:
    at = f"{shown}:{line}" if line else shown
    return AssetError(
        "bad_params_file",
        f"'{at}' {why}; the file holds the values accepted in the viewer, "
        f"one \"<param_id>\": <value> per line inside one object - correct "
        f"the line, or delete the file to return every knob to its default",
        file=shown, **({"line": line} if line else {}),
    )


def _line_of(text: str, key: str) -> int | None:
    needle = json.dumps(key)
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(needle):
            return n
    return None


def loads(data: bytes, shown: str) -> dict[str, object]:
    """The accepted values a ``params.json`` holds, by param id.
    ``shown`` names the file in an error."""
    try:
        # An editor may have saved the file with a byte order mark.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise _refuse(shown, f"is not valid UTF-8 (byte {e.start})") from e
    duplicated: list[str] = []

    def pairs(found: list[tuple[str, object]]) -> dict:
        out: dict = {}
        for k, v in found:
            if k in out:
                duplicated.append(k)
            out[k] = v
        return out

    def nonfinite(literal: str):
        raise _refuse(shown, f"holds the non-finite number '{literal}'")

    try:
        doc = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except json.JSONDecodeError as e:
        raise _refuse(shown, f"is not valid JSON ({e.msg})", e.lineno) from e
    if not isinstance(doc, dict):
        raise _refuse(shown, "does not hold a JSON object")
    if duplicated:
        key = duplicated[0]
        raise _refuse(shown, f"holds '{key}' twice", _line_of(text, key))
    for key, value in doc.items():
        if not PARAM_ID_RE.fullmatch(key):
            raise _refuse(
                shown, f"holds the key {key!r}, which is no param id",
                _line_of(text, key),
            )
        if param_kind(value) is None:
            raise _refuse(
                shown,
                f"holds a value for '{key}' that no knob takes (a knob is "
                f"a number, a vector of 2 to 4 numbers, a text, or a "
                f"boolean)",
                _line_of(text, key),
            )
    return doc


def dumps(values: dict[str, object]) -> str:
    """Canonical text: one id per line, in id order, so that a changed
    value is a one-line diff."""
    lines = [
        f"{json.dumps(key)}: "
        + json.dumps(values[key], separators=(", ", ": "), allow_nan=False)
        for key in sorted(values)
    ]
    if not lines:
        return "{\n}\n"
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write(path: Path, values: dict[str, object]) -> None:
    """Write a project's ``params.json`` whole and atomically
    (``core.atomic``). The in-flight temp file is this write's own, so
    a failed write removes it directly (DESIGN 4.5)."""
    path = Path(path)
    try:
        atomic_replace(path, dumps(values).encode("utf-8"))
    except ReplaceError as err:
        err.tmp.unlink(missing_ok=True)
        e = err.cause
        if err.step == "write":
            raise AssetError(
                "params_unwritable",
                f"cannot write '{path.name}' in '{path.parent}' "
                f"({e.__class__.__name__}); check directory permissions",
            ) from e
        raise AssetError(
            "params_locked",
            f"cannot replace '{path.name}' ({e.__class__.__name__}: held "
            f"open by another program); close it and retry",
        ) from e
