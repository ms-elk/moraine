"""Copies a recipe wrote by hand (DESIGN 4.8).

A loop whose body places an unchanged part under a per-copy transform
writes as nodes what ``array`` emits as instances: one prototype and
its transforms, kept as instancing through to the export (Invariant
6). The build finds the pattern in the IR it made and warns with the
``array`` to write. Copies placed by no step are left alone: no
operator emits them as instances, and a warning that names no fix
trains its reader to pass warnings by.

The pattern is sibling nodes whose content, all but a transform, is
equal: ``transform`` nodes whose inputs have one content. Content is
what a node computes from (its op, its params, its inputs' content,
and its id where the id salts a stream), never its id or where it was
written, since a loop names every copy.
"""

from __future__ import annotations

import hashlib
import json

from .. import ops
from ..core.brief import LISTED
from ..core.asset import Node, Asset
from ..core.sites import where
from ..errors import MoraineError

# The op whose params are a placement and nothing else.
PLACING = "transform"
# Copies before the pattern is worth a warning: a pair is a left and a
# right more often than a row.
MIN_COPIES = 3
_DEFAULTS = {
    "translate": [0.0, 0.0, 0.0], "rotate_deg": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
}
_STEP_TOL = 1e-9


def _salted(op: str) -> bool:
    try:
        return bool(getattr(ops.get(op), "USES_NODE_ID", False))
    except MoraineError:
        return True  # an op the registry does not know: its own content


def _contents(sc: Asset) -> dict[str, str]:
    """Each node's content as a hash, inputs first."""
    out: dict[str, str] = {}

    def of(node_id: str) -> str:
        if node_id not in out:
            node = sc.nodes[node_id]
            payload = [node.op, node.params, [of(i) for i in node.inputs]]
            if _salted(node.op):
                payload.append(node.id)
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            out[node_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return out[node_id]

    for node_id in sc.nodes:
        of(node_id)
    return out


def _vector(node: Node, key: str) -> list[float] | None:
    value = node.params.get(key, _DEFAULTS[key])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value, value, value]
    if (
        isinstance(value, list) and len(value) == 3
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in value)
    ):
        return [float(c) for c in value]
    return None


def _step(copies: list[Node], key: str) -> list[float] | None:
    """The constant difference of one placement key across the copies
    in the order they were written, or None where there is none."""
    rows = [_vector(n, key) for n in copies]
    if any(r is None for r in rows):
        return None
    step = [b - a for a, b in zip(rows[0], rows[1])]
    for before, after in zip(rows[1:], rows[2:]):
        if any(abs((b - a) - s) > _STEP_TOL for a, b, s in zip(before, after, step)):
            return None
    return step


def _array(copies: list[Node]) -> str | None:
    """The ``array`` that places the copies, where their placement
    follows a step ``array`` composes the same way: a row (one
    translation step under one rotation and one scale) or a ring about
    the origin (one rotation step about one axis, no translation)."""
    scaled = _step(copies, "scale")
    moved, turned = _step(copies, "translate"), _step(copies, "rotate_deg")
    if scaled is None or any(scaled) or moved is None or turned is None:
        return None
    first = copies[0]
    if any(moved) and not any(turned):
        step = f"step_translate = {_literal(moved)}"
    elif any(turned) and not any(moved):
        axis = [i for i, s in enumerate(turned) if s]
        still = [c for i, c in enumerate(_vector(first, "rotate_deg")) if i not in axis]
        if len(axis) != 1 or any(still) or any(_vector(first, "translate")):
            return None
        step = f"step_rotate_deg = {_literal(turned)}"
    else:
        return None
    return (
        f"mo.array(<the first copy, '{first.id}'>, id = '<name>', "
        f"count = {len(copies)}, {step})"
    )


def _literal(vector: list[float]) -> str:
    return "[" + ", ".join(f"{round(c, 9):g}" for c in vector) + "]"


def find(sc: Asset) -> list[dict]:
    """The ``copies_not_instanced`` warnings of one asset (DESIGN 4.8),
    one per part written as copies, in the order the first copy of each
    was written."""
    contents = _contents(sc)
    groups: dict[str, list[Node]] = {}
    for node in sc.nodes.values():
        if node.op == PLACING and len(node.inputs) == 1:
            groups.setdefault(contents[node.inputs[0]], []).append(node)
    out = []
    for copies in groups.values():
        if len(copies) < MIN_COPIES:
            continue
        if len({contents[n.id] for n in copies}) < len(copies):
            continue  # one placement written twice is no row of copies
        spelled = _array(copies)
        if spelled is None:
            continue
        out.append({
            "warning": "copies_not_instanced",
            **where(copies[0]),
            "count": len(copies),
            "nodes": [n.id for n in copies[:LISTED]],
            "fix": spelled,
            "note": (
                f"{len(copies)} nodes place one unchanged part by one step: "
                f"copies written as nodes ship as {len(copies)} meshes, "
                f"where instances ship as one prototype and its "
                f"transforms; write the array under 'fix' (an operator "
                f"that takes a mesh, as boolean does, takes the array "
                f"through mo.bake_instances)"
            ),
        })
    return out
