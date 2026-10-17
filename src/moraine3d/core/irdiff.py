"""The graph change between two IRs of one asset (DESIGN 6).

A recipe edit inside a loop or a function reaches nodes the edited
line does not name. The diff says which: nodes and asserts added,
removed, and changed, and the outputs the change makes dirty. It reads
the IR alone and compares by id, so the order entries are written in
is no change, and neither is where an entry was written from.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .brief import LISTED
from .asset import Node, Asset
from .sites import where

# The root keys a declaration writes, beside the nodes, the asserts,
# and the outputs the diff reports entry by entry.
_DECLARED = ("params", "groups", "checks", "asset_checks", "budgets", "provenance")
_POSITION = ("site", "via")


def _text(value) -> str:
    """A value as the cache key reads it: a number written another way
    is another value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _plain(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in _POSITION}


@dataclass
class AssetDiff:
    """What changed between two IRs of one asset. ``old`` None is an
    asset with nothing to compare against, where every entry is new."""

    new: Asset
    old: Asset | None
    nodes_added: list[Node] = field(default_factory=list)
    nodes_removed: list[Node] = field(default_factory=list)
    nodes_changed: list[tuple[Node, Node]] = field(default_factory=list)
    asserts_added: list[dict] = field(default_factory=list)
    asserts_removed: list[dict] = field(default_factory=list)
    asserts_changed: list[tuple[dict, dict]] = field(default_factory=list)
    outputs_added: list[str] = field(default_factory=list)
    outputs_removed: list[str] = field(default_factory=list)
    outputs_dirty: list[str] = field(default_factory=list)
    declarations: list[str] = field(default_factory=list)
    # An asset the named base holds no IR of: nothing to compare with,
    # which is neither "all added" nor "unchanged".
    uncompared: bool = False

    @property
    def empty(self) -> bool:
        return self.old is not None and not (
            self.nodes_added or self.nodes_removed or self.nodes_changed
            or self.asserts_added or self.asserts_removed or self.asserts_changed
            or self.outputs_added or self.outputs_removed or self.outputs_dirty
            or self.declarations
        )

    @property
    def counts(self) -> dict:
        """The counts a reload logs (DESIGN 6): added, removed,
        changed, for nodes and for asserts."""
        return {
            "nodes": [len(self.nodes_added), len(self.nodes_removed), len(self.nodes_changed)],
            "asserts": [
                len(self.asserts_added), len(self.asserts_removed), len(self.asserts_changed),
            ],
        }


def _node_content(node: Node) -> str:
    return _text([node.op, node.inputs, node.params])


def _structures(sc: Asset, ids: list[str]) -> dict[str, str]:
    """Each named node's content with every input read by its own
    content in place of its id, so a subgraph compares equal under
    another naming."""
    memo: dict[str, str] = {}
    stack = list(ids)
    while stack:
        node_id = stack[-1]
        if node_id in memo:
            stack.pop()
            continue
        node = sc.nodes[node_id]
        pending = [i for i in node.inputs if i not in memo]
        if pending:
            stack.extend(pending)
            continue
        text = _text([node.op, [memo[i] for i in node.inputs], node.params])
        memo[node_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stack.pop()
    return {node_id: memo[node_id] for node_id in ids}


def _renamed(diff: AssetDiff) -> list[tuple[Node, Node]]:
    """Of the nodes added and removed, the pairs that are one subgraph
    under two ids. The content is all a pair shares: a node removed
    and an equal one written elsewhere read the same."""
    if not diff.nodes_added or not diff.nodes_removed:
        return []
    was = _structures(diff.old, [n.id for n in diff.nodes_removed])
    now = _structures(diff.new, [n.id for n in diff.nodes_added])
    waiting: dict[str, list[Node]] = {}
    for node in diff.nodes_removed:
        waiting.setdefault(was[node.id], []).append(node)
    pairs = []
    for node in diff.nodes_added:
        held = waiting.get(now[node.id])
        if held:
            # Several nodes of one content (a loop's copies): the one
            # whose id reads closest is the one this node was.
            best = max(held, key=lambda old: difflib.SequenceMatcher(
                None, old.id, node.id, autojunk=False).ratio())
            held.remove(best)
            pairs.append((best, node))
    return pairs


def _upstream(sc: Asset, start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(sc.nodes[node_id].inputs)
    return seen


def asset_diff(old: Asset | None, new: Asset,
               moved: frozenset[str] | set[str] = frozenset()) -> AssetDiff:
    """The change from ``old`` to ``new``. ``moved`` names nodes of
    ``new`` whose value changed though their own line did not, a
    reference to an asset that changed: they are no change of this
    asset, and the outputs downstream of them are dirty."""
    diff = AssetDiff(new=new, old=old)
    if old is None:
        diff.nodes_added = list(new.nodes.values())
        diff.asserts_added = list(new.asserts or [])
        diff.outputs_added = list(new.outputs)
        return diff
    for node_id, node in new.nodes.items():
        before = old.nodes.get(node_id)
        if before is None:
            diff.nodes_added.append(node)
        elif _node_content(before) != _node_content(node):
            diff.nodes_changed.append((before, node))
    diff.nodes_removed = [n for i, n in old.nodes.items() if i not in new.nodes]

    was = {a["id"]: a for a in old.asserts or []}
    now = {a["id"]: a for a in new.asserts or []}
    for aid, entry in now.items():
        if aid not in was:
            diff.asserts_added.append(entry)
        elif _text(_plain(was[aid])) != _text(_plain(entry)):
            diff.asserts_changed.append((was[aid], entry))
    diff.asserts_removed = [a for i, a in was.items() if i not in now]

    touched = (
        {n.id for n in diff.nodes_added}
        | {n.id for _, n in diff.nodes_changed}
        | set(moved)
    )
    for name, node_id in new.outputs.items():
        if name not in old.outputs:
            diff.outputs_added.append(name)
        elif old.outputs[name] != node_id or _upstream(new, node_id) & touched:
            diff.outputs_dirty.append(name)
    diff.outputs_removed = [n for n in old.outputs if n not in new.outputs]

    for key in _DECLARED:
        a, b = getattr(old, key), getattr(new, key)
        if key == "params":
            a, b = ([_plain(d) for d in x or []] for x in (a, b))
        if _text(a) != _text(b):
            diff.declarations.append(key)
    return diff


# -- the report ----------------------------------------------------------------


def _listed(ids: list[str]) -> dict:
    out: dict = {"count": len(ids)}
    if ids:
        out["ids"] = ids[:LISTED]
    return out


def _changed_params(before: dict, after: dict) -> dict:
    return {
        key: [before.get(key), after.get(key)]
        for key in dict.fromkeys([*before, *after])
        if (key in before) != (key in after)
        or _text(before.get(key)) != _text(after.get(key))
    }


def _node_row(before: Node, after: Node) -> dict:
    row: dict = {"id": after.id, **where(after)}
    if before.op != after.op:
        row["op"] = [before.op, after.op]
    if before.inputs != after.inputs:
        row["inputs"] = [before.inputs, after.inputs]
    changed = _changed_params(before.params, after.params)
    if changed:
        row["params"] = changed
    return row


def _assert_row(before: dict, after: dict) -> dict:
    return {
        "id": after["id"], **where(after),
        "keys": _changed_params(_plain(before), _plain(after)),
    }


def _by_site(holders: list) -> list[dict]:
    """Entries counted by the recipe line that made them, the lines
    with the most first: one edited expression inside a loop reads as
    one row. Present when the entries are too many to list."""
    counts: dict[str, int] = {}
    for holder in holders:
        site = where(holder).get("site")
        if site is not None:
            counts[site] = counts.get(site, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"site": site, "count": n} for site, n in rows[:LISTED]]


def _by_caller(holders: list) -> list[dict]:
    """Entries made inside a function, counted by the outermost call
    they were reached through: where every edit goes through one
    helper, the helper's line holds them all and the call says which
    part changed. Empty where no entry came through a call."""
    counts: dict[str, int] = {}
    for holder in holders:
        via = where(holder).get("via")
        if via:
            counts[via[0]] = counts.get(via[0], 0) + 1
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"site": site, "count": n} for site, n in rows[:LISTED]]


def _asserts_section(added, removed, changed) -> dict:
    out = {
        "added": _listed([a["id"] for a in added]),
        "removed": _listed([a["id"] for a in removed]),
        "changed": {"count": len(changed)},
    }
    if changed:
        out["changed"]["rows"] = [_assert_row(a, b) for a, b in changed[:LISTED]]
    return out


def report(diff: AssetDiff, salted: Callable[[str], bool] = lambda op: False) -> dict:
    """One asset's change as ``build --diff`` reports it (DESIGN 6).
    ``salted`` says whether an op's streams are salted with the node
    id, where a new id is a new result."""
    out: dict = {}
    nodes: dict = {
        "added": _listed([n.id for n in diff.nodes_added]),
        "removed": _listed([n.id for n in diff.nodes_removed]),
        "changed": {"count": len(diff.nodes_changed)},
    }
    if diff.nodes_changed:
        nodes["changed"]["rows"] = [
            _node_row(a, b) for a, b in diff.nodes_changed[:LISTED]
        ]
    renamed = _renamed(diff) if diff.old is not None else []
    if renamed:
        nodes["renamed"] = {
            "count": len(renamed),
            "rows": [
                {
                    "from": a.id, "to": b.id, **where(b),
                    **({"stream_changed": True} if salted(b.op) else {}),
                }
                for a, b in renamed[:LISTED]
            ],
        }
    reached = diff.nodes_added + [n for _, n in diff.nodes_changed]
    if len(reached) > LISTED:
        nodes["by_site"] = _by_site(reached)
        callers = _by_caller(reached)
        if callers:
            nodes["by_caller"] = callers
    out["nodes"] = nodes

    def owned(entries, owner, pair=False):
        pick = (lambda e: e[1]) if pair else (lambda e: e)
        return [e for e in entries if (pick(e).get("owner") == "human") == (owner == "human")]

    asserts = _asserts_section(
        owned(diff.asserts_added, "agent"), owned(diff.asserts_removed, "agent"),
        owned(diff.asserts_changed, "agent", pair=True),
    )
    human = (
        owned(diff.asserts_added, "human"), owned(diff.asserts_removed, "human"),
        owned(diff.asserts_changed, "human", pair=True),
    )
    if any(human):
        asserts["human"] = _asserts_section(*human)
    out["asserts"] = asserts
    out["outputs"] = {
        "added": _listed(diff.outputs_added),
        "removed": _listed(diff.outputs_removed),
        "dirty": _listed(diff.outputs_dirty),
    }
    if diff.declarations:
        out["declarations"] = diff.declarations
    return out
