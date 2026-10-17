"""The graph change of a build's closure (DESIGN 6).

``core.irdiff`` compares two IRs of one asset. A build makes a closure
of them, and a reference ties each asset to the one it reads, so this
module pairs the assets of two closures and carries a change across
the references: an output of a referenced asset that became dirty
makes every output downstream of its reference dirty.

An asset is paired with the one the same reference read before. A
variant is named by its overrides, so a changed override opens another
file, and pairing by the file's name would report an asset removed and
an asset added where one value moved.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import ops
from ..core import irdiff
from ..core import asset as asset_mod
from ..core.irdiff import AssetDiff
from ..core.asset import Asset
from ..core.brief import LISTED
from ..errors import MoraineError

# What a path held before the build, or None.
Base = Callable[[Path], "Asset | None"]


def from_disk(path: Path) -> Asset | None:
    """The IR a file holds, or None where there is none to read. An IR
    that cannot be read is no base to compare against."""
    try:
        return asset_mod.load(path) if Path(path).is_file() else None
    except MoraineError:
        return None


def _target(sc: Asset, node) -> tuple[Path, str] | None:
    """The IR a reference node reads and the output it takes, or None
    for a reference that resolves to none."""
    op = ops.get(node.op)
    try:
        spec = op.ref_params(node.params)
        return op.resolve_ref(sc.path, spec), spec.output
    except MoraineError:
        return None


def closure_diff(target: Path, irs: dict[Path, Asset], base: Base,
                 base_target: Path | None = None) -> dict[Path, AssetDiff]:
    """The change of every asset the build of ``target`` made, by the
    path its IR is written at. ``irs`` holds the build's IRs,
    ``base`` answers with what a path held before, and ``base_target``
    names the IR to compare the target against where that is not the
    target's own path. Under a named base an asset an existing
    reference read, whose IR the base does not hold, is left uncompared:
    the revision named is the one to read, and it does not say what
    that asset was. An asset only a new reference reads is new."""
    done: dict[Path, AssetDiff] = {}
    named = base_target is not None

    def visit(path: Path, old: Asset | None, fresh: bool = False) -> AssetDiff:
        if path in done:
            return done[path]
        new = irs[path]
        if old is None and named and not fresh:
            done[path] = AssetDiff(new=new, old=None, uncompared=True)
            return done[path]
        moved: set[str] = set()
        # Seated before the references are walked, so an asset that
        # reaches itself through a cycle the build refused elsewhere
        # cannot recurse.
        done[path] = AssetDiff(new=new, old=old)
        for node in new.nodes.values():
            if not _is_reference(node.op):
                continue
            read = _target(new, node)
            if read is None or read[0] not in irs:
                continue
            child_path, output = read
            before = old.nodes.get(node.id) if old is not None else None
            old_child = None
            fresh = False
            if before is not None and _is_reference(before.op):
                was = _target(old, before)
                old_child = base(was[0]) if was is not None else None
            elif old is not None and named:
                fresh = True  # the reference is this edit's: so is its asset
            elif old is not None:
                old_child = base(child_path)
            child = visit(child_path, old_child, fresh)
            if child.uncompared:
                continue
            if output in child.outputs_dirty or output in child.outputs_added:
                moved.add(node.id)
        done[path] = irdiff.asset_diff(old, new, moved)
        return done[path]

    visit(target, base(base_target if base_target is not None else target))
    # An asset of the closure no reference of the target reaches on the
    # way (a sibling's own sub-asset) is reached through its consumer
    # above; whatever is left has no consumer in this build.
    for path in irs:
        if path not in done:
            visit(path, None if named else base(path))
    return done


def _is_salted(op: str) -> bool:
    try:
        return bool(getattr(ops.get(op), "USES_NODE_ID", False))
    except MoraineError:
        return False


def _is_reference(op: str) -> bool:
    try:
        return bool(getattr(ops.get(op), "IS_REFERENCE", False))
    except MoraineError:
        return False


def counts(diffs: dict[Path, AssetDiff]) -> dict:
    """The closure's counts, summed (DESIGN 6): added, removed, and
    changed, for nodes and for asserts. An asset with nothing to compare
    against counts as added."""
    total = {"nodes": [0, 0, 0], "asserts": [0, 0, 0]}
    for diff in diffs.values():
        for key, row in diff.counts.items():
            total[key] = [a + b for a, b in zip(total[key], row)]
    return total


def report(diffs: dict[Path, AssetDiff], shown: Callable[[Path], str],
           base: str) -> dict:
    """``build --diff``'s report (DESIGN 6): the assets that changed,
    each with the IR it was compared against, and how many did not."""
    assets = {}
    unchanged = 0
    uncompared: list[str] = []
    for path in sorted(diffs, key=lambda p: p.as_posix()):
        diff = diffs[path]
        if diff.uncompared:
            uncompared.append(shown(path))
            continue
        if diff.empty:
            unchanged += 1
            continue
        against = diff.old.path if diff.old is not None else None
        assets[shown(path)] = {
            "base": shown(against) if against is not None else None,
            **irdiff.report(diff, _is_salted),
        }
    out = {"base": base, "assets": assets, "unchanged": unchanged}
    if uncompared:
        out["not_compared"] = {"count": len(uncompared), "assets": uncompared[:LISTED]}
    return out
