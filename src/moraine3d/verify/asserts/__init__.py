"""Assert vocabulary, evaluation side (DESIGN 3, 5.2): one kind, one
module, mirroring ``core/asserts`` kind for kind (a test keeps the two
registries equal).

Each kind module exposes:
- ``check(entry, a, ctx)``: fill the report entry for one parsed
  assert ``a`` — at least ``ok``, and on a red entry ``fix``, the
  shortest path to a fix (DESIGN 8.2) — reading the asset through the
  ``common.Context`` memos. An entry the kind cannot evaluate (an image
  output, an open surface, a missing tag) FAILS with the reason: an
  unevaluatable constraint reporting ok would be the silent-pick
  failure mode the vocabulary exists to prevent.
- ``ECHOED`` (optional): the entry keys the kind echoes into its header
  and their defaults (``common.REQUIRED`` for a key the parser
  guarantees), so a weakened setting is visible in the report, not
  just the file diff.
- ``report_warnings(entries)`` (optional): advisory top-level rows
  the kind raises over a finished entry list.
- ``cedes(a, ctx)`` (optional): what one entry of the kind polices
  instead of the built-in guard — ``(cross-output pairs, outputs whose
  copy pairs)`` — collected over the whole list into ``ctx.ceded``
  before any check runs, so the guard reads one roster and a ceding
  kind owns its own rule.

The header every entry carries — ``assert``, ``kind``, ``builtin``,
the subject (``outputs`` with the group resolution under ``members``,
or ``faces``), the echoed keys, and ``owner`` — is written here, so
the report reads the same across kinds and a weakened human-owned
assertion is visible in the report, not just the file diff. Kind
semantics are documented in the shipped reference (``moraine docs``)
and pinned by golden tests (DESIGN 5.2, 10).
"""

from __future__ import annotations

from ...core.mesh import Value
from ...core.sites import where
from . import clearance, common, contact, count, embedded, extent, mated, no_overlap
from .common import HONESTY_KEYS

__all__ = ["HONESTY_KEYS", "check_asserts", "warnings"]

KINDS = {
    "clearance": clearance,
    "contact": contact,
    "count": count,
    "embedded": embedded,
    "extent": extent,
    "mated": mated,
    "no_overlap": no_overlap,
}


def check_asserts(
    values: dict[str, Value], asserts: list[dict], proto_manifold=None,
    *, ref_embeds: dict | None = None, groups: dict[str, list[str]] | None = None,
    copy_bounds=None,
) -> list[dict]:
    """Asset assertions (DESIGN 5.2): declarative constraints checked
    deterministically, gating like tier-1 checks — one entry per
    parsed assert, in order, each filled by its kind's module.

    ``asserts`` is the parsed list with the built-in entries injected
    (``validate_outputs`` does so on every run — overlap is checked by
    default, not by authorship); ``proto_manifold`` and ``copy_bounds``
    are the caller's memos; ``ref_embeds``
    (``Evaluation.embedded_via_refs()``) the cede coverage arriving
    through references, folded into ``ctx.ceded`` with the kinds' own;
    ``groups`` the asset's root ``groups`` object
    (DESIGN 4.1) — a name an entry takes where it takes an output
    resolves to the union of the group's members, and the entry echoes
    the resolution under ``members``. A kind the parser accepts but no
    module here evaluates raises rather than falling through to
    another kind's semantics.
    """
    ctx = common.Context(
        values, proto_manifold=proto_manifold, copy_bounds=copy_bounds, groups=groups
    )
    pairs: set[frozenset] = set()
    self_outputs: set[str] = set()
    for a in asserts:
        hook = getattr(KINDS.get(a["kind"]), "cedes", None)
        if hook is not None:
            p, s = hook(a, ctx)
            pairs |= p
            self_outputs |= s
    if ref_embeds:
        pairs.update(frozenset(p) for p in ref_embeds.get("pairs", []))
        self_outputs.update(ref_embeds.get("self", []))
    ctx.ceded = (pairs, self_outputs)
    entries: list[dict] = []
    for a in asserts:
        mod = KINDS.get(a["kind"])
        if mod is None:
            # A kind the parser accepts but no module here evaluates must
            # never fall through to another kind's semantics — that is the
            # silent-pick class this vocabulary exists to prevent.
            raise ValueError(
                f"unhandled assert kind '{a['kind']}': add its module to "
                f"verify.asserts"
            )
        entry: dict = {"assert": a["id"], "kind": a["kind"]}
        if a.get("builtin"):
            entry["builtin"] = True  # not in the file: the reader must know
        if "faces" in a:  # mated's two specs, extent's face-set form
            entry["faces"] = a["faces"]
        else:
            entry["outputs"] = a["outputs"]
            named = (
                [n for n in a["outputs"] if n in ctx.groups] if a["outputs"] != "all" else []
            )
            if named:
                # The resolution rides the entry: a reader sees which
                # outputs the group stood for when this report was made.
                entry["members"] = {n: list(ctx.groups[n]) for n in named}
        for key, default in getattr(mod, "ECHOED", {}).items():
            entry[key] = a[key] if default is common.REQUIRED else a.get(key, default)
        if "from_param" in a:
            # The knob a threshold follows (DESIGN 4.8): an accepted
            # value moves the requirement, and the entry says which.
            entry["from_param"] = dict(a["from_param"])
        entry["owner"] = a["owner"]  # required by the parser; KeyError = unparsed
        entry.update(where(a))  # the recipe line that declared it
        mod.check(entry, a, ctx)
        entries.append(entry)
    for e in entries:
        common.settle_floor(e)
    return entries


def warnings(entries: list[dict]) -> list[dict]:
    """The advisory rows the kinds raise over a finished entry list —
    each kind's ``report_warnings`` hook, in registry order."""
    out: list[dict] = []
    for mod in KINDS.values():
        hook = getattr(mod, "report_warnings", None)
        if hook is not None:
            out.extend(hook(entries))
    return out
