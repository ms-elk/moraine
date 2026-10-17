"""Parameter interaction tiers (DESIGN 7.2): classify an asset's public
knobs from the graph.

Three tiers, derived — never authored — from op-declared metadata plus
the declaration's position in the root graph:

- ``seed``: the bound param is in the op's ``SEED_PARAMS`` — an
  unordered stochastic stream selector, presented as a reroll control.
- ``factor``: the bound param is in the op's ``UNIFORM_PARAMS`` and the
  graph guarantees the promise holds at the asset level — the client
  may drive the named material's uniform live, with no evaluation.
- ``evaluation``: everything else, and every doubt. Only positive
  metadata plus the graph checks below can produce ``factor``; the
  default is the safe tier, so a misdeclared or unclassifiable knob can
  degrade to an extra evaluation but never to stale geometry.

The classifier runs on every schema build: it is a dict walk over the
root asset, and a text override can rename a material mid-session,
moving its knobs between tiers. The one fact the walk cannot see — the
names un-renamed references carry into the root — comes from an
evaluation (``reference_material_names``), handed in by the caller.
"""

from __future__ import annotations

from collections.abc import Iterable

from .. import ops
from ..core.asset import Asset
from ..errors import MoraineError
from .evaluator import _material_policy


def reference_material_names(results: dict, origins: dict) -> frozenset[str]:
    """Material names that arrive in the root asset through un-renamed
    references, read off one evaluation: ``results`` per
    ``Evaluation.results()`` and ``origins`` per
    ``Evaluation.material_origins()``. An output whose material arrives
    through references (non-empty ``refs``) ships the name its stats
    carry; an output this asset authors contributes nothing, its name
    being in the authored set already."""
    names: set[str] = set()
    for output, entry in results.items():
        if not (origins.get(output) or {}).get("refs"):
            continue
        name = (entry.get("stats") or {}).get("material")
        if isinstance(name, str):
            names.add(name)
    return frozenset(names)


def _authored_names(node, op) -> list[str]:
    """Material identities this node binds into the asset's own namespace
    (the knob panel's authored-identity scope, DESIGN 7.2): ``material``'s
    own name; a reference's rename targets."""
    author = getattr(op, "authors_material", None)
    if author is None or not author(node.params):
        return []
    if getattr(op, "IS_REFERENCE", False):
        rename = node.params.get("material_rename")
        if isinstance(rename, dict):
            return [str(v) for v in rename.values()]
        return []
    name = node.params.get("name")
    return [name] if isinstance(name, str) else []


def classify(
    sc: Asset, *, reference_names: Iterable[str] = ()
) -> dict[str, dict]:
    """``param_id -> {"tier": ...}`` (factor entries add ``"uniform":
    {"material", "channel"}``) for every declaration of ``sc``.

    ``factor`` requires, beyond the op's ``UNIFORM_PARAMS`` declaration:

    - The declared [min, max] sits inside the channel's valid range
      (``ops.UNIFORM_RANGES``). The factor tier skips evaluation, so the
      declaration is the only validation a scrub gets: a wider range
      would let an unevaluated, op-invalid value poison the overlay and
      reach the asset file through Accept.
    - The bound node's material name is unique among the identities the
      root asset authors (``material`` nodes and ``asset_ref`` rename
      targets alike) and absent from ``reference_names`` — the names
      un-renamed references carry into the root, which the graph walk
      cannot see (they live in the referenced files) and the caller
      reads off an evaluation (``reference_material_names``). The
      client applies the uniform by material name across every loaded
      output, so a shared name would let one knob's preview repaint a
      sibling output that a real evaluation of the edit would not
      touch — a kit binding the root's name with identical content is
      the case nothing else catches: ``validate`` passes it, and an
      Accept of the scrubbed value would leave the name bound to two
      contents. Without ``reference_names`` (a caller holding no
      evaluation) the check is graph-only, and such a knob classifies
      ``factor``.
    - Every downstream consumer reachable over carried inputs holds
      ``MATERIAL_POLICY`` ``"preserved"`` or ``"dropped"`` — vacuous
      under today's op contract (materials never feed computed bytes; a
      dropping consumer merely makes the knob inert on that path, which
      a by-name client update renders correctly by updating nothing),
      but the walk demotes automatically the day a material-*consuming*
      policy value appears (a bake reading factors into pixels).

    An asset holding any unresolvable op classifies every knob as
    ``evaluation``: the deciding facts — the authored-identity set, the
    consumer edges — cannot be traced past an op the registry cannot
    resolve, so no ``factor`` promise can be kept. Classification must
    not raise on an asset evaluation would reject.
    """
    authored: list[str] = []
    consumers: dict[str, list[str]] = {}
    mods: dict[str, object] = {}
    for node in sc.nodes.values():
        try:
            op = ops.get(node.op)
        except MoraineError:
            return {decl["id"]: {"tier": "evaluation"} for decl in sc.params or []}
        mods[node.id] = op
        authored.extend(_authored_names(node, op))
        carried = getattr(op, "CARRIED_INPUTS", None)
        for idx, src in enumerate(node.inputs):
            if carried is not None and idx not in carried:
                continue
            consumers.setdefault(src, []).append(node.id)
    # One entry per reference-borne name: a root identity it matches
    # becomes a pair, however many outputs carry it.
    authored.extend(set(reference_names))

    out: dict[str, dict] = {}
    for decl in sc.params or []:
        out[decl["id"]] = _classify_one(sc, decl, mods, authored, consumers)
    return out


def _classify_one(
    sc: Asset,
    decl: dict,
    mods: dict[str, object],
    authored: list[str],
    consumers: dict[str, list[str]],
) -> dict:
    if "node" not in decl:
        # A knob the build did not record as reaching one node param
        # whole moves the graph only through a rebuild (DESIGN 7.2).
        return {"tier": "evaluation"}
    node = sc.nodes[decl["node"]]
    op = mods.get(node.id)
    if op is None:
        return {"tier": "evaluation"}
    param = decl["param"]
    if param in getattr(op, "SEED_PARAMS", frozenset()):
        return {"tier": "seed"}
    channel = getattr(op, "UNIFORM_PARAMS", {}).get(param)
    if channel is None:
        return {"tier": "evaluation"}
    lo, hi = ops.UNIFORM_RANGES[channel]
    if not (
        isinstance(decl.get("min"), (int, float))
        and isinstance(decl.get("max"), (int, float))
        and lo <= decl["min"]
        and decl["max"] <= hi
    ):
        # The declared range must sit inside the channel's valid range:
        # the factor tier skips evaluation, so the declaration is the
        # only validation a scrub gets — a range the op could reject
        # would let an unevaluated value poison the overlay and Accept
        # write an unevaluable asset file. Demote instead: the safe tier
        # evaluates, and evaluation rejects the value with the op's own
        # actionable error.
        return {"tier": "evaluation"}
    name = node.params.get("name")
    if not isinstance(name, str) or authored.count(name) != 1:
        return {"tier": "evaluation"}
    stack = [node.id]
    seen: set[str] = set()
    while stack:
        for consumer in consumers.get(stack.pop(), ()):
            if consumer in seen:
                continue
            seen.add(consumer)
            cop = mods.get(consumer)
            if cop is None or _material_policy(cop) not in ("preserved", "dropped"):
                return {"tier": "evaluation"}
            stack.append(consumer)
    return {"tier": "factor", "uniform": {"material": name, "channel": channel}}
