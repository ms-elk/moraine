"""The IR (DESIGN 4, 4.1).

The IR is a JSON operator graph, the deterministic expansion of a
recipe and the input to everything past the build. Canonical
serialization is one node per line with fixed key order, so a build's
output reads and diffs line by line.

Day-One Invariants honored here (DESIGN 11):
- ``build_version`` stamped at the root by the build that wrote the file.
- Deterministic unique node ``id``; all references are by id.
- Named outputs (``outputs``: name -> node id).
- Reserved name ``time`` (the animation input to come; refused as a
  public parameter id so the name is free when it lands).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import BuildError, AssetError
from . import asserts as vocab
from .atomic import ReplaceError, atomic_replace
from .paths import PORTABLE_NAME_FRAGMENT
from .sites import where

# Output names become cache/export filenames and engine object names, so
# they are restricted to a portable ASCII identifier form and must be
# unique case-insensitively (Windows/macOS filesystems fold case).
OUTPUT_NAME_RE = re.compile(rf"^{PORTABLE_NAME_FRAGMENT}$")

# Fixed key orders for canonical serialization (DESIGN 4.1).
ROOT_KEY_ORDER = (
    "build_version", "nodes", "params", "outputs", "groups", "checks",
    "asset_checks", "asserts", "budgets", "provenance",
)
NODE_KEY_ORDER = ("id", "op", "inputs", "params", "site", "via")
# Accepted key sets derive from the serialization orders (the one
# vocabulary literal per key family): a key the loader does not know is
# rejected, never silently carried or dropped by dumps — a typo'd
# "param" would otherwise evaluate the node with every default.
ROOT_KEYS = frozenset(ROOT_KEY_ORDER)
NODE_KEYS = frozenset(NODE_KEY_ORDER)

# Public parameter declarations (DESIGN 4.1, 7.2): the root "params"
# list marks an asset's public knobs. Each entry carries an immutable
# ``id`` (the param_id, so UI bindings and asset_ref overrides survive
# renames) and the ``value`` the build resolved for it. ``node`` and
# ``param``, present together or not at all, record that the value
# reached exactly that one node param whole, so patching the param
# equals a rebuild; the two values must agree.
# The key set derives from the serialization order: one vocabulary
# literal per key family, so a new declaration key cannot be accepted by
# the loader yet silently dropped by dumps (_param_json iterates this).
PARAM_KEY_ORDER = (
    "id", "value", "from", "default", "mismatch", "node", "param", "label",
    "min", "max", "step", "site", "via",
)
# Where a knob's value came from when it is not the recipe's default
# (DESIGN 4.8): the project's ``params.json``, or a consumer.
PARAM_SOURCES = ("accepted", "override")
PARAM_DECL_KEYS = frozenset(PARAM_KEY_ORDER)
PARAM_ID_RE = re.compile(rf"^{PORTABLE_NAME_FRAGMENT}$")
# Reserved names (DESIGN 4.1): ``time`` is the animation input to come;
# refusing it as a param id today keeps the name free without carrying
# an unused field in every file.
RESERVED_PARAM_IDS = frozenset({"time"})

# Per-output validation policy keys accepted under the root "checks" object
# (DESIGN 4.1). ``not_sunk: false`` waives the below-ground check,
# ``not_enclosed: false`` the full-enclosure check, for that output —
# the waiver lives in the asset so intent is documented and survives edits.
# Waivers only: the sole accepted value is ``false`` (an opt-in spelling
# stays reserved).
KNOWN_CHECKS = frozenset({"not_sunk", "not_enclosed"})

# Asset-level validation policy keys accepted under the root
# "asset_checks" object (DESIGN 4.1): checks over the outputs as a
# whole, same waiver-only spelling. ``grounded: false`` waives the
# asset-level ground-contact check and documents support outside the
# asset (a hanging fixture, a parts-bin sub-asset).
KNOWN_ASSET_CHECKS = frozenset({"grounded"})

# Asset-declared budgets (DESIGN 4.1/9.1): acceptance criteria
# the human owns, kept in the reviewable recipe. ``materials`` caps the
# distinct material identities the asset ships (content identities over
# every mesh output, reference-borne identities included, the neutral
# default counted once); declared budgets gate ``validate`` like asserts,
# undeclared ones are not checked at all — there is no built-in default.
KNOWN_BUDGETS = frozenset({"materials"})


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)  # upstream node ids, positional
    params: dict = field(default_factory=dict)
    # The recipe file and line that created the node (DESIGN 4.1). The
    # build writes one on every node; an IR written by hand may omit it.
    # Location, not content: it joins no cache key.
    site: str | None = None
    # The calls the node was reached through, outermost first, for a
    # node made inside a function (DESIGN 4.8). Location like the site.
    via: list[str] | None = None


@dataclass
class Asset:
    # The stamp of the build that wrote the document (DESIGN 4.1); the
    # identity check against the running build happens where an IR is
    # opened from disk, above this layer.
    build_version: str
    nodes: dict[str, Node]  # id -> Node, insertion-ordered
    outputs: dict[str, str]  # output name -> node id
    # Output groups (DESIGN 4.1): group name -> member output names; a
    # name an assert takes wherever it takes an output (DESIGN 5.2).
    groups: dict[str, list[str]] | None = None
    # Set by load(): absolute file path; None for in-memory assets (loads()).
    # asset_ref resolution needs it (DESIGN 4.6).
    path: Path | None = None
    # Per-output validation policy: output name -> {check name -> False}
    # (KNOWN_CHECKS). Ignored by evaluation; consumed by validate.
    checks: dict[str, dict] | None = None
    # Asset-level validation policy: check name -> False
    # (KNOWN_ASSET_CHECKS). Ignored by evaluation; consumed by validate.
    asset_checks: dict | None = None
    # Public parameter declarations (PARAM_DECL_KEYS), in UI order.
    # Ignored by evaluation; consumed by the parameter panel / overrides.
    params: list[dict] | None = None
    # Asset assertions (asserts.KEYS). Ignored by evaluation; checked and
    # gated by validate — including through asset_ref: every consuming
    # root's validate checks a sub-asset's asserts once per override
    # identity (append_ref_asserts, DESIGN 4.6).
    asserts: list[dict] | None = None
    # Asset-declared budgets (KNOWN_BUDGETS): budget name -> positive int.
    # Ignored by evaluation; gated by validate (DESIGN 9.1).
    budgets: dict[str, int] | None = None
    # Vendoring provenance (DESIGN 4.6): where a vendored sub-asset came
    # from ({"source": ..., "content_hash": ...}). Ignored by evaluation.
    provenance: dict | None = None


def param_kind(value) -> str | None:
    """UI kind of a bindable node-param value; None = not bindable.

    Numbers must be finite: an inf value (reachable via an overflow
    literal like 1e999, which json parses without hitting
    ``parse_constant``) would produce an invalid-JSON param_schema
    payload downstream.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number" if math.isfinite(value) else None
    if isinstance(value, str):
        return "text"
    if (
        isinstance(value, list)
        and 2 <= len(value) <= 4
        and all(
            isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)
            for c in value
        )
    ):
        return "vector"
    return None


def output_closure(sc: Asset, name: str):
    """Yield each node in the dependency closure of a named output, once.

    Order is unspecified. The graph is validated at load (every referenced
    id exists, acyclic), so the walk needs no missing-id or cycle
    handling; ``asset_ref`` nodes are leaves (zero inputs), so the closure
    never crosses into a referenced asset.
    """
    stack, seen = [sc.outputs[name]], set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = sc.nodes[nid]
        stack.extend(node.inputs)
        yield node


# Node and assert ids are paths (DESIGN 4.8): portable names joined by
# the scope separator, so an id never starts or ends with it.
ID_SEPARATOR = "/"
ID_GRAMMAR = (
    f"segments matching {PORTABLE_NAME_FRAGMENT} joined by '{ID_SEPARATOR}'"
)


def id_segments(value: object) -> list[str] | None:
    """The segments of an id, or None when ``value`` is not one:
    a string of portable names joined by the separator."""
    if not isinstance(value, str) or not value:
        return None
    segments = value.split(ID_SEPARATOR)
    if not all(PARAM_ID_RE.fullmatch(s) for s in segments):
        return None
    return segments


def _check_site(owner: str, entry: dict) -> str | None:
    site = entry.get("site")
    if site is not None and (not isinstance(site, str) or not site):
        raise AssetError(
            "bad_site",
            f"{owner} 'site' must be a non-empty string, the recipe file "
            f"and line the build recorded",
        )
    return site


def _check_via(owner: str, entry: dict) -> list[str] | None:
    via = entry.get("via")
    if via is None:
        return None
    if (
        not isinstance(via, list) or not via
        or not all(isinstance(s, str) and s for s in via)
        or entry.get("site") is None
    ):
        raise AssetError(
            "bad_site",
            f"{owner} 'via' must be a non-empty list of call sites beside "
            f"a 'site', the calls the build recorded on the way to it",
        )
    return list(via)


def _check_acyclic(nodes: dict[str, Node]) -> None:
    """Reject cyclic graphs with the offending node named (actionable).

    Iterative DFS: a deep-but-valid linear chain must not hit the
    interpreter recursion limit (a raw RecursionError instead of an
    AssetError) — chain depth is author-controlled and unbounded.
    """
    state: dict[str, int] = {}  # 0 = visiting (on path), 1 = done
    for root in nodes:
        if state.get(root) == 1:
            continue
        # (node, entered): entered=False is the pre-visit, True the
        # post-visit that retires the node from the current path.
        stack: list[tuple[str, bool]] = [(root, False)]
        path: list[str] = []
        while stack:
            nid, entered = stack.pop()
            if entered:
                state[nid] = 1
                path.pop()
                continue
            s = state.get(nid)
            if s == 1:
                continue
            if s == 0:
                cycle = path[path.index(nid):] + [nid]
                raise AssetError(
                    "cycle", f"node graph has a cycle: {' -> '.join(cycle)}", cycle=cycle
                )
            state[nid] = 0
            path.append(nid)
            stack.append((nid, True))
            for ref in nodes[nid].inputs:
                stack.append((ref, False))


def _reject_nonfinite(literal: str) -> None:
    # Python's json accepts NaN/Infinity literals by default; an asset with
    # one would evaluate to poisoned geometry that only surfaces far
    # downstream, so reject at the door.
    raise AssetError(
        "bad_json", f"non-finite number literal '{literal}' is not valid asset JSON"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.loads resolves duplicate object keys last-wins before any
    # validation can see them — a duplicated output name would silently
    # drop a binding while validate stays green. Never silent (P6).
    obj: dict = {}
    for k, v in pairs:
        if k in obj:
            raise AssetError(
                "duplicate_key",
                f"duplicate JSON key '{k}' in one object - the later value "
                f"would silently replace the earlier one; remove one",
            )
        obj[k] = v
    return obj


def check_output(name, ref, nodes, folded: dict[str, str]) -> None:
    """One ``outputs`` entry: the name's form, its case-folded
    uniqueness (``folded`` collects the names seen), and its node."""
    if not isinstance(name, str) or not OUTPUT_NAME_RE.fullmatch(name):
        raise AssetError(
            "bad_output_name",
            f"output name '{name}' must match {PORTABLE_NAME_FRAGMENT} - "
            "it becomes cache/export filenames and engine object names",
        )
    other = folded.setdefault(name.lower(), name)
    if other != name:
        raise AssetError(
            "output_name_collision",
            f"output names '{other}' and '{name}' collide case-insensitively "
            "(cache/export files live on case-folding filesystems)",
        )
    if not isinstance(ref, str) or ref not in nodes:
        raise AssetError("bad_output_ref", f"output '{name}' references unknown node '{ref}'")


def check_group(gname, members, raw_outputs, folded: dict[str, str]) -> None:
    """One ``groups`` entry (DESIGN 4.1): a piece built from several
    outputs named once, so an assert can take the piece where it takes
    an output (DESIGN 5.2). Group and output names share one namespace
    — an assert resolves a name without a second key saying which kind
    it is."""
    if not isinstance(gname, str) or not OUTPUT_NAME_RE.fullmatch(gname):
        raise AssetError(
            "bad_group",
            f"group name '{gname}' must match {PORTABLE_NAME_FRAGMENT} "
            f"(the same rule as output names)",
        )
    other = folded.setdefault(gname.lower(), gname)
    if other != gname or gname in raw_outputs:
        raise AssetError(
            "bad_group",
            f"group '{gname}' collides with output '{other}': groups and "
            f"outputs share one namespace (an assert names either); "
            f"rename the group",
        )
    if (
        not isinstance(members, list) or len(members) < 2
        or not all(isinstance(m, str) for m in members)
        or len(set(members)) != len(members)
    ):
        raise AssetError(
            "bad_group",
            f"group '{gname}' must list 2 or more distinct output names "
            f"(a group of one is that output; write its name)",
        )
    for m in members:
        if m not in raw_outputs:
            raise AssetError(
                "bad_group",
                f"group '{gname}' names unknown output '{m}'; asset "
                f"outputs: {sorted(raw_outputs)} (groups hold outputs, "
                f"never other groups)",
            )


def check_waiver(name, policy, raw_outputs) -> None:
    """One ``checks`` entry: an output's waivers."""
    if name not in raw_outputs:
        raise AssetError(
            "checks_unknown_output",
            f"'checks' names unknown output '{name}'; "
            f"asset outputs: {sorted(raw_outputs)}",
        )
    if not isinstance(policy, dict):
        raise AssetError(
            "bad_checks", f"checks['{name}'] must be an object like {{\"not_sunk\": false}}"
        )
    for key, val in policy.items():
        if key == "grounded":
            raise AssetError(
                "unknown_check",
                f"checks['{name}']: per-output 'grounded' was split; "
                f"an elevated output passes by default (delete this "
                f"waiver), a deliberately below-ground output is "
                f'waived with {{"not_sunk": false}}, and whole-asset '
                f'grounding is waived under root "asset_checks": '
                f'{{"grounded": false}}',
            )
        if key not in KNOWN_CHECKS:
            raise AssetError(
                "unknown_check",
                f"checks['{name}'] has unknown check '{key}'; "
                f"known: {sorted(KNOWN_CHECKS)}",
            )
        if val is not False:
            raise AssetError(
                "bad_check_value",
                f"checks['{name}'].{key} must be false (waivers only; "
                f"delete the entry to keep the check on)",
            )


def check_asset_check(key, val) -> None:
    """One ``asset_checks`` entry."""
    if key not in KNOWN_ASSET_CHECKS:
        raise AssetError(
            "unknown_check",
            f"'asset_checks' has unknown check '{key}'; "
            f"known: {sorted(KNOWN_ASSET_CHECKS)}",
        )
    if val is not False:
        raise AssetError(
            "bad_check_value",
            f"asset_checks.{key} must be false (waivers only; "
            f"delete the entry to keep the check on)",
        )


def check_budget(key, val) -> None:
    """One ``budgets`` entry."""
    if key not in KNOWN_BUDGETS:
        raise AssetError(
            "unknown_budget",
            f"budgets has unknown budget '{key}'; known: {sorted(KNOWN_BUDGETS)}",
        )
    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise AssetError(
            "bad_budget_value",
            f"budgets.{key} must be a positive integer (the maximum the "
            f"asset may use; validate fails above it)",
        )


def check_assert(label: str, a, raw_outputs, group_names: dict, seen_ids: set[str]) -> None:
    """One ``asserts`` entry; ``label`` names it before its id is
    known (``asserts[3]``) and ``seen_ids`` collects the ids seen."""
    if not isinstance(a, dict):
        raise AssetError("bad_assert", f"{label} must be an object")
    kind = a.get("kind")
    mod = vocab.KINDS.get(kind) if isinstance(kind, str) else None
    if mod is not None and not getattr(mod, "AUTHORABLE", True):
        # Ahead of the key and id gates: an entry of the retired
        # kind carries that kind's keys and often its natural id,
        # and its one fix is removal — the generic gates would
        # send the author through two edits before saying so.
        raise AssetError("bad_assert_kind", f"{label}: {mod.REFUSAL}")
    unknown = set(a) - vocab.KEYS
    if unknown:
        raise AssetError(
            "bad_assert",
            f"{label} has unknown key(s) {sorted(unknown)}; "
            f"accepted: {sorted(vocab.KEYS)}",
        )
    aid = a.get("id")
    if id_segments(aid) is None:
        raise AssetError(
            "bad_assert_id", f"{label} needs a string 'id' of {ID_GRAMMAR}",
        )
    if aid in seen_ids:
        raise AssetError("duplicate_assert_id", f"duplicate assert id '{aid}'")
    seen_ids.add(aid)
    if aid in vocab.RESERVED_IDS:
        raise AssetError(
            "bad_assert_id",
            f"assert id '{aid}' is reserved for the built-in '{aid}' "
            f"assert's report entry; pick another id",
        )
    if mod is None:
        raise AssetError(
            "bad_assert_kind",
            f"assert '{aid}' 'kind' must be one of "
            f"{sorted(vocab.AUTHORABLE_KINDS)}",
        )
    _check_site(f"assert '{aid}'", a)
    _check_via(f"assert '{aid}'", a)
    follows = a.get("from_param")
    if follows is not None and not (
        isinstance(follows, dict) and follows
        and all(
            k in vocab.THRESHOLDS and k in a
            and isinstance(v, str) and PARAM_ID_RE.fullmatch(v)
            for k, v in follows.items()
        )
    ):
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' 'from_param' names the knob each of its "
            f"thresholds was taken from, as in {{\"max\": \"<param_id>\"}}, "
            f"over the keys {list(vocab.THRESHOLDS)} the entry holds",
        )
    for key in sorted(set(a) - vocab.COMMON):
        if key not in mod.KEYS:
            raise AssetError(
                "bad_assert", f"assert '{aid}': {vocab.key_refusal(kind, key)}"
            )
    mod.validate_entry(aid, a, raw_outputs, group_names)
    if a.get("owner") not in vocab.OWNERS:
        # Required, never defaulted: an omitted owner defaulting to
        # "agent" would silently make a forgotten human requirement
        # agent-editable (DESIGN 5.2 ownership).
        raise AssetError(
            "bad_assert",
            f"assert '{aid}' needs 'owner': \"human\" (a requirement "
            f"the human stated) or \"agent\" (the agent's own "
            f"invariant)",
        )


def check_param(
    label: str, decl, nodes, seen_ids: set[str],
    seen_bindings: set[tuple[str, str]],
) -> None:
    """One ``params`` entry; ``label`` names it before its id is known
    and the two sets collect the ids and the bindings seen."""
    if not isinstance(decl, dict):
        raise AssetError("bad_param_decl", f"{label} must be an object")
    unknown = set(decl) - PARAM_DECL_KEYS
    if unknown:
        raise AssetError(
            "bad_param_decl",
            f"{label} has unknown key(s) {sorted(unknown)}; "
            f"accepted: {sorted(PARAM_DECL_KEYS)}",
        )
    pid = decl.get("id")
    if not isinstance(pid, str) or not PARAM_ID_RE.fullmatch(pid):
        raise AssetError(
            "bad_param_id",
            f"{label} needs a string 'id' matching "
            f"{PORTABLE_NAME_FRAGMENT} - the immutable param_id",
        )
    if pid in RESERVED_PARAM_IDS:
        raise AssetError(
            "reserved_param_id",
            f"param id '{pid}' is reserved (the animation input to "
            f"come); choose another name",
        )
    if pid in seen_ids:
        raise AssetError("duplicate_param_id", f"duplicate param id '{pid}'")
    seen_ids.add(pid)
    _check_site(f"param '{pid}'", decl)
    _check_via(f"param '{pid}'", decl)
    if "value" not in decl:
        raise AssetError(
            "bad_param_decl",
            f"param '{pid}' needs 'value', the value the build "
            f"resolved for the knob",
        )
    kind = param_kind(decl["value"])
    if kind is None:
        raise AssetError(
            "bad_param_value",
            f"param '{pid}' holds a value the parameter UI cannot "
            f"represent; a knob is a finite number, a boolean, a "
            f"string, or a 2-4 finite-number vector",
        )
    if ("node" in decl) != ("param" in decl):
        raise AssetError(
            "bad_param_decl",
            f"param '{pid}' carries one of 'node' and 'param'; the "
            f"two name one node param together or are both absent",
        )
    if "node" in decl:
        node_id = decl["node"]
        if not isinstance(node_id, str) or node_id not in nodes:
            raise AssetError(
                "bad_param_node",
                f"param '{pid}' references unknown node '{node_id}'",
            )
        pname = decl["param"]
        if not isinstance(pname, str) or pname not in nodes[node_id].params:
            raise AssetError(
                "bad_param_ref",
                f"param '{pid}' binds '{pname}' which is not present "
                f"in node '{node_id}' params",
            )
        if (node_id, pname) in seen_bindings:
            raise AssetError(
                "duplicate_param_binding",
                f"param '{pid}' binds '{node_id}.{pname}', which "
                f"another declaration already binds - two controls "
                f"fighting over one value; remove one",
            )
        seen_bindings.add((node_id, pname))
        bound = nodes[node_id].params[pname]
        if param_kind(bound) != kind or bound != decl["value"]:
            raise AssetError(
                "param_value_mismatch",
                f"param '{pid}' holds {decl['value']!r} while "
                f"'{node_id}.{pname}' holds {bound!r}; the entry "
                f"records that the knob's value reached that param "
                f"whole, so the two agree",
            )
    if "from" in decl and decl["from"] not in PARAM_SOURCES:
        raise AssetError(
            "bad_param_decl",
            f"param '{pid}' 'from' must be one of {list(PARAM_SOURCES)}, "
            f"and is absent for the recipe's default",
        )
    if ("default" in decl) != ("from" in decl) or (
        "default" in decl and param_kind(decl["default"]) != kind
    ):
        raise AssetError(
            "bad_param_decl",
            f"param '{pid}' 'default' is the recipe's value of the knob's "
            f"kind, present exactly where 'from' names another source",
        )
    if "mismatch" in decl and param_kind(decl["mismatch"]) is None:
        raise AssetError(
            "bad_param_decl",
            f"param '{pid}' 'mismatch' holds the accepted value the knob "
            f"could not take, itself a value some knob could",
        )
    label_text = decl.get("label")
    if label_text is not None and not isinstance(label_text, str):
        raise AssetError("bad_param_decl", f"param '{pid}' 'label' must be a string")
    for key in ("min", "max", "step"):
        # Presence, not truthiness: an explicit null must fail here,
        # not as a TypeError in the range comparisons below.
        if key not in decl:
            continue
        v = decl[key]
        if (
            not isinstance(v, (int, float)) or isinstance(v, bool)
            or not math.isfinite(v)
        ):
            raise AssetError(
                "bad_param_decl", f"param '{pid}' '{key}' must be a finite number"
            )
    if kind not in ("number", "vector"):
        # A range on a text/boolean param would reach the override
        # range check as a number-vs-non-number comparison — a raw
        # TypeError, not a structured error. Reject at the door.
        range_keys = [k for k in ("min", "max", "step") if k in decl]
        if range_keys:
            raise AssetError(
                "bad_param_decl",
                f"param '{pid}' is a {kind} and cannot carry "
                f"{range_keys} - range keys apply to number/vector "
                f"params only",
            )
    if kind in ("number", "vector"):
        if "min" not in decl or "max" not in decl:
            raise AssetError(
                "bad_param_decl",
                f"param '{pid}' is a {kind} and needs 'min' and 'max' "
                f"(the slider range)",
            )
        if not decl["min"] < decl["max"]:
            raise AssetError(
                "bad_param_decl", f"param '{pid}' needs min < max"
            )
        if "step" in decl and not decl["step"] > 0:
            raise AssetError(
                "bad_param_decl", f"param '{pid}' 'step' must be > 0"
            )
        # The value against the range is not judged here: a value
        # outside it is a finding validate reports beside every
        # other entry (params.range_entries, DESIGN 6), and a
        # refusal at the door would replace that report.


def _check_followed(params: list | None, asserts: list | None) -> None:
    """A knob a threshold follows is no `direct` one (DESIGN 7.2): its
    value is read in a second place, and a patch of the one node param
    would leave the threshold behind."""
    followed = {
        pid: a["id"] for a in asserts or []
        for pid in (a.get("from_param") or {}).values()
    }
    for decl in params or []:
        if "node" in decl and decl["id"] in followed:
            raise AssetError(
                "bad_param_decl",
                f"param '{decl['id']}' records 'node' and 'param', and "
                f"assert '{followed[decl['id']]}' takes a threshold from "
                f"it; a knob read in two places is rebuilt, so the entry "
                f"carries neither",
            )


def loads(text: str) -> Asset:
    try:
        doc = json.loads(
            text,
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as e:
        raise AssetError("bad_json", f"invalid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise AssetError("bad_root", "asset root must be a JSON object")
    unknown_root = set(doc) - ROOT_KEYS
    if unknown_root:
        hint = (
            "; 'time' is a reserved name, not a root key - delete that line"
            if "time" in unknown_root else
            "; the root carries 'build_version', the stamp 'moraine build' "
            "writes - rebuild the recipe rather than edit the IR"
            if "version" in unknown_root else ""
        )
        raise AssetError(
            "bad_root",
            f"asset root has unknown key(s) {sorted(unknown_root)}; "
            f"accepted: {list(ROOT_KEY_ORDER)}{hint}",
        )

    build_version = doc.get("build_version")
    if not isinstance(build_version, str) or not build_version:
        raise AssetError(
            "bad_build_version",
            "root 'build_version' must be a non-empty string, the stamp "
            "'moraine build' writes - rebuild the recipe rather than edit "
            "the IR",
        )

    raw_nodes = doc.get("nodes")
    if not isinstance(raw_nodes, list):
        raise AssetError("bad_nodes", "root 'nodes' must be a list")
    nodes: dict[str, Node] = {}
    for i, rn in enumerate(raw_nodes):
        if not isinstance(rn, dict):
            raise AssetError("bad_node", f"nodes[{i}] must be an object")
        unknown_node = set(rn) - NODE_KEYS
        if unknown_node:
            raise AssetError(
                "bad_node",
                f"nodes[{i}] has unknown key(s) {sorted(unknown_node)}; "
                f"accepted: {list(NODE_KEY_ORDER)}",
            )
        node_id = rn.get("id")
        op = rn.get("op")
        inputs = rn.get("inputs", [])
        params = rn.get("params", {})
        if not isinstance(node_id, str) or not node_id:
            raise AssetError("bad_node_id", f"nodes[{i}] missing string 'id'")
        if id_segments(node_id) is None:
            raise AssetError(
                "bad_node_id",
                f"node id '{node_id}' is not {ID_GRAMMAR}; '{ID_SEPARATOR}' "
                f"separates scopes, so an id neither starts nor ends with "
                f"it nor doubles it",
            )
        if not isinstance(op, str) or not op:
            raise AssetError("bad_node_op", f"node '{node_id}' missing string 'op'")
        if not isinstance(inputs, list) or not all(isinstance(r, str) for r in inputs):
            raise AssetError(
                "bad_node_inputs", f"node '{node_id}' 'inputs' must be a list of node ids"
            )
        if not isinstance(params, dict):
            raise AssetError("bad_node_params", f"node '{node_id}' 'params' must be an object")
        if node_id in nodes:
            raise AssetError("duplicate_id", f"duplicate node id '{node_id}'")
        nodes[node_id] = Node(
            id=node_id, op=op, inputs=list(inputs), params=params,
            site=_check_site(f"node '{node_id}'", rn),
            via=_check_via(f"node '{node_id}'", rn),
        )

    for node in nodes.values():
        for ref in node.inputs:
            if ref not in nodes:
                raise AssetError(
                    "bad_input_ref", f"node '{node.id}' input references unknown node '{ref}'"
                )
    _check_acyclic(nodes)

    raw_outputs = doc.get("outputs")
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        raise AssetError("bad_outputs", "root 'outputs' must be a non-empty object")
    folded: dict[str, str] = {}
    for name, ref in raw_outputs.items():
        check_output(name, ref, nodes, folded)

    groups = doc.get("groups")
    if groups is not None:
        example = '{"shelf_a": ["shelf_a_frame", "shelf_a_posts"]}'
        if not isinstance(groups, dict) or not groups:
            raise AssetError(
                "bad_group",
                f"root 'groups' must be a non-empty object naming output "
                f"groups like {example} (omit the key when there are none)",
            )
        for gname, members in groups.items():
            check_group(gname, members, raw_outputs, folded)

    provenance = doc.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise AssetError("bad_provenance", "root 'provenance' must be an object")

    checks = doc.get("checks")
    if checks is not None:
        if not isinstance(checks, dict):
            raise AssetError("bad_checks", "root 'checks' must be an object")
        for name, policy in checks.items():
            check_waiver(name, policy, raw_outputs)

    asset_checks = doc.get("asset_checks")
    if asset_checks is not None:
        if not isinstance(asset_checks, dict):
            raise AssetError(
                "bad_asset_checks",
                "root 'asset_checks' must be an object like {\"grounded\": false}",
            )
        for key, val in asset_checks.items():
            check_asset_check(key, val)

    budgets = doc.get("budgets")
    if budgets is not None:
        if not isinstance(budgets, dict):
            raise AssetError(
                "bad_budgets", "root 'budgets' must be an object like {\"materials\": 4}"
            )
        for key, val in budgets.items():
            check_budget(key, val)

    asserts = doc.get("asserts")
    if asserts is not None:
        if not isinstance(asserts, list):
            raise AssetError("bad_asserts", "root 'asserts' must be a list of assertions")
        seen_assert_ids: set[str] = set()
        for i, a in enumerate(asserts):
            check_assert(f"asserts[{i}]", a, raw_outputs, groups or {}, seen_assert_ids)

    params = doc.get("params")
    if params is not None:
        if not isinstance(params, list):
            raise AssetError("bad_params", "root 'params' must be a list of declarations")
        seen_ids: set[str] = set()
        seen_bindings: set[tuple[str, str]] = set()
        for i, decl in enumerate(params):
            check_param(f"params[{i}]", decl, nodes, seen_ids, seen_bindings)
    _check_followed(params, asserts)

    return Asset(
        build_version=build_version,
        nodes=nodes,
        outputs=dict(raw_outputs),
        groups={g: list(m) for g, m in groups.items()} if groups is not None else None,
        checks=checks,
        asset_checks=asset_checks,
        provenance=provenance,
        params=params,
        asserts=asserts,
        budgets=budgets,
    )


ASSET_EXT = ".mora"
# The directory that marks a workspace root (DESIGN 4.2).
MORAINE_DIR = ".moraine"
# Where the build writes (DESIGN 4): a project's IR directly under it
# and under its ``subassets/``, each variant beside the IR of its asset.
BUILD_DIR = "build"
SUBASSETS_DIR = "subassets"
LIB_DIR = "lib"
# Names a project directory may contain, never be (DESIGN 4.2): a child
# directory with one of these names is layout, not a project candidate.
# Compared case-insensitively: the supported platforms' filesystems are
# case-insensitive by default, so Exports/ is the same directory the
# tool writes as exports/. ``custom`` is held and read by nothing: a name
# a project has taken cannot be reserved afterwards.
RESERVED_DIRS = frozenset({
    SUBASSETS_DIR, LIB_DIR, BUILD_DIR, "exports", "external", "custom",
})
# A variant IR is named ``<stem>.<identity>.mora`` (DESIGN 4.6), so a
# asset file whose own stem ends that way could not be told from one.
VARIANT_STEM = re.compile(r"\.[0-9a-f]{24}$")


# Extensions compare case-folded: the supported platforms' filesystems
# are case-insensitive by default.


def is_ir_file(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(ASSET_EXT)


def check_stamp(sc: Asset, expected: str, where: str, *,
                source: Callable[[], Path | None]) -> None:
    """Refuse an IR another build wrote (DESIGN 4.1): the runtime reads
    the IR its own build wrote, whose stamp is ``expected``, so any
    other is stale. ``source`` answers with the file the IR was written
    from, or None where it has none, and is asked only on a refusal:
    the fix names the rebuild only where there is a file to rebuild."""
    if sc.build_version == expected:
        return
    written_from = source()
    if written_from is not None and written_from.is_file():
        fix = (
            "the file is stale - run 'moraine build' on its project to "
            "write it again"
        )
    else:
        fix = (
            "no recipe sits beside it to build it again, so it is an IR "
            "to read, not to evaluate - write the asset as a recipe "
            "('moraine docs recipe' maps each IR line to its call)"
        )
    raise BuildError(
        "version_mismatch",
        f"'{where}' carries build_version '{sc.build_version}' and this "
        f"build is '{expected}'; {fix}",
        file=where, found=sc.build_version, expected=expected,
    )


def load(path: str | Path) -> Asset:
    p = Path(path)
    if not p.is_file():
        raise AssetError("asset_not_found", f"asset file not found: {p}")
    try:
        text = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise AssetError(
            "bad_encoding",
            f"IR file '{p}' is not valid UTF-8 (byte {e.start}); "
            f"'moraine build' on its project writes it again",
        ) from e
    except OSError as e:
        raise AssetError(
            "asset_unreadable",
            f"cannot read asset file '{p}' ({e.__class__.__name__}: {e}); "
            f"check that the file is readable and not held open exclusively",
        ) from e
    sc = loads(text)
    sc.path = p.resolve()
    return sc


def ir_home(ir_path: Path) -> Path:
    """The directory an IR's own references resolve from (DESIGN 4.6):
    the ``build/`` of the project its asset belongs to. A sub-asset's
    IR sits one level below it, and a variant beside the IR of its
    asset. An IR outside any ``build/`` is its own home, laid out the
    same way."""
    path = Path(ir_path).resolve()
    parent = path.parent
    # Case-folded like the workspace's reserved names: resolve() restores
    # on-disk casing on case-insensitive filesystems, and a "Subassets/"
    # directory is the same layout slot.
    if parent.name.lower() == SUBASSETS_DIR:
        return parent.parent
    return parent


def project_root(asset_path: Path) -> Path:
    """Project directory owning an asset, named by its recipe or by its
    IR (DESIGN 4.2, 4.6).

    The root asset sits directly in the project directory and
    sub-assets exactly one level down in ``subassets/``; the IR mirrors
    that layout under the project's ``build/``. (Consequently a project
    directory must not itself be named ``subassets`` — DESIGN 4.2.)
    """
    if is_ir_file(asset_path):
        home = ir_home(asset_path)
        return home.parent if home.name.lower() == BUILD_DIR else home
    parent = Path(asset_path).resolve().parent
    if parent.name.lower() == SUBASSETS_DIR:
        return parent.parent
    return parent


def _node_json(node: Node) -> str:
    ordered: dict = {"id": node.id, "op": node.op}
    if node.inputs:  # omitted when empty so source nodes stay terse
        ordered["inputs"] = node.inputs
    ordered["params"] = node.params
    ordered.update(where(node))
    return json.dumps(ordered, separators=(", ", ": "), allow_nan=False)


def _param_json(decl: dict) -> str:
    ordered = {k: decl[k] for k in PARAM_KEY_ORDER if k in decl}
    return json.dumps(ordered, separators=(", ", ": "), allow_nan=False)


def _list_block(key: str, entries: list[str]) -> str:
    """A root list serialized one entry per line (nodes, params,
    asserts): minimal-token diffs. An empty list keeps "[" and "]" on
    adjacent lines (the frozen byte format)."""
    if not entries:
        return f'"{key}": [\n]'
    return f'"{key}": [\n' + ",\n".join(entries) + "\n]"


def _inline(key: str, obj) -> str:
    return f'"{key}": ' + json.dumps(obj, separators=(", ", ": "), allow_nan=False)


def _object_block(key: str, obj: dict) -> str:
    """A root object whose entries grow per output (outputs, groups, checks)
    serialized one entry per line, like the root lists: the fold keys
    on whether entries accumulate, not on JSON container type — a
    growing key must review, diff, and merge entry-wise. Objects with
    a fixed small keyset (budgets, provenance) stay inline. An empty
    object keeps "{" and "}" on adjacent lines (the frozen byte
    format)."""
    if not obj:
        return f'"{key}": {{\n}}'
    entries = [
        json.dumps(k, allow_nan=False)
        + ": "
        + json.dumps(v, separators=(", ", ": "), allow_nan=False)
        for k, v in obj.items()
    ]
    return f'"{key}": {{\n' + ",\n".join(entries) + "\n}"


def _nonfinite_paths(value, path: str = "") -> list[str]:
    """Paths (``size[0]``, ``a.b[1]``) of every non-finite float
    anywhere under a JSON value."""
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [
            p for k, v in value.items()
            for p in _nonfinite_paths(v, f"{path}.{k}" if path else str(k))
        ]
    if isinstance(value, list):
        return [
            p for i, v in enumerate(value)
            for p in _nonfinite_paths(v, f"{path}[{i}]")
        ]
    return []


def _nonfinite_error(asset: Asset) -> AssetError:
    """The actionable error for an asset the serializer cannot write:
    names the node and param path that carry the non-finite value
    (or the root section, for a value outside every node)."""
    for node in asset.nodes.values():
        paths = _nonfinite_paths(node.params)
        if paths:
            return AssetError(
                "nonfinite_number",
                f"node '{node.id}' param '{paths[0]}' is not a finite "
                f"number (an overflow literal such as 1e400 parses to "
                f"inf), so the asset has no canonical text; replace it "
                f"with a finite number",
                node=node.id, param=paths[0],
            )
    for key in ROOT_KEY_ORDER:
        paths = _nonfinite_paths(getattr(asset, key, None), key)
        if paths:
            return AssetError(
                "nonfinite_number",
                f"'{paths[0]}' is not a finite number (an overflow "
                f"literal such as 1e400 parses to inf), so the asset "
                f"has no canonical text; replace it with a finite number",
                path=paths[0],
            )
    return AssetError(  # pragma: no cover - dumps refused a value the walk did not see
        "nonfinite_number",
        "the asset carries a non-finite number and has no canonical "
        "text; replace it with a finite number",
    )


def dumps(asset: Asset) -> str:
    """Canonical serialization: one node per line, fixed key order
    (ROOT_KEY_ORDER); one comma rule — sections join with ",\\n".
    Root containers whose entries grow per output — the lists, and
    the ``outputs``/``checks`` objects — serialize one entry per
    line (DESIGN 4.1). Standard JSON only (allow_nan=False): an inf
    that slipped past the loader (an overflow literal like 1e999
    parses without hitting parse_constant) raises ``nonfinite_number``
    naming the node and param instead of writing 'Infinity'."""
    try:
        return _dumps(asset)
    except ValueError as e:
        raise _nonfinite_error(asset) from e


def _dumps(asset: Asset) -> str:
    sections = [
        _inline("build_version", asset.build_version),
        _list_block("nodes", [_node_json(n) for n in asset.nodes.values()]),
    ]
    if asset.params is not None:
        sections.append(_list_block("params", [_param_json(d) for d in asset.params]))
    sections.append(_object_block("outputs", asset.outputs))
    if asset.groups is not None:
        sections.append(_object_block("groups", asset.groups))
    if asset.checks is not None:
        sections.append(_object_block("checks", asset.checks))
    if asset.asset_checks is not None:
        sections.append(_inline("asset_checks", asset.asset_checks))
    if asset.asserts is not None:
        sections.append(
            _list_block(
                "asserts",
                [
                    json.dumps(
                        {k: a[k] for k in vocab.KEY_ORDER if k in a},
                        separators=(", ", ": "),
                        allow_nan=False,
                    )
                    for a in asset.asserts
                ],
            )
        )
    if asset.budgets is not None:
        sections.append(_inline("budgets", asset.budgets))
    if asset.provenance is not None:
        sections.append(_inline("provenance", asset.provenance))
    return "{\n" + ",\n".join(sections) + "\n}\n"


def write_text(asset_path: Path, text: str) -> None:
    """Write an IR's canonical text to ``asset_path``, atomically
    (``core.atomic``): a concurrent reader — the file watcher included —
    sees the old bytes or the new, never a partial asset. The one
    writer of IR files; the in-flight temp file must not linger beside
    the IR — it is Moraine's own file outside the cache, so it is
    unlinked directly (the cache choke point would refuse it), the
    export-write precedent (DESIGN 4.5)."""
    try:
        atomic_replace(Path(asset_path), text.encode("utf-8"))
    except ReplaceError as err:
        err.tmp.unlink(missing_ok=True)
        e = err.cause
        if err.step == "write":
            raise AssetError(
                "asset_unwritable",
                f"cannot write '{Path(asset_path).name}' in "
                f"'{Path(asset_path).parent}' ({e.__class__.__name__}); "
                "check directory permissions",
            ) from e
        raise AssetError(
            "asset_locked",
            f"cannot replace '{Path(asset_path).name}' "
            f"({e.__class__.__name__}: held open by another program); "
            "close it and retry",
        ) from e


def content_hash(
    files: set[Path] | frozenset[Path], root: Path, *, missing_ok: bool = False,
    read: Mapping[Path, bytes | None] | None = None,
) -> str:
    """Content hash naming one exact asset revision (DESIGN 6): sha256
    over the sorted (root-relative posix path, raw bytes) of every
    contributing recipe file, 24-hex like cache keys. Raw bytes are the
    point: the recipe is the save file, so an edited byte is a
    different revision. ``root`` is the workspace root (or the project directory
    outside a workspace, ``workspace.enclosing_root``): every writer of
    the hash must frame it from the same root or the same revision
    hashes differently per writer.

    An unreadable file raises: the callers stamp revisions whose files
    the evaluation just read, and a hash silently missing one of them
    would name a revision that never existed on disk. The one sanctioned
    skip is ``missing_ok`` for a file that does not exist at all — a
    ``asset_ref`` registers its target before the existence check, and a
    dangling reference's revision is exactly the readable files minus
    the missing one (the reload path's case).

    ``read`` holds what a build found, by resolved path: the bytes it
    read, or None for a file that was not there. That stands for the
    file whatever the disk holds by now, since the revision an
    evaluation names is the one it was built from.
    """
    held = read or {}
    by_path: dict[Path, bytes] = {}
    for f in files:
        fp = Path(f).resolve()
        if fp in held:
            if held[fp] is not None:
                by_path[fp] = held[fp]
            continue
        try:
            by_path[fp] = fp.read_bytes()
        except FileNotFoundError:
            if not missing_ok:
                raise
    return content_hash_of_bytes(by_path, root)


def content_hash_of_bytes(by_path: dict[Path, bytes], root: Path) -> str:
    """`content_hash` over bytes already in hand — for a caller that read
    the files itself (the server snapshots them for the reload diff and
    must not pay a second read, nor let the hash and the snapshot see
    two different file states)."""
    entries = []
    for fp, data in by_path.items():
        try:
            rel = fp.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - contributing files live under root
            rel = fp.as_posix()
        entries.append((rel, data))
    h = hashlib.sha256()
    for rel, data in sorted(entries):
        name = rel.encode("utf-8")
        h.update(len(name).to_bytes(4, "big") + name)
        h.update(len(data).to_bytes(8, "big") + data)
    return h.hexdigest()[:24]
