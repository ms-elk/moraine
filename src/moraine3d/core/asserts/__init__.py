"""Assert vocabulary, parse side (DESIGN 3, 5.2): one kind, one module.

Each kind module exposes:
- ``KEY_KINDS``: the keys the kind accepts besides ``id`` / ``kind`` /
  ``owner`` / ``site`` / ``via``, each with the number type it holds
  (``core.numbers``, DESIGN 4.8): a length or an angle is a float, a
  count of copies and a copy's index an integer.
- ``KEYS``: the accepted keys, the keys of ``KEY_KINDS`` — the one
  table the parser gates on, so a kind can never silently accept a key
  it does not implement (the parser would pass it through and the
  check ignore it).
- ``KEY_MEANING``: what each accepted key means on this kind, one
  phrase per key — the refusal a key draws on another kind names its
  owners from these, so the error channel cannot lag the vocabulary.
- ``SUBJECT``: what the kind measures, as a predicate phrase ("measures
  between its two named outputs"), for the refusals that contrast a
  key with the kind's shape.
- ``REFUSALS`` (optional): the kind's own answer to a key it refuses,
  keyed by that key, where the generic owner list would not name the
  right fix (embedded's band in place of ``tol``, clearance's ``self``
  in place of ``every``).
- ``validate_entry(aid, a, raw_outputs, groups)``: the parse-time
  shape of one entry (raises AssetError); ``raw_outputs`` are the
  asset's declared output names, ``groups`` its root output groups.
  Evaluation facts (tag presence, copy indices) are judged at validate
  time, never here.
- ``TOL_UNIT`` (required when ``"tol"`` is in KEYS): what the key
  measures, for the number's error message.
- ``AUTHORABLE`` (optional, default True): False for a kind the report
  carries but an asset never states; such a module declares ``ENTRY``,
  the report entry the registry injects into every validate (its id
  is reserved), and ``REFUSAL``, the parser's answer to an authored
  entry.
- The kind's defaults (``DEFAULT_TOL`` and the like), each with the
  rationale that makes it a decision rather than a magic number.

The evaluation side mirrors this package kind for kind
(``verify/asserts``); a test keeps the two registries equal. Kind
semantics are documented in the shipped reference (``moraine docs``)
and pinned by golden tests (DESIGN 5.2, 10).
"""

from __future__ import annotations

from ...errors import AssetError
from .. import numbers
from . import clearance, contact, count, embedded, extent, mated, no_overlap

# Registry order is the order refusals list a key's owners in.
KINDS = {
    "contact": contact,
    "clearance": clearance,
    "mated": mated,
    "embedded": embedded,
    "extent": extent,
    "count": count,
    "no_overlap": no_overlap,
}
# The kinds an asset may state, and the entries the built-in kinds add
# to every validate (each under its reserved id).
AUTHORABLE_KINDS = frozenset(
    k for k, m in KINDS.items() if getattr(m, "AUTHORABLE", True)
)
BUILTIN_ENTRIES = tuple(
    m.ENTRY for m in KINDS.values() if not getattr(m, "AUTHORABLE", True)
)
RESERVED_IDS = frozenset(e["id"] for e in BUILTIN_ENTRIES)
OWNERS = ("agent", "human")


def typed_entry(entry: dict) -> None:
    """Hold every number of an authored entry in the type its key
    declares (DESIGN 4.8), in place. An entry of an unknown kind is
    left for the parser to refuse."""
    kinds = getattr(KINDS.get(entry.get("kind")), "KEY_KINDS", {})
    for key, kind in kinds.items():
        if key not in entry:
            continue
        try:
            entry[key] = numbers.typed(entry[key], kind)
        except numbers.FloatForInteger as e:
            raise AssetError(
                "bad_assert",
                numbers.integer_fix(f"assert '{entry.get('id')}' '{key}'", e),
            ) from None


def builtin_entries() -> list[dict]:
    """Fresh copies of the built-in entries, to prepend to an asset's
    asserts for one validate."""
    return [dict(e) for e in BUILTIN_ENTRIES]


# Asset assertions (DESIGN 5.2): declarative constraints as asset
# content, checked deterministically by `validate` and gating like
# tier-1 checks — never waivable (they are the opposite of waivers: a
# failing assertion means fix the geometry). ``owner``
# records who the constraint belongs to and is required on every entry —
# never defaulted, because a default of "agent" would silently hand a
# forgotten human requirement to the agent to weaken: "human"-owned
# entries encode requirements — the agent proposes changes but does not
# edit them without approval (enforced by process and by visibility in
# the validate report, not by the parser); "agent"-owned entries are
# invariants the agent records to protect its own construction and
# may freely edit.
KEY_ORDER = (
    "id", "kind", "outputs", "faces", "measure", "others", "self", "connected",
    "every", "mode", "value", "min", "max", "tol", "min_area", "angle_tol_deg",
    "from_param", "owner", "site", "via",
)
KEYS = frozenset(KEY_ORDER)  # derived: one literal per family
# The keys that judge (DESIGN 5.2): the thresholds a human-owned entry
# may take whole from a knob (``mo.param_value``, DESIGN 4.8), which
# ``from_param`` then records, threshold by threshold.
THRESHOLDS = ("value", "tol", "min", "max", "min_area")
# What every entry carries beside its kind's own keys.
COMMON = frozenset({"id", "kind", "from_param", "owner", "site", "via"})
# `tol`, `min`, `max`, and `value` carry the kind's own measure — meters
# on every length kind, copies on `count` — and never a unit that depends
# on which other keys are present: a key whose unit depended on its own
# presence (or on a companion key) would make an entry unreadable on its
# face (DESIGN 5.2).
# Overlap is not an authored kind: the built-in overlap guard
# (no_overlap.ENTRY) checks every pair in every validate, and an
# intended overlap is a declared `embedded`.

# The sentence a refused key draws, by key: ``{owners}`` is the kinds
# that accept it with what it means on each (from their KEY_MEANING),
# ``{names}`` the same kinds bare, ``{kind}`` the refusing kind and
# ``{subject}`` its SUBJECT phrase. A key without a shape reads "is
# only for {owners}".
_SHAPES = {
    "min": "'{key}' is a bound for {owners}; {kind} takes 'tol'",
    "max": "'{key}' is a bound for {owners}; {kind} takes 'tol'",
    "others": "'{key}' is only for {owners}; universal contact is contradictory",
    "self": (
        "'{key}' is only for {names} (per-copy pairs of instanced outputs); "
        "{kind} {subject}"
    ),
    "faces": "'{key}' is only for {owners}; {kind} names outputs",
    "mode": "'{key}' is only for {owners}; {kind} names outputs",
    "angle_tol_deg": "'{key}' is only for {owners}; {kind} names outputs",
}


def _and(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def key_refusal(kind: str, key: str) -> str:
    """Why ``key`` is not accepted on ``kind``: the kind's own redirect
    where it has one (REFUSALS), else the key's sentence naming the
    kinds that take it and what it means on each — derived from the
    modules, so a kind that starts accepting a key is named the day it
    lands."""
    mod = KINDS[kind]
    own = getattr(mod, "REFUSALS", {}).get(key)
    if own is not None:
        return own
    owners = [(k, m) for k, m in KINDS.items() if key in getattr(m, "KEYS", ())]
    shape = _SHAPES.get(key)
    if shape is None and kind == "mated":
        return (
            f"mated names two tagged faces under 'faces', not '{key}' - write "
            '"faces": [[output, tag], [output, tag]]'
        )
    return (shape or "'{key}' is only for {owners}").format(
        key=key,
        kind=kind,
        subject=mod.SUBJECT,
        owners=_and([f"{k} ({m.KEY_MEANING[key]})" for k, m in owners]),
        names=_and([k for k, _ in owners]),
    )


def output_names(a: dict) -> set[str] | str:
    """The output names an assert entry references: ``"all"`` for the
    built-in guard's blanket form, else the set of names — from
    ``outputs`` (contact, clearance, embedded, count, extent's output
    form) or from ``faces`` (mated's two specs; extent's one flat spec).
    The one reader of the entry shapes."""
    if a.get("kind") == "mated":
        return {f[0] for f in a["faces"]}
    if "faces" in a:
        return {a["faces"][0]}
    outs = a["outputs"]
    return "all" if outs == "all" else set(outs)
