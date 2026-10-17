"""Public parameters: viewer schema and per-use overrides
(DESIGN 7.2; second consumer DESIGN 4.6).

An asset's root ``params`` list (asset.py) declares its public knobs,
each with the value the build resolved. This module turns those entries
into the ``param_schema`` message for clients, judges each value against
its declared range, and patches a knob the build recorded as reaching
one node param whole (``apply_overrides``). A knob without that record
reaches the graph through the recipe alone, so only a rebuild moves it.
"""

from __future__ import annotations

from ..errors import MoraineError
from . import accepted
from .asset import Asset, param_kind
from .sites import where


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _all_int(value: object) -> bool:
    if isinstance(value, list):
        return all(_is_int(c) for c in value)
    return _is_int(value)


def build_schema(sc: Asset, tiers: dict[str, dict] | None = None) -> list[dict]:
    """``param_schema`` payload: declarations joined with current values.

    Call after any overrides are applied, so ``value`` reflects what the
    running preview actually shows. ``tiers`` is the interaction-tier
    map (``engine.tiers.classify``, DESIGN 7.2), optional because this
    module cannot compute it (core must not import ops): with it, each
    entry carries ``tier`` (and ``uniform`` for factor entries); without
    it the schema is tierless — the server always passes one.
    """
    schema: list[dict] = []
    for decl in sc.params or []:
        value = decl["value"]
        entry = {
            "param_id": decl["id"],
            "label": decl.get("label", decl["id"]),
            "bound": "direct" if is_direct(decl) else "build",
        }
        if is_direct(decl):
            entry["node_id"] = decl["node"]
            entry["param"] = decl["param"]
        entry["kind"] = param_kind(value)
        entry["value"] = value
        if tiers is not None:
            entry.update(tiers[decl["id"]])
        # Where another value took the recipe's place (DESIGN 4.1), so
        # the panel can say so and show the default beside it.
        for key in ("from", "default", "min", "max", "step"):
            if key in decl:
                entry[key] = decl[key]
        # Integer-valued params (scalar or all-int vector) without a
        # declared step get step 1: the client's fractional default step
        # would otherwise generate values the op's integer validation
        # rejects on every drag — a control that can never succeed.
        if "step" not in entry and _all_int(value):
            entry["step"] = 1
        schema.append(entry)
    return schema


def is_direct(decl: dict) -> bool:
    """Whether the build recorded the knob's value as reaching exactly
    one node param whole (DESIGN 7.2): patching that param then equals
    a rebuild."""
    return "node" in decl


def edit_site(decl: dict) -> str:
    """Where the knob's own value is edited, for a fix text."""
    if "site" in decl:
        return f"the default at {decl['site']}"
    if is_direct(decl):
        return f"'{decl['node']}.{decl['param']}'"
    return f"the default of param '{decl['id']}' in the recipe"


def accepted_file(decl: dict) -> str:
    """The ``params.json`` a knob's accepted value sits in, as a fix
    names it."""
    return accepted.beside(decl["site"]) if "site" in decl else accepted.PARAMS_FILE


def in_range(value: object, lo: float, hi: float) -> bool:
    """The one range rule (DESIGN 4.6, 6): every component inside the
    declared [min, max], both ends inclusive."""
    components = value if isinstance(value, list) else [value]
    return all(lo <= c <= hi for c in components)


def range_entries(sc: Asset) -> list[dict]:
    """The validate report's ``params`` list (DESIGN 6): one entry per
    declared knob that carries a range, in declaration order, each with
    the value the evaluation used and the range it was declared under.
    A value outside its range fails its entry, never the load, so one
    knob cannot take every other entry of the report with it."""
    entries: list[dict] = []
    for decl in sc.params or []:
        held = decl.get("from") == "accepted" or "mismatch" in decl
        if "min" not in decl and not held:
            # A text or a boolean knob declares no range, and has an
            # entry for what the human's file holds for it alone.
            continue
        value = decl["value"]
        entry: dict = {"param": decl["id"], "ok": True, "value": value}
        if "from" in decl:
            entry["from"] = decl["from"]
        if "min" in decl:
            lo, hi = decl["min"], decl["max"]
            entry.update(ok=in_range(value, lo, hi), min=lo, max=hi)
        entry.update(where(decl))
        if "mismatch" in decl:
            entry["ok"] = False
            entry["mismatch"] = decl["mismatch"]
            entry["fix"] = (
                f"'{accepted_file(decl)}' holds {decl['mismatch']!r} for "
                f"param '{decl['id']}', which is declared as a "
                f"{param_kind(value)} like {value!r}, so the build used "
                f"the recipe's default; accept a value for it in the "
                f"viewer, which rewrites the line"
            )
        elif not entry["ok"] and decl.get("from") == "accepted":
            entry["fix"] = (
                f"param '{decl['id']}' is {value}, outside its declared "
                f"[{lo}, {hi}]; the value was accepted into "
                f"'{accepted_file(decl)}' and the range declared at "
                f"{decl.get('site', 'the recipe')} no longer holds it - "
                f"accept a value inside the range in the viewer, or widen "
                f"the declaration if the range was wrong"
            )
        elif not entry["ok"] and decl.get("from") == "override":
            entry["fix"] = (
                f"param '{decl['id']}' is {value}, outside its declared "
                f"[{lo}, {hi}]; the value was given for this run alone and "
                f"no file holds it, so the other entries judge the asset "
                f"at a value nothing can set while the range stands - "
                f"widen the declaration at {decl.get('site', 'the recipe')} "
                f"if this value should be in reach"
            )
        elif not entry["ok"]:
            entry["fix"] = (
                f"param '{decl['id']}' is {value}, outside its declared "
                f"[{lo}, {hi}]; set {edit_site(decl)} inside "
                f"the range, or widen the declaration if the range was wrong"
            )
        entries.append(entry)
    return entries


def unknown_param(pid: str, declared) -> MoraineError:
    return MoraineError(
        "param_unknown",
        f"no public param '{pid}' in this asset; declared: "
        f"{sorted(declared)}",
    )


def offered_value(decl: dict, value: object) -> object:
    """The value a consumer offers for a knob, judged against the
    declaration and returned in the declared number type (DESIGN 4.6).

    The declared kind must match, and a number/vector value must sit
    inside the declared [min, max]: the range is the only validity
    knowledge an opaque reference exposes. ``decl["value"]`` is the
    knob's own value, whose kind is the declaration's. The one rule
    every consumer's value passes, a reference's override and the
    viewer's overlay alike. A read command's ``--param`` is no
    consumer: its value passes ``typed_value`` and the report judges
    the range.
    """
    value = typed_value(decl, value)
    pid = decl["id"]
    # Every number/vector declaration carries min/max and only those
    # kinds may carry them (asset.py enforces both); per-component.
    # The kind gate keeps a non-numeric value out of the comparison
    # (a raw TypeError, not an error).
    lo, hi = decl.get("min"), decl.get("max")
    if (
        lo is not None and param_kind(value) in ("number", "vector")
        and not in_range(value, lo, hi)
    ):
        raise MoraineError(
            "param_out_of_range",
            f"param '{pid}' expects a value in [{lo}, {hi}]; got "
            f"{value!r} - pick a value in range, or widen the "
            f"declaration where '{pid}' is declared (for a sibling "
            f"project's knob, that is that project's file - widen it "
            f"there, not in this asset)",
            param=pid,
            min=lo,
            max=hi,
        )
    return value


def typed_value(decl: dict, value: object) -> object:
    """A value for a knob in the knob's declared kind and number type
    (DESIGN 4.6), refused where the kind differs. The range is not
    judged here: an accepted value a narrowed range left outside it is
    the report's to fail, and only a consumer's is refused."""
    pid = decl["id"]
    current = decl["value"]
    declared = param_kind(current)
    got = param_kind(value)
    if got != declared or (
        declared == "vector" and len(value) != len(current)
    ):
        raise MoraineError(
            "param_type_mismatch",
            f"param '{pid}' expects a {declared} matching the declared "
            f"value; got {value!r}",
        )
    # JSON clients send every number as a float; an int-valued param
    # (e.g. a seed) accepts integral floats so sliders work at all —
    # per-component for vectors (an int-strict [w, h] size must not
    # reject every override a JSON client can express). Symmetric in
    # the other direction: a float-valued param keeps float when a
    # client sends an end-of-range integral as a JSON integer — an
    # int-flipped value would hand the knob the integer default step
    # (a 0..1 slider degrades into a toggle) and churn cache keys on
    # a value the op reads identically.
    if isinstance(current, list) and isinstance(value, list):
        value = [
            int(v) if (
                _is_int(c) and isinstance(v, float) and v.is_integer()
            ) else float(v) if (isinstance(c, float) and _is_int(v))
            else v
            for c, v in zip(current, value)
        ]
    elif _is_int(current) and isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(current, float) and _is_int(value):
        value = float(value)
    return value


def apply_overrides(sc: Asset, overrides: dict[str, object]) -> None:
    """Patch offered values onto a freshly loaded Asset, in place.

    Each value passes ``offered_value``; the asset's own value is
    reported instead (``range_entries``). Op-level validation still
    happens at evaluation, where a bad value fails with the op's
    actionable error. Only a knob recorded as reaching one node param
    whole can be patched; any other is refused, since its value reaches
    the graph through the recipe. ``None`` is the unset (DESIGN 7.2):
    the knob goes back to the recipe's default, as the entry without
    another source. Applies per-entry: on a raise the Asset may be
    partially mutated, so callers must not reuse it.
    """
    decls = {decl["id"]: decl for decl in sc.params or []}
    for pid, value in overrides.items():
        decl = decls.get(pid)
        if decl is None:
            raise unknown_param(pid, decls)
        if not is_direct(decl):
            raise MoraineError(
                "param_build_bound",
                f"param '{pid}' reaches the graph through its recipe, so "
                f"a value cannot be patched into this IR; open the project "
                f"or the recipe instead of the IR file",
                param=pid,
            )
        if value is None:
            if "from" not in decl:
                continue
            value = decl.pop("default")
            sc.nodes[decl["node"]].params[decl["param"]] = value
            decl["value"] = value
            decl.pop("from")
            decl.pop("mismatch", None)
            continue
        value = offered_value(decl, value)
        sc.nodes[decl["node"]].params[decl["param"]] = value
        # As a rebuild under the same value writes the entry (DESIGN
        # 7.2): the value is a consumer's, whatever it replaced, and
        # the default stays the recipe's.
        if "from" not in decl:
            decl["default"] = decl["value"]
        decl["value"] = value
        decl["from"] = "override"
        decl.pop("mismatch", None)
