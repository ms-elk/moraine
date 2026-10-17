"""The test-side translator (DESIGN 10): an IR fixture to a flat recipe.

One call per node, ids kept, definitions ordered before their uses.
Test infrastructure, never a command: it exists so the IR corpus can
prove the builtins cover the IR, and so an asset written before recipes
has a recipe to start from. The flat form is a test form, not a
document: nothing here tries to write the recipe a person would.

A fixture is an IR document whose knobs name the node param their
value reached (``node`` and ``param``), which is what lets the
translator wire a knob's value into the call that consumes it. Human
asserts and budgets go to the requirements file, since the file
decides the owner (DESIGN 5.2).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from moraine3d import ops

RECIPE_EXT = ".star"
REQUIREMENTS_EXT = ".requirements.star"
IR_EXT = ".mora"

# The ops whose optional inputs a recipe names by keyword: the param
# the IR lists the filled slots under, and each slot's keyword.
SLOTTED = {
    name: (ops.get(name).SLOT_PARAM, {s: k for k, s in ops.get(name).SLOT_INPUTS.items()})
    for name in ops.names() if hasattr(ops.get(name), "SLOT_INPUTS")
}

_IDENT = re.compile(r"\W")
# Names a Starlark call cannot take as a keyword.
_RESERVED = {
    "and", "break", "continue", "def", "elif", "else", "for", "if", "in",
    "lambda", "load", "not", "or", "pass", "return",
}


def literal(v) -> str:
    """A JSON value as a Starlark literal."""
    if v is None:
        return "None"
    if v is True:
        return "True"
    if v is False:
        return "False"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


def _keywords(d: dict, values: dict | None = None) -> list[str]:
    out = []
    for k, v in d.items():
        if k in _RESERVED:
            raise ValueError(f"'{k}' cannot be written as a keyword")
        out.append(f"{k} = {(values or {}).get(k) or literal(v)}")
    return out


def ordered(nodes: list[dict]) -> list[dict]:
    """The nodes with every definition before its uses, otherwise in
    file order."""
    by_id = {n["id"]: n for n in nodes}
    done: set[str] = set()
    out: list[dict] = []
    for n in nodes:
        stack = [(n, iter(n.get("inputs", [])))]
        while stack:
            cur, it = stack[-1]
            for dep in it:
                if dep not in done and dep in by_id:
                    stack.append((by_id[dep], iter(by_id[dep].get("inputs", []))))
                    break
            else:
                stack.pop()
                if cur["id"] not in done:
                    done.add(cur["id"])
                    out.append(cur)
    return out


def _names(prefix: str, ids: list[str]) -> dict[str, str]:
    taken: set[str] = set()
    out = {}
    for i in ids:
        base = prefix + _IDENT.sub("_", i)
        name, k = base, 1
        while name in taken:
            k += 1
            name = f"{base}_{k}"
        taken.add(name)
        out[i] = name
    return out


def _assert_line(a: dict) -> str:
    rest = {k: v for k, v in a.items() if k not in ("kind", "owner", "site")}
    return f"mo.assert_{a['kind']}({', '.join(_keywords(rest))})"


def recipe_path(asset: str) -> str:
    """An IR's reference path as the recipe writes it."""
    if asset.endswith(IR_EXT):
        return asset[: -len(IR_EXT)] + RECIPE_EXT
    return asset


def _reference(node: dict, var: str, wired: dict) -> str:
    params = dict(node["params"])
    args = []
    if "project" in params:
        # Beside a project the IR names the sibling's root IR, which
        # the build writes and no recipe does.
        params.pop("asset", None)
    if "asset" in params:
        asset = params.pop("asset")
        args.append(wired.get("asset") or literal(recipe_path(asset)))
        if "output" in params:
            output = params.pop("output")
            args.append(wired.get("output") or literal(output))
    args.append(f"id = {literal(node['id'])}")
    args += _keywords(params, wired)
    return f"{var} = mo.asset_ref({', '.join(args)})"


def flatten(doc: dict) -> tuple[str, str]:
    """The asset file's text and its requirements file's text ("" when
    the fixture has no human assert and no budget)."""
    nodes = ordered(doc["nodes"])
    var = _names("n_", [n["id"] for n in nodes])
    knob = _names("p_", [p["id"] for p in doc.get("params") or []])
    bound = {
        (p["node"], p["param"]): knob[p["id"]]
        for p in doc.get("params") or [] if "node" in p
    }

    by_id = {n["id"]: n for n in nodes}
    asset: list[str] = []
    for p in doc.get("params") or []:
        value = p["value"]
        if p.get("param") == "asset" and by_id[p["node"]]["op"] == "asset_ref":
            value = recipe_path(value)
        args = [literal(p["id"]), literal(value)]
        args += _keywords({k: p[k] for k in ("min", "max", "label", "step") if k in p})
        asset.append(f"{knob[p['id']]} = mo.param({', '.join(args)})")
    if asset:
        asset.append("")
    for n in nodes:
        wired = {k: bound[(n["id"], k)] for k in n.get("params", {}) if (n["id"], k) in bound}
        if n["op"] == "asset_ref" and not n.get("inputs"):
            asset.append(_reference(n, var[n["id"]], wired))
            continue
        args = [var[i] for i in n.get("inputs", [])]
        params = dict(n.get("params", {}))
        slots = SLOTTED.get(n["op"])
        if slots is not None:
            param, keyword = slots
            filled = params.pop(param, [])
            lead = len(args) - len(filled)
            args = args[:lead] + [
                f"{keyword[slot]} = {arg}" for slot, arg in zip(filled, args[lead:])
            ]
        args.append(f"id = {literal(n['id'])}")
        args += _keywords(params, wired)
        asset.append(f"{var[n['id']]} = mo.{n['op']}({', '.join(args)})")
    asset.append("")
    for name, node in (doc.get("outputs") or {}).items():
        asset.append(f"mo.output({literal(name)}, {var[node]})")
    for name, members in (doc.get("groups") or {}).items():
        asset.append(f"mo.group({literal(name)}, {literal(members)})")
    for out, checks in (doc.get("checks") or {}).items():
        for check in checks:
            asset.append(f"mo.waive({literal(out)}, {literal(check)})")
    for check in doc.get("asset_checks") or {}:
        asset.append(f"mo.waive_asset({literal(check)})")
    if doc.get("provenance"):
        prov = doc["provenance"]
        asset.append(
            f"mo.vendored({literal(prov['source'])}, {literal(prov['content_hash'])})"
        )

    requirements: list[str] = []
    for a in doc.get("asserts") or []:
        (requirements if a["owner"] == "human" else asset).append(_assert_line(a))
    for name, v in (doc.get("budgets") or {}).items():
        requirements.append(f"mo.budget({name} = {literal(v)})")

    return (
        "\n".join(asset) + "\n",
        "\n".join(requirements) + "\n" if requirements else "",
    )


def write_recipe(ir_file: Path, project: Path, name: str | None = None) -> Path:
    """Translate one IR fixture file into ``project``: a fixture beside
    a project root becomes the root asset, one under ``subassets/`` a
    sub-asset. Returns the asset file."""
    doc = json.loads(Path(ir_file).read_text(encoding="utf-8"))
    return write_doc(doc, project, name or Path(ir_file).name[: -len(IR_EXT)],
                     sub=Path(ir_file).parent.name == "subassets")


def write_doc(doc: dict, project: Path, name: str, *, sub: bool = False) -> Path:
    where = project / "subassets" if sub else project
    where.mkdir(parents=True, exist_ok=True)
    asset, requirements = flatten(doc)
    target = where / f"{name}{RECIPE_EXT}"
    for path, text in ((target, asset), (where / f"{name}{REQUIREMENTS_EXT}", requirements)):
        if text:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    return target


def write_project(fixture: Path, project: Path) -> Path:
    """Translate a fixture project directory (a root IR and its
    ``subassets/``) into ``project``. Returns the root asset file."""
    roots = sorted(p for p in Path(fixture).glob(f"*{IR_EXT}") if p.is_file())
    if len(roots) != 1:
        raise ValueError(f"{fixture} holds {len(roots)} root fixtures")
    for sub in sorted((Path(fixture) / "subassets").glob(f"*{IR_EXT}")):
        write_recipe(sub, project)
    return write_recipe(roots[0], project)


# -- equivalence ---------------------------------------------------------------


def _numbers(v):
    """A value with every number as a float, so ``1`` and ``1.0`` in an
    override compare equal (the declaration decides the type)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_numbers(x) for x in v]
    if isinstance(v, dict):
        return {k: _numbers(x) for k, x in v.items()}
    return v


def _node(n: dict) -> dict:
    out = {k: v for k, v in n.items() if k != "site"}
    out.setdefault("inputs", [])
    if n["op"] == "asset_ref" and "overrides" in out.get("params", {}):
        out["params"] = dict(out["params"], overrides=_numbers(out["params"]["overrides"]))
    return out


def differences(fixture: dict, built: dict, *, order: bool = True) -> list[str]:
    """Where the build of a fixture's flat recipe departs from the
    fixture (DESIGN 10): node ids, outputs, groups, params, asserts,
    checks, and budgets. Bytes are not compared, since the build adds
    ``site`` and orders nodes by definition. ``order`` off compares a
    recipe written by hand, whose definitions come in its own order."""
    out: list[str] = []
    a = {n["id"]: _node(n) for n in fixture["nodes"]}
    b = {n["id"]: _node(n) for n in built["nodes"]}
    if order and list(b) != [n["id"] for n in ordered(fixture["nodes"])]:
        out.append("node order")
    for nid in sorted(set(a) | set(b)):
        if a.get(nid) != b.get(nid):
            out.append(f"node {nid}: {a.get(nid)} != {b.get(nid)}")
    for key in ("outputs", "groups", "checks", "asset_checks", "budgets", "provenance"):
        if fixture.get(key) != built.get(key):
            out.append(f"{key}: {fixture.get(key)} != {built.get(key)}")
    knobs = lambda d: [  # noqa: E731
        {k: v for k, v in p.items() if k not in ("node", "param", "site")}
        for p in d.get("params") or []
    ]
    if knobs(fixture) != knobs(built):
        out.append(f"params: {knobs(fixture)} != {knobs(built)}")
    entries = lambda d: {  # noqa: E731
        e["id"]: {k: v for k, v in e.items() if k != "site"}
        for e in d.get("asserts") or []
    }
    fa, ba = entries(fixture), entries(built)
    for aid in sorted(set(fa) | set(ba)):
        if fa.get(aid) != ba.get(aid):
            out.append(f"assert {aid}: {fa.get(aid)} != {ba.get(aid)}")
    return out
