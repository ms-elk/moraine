"""What the build records of where a knob's value reached (DESIGN
7.2 `bound`), read from the recipe's text."""

from __future__ import annotations

import asyncio
import json

import pytest

from moraine3d.app.server import ClientState, MoraineServer
from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.engine import tiers
from moraine3d.recipe import uses
from moraine3d.recipe.build import build

from support.recipes import build_recipe, ir, write_files

BODY = 'body = mo.box(id = "bx", size = [1, 1, 1])\n'
SHIP = 'mo.output("main", mt)\n'


def _bound(doc: dict) -> dict:
    return {
        d["id"]: (d["node"], d["param"]) if "node" in d else None
        for d in doc.get("params", [])
    }


def _recipe(knob_lines: str, use: str) -> str:
    return knob_lines + BODY + use + SHIP


# -- a value that reached one param whole --------------------------------------


def test_a_value_handed_whole_to_one_param_is_recorded(tmp_path):
    """DESIGN 7.2: `bound` is `direct` only when the build has recorded
    that the declared value reached exactly one node param whole and
    unmodified, and nowhere else. DESIGN 4.1: `node` and `param` are
    that record, and the two values agree."""
    doc = ir(build_recipe(tmp_path, _recipe(
        'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
        'c = mo.param("tint", [0.8, 0.2, 0.2, 1.0], min = 0.0, max = 1.0)\n',
        'mt = mo.material(body, id = "mt", name = "paint", roughness = r,\n'
        '                 base_color = c)\n',
    ), write=False))
    assert _bound(doc) == {
        "rough": ("mt", "roughness"), "tint": ("mt", "base_color"),
    }
    sc = asset_mod.loads(json.dumps(doc))
    assert {k: v["tier"] for k, v in tiers.classify(sc).items()} == {
        "rough": "factor", "tint": "factor",
    }


def test_the_call_may_be_written_as_an_argument_or_over_several_lines(tmp_path):
    """DESIGN 7.2: the record names the node the value reached. A call
    written as the argument of another is made once, as a statement's
    own call is, and a node's site is the line its call begins on."""
    doc = ir(build_recipe(tmp_path, (
        'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
        's = mo.param("seed", 7, min = 0, max = 99)\n'
        + BODY +
        'mo.output("main", mo.material(\n'
        '    mo.noise_displace(body, id = "nd", seed = s, amplitude = 0.01),\n'
        '    id = "mt",\n'
        '    name = "paint",\n'
        '    roughness = r,\n'
        '))\n'
    ), write=False))
    assert _bound(doc) == {"rough": ("mt", "roughness"), "seed": ("nd", "seed")}
    sc = asset_mod.loads(json.dumps(doc))
    assert tiers.classify(sc)["seed"] == {"tier": "seed"}


def test_a_knob_declared_where_it_is_used_is_recorded(tmp_path):
    """DESIGN 7.2: a value bound to no name is written nowhere else."""
    doc = ir(build_recipe(tmp_path, BODY + (
        'mt = mo.material(body, id = "mt", name = "paint",\n'
        '    roughness = mo.param("rough", 0.5, min = 0.0, max = 1.0))\n'
    ) + SHIP, write=False))
    assert _bound(doc) == {"rough": ("mt", "roughness")}


# -- every other use leaves the knob rebuilt -----------------------------------

KNOB = 'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
PLAIN = 'mt = mo.material(body, id = "mt", name = "paint", roughness = r)\n'


@pytest.mark.parametrize("recipe", [
    pytest.param(
        _recipe(KNOB, 'mt = mo.material(body, id = "mt", name = "p", roughness = r * 1.0)\n'),
        id="combined",
    ),
    pytest.param(
        _recipe(KNOB, 'mt = mo.material(body, id = "mt", name = "p", base_color = [r, r, r])\n'),
        id="inside a vector",
    ),
    pytest.param(
        _recipe(KNOB, PLAIN + 'm2 = mo.material(body, id = "m2", name = "q", roughness = r)\n'),
        id="used twice",
    ),
    pytest.param(
        _recipe(KNOB, PLAIN + 'extra = 1 if r > 0.3 else 2\n'),
        id="consulted by a condition",
    ),
    pytest.param(
        _recipe(KNOB, PLAIN + 'other = r\n'),
        id="bound to a second name",
    ),
    pytest.param(
        _recipe(KNOB, (
            'def painted(id, rough):\n'
            '    return mo.material(body, id = id, name = "p", roughness = rough)\n'
            'mt = painted("mt", r)\n'
        )),
        id="handed to a function",
    ),
    pytest.param(
        _recipe(KNOB, (
            'def painted(id):\n'
            '    return mo.material(body, id = id, name = "p", roughness = r)\n'
            'mt = painted("mt")\n'
        )),
        id="read inside a function",
    ),
    pytest.param(
        _recipe(KNOB, (
            'made = [mo.material(body, id = "m%d" % i, name = "p%d" % i, roughness = r)\n'
            '        for i in range(2)]\n'
            'mt = made[0]\n'
        )),
        id="read inside a comprehension",
    ),
    pytest.param(
        _recipe(KNOB, 'mt = mo.material(body, id = "mt", name = "p", **{"roughness": r})\n'),
        id="inside an expansion",
    ),
    pytest.param(
        _recipe(
            'n = mo.param("count", 3, min = 1, max = 6)\n',
            'mt = mo.material(mo.transform(body, id = "tf", scale = n), id = "mt", name = "p")\n',
        ),
        id="held in another number type",
    ),
])
def test_any_other_use_leaves_the_knob_rebuilt(tmp_path, recipe):
    """DESIGN 7.2: a value placed inside a vector, used twice, combined
    with anything, or consulted by a condition is `build`. Every doubt
    lands in the safe tier."""
    result = build_recipe(tmp_path, recipe, write=False)
    assert set(_bound(ir(result)).values()) == {None}
    schema = params_mod.build_schema(
        result.target.asset, tiers.classify(result.target.asset),
    )
    assert {(p["bound"], p["tier"]) for p in schema} == {("build", "evaluation")}


def test_a_value_passed_on_to_a_reference_is_rebuilt(tmp_path):
    """DESIGN 4.6: an asset forwards a deeper knob by declaring a knob
    of its own and passing its value on in the `overrides` of its own
    reference. The value sits inside the overrides, and the build
    resolves what a reference names, so the knob is `build`."""
    write_files(tmp_path, {
        "asset.star": (
            'w = mo.param("width", 1.0, min = 0.5, max = 2.0)\n'
            'o = mo.param("part", "p")\n'
            'mo.output("a", mo.asset_ref("subassets/part.star", "p", id = "a",'
            ' overrides = {"w": w}))\n'
            'mo.output("b", mo.asset_ref("subassets/part.star", output = o, id = "b"))\n'
        ),
        "subassets/part.star": (
            'w = mo.param("w", 0.5, min = 0.1, max = 3.0)\n'
            'mo.output("p", mo.box(id = "p", size = [w, 1, 1]))\n'
        ),
    })
    result = build(tmp_path / "asset.star", write=False)
    assert _bound(ir(result)) == {"width": None, "part": None}


def test_a_knob_a_threshold_follows_is_rebuilt(tmp_path):
    """DESIGN 7.2: a value a threshold follows (`mo.param_value`) is
    read in a second place, so the knob is `build`, and an IR that
    records both is refused."""
    recipe = _recipe(KNOB, PLAIN)
    alone = build_recipe(tmp_path / "a", recipe, write=False)
    assert _bound(ir(alone)) == {"rough": ("mt", "roughness")}
    result = build_recipe(tmp_path / "b", recipe, write=False, files={
        "asset.requirements.star": (
            'mo.assert_extent(id = "wide", outputs = ["main"], measure = "size_x",'
            ' max = mo.param_value("rough"))\n'
        ),
    })
    doc = ir(result)
    assert _bound(doc) == {"rough": None}
    rebuilt = build(tmp_path / "b" / "asset.star", offered={"rough": 0.9}, write=False)
    assert next(a for a in ir(rebuilt)["asserts"] if a["id"] == "wide")["max"] == 0.9
    doc["params"][0].update(node="mt", param="roughness")
    with pytest.raises(Exception) as e:
        asset_mod.loads(json.dumps(doc))
    assert e.value.code == "bad_param_decl" and "'wide'" in e.value.message


def test_a_file_pythons_parser_cannot_read_is_named_and_rebuilt(tmp_path, monkeypatch):
    """DESIGN 7.2: every doubt lands in the safe tier. An asset file
    whose text the reading cannot take is built as ever, its knobs are
    `build`, and the build's report names the file."""
    monkeypatch.setattr(uses, "parse", lambda source: None)
    result = build_recipe(tmp_path, _recipe(KNOB, PLAIN), write=False)
    assert _bound(ir(result)) == {"rough": None}
    assert result.report(wrote=False)["unread"]["files"] == ["asset.star"]


def test_text_pythons_parser_refuses_gives_no_reading():
    """DESIGN 7.2: every doubt lands in the safe tier. Text the
    reading cannot take gives no reading, never a guess."""
    assert uses.read("x = = 1\n") is None
    assert uses.read("x = 1\n") is not None


# -- a patch is a rebuild ------------------------------------------------------


def _knobs() -> str:
    """Three knobs, each handed whole to one param. The body is a
    sphere, whose shading `noise_displace` takes."""
    return (
        'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
        's = mo.param("seed", 7, min = 0, max = 99)\n'
        'c = mo.param("tint", [0.8, 0.2, 0.2, 1.0], min = 0.0, max = 1.0)\n'
        'body = mo.sphere(id = "bx", radius = 0.5, segments = 16, rings = 8)\n'
        'nd = mo.noise_displace(body, id = "nd", seed = s, amplitude = 0.01)\n'
        'mt = mo.material(nd, id = "mt", name = "paint", roughness = r, base_color = c)\n'
        + SHIP
    )


@pytest.mark.parametrize("offered", [
    {"rough": 0.0}, {"rough": 1}, {"seed": 99}, {"seed": 12.0},
    {"tint": [0, 1, 0.5, 1]}, {"rough": 0.25, "seed": 3, "tint": [1.0, 1.0, 1.0, 1.0]},
])
def test_a_direct_patch_and_a_rebuild_write_the_same_ir(tmp_path, offered):
    """DESIGN 7.2: `direct` is an optimization, not a second
    semantics, and a test pins that a direct patch and a rebuild
    produce the same IR."""
    write_files(tmp_path, {"asset.star": _knobs()})
    plain = build(tmp_path / "asset.star", write=False)
    assert all(params_mod.is_direct(d) for d in plain.target.asset.params)
    patched = asset_mod.loads(plain.target.text)
    params_mod.apply_overrides(patched, dict(offered))
    rebuilt = build(tmp_path / "asset.star", offered=dict(offered), write=False)
    assert asset_mod.dumps(patched) == rebuilt.target.text


def test_a_patch_over_an_accepted_value_is_a_rebuild_too(tmp_path):
    """DESIGN 7.2: the pin holds whatever the knob's value came from:
    an accepted value is the file state an overlay is laid over."""
    write_files(tmp_path, {
        "asset.star": _knobs(), "params.json": '{\n"rough": 0.75,\n"seed": "x"\n}\n',
    })
    plain = build(tmp_path / "asset.star", write=False)
    assert all(params_mod.is_direct(d) for d in plain.target.asset.params)
    offered = {"rough": 0.2, "seed": 4}
    patched = asset_mod.loads(plain.target.text)
    params_mod.apply_overrides(patched, dict(offered))
    rebuilt = build(tmp_path / "asset.star", offered=dict(offered), write=False)
    assert asset_mod.dumps(patched) == rebuilt.target.text


# -- the server ----------------------------------------------------------------


class _StubWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, text):
        self.sent.append(json.loads(text))

    def messages(self):
        return list(self.sent)


def _server(tmp_path, monkeypatch, recipe: str) -> MoraineServer:
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    write_files(tmp_path, {"asset.star": recipe})
    server = MoraineServer(tmp_path / "asset.star")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def test_a_recipes_knobs_take_their_tiers(tmp_path, monkeypatch):
    """DESIGN 7.2: `seed`: the bound param is in the op's
    `SEED_PARAMS`. `factor`: the bound param is in the op's
    `UNIFORM_PARAMS`. A factor scrub answers with the schema alone."""
    server = _server(tmp_path, monkeypatch, _knobs())
    schema = {p["param_id"]: p for p in server.param_payload["params"]}
    assert {k: (p["bound"], p["tier"]) for k, p in schema.items()} == {
        "rough": ("direct", "factor"), "seed": ("direct", "seed"),
        "tint": ("direct", "factor"),
    }
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    assert [m["type"] for m in ws.messages()] == ["param_schema"]
    rows = {p["param_id"]: p for p in ws.messages()[0]["params"]}
    assert rows["rough"]["value"] == 0.9 and rows["rough"]["bound"] == "direct"


def test_an_overlay_of_direct_knobs_is_patched_without_a_rebuild(tmp_path, monkeypatch):
    """DESIGN 7.2: a `build`-bound knob in the batch first reruns the
    recipe in-process with the overlay's values; a `direct` one
    patches the IR value without a rebuild."""
    from moraine3d.recipe import target as target_mod

    server = _server(tmp_path, monkeypatch, _recipe(
        'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
        'h = mo.param("height", 1.0, min = 0.5, max = 2.0)\n',
        'tf = mo.transform(body, id = "tf", scale = [1, h, 1])\n'
        'mt = mo.material(tf, id = "mt", name = "paint", roughness = r)\n',
    ))
    built = []
    opened = target_mod.open_target

    def counted(*a, **k):
        built.append(k.get("offered"))
        return opened(*a, **k)

    monkeypatch.setattr(target_mod, "open_target", counted)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"rough": 0.9}}))
    assert built == []
    assert [m["type"] for m in ws.messages()] == ["asset_updated", "param_schema"]
    asyncio.run(server._set_param(ws, {"params": {"height": 1.5}}))
    assert built == [{"rough": 0.9, "height": 1.5}]
    rows = {p["param_id"]: p["value"] for p in ws.messages()[-1]["params"]}
    assert rows == {"rough": 0.9, "height": 1.5}


def test_a_patch_is_judged_by_the_declaration(tmp_path, monkeypatch):
    """DESIGN 4.6: errors reuse the declaration contract, and the
    declared min/max is part of it, whichever way the value lands."""
    server = _server(tmp_path, monkeypatch, _knobs())
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"seed": 100}}))
    err = next(m for m in ws.messages() if m["type"] == "asset_error")
    assert err["error"]["code"] == "param_out_of_range"
    assert server.clients[ws].overrides == {}
