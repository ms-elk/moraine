"""The viewer's accept (DESIGN 7.2): the pick written to the project's
``params.json``, the check of the accepted state, and the warning
that stands while it is red."""

from __future__ import annotations

import asyncio
import json

import pytest

from moraine3d.app import server as server_mod
from moraine3d.app.server import ClientState, MoraineServer

from support.recipes import workspace, write_files

ASSET = (
    'h = mo.param("height", 2.0, min = 1.0, max = 3.0)\n'
    'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
    'body = mo.transform(mo.box(id = "bx", size = [0.2, h, 0.2]), id = "body",'
    ' translate = [0, h / 2, 0])\n'
    'mo.output("main", mo.material(body, id = "mt", name = "paint", roughness = r))\n'
)
# The human's requirement and the agent's invariant, both red above 2.4 m.
REQUIREMENTS = (
    'mo.assert_extent(id = "headroom", outputs = ["main"], measure = "max_y", max = 2.4)\n'
)
INVARIANT = (
    'mo.assert_extent(id = "stays_low", outputs = ["main"], measure = "size_y", max = 2.4)\n'
)


class _StubWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, text):
        self.sent.append(json.loads(text))

    def messages(self, kind: str | None = None):
        return [m for m in self.sent if kind is None or m["type"] == kind]


def _serve(tmp_path, monkeypatch, files: dict | None = None, asset: str = "asset.star"):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    project = write_files(tmp_path / "p", {"asset.star": ASSET, **(files or {})})
    server = MoraineServer(project / asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    ws = _StubWS()
    server.clients[ws] = ClientState()
    return server, ws, project


def _events(monkeypatch) -> list:
    events: list = []
    monkeypatch.setattr(
        server_mod.session, "record", lambda ev, **f: events.append((ev, f))
    )
    return events


def _reload(server) -> None:
    """One watch reload, and the check it starts where it starts one."""
    async def go():
        server._accept_task = None
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        if server._accept_task is not None:
            await server._accept_task

    asyncio.run(go())


def _accept(server, ws):
    """One accept and the watch reload that lands it."""
    asyncio.run(server._accept(ws, {"type": "accept"}))
    _reload(server)


# -- the write -----------------------------------------------------------------


def test_accept_writes_the_overlay_to_the_projects_file(tmp_path, monkeypatch):
    """DESIGN 7.2: accept writes this connection's preview overlay to
    the project's `params.json`, and answers the accepting client
    `accepted {params}`. It writes exactly the state the tab displays.
    Accept never rewrites the recipe source."""
    server, ws, project = _serve(tmp_path, monkeypatch)
    recipes = {p: p.read_bytes() for p in project.rglob("*.star")}
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5, "rough": 1}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("accepted") == [
        {"type": "accepted", "params": ["height", "rough"]}
    ]
    assert (project / "params.json").read_bytes() == (
        b'{\n"height": 2.5,\n"rough": 1.0\n}\n'
    )
    assert {p: p.read_bytes() for p in project.rglob("*.star")} == recipes
    assert [p.name for p in project.iterdir() if "tmp" in p.name] == []


def test_the_reload_that_follows_shows_the_accepted_state(tmp_path, monkeypatch):
    """DESIGN 7.2: the durable confirmation is the watch reload that
    follows and reaches every client. After an accept the file state
    is the accepted state, and the reload drops the overlay."""
    server, ws, _ = _serve(tmp_path, monkeypatch)
    asyncio.run(server._set_param(ws, {"params": {"height": 2.25}}))
    _accept(server, ws)
    assert server.clients[ws].overrides == {}
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["height"]["value"] == 2.25
    assert server.param_payload["overridden"] == []
    assert server.asset_payload["stats"]["main"]["bounds"]["max"][1] == 2.25


def test_a_second_accept_moves_only_its_own_line(tmp_path, monkeypatch):
    """DESIGN 7.2: the file is written whole through the one project
    writer, so the diff is the changed lines alone, and a knob the
    overlay does not hold keeps its accepted value."""
    server, ws, project = _serve(tmp_path, monkeypatch)
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5, "rough": 0.9}}))
    _accept(server, ws)
    before = (project / "params.json").read_bytes().split(b"\n")
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.25}))
    _accept(server, ws)
    after = (project / "params.json").read_bytes().split(b"\n")
    assert [(a, b) for a, b in zip(before, after) if a != b] == [
        (b'"rough": 0.9', b'"rough": 0.25')
    ]


def test_accept_drops_the_line_of_an_id_no_knob_declares(tmp_path, monkeypatch):
    """DESIGN 7.2: the overlay holds declared knob values and nothing
    else. A line the file holds for an id the asset declares no knob
    for leaves with the write, and the answer names it."""
    server, ws, project = _serve(
        tmp_path, monkeypatch, {"params.json": '{\n"gone": 3,\n"rough": 0.7\n}\n'},
    )
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("accepted") == [
        {"type": "accepted", "params": ["height"], "dropped": ["gone"]}
    ]
    assert json.loads((project / "params.json").read_text(encoding="utf-8")) == {
        "height": 2.5, "rough": 0.7,
    }


def test_accept_brings_a_value_back_inside_its_range(tmp_path, monkeypatch):
    """DESIGN 7.2: a knob whose value sits outside its declared range
    is reported, not refused, and the human repairs it from the panel:
    a preview inside the range is accepted over it."""
    server, ws, project = _serve(
        tmp_path, monkeypatch, {"params.json": '{\n"height": 3.0\n}\n'},
    )
    write_files(project, {"asset.star": ASSET.replace("max = 3.0", "max = 2.5")})
    _reload(server)
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert (rows["height"]["value"], rows["height"]["max"]) == (3.0, 2.5)
    asyncio.run(server._set_param(ws, {"params": {"height": 2.0}}))
    _accept(server, ws)
    assert ws.messages("accept_validated")[-1]["ok"] is True


def test_an_accept_of_an_unset_removes_the_line(tmp_path, monkeypatch):
    """DESIGN 7.2: a `null` value unsets the knob in the overlay, the
    preview shows the recipe's default under the knob's tier contract,
    and an accept of it removes the line. A `direct` knob's unset is
    a patch and a `build`-bound one's a rebuild; each shows the
    default and names no other source."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "params.json": '{\n"height": 2.5,\n"rough": 0.9\n}\n',
    })
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["rough"]["bound"] == "direct" and rows["height"]["bound"] == "build"
    assert (rows["rough"]["from"], rows["rough"]["default"]) == ("accepted", 0.5)
    assert (rows["height"]["from"], rows["height"]["default"]) == ("accepted", 2.0)
    assert (rows["rough"]["accepted"], rows["height"]["accepted"]) == (0.9, 2.5)
    # The factor's unset: the schema alone, no evaluation.
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": None}))
    schema = ws.messages("param_schema")[-1]
    assert schema["overridden"] == ["rough"]
    rough = next(p for p in schema["params"] if p["param_id"] == "rough")
    assert rough["value"] == 0.5 and "from" not in rough and "default" not in rough
    assert rough["accepted"] == 0.9
    assert ws.messages("asset_updated") == []
    # The build-bound knob's unset: an Apply, which rebuilds.
    asyncio.run(server._set_param(ws, {"params": {"height": None}}))
    schema = ws.messages("param_schema")[-1]
    height = next(p for p in schema["params"] if p["param_id"] == "height")
    assert height["value"] == 2.0 and "from" not in height
    assert height["accepted"] == 2.5
    assert ws.messages("asset_updated")[-1]["stats"]["main"]["bounds"]["max"][1] == 2.0
    _accept(server, ws)
    assert ws.messages("accepted")[-1] == {
        "type": "accepted", "params": ["height", "rough"],
    }
    assert (project / "params.json").read_bytes() == b"{\n}\n"
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["height"]["value"] == 2.0 and "from" not in rows["height"]
    assert "accepted" not in rows["height"] and "accepted" not in rows["rough"]


def test_a_preview_leaves_the_accepted_value_in_the_schema(tmp_path, monkeypatch):
    """DESIGN 7.2: `accepted` is the value the human's file holds, read
    from the file, so it stands in the schema of a preview whose
    `value` the overlay replaced, and follows the file on an accept."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "params.json": '{\n"height": 2.5\n}\n',
    })
    asyncio.run(server._set_param(ws, {"params": {"height": 3.0}}))
    height = next(
        p for p in ws.messages("param_schema")[-1]["params"] if p["param_id"] == "height"
    )
    assert (height["value"], height["accepted"], height["default"]) == (3.0, 2.5, 2.0)
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert "accepted" not in rows["rough"]
    _accept(server, ws)
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["height"]["accepted"] == 3.0


def test_an_unset_of_one_knob_leaves_the_other_lines(tmp_path, monkeypatch):
    """DESIGN 7.2: an accept writes exactly the state the tab displays,
    so the unset of one knob removes its line and no other."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "params.json": '{\n"height": 2.5,\n"rough": 0.9\n}\n',
    })
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": None}))
    _accept(server, ws)
    assert (project / "params.json").read_bytes() == b'{\n"height": 2.5\n}\n'
    assert ws.messages("accept_validated")[-1]["moves_no_check"] is True


# -- refusals ------------------------------------------------------------------


def test_an_empty_overlay_is_refused(tmp_path, monkeypatch):
    """DESIGN 7.2: an empty overlay reaching the runtime anyway is
    refused with the actionable error. An overlay holding only ids an
    edit removed is empty."""
    server, ws, project = _serve(tmp_path, monkeypatch)
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("asset_error")[0]["error"]["code"] == "accept_empty"
    server.clients[ws].overrides["gone"] = 1.0
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("asset_error")[1]["error"]["code"] == "accept_empty"
    assert not (project / "params.json").exists()


def test_a_sub_asset_served_alone_takes_no_accept(tmp_path, monkeypatch):
    """DESIGN 4.6: `params.json` holds the values of its project's
    root asset alone, so a sub-asset's knob is set by the asset that
    references it. The refusal names that route."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "subassets/part.star": (
            'w = mo.param("w", 0.5, min = 0.1, max = 3.0)\n'
            'mo.output("p", mo.box(id = "p", size = [w, 1, 1]))\n'
        ),
    }, asset="subassets/part.star")
    asyncio.run(server._set_param(ws, {"params": {"w": 1.0}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    err = ws.messages("asset_error")[0]["error"]
    assert err["code"] == "accept_not_root" and "overrides" in err["message"]
    assert not list(project.rglob("params.json"))
    assert server.clients[ws].overrides == {"w": 1.0}


# -- the check of the accepted state -------------------------------------------


def test_the_accepted_state_is_validated_after_the_reload(tmp_path, monkeypatch):
    """DESIGN 7.2: after the reload that lands an accept the runtime
    runs `validate` over the accepted file state of that project and
    sends every client the report's gating summary, the failing
    entries by id."""
    server, ws, _ = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
        "asset.star": ASSET + INVARIANT,
    })
    other = _StubWS()
    server.clients[other] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 2.2}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("accept_validated") == []
    _reload(server)
    assert ws.messages("accept_validated") == [{
        "type": "accept_validated", "project": server.project, "ok": True,
        "failed": [], "params": ["height"],
    }]
    assert other.messages("accept_validated") == ws.messages("accept_validated")


def test_an_accept_that_can_move_no_entry_is_not_validated(tmp_path, monkeypatch):
    """DESIGN 7.2: an accept that can move no entry of the report is
    not validated, and the reload that lands it answers `checked:
    false` with `moves_no_check`. The accept's event carries no summary
    (DESIGN 6)."""
    events = _events(monkeypatch)
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    assert ws.messages("accepted") == [{"type": "accepted", "params": ["rough"]}]
    assert json.loads((project / "params.json").read_text(encoding="utf-8")) == {"rough": 0.9}
    _reload(server)
    assert server._accept_task is None
    assert ws.messages("accept_validated") == [{
        "type": "accept_validated", "project": server.project,
        "params": ["rough"], "checked": False, "moves_no_check": True,
    }]
    accepts = [f for ev, f in events if ev == "accept"]
    assert [(a["ok"], a["param_ids"], "summary" in a) for a in accepts] == [
        (True, ["rough"], False),
    ]


# A shell around a core, and one knob on each channel of the shell's
# material: what a verdict could read of a material is all here.
CHANNELS = (
    'c = mo.param("color", [0.5, 0.5, 0.5, 1.0], min = 0.0, max = 1.0)\n'
    'm = mo.param("metal", 0.5, min = 0.0, max = 1.0)\n'
    'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
    'e = mo.param("glow", [0.0, 0.0, 0.0], min = 0.0, max = 1.0)\n'
    'n = mo.param("relief", 1.0, min = 0.0, max = 2.0)\n'
    'shell = mo.transform(mo.box(id = "shell_box", size = [2.0, 2.0, 2.0]),'
    ' id = "shell_up", translate = [0, 1.0, 0])\n'
    'mo.output("shell", mo.material(shell, id = "mt", name = "glass", base_color = c,'
    ' metallic = m, roughness = r, emissive = e, normal_scale = n))\n'
    'mo.output("core", mo.transform(mo.box(id = "core_box", size = [0.5, 0.5, 0.5]),'
    ' id = "core_up", translate = [0, 1.0, 0]))\n'
)


def test_only_a_channel_a_verdict_reads_moves_an_entry(tmp_path, monkeypatch):
    """DESIGN 7.2: the channels a verdict reads are declared beside the
    checks, and every other channel moved across its range leaves each
    entry unchanged. A base color's alpha decides whether an output
    encloses another, so a color knob moves one."""
    from moraine3d import ops
    from moraine3d.engine import validation

    server, ws, _ = _serve(tmp_path, monkeypatch, {
        "asset.star": CHANNELS,
        "asset.requirements.star": 'mo.budget(materials = 1)\n',
    })
    knobs = {p["param_id"]: p for p in server.param_payload["params"]}
    assert {k["tier"] for k in knobs.values()} == {"factor"}
    assert {k["uniform"]["channel"] for k in knobs.values()} == set(ops.UNIFORM_RANGES)

    def verdict(pid, end):
        was = knobs[pid]["value"]
        value = [end] * len(was) if isinstance(was, list) else end
        opened = server._open(server.asset_path, {pid: value})
        _, ev, _ = server._evaluate_results({pid: value}, opened=opened)
        report, _ = validation.report(ev.root, ev)
        return report["ok"], validation.failing(report)

    still = {
        pid: verdict(pid, knob["min"]) == verdict(pid, knob["max"])
        for pid, knob in knobs.items()
    }
    assert still == {pid: validation.moves_no_entry(knob) for pid, knob in knobs.items()}
    assert still == {
        "color": False, "metal": True, "rough": True, "glow": True, "relief": True,
    }
    enclosed = {
        "section": "checks", "entry": "not_enclosed", "owner": "builtin",
        "output": "core",
    }
    assert enclosed in verdict("color", 1.0)[1]
    assert enclosed not in verdict("color", 0.0)[1]


def test_an_accept_of_a_color_is_validated(tmp_path, monkeypatch):
    """DESIGN 7.2: a base color's alpha decides whether an output
    encloses another, so a color knob is validated. The shell made
    transparent takes the core's `not_enclosed` out of the verdict."""
    server, ws, _ = _serve(tmp_path, monkeypatch, {"asset.star": CHANNELS})
    asyncio.run(server._set_param(
        ws, {"param_id": "color", "value": [0.5, 0.5, 0.5, 0.4]},
    ))
    _accept(server, ws)
    said = ws.messages("accept_validated")
    assert [("moves_no_check" in m, m["params"]) for m in said] == [(False, ["color"])]
    assert "not_enclosed" not in [row["entry"] for row in said[0]["failed"]]


def test_the_question_is_asked_of_the_state_the_reload_built(tmp_path, monkeypatch):
    """DESIGN 7.2: whether an accept can move a check is asked of the
    state the reload built, so a recipe edit that lands with the accept
    is taken as it stands. A knob the edit put to a second use is no
    longer `factor`-tier, and its accept is validated."""
    server, ws, project = _serve(tmp_path, monkeypatch)
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    edited = ASSET.replace("size = [0.2, h, 0.2]", "size = [0.2 + r, h, 0.2]")
    assert edited != ASSET
    with open(project / "asset.star", "w", encoding="utf-8", newline="\n") as f:
        f.write(edited)
    _reload(server)
    assert ws.messages("accept_validated") == [{
        "type": "accept_validated", "project": server.project, "ok": True,
        "failed": [], "params": ["rough"],
    }]


def test_an_accept_under_a_standing_warning_is_validated(tmp_path, monkeypatch):
    """DESIGN 7.2: an accept made while the warning stands is
    validated all the same, and the warning names the knobs of every
    accept it stands for, so a pick that moved no entry does not take
    the red's name."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    _accept(server, ws)
    said = ws.messages("accept_validated")
    assert [(m["ok"], m["params"], "moves_no_check" in m) for m in said] == [
        (False, ["height"], False), (False, ["height", "rough"], False),
    ]
    # The green that clears it names the same knobs.
    write_files(project, {"asset.requirements.star": REQUIREMENTS.replace("2.4", "3.0")})
    _reload(server)
    last = ws.messages("accept_validated")[-1]
    assert (last["ok"], last["params"]) == (True, ["height", "rough"])


def test_a_factor_accepted_with_another_knob_is_validated(tmp_path, monkeypatch):
    """DESIGN 7.2: the accept that skips the check holds knobs that
    move no entry and no other. One knob of another tier in the pick,
    and the accepted state is validated."""
    server, ws, _ = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    asyncio.run(server._set_param(ws, {"params": {"rough": 0.9, "height": 2.2}}))
    _accept(server, ws)
    assert [m["ok"] for m in ws.messages("accept_validated")] == [True]


def test_a_factor_that_fails_its_own_entry_is_validated(tmp_path, monkeypatch):
    """DESIGN 7.2: an accept of a knob that fails its own entry in the
    file state is validated all the same, since the new value turns the
    entry green."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "params.json": '{\n"rough": 4.0\n}\n',
    })
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.4}))
    _accept(server, ws)
    assert [m["ok"] for m in ws.messages("accept_validated")] == [True]


def test_the_check_never_gates_the_write(tmp_path, monkeypatch):
    """DESIGN 7.2: the check never gates the write. The human's pick
    is a requirement, and a red entry after it says the recipe must
    catch up. The summary holds the failing entries with their owner,
    human-owned first, and nothing that explains them: the agent
    resolves the red, and `validate` tells it the rest."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
        "asset.star": ASSET + INVARIANT,
    })
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    assert json.loads((project / "params.json").read_text(encoding="utf-8")) == {
        "height": 2.8,
    }
    (said,) = ws.messages("accept_validated")
    assert said["ok"] is False
    assert [(r["entry"], r["owner"], r["section"]) for r in said["failed"]] == [
        ("headroom", "human", "asserts"), ("stays_low", "agent", "asserts"),
    ]
    assert said["failed"][0]["site"] == "asset.requirements.star:1"
    assert set(said) == {"type", "project", "ok", "failed", "params"}


def test_the_warning_reaches_every_client_and_a_connecting_one(tmp_path, monkeypatch):
    """DESIGN 7.2: the runtime holds the warning with the project,
    sends it whole to every client each time and replays it to a
    connecting one, so a tab opened after the accept shows what the
    accepting one does."""
    server, ws, _ = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    other = _StubWS()
    server.clients[other] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    (said,) = ws.messages("accept_validated")
    assert said["ok"] is False and said["params"] == ["height"]
    assert other.messages("accept_validated") == [said]
    late = _StubWS()
    server.clients[late] = ClientState()
    asyncio.run(server._send_snapshot(late))
    assert late.messages("accept_validated") == [said]
    # Each reload's re-check and the green verdict reach the same
    # tabs, and a tab connecting after the green has nothing to replay.
    _reload(server)
    write_files(server.project_dir, {
        "asset.requirements.star": REQUIREMENTS.replace("2.4", "3.0"),
    })
    _reload(server)
    assert [m["ok"] for m in late.messages("accept_validated")] == [False, False, True]
    assert ws.messages("accept_validated") == other.messages("accept_validated")
    last = _StubWS()
    server.clients[last] = ClientState()
    asyncio.run(server._send_snapshot(last))
    assert last.messages("accept_validated") == []


def test_a_standing_warning_follows_every_reload_until_it_clears(tmp_path, monkeypatch):
    """DESIGN 7.2: while the warning stands, each watch reload of that
    project re-runs the same `validate` and updates or clears it, so
    it is never stale, and a green project pays nothing."""
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
        "asset.star": ASSET + INVARIANT,
    })
    checks = []
    check = server._accept_verdict
    monkeypatch.setattr(
        server, "_accept_verdict", lambda sc, ev: checks.append(1) or check(sc, ev),
    )
    _reload(server)
    assert checks == []
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    # The agent drops its own invariant: one red entry is left.
    write_files(project, {"asset.star": ASSET})
    _reload(server)
    assert [r["entry"] for r in ws.messages("accept_validated")[-1]["failed"]] == [
        "headroom"
    ]
    # The human revises the requirement: the warning clears.
    write_files(project, {"asset.requirements.star": REQUIREMENTS.replace("2.4", "3.0")})
    _reload(server)
    assert ws.messages("accept_validated")[-1]["ok"] is True
    assert len(checks) == 3
    _reload(server)
    assert len(checks) == 3


def test_a_broken_recipe_keeps_the_check_for_the_reload_that_builds(tmp_path, monkeypatch):
    """DESIGN 7.2: where the accepted state does not build, the
    message carries `checked: false` with the `error`, and the verdict
    waits for a state that does."""
    events = _events(monkeypatch)
    server, ws, project = _serve(tmp_path, monkeypatch)
    other = _StubWS()
    server.clients[other] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    write_files(project, {"asset.star": ASSET + "while True:\n    pass\n"})
    _reload(server)
    (said,) = ws.messages("accept_validated")
    assert said["checked"] is False and said["params"] == ["height"]
    assert said["error"]["code"] == "unsupported_syntax" and "ok" not in said
    assert other.messages("accept_validated") == [said]
    assert [f for ev, f in events if ev == "accept"] == []
    write_files(project, {"asset.star": ASSET})
    _reload(server)
    assert ws.messages("accept_validated")[-1]["ok"] is True
    assert len([f for ev, f in events if ev == "accept"]) == 1


def test_an_accept_written_while_a_check_runs_is_the_next_checks(tmp_path, monkeypatch):
    """DESIGN 7.2: the check judges the state its reload evaluated,
    for the accepts that reload landed. An accept written while it
    runs is judged by the reload its own write brings, never by the
    verdict of the state before it."""
    import threading

    events = _events(monkeypatch)
    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    check = server._accept_verdict
    started, release = threading.Event(), threading.Event()

    def slow(sc, ev):
        started.set()
        release.wait(10)
        return check(sc, ev)

    async def go():
        await server._set_param(ws, {"params": {"height": 2.2}})
        await server._accept(ws, {"type": "accept"})
        monkeypatch.setattr(server, "_accept_verdict", slow)
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        first = server._accept_task
        await asyncio.to_thread(started.wait, 10)
        monkeypatch.setattr(server, "_accept_verdict", check)
        # The second pick breaks the requirement, and its reload has
        # not come when the first check ends.
        await server._set_param(ws, {"params": {"height": 2.8}})
        await server._accept(ws, {"type": "accept"})
        release.set()
        await first
        assert [m["ok"] for m in ws.messages("accept_validated")] == [True]
        assert len(server._accepts) == 1
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        await server._accept_task

    asyncio.run(go())
    assert [m["ok"] for m in ws.messages("accept_validated")] == [True, False]
    accepts = [f for ev, f in events if ev == "accept"]
    assert [a["summary"]["validate_ok"] for a in accepts] == [True, False]


def test_the_check_judges_what_the_reload_evaluated(tmp_path, monkeypatch):
    """DESIGN 7.2: the check judges the state that reload evaluated.
    It builds nothing of its own, so the state it judges is the one
    the viewer shows."""
    from moraine3d.recipe import target as target_mod

    server, ws, _ = _serve(tmp_path, monkeypatch)
    built = []
    opened = target_mod.open_target
    monkeypatch.setattr(
        target_mod, "open_target", lambda *a, **k: built.append(1) or opened(*a, **k),
    )

    async def go():
        await server._set_param(ws, {"params": {"height": 2.5}})
        await server._accept(ws, {"type": "accept"})
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        after_reload = len(built)
        await server._accept_task
        assert len(built) == after_reload

    asyncio.run(go())
    assert ws.messages("accept_validated")[-1]["ok"] is True


def test_the_loops_exit_does_not_wait_for_a_check(tmp_path, monkeypatch):
    """DESIGN 7.2: `validate` over a large asset outlasts what a
    reload may wait, and so what a stopping server may: the check runs
    in a thread the loop's exit leaves behind."""
    import threading
    import time

    release = threading.Event()

    async def go():
        server_mod._apart(release.wait, 30)
        await asyncio.sleep(0)

    t0 = time.perf_counter()
    asyncio.run(go())
    waited = time.perf_counter() - t0
    release.set()
    assert waited < 5


def test_the_check_holds_up_no_reload(tmp_path, monkeypatch):
    """DESIGN 7.2: validating the saved state adds no validation cost
    to a gesture, and the viewer follows the file (DESIGN 8.1) while
    the check runs. A state replaced before its check ended is judged
    by the reload that replaced it."""
    import threading

    server, ws, project = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    check = server._accept_verdict
    started, release = threading.Event(), threading.Event()
    checked = []

    def slow(sc, ev):
        started.set()
        release.wait(10)
        checked.append(1)
        return check(sc, ev)

    async def go():
        monkeypatch.setattr(server, "_accept_verdict", slow)
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        first = server._accept_task
        await asyncio.to_thread(started.wait, 10)
        # The human revises the requirement while the check runs.
        write_files(project, {
            "asset.requirements.star": REQUIREMENTS.replace("2.4", "3.0"),
        })
        monkeypatch.setattr(server, "_accept_verdict", check)
        await server._reload_and_broadcast(drop_overrides=True, trigger="watch")
        assert not first.done()
        # The evaluation gate is free while the asserts are judged, so
        # a screenshot or a pin is not kept waiting.
        assert not server._offloop_eval_gate.locked()
        assert server.asset_payload["stats"]["main"]["bounds"]["max"][1] == 2.8
        release.set()
        await first
        await server._accept_task

    asyncio.run(go())
    assert checked == [1]
    (said,) = ws.messages("accept_validated")
    assert said["ok"] is True and said["params"] == ["height"]


# -- the warning's end ----------------------------------------------------------


def test_the_tool_never_undoes_on_its_own(tmp_path, monkeypatch):
    """DESIGN 7.2: the tool never undoes on its own. If it did, an
    agent-owned red would veto the human's pick."""
    server, ws, project = _serve(tmp_path, monkeypatch, {"asset.star": ASSET + INVARIANT})
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    _reload(server)
    assert json.loads((project / "params.json").read_text(encoding="utf-8")) == {
        "height": 2.8,
    }


def test_a_re_root_ends_the_warning(tmp_path, monkeypatch):
    """DESIGN 7.2: a re-root ends the warning. The warning belongs to
    the project it was raised in."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    from moraine3d.core import workspace as workspace_mod

    root = workspace(tmp_path / "ws", {
        "alpha": {"alpha.star": ASSET + INVARIANT},
        "beta": {"beta.star": ASSET},
    })
    found = workspace_mod.discover(root)
    server = MoraineServer(found.resolve_asset("alpha"), workspace=found, project="alpha")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 2.8}}))
    _accept(server, ws)
    assert server._warning is not None
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert server._warning is None
    late = _StubWS()
    server.clients[late] = ClientState()
    asyncio.run(server._send_snapshot(late))
    assert late.messages("accept_validated") == []


# -- the record ----------------------------------------------------------------


def test_an_accept_no_check_reaches_is_recorded_without_a_verdict(tmp_path, monkeypatch):
    """DESIGN 6: one `accept` per viewer accept, its `validate`
    summary left out where the project was left or the server stopped
    before the check."""
    from moraine3d.core import workspace as workspace_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    events = _events(monkeypatch)
    root = workspace(tmp_path / "ws", {
        "alpha": {"alpha.star": ASSET}, "beta": {"beta.star": ASSET},
    })
    found = workspace_mod.discover(root)
    server = MoraineServer(found.resolve_asset("alpha"), workspace=found, project="alpha")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    asyncio.run(server._set_param(ws, {"params": {"rough": 0.9}}))
    asyncio.run(server._accept(ws, {"type": "accept"}))
    server._settle_unchecked()  # what the shutdown hook runs
    accepts = [f for ev, f in events if ev == "accept"]
    assert [(a["project"], a["param_ids"], a["ok"]) for a in accepts] == [
        ("alpha", ["height"], True), ("beta", ["rough"], True),
    ]
    assert all("summary" not in a for a in accepts)
    assert server._accepts == []


def test_the_accept_event_names_the_knobs_and_the_verdict(tmp_path, monkeypatch):
    """DESIGN 6: one `accept` per viewer accept, with the sorted
    `param_ids`, the outcome, and the `validate` summary of the
    accepted state. Names, never values."""
    events = _events(monkeypatch)
    server, ws, _ = _serve(tmp_path, monkeypatch, {
        "asset.requirements.star": REQUIREMENTS,
    })
    asyncio.run(server._accept(ws, {"type": "accept"}))
    asyncio.run(server._set_param(ws, {"params": {"rough": 0.9, "height": 2.8}}))
    _accept(server, ws)
    accepts = [f for ev, f in events if ev == "accept"]
    assert [(a["ok"], a["code"], a["param_ids"]) for a in accepts] == [
        (False, "accept_empty", []), (True, None, ["height", "rough"]),
    ]
    assert accepts[1]["summary"] == {"validate_ok": False, "failed": ["headroom"]}
    sans_ms = [{k: v for k, v in a.items() if k != "ms"} for a in accepts]
    assert "0.9" not in json.dumps(sans_ms) and "2.8" not in json.dumps(sans_ms)


def test_an_accepts_line_is_no_edited_line(tmp_path, monkeypatch):
    """DESIGN 6: `changed_lines` measures the recipe files' edit
    since the previous file state. The line an accept writes is the
    human's pick, and the revision still moves with it."""
    events = _events(monkeypatch)
    server, ws, _ = _serve(tmp_path, monkeypatch)
    before = server.asset_hash
    asyncio.run(server._set_param(ws, {"params": {"height": 2.5}}))
    _accept(server, ws)
    reload = [f for ev, f in events if ev == "reload"][-1]
    assert reload["changed_lines"] == 0
    assert reload["asset_hash"] == server.asset_hash != before


@pytest.mark.parametrize("name", ["params.json", "other.json"])
def test_a_change_to_the_file_is_a_change_the_watch_takes(tmp_path, monkeypatch, name):
    """DESIGN 4.6: `serve` watches every `params.json`, and editing
    it hot-reloads its dependents. It is watched before it exists, so
    the first accept is an edit like any other. No other JSON a
    project holds is one of the build's files (DESIGN 4)."""
    server, ws, project = _serve(tmp_path, monkeypatch)
    assert (project / "params.json").resolve() in server.watched
    write_files(project, {name: '{\n"height": 2.5\n}\n'})

    async def feed(_root):
        yield [("added", str(project / name))]

    monkeypatch.setattr(server_mod, "awatch", feed)
    asyncio.run(server._watch(None))
    rows = {p["param_id"]: p["value"] for p in server.param_payload["params"]}
    assert rows["height"] == (2.5 if name == "params.json" else 2.0)
