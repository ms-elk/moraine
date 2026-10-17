"""Server-side Viewer Protocol behavior (DESIGN 7.2): screenshot
request validation, the no-viewer fail-fast, and multi-asset serve
(set_asset / workspace roster / project identity). Handler-level
tests — no sockets, no app lifecycle (startup would evaluate the
asset and start the file watcher).
"""

import asyncio
import base64
import json

import pytest

from moraine3d.app import server as server_mod
from moraine3d.app import session as session_mod
from moraine3d.app.server import ClientState, MoraineServer
from moraine3d.cache import records as records_mod
from moraine3d.core import asset as asset_mod
from moraine3d.core import workspace as workspace_mod
from support import projects

_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"outputs": {"main": "bx"}
}
"""


class _Request:
    def __init__(self, body: dict, host: str = "127.0.0.1:8642", headers: dict | None = None,
                 method: str = "POST", query: dict | None = None):
        self._body = body
        self.host = host
        self.headers = headers or {}
        self.method = method
        self.query = query or {}

    async def json(self):
        return self._body


def _post_screenshot(server: MoraineServer, body: dict) -> tuple[int, dict]:
    resp = asyncio.run(server.screenshot(_Request(body)))
    return resp.status, json.loads(resp.body)


def _server(tmp_path) -> MoraineServer:
    return MoraineServer(projects.write_project(tmp_path, _ASSET, name="asset"))


def _ws_server(tmp_path, monkeypatch) -> MoraineServer:
    """Two-project workspace server (distinct box sizes so payloads are
    tellable apart). These tests evaluate for real, so the cache dir is
    isolated — conftest only pins the size cap."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    for name, sx in (("alpha", "1.0"), ("beta", "2.0")):
        projects.write_project(
            ws / name, _ASSET.replace("[1.0, 1.0, 1.0]", f"[{sx}, 1.0, 1.0]"),
            name=name,
        )
    disc = workspace_mod.discover(ws)
    initial = disc.resolve_asset(disc.projects[0])
    return MoraineServer(initial, workspace=disc, project=disc.projects[0])


def _capture_broadcasts(server, monkeypatch) -> list[dict]:
    sent: list[dict] = []

    async def record(msg):
        sent.append(msg)

    monkeypatch.setattr(server, "_broadcast", record)
    return sent


def test_screenshot_no_viewer_fails_fast(tmp_path):
    status, payload = _post_screenshot(_server(tmp_path), {"view": "iso"})
    assert status == 503
    err = payload["error"]
    assert err["code"] == "no_viewer"
    # Actionable: the message names the exact URL to open.
    assert "http://127.0.0.1:8642/" in err["message"]


def test_screenshot_rejects_bad_view(tmp_path):
    status, payload = _post_screenshot(_server(tmp_path), {"view": "sideways"})
    assert status == 400
    assert payload["error"]["code"] == "bad_view"


def test_screenshot_rejects_bad_size(tmp_path):
    status, payload = _post_screenshot(
        _server(tmp_path), {"view": "iso", "size": [8192, 8192]}
    )
    assert status == 400
    assert payload["error"]["code"] == "bad_size"


_PNG_B64 = base64.b64encode(b"\x89PNG fake").decode("ascii")


def _shoot(server, body, monkeypatch, *, reply_extra=None):
    """Drive the async screenshot path handler-level, standing in for
    ws_handler: wait for the request broadcast, resolve the pending
    future with a screenshot_result, and return
    (request_msg, status, response_payload)."""
    sent = _capture_broadcasts(server, monkeypatch)
    server.clients[object()] = ClientState()  # pass the no_viewer gate

    async def go():
        task = asyncio.ensure_future(server.screenshot(_Request(body)))
        while not server.pending_shots and not task.done():
            await asyncio.sleep(0)
        req = next((m for m in sent if m["type"] == "screenshot_request"), None)
        if req is not None:
            reply = {
                "type": "screenshot_result",
                "request_id": req["request_id"],
                "protocol": req["protocol"],
                "png": _PNG_B64,
            }
            reply.update(reply_extra or {})
            server._deliver_screenshot_result(reply)
        resp = await task
        return req, resp.status, json.loads(resp.body)

    return asyncio.run(go())


def test_screenshot_renders_file_state_not_overrides(tmp_path, monkeypatch):
    """The dedicated render session renders the file's current state: a
    client's set_param preview overlay never reaches the shot, and the
    shot leaves the client's state — overlay included — untouched
    (DESIGN 7.2, 11)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    state = ClientState()
    state.overrides = {"width": [3.0, 1.0, 1.0]}
    server.clients[object()] = state
    file_state, *_ = server._evaluate_results()
    overridden, *_ = server._evaluate_results(dict(state.overrides))

    req, status, payload = _shoot(server, {}, monkeypatch)
    assert status == 200
    want = {n: f"/cache/{r['glb_file']}" for n, r in file_state.items() if "glb_file" in r}
    seen = {n: f"/cache/{r['glb_file']}" for n, r in overridden.items() if "glb_file" in r}
    assert req["glb_urls"] == want
    assert req["glb_urls"] != seen  # the overlay really changes the output
    assert state.overrides == {"width": [3.0, 1.0, 1.0]}
    assert server.asset_payload is None and server.param_payload is None


def test_screenshot_request_carries_camera_urls_and_meta(tmp_path, monkeypatch):
    """screenshot_request carries the renderer-neutral resolved camera and
    file-state GLB urls; the response meta names project, asset, content
    hash, version, watch liveness, and the resolved camera — a stale or
    foreign shot is detectable from the report alone (DESIGN 6)."""
    from moraine3d import dist_version
    from moraine3d.core.asset import content_hash as _asset_content_hash

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    req, status, payload = _shoot(
        server, {"view": "front", "size": [64, 64]}, monkeypatch
    )
    assert status == 200
    assert req["protocol"] == 1
    assert set(req["camera"]) == {"position", "target", "up", "fov"}
    assert req["glb_urls"] and all(
        u.startswith("/cache/") and u.endswith(".glb") for u in req["glb_urls"].values()
    )
    assert req["size"] == [64, 64]
    meta = payload["meta"]
    assert meta["project"] == tmp_path.name
    # Workspace-relative: reports get quoted into committed journals, so
    # the meta must never carry an absolute local path.
    assert meta["asset"] == "asset.star"
    assert meta["asset_hash"] == _asset_content_hash(
        {server.asset_path}, server.watch_root
    )
    assert meta["moraine_version"] == dist_version()
    assert meta["watch"] == "alive"
    assert meta["view"] == "front"
    assert meta["camera"]["azimuth"] == 0.0 and meta["camera"]["elevation"] == 0.0
    assert payload["png"] == _PNG_B64


def test_screenshot_free_camera_accepted(tmp_path, monkeypatch):
    """A free camera (azimuth/elevation) resolves without a named view:
    the report carries the angles and no view key (DESIGN 7.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    req, status, payload = _shoot(
        server, {"azimuth": 135.0, "elevation": 15.0}, monkeypatch
    )
    assert status == 200
    meta = payload["meta"]
    assert "view" not in meta
    assert meta["camera"]["azimuth"] == 135.0
    assert meta["camera"]["elevation"] == 15.0


def test_screenshot_camera_conflict_and_bad_camera(tmp_path):
    """Camera validation fails fast with the camera.py codes — before the
    no_viewer gate, like bad_view always has."""
    server = _server(tmp_path)
    status, payload = _post_screenshot(server, {"view": "front", "azimuth": 30.0})
    assert status == 400 and payload["error"]["code"] == "camera_conflict"
    status, payload = _post_screenshot(server, {"elevation": 91.0})
    assert status == 400 and payload["error"]["code"] == "bad_camera"
    status, payload = _post_screenshot(server, {"target": [1.0, 2.0]})
    assert status == 400 and payload["error"]["code"] == "bad_camera"


def test_screenshot_eval_failure_is_loud(tmp_path, monkeypatch):
    """A failing file-state evaluation is the response — never stale
    pixels (DESIGN 6): structured error, no request broadcast,
    no dangling pending shot."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    server.asset_path.write_text("{broken", encoding="utf-8")
    sent = _capture_broadcasts(server, monkeypatch)
    server.clients[object()] = ClientState()
    status, payload = _post_screenshot(server, {})
    assert status == 500
    assert payload["error"]["code"]
    assert not server.pending_shots
    assert [m for m in sent if m.get("type") == "screenshot_request"] == []


def test_screenshot_client_error_reported(tmp_path, monkeypatch):
    """A client that cannot render (GLB fetch/parse failure) reports the
    reason instead of timing out; the server relays it as
    screenshot_failed (DESIGN 2 P2: actionable errors)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    _, status, payload = _shoot(
        server, {}, monkeypatch, reply_extra={"error": "GLB fetch failed", "png": None}
    )
    assert status == 502
    assert payload["error"]["code"] == "screenshot_failed"
    assert "GLB fetch failed" in payload["error"]["message"]


def test_screenshot_stale_result_ignored_current_client_wins(tmp_path, monkeypatch):
    """A result without the protocol echo comes from a pre-upgrade tab
    rendering its own shared asset — ignored, not resolved, so a current
    client's later reply still wins the request (DESIGN 7.2): one stale
    tab must not starve the screenshot channel."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    server.clients[object()] = ClientState()

    async def go():
        task = asyncio.ensure_future(server.screenshot(_Request({})))
        while not server.pending_shots and not task.done():
            await asyncio.sleep(0)
        req = next(m for m in sent if m["type"] == "screenshot_request")
        stale = {
            "type": "screenshot_result",
            "request_id": req["request_id"],
            "png": _PNG_B64,  # no protocol echo: pre-upgrade client
        }
        server._deliver_screenshot_result(stale)
        assert server.pending_shots  # still waiting, not resolved
        good = {**stale, "protocol": req["protocol"]}
        server._deliver_screenshot_result(good)
        resp = await task
        return resp.status, json.loads(resp.body)

    status, payload = asyncio.run(go())
    assert status == 200
    assert payload["png"] == _PNG_B64
    assert not server.stale_shots  # cleaned up on success


def test_screenshot_only_stale_replies_reports_stale_client(tmp_path, monkeypatch):
    """If nobody current answers within the timeout after a stale reply,
    the failure names the real problem — an old client tab — instead of
    'viewer did not respond' (DESIGN 6)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("moraine3d.app.server.SCREENSHOT_TIMEOUT_S", 0.05)
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    server.clients[object()] = ClientState()

    async def go():
        task = asyncio.ensure_future(server.screenshot(_Request({})))
        while not server.pending_shots and not task.done():
            await asyncio.sleep(0)
        req = next(m for m in sent if m["type"] == "screenshot_request")
        server._deliver_screenshot_result(
            {"type": "screenshot_result", "request_id": req["request_id"], "png": _PNG_B64}
        )
        resp = await task
        return resp.status, json.loads(resp.body)

    status, payload = asyncio.run(go())
    assert status == 502
    assert payload["error"]["code"] == "stale_client"
    assert "reload" in payload["error"]["message"]
    assert not server.stale_shots and not server.pending_shots  # no leak


def test_screenshot_non_dict_body_is_not_a_crash(tmp_path):
    """A valid-JSON non-object body (null, a string) must answer through
    the terse-JSON error contract, never aiohttp's HTML 500 (DESIGN 2 P2)."""
    server = _server(tmp_path)
    for body in (None, "iso", [512, 512]):
        status, payload = _post_screenshot(server, body)
        assert status == 503  # falls through to the no_viewer gate
        assert payload["error"]["code"] == "no_viewer"


def test_screenshot_meta_reports_watch_dead(tmp_path, monkeypatch):
    """With the watch dead the shot itself is still fresh (it re-evaluates
    the file), but the report must say the viewer display may be stale."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    asyncio.run(server._mark_watch_dead(RuntimeError("gone")))
    _, status, payload = _shoot(server, {}, monkeypatch)
    assert status == 200
    assert payload["meta"]["watch"] == "dead"


def test_asset_content_hash_names_the_revision(tmp_path):
    """The asset content hash is 24-hex (cache-key convention), covers
    every contributing file, and changes when any file's bytes change or
    a file moves — byte identity is content identity for machine-written
    .mora files (DESIGN 6)."""
    from moraine3d.core.asset import content_hash as _asset_content_hash

    root = tmp_path
    a = root / "alpha" / "alpha.star"
    b = root / "beta" / "beta.star"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text(_ASSET, encoding="utf-8")
    b.write_text(_ASSET.replace("1.0", "2.0"), encoding="utf-8")

    h1 = _asset_content_hash({a, b}, root)
    assert len(h1) == 24 and int(h1, 16) >= 0
    # Set iteration order must not matter (sorted inside).
    assert _asset_content_hash({b, a}, root) == h1
    # Any byte change is a new revision.
    b.write_text(_ASSET.replace("1.0", "3.0"), encoding="utf-8")
    h2 = _asset_content_hash({a, b}, root)
    assert h2 != h1
    # Same bytes at a different path is a different asset set.
    c = root / "gamma" / "gamma.mora"
    c.parent.mkdir()
    c.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    assert _asset_content_hash({c, b}, root) != h2


def test_watch_death_is_reported(tmp_path, monkeypatch):
    """Watch death is detected and reported instead of serving stale
    pixels silently (DESIGN 7.2): the flag is set, the failure is
    broadcast once per outage and logged every time, and the message
    says the watch re-arms (the display may be stale meanwhile)."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._mark_watch_dead(RuntimeError("inotify limit")))
    assert server.watch_dead is not None
    assert server.watch_dead["code"] == "watch_dead"
    assert "re-arms" in server.watch_dead["message"]
    assert "RuntimeError" in server.watch_dead["message"]
    assert sent == [{"type": "watch_dead", "error": server.watch_dead}]
    # A re-arm attempt that dies at birth is the same outage: no flap.
    asyncio.run(server._mark_watch_dead(RuntimeError("still exhausted")))
    assert len(sent) == 1
    events = [e for e in session_mod.read_events(tmp_path / "log") if e["event"] == "watch_dead"]
    assert len(events) == 1 and events[0]["root"] == str(server.watch_root)


def test_run_watch_reports_death_and_returns_it(tmp_path, monkeypatch):
    """Per-event reload failures are retried in the loop, so the one
    failure mode _run_watch cannot retry through is awatch itself dying
    (root deleted, OS watch resources exhausted) — reported through the
    death callback and returned as True so the caller re-arms; shutdown
    cancellation is not a death (False, no flag, no broadcast)."""

    async def _dying(_root):
        raise RuntimeError("ReadDirectoryChangesW exhausted")
        yield  # pragma: no cover - makes this an async generator

    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    monkeypatch.setattr("moraine3d.app.server.awatch", _dying)

    async def noop(_changes):
        pass

    died = asyncio.run(server._run_watch(
        server.watch_root, noop, server._mark_watch_dead, tag="watch_error",
    ))
    assert died is True and server.watch_dead is not None
    assert [m["type"] for m in sent] == ["watch_dead"]

    async def _cancelled(_root):
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    server2 = _server(tmp_path)
    sent2 = _capture_broadcasts(server2, monkeypatch)
    monkeypatch.setattr("moraine3d.app.server.awatch", _cancelled)
    died = asyncio.run(server2._run_watch(
        server2.watch_root, noop, server2._mark_watch_dead, tag="watch_error",
    ))
    assert died is False and server2.watch_dead is None and sent2 == []


def test_watch_rearms_after_death(tmp_path, monkeypatch):
    """The file watch re-arms on death (DESIGN 7.2): a dead watch is
    retried with backoff; an attempt that dies at birth stays the same
    outage (no second watch_dead), and one that outlives the birth
    window is armed — `watch_alive` clears the sticky flag, the re-arm
    is logged, and one catch-up reload re-derives the display from the
    file for the edits that landed while dead."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    monkeypatch.setattr(server_mod, "_REARM_S", 0.01)
    monkeypatch.setattr(server_mod, "_BIRTH_S", 0.05)
    reloads: list[dict] = []

    async def reload(**kw):
        reloads.append(kw)

    monkeypatch.setattr(server, "_reload_and_broadcast", reload)
    attempts: list[int] = []

    async def flaky(_root):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("exhausted")  # dies at birth, twice
        await asyncio.sleep(0.2)  # outlives the birth window: armed
        raise asyncio.CancelledError()  # the test's shutdown
        yield  # pragma: no cover

    monkeypatch.setattr(server_mod, "awatch", flaky)
    asyncio.run(server._watch(None))
    assert len(attempts) == 3
    assert [m["type"] for m in sent] == ["watch_dead", "watch_alive"]
    assert server.watch_dead is None
    assert reloads == [{"drop_overrides": True, "trigger": "watch"}]
    kinds = [e["event"] for e in session_mod.read_events(tmp_path / "log")
             if e["event"].startswith("watch_")]
    assert kinds == ["watch_dead", "watch_alive"]


def test_watch_batches_route_by_reachability(tmp_path, monkeypatch):
    """The one fixed watch root is narrowed per event (DESIGN 7.2): a
    change to an asset the displayed project reaches reloads once, and
    that reload drops every client's preview overlay (the file is the
    sole source of truth); a change to another roster project's asset
    reloads nothing and marks that project changed-since-displayed in
    one workspace broadcast; non-asset churn under the root (an export)
    neither reloads nor broadcasts."""
    server = _ws_server(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    sent = _capture_broadcasts(server, monkeypatch)
    client = _StubWS()
    server.clients[client] = ClientState()
    server.clients[client].overrides = {"width": [3.0, 1.0, 1.0]}
    real_reload = server._reload_and_broadcast
    reloads: list[dict] = []

    async def reload(**kw):
        reloads.append({
            "drop_overrides": kw.get("drop_overrides", False),
            "trigger": kw.get("trigger", "watch"),
            "changed_at_reload": sorted(server.changed),
        })
        await real_reload(**kw)

    monkeypatch.setattr(server, "_reload_and_broadcast", reload)
    batches = [
        [("modified", str(server.asset_path))],
        [("modified", str(ws / "beta" / "beta.star"))],
        [("modified", str(ws / "exports" / "x.glb"))],
    ]

    async def feed(_root):
        for batch in batches:
            yield batch

    monkeypatch.setattr(server_mod, "awatch", feed)
    asyncio.run(server._watch(None))
    # One reload, for the displayed asset, before the sibling was marked.
    assert reloads == [
        {"drop_overrides": True, "trigger": "watch", "changed_at_reload": []}
    ]
    assert server.clients[client].overrides == {}
    kinds = [m["type"] for m in client.messages()]
    assert kinds[:2] == ["preview_dropped", "asset_updated"]
    assert kinds.count("preview_dropped") == 1
    # One workspace broadcast, for the sibling; nothing for the export.
    assert sent == [{
        "type": "workspace", "projects": ["alpha", "beta"], "changed": ["beta"],
    }]
    assert server.changed == {"beta"}


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_listener_half_death_is_probed_reported_and_rearmed(tmp_path, monkeypatch):
    """A listener that stops accepting while its sockets keep streaming
    (DESIGN 7.2 `listen_dead`) is found by the loopback probe, reported
    once per outage (broadcast, session log), re-bound in place, and
    announced alive only once a probe passes through the new socket;
    the re-arm count reaches the status contract."""
    from aiohttp import web

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(server_mod, "_LISTEN_PROBE_S", 0.05)
    monkeypatch.setattr(server_mod, "_LISTEN_PROBE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(server_mod, "_REARM_S", 0.05)
    monkeypatch.setattr(server_mod, "_BIRTH_S", 0.05)
    server = _server(tmp_path)
    server.open_browser = False
    port = _free_port()
    server.port = port
    sent = _capture_broadcasts(server, monkeypatch)

    async def go():
        runner = web.AppRunner(server.build_app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        server._site = site
        watchdog = asyncio.create_task(server._rearm_forever(
            lambda: server._run_listen(runner, port), server._mark_listen_alive,
        ))
        try:
            assert server_mod._site_listening(site) and await server._listen_probe(port)
            # The proactor's failure mode: the listening socket closes,
            # the process and every accepted connection live on.
            site._server.close()
            assert not server_mod._site_listening(site)
            for _ in range(100):
                await asyncio.sleep(0.05)
                if server.listen_rearmed:
                    break
            assert server.listen_rearmed == 1 and server.listen_dead is None
            assert server._site is not None and server._site is not site
            assert await server._listen_probe(port)
            status = json.loads((await server.status_api(_Request({}, method="GET"))).body)
            assert status["listen"] == "alive" and status["listen_rearmed"] == 1
        finally:
            watchdog.cancel()
            await runner.cleanup()

    asyncio.run(go())
    kinds = [m["type"] for m in sent if m["type"].startswith("listen_")]
    assert kinds == ["listen_dead", "listen_alive"]
    events = [e["event"] for e in session_mod.read_events(tmp_path / "log")
              if e["event"].startswith("listen_")]
    assert events == ["listen_dead", "listen_alive"]


def test_listen_probe_budget_is_wall_time_off_the_loop(tmp_path, monkeypatch):
    """The loopback probe's budget is wall time in a worker thread
    (DESIGN 7.2 `listen_dead`): an evaluation blocking the event loop
    for longer than the budget delays the verdict but never turns a
    completed handshake into a timeout, so a healthy listener is never
    declared dead by a slow asset; a closed port still reads False."""
    import socket
    import time

    monkeypatch.setattr(server_mod, "_LISTEN_PROBE_TIMEOUT_S", 0.2)
    server = _server(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        async def stalled():
            probe = asyncio.create_task(server._listen_probe(port))
            await asyncio.sleep(0)  # the probe's thread is under way
            time.sleep(0.5)  # a loop-side evaluation, longer than the budget
            return await probe

        assert asyncio.run(stalled()) is True
    assert asyncio.run(server._listen_probe(port)) is False


def test_rebind_refused_by_a_foreign_listener_is_a_death(tmp_path, monkeypatch):
    """Only a site this server bound counts (DESIGN 7.2 `listen_dead`):
    when another process holds the port, the re-bind is refused and the
    attempt dies — no probe against the foreign listener can announce a
    re-arm, so the server never reports alive on a port it does not
    own."""
    import socket

    from aiohttp import web

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    server.open_browser = False
    sent = _capture_broadcasts(server, monkeypatch)
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", 0))
        foreign.listen()
        port = foreign.getsockname()[1]
        server.port = port

        async def go():
            runner = web.AppRunner(server.build_app(), access_log=None)
            await runner.setup()
            try:
                return await server._run_listen(runner, port)
            finally:
                await runner.cleanup()

        assert asyncio.run(go()) is True  # died at birth: retry later
    assert server._site is None and server.listen_rearmed == 0
    assert server.listen_dead is not None and "re-bind failed" in server.listen_dead["message"]
    assert [m["type"] for m in sent if m["type"].startswith("listen_")] == ["listen_dead"]


def test_serve_stop_is_recorded_at_shutdown(tmp_path, monkeypatch):
    """The session log carries the server's lifetime from the server
    itself (DESIGN 6): one `serve_start` at startup and one `serve_stop`
    at shutdown, both naming the same root and port."""
    from aiohttp import web

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    server.open_browser = False
    server.port = 8642

    async def go():
        runner = web.AppRunner(server.build_app(), access_log=None)
        await runner.setup()  # on_startup: the initial build, serve_start
        await runner.cleanup()  # on_shutdown: serve_stop

    asyncio.run(go())
    events = [e for e in session_mod.read_events(tmp_path / "log")
              if e["event"].startswith("serve_")]
    assert [e["event"] for e in events] == ["serve_start", "serve_stop"]
    start, stop = events
    assert start["root"] == stop["root"] == str(server.watch_root)
    assert start["port"] == stop["port"] == 8642
    assert start["project"] == server.project
    assert start["asset_hash"] == server.asset_hash
    assert "projects" not in start  # single-project serve names no roster


def test_status_api_reports_root_project_and_watch(tmp_path, monkeypatch):
    """`GET /api/status` answers the CLI's liveness probe from memory
    (DESIGN 6): the served root and project, the revision, and whether
    the file watch is armed — `validate`/`stats` carry it as `serve`."""
    server = _server(tmp_path)
    _capture_broadcasts(server, monkeypatch)
    resp = asyncio.run(server.status_api(_Request({}, method="GET")))
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body["root"] == str(server.watch_root) and body["watch"] == "alive"
    assert body["project"] == server.project and body["asset"] == "asset.star"
    assert set(body) == {
        "root", "project", "asset", "asset_hash", "port", "watch",
        "listen", "listen_rearmed",
    }
    assert body["listen"] == "alive" and body["listen_rearmed"] == 0
    asyncio.run(server._mark_watch_dead(RuntimeError("gone")))
    body = json.loads(asyncio.run(server.status_api(_Request({}, method="GET"))).body)
    assert body["watch"] == "dead"
    resp = asyncio.run(server.status_api(
        _Request({}, method="GET", headers={"Origin": "http://evil.example"})
    ))
    assert resp.status == 403


def test_snapshot_replays_watch_dead(tmp_path, monkeypatch):
    """watch_dead is sticky and replayed on connect, after the state
    payloads: a tab opened after the watch died must still learn its
    display will not update (DESIGN 7.2)."""

    class _Stub:
        def __init__(self):
            self.sent: list[str] = []

        async def send_str(self, text):
            self.sent.append(text)

    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())
    _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._mark_watch_dead(RuntimeError("gone")))
    stub = _Stub()
    asyncio.run(server._send_snapshot(stub))
    types = [json.loads(t)["type"] for t in stub.sent]
    assert types == ["workspace", "asset_updated", "param_schema", "watch_dead"]


def test_snapshot_never_replays_listen_dead(tmp_path, monkeypatch):
    """listen_dead is never replayed: a new connection is itself the
    proof the listener accepts, so a tab clears it on connect and the
    snapshot omits it — while watch_dead, an outage the new tab cannot
    see for itself, is still replayed (DESIGN 7.2 `listen_dead`)."""

    class _FakeWS:
        def __init__(self):
            self.sent: list[str] = []

        async def prepare(self, req):
            pass

        async def send_str(self, text):
            self.sent.append(text)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._mark_listen_dead("the listening socket is closed"))
    asyncio.run(server._mark_watch_dead(RuntimeError("gone")))
    # Both outages were told to the tabs connected at the time.
    assert [m["type"] for m in sent] == ["listen_dead", "watch_dead"]
    fake = _FakeWS()
    monkeypatch.setattr("moraine3d.app.server.web.WebSocketResponse", lambda **kw: fake)
    asyncio.run(server.ws_handler(_Request({})))
    types = [json.loads(t)["type"] for t in fake.sent]
    assert types == ["workspace", "asset_updated", "param_schema", "watch_dead"]
    assert "listen_dead" not in types
    # The omission is the snapshot's, not a server-side clear: the
    # outage stays marked until the listener re-arms.
    assert server.listen_dead is not None


def test_asset_updated_carries_project_identity(tmp_path, monkeypatch):
    """`asset_updated` carries the active project's identity in both
    modes (DESIGN 7.2, the select seam); single-asset mode has
    no workspace roster and watches the project directory."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    asset_payload, _ = server._evaluate()
    assert asset_payload["project"] == tmp_path.name
    assert server._workspace_payload() is None
    assert server.watch_root == tmp_path.resolve()


def test_workspace_server_caches_roster_and_watch_root(tmp_path, monkeypatch):
    """Workspace mode carries the discovery roster in the `workspace`
    message and watches the workspace root — one fixed awatch,
    narrowed per-event (DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    assert server._workspace_payload() == {
        "type": "workspace", "projects": ["alpha", "beta"],
        "changed": [],
    }
    assert server.watch_root == (tmp_path / "ws").resolve()
    assert server.project == "alpha"


def test_single_asset_serve_of_workspace_member_watches_workspace(tmp_path, monkeypatch):
    """Single-asset serve of a project inside a workspace watches the
    workspace root, so sibling asset files reached by cross-project refs
    (DESIGN 4.6) sit inside the awatch tree and hot-reload; the mode
    stays single-asset (no roster, no picker)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    projects.write_project(
        ws / "alpha",
        {
            "build_version": "fixture",
            "nodes": [
                {
                    "id": "rf01",
                    "op": "asset_ref",
                    "params": {"project": "beta", "output": "main"},
                }
            ],
            "outputs": {"main": "rf01"},
        },
        name="alpha",
    )
    projects.write_project(ws / "beta", _ASSET, name="beta")
    server = MoraineServer(ws / "alpha" / "alpha.star")
    assert server._workspace_payload() is None
    assert server.watch_root == ws.resolve()
    server._evaluate()
    assert (ws / "beta" / "beta.star").resolve() in server.watched


def test_snapshot_replay_order(tmp_path, monkeypatch):
    """On connect the snapshot replays workspace -> asset_updated ->
    param_schema: the picker must exist client-side before asset_updated
    names the active project to select in it (DESIGN 7.2)."""

    class _Stub:
        def __init__(self):
            self.sent: list[str] = []

        async def send_str(self, text):
            self.sent.append(text)

    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())
    stub = _Stub()
    asyncio.run(server._send_snapshot(stub))
    types = [json.loads(t)["type"] for t in stub.sent]
    assert types == ["workspace", "asset_updated", "param_schema"]


def test_set_asset_switches_active_project(tmp_path, monkeypatch):
    """`set_asset` re-roots the single active asset: identity, watched
    set, and payloads move to the target; every client's set_param
    overlay is dropped (asset-scoped preview, DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    state = ClientState()
    state.overrides = {"stale": 1}
    server.clients[_StubWS()] = state
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert server.project == "beta"
    assert server.asset_path.parent.name == "beta"
    assert server.asset_path in server.watched
    assert state.overrides == {}
    assert server.asset_payload["project"] == "beta"
    # The switch is a real re-root, not a relabel: beta's box is 2 m wide.
    assert server.asset_payload["stats"]["main"]["bounds"]["max"][0] == 1.0


def test_set_asset_rejects_unknown_project(tmp_path, monkeypatch):
    """A target no project directory answers to is refused with the
    valid names and no state change (DESIGN 7.2: name-based
    restriction)."""
    server = _ws_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "gamma"}))
    err = sent[-1]["error"]
    assert err["code"] == "unknown_project"
    assert err["detail"]["projects"] == ["alpha", "beta"]
    assert server.project == "alpha"
    assert server.asset_payload is None  # nothing evaluated


def _add_project(ws, name, sx="3.0"):
    projects.write_project(
        ws / name, _ASSET.replace("[1.0, 1.0, 1.0]", f"[{sx}, 1.0, 1.0]"), name=name,
    )


def test_set_asset_admits_a_project_created_after_startup(tmp_path, monkeypatch):
    """The roster is re-read from disk by a message that resolves a
    name (DESIGN 7.2): a project directory created after start-up is a
    valid `set_asset` target, and the picker learns of it through a
    fresh `workspace` broadcast before the switch lands."""
    server = _ws_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    _add_project(tmp_path / "ws", "gamma")
    asyncio.run(server._set_asset({"type": "set_asset", "project": "gamma"}))
    assert server.project == "gamma"
    assert server.workspace.projects == ("alpha", "beta", "gamma")
    # The roster lands before anything the switch sends.
    assert sent[0]["type"] == "workspace"
    assert sent[0]["projects"] == ["alpha", "beta", "gamma"]


def test_screenshot_and_pins_admit_a_late_project(tmp_path, monkeypatch):
    """The HTTP channels re-read the roster the same way (DESIGN 7.2):
    a screenshot or a pins read addressed to a project created after
    start-up resolves instead of answering `unknown_project`."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    _add_project(tmp_path / "ws", "gamma")
    status, payload = _post_screenshot(server, {"project": "gamma"})
    assert status == 503 and payload["error"]["code"] == "no_viewer"  # resolved
    resp = asyncio.run(server.pins_api(_Request({}, method="GET", query={"project": "gamma"})))
    assert json.loads(resp.body) == {"project": "gamma", "displayed": "alpha", "pins": []}
    assert server.workspace.projects == ("alpha", "beta", "gamma")


def test_watch_rescans_the_roster_for_a_new_or_removed_project(tmp_path, monkeypatch):
    """The watch re-reads the roster when a batch touches an asset file
    directly under a root child the roster does not list, or deletes
    one under a child it does (DESIGN 7.2, 4.2): the picker grows and
    shrinks without a message, and `changed` never names a project
    that is gone."""
    server = _ws_server(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    touched = server._roster_touched
    assert touched([("added", str(ws / "gamma" / "gamma.star"))])
    assert touched([("deleted", str(ws / "beta" / "beta.star"))])
    assert not touched([("modified", str(ws / "alpha" / "alpha.star"))])
    # A directory renamed arrives as its old name deleted and its new
    # name added, with no asset file in the batch.
    assert touched([("deleted", str(ws / "alpha")), ("added", str(ws / "gamma"))])
    assert touched([("added", str(ws / "gamma"))])
    assert not touched([("modified", str(ws / "alpha"))])
    (ws / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
    assert not touched([("modified", str(ws / "AGENTS.md"))])
    assert not touched([("added", str(ws / "gamma" / "gamma.requirements.star"))])
    assert not touched([("added", str(ws / "gamma" / "subassets" / "x.star"))])
    assert not touched([("added", str(ws / "lib" / "x.star"))])
    assert not touched([("added", str(ws / ".git" / "x.star"))])
    sent = _capture_broadcasts(server, monkeypatch)
    _add_project(ws, "gamma")
    assert asyncio.run(server._refresh_roster()) is True
    assert sent[-1] == {
        "type": "workspace", "projects": ["alpha", "beta", "gamma"], "changed": [],
    }
    assert asyncio.run(server._refresh_roster()) is False  # unchanged: no broadcast
    server.changed.add("gamma")
    (ws / "gamma" / "gamma.star").unlink()
    assert asyncio.run(server._refresh_roster()) is True
    assert server.workspace.projects == ("alpha", "beta")
    assert server.changed == set()
    assert sent[-1]["projects"] == ["alpha", "beta"]
    # A project renamed on disk: the new name is served, the old is gone.
    import shutil
    shutil.rmtree(ws / "gamma")
    (ws / "beta").rename(ws / "delta")
    (ws / "delta" / "beta.star").rename(ws / "delta" / "delta.star")
    assert asyncio.run(server._refresh_roster()) is True
    assert server.workspace.projects == ("alpha", "delta")


def test_roster_rescan_failure_keeps_the_last_roster(tmp_path, monkeypatch):
    """A re-scan that the discovery sweep refuses (a reserved name
    holding an asset, DESIGN 4.2) keeps the last roster and surfaces
    as the message's own error (DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    ws = tmp_path / "ws"
    _add_project(ws, "lib")
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert sent[-1]["type"] == "asset_error"
    assert sent[-1]["error"]["code"] == "reserved_project_name"
    assert server.project == "alpha"
    assert server.workspace.projects == ("alpha", "beta")
    status, payload = _post_screenshot(server, {"project": "beta"})
    assert status == 400 and payload["error"]["code"] == "reserved_project_name"


def test_set_asset_rejected_in_single_mode(tmp_path, monkeypatch):
    """Single-asset serve has no valid set_asset targets; the error
    names the fix (restart with the workspace root) — DESIGN 7.2."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "proj"}))
    err = sent[-1]["error"]
    assert err["code"] == "unknown_project"
    assert "workspace-root" in err["message"]
    assert server.asset_path == (tmp_path / "asset.star").resolve()


def test_set_asset_malformed_is_socket_safe(tmp_path, monkeypatch):
    """A malformed set_asset must not take the WebSocket down: it is
    reported as asset_error, never raised (same rule as set_param)."""
    server = _ws_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    for data in ({"type": "set_asset", "project": 5}, {"type": "set_asset"}):
        asyncio.run(server._set_asset(data))
        assert sent[-1]["error"]["code"] == "set_asset_invalid"
    assert server.project == "alpha"


def test_set_asset_resolve_failure_keeps_state(tmp_path, monkeypatch):
    """An unresolvable target swaps nothing — there is nothing to
    watch; the underlying resolution error reaches the client
    (DESIGN 7.2 failure asymmetry)."""
    server = _ws_server(tmp_path, monkeypatch)
    projects.write_project(tmp_path / "ws" / "beta", _ASSET, name="extra")
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert sent[-1]["error"]["code"] == "ambiguous_asset"
    assert server.project == "alpha"
    assert server.asset_path.parent.name == "alpha"


def test_set_asset_eval_failure_swaps_but_keeps_last_good(tmp_path, monkeypatch):
    """A resolvable-but-broken target DOES swap: the display keeps the
    last good state (DESIGN 8.1) while the new asset joins the watched
    set, so fixing its file completes the switch via hot reload — and
    asset_error reports the post-swap active project so the client's
    picker follows the server instead of snapping back (DESIGN 7.2
    failure asymmetry)."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())
    beta_asset = tmp_path / "ws" / "beta" / "beta.star"
    beta_asset.write_text("{not a recipe", encoding="utf-8")
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    err = next(m for m in sent if m["type"] == "asset_error")
    assert err["project"] == "beta"  # the swap is reported
    assert server.asset_payload["project"] == "alpha"  # last good display
    assert server.project == "beta"
    assert beta_asset.resolve() in server.watched


def test_rejected_set_asset_reports_unmoved_project(tmp_path, monkeypatch):
    """A rejected switch reports the unchanged active project on
    asset_error, so the client's picker snaps back only when the server
    really did not move (DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"type": "set_asset", "project": "gamma"}))
    assert sent[-1]["project"] == "alpha"


def test_broadcast_aborts_on_newer_generation(tmp_path, monkeypatch):
    """A send loop suspended on a slow socket must not deliver an older
    asset_updated after a newer evaluation broadcast: the generation
    check aborts the stale loop (DESIGN 7.2 — clients never settle on a
    stale asset)."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())

    stale_sent: list[str] = []
    fast_sent: list[str] = []

    class _Bumping:
        async def send_str(self, text):
            server._payload_gen += 1  # a switch completes mid-send
            stale_sent.append(text)

    class _Fast:
        async def send_str(self, text):
            fast_sent.append(text)

    # Dict insertion order keeps the iteration deterministic (the code
    # copies the keys via list(self.clients) either way).
    server.clients = {_Bumping(): ClientState(), _Fast(): ClientState()}
    asyncio.run(server._broadcast_state())
    assert stale_sent  # the first client began receiving
    assert fast_sent == []  # aborted before the second client


def test_set_asset_recovers_from_broken_startup(tmp_path, monkeypatch):
    """When the initial project fails at startup (no payloads), the
    picker is the recovery path: a set_asset to a healthy sibling
    populates the first good state (DESIGN 8.1)."""
    server = _ws_server(tmp_path, monkeypatch)
    (tmp_path / "ws" / "alpha" / "alpha.star").write_text("{not a recipe", encoding="utf-8")
    asyncio.run(server._reload_and_broadcast())
    assert server.asset_payload is None
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert server.asset_payload["project"] == "beta"


def test_maybe_evict_debounce_and_gates(tmp_path, monkeypatch):
    """serve runs the post-growth eviction check debounced to at most
    every 10 minutes, only after a re-eval that wrote cache bytes, and
    never when the cap is 0 (DESIGN 4.5)."""
    from moraine3d.cache import store as cache_mod

    # Isolate: note_write stamps CACHEDIR.TAG into the active cache dir,
    # which must never be the developer's real cache during tests.
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(cache_mod, "evict", lambda: calls.append(1) or {"evicted": 0})

    # cap 0 (pinned by conftest): never checks, even after writes.
    cache_mod.note_write(100)
    asyncio.run(server._maybe_evict())
    assert calls == []

    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "1000")
    # Zero-write session: no check.
    cache_mod.take_wrote()
    asyncio.run(server._maybe_evict())
    assert calls == []

    # Wrote bytes + cap active: checks once and consumes the counter...
    cache_mod.note_write(100)
    asyncio.run(server._maybe_evict())
    assert calls == [1]
    assert cache_mod.session_wrote() == 0

    # ...then debounces: another write within 10 minutes does not check.
    cache_mod.note_write(100)
    asyncio.run(server._maybe_evict())
    assert calls == [1]

    # Past the window it checks again.
    server._last_evict -= 601.0
    asyncio.run(server._maybe_evict())
    assert calls == [1, 1]


def test_maybe_evict_survives_bad_cap_env(tmp_path, monkeypatch, capsys):
    """A malformed MORAINE_CACHE_MAX_BYTES must never take down serve:
    housekeeping reports through the error channel, it does not kill the
    watch task or the WebSocket handler it runs under."""
    from moraine3d.cache import store as cache_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "4GB")
    server = _server(tmp_path)
    cache_mod.note_write(100)
    asyncio.run(server._maybe_evict())  # must not raise
    assert "evict_error" in capsys.readouterr().out


_PARAM_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"params": [
{"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx", "param": "size", "min": 0.1, "max": 5.0}
],
"outputs": {"main": "bx"}
}
"""


def _capture_broadcasts(server, monkeypatch) -> list[dict]:
    sent: list[dict] = []

    async def record(msg):
        sent.append(msg)

    monkeypatch.setattr(server, "_broadcast", record)
    return sent


class _StubWS:
    """A registrable fake client socket for per-client send assertions."""

    def __init__(self):
        self.sent: list[str] = []

    async def send_str(self, text):
        self.sent.append(text)

    def messages(self) -> list[dict]:
        return [json.loads(t) for t in self.sent]


def test_set_param_reverts_override_on_unexpected_error(tmp_path, monkeypatch):
    """The revert-on-failure covers every exception class: a non-Moraine
    failure that skipped it would leave the poisoned override re-applied
    (and re-failing) on every later set_param, wedging that client's
    overlay until a file edit drops it — and the error reaches the
    overlaying client, never the room (DESIGN 7.2 last-good-state
    contract)."""
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()

    def boom(overrides=None):
        raise ValueError("kernel exploded")

    monkeypatch.setattr(server, "_evaluate", boom)
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [2.0, 2.0, 2.0]}))
    assert server.clients[ws].overrides == {}
    errors = [m for m in ws.messages() if m.get("type") == "asset_error"]
    assert errors and errors[0]["error"]["code"] == "internal"
    assert "project" in errors[0]
    assert sent == []  # nothing about the rejected preview is broadcast


def test_set_param_preview_is_client_local(tmp_path, monkeypatch):
    """A set_param preview is client-local (DESIGN 7.2): the re-evaluated
    asset_updated/param_schema pair reaches only the overlaying client,
    the shared file-state payloads and generation stay untouched, and a
    tab connecting mid-preview replays file state."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    file_asset = server.asset_payload
    gen = server._payload_gen
    a, b = _StubWS(), _StubWS()
    server.clients[a] = ClientState()
    server.clients[b] = ClientState()
    asyncio.run(server._set_param(a, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    # A got its preview pair; B got nothing.
    types = [m["type"] for m in a.messages()]
    assert types == ["asset_updated", "param_schema"]
    assert a.messages()[0]["glb_urls"] != file_asset["glb_urls"]
    assert b.sent == []
    # Shared state is still file state, same generation.
    assert server.asset_payload is file_asset
    assert server._payload_gen == gen
    assert server.clients[a].overrides == {"width": [3.0, 1.0, 1.0]}
    assert server.clients[b].overrides == {}
    # A late tab replays file state, not A's preview.
    late = _StubWS()
    asyncio.run(server._send_snapshot(late))
    replay = [m for m in late.messages() if m["type"] == "asset_updated"]
    assert replay and replay[0]["glb_urls"] == file_asset["glb_urls"]


def test_revert_drops_the_overlay_and_answers_file_state(tmp_path, monkeypatch):
    """Revert is the A/B half of accept (DESIGN 7.2): it drops this
    tab's overlay and replays the file-state pair to it alone —
    announced through the same preview_dropped a reload sends, with no
    re-evaluation, no broadcast, and no generation bump, since nothing
    about file state changed."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    file_asset = server.asset_payload
    gen = server._payload_gen
    sent = _capture_broadcasts(server, monkeypatch)
    a, b = _StubWS(), _StubWS()
    server.clients[a] = ClientState()
    server.clients[b] = ClientState()
    asyncio.run(server._set_param(a, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    a.sent.clear()

    def no_eval(*args, **kwargs):
        raise AssertionError("revert must not evaluate")

    monkeypatch.setattr(server, "_evaluate", no_eval)
    asyncio.run(server._revert(a))
    assert server.clients[a].overrides == {}
    msgs = a.messages()
    assert [m["type"] for m in msgs] == [
        "preview_dropped", "asset_updated", "param_schema",
    ]
    assert msgs[0]["trigger"] == "revert"
    assert msgs[1]["glb_urls"] == file_asset["glb_urls"]
    assert msgs[2] == server.param_payload
    # Client-local, like the preview it undoes: the room hears nothing.
    assert b.sent == [] and sent == []
    assert server.asset_payload is file_asset and server._payload_gen == gen


def test_revert_without_an_overlay_is_refused(tmp_path, monkeypatch):
    """An empty overlay gets the actionable refusal, not a dead control
    (DESIGN 7.2, accept's rule): the tab already shows the file. The
    refusal is not a file-state failure, so it never becomes the
    replayed last_error."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._revert(ws))
    errors = [m for m in ws.messages() if m.get("type") == "asset_error"]
    assert errors and errors[0]["error"]["code"] == "revert_empty"
    assert sent == [] and server.last_error is None


def test_asset_updated_carries_the_stats_asset_rollup(tmp_path, monkeypatch):
    """The viewer HUD's totals and `moraine stats` are one derivation
    (DESIGN 7.2): asset_updated carries the CLI report's asset rollup,
    field for field."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    from moraine3d.app.cli import _stats_report

    assert server.asset_payload["asset"] == _stats_report(asset)["asset"]


_BUDGET_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"params": [
{"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx", "param": "size", "min": 0.1, "max": 5.0}
],
"budgets": {"materials": 2},
"outputs": {"main": "bx"}
}
"""


def test_payload_rollup_omits_the_reports_budget_annotation(tmp_path, monkeypatch):
    """The payload carries the rollup, not the `stats` report's
    declared-budget annotation (DESIGN 7.2): that field is the CLI
    report's own, and showing budgets beside the counts is not a
    shipped display feature — the boundary is pinned so the two cannot
    drift into each other silently."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_BUDGET_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    from moraine3d.app.cli import _stats_report

    payload = server.asset_payload["asset"]
    report = _stats_report(asset)["asset"]
    assert report["materials"]["budget"] == 2
    assert "budget" not in payload["materials"]
    report["materials"].pop("budget")
    assert payload == report


def test_preview_payload_rolls_up_the_preview(tmp_path, monkeypatch):
    """The totals describe what the tab displays (DESIGN 7.2): a
    preview payload's rollup is the preview's, and the shared file-state
    payload keeps the file's."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    file_size = server.asset_payload["asset"]["bounds"]["size"]
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    preview = ws.messages()[0]["asset"]
    assert preview["bounds"]["size"][0] == 3.0
    assert preview["parts"] == 1
    assert server.asset_payload["asset"]["bounds"]["size"] == file_size


def _viewer_events(capsys) -> list[dict]:
    return [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event":"viewer"')
    ]


def test_serve_opens_the_browser_when_nobody_is_watching(tmp_path, monkeypatch, capsys):
    """DESIGN 7.1: once serving, `serve` opens the human's default
    browser on the viewer when no client has connected by the end of
    the grace window - the human meets the viewer when authoring
    starts, not when a screenshot first needs a tab."""
    server = _server(tmp_path)
    server.port = 8642
    server.viewer_open_grace_s = 0
    opened: list[str] = []
    monkeypatch.setattr(server_mod.webbrowser, "open", lambda url: opened.append(url) or True)
    asyncio.run(server._open_viewer())
    assert opened == ["http://127.0.0.1:8642/"]
    assert _viewer_events(capsys) == [{"event": "viewer", "state": "opened"}]


def test_serve_does_not_open_over_a_connected_tab(tmp_path, monkeypatch, capsys):
    """DESIGN 7.1: a client connected within the grace window is a tab
    already showing this server - the ordinary case after a restart,
    since tabs reconnect on their own - so no second tab is opened."""
    server = _server(tmp_path)
    server.port = 8642
    server.viewer_open_grace_s = 0
    server.clients[_StubWS()] = ClientState()

    def never(url):
        raise AssertionError("must not open a browser over a connected tab")

    monkeypatch.setattr(server_mod.webbrowser, "open", never)
    asyncio.run(server._open_viewer())
    assert _viewer_events(capsys) == [{"event": "viewer", "state": "connected"}]


def test_grace_window_is_sampled_not_checked_once(tmp_path, monkeypatch, capsys):
    """DESIGN 7.1: the window is sampled, so a tab whose reconnect the
    browser throttled still wins it - a hidden tab's retry timer can be
    seconds late, and a single check at the end would open a second tab
    over it."""
    server = _server(tmp_path)
    server.port = 8642
    server.viewer_open_grace_s = 5.0
    monkeypatch.setattr(
        server_mod, "_VIEWER_OPEN_POLL_S", 0.01, raising=True,
    )
    monkeypatch.setattr(server_mod.webbrowser, "open", lambda url: (_ for _ in ()).throw(
        AssertionError("must not open over a tab that connected late")))

    async def go():
        task = asyncio.ensure_future(server._open_viewer())
        await asyncio.sleep(0.05)  # well inside the window
        server.clients[_StubWS()] = ClientState()
        await task

    asyncio.run(go())
    assert _viewer_events(capsys) == [{"event": "viewer", "state": "connected"}]


def test_serve_without_a_browser_reports_and_carries_on(tmp_path, monkeypatch, capsys):
    """DESIGN 7.1: a host with no browser gets nothing, silently - the
    open is a convenience for a co-present human, never a requirement
    of the loop - and the event says so; a launcher that raises is the
    same case, never an exception out of the startup task."""
    server = _server(tmp_path)
    server.port = 8642
    server.viewer_open_grace_s = 0
    monkeypatch.setattr(server_mod.webbrowser, "open", lambda url: False)
    asyncio.run(server._open_viewer())
    assert _viewer_events(capsys) == [{"event": "viewer", "state": "unavailable"}]

    def boom(url):
        raise OSError("no display")

    monkeypatch.setattr(server_mod.webbrowser, "open", boom)
    asyncio.run(server._open_viewer())
    assert _viewer_events(capsys) == [{"event": "viewer", "state": "unavailable"}]


def test_open_browser_never_blocks_past_its_budget(monkeypatch):
    """`webbrowser.open` is unbounded - its generic-launcher path waits
    for the browser process to exit, so a `$BROWSER` that stays in the
    foreground would hang the caller (and, on the server, hold an
    executor thread past shutdown). The launcher runs on a daemon
    thread and a launcher still running when the budget expires counts
    as opened: the browser started, it just has not exited."""
    import threading
    import time as time_mod

    started = threading.Event()
    release = threading.Event()

    def hangs(url):
        started.set()
        release.wait(30)  # a foreground browser: never returns in time
        return True

    monkeypatch.setattr(server_mod.webbrowser, "open", hangs)
    monkeypatch.setattr(server_mod, "BROWSER_OPEN_TIMEOUT_S", 0.05)
    t0 = time_mod.perf_counter()
    assert server_mod.open_browser("http://127.0.0.1:8642/") is True
    assert time_mod.perf_counter() - t0 < 5.0
    assert started.is_set()
    release.set()


def test_screenshot_addresses_the_callers_project(tmp_path, monkeypatch):
    """A screenshot renders the project the request addresses, not the
    displayed one (DESIGN 7.2): several sessions share one server and
    one tab, each shooting its own asset through it — and the foreign
    evaluation never reframes the watch set or moves the display."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    watched = set(server.watched)
    req_a, status, shot_a = _shoot(server, {}, monkeypatch)
    assert status == 200 and shot_a["meta"]["project"] == "alpha"
    req_b, status, shot_b = _shoot(server, {"project": "beta"}, monkeypatch)
    assert status == 200
    assert shot_b["meta"]["project"] == "beta"
    assert shot_b["meta"]["asset"] == "beta/beta.star"
    assert req_b["glb_urls"] != req_a["glb_urls"]  # beta's geometry
    assert server.project == "alpha" and server.watched == watched
    status, payload = _post_screenshot(server, {"project": "zeta"})
    assert status == 400 and payload["error"]["code"] == "unknown_project"


def test_single_project_server_refuses_a_foreign_project(tmp_path):
    """A single-project server answers only for the project it serves
    (DESIGN 7.2): a foreign name is `not_served`, never shot as the
    displayed asset under that name; its own name passes the gate."""
    server = _server(tmp_path)
    status, payload = _post_screenshot(server, {"project": "other"})
    assert status == 400 and payload["error"]["code"] == "not_served"
    assert "workspace root" in payload["error"]["message"]
    status, payload = _post_screenshot(server, {"project": tmp_path.name})
    assert status == 503 and payload["error"]["code"] == "no_viewer"  # accepted


def test_pins_answer_only_for_the_displayed_project(tmp_path, monkeypatch):
    """Pins are gestures in what the human sees (DESIGN 7.2): addressed
    to a project the viewer does not display, `list` is empty and names
    the displayed one, and a pin into it is refused (`not_displayed`)
    rather than resolved against the displayed asset."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    resp = asyncio.run(server.pins_api(_Request({}, method="GET", query={"project": "beta"})))
    assert json.loads(resp.body) == {"project": "beta", "displayed": "alpha", "pins": []}
    resp = asyncio.run(server.pins_api(_Request({"project": "beta", "output": "main", "face": 0})))
    assert resp.status == 409
    assert json.loads(resp.body)["error"]["code"] == "not_displayed"
    resp = asyncio.run(server.pins_api(_Request({}, method="DELETE", query={"project": "beta"})))
    assert resp.status == 409
    resp = asyncio.run(server.pins_api(_Request({}, method="GET", query={"project": "alpha"})))
    assert json.loads(resp.body)["project"] == "alpha" and "displayed" not in json.loads(resp.body)
    resp = asyncio.run(server.pins_api(_Request({}, method="GET", query={"project": "zeta"})))
    assert resp.status == 400 and json.loads(resp.body)["error"]["code"] == "unknown_project"


def test_changed_marks_projects_edited_while_another_is_displayed(tmp_path, monkeypatch):
    """`workspace.changed` (DESIGN 7.2): an asset file changing under a
    roster project other than the displayed one marks it, the roster is
    re-sent, and displaying the project clears its mark."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    ws = tmp_path / "ws"
    assert server._note_changed([("modified", str(ws / "alpha" / "alpha.star"))]) == set()
    assert server._note_changed([("modified", str(ws / "beta" / "notes.md"))]) == set()
    assert server._note_changed([("modified", str(ws / "beta" / "beta.star"))]) == {"beta"}
    assert server._note_changed([("modified", str(ws / "beta" / "beta.star"))]) == set()
    assert server._workspace_payload()["changed"] == ["beta"]
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._set_asset({"project": "beta"}))
    assert server.changed == set()
    rosters = [m for m in sent if m["type"] == "workspace"]
    assert rosters and rosters[-1]["changed"] == []


def test_snap_back_keeps_the_overlay_ids_when_the_file_breaks(tmp_path, monkeypatch):
    """The snap-back after a rejected override falls back to the shared
    file-state schema when even loading fails - but this connection's
    overlay is still standing, so `overridden` must still name it
    (DESIGN 7.2): an empty claim would gray out the landings on a
    preview the human can still accept or revert, and with the watch
    dead no reload comes to make it true."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    overrides = {"width": [3.0, 1.0, 1.0]}

    def broken(*a, **kw):
        raise MoraineError("bad_asset", "the file broke underneath")

    monkeypatch.setattr(server, "_overlaid_asset", broken)
    snap = server._preview_schema(overrides)
    assert snap["type"] == "param_schema"
    assert snap["overridden"] == ["width"]
    # The shared payload it borrowed is untouched: file state, no overlay.
    assert server.param_payload["overridden"] == []


def test_param_schema_carries_the_overlays_ids(tmp_path, monkeypatch):
    """`param_schema.overridden` (DESIGN 7.2): the ids this connection's
    overlay holds — file state carries none, a preview carries its
    keys, the snap-back after a rejected override carries what
    survived, and the pair a revert replays carries none again."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    assert server.param_payload["overridden"] == []
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    assert ws.messages()[-1]["type"] == "param_schema"
    assert ws.messages()[-1]["overridden"] == ["width"]
    ws.sent.clear()
    # A rejected override snaps back to the schema of what survived.
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [99.0, 1.0, 1.0]}))
    snap = [m for m in ws.messages() if m["type"] == "param_schema"]
    assert snap and snap[-1]["overridden"] == ["width"]
    ws.sent.clear()
    asyncio.run(server._revert(ws))
    assert ws.messages()[-1]["type"] == "param_schema"
    assert ws.messages()[-1]["overridden"] == []


# -- parameter interaction tiers over the wire (DESIGN 7.2) -------------------

_TIER_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "mt", "op": "material", "inputs": ["bx"], "params": {"name": "paint", "roughness": 0.5}}
],
"params": [
{"id": "rough", "value": 0.5, "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0},
{"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx", "param": "size", "min": 0.1, "max": 5.0}
],
"outputs": {
"main": "mt"
}
}
"""


def _tier_server(tmp_path, monkeypatch) -> MoraineServer:
    """The tier asset served as an IR handed over by path: its knobs
    carry the record that their value reached one node param whole
    (DESIGN 7.2 `direct`), the record the seed and factor tiers read."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_TIER_ASSET, encoding="utf-8", newline="\n")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def test_schema_carries_tier_and_uniform_over_the_wire(tmp_path, monkeypatch):
    """The served param_schema carries each knob's interaction tier, and
    factor entries their uniform target (DESIGN 7.2 param_schema shape)."""
    server = _tier_server(tmp_path, monkeypatch)
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["rough"]["tier"] == "factor"
    assert rows["rough"]["uniform"] == {"material": "paint", "channel": "roughness"}
    assert rows["width"]["tier"] == "evaluation"


def test_factor_set_param_validates_without_evaluating(tmp_path, monkeypatch):
    """A factor-tier set_param updates the overlay and replies with the
    schema alone — no evaluation, no cache write, no asset_updated
    (DESIGN 7.2: the client already drives the uniform live; the exit
    criterion is zero evaluations per factor gesture)."""
    server = _tier_server(tmp_path, monkeypatch)
    before = _persistent_files(tmp_path)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    types = [m["type"] for m in ws.messages()]
    assert types == ["param_schema"]
    rows = {p["param_id"]: p for p in ws.messages()[0]["params"]}
    assert rows["rough"]["value"] == 0.9
    assert server.clients[ws].overrides == {"rough": 0.9}
    assert _persistent_files(tmp_path) == before
    assert not (tmp_path / "cache" / "preview").exists()


def test_factor_set_param_out_of_range_reverts_and_snaps_back(tmp_path, monkeypatch):
    """A rejected factor value takes the same revert + snap-back path as
    an evaluation failure (DESIGN 7.2: last good state stays up, the
    panel snaps back instead of displaying the rejected value)."""
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 3.0}))
    assert server.clients[ws].overrides == {}
    types = [m["type"] for m in ws.messages()]
    assert types == ["asset_error", "param_schema"]
    assert ws.messages()[0]["error"]["code"] == "param_out_of_range"
    rows = {p["param_id"]: p for p in ws.messages()[1]["params"]}
    assert rows["rough"]["value"] == 0.5  # file state, not the rejected 3.0


def test_batch_set_param_is_one_evaluation(tmp_path, monkeypatch):
    """The batch form applies N staged edits in one evaluation (DESIGN
    7.2: Apply must not cost one evaluation per staged knob), and the
    preview pair reflects every key."""
    server = _tier_server(tmp_path, monkeypatch)
    calls = []
    real = server._evaluate

    def counting(overrides=None, pre=None):
        calls.append(dict(overrides or {}))
        return real(overrides, pre)

    monkeypatch.setattr(server, "_evaluate", counting)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(
        ws, {"params": {"width": [2.0, 1.0, 1.0], "rough": 0.9}}
    ))
    assert len(calls) == 1
    assert calls[0] == {"width": [2.0, 1.0, 1.0], "rough": 0.9}
    types = [m["type"] for m in ws.messages()]
    assert types == ["asset_updated", "param_schema"]


def test_one_entry_batch_still_evaluates(tmp_path, monkeypatch):
    """The form carries the contract, not the key count (DESIGN 7.2): a
    one-entry batch naming a factor knob still evaluates and still
    answers with asset_updated — never the factor fast path — and its
    preview event carries param_ids like any batch."""
    from moraine3d.app import server as server_mod

    events = []
    monkeypatch.setattr(
        server_mod.session, "record", lambda ev, **f: events.append((ev, f))
    )
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"rough": 0.4}}))
    types = [m["type"] for m in ws.messages()]
    assert types == ["asset_updated", "param_schema"]
    assert not [f for ev, f in events if ev == "preview"]  # a scrub leaves no event


_SEED_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [0.2, 0.2, 0.2]}},
{"id": "gr", "op": "grid", "params": {}},
{"id": "sc", "op": "scatter", "inputs": ["gr", "bx"], "params": {"count": 4, "seed": 1}}
],
"params": [
{"id": "spread", "value": 1, "node": "sc", "param": "seed", "min": 0, "max": 9999, "step": 1}
],
"outputs": {
"main": "sc"
}
}
"""


def test_seed_set_param_evaluates_every_pick_and_answers_the_pair(tmp_path, monkeypatch):
    """A seed knob evaluates like an evaluation knob, and rerolls are
    never coalesced: each pick is a distinct proposal with its own
    reply, the asset_updated + param_schema pair, never the factor fast
    path (DESIGN 7.2 tier table, seed row)."""
    events = []
    monkeypatch.setattr(
        server_mod.session, "record", lambda ev, **f: events.append((ev, f))
    )
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_SEED_ASSET, encoding="utf-8", newline="\n")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["spread"]["tier"] == "seed"
    calls = []
    real = server._evaluate

    def counting(overrides=None, pre=None):
        calls.append(dict(overrides or {}))
        return real(overrides, pre)

    monkeypatch.setattr(server, "_evaluate", counting)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    for pick in (7, 4242):
        asyncio.run(server._set_param(ws, {"param_id": "spread", "value": pick}))
    # One evaluation per pick, one pair per pick, in order.
    assert calls == [{"spread": 7}, {"spread": 4242}]
    msgs = ws.messages()
    assert [m["type"] for m in msgs] == ["asset_updated", "param_schema"] * 2
    assert msgs[0] != msgs[2]  # two picks, two distinct proposals
    rows = {p["param_id"]: p for p in msgs[-1]["params"]}
    assert rows["spread"]["value"] == 4242 and rows["spread"]["tier"] == "seed"
    assert server.clients[ws].overrides == {"spread": 4242}
    assert not [f for ev, f in events if ev == "preview"]  # a reroll leaves no event


def test_batch_set_param_failure_reverts_every_key(tmp_path, monkeypatch):
    """A failing Apply reverts the whole batch atomically (DESIGN 7.2: a
    partial overlay would preview a state the client never staged)."""
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    server.clients[ws].overrides["rough"] = 0.7  # a prior good override
    asyncio.run(server._set_param(
        ws, {"params": {"width": [9.0, 1.0, 1.0], "rough": 0.9}}  # width out of range
    ))
    assert server.clients[ws].overrides == {"rough": 0.7}
    assert ws.messages()[0]["type"] == "asset_error"
    assert ws.messages()[0]["error"]["code"] == "param_out_of_range"


def test_batch_set_param_rejects_malformed_params(tmp_path, monkeypatch):
    """A malformed batch form is refused without touching the overlay
    (the malformed-message rule of DESIGN 7.2: it must not take the
    WebSocket down, and it must not half-apply)."""
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    for bad in ({}, [], "x", 5):
        asyncio.run(server._set_param(ws, {"params": bad}))
    assert server.clients[ws].overrides == {}
    assert all(m["error"]["code"] == "param_invalid" for m in ws.messages())


def test_reload_announces_dropped_overlay_to_its_owner(tmp_path, monkeypatch):
    """DESIGN 7.2: a reload that discards a client's non-empty overlay
    sends that client preview_dropped — the file winning must read as an
    event, never as the preview silently vanishing — and a client with
    no overlay hears nothing."""
    server = _tier_server(tmp_path, monkeypatch)
    scrubbing = _StubWS()
    idle = _StubWS()
    server.clients[scrubbing] = ClientState()
    server.clients[idle] = ClientState()
    asyncio.run(server._set_param(scrubbing, {"param_id": "rough", "value": 0.9}))
    scrubbing.sent.clear()
    idle.sent.clear()
    asyncio.run(server._reload_and_broadcast(drop_overrides=True, trigger="watch"))
    dropped = [m for m in scrubbing.messages() if m["type"] == "preview_dropped"]
    assert dropped == [{"type": "preview_dropped", "trigger": "watch"}]
    assert server.clients[scrubbing].overrides == {}
    assert all(m["type"] != "preview_dropped" for m in idle.messages())


def test_failing_reload_still_announces_the_drop(tmp_path, monkeypatch):
    """DESIGN 7.2: the announcement rides the drop itself, not the
    broadcast — a file edit that breaks evaluation still dropped the
    overlay, and the client must hear that alongside the asset_error."""
    server = _tier_server(tmp_path, monkeypatch)
    asset = tmp_path / "asset.mora"
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    ws.sent.clear()
    asset.write_text("{not an asset", encoding="utf-8")
    asyncio.run(server._reload_and_broadcast(drop_overrides=True, trigger="watch"))
    types = [m["type"] for m in ws.messages()]
    assert "preview_dropped" in types and "asset_error" in types


def test_wide_declared_range_never_reaches_factor_path(tmp_path, monkeypatch):
    """DESIGN 7.2: a knob whose declared range exceeds the channel's
    valid range classifies evaluation, so an op-invalid value is caught
    by evaluation and reverted, so it cannot poison the overlay
    unevaluated."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(
        _TIER_ASSET.replace(
            '"param": "roughness", "min": 0.0, "max": 1.0',
            '"param": "roughness", "min": 0.0, "max": 2.0',
        ),
        encoding="utf-8", newline="\n",
    )
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["rough"]["tier"] == "evaluation"
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 1.5}))
    assert ws.messages()[0]["type"] == "asset_error"  # the op's own refusal
    assert server.clients[ws].overrides == {}  # reverted, not wedged


_KIT_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "kb", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
{"id": "km", "op": "material", "inputs": ["kb"], "params": {"name": "paint", "roughness": 0.5}}
],
"outputs": {
"main": "km"
}
}
"""


def _kit_server(tmp_path, monkeypatch, ref_params: str) -> MoraineServer:
    """A served root binding ``paint`` (roughness knob) beside a
    reference to a kit whose output binds ``paint`` with the same
    content; ``ref_params`` is the asset_ref's params JSON."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "part.mora").write_text(
        _KIT_ASSET, encoding="utf-8", newline="\n"
    )
    root = _TIER_ASSET.replace(
        '{"id": "mt", "op": "material", "inputs": ["bx"], "params": {"name": "paint", "roughness": 0.5}}',
        '{"id": "mt", "op": "material", "inputs": ["bx"], "params": {"name": "paint", "roughness": 0.5}},\n'
        f'{{"id": "rf", "op": "asset_ref", "params": {ref_params}}}',
    ).replace('"main": "mt"', '"a": "mt",\n"b": "rf"')
    (proj / "proj.mora").write_text(root, encoding="utf-8", newline="\n")
    server = MoraineServer(proj / "proj.mora")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def test_kit_sharing_a_root_material_name_serves_evaluation_tier(tmp_path, monkeypatch):
    """The served schema classifies a root knob ``evaluation`` when an
    un-renamed reference carries the knob's material name into the root
    (DESIGN 7.2): the fact lives in the referenced file, so the server
    reads it off the file-state evaluation and feeds every
    classification — and a scrub of that knob takes the evaluation path,
    never the factor fast path that would repaint the kit's part."""
    server = _kit_server(
        tmp_path, monkeypatch, '{"asset": "subassets/part.mora", "output": "main"}'
    )
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["rough"]["tier"] == "evaluation"
    assert "uniform" not in rows["rough"]
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    assert [m["type"] for m in ws.messages()] == ["asset_updated", "param_schema"]
    rows = {p["param_id"]: p for p in ws.messages()[1]["params"]}
    assert rows["rough"]["tier"] == "evaluation"


def test_renamed_kit_material_serves_factor_tier(tmp_path, monkeypatch):
    """A reference renaming the kit's material carries nothing un-renamed
    into the root (DESIGN 7.2 / 4.6), so the served schema keeps the root
    knob factor."""
    server = _kit_server(
        tmp_path,
        monkeypatch,
        '{"asset": "subassets/part.mora", "output": "main", '
        '"material_rename": {"paint": "kit_paint"}}',
    )
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert rows["rough"]["tier"] == "factor"
    assert rows["rough"]["uniform"] == {"material": "paint", "channel": "roughness"}


def test_factor_overrides_join_a_later_apply(tmp_path, monkeypatch):
    """Factor scrubs stay in the overlay (value-honesty, DESIGN 7.2), so
    a later evaluation-tier apply evaluates with the scrubbed factor
    values included."""
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    asyncio.run(server._set_param(ws, {"params": {"width": [2.0, 1.0, 1.0]}}))
    schema = [m for m in ws.messages() if m["type"] == "param_schema"][-1]
    rows = {p["param_id"]: p for p in schema["params"]}
    assert rows["rough"]["value"] == 0.9
    assert rows["width"]["value"] == [2.0, 1.0, 1.0]


def test_preview_events_carry_tier_and_param_ids(tmp_path, monkeypatch):
    """The session log's preview event names the knob's interaction tier,
    and a batch apply carries sorted param_ids in place of param_id
    (DESIGN 6: names, never values)."""
    from moraine3d.app import server as server_mod

    events = []
    monkeypatch.setattr(
        server_mod.session, "record", lambda ev, **f: events.append((ev, f))
    )
    server = _tier_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [2.0, 1.0, 1.0]}))
    asyncio.run(server._set_param(
        ws, {"params": {"width": [3.0, 1.0, 1.0], "rough": 0.4}}
    ))
    schemas = [m for m in ws.messages() if m["type"] == "param_schema"]
    assert len(schemas) == 3  # the factor path answers with the schema alone
    rows = {p["param_id"]: p for p in schemas[-1]["params"]}
    assert rows["rough"]["tier"] == "factor" and rows["width"]["tier"] == "evaluation"
    assert schemas[-1]["overridden"] == ["rough", "width"]
    assert not [f for ev, f in events if ev == "preview"]


# -- viewer-side accept (DESIGN 7.2) ------------------------------------------


def _knob_server(tmp_path, monkeypatch) -> MoraineServer:
    """A recipe with two knobs, one of them feeding two nodes."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "parts.star").write_text(
        'def post(id, h):\n'
        '    return mo.box(id = id, size = [0.2, h, 0.2])\n',
        encoding="utf-8", newline="\n",
    )
    (tmp_path / "asset.star").write_text(
        'load("lib/parts.star", "post")\n'
        'h = mo.param("height", 2.0, min = 1.0, max = 3.0)\n'
        'r = mo.param("rough", 0.5, min = 0.0, max = 1.0)\n'
        'body = mo.transform(post("post", h), id = "body", translate = [0, h / 2, 0])\n'
        'mo.output("main", mo.material(body, id = "mt", name = "paint", roughness = r))\n',
        encoding="utf-8", newline="\n",
    )
    (tmp_path / "asset.requirements.star").write_text(
        'mo.budget(materials = 2)\n', encoding="utf-8", newline="\n",
    )
    server = MoraineServer(tmp_path / "asset.star")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def test_a_reload_builds_and_watches_what_the_build_read(tmp_path, monkeypatch):
    """DESIGN 4.6: `serve` watches every reachable recipe file, every
    requirements file, every `lib/` file they load, and every
    `params.json`. The IR a reload's own build writes is never one of
    them."""
    server = _knob_server(tmp_path, monkeypatch)
    assert {p.relative_to(tmp_path.resolve()).as_posix() for p in server.watched} == {
        "asset.star", "asset.requirements.star", "lib/parts.star", "params.json",
    }
    assert (tmp_path / "build" / "asset.mora").is_file()
    assert server.asset_payload["outputs"] == {"main": "mt"}
    assert len(server.asset_hash) == 24


def test_a_reload_names_the_revision_it_built(tmp_path, monkeypatch):
    """DESIGN 6: a `reload` carries the revision it reached; a save
    that lands while it evaluates is the next reload's."""
    from moraine3d.engine import evaluator

    server = _knob_server(tmp_path, monkeypatch)
    built = server.asset_hash
    asset = tmp_path / "asset.star"
    results = evaluator.Evaluation.results

    def saved_meanwhile(self, only=None):
        with open(asset, "a", encoding="utf-8", newline="\n") as f:
            f.write("x = 1\n")
        return results(self, only)

    monkeypatch.setattr(evaluator.Evaluation, "results", saved_meanwhile)
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert server.asset_hash == built
    monkeypatch.setattr(evaluator.Evaluation, "results", results)
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert server.asset_hash != built


def test_a_file_written_while_a_reload_evaluates_is_the_next_revision(tmp_path, monkeypatch):
    """DESIGN 6: the hash is over what the build found, so a
    requirements file that was not there when it looked joins the next
    revision."""
    from moraine3d.engine import evaluator

    server = _knob_server(tmp_path, monkeypatch)
    requirements = tmp_path / "asset.requirements.star"
    text = requirements.read_text(encoding="utf-8")
    requirements.unlink()
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    built = server.asset_hash
    results = evaluator.Evaluation.results

    def written_meanwhile(self, only=None):
        with open(requirements, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return results(self, only)

    monkeypatch.setattr(evaluator.Evaluation, "results", written_meanwhile)
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert server.asset_hash == built
    monkeypatch.setattr(evaluator.Evaluation, "results", results)
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert server.asset_hash != built


def test_a_reload_that_breaks_names_what_the_disk_holds(tmp_path, monkeypatch):
    """DESIGN 6: a `reload` carries the revision it reached; one that
    failed before its build read a file names the file as it is on
    disk, never as an earlier reload read it."""
    from moraine3d.core import asset as asset_mod

    server = _knob_server(tmp_path, monkeypatch)
    built = server.asset_hash
    with open(tmp_path / "asset.star", "a", encoding="utf-8", newline="\n") as f:
        f.write("x = 1\n")

    def broken(*a, **k):
        raise RuntimeError("no build")

    monkeypatch.setattr(server, "_open", broken)
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert server.asset_hash != built
    assert server.asset_hash == asset_mod.content_hash_of_bytes(
        {f: f.read_bytes() for f in server.watched if f.exists()},
        server.watch_root,
    )


def test_a_requirements_file_not_yet_written_is_watched(tmp_path, monkeypatch):
    """DESIGN 4.6: `serve` watches every requirements file of the
    closure, so the file the human writes later reloads the asset and
    its first failing requirement reaches the report."""
    server = _knob_server(tmp_path, monkeypatch)
    (tmp_path / "asset.requirements.star").unlink()
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    assert (tmp_path / "asset.requirements.star").resolve() in server.watched
    assert server.asset_hash is not None


def test_an_overlay_the_file_outgrew_is_a_file_state_pass(tmp_path, monkeypatch):
    """DESIGN 7.2: the file wins over a value offered for a knob an
    edit removed. With every overlaid id gone the pass is file state,
    so the IR on disk is the one it evaluated."""
    server = _knob_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    server.clients[ws].overrides = {"height": 2.5}
    asset = tmp_path / "asset.star"
    text = asset.read_text(encoding="utf-8").replace(
        'h = mo.param("height", 2.0, min = 1.0, max = 3.0)', "h = 1.5",
    )
    with open(asset, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    overrides = server.clients[ws].overrides
    opened = server._open(asset.resolve(), overrides)
    assert overrides == {}
    written = (tmp_path / "build" / "asset.mora").read_text(encoding="utf-8")
    assert '"size": [0.2, 1.5, 0.2]' in written
    assert [d["id"] for d in opened.asset.params] == ["rough"]


def test_a_knob_of_a_recipe_moves_through_a_rebuild(tmp_path, monkeypatch):
    """DESIGN 7.2: `bound` is `build` by default, the value reaches the
    graph through a rebuild, and a `build`-bound knob is always
    `evaluation`. It is `direct` only where the value reached one node
    param whole. An apply rebuilds the recipe in-process with the
    overlay's values, so a value used twice moves everywhere it is
    used, and nothing is written."""
    server = _knob_server(tmp_path, monkeypatch)
    schema = {p["param_id"]: p for p in server.param_payload["params"]}
    assert {k: (p["bound"], p["tier"]) for k, p in schema.items()} == {
        "height": ("build", "evaluation"), "rough": ("direct", "factor"),
    }
    assert "node_id" not in schema["height"]
    assert (schema["rough"]["node_id"], schema["rough"]["param"]) == ("mt", "roughness")
    before = (tmp_path / "build" / "asset.mora").read_bytes()
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 3.0}}))
    asset, params = ws.messages()[-2:]
    assert (asset["type"], params["type"]) == ("asset_updated", "param_schema")
    bounds = asset["stats"]["main"]["bounds"]
    assert (bounds["min"][1], bounds["max"][1]) == (0.0, 3.0)
    assert params["overridden"] == ["height"]
    assert (tmp_path / "build" / "asset.mora").read_bytes() == before
    assert server.asset_payload["stats"]["main"]["bounds"]["max"][1] == 2.0


def test_a_preview_outside_the_range_is_refused_and_reverted(tmp_path, monkeypatch):
    """DESIGN 4.6: the declared range is enforced in the build's one
    param resolution, which both consumers feed. DESIGN 7.2: a failing
    override is reverted and reported to the overlaying client."""
    server = _knob_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"params": {"height": 9.0}}))
    err = next(m for m in ws.messages() if m["type"] == "asset_error")
    assert err["error"]["code"] == "param_out_of_range"
    assert server.clients[ws].overrides == {}


def test_a_broken_recipe_keeps_the_last_good_state_and_the_watch(tmp_path, monkeypatch):
    """DESIGN 8.1: a build failure takes the path of an evaluation
    failure, its error naming the recipe file and line, and the viewer
    keeps the last good state. The files the failed build had named
    stay watched, so fixing the one that broke reloads."""
    server = _knob_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    good = server.asset_payload
    (tmp_path / "lib" / "parts.star").write_text(
        'def post(id, h):\n    while h:\n        pass\n', encoding="utf-8",
    )
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    err = next(m for m in sent if m["type"] == "asset_error")
    assert err["error"]["code"] == "unsupported_syntax"
    assert err["error"]["detail"]["site"] == "lib/parts.star:2"
    assert server.asset_payload is good
    assert (tmp_path / "lib" / "parts.star").resolve() in server.watched


def _outside_server(tmp_path, monkeypatch) -> MoraineServer:
    """The tier asset with both knobs declared over ranges the file's
    values sit outside: roughness 0.5 under [0.6, 1], size 1 under
    [2, 5]."""
    text = _TIER_ASSET.replace(
        '"param": "roughness", "min": 0.0', '"param": "roughness", "min": 0.6'
    ).replace('"param": "size", "min": 0.1', '"param": "size", "min": 2.0')
    assert text.count('"min": 0.6') == 1 and text.count('"min": 2.0') == 1
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(text, encoding="utf-8", newline="\n")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def test_schema_carries_a_value_outside_its_range_as_it_is(tmp_path, monkeypatch):
    """A knob whose value sits outside its declared range is served, not
    refused: the asset evaluates and the schema carries the value the
    file holds beside the range (DESIGN 7.2)."""
    server = _outside_server(tmp_path, monkeypatch)
    assert server.asset_payload is not None
    rows = {p["param_id"]: p for p in server.param_payload["params"]}
    assert (rows["rough"]["value"], rows["rough"]["min"]) == (0.5, 0.6)
    assert (rows["width"]["value"], rows["width"]["min"]) == ([1.0, 1.0, 1.0], 2.0)


def test_a_knob_outside_its_range_leaves_its_siblings_usable(tmp_path, monkeypatch):
    """The declaration contract judges the values a client offers
    (DESIGN 4.6, 7.2): a preview of one knob is accepted while another
    knob's file value sits outside its range, and an offered vector is
    judged whole."""
    server = _outside_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.8}))
    assert [m["type"] for m in ws.messages()] == ["param_schema"]
    assert ws.messages()[-1]["overridden"] == ["rough"]
    ws.sent.clear()
    asyncio.run(server._set_param(ws, {"params": {"width": [2.0, 1.0, 1.0]}}))
    assert ws.messages()[0]["error"]["code"] == "param_out_of_range"
    assert ws.messages()[-1]["overridden"] == ["rough"]


def test_an_ir_opened_by_path_takes_no_accept(tmp_path, monkeypatch):
    """DESIGN 7.2: accept writes the overlay to the project's
    `params.json`. DESIGN 6: an IR handed over by path is evaluated as
    given and never classifies a project, so it has no file to accept
    into and nothing is written beside it."""
    server = _tier_server(tmp_path, monkeypatch)
    asset = tmp_path / "asset.mora"
    before = asset.read_bytes()
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "rough", "value": 0.9}))
    ws.sent.clear()
    asyncio.run(server._accept(ws))
    assert ws.messages()[0]["error"]["code"] == "accept_no_project"
    assert asset.read_bytes() == before
    assert not list(tmp_path.rglob("params.json"))


# -- ephemeral preview bucket (DESIGN 4.5) ------------------------------------


def _param_server(tmp_path, monkeypatch) -> MoraineServer:
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    return server


def _persistent_files(tmp_path) -> list[str]:
    cdir = tmp_path / "cache"
    return sorted(p.name for p in cdir.iterdir() if p.is_file())


def test_preview_urls_point_into_the_bucket(tmp_path, monkeypatch):
    """A preview evaluation's dirty outputs serve over /preview/ while
    file-state payloads stay /cache/-only, and the preview's records land
    in the bucket — the persistent store's file set is untouched by
    scrubbing (DESIGN 4.5: the persistent cache indexes file history,
    the bucket indexes exploration)."""
    server = _param_server(tmp_path, monkeypatch)
    assert all(
        u.startswith("/cache/")
        for u in server.asset_payload["glb_urls"].values()
    )
    before = _persistent_files(tmp_path)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    preview = ws.messages()[0]
    assert preview["type"] == "asset_updated"
    assert all(u.startswith("/preview/") for u in preview["glb_urls"].values())
    assert _persistent_files(tmp_path) == before
    bucket = tmp_path / "cache" / "preview"
    served = preview["glb_urls"]["main"].removeprefix("/preview/")
    assert (bucket / served).is_file()


def test_all_pruned_overlay_is_file_state_not_preview(tmp_path, monkeypatch):
    """An overlay whose every id was edited away IS file state (DESIGN
    4.5/7.2): its evaluation populates the persistent store and serves
    over /cache/, never the bucket."""
    server = _param_server(tmp_path, monkeypatch)
    payload, _params = server._evaluate({"gone": 1.0})  # pruned to empty
    assert all(u.startswith("/cache/") for u in payload["glb_urls"].values())


class _FileReq:
    def __init__(self, name: str):
        self.match_info = {"filename": name}


def test_preview_route_serves_bucket_files_only(tmp_path, monkeypatch):
    """The /preview/ route is the /cache/ route's twin (DESIGN 4.5):
    grammar-gated, rooted in the bucket — a name outside the grammar or
    a file living only in the persistent store 404s."""
    import pytest
    from aiohttp import web

    server = _param_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    name = ws.messages()[0]["glb_urls"]["main"].removeprefix("/preview/")
    resp = asyncio.run(server.preview_file(_FileReq(name)))
    assert isinstance(resp, web.FileResponse)
    with pytest.raises(web.HTTPNotFound):
        asyncio.run(server.preview_file(_FileReq("../" + name)))
    # A file-state GLB exists only in the persistent store: the bucket
    # route must not reach up into it.
    persistent = server.asset_payload["glb_urls"]["main"].removeprefix("/cache/")
    with pytest.raises(web.HTTPNotFound):
        asyncio.run(server.preview_file(_FileReq(persistent)))


def test_server_purges_bucket_and_survives_purge_failure(tmp_path, monkeypatch, capsys):
    """The serve start/stop hygiene empties the bucket (DESIGN 4.5:
    session-scoped lifetime), and a failing purge degrades to one stderr
    line — housekeeping must never keep the server from starting or
    stopping."""
    from moraine3d.cache import store as cache_mod

    server = _param_server(tmp_path, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "width", "value": [3.0, 1.0, 1.0]}))
    bucket = tmp_path / "cache" / "preview"
    assert any(bucket.iterdir())
    asyncio.run(server._purge_preview_bucket())
    assert list(bucket.iterdir()) == []

    def boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cache_mod, "purge_preview", boom)
    asyncio.run(server._purge_preview_bucket())  # must not raise
    assert "preview purge skipped" in capsys.readouterr().err


# -- deixis pins (DESIGN 7.2) -------------------------------------------------

_TAG_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}},
{"id": "tg", "op": "tag", "inputs": ["bx"],
 "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}}
],
"outputs": {"main": "tg"}
}
"""

_ARRAY_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "ar", "op": "array", "inputs": ["bx"],
 "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}}
],
"outputs": {"row": "ar"}
}
"""


def _pin_server(tmp_path, monkeypatch, asset_text=_TAG_ASSET) -> MoraineServer:
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(asset_text, encoding="utf-8")
    return MoraineServer(asset)


def test_select_pin_resolves_and_broadcasts(tmp_path, monkeypatch):
    """A viewer pin resolves against file state into structural vocabulary
    — {output, instance, face set, position} — and the full pin state is
    broadcast to every client (DESIGN 7.2); the clicked world position
    wins over the face centroid."""
    server = _pin_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._select(ws, {
        "type": "select", "action": "pin", "output": "main", "face": 4,
        "position": [0.1, 1.0, 0.2],
    }))
    pins_msgs = [m for m in sent if m["type"] == "pins"]
    assert pins_msgs and len(pins_msgs[-1]["pins"]) == 1
    pin = pins_msgs[-1]["pins"][0]
    assert pin["id"] == 1
    assert pin["output"] == "main" and pin["node_id"] == "tg"
    assert pin["instance"] is None and pin["face"] == 4
    assert pin["groups"] == ["top"]
    # The click, re-expressed on the file-state face (never the raw
    # possibly-preview-space point): it stays on the top plane.
    assert pin["position"][1] == 1.0
    assert "bary" in pin and abs(sum(pin["bary"]) - 1.0) < 1e-6
    assert pin["face_center"][1] == 1.0
    assert pin["source"] == "viewer"
    assert len(pin["asset_hash"]) == 24
    assert ws.sent == []  # success answers through the broadcast alone


def test_select_pin_instanced_requires_instance(tmp_path, monkeypatch):
    """On an instanced output the pin carries the raycast's copy index —
    required, never guessed — and the failure answers the pointing
    client only, leaving the store untouched (DESIGN 7.2)."""
    server = _pin_server(tmp_path, monkeypatch, _ARRAY_ASSET)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._select(ws, {
        "type": "select", "action": "pin", "output": "row", "face": 0,
    }))
    assert server.pins == []
    assert [m for m in sent if m["type"] == "pins"] == []
    err = ws.messages()[-1]
    assert err["type"] == "asset_error"
    assert err["error"]["code"] == "missing_instance"
    asyncio.run(server._select(ws, {
        "type": "select", "action": "pin", "output": "row", "face": 0,
        "instance": 2,
    }))
    assert server.pins[-1]["instance"] == 2


def test_select_errors_answer_the_pointing_client(tmp_path, monkeypatch):
    """Malformed and unresolvable selects are structured errors to the
    pointing client, never a broadcast, a silent pick, or a dead socket
    (DESIGN 7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    cases = [
        ({"action": "pin", "output": "nope", "face": 0}, "unknown_output"),
        ({"action": "pin", "output": "main", "face": 9999}, "bad_face"),
        ({"action": "pin", "output": "main", "face": 0, "position": [1.0]}, "pin_invalid"),
        ({"action": "pin", "output": 7, "face": 0}, "pin_invalid"),
        ({"action": "unpin", "id": 42}, "unknown_pin"),
        ({"action": "warp"}, "select_invalid"),
    ]
    for data, code in cases:
        asyncio.run(server._select(ws, {"type": "select", **data}))
        assert ws.messages()[-1]["error"]["code"] == code, data
    assert server.pins == []
    assert [m for m in sent if m["type"] == "pins"] == []


def test_select_replace_is_atomic_sole_selection(tmp_path, monkeypatch):
    """A replace select makes the newly resolved pin the sole selection
    in one mutation — a single pins broadcast, and the store is cleared
    only after a successful resolution, so a failed replace never wipes
    the selection (DESIGN 7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._select(ws, {
        "type": "select", "action": "pin", "output": "main", "face": 0}))
    asyncio.run(server._select(ws, {
        "type": "select", "action": "pin", "output": "main", "face": 2}))
    before = len(sent)
    asyncio.run(server._select(ws, {
        "type": "select", "action": "replace", "output": "main", "face": 4}))
    assert [p["face"] for p in server.pins] == [4]
    replace_msgs = [m for m in sent[before:] if m["type"] == "pins"]
    assert len(replace_msgs) == 1  # one atomic mutation, one broadcast
    asyncio.run(server._select(ws, {
        "type": "select", "action": "replace", "output": "nope", "face": 0}))
    assert [p["face"] for p in server.pins] == [4]  # failure wiped nothing
    assert ws.messages()[-1]["error"]["code"] == "unknown_output"


def test_pin_hash_matches_server_revision_with_refs(tmp_path, monkeypatch):
    """A pin's asset_hash is framed over the full watched file set — the
    same framing as the reload stamp — so a pin on one output is not
    marked stale merely because another output pulls in an asset_ref
    (DESIGN 7.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "part.mora").write_text(_ASSET, encoding="utf-8")
    (proj / "proj.mora").write_text("""{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "rf", "op": "asset_ref", "params": {"asset": "subassets/part.mora", "output": "main"}}
],
"outputs": {"local": "bx", "ref": "rf"}
}
""", encoding="utf-8")
    server = MoraineServer(proj / "proj.mora")
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    pin = asyncio.run(server._add_pin("local", face=0, source="viewer"))
    assert pin["asset_hash"] == server.asset_hash


def test_pin_resolves_against_the_held_file_state(tmp_path, monkeypatch):
    """A pin reads the records the last reload holds, stamped with that
    revision: no build stands between a click and its pin (DESIGN
    7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))

    def no_open(*args, **kwargs):
        raise AssertionError("a pin must not rebuild the asset")

    monkeypatch.setattr(server, "_open", no_open)
    pin = asyncio.run(server._add_pin("main", face=4, source="viewer"))
    assert pin["output"] == "main" and pin["groups"] == ["top"]
    assert pin["asset_hash"] == server.asset_hash


def test_pin_rebuilds_when_the_held_record_is_evicted(tmp_path, monkeypatch):
    """A record evicted from under the held evaluation does not fail the
    pin: resolution falls back to a fresh file-state pass (DESIGN
    7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    _, _, ev = server._file_eval
    rec = ev.records(only="main")["main"]
    rec._value = None  # the value was never loaded into memory
    records_mod.record_paths(rec.key)[0].unlink()
    opens = []
    real_open = server._open
    monkeypatch.setattr(
        server, "_open", lambda *a, **k: opens.append(a) or real_open(*a, **k)
    )
    pin = asyncio.run(server._add_pin("main", face=4, source="viewer"))
    assert pin["groups"] == ["top"] and len(opens) == 1


def test_pin_after_a_re_root_never_reads_the_old_project(tmp_path, monkeypatch):
    """A re-root swaps the asset before its reload lands: a pin in that
    window resolves the displayed asset afresh, never the previous
    project's held evaluation (DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    assert server._file_eval is not None
    disc = server.workspace
    server._bind_asset(disc.resolve_asset("beta"), "beta")
    opens = []
    real_open = server._open
    monkeypatch.setattr(
        server, "_open", lambda *a, **k: opens.append(a) or real_open(*a, **k)
    )
    pin = asyncio.run(server._add_pin("main", face=0, source="viewer"))
    assert len(opens) == 1
    # Face 0 is an x-facing side: beta's box spans 2 m along x.
    assert abs(pin["face_center"][0]) == pytest.approx(1.0)


def test_pins_cap_evicts_fifo(tmp_path, monkeypatch):
    """Pins are a gesture stream, not storage: past the cap the oldest is
    evicted FIFO instead of erroring, and ids are never reused
    (DESIGN 7.2)."""
    from moraine3d.app import server as server_mod

    monkeypatch.setattr(server_mod, "_PINS_MAX", 3)
    server = _pin_server(tmp_path, monkeypatch)
    for _ in range(4):
        asyncio.run(server._add_pin("main", face=4, source="viewer"))
    assert [p["id"] for p in server.pins] == [2, 3, 4]


def test_pins_survive_reload_die_on_set_asset(tmp_path, monkeypatch):
    """Pins survive file reloads — the agent's fix must not erase the
    gesture that asked for it; each pin names the revision it was
    resolved against so staleness is derivable — and a re-root clears
    them with the asset they pointed into, told to every client as an
    empty pins broadcast (DESIGN 7.2)."""
    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._add_pin("main", face=0, source="viewer"))
    placed_hash = server.pins[0]["asset_hash"]
    asyncio.run(server._reload_and_broadcast(drop_overrides=True))
    assert len(server.pins) == 1 and server.pins[0]["asset_hash"] == placed_hash
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    assert server.pins == []
    empty = [m for m in sent if m["type"] == "pins"][-1]
    assert empty["pins"] == [] and empty["project"] == "beta"


def test_snapshot_replays_pins_after_state_pair(tmp_path, monkeypatch):
    """A connecting tab replays existing pins after the state pair —
    markers need the asset they point into — and an empty store replays
    nothing (DESIGN 7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    stub = _StubWS()
    asyncio.run(server._send_snapshot(stub))
    assert [m["type"] for m in stub.messages()] == ["asset_updated", "param_schema"]
    asyncio.run(server._add_pin("main", face=4, source="viewer"))
    late = _StubWS()
    asyncio.run(server._send_snapshot(late))
    assert [m["type"] for m in late.messages()] == [
        "asset_updated", "param_schema", "pins",
    ]


def test_predicate_pin_resolves_to_weighted_centroid(tmp_path, monkeypatch):
    """An agent-placed predicate pin resolves to the area-weighted
    centroid of the matched faces and stores the predicate with its
    selection summary; an empty match is the standard empty_selection
    error, never a silent pick (DESIGN 7.2)."""
    import pytest
    from moraine3d.errors import MoraineError

    server = _pin_server(tmp_path, monkeypatch)
    pin = asyncio.run(server._add_pin(
        "main", predicate={"normal_y": {"gt": 0.9}}, source="api"
    ))
    assert pin["source"] == "api"
    assert pin["predicate"] == {"normal_y": {"gt": 0.9}}
    assert pin["summary"]["matched"] == 2
    assert pin["position"][1] == 1.0  # the 2 m box's top plane
    assert abs(pin["position"][0]) < 1e-6 and abs(pin["position"][2]) < 1e-6
    with pytest.raises(MoraineError) as exc:
        asyncio.run(server._add_pin("main", predicate={"area": {"lt": 0.0}}, source="api"))
    assert exc.value.code == "empty_selection"
    assert len(server.pins) == 1


def test_pins_api_roundtrip(tmp_path, monkeypatch):
    """The agent's pin surface (DESIGN 6): GET reads the active project's
    pins with the revision hash, POST places by predicate or face with
    no viewer tab connected, DELETE removes by id or clears — every
    mutation broadcasting full pin state to the viewers."""
    server = _pin_server(tmp_path, monkeypatch)
    sent = _capture_broadcasts(server, monkeypatch)

    resp = asyncio.run(server.pins_api(_Request({}, method="GET")))
    assert resp.status == 200
    assert json.loads(resp.body)["pins"] == []

    resp = asyncio.run(server.pins_api(_Request(
        {"output": "main", "predicate": {"normal_y": {"gt": 0.9}}}
    )))
    assert resp.status == 200
    out = json.loads(resp.body)
    assert out["pin"]["source"] == "api" and out["pins"] == 1
    assert out["pin"]["summary"]["matched"] == 2
    # The matched-face list is a display-only field: it rides the pins
    # broadcast for the highlight, never the agent's textual surfaces.
    assert "faces" not in out["pin"]
    broadcast = [m for m in sent if m["type"] == "pins"][-1]
    assert broadcast["pins"][0]["faces"] == [4, 5]

    resp = asyncio.run(server.pins_api(_Request({"output": "main", "face": 4})))
    assert resp.status == 200
    face_pin = json.loads(resp.body)["pin"]
    assert face_pin["groups"] == ["top"] and face_pin["face"] == 4

    report = json.loads(asyncio.run(server.pins_api(_Request({}, method="GET"))).body)
    assert [p["id"] for p in report["pins"]] == [1, 2]
    assert all("faces" not in p for p in report["pins"])
    assert report["project"] == server.project

    resp = asyncio.run(server.pins_api(
        _Request({}, method="DELETE", query={"id": "1"})
    ))
    assert json.loads(resp.body) == {"cleared": 1}
    assert [p["id"] for p in server.pins] == [2]
    resp = asyncio.run(server.pins_api(_Request({}, method="DELETE")))
    assert json.loads(resp.body) == {"cleared": 1}
    assert server.pins == []
    assert [m for m in sent if m["type"] == "pins"]  # mutations broadcast


def test_pins_api_rejects_bad_requests(tmp_path, monkeypatch):
    """Pin placement failures are structured, client-fixable errors: a
    body without exactly one of predicate/face, an empty predicate
    match, an unknown id — and the endpoint shares the screenshot
    channel's origin gate (DESIGN 6, 7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    _capture_broadcasts(server, monkeypatch)
    cases = [
        ({"output": "main"}, "pin_invalid"),
        ({"output": "main", "face": 0, "predicate": {"area": {"gt": 0}}}, "pin_invalid"),
        ({"output": "main", "predicate": {"area": {"lt": 0.0}}}, "empty_selection"),
        ({"output": "main", "predicate": {"bogus": {"gt": 0}}}, "bad_predicate"),
    ]
    for body, code in cases:
        resp = asyncio.run(server.pins_api(_Request(body)))
        assert resp.status == 400, body
        assert json.loads(resp.body)["error"]["code"] == code, body
    resp = asyncio.run(server.pins_api(
        _Request({}, method="DELETE", query={"id": "9"})
    ))
    assert resp.status == 400
    assert json.loads(resp.body)["error"]["code"] == "unknown_pin"
    resp = asyncio.run(server.pins_api(_Request(
        {"output": "main", "face": 4},
        headers={"Origin": "http://evil.example"},
    )))
    assert resp.status == 403
    assert server.pins == []


def test_reload_drops_every_clients_overlay(tmp_path, monkeypatch):
    """A file-state reload drops every client's preview overlay and
    broadcasts file state to all — the file is the sole source of truth,
    a reload drops the preview (DESIGN 7.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = tmp_path / "asset.mora"
    asset.write_text(_PARAM_ASSET, encoding="utf-8")
    server = MoraineServer(asset)
    a, b = ClientState(), ClientState()
    a.overrides = {"width": [3.0, 1.0, 1.0]}
    b.overrides = {"width": [2.0, 1.0, 1.0]}
    server.clients[_StubWS()] = a
    server.clients[_StubWS()] = b
    asyncio.run(server._reload_and_broadcast(drop_overrides=True))
    assert a.overrides == {} and b.overrides == {}
    assert server.asset_payload is not None  # file state broadcast to all


def test_reload_reports_unexpected_errors_instead_of_raising(tmp_path, monkeypatch):
    """Failures are reported, never silent (DESIGN 8.2): a raw
    PermissionError from an editor holding the asset file mid-save must
    become an asset_error broadcast, not an exception that kills the
    watch task for the rest of the session."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)

    def locked():
        raise PermissionError(13, "asset.mora held by editor")

    monkeypatch.setattr(server, "_evaluate", locked)
    asyncio.run(server._reload_and_broadcast())
    errors = [m for m in sent if m.get("type") == "asset_error"]
    assert errors and errors[0]["error"]["code"] == "internal"
    assert "PermissionError" in errors[0]["error"]["message"]


def test_snapshot_replays_last_asset_error(tmp_path, monkeypatch):
    """DESIGN 7.2: a tab opened after a failed file-state evaluation
    receives the last asset_error after the state pair (the display is
    the last good state, the error says why it is not the file); a
    successful evaluation clears it, and the last-good asset_updated is
    never relabeled or dropped across a failed switch."""

    class _Stub:
        def __init__(self):
            self.sent: list[str] = []

        async def send_str(self, text):
            self.sent.append(text)

    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast())
    _capture_broadcasts(server, monkeypatch)
    beta_asset = tmp_path / "ws" / "beta" / "beta.star"
    beta_asset.write_text("{not a recipe", encoding="utf-8")
    asyncio.run(server._set_asset({"type": "set_asset", "project": "beta"}))
    stub = _Stub()
    asyncio.run(server._send_snapshot(stub))
    msgs = [json.loads(t) for t in stub.sent]
    assert [m["type"] for m in msgs] == [
        "workspace", "asset_updated", "param_schema", "asset_error",
    ]
    assert msgs[1]["project"] == "alpha"  # last good, untouched
    assert msgs[3]["project"] == "beta" and msgs[3]["error"]["code"] == "recipe_error"
    assert msgs[3]["error"]["detail"]["site"] == "beta/beta.star:1"
    # A rejected set_param leaves the display current: not replayed.
    asyncio.run(server._set_asset({"type": "set_asset", "project": "alpha"}))
    ws = _StubWS()
    server.clients[ws] = ClientState()
    asyncio.run(server._set_param(ws, {"param_id": "nope", "value": 1}))
    stub = _Stub()
    asyncio.run(server._send_snapshot(stub))
    assert [json.loads(t)["type"] for t in stub.sent] == [
        "workspace", "asset_updated", "param_schema",
    ]


def test_screenshot_endpoint_checks_origin(tmp_path):
    """DESIGN 7.2 single-user contract: a browser page reaches
    /api/screenshot only from this server's own origin — http, a
    loopback host, the bound port; a foreign host, another loopback
    port, another scheme, or an opaque Origin is refused with JSON 403,
    and a missing Origin (the CLI's urllib) passes through to the
    handler."""
    server = _server(tmp_path)
    server.port = 8642
    for origin in (
        "http://evil.example", "http://localhost:3000", "http://127.0.0.1:8643",
        "https://localhost:8642", "http://localhost", "null", "http://localhost:x",
    ):
        resp = asyncio.run(server.screenshot(
            _Request({"view": "iso"}, headers={"Origin": origin})
        ))
        assert resp.status == 403, origin
        assert json.loads(resp.body)["error"]["code"] == "forbidden"
    for headers in (
        {}, {"Origin": "http://127.0.0.1:8642"}, {"Origin": "http://localhost:8642"},
        {"Origin": "http://[::1]:8642"},
    ):
        resp = asyncio.run(server.screenshot(_Request({"view": "iso"}, headers=headers)))
        assert resp.status == 503, headers  # reached the no_viewer gate
    # A port-less Origin is the scheme's default port: admitted only when
    # that is the bound port.
    server.port = 80
    resp = asyncio.run(server.screenshot(
        _Request({"view": "iso"}, headers={"Origin": "http://localhost"})
    ))
    assert resp.status == 503
    # Unbound (no port yet) admits no browser page.
    server.port = None
    resp = asyncio.run(server.screenshot(
        _Request({"view": "iso"}, headers={"Origin": "http://localhost:8642"})
    ))
    assert resp.status == 403


def test_ws_door_checks_origin(tmp_path, monkeypatch):
    """DESIGN 7.2 security boundary: `set_asset` rests on WebSocket
    `Origin` checking, so a browser page opens /ws only from this
    server's own origin. A foreign host, another loopback port, or
    another scheme is refused at the door (403, no handshake, no
    snapshot, no client), while the server's own origin and a missing
    Origin (the CLI, tools) complete the handshake and get the
    snapshot."""
    import pytest
    from aiohttp import web

    class _FakeWS:
        def __init__(self):
            self.prepared = False
            self.sent: list[str] = []

        async def prepare(self, req):
            self.prepared = True

        async def send_str(self, text):
            self.sent.append(text)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    server = _server(tmp_path)
    server.port = 8642
    server.asset_payload = {"type": "asset_updated"}
    for origin in (
        "http://evil.example", "http://127.0.0.1:8643", "https://localhost:8642",
    ):
        fake = _FakeWS()
        monkeypatch.setattr(
            "moraine3d.app.server.web.WebSocketResponse", lambda _f=fake, **kw: _f
        )
        with pytest.raises(web.HTTPForbidden) as info:
            asyncio.run(server.ws_handler(_Request({}, headers={"Origin": origin})))
        assert info.value.status == 403 and "cross-origin" in info.value.text, origin
        assert not fake.prepared and fake.sent == [] and server.clients == {}, origin
    for headers in (
        {}, {"Origin": "http://127.0.0.1:8642"}, {"Origin": "http://localhost:8642"},
    ):
        fake = _FakeWS()
        monkeypatch.setattr(
            "moraine3d.app.server.web.WebSocketResponse", lambda _f=fake, **kw: _f
        )
        ws = asyncio.run(server.ws_handler(_Request({}, headers=headers)))
        assert ws is fake and fake.prepared, headers
        assert [json.loads(t)["type"] for t in fake.sent] == ["asset_updated"], headers


def test_ws_client_dying_mid_snapshot_leaves_no_client(tmp_path, monkeypatch):
    """A socket that dies while the connect snapshot is sent leaves no
    client registered: the client roster is what the screenshot
    endpoint's no-viewer fail-fast consults (DESIGN 7.2), and a phantom
    entry would turn ``no_viewer`` into a render timeout."""
    import pytest

    class _DeadWS:
        async def prepare(self, req):
            pass

        async def send_str(self, text):
            raise ConnectionResetError("gone")

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    dead = _DeadWS()
    monkeypatch.setattr("moraine3d.app.server.web.WebSocketResponse", lambda **kw: dead)
    server = _server(tmp_path)
    server.asset_payload = {"type": "asset_updated"}
    with pytest.raises(ConnectionResetError):
        asyncio.run(server.ws_handler(_Request({})))
    assert dead not in server.clients


def test_screenshot_non_utf8_body_is_not_a_crash(tmp_path):
    """A body that is not UTF-8 answers through the JSON error contract
    like a non-object body, never aiohttp's HTML 500 (DESIGN 2 P2)."""

    class _BadBody(_Request):
        async def json(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    resp = asyncio.run(_server(tmp_path).screenshot(_BadBody({})))
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["code"] == "no_viewer"


def test_ws_non_object_messages_are_ignored(tmp_path, monkeypatch):
    """A valid-JSON non-object WebSocket frame ([1,2], null) is skipped,
    never an AttributeError that closes that client's socket."""
    from aiohttp import WSMsgType

    class _Msg:
        def __init__(self, data):
            self.type = WSMsgType.TEXT
            self.data = data

    class _FakeWS:
        def __init__(self, frames):
            self._frames = frames
            self.sent: list[str] = []

        async def prepare(self, req):
            pass

        async def send_str(self, text):
            self.sent.append(text)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._frames:
                raise StopAsyncIteration
            return _Msg(self._frames.pop(0))

    frames = ["[1, 2]", "null", '"set_param"', "{not json",
              '{"type": "set_asset", "project": 7}']
    fake = _FakeWS(frames)
    monkeypatch.setattr("moraine3d.app.server.web.WebSocketResponse", lambda **kw: fake)
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    ws = asyncio.run(server.ws_handler(_Request({})))
    assert ws is fake
    # Only the well-formed-but-invalid set_asset produced a report.
    assert [m["error"]["code"] for m in sent] == ["set_asset_invalid"]
    assert fake not in server.clients  # the handler ran to its end and cleaned up


# -- session timeline (DESIGN 7.2 ``timeline``) --------------------------------
# Round semantics are pinned in test_session.py; these pin the transport:
# congruence with the CLI report, replay, dedupe, and the stale flag.


def test_timeline_payload_is_the_cli_reports_session(tmp_path, monkeypatch):
    """The timeline's session object is the CLI report's, field for
    field, so the panel and ``moraine sessions`` read the same numbers
    (DESIGN 7.2 ``timeline``) — and the payload carries names, never
    paths (DESIGN 6)."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    root = str(server.watch_root)
    session_mod.record(
        "serve_start", root=root, port=8642,
        project=server.project, asset_hash="a" * 24,
    )
    session_mod.record(
        "reload", root=root, project=server.project, trigger="watch",
        ok=True, ms=5, asset_hash="b" * 24, changed_lines=3,
    )
    session_mod.record(
        "command", cmd="validate", ok=True, ms=20, out_bytes=100,
        root=root, project=server.project, asset_hash="b" * 24,
        summary={"outputs": 1, "failed_outputs": 0, "failed_checks": [],
                 "asserts_failed": 0, "budgets_failed": 0, "warnings": 0},
    )
    asyncio.run(server._refresh_timeline())
    assert [m["type"] for m in sent] == ["timeline"]
    payload = server.timeline_payload
    report = session_mod.summarize(
        session_mod.read_events(session_mod.log_dir()), server.watch_root,
        last=1,
    )
    assert payload["session"] == report["sessions"][-1]
    assert payload["root"] == report["root"]
    assert payload["session"]["rounds"] == 1
    # Names, never paths: the JSON-escaped absolute root must not occur.
    assert json.dumps(root)[1:-1] not in json.dumps(payload)


def test_timeline_absent_when_logging_off(tmp_path, monkeypatch):
    """Absence means logging is off: with the log disabled no timeline
    message is ever sent, so the panel never appears (DESIGN 7.2
    ``timeline``, the workspace-absence rule)."""
    monkeypatch.setenv(session_mod.LOG_DIR_ENV, "")
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._refresh_timeline())
    assert server.timeline_payload is None
    assert sent == []


def test_timeline_rebroadcast_deduped(tmp_path, monkeypatch):
    """Day files are machine-shared: a log write under an unrelated root
    wakes the watch with an identical derivation for this root, which is
    not re-broadcast (DESIGN 7.2 ``timeline``)."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    session_mod.record("serve_start", root=str(server.watch_root), port=8642)
    asyncio.run(server._refresh_timeline())
    session_mod.record(
        "command", cmd="stats", ok=True, ms=1, out_bytes=10,
        root=str(tmp_path) + "-elsewhere",
    )
    asyncio.run(server._refresh_timeline())
    assert len([m for m in sent if m["type"] == "timeline"]) == 1


def test_timeline_replayed_in_snapshot(tmp_path):
    """The timeline is replayed on connect when available — after the
    state pair — so a tab opened mid-session reads the same economy a
    live tab does; None replays nothing (DESIGN 7.2 ``timeline``)."""
    server = _server(tmp_path)
    server.asset_payload = {"type": "asset_updated"}
    server.param_payload = {"type": "param_schema"}
    server.timeline_payload = {"type": "timeline", "root": "r", "session": {}}
    ws = _StubWS()
    asyncio.run(server._send_snapshot(ws))
    assert [m["type"] for m in ws.messages()] == [
        "asset_updated", "param_schema", "timeline",
    ]
    server.timeline_payload = None
    ws2 = _StubWS()
    asyncio.run(server._send_snapshot(ws2))
    assert "timeline" not in [m["type"] for m in ws2.messages()]


def test_timeline_watch_death_marks_stale(tmp_path, monkeypatch):
    """Log-watch death is detection and report only: the payload gains
    ``stale: true``, is broadcast, and replays sticky on connect — a
    frozen panel must say it is frozen rather than show a silently
    stale economy (DESIGN 7.2 ``timeline``)."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    server.timeline_payload = {"type": "timeline", "root": "r", "session": {}}
    asyncio.run(server._mark_timeline_stale("boom"))
    assert sent[-1]["type"] == "timeline" and sent[-1]["stale"] is True
    ws = _StubWS()
    asyncio.run(server._send_snapshot(ws))
    replayed = [m for m in ws.messages() if m["type"] == "timeline"]
    assert replayed and replayed[0]["stale"] is True


def test_stale_mark_without_payload_sends_nothing(tmp_path, monkeypatch):
    """With no payload there is no panel to freeze, and a session-less
    stale message would break the message's own shape (DESIGN 7.2
    ``timeline``: nothing is sent)."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    asyncio.run(server._mark_timeline_stale("boom"))
    assert server.timeline_payload is None
    assert sent == []


def test_timeline_marks_stale_when_the_session_vanishes(tmp_path, monkeypatch):
    """A derivation that vanishes under an existing payload freezes the
    panel and says so (DESIGN 7.2 ``timeline`` ``stale``): the old
    economy must not stay up as if live."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    session_mod.record("serve_start", root=str(server.watch_root), port=8642)
    asyncio.run(server._refresh_timeline())
    assert server.timeline_payload is not None
    for f in session_mod.log_dir().glob("*.jsonl"):
        f.unlink()
    asyncio.run(server._refresh_timeline())
    assert server.timeline_payload["stale"] is True
    assert sent[-1]["stale"] is True


def test_watch_log_returns_when_logging_is_off(tmp_path, monkeypatch):
    """Logging off means no timeline and no watch (DESIGN 7.2
    ``timeline``: absence means logging is off)."""
    monkeypatch.setenv(session_mod.LOG_DIR_ENV, "")
    server = _server(tmp_path)
    asyncio.run(server._watch_log(None))
    assert server.timeline_payload is None


def test_watch_log_waits_for_a_late_log_directory(tmp_path, monkeypatch):
    """A log directory missing at serve start may still appear when a
    later writer succeeds — the watch waits for it instead of leaving
    the panel dead for the server's lifetime (DESIGN 7.2
    ``timeline``)."""
    server = _server(tmp_path)
    monkeypatch.setattr(server_mod, "_LOG_DIR_RETRY_S", 0.01)
    sent = _capture_broadcasts(server, monkeypatch)

    async def go():
        task = asyncio.ensure_future(server._watch_log(None))
        await asyncio.sleep(0.05)
        assert server.timeline_payload is None
        session_mod.record("serve_start", root=str(server.watch_root), port=1)
        for _ in range(200):
            if server.timeline_payload is not None:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    assert server.timeline_payload is not None
    assert sent and sent[-1]["type"] == "timeline"


_REF_PARAM_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "rf", "op": "asset_ref", "params": {"asset": "subassets/part.mora", "output": "main"}}
],
"params": [
{"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx", "param": "size", "min": -1.0, "max": 5.0}
],
"outputs": {"local": "bx", "ref": "rf"}
}
"""


def _ref_param_server(tmp_path, monkeypatch) -> MoraineServer:
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "part.mora").write_text(_ASSET, encoding="utf-8")
    (proj / "proj.mora").write_text(_REF_PARAM_ASSET, encoding="utf-8")
    return MoraineServer(proj / "proj.mora")


def test_failed_preview_keeps_the_watch_set(tmp_path, monkeypatch):
    """A preview evaluation never reframes the hot-reload watch set — a
    failing override must not shrink it, or a later sub-asset edit stops
    reloading until the root itself is touched (DESIGN 7.2: the reload
    channel renders file state; previews are per-client overlays)."""
    import pytest
    from moraine3d.errors import MoraineError

    server = _ref_param_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    before = set(server.watched)
    assert len(before) == 2  # the root and the reference target
    with pytest.raises(MoraineError):
        # Passes the declared range, fails the op: the evaluation dies
        # before the reference output runs, so its file reach is partial.
        server._evaluate_results({"width": [-0.5, 1.0, 1.0]})
    assert set(server.watched) == before


def test_pin_resolves_while_another_outputs_ref_dangles(tmp_path, monkeypatch):
    """The watched set legitimately holds a dangling reference target
    (kept so re-creating it hot-reloads); a pin on a healthy output
    still resolves there, stamped over the readable files — the same
    sanctioned skip the reload stamp uses (DESIGN 7.2)."""
    server = _ref_param_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    (tmp_path / "proj" / "subassets" / "part.mora").unlink()
    asyncio.run(server._reload_and_broadcast())  # fails, keeps last good state
    pin = asyncio.run(server._add_pin("local", face=0, source="viewer"))
    assert pin["output"] == "local" and pin["asset_hash"]


def test_screenshot_result_with_unhashable_request_id_is_ignored(tmp_path, monkeypatch):
    """A malformed message never takes the WebSocket down: a
    screenshot_result whose request_id is not a string is dropped like
    any other malformed field, not raised out of the message loop
    (DESIGN 7.2)."""
    server = _pin_server(tmp_path, monkeypatch)
    server._deliver_screenshot_result(
        {"type": "screenshot_result", "request_id": ["x"]}
    )  # must not raise
    assert server.pending_shots == {}


def test_watch_forever_keeps_backoff_across_a_late_death(tmp_path, monkeypatch):
    """The shared re-arm discipline (DESIGN 7.2): a watch that dies after
    the birth window is announced dead again, and because it did not
    hold for the stable period the backoff keeps doubling rather than
    resetting — no flapping every few seconds; a watch that held for
    the stable period resets it."""
    server = _server(tmp_path)
    monkeypatch.setattr(server_mod, "_REARM_S", 0.01)
    monkeypatch.setattr(server_mod, "_BIRTH_S", 0.02)
    monkeypatch.setattr(server_mod, "_STABLE_S", 10.0)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(d):
        sleeps.append(d)
        await real_sleep(0)

    monkeypatch.setattr(server_mod.asyncio, "sleep", fake_sleep)
    events: list[str] = []

    async def on_dead(e):
        events.append("dead")

    async def on_alive():
        events.append("alive")

    async def noop(_changes):
        pass

    attempts: list[int] = []

    async def flaky(_root):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first death")
        if len(attempts) == 2:
            await real_sleep(0.05)  # outlives the birth window, then dies late
            raise RuntimeError("late death")
        await real_sleep(0.05)
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    monkeypatch.setattr(server_mod, "awatch", flaky)
    asyncio.run(server._watch_forever(
        server.watch_root, noop, on_dead, on_alive, tag="t",
    ))
    assert events == ["dead", "alive", "dead", "alive"]
    # Two waits: the first at the base delay, the second doubled — the
    # late death did not reset the schedule.
    assert sleeps == [0.01, 0.02]


def test_timeline_stale_clears_when_the_log_watch_rearms(tmp_path, monkeypatch):
    """The log watch re-arms under the same discipline as the asset
    watch (DESIGN 7.2 `timeline`): the frozen flag comes off the payload
    and is broadcast, and a refresh follows."""
    server = _server(tmp_path)
    sent = _capture_broadcasts(server, monkeypatch)
    refreshed: list[int] = []

    async def refresh():
        refreshed.append(1)

    monkeypatch.setattr(server, "_refresh_timeline", refresh)
    server.timeline_payload = {"type": "timeline", "root": "r", "gap_minutes": 30,
                               "session": {}}
    asyncio.run(server._mark_timeline_stale("gone"))
    assert sent[-1]["stale"] is True
    asyncio.run(server._mark_timeline_fresh())
    assert "stale" not in sent[-1] and "stale" not in server.timeline_payload
    assert refreshed == [1]


def test_cache_control_never_caches_a_miss():
    """Cache and preview-bucket entries are content-addressed and served
    immutable; a miss on those paths is never cacheable (DESIGN 7.2): a
    404 cached under the very name a re-evaluation restores stranded a
    tab across reconnects and restarts on one origin. Viewer files
    revalidate every load."""
    assert server_mod.cache_control("/cache/abc.glb", 200) == "public, max-age=31536000, immutable"
    assert server_mod.cache_control("/preview/abc.glb", 200) == "public, max-age=31536000, immutable"
    assert server_mod.cache_control("/cache/abc.glb", 404) == "no-store"
    assert server_mod.cache_control("/preview/abc.glb", 404) == "no-store"
    # A served body stays cacheable when revalidated or partial: a
    # no-store on the 304 would evict the stored copy it confirms.
    for status in (206, 304):
        assert server_mod.cache_control("/cache/abc.glb", status).endswith("immutable")
    assert server_mod.cache_control("/viewer/app.js", 200) == "no-cache"
    assert server_mod.cache_control("/", 200) == "no-cache"


def test_cache_miss_on_a_served_record_rematerializes(tmp_path, monkeypatch):
    """A 404 on a record the server is serving means the cache was
    emptied under it (DESIGN 7.2): one file-state reload (trigger
    `cache_lost`) rebuilds the record under its content-addressed name
    and re-broadcasts — one reload per outage, and a name the payload
    does not reference heals nothing."""
    import pytest
    from aiohttp import web

    from moraine3d.app.server import cache_dir
    from moraine3d.cache.store import unlink_cache_file

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    server = _server(tmp_path)
    recorded: list[dict] = []
    monkeypatch.setattr(
        session_mod, "record", lambda kind, **kw: recorded.append({"kind": kind, **kw})
    )
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    name = server.asset_payload["glb_urls"]["main"].removeprefix("/cache/")
    path = cache_dir() / name
    key = name.split("-")[0]
    for p in (path, cache_dir() / f"{key}.meta.json", cache_dir() / f"{key}.npz"):
        unlink_cache_file(p)
    assert not path.exists()
    tab = _StubWS()  # a tab holding the now-dangling urls
    server.clients[tab] = ClientState()
    scrubbing = _StubWS()  # a tab mid-preview: the heal drops and says so
    server.clients[scrubbing] = ClientState()
    server.clients[scrubbing].overrides["k"] = 1.0

    async def scenario():
        with pytest.raises(web.HTTPNotFound):
            await server.cache_file(_FileReq(name))
        task = server._heal_task
        assert task is not None
        # A second miss during the reload is the same outage.
        with pytest.raises(web.HTTPNotFound):
            await server.cache_file(_FileReq(name))
        assert server._heal_task is task
        await task
        assert path.exists()
        resp = await server.cache_file(_FileReq(name))
        assert isinstance(resp, web.FileResponse)
        # An unreferenced name is not this server's record to rebuild.
        with pytest.raises(web.HTTPNotFound):
            await server.cache_file(_FileReq("0" * 24 + "-g4-other.glb"))
        assert server._heal_task is task

    asyncio.run(scenario())
    updates = [m for m in tab.messages() if m["type"] == "asset_updated"]
    assert len(updates) == 1 and updates[0]["glb_urls"]["main"] == f"/cache/{name}"
    assert [r["trigger"] for r in recorded if r["kind"] == "reload"] == ["startup", "cache_lost"]
    # A file-state reload like any other: the overlay is dropped on the
    # server and announced to the tab that held it, never left behind.
    assert {"type": "preview_dropped", "trigger": "cache_lost"} in scrubbing.messages()
    assert server.clients[scrubbing].overrides == {}
    assert all(m["type"] != "preview_dropped" for m in tab.messages())


def test_viewer_renderers_share_the_calibrated_exposure():
    """DESIGN 7.1: one lighting rig for the interactive view and the
    screenshot scene alike, rendered through one fixed linear exposure -
    every WebGLRenderer the viewer constructs is calibrated by the same
    call, so an evidence render cannot drift from what the human sees."""
    import re
    from pathlib import Path

    import moraine3d

    src = (Path(moraine3d.__file__).parent / "app" / "viewer" / "app.js").read_text(encoding="utf-8")
    constructions = [m.start() for m in re.finditer(r"new THREE\.WebGLRenderer\(", src)]
    assert len(constructions) == 2
    for at in constructions:
        window = src[at:at + 400]
        assert "calibrateExposure(" in window, src[at:at + 120]
    assert "r.toneMapping = THREE.LinearToneMapping" in src
    assert re.search(r"const RIG_EXPOSURE = 0\.\d+;", src)



def test_channels_refuse_a_request_sent_for_another_root(tmp_path, monkeypatch):
    """A channel request names the root it speaks for, and a server of
    another root refuses it (DESIGN 6) before resolving the project,
    where a project of the same name would be shot or pinned in the
    caller's place; the server's own root passes the gate."""
    import urllib.parse

    from moraine3d.app.server import ROOT_HEADER

    server = _ws_server(tmp_path, monkeypatch)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    other = {ROOT_HEADER: urllib.parse.quote(str(tmp_path / "elsewhere"))}
    resp = asyncio.run(server.screenshot(_Request({"project": "alpha"}, headers=other)))
    payload = json.loads(resp.body)
    assert resp.status == 400 and payload["error"]["code"] == "wrong_workspace"
    assert payload["error"]["detail"]["serves"] == str(server.watch_root)
    resp = asyncio.run(server.pins_api(
        _Request({}, method="GET", query={"project": "alpha"}, headers=other)
    ))
    assert resp.status == 400
    assert json.loads(resp.body)["error"]["code"] == "wrong_workspace"
    own = {ROOT_HEADER: urllib.parse.quote(str(tmp_path / "ws"))}
    resp = asyncio.run(server.pins_api(
        _Request({}, method="GET", query={"project": "alpha"}, headers=own)
    ))
    assert resp.status == 200


def test_a_shot_under_given_knob_values_leaves_the_file_state_alone(tmp_path, monkeypatch):
    """DESIGN 6: `screenshot --param` shoots the asset under the values
    given for its knobs, built as a consumer's offer and evaluated into
    the preview bucket; the report names them, a value the declaration
    refuses is the offer's error, and the next plain shot is the file
    state's again."""
    from support.recipes import write_files

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        'body = mo.transform(mo.box(id = "bx", size = [w, 1, 1]), id = "up",'
        ' translate = [0, 0.5, 0])\n'
        'mo.output("main", body)\n'
    )
    project = write_files(tmp_path / "proj", {"asset.star": asset})
    server = MoraineServer(project / "asset.star")
    req, status, plain = _shoot(server, {}, monkeypatch)
    assert status == 200 and "params_given" not in plain["meta"]
    assert req["glb_urls"]["main"].startswith("/cache/")

    req_given, status, shot = _shoot(server, {"params": {"w": 1.5}}, monkeypatch)
    assert status == 200 and shot["meta"]["params_given"] == {"w": 1.5}
    assert req_given["glb_urls"]["main"].startswith("/preview/")
    assert req_given["glb_urls"]["main"].rsplit("/", 1)[1] != req["glb_urls"]["main"].rsplit("/", 1)[1]
    assert not (project / "params.json").exists()

    # Outside the declared range the asset is shot all the same, as
    # `validate --param` judges it (DESIGN 6).
    req_out, status, shot = _shoot(server, {"params": {"w": 9}}, monkeypatch)
    assert status == 200 and shot["meta"]["params_given"] == {"w": 9}
    assert req_out["glb_urls"]["main"].startswith("/preview/")
    status, refused = _post_screenshot(server, {"params": [1]})
    assert status == 400 and refused["error"]["code"] == "bad_params"
    again, status, _ = _shoot(server, {}, monkeypatch)
    assert again["glb_urls"] == req["glb_urls"]
