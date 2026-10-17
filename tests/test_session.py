"""Session log and its summary report (DESIGN 6): what the runtime
records, how it never disturbs a command, and how sessions and rounds
are derived from the record.
"""

import asyncio
import json
import os
from pathlib import Path

from moraine3d.app import session
from moraine3d.app.cli import main
from moraine3d.app.server import ClientState, MoraineServer
from moraine3d.core import asset as asset_mod
from moraine3d.core import workspace as workspace_mod
from support import projects

_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"outputs": {"main": "bx"},
"checks": {"main": {"not_sunk": false}}
}
"""
# The bare box is centered and would fail ``not_sunk``; the waiver keeps
# the happy-path asset green, and dropping it is the red-path asset.
_SUNK = _ASSET.replace(',\n"checks": {"main": {"not_sunk": false}}', "")


def _project(tmp_path, name="proj", asset=_ASSET) -> Path:
    d = tmp_path / name
    projects.write_project(d, asset, name=name)
    return d


def _events(tmp_path) -> list[dict]:
    return session.read_events(tmp_path / "log")


def _run(argv, capsys) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


# -- writer -------------------------------------------------------------------


def test_command_event_names_the_revision_and_outcome(tmp_path, capsys):
    """DESIGN 6: every CLI invocation appends one ``command`` event with
    the timing, the stdout volume, the project and its asset content
    hash, and the command's compact outcome."""
    proj = _project(tmp_path)
    code, report = _run(["validate", str(proj)], capsys)
    assert code == 0
    evs = _events(tmp_path)
    assert [e["event"] for e in evs] == ["command"]
    ev = evs[0]
    assert ev["cmd"] == "validate" and ev["ok"] is True and "code" not in ev
    assert ev["pid"] == os.getpid() and ev["t"].endswith("Z") and ev["v"]
    assert ev["ms"] >= 0 and ev["out_bytes"] == len(json.dumps(report, separators=(",", ":"))) + 1
    assert ev["project"] == "proj"
    assert Path(ev["root"]) == proj.resolve()  # no workspace: the project dir
    assert ev["asset_hash"] == asset_mod.content_hash({proj / "proj.star"}, proj.resolve())
    assert ev["summary"] == {
        "outputs": 1, "failed_outputs": 0, "failed_checks": [],
        "asserts_failed": 0, "budgets_failed": 0, "params_failed": 0,
        "warnings": 0,
    }


def test_failed_validate_records_the_red_checks(tmp_path, capsys):
    """DESIGN 6: the verify outcome names the gating checks that went
    red, so the record can say what caught a defect."""
    proj = _project(tmp_path, asset=_SUNK)
    code, _ = _run(["validate", str(proj)], capsys)
    assert code == 1
    ev = _events(tmp_path)[0]
    assert ev["ok"] is False and "code" not in ev  # red, not an error
    assert ev["summary"]["failed_outputs"] == 1
    assert "not_sunk" in ev["summary"]["failed_checks"]


def test_failed_asset_check_joins_the_red_checks(tmp_path, capsys):
    """DESIGN 6: the summary's ``failed_checks`` names every gating check
    that went red — the asset-level grounded check included, which fails
    no individual output."""
    floating = _SUNK.replace(
        '{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}',
        '{"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},\n'
        '{"id": "up", "op": "transform", "inputs": ["bx"], '
        '"params": {"translate": [0.0, 2.0, 0.0]}}',
    ).replace('"outputs": {"main": "bx"}', '"outputs": {"main": "up"}')
    proj = _project(tmp_path, asset=floating)
    code, _ = _run(["validate", str(proj)], capsys)
    assert code == 1
    ev = _events(tmp_path)[0]
    assert ev["summary"]["failed_outputs"] == 0
    assert ev["summary"]["failed_checks"] == ["grounded"]


def test_error_path_logs_the_code_and_no_revision(tmp_path, capsys):
    """DESIGN 6: a command that fails before evaluating still leaves one
    event — ok false with the error code and no project or hash."""
    code, _ = _run(["validate", str(tmp_path / "missing")], capsys)
    assert code == 1
    ev = _events(tmp_path)[0]
    assert ev["cmd"] == "validate" and ev["ok"] is False
    assert ev["code"] == "asset_not_found"
    assert "asset_hash" not in ev and "project" not in ev


def test_sweep_event_carries_per_project_records(tmp_path, capsys):
    """DESIGN 6: a workspace sweep is one command event whose
    ``projects`` map carries each project's hash and outcome, framed by
    the workspace root."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _project(ws, "alpha")
    _project(ws, "beta")
    code, _ = _run(["stats", str(ws)], capsys)
    assert code == 0
    ev = _events(tmp_path)[0]
    assert Path(ev["root"]) == ws.resolve()
    assert set(ev["projects"]) == {"alpha", "beta"}
    for name in ("alpha", "beta"):
        rec = ev["projects"][name]
        assert rec["ok"] is True and len(rec["asset_hash"]) == 24
        assert rec["summary"]["nodes"] == 1
    assert "project" not in ev and "sweep" not in ev


def test_partial_evaluation_is_marked(tmp_path, capsys):
    """DESIGN 6: a subgraph evaluation (``--output``) need not reach the
    whole revision, so its hash is flagged ``partial`` for the reader."""
    proj = _project(tmp_path)
    code, _ = _run(["eval", str(proj), "--output", "main"], capsys)
    assert code == 0
    assert _events(tmp_path)[0]["partial"] is True
    code, _ = _run(["eval", str(proj)], capsys)
    assert "partial" not in _events(tmp_path)[1]
    # eval's summary carries the same counts as stats (DESIGN 6).
    summary = _events(tmp_path)[1]["summary"]
    assert summary["nodes"] == 1 and summary["outputs"] == 1
    assert summary["tris"] == 12 and summary["instances"] == 0


def test_logging_never_changes_stdout_or_exit(tmp_path, capsys, monkeypatch):
    """DESIGN 6: an unwritable log directory costs one stderr line and
    nothing else — stdout and the exit code are the command's contract."""
    proj = _project(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(session.LOG_DIR_ENV, str(blocker / "log"))
    session.reset()
    code = main(["stats", str(proj)])
    out = capsys.readouterr()
    assert code == 0
    report = json.loads(out.out)
    assert report["asset"]["nodes"] == 1
    assert "session log skipped" in out.err and out.err.count("\n") == 1
    # Later events in the same process (a serve's many reloads): silent.
    session.record("probe")
    session.record("probe")
    assert capsys.readouterr().err == ""


def test_empty_override_turns_logging_off(tmp_path, capsys, monkeypatch):
    """DESIGN 6: an empty ``MORAINE_LOG_DIR`` is the off switch."""
    proj = _project(tmp_path)
    monkeypatch.setenv(session.LOG_DIR_ENV, "")
    assert session.log_dir() is None
    assert main(["stats", str(proj)]) == 0
    capsys.readouterr()
    assert not list(tmp_path.rglob("*.jsonl"))


def test_log_holds_no_asset_content(tmp_path, capsys):
    """DESIGN 6: the log records hashes, names, counts, and timings —
    never the asset text."""
    proj = _project(tmp_path, asset=_ASSET.replace('"bx"', '"secretnode"'))
    assert main(["validate", str(proj)]) == 0
    capsys.readouterr()
    raw = "".join(f.read_text(encoding="utf-8") for f in (tmp_path / "log").glob("*.jsonl"))
    assert "secretnode" not in raw


def _day_file(d, day: str, text=None) -> Path:
    f = d / f"{day}.jsonl"
    f.write_text(
        text if text is not None
        else f'{{"t":"{day}T00:00:00Z","v":"0","pid":1,"event":"command"}}\n',
        encoding="utf-8",
    )
    return f


def test_retirement_drops_only_expired_moraine_day_files(tmp_path):
    """DESIGN 6: the log is a rolling window — the first write of a new
    day retires day files past the retention window, and only them:
    the name must be a whole day-file name, the date past the cutoff,
    and the first line a Moraine event (a foreign file in an overridden
    log dir is never deletable by name accident)."""
    d = tmp_path / "log"
    d.mkdir()
    from datetime import date, timedelta
    recent_day = (date.today() - timedelta(days=3)).isoformat()
    expired = _day_file(d, "2020-01-01")
    foreign = _day_file(d, "2020-01-02", text="someone else's file\n")
    named = _day_file(d, "2020-01-03")
    named = named.rename(d / "notes.jsonl")
    recent = _day_file(d, recent_day)
    session.record("probe")  # first write of today's file runs retirement
    assert not expired.exists()
    assert foreign.exists() and named.exists() and recent.exists()
    assert len(session.read_events(d)) >= 1  # today's event landed first


def test_retirement_keep_zero_keeps_everything(tmp_path, monkeypatch):
    """DESIGN 6: ``MORAINE_LOG_KEEP_DAYS=0`` turns retirement off."""
    d = tmp_path / "log"
    d.mkdir()
    expired = _day_file(d, "2020-01-01")
    monkeypatch.setenv(session.KEEP_DAYS_ENV, "0")
    session.record("probe")
    assert expired.exists()


# -- server events ------------------------------------------------------------


def _ws_server(tmp_path) -> tuple[MoraineServer, Path]:
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _project(ws, "alpha")
    _project(ws, "beta", _ASSET.replace("[1.0, 1.0, 1.0]", "[2.0, 1.0, 1.0]"))
    disc = workspace_mod.discover(ws)
    initial = disc.resolve_asset(disc.projects[0])
    return MoraineServer(initial, workspace=disc, project=disc.projects[0]), ws


def test_reload_event_counts_the_edit(tmp_path, monkeypatch):
    """DESIGN 6: a watch-triggered reload records the revision it reached
    and the edit size (lines added + removed) since the previous file
    state; the startup and set_asset reloads seat the snapshot without
    an edit size, since nothing was edited."""
    server, ws = _ws_server(tmp_path)
    monkeypatch.setattr(server, "_broadcast", _noop)
    monkeypatch.setattr(server, "_broadcast_state", _noop)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    asset = ws / "alpha" / "alpha.star"
    text = asset.read_text(encoding="utf-8")
    assert "size = [1.0, 1.0, 1.0])" in text
    asset.write_text(
        text.replace('size = [1.0, 1.0, 1.0])',
                     'size = [1.5, 1.0, 1.0])\n'
                     'n_bx2 = mo.box(id = "bx2", size = [0.5, 0.5, 0.5])'),
        encoding="utf-8", newline="\n",
    )
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    asyncio.run(server._set_asset({"project": "beta"}))
    evs = [e for e in _events(tmp_path) if e["event"] == "reload"]
    assert [e["trigger"] for e in evs] == ["startup", "watch", "set_asset"]
    assert all(e["ok"] and Path(e["root"]) == ws.resolve() for e in evs)
    assert "changed_lines" not in evs[0]
    assert evs[1]["project"] == "alpha" and evs[1]["changed_lines"] == 3  # 1 removed + 2 added
    assert evs[1]["asset_hash"] != evs[0]["asset_hash"]
    assert evs[1]["asset_hash"] == asset_mod.content_hash({asset}, ws.resolve())
    assert evs[2]["project"] == "beta" and "changed_lines" not in evs[2]


def test_reload_event_counts_the_graph_change(tmp_path, monkeypatch):
    """DESIGN 6: `serve` runs the same diff on every watch reload and
    logs its counts, nodes and asserts added, removed, and changed. A
    startup or a re-root has no edit to count, and the reload after a
    broken revision is counted against the last good state."""
    server, ws = _ws_server(tmp_path)
    monkeypatch.setattr(server, "_broadcast", _noop)
    monkeypatch.setattr(server, "_broadcast_state", _noop)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    asset = ws / "alpha" / "alpha.star"
    text = asset.read_text(encoding="utf-8")

    def save(new: str) -> None:
        asset.write_text(new, encoding="utf-8", newline="\n")
        asyncio.run(server._reload_and_broadcast(trigger="watch"))

    grown = text.replace(
        'size = [1.0, 1.0, 1.0])',
        'size = [1.5, 1.0, 1.0])\n'
        'n_bx2 = mo.box(id = "bx2", size = [0.5, 0.5, 0.5])',
    )
    save(grown)
    save("{not a recipe")
    save(grown.replace("[0.5, 0.5, 0.5]", "[0.6, 0.5, 0.5]"))
    save(text)
    asyncio.run(server._set_asset({"project": "beta"}))
    evs = [e for e in _events(tmp_path) if e["event"] == "reload"]
    assert [e["trigger"] for e in evs] == ["startup"] + ["watch"] * 4 + ["set_asset"]
    assert [e.get("diff") for e in evs] == [
        None,
        {"nodes": [1, 0, 1], "asserts": [0, 0, 0]},
        None,
        {"nodes": [0, 0, 1], "asserts": [0, 0, 0]},
        {"nodes": [0, 1, 1], "asserts": [0, 0, 0]},
        None,
    ]


def test_failed_reload_is_recorded_and_a_scrub_is_not(tmp_path, monkeypatch):
    """DESIGN 6: a broken revision still gets its reload event (ok false,
    the error code, the hash of the broken files); a set_param scrub,
    failing or not, is the human's own gesture and leaves no event."""
    server, ws = _ws_server(tmp_path)
    monkeypatch.setattr(server, "_broadcast", _noop)
    monkeypatch.setattr(server, "_broadcast_state", _noop)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    asset = ws / "alpha" / "alpha.star"
    asset.write_text("{not a recipe", encoding="utf-8")
    asyncio.run(server._reload_and_broadcast(trigger="watch"))
    monkeypatch.setattr(server, "_send", _noop)
    client = object()
    server.clients[client] = ClientState()
    asyncio.run(server._set_param(client, {"param_id": "nope", "value": 1}))
    evs = _events(tmp_path)
    bad = [e for e in evs if e["event"] == "reload"][-1]
    assert bad["ok"] is False and bad["code"] and len(bad["asset_hash"]) == 24
    assert not [e for e in evs if e["event"] == "preview"]


async def _noop(*_a, **_k):
    return None


def test_pins_leave_no_event(tmp_path, monkeypatch):
    """DESIGN 6: a pin is the human's own gesture, which no surface reads
    back, so a placement, a failed resolution, and a clear leave no
    event in the log."""
    server, ws = _ws_server(tmp_path)
    monkeypatch.setattr(server, "_broadcast", _noop)
    monkeypatch.setattr(server, "_broadcast_state", _noop)
    monkeypatch.setattr(server, "_send", _noop)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    client = object()
    server.clients[client] = ClientState()
    asyncio.run(server._select(client, {
        "type": "select", "action": "pin", "output": "main", "face": 0,
        "position": [0.25, 0.5, 0.1259],
    }))
    asyncio.run(server._select(client, {
        "type": "select", "action": "pin", "output": "nope", "face": 0,
    }))
    asyncio.run(server._select(client, {"type": "select", "action": "clear"}))
    evs = _events(tmp_path)
    assert not [e for e in evs if e["event"] == "pin"]
    assert "0.1259" not in json.dumps(evs)


def test_reader_passes_over_the_events_earlier_builds_wrote(tmp_path):
    """DESIGN 6: a day file written by an earlier build may still carry
    ``preview`` and ``pin`` events; the report neither counts them nor
    reads them as errors, and never carries ``previews`` or ``pins``."""
    events = [
        _ev("00:00", "serve_start"),
        _ev("00:01", "pin", project="p", action="add", source="viewer", ok=True, ms=5),
        _ev("00:02", "preview", project="p", ok=False, code="param_invalid"),
        _ev("00:03", "pin", project="p", action="add", source="viewer",
            ok=False, code="bad_face", ms=5),
    ]
    rep = session.summarize(events, Path("/ws"))
    s = rep["sessions"][0]
    p = s["projects"]["p"]
    assert p["eval_errors"] == 0 and p["rounds"] == 0
    assert not {"pins", "previews"} & (set(p) | set(s) | set(rep["totals"]))


def test_an_accept_that_carries_its_check_verifies_the_round(tmp_path):
    """DESIGN 6: the ``validate`` an accept runs is a verification of
    the accepted revision, so a round the human closed by accepting
    reads ``ended_by: accept``; an accept that could move no check
    (no summary) opens none."""
    events = [
        _ev("00:00", "reload", project="p", asset_hash="a" * 24, ok=True, trigger="watch"),
        _ev("00:01", "accept", project="p", ok=True, ms=5, param_ids=["rough"],
            asset_hash="b" * 24, summary={"validate_ok": False, "failed": ["rise"]}),
        _ev("00:02", "reload", project="p", asset_hash="c" * 24, ok=True, trigger="watch"),
        _ev("00:03", "accept", project="p", ok=True, ms=5, param_ids=["seed"],
            asset_hash="c" * 24),
    ]
    p = session.summarize(events, Path("/ws"))["sessions"][0]["projects"]["p"]
    assert p["accepts"] == 2
    [closed] = p["rounds_detail"]
    assert closed["ended_by"] == "accept"
    assert closed["validate_ok"] is False and closed["failed_checks"] == ["rise"]


def test_cli_and_server_agree_on_the_revision(tmp_path, capsys, monkeypatch):
    """DESIGN 6: the CLI frames a workspace member's hash from the
    enclosing workspace root, exactly as the server does, so one
    revision has one name across writers."""
    server, ws = _ws_server(tmp_path)
    monkeypatch.setattr(server, "_broadcast", _noop)
    monkeypatch.setattr(server, "_broadcast_state", _noop)
    asyncio.run(server._reload_and_broadcast(trigger="startup"))
    assert main(["stats", str(ws / "alpha")]) == 0
    capsys.readouterr()
    evs = _events(tmp_path)
    reload, cmd = evs[0], evs[-1]
    assert reload["event"] == "reload" and cmd["event"] == "command"
    assert cmd["asset_hash"] == reload["asset_hash"]
    assert cmd["root"] == reload["root"]


# -- reader -------------------------------------------------------------------


def _ev(t, event, root="/ws", **f):
    # None fields are omitted, as the writer omits them (session.record).
    return {"t": f"2026-01-01T10:{t}Z", "v": "0", "pid": 1, "event": event,
            **({"root": root} if root else {}),
            **{k: v for k, v in f.items() if v is not None}}


def _cmd(t, cmd, project="p", h="h1", ok=True, **f):
    return _ev(t, "command", cmd=cmd, project=project, asset_hash=h, ok=ok, ms=10, out_bytes=100, **f)


def test_rounds_follow_verified_revisions(tmp_path):
    """DESIGN 6: a round is one revision carrying at least one verify
    command; revisions never verified fold into the next one (draft
    corrections are not rounds), and the round ends on its last
    verifier."""
    root = Path("/ws")
    events = [
        _cmd("00:00", "validate", h="a"),
        _cmd("00:10", "stats", h="a"),
        _ev("00:20", "reload", project="p", asset_hash="b", ok=True, changed_lines=3),  # draft, never verified
        _ev("00:30", "reload", project="p", asset_hash="c", ok=True, changed_lines=5),
        _cmd("00:40", "validate", h="c", ok=False, summary={"failed_checks": ["grounded"]}),
        _ev("00:50", "screenshot", project="p", asset_hash="c", ok=True, bytes=4000),
        _ev("01:00", "reload", project="p", asset_hash="d", ok=True, changed_lines=1),
        _cmd("01:10", "stats", h="d", summary={"nodes": 9}),
        _ev("01:20", "reload", project="p", asset_hash="e", ok=True, changed_lines=2),  # unverified tail
    ]
    rep = session.summarize(events, root)
    assert rep["root"] == "ws"
    s = rep["sessions"][0]
    p = s["projects"]["p"]
    assert p["rounds"] == 3 and p["nodes"] == 9 and p["shots"] == 1 and p["shot_bytes"] == 4000
    r1, r2, r3 = p["rounds_detail"]
    assert r1["hash"] == "a" and r1["ended_by"] == "stats" and r1["validate_ok"] is True
    assert r2["hash"] == "c" and r2["changed_lines"] == 8  # the draft folded forward
    assert r2["validate_ok"] is False and r2["failed_checks"] == ["grounded"]
    assert r2["ended_by"] == "screenshot" and r2["shots"] == 1
    assert r3["hash"] == "d" and r3["ended_by"] == "stats"
    assert p["changed_lines"] == 11  # every edit, the unverified tail included
    assert p["reloads"] == 4
    assert s["commands"] == {"validate": 2, "stats": 2}
    assert rep["totals"]["rounds"] == 3 and rep["totals"]["shots"] == 1


def test_sessions_split_on_idle_gaps_and_select_by_root(tmp_path):
    """DESIGN 6: events more than the gap apart start a new session;
    only events under the requested root count, root-less commands
    join the session whose span (plus the gap) holds them, and partial
    hashes never open a round."""
    root = Path("/ws")
    events = [
        _cmd("00:00", "validate", h="a"),
        _cmd("00:05", "docs", project=None, h=None, root=None),          # root-less, inside
        _cmd("00:10", "eval", h="zzz", partial=True),                      # partial: no new round
        _cmd("00:20", "stats", h="a"),
        _cmd("00:25", "stats", h="other", root="/elsewhere"),             # other root: excluded
        _cmd("59:00", "validate", h="b"),                                  # > 30 min later
        _cmd("59:30", "cache.stats", project=None, h=None, root=None),
    ]
    rep = session.summarize(events, root, last=None)
    assert len(rep["sessions"]) == 2
    first, second = rep["sessions"]
    assert first["commands"] == {"validate": 1, "docs": 1, "eval": 1, "stats": 1}
    assert first["projects"]["p"]["rounds"] == 1
    assert second["commands"] == {"validate": 1, "cache.stats": 1}
    assert session.summarize(events, root)["sessions"] == [second]
    assert session.summarize(events, root, last=2)["sessions"] == [first, second]
    wider = session.summarize(events, Path("/"), last=None)
    assert "/elsewhere" not in json.dumps(wider) and wider["root"] == "/"
    assert wider["totals"]["commands"]["stats"] == 2


def test_hash_less_events_before_any_revision_open_no_round():
    """DESIGN 6: a partial evaluation opens no round and hash-less events
    attach to the current round — so hash-less events that precede the
    first revision fold into its round, and alone they leave the project
    with no round at all rather than one named by no hash."""
    root = Path("/ws")
    events = [
        _cmd("00:00", "eval", h="zzz", partial=True),
        _cmd("00:05", "validate", h=None, ok=False, code="unknown_op"),
        _cmd("00:10", "validate", h="a"),
    ]
    p = session.summarize(events, root, last=None)["sessions"][0]["projects"]["p"]
    assert p["rounds"] == 1
    (r,) = p["rounds_detail"]
    assert r["hash"] == "a" and r["commands"] == {"eval": 1, "validate": 2}
    assert p["eval_errors"] == 1

    alone = session.summarize(events[:2], root, last=None)["sessions"][0]["projects"]["p"]
    assert alone["rounds"] == 0 and alone["rounds_detail"] == []
    assert alone["eval_errors"] == 1


def test_failed_shots_count_as_errors(tmp_path):
    """DESIGN 6: per-project error accounting covers failed commands,
    reloads, and renders — a session of failing renders must not read
    as error-free."""
    events = [
        _cmd("00:00", "validate", h="a"),
        _ev("00:30", "screenshot", project="p", ok=False, code="no_viewer"),
    ]
    proj = session.summarize(events, Path("/ws"))["sessions"][0]["projects"]["p"]
    assert proj["eval_errors"] == 1


def test_sweep_wall_time_is_not_duplicated_per_project(tmp_path):
    """DESIGN 6: a sweep's wall time and output volume belong to the
    command once — exploding them into every project's round would
    overstate per-round time N-fold."""
    events = [_ev("00:00", "command", cmd="validate", ok=True, ms=4000, out_bytes=999,
                  projects={"a": {"asset_hash": "ha", "ok": True},
                            "b": {"asset_hash": "hb", "ok": True}})]
    rep = session.summarize(events, Path("/ws"))
    s0 = rep["sessions"][0]
    assert s0["commands"] == {"validate": 1} and s0["out_bytes"] == 999
    for name in ("a", "b"):
        assert s0["projects"][name]["rounds_detail"][0]["ms"] == 0


def test_reader_never_double_counts_or_adopts_screenshot_commands(tmp_path):
    """DESIGN 6: the server's ``screenshot`` event owns the render — a
    project-carrying screenshot *command* event (earlier builds wrote
    them) must not double the round's count, and a root-less command
    naming a project (another server may have answered the port) is
    unattributable and dropped rather than adopted into the report."""
    events = [
        _cmd("00:00", "validate", h="a"),
        _ev("00:10", "screenshot", project="p", asset_hash="a", ok=True, bytes=100),
        # old-log shape: CLI screenshot command event carrying the project
        _cmd("00:11", "screenshot", h="a"),
        # foreign server's shot, root-less but project-carrying
        _ev("00:12", "command", root=None, cmd="screenshot", ok=True,
            project="foreign", asset_hash="zz"),
    ]
    rep = session.summarize(events, Path("/ws"))
    s0 = rep["sessions"][0]
    assert set(s0["projects"]) == {"p"}
    r0 = s0["projects"]["p"]["rounds_detail"][0]
    assert r0["commands"]["screenshot"] == 1
    assert s0["projects"]["p"]["shots"] == 1
    assert "foreign" not in json.dumps(rep)


def test_sessions_narrows_to_a_workspace_member(tmp_path, capsys):
    """DESIGN 6: PATH may be a workspace member's project directory —
    its events are framed by the workspace root, so the report narrows
    the workspace's sessions to that project instead of selecting by
    path (which would find nothing)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _project(ws, "alpha")
    _project(ws, "beta")
    for name in ("alpha", "beta"):
        assert main(["validate", str(ws / name)]) == 0
        capsys.readouterr()
    code, rep = _run(["sessions", str(ws / "alpha")], capsys)
    assert code == 0
    assert rep["root"] == "ws" and rep["project"] == "alpha"
    assert set(rep["sessions"][0]["projects"]) == {"alpha"}
    code, rep = _run(["sessions", str(ws)], capsys)
    assert code == 0
    assert set(rep["sessions"][0]["projects"]) == {"alpha", "beta"}


def test_cli_evaluation_failure_is_attributed_to_the_project(tmp_path, capsys):
    """DESIGN 6: per-project error accounting must not depend on the
    entry point — a CLI command failing mid-evaluation still leaves an
    event naming the project it ran against."""
    proj = _project(tmp_path)
    (proj / "proj.star").write_text("{not a recipe", encoding="utf-8")
    code, _ = _run(["validate", str(proj)], capsys)
    assert code == 1
    ev = _events(tmp_path)[0]
    assert ev["ok"] is False and ev["code"]
    assert ev["project"] == "proj" and "asset_hash" not in ev
    rep = session.summarize(_events(tmp_path), proj.resolve())
    assert rep["sessions"][0]["projects"]["proj"]["eval_errors"] == 1


def test_last_must_be_positive(tmp_path, capsys):
    """DESIGN 8.2: errors are actionable — ``--last 0`` is a validation
    error, never a no_sessions report claiming the log is empty."""
    code, out = _run(["sessions", str(tmp_path), "--last", "0"], capsys)
    assert code == 1 and out["error"]["code"] == "bad_last"


def test_content_hash_is_strict_except_for_missing_refs(tmp_path):
    """DESIGN 6: an unreadable file raises — a hash silently missing a
    file would name a revision that never existed on disk; the one
    sanctioned skip is a file that does not exist at all (a dangling
    asset_ref's revision)."""
    import pytest

    a = tmp_path / "a.mora"
    a.write_text("x", encoding="utf-8")
    gone = tmp_path / "gone.mora"
    with pytest.raises(FileNotFoundError):
        asset_mod.content_hash({a, gone}, tmp_path)
    assert asset_mod.content_hash({a, gone}, tmp_path, missing_ok=True) ==         asset_mod.content_hash({a}, tmp_path)
    unreadable = tmp_path / "dir.mora"
    unreadable.mkdir()
    with pytest.raises(OSError):
        asset_mod.content_hash({a, unreadable}, tmp_path, missing_ok=True)


def test_read_recent_stops_before_old_day_files(tmp_path, monkeypatch):
    """DESIGN 6: the default report reads day files newest-first and
    stops once one session more than requested exists — a month of
    retained log must not be parsed to report the last session."""
    d = tmp_path / "log"
    d.mkdir()
    line = ('{{"t":"{day}T{hm}:00Z","v":"0","pid":1,"event":"command",'
            '"cmd":"stats","ok":true,"root":"/ws","project":"p",'
            '"asset_hash":"h"}}\n')
    (d / "2026-01-01.jsonl").write_text(line.format(day="2026-01-01", hm="10:00"), encoding="utf-8")
    (d / "2026-01-02.jsonl").write_text(line.format(day="2026-01-02", hm="10:00"), encoding="utf-8")
    (d / "2026-01-03.jsonl").write_text(line.format(day="2026-01-03", hm="10:00"), encoding="utf-8")
    parsed: list[str] = []
    real = session._parse_file

    def counting(f):
        parsed.append(f.name)
        return real(f)

    monkeypatch.setattr(session, "_parse_file", counting)
    events = session.read_recent(d, Path("/ws"), gap_minutes=30, last=1)
    assert parsed == ["2026-01-03.jsonl", "2026-01-02.jsonl"]  # oldest never read
    rep = session.summarize(events, Path("/ws"), last=1)
    assert rep["sessions"][0]["start"].startswith("2026-01-03")


def test_read_recent_reads_the_window_for_a_root_with_one_session(tmp_path):
    """DESIGN 6: reading stops once one session more than requested
    exists. A root whose only session sits in the newest day file never
    meets that, so every day file is read, and the report is that one
    session over events in time order."""
    d = tmp_path / "log"
    d.mkdir()
    line = ('{{"t":"{t}:00Z","v":"0","pid":1,"event":"command",'
            '"cmd":"stats","ok":true,"root":"{root}","project":"p",'
            '"asset_hash":"h"}}\n')
    (d / "2026-01-01.jsonl").write_text(
        line.format(t="2026-01-01T10:00", root="/other"), encoding="utf-8")
    (d / "2026-01-02.jsonl").write_text(
        line.format(t="2026-01-02T10:00", root="/other"), encoding="utf-8")
    (d / "2026-01-03.jsonl").write_text(
        line.format(t="2026-01-03T09:00", root="/other")
        + line.format(t="2026-01-03T10:00", root="/ws")
        + line.format(t="2026-01-03T10:05", root="/ws"), encoding="utf-8")
    events = session.read_recent(d, Path("/ws"), gap_minutes=30, last=1)
    assert [e["t"][:16] for e in events] == [
        "2026-01-01T10:00", "2026-01-02T10:00", "2026-01-03T09:00",
        "2026-01-03T10:00", "2026-01-03T10:05",
    ]
    rep = session.summarize(events, Path("/ws"), last=1)
    assert len(rep["sessions"]) == 1
    assert rep["sessions"][0]["start"].startswith("2026-01-03T10:00")


def test_report_carries_names_never_paths(tmp_path, capsys):
    """DESIGN 6: the report is meant to be quoted — it names the root and
    projects, never an absolute path; the CLI resolves PATH as a plain
    directory so a parent of standalone projects works."""
    parent = tmp_path / "bench"
    _project(parent, "one")
    _project(parent, "two")
    for name in ("one", "two"):
        assert main(["validate", str(parent / name)]) == 0
        capsys.readouterr()
    code, rep = _run(["sessions", str(parent)], capsys)
    assert code == 0
    assert rep["root"] == "bench" and set(rep["sessions"][0]["projects"]) == {"one", "two"}
    assert str(tmp_path) not in json.dumps(rep) and str(tmp_path).replace("\\", "/") not in json.dumps(rep)
    # The report command itself is logged like any other, under no root.
    assert _events(tmp_path)[-1]["cmd"] == "sessions"


def test_sessions_errors_are_actionable(tmp_path, capsys):
    """DESIGN 6 / 8.2: nothing under the root is an error naming the
    root and the log location, never an empty green report."""
    code, out = _run(["sessions", str(tmp_path)], capsys)
    assert code == 1 and out["error"]["code"] == "no_sessions"
    assert "log" in out["error"]["message"]
    code, out = _run(["sessions", str(tmp_path / "nope")], capsys)
    assert code == 1 and out["error"]["code"] == "no_such_directory"


def test_reader_skips_malformed_lines(tmp_path):
    """DESIGN 6: the reader is forward-compatible — malformed or foreign
    lines are skipped, events are returned in time order."""
    d = tmp_path / "log"
    d.mkdir()
    (d / "2026-01-02.jsonl").write_text(
        '{"t":"2026-01-02T00:00:00Z","event":"command"}\n'
        'garbage\n'
        '[1,2]\n'
        '{"no_t":1,"event":"x"}\n',
        encoding="utf-8",
    )
    (d / "2026-01-01.jsonl").write_text('{"t":"2026-01-01T00:00:00Z","event":"reload"}\n', encoding="utf-8")
    evs = session.read_events(d)
    assert [e["event"] for e in evs] == ["reload", "command"]


def test_accept_events_count_in_the_report(tmp_path):
    """DESIGN 6: the per-project summary counts landed viewer accepts —
    the picks that reached the file without an agent round trip (7.2) —
    and a failed accept joins eval_errors like a failed reload."""
    events = [
        _ev("00:00", "serve_start"),
        _ev("00:01", "accept", project="p", ok=True, ms=5, param_ids=["rough"]),
        _ev("00:02", "accept", project="p", ok=True, ms=5, param_ids=["seed", "width"]),
        _ev("00:03", "accept", project="p", ok=False, code="accept_empty",
            ms=1, param_ids=[]),
    ]
    rep = session.summarize(events, Path("/ws"))
    s = rep["sessions"][0]
    p = s["projects"]["p"]
    assert p["accepts"] == 2
    assert p["eval_errors"] == 1
    assert s["accepts"] == 2 and rep["totals"]["accepts"] == 2


def test_reader_memo_sees_appended_events(tmp_path):
    """The reader memoizes parsed day files by size and mtime, so an
    append is always re-read: the timeline refresh must never serve a
    stale parse of the current day file (DESIGN 7.2 ``timeline``)."""
    d = tmp_path / "log"
    d.mkdir()
    f = d / "2026-01-01.jsonl"
    row = '{"t":"2026-01-01T00:00:%02dZ","event":"command","cmd":"stats"}\n'
    f.write_text(row % 0, encoding="utf-8")
    assert len(session.read_events(d)) == 1
    with f.open("a", encoding="utf-8") as fh:
        fh.write(row % 1)
    assert len(session.read_events(d)) == 2


def test_retirement_rejects_near_miss_foreign_rows(tmp_path):
    """The provenance gate is the full row shape the writer stamps —
    t/v/pid/event with a parseable timestamp. A foreign day-rotated
    JSONL whose first line merely carries generic "event" and "v" keys
    is never deletable, however expired (the DESIGN 4.5 deletion-safety
    discipline, scaled down to the log dir)."""
    d = tmp_path / "log"
    d.mkdir()
    near_miss = _day_file(d, "2020-01-04", text='{"event": "start", "v": 2}\n')
    bad_time = _day_file(
        d, "2020-01-05",
        text='{"t": "yesterday", "v": "0", "pid": 1, "event": "x"}\n',
    )
    session.record("probe")  # first write of today's file runs retirement
    assert near_miss.exists() and bad_time.exists()


def test_project_report_reads_past_other_projects_sessions(tmp_path):
    """A project-narrowed report reaches that project's newest session
    even when other projects under the same root fill the newest day
    files: the reader's early stop counts sessions under the same
    narrowing the summarizer applies (DESIGN 6)."""
    d = tmp_path / "log"
    d.mkdir()
    root = tmp_path / "ws"

    def row(t, project):
        return json.dumps({
            "t": t, "v": "0", "pid": 1, "event": "command",
            "cmd": "validate", "ok": True,
            "root": str(root), "project": project,
        })

    (d / "2026-08-27.jsonl").write_text(
        row("2026-08-27T10:00:00.000Z", "beta") + "\n", encoding="utf-8"
    )
    (d / "2026-08-28.jsonl").write_text(
        row("2026-08-28T09:00:00.000Z", "alpha") + "\n"
        + row("2026-08-28T12:00:00.000Z", "alpha") + "\n",
        encoding="utf-8",
    )
    report = session.latest_report(root, project="beta")
    assert report is not None
    assert "beta" in report["sessions"][-1]["projects"]


def test_parse_memo_extends_on_append_and_survives_truncation(tmp_path):
    """The day-file memo re-parses only the appended tail of the growing
    current file and falls back to a full parse when the file shrinks —
    either way the parse equals a cold one (DESIGN 6: the timeline
    refresh runs per recorded event, so the current file must not cost
    O(day) per append)."""
    d = tmp_path / "log"
    d.mkdir()
    f = d / "2026-08-28.jsonl"

    def row(n):
        return f'{{"t":"2026-08-28T00:00:0{n}Z","v":"0","pid":1,"event":"command"}}\n'

    f.write_text(row(1), encoding="utf-8")
    assert len(session._parse_file(f)) == 1
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(row(2) + '{"t":"2026-08-28T00:00:03Z","v":"0"')  # torn tail
    assert len(session._parse_file(f)) == 2  # tail parsed, torn line skipped
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(',"pid":1,"event":"command"}\n')  # the torn line completes
    assert len(session._parse_file(f)) == 3
    f.write_text(row(9), encoding="utf-8")  # shrunk: full re-parse
    assert [e["t"] for e in session._parse_file(f)] == ["2026-08-28T00:00:09Z"]


def test_day_file_lines_are_lf_terminated(tmp_path):
    """DESIGN 6: the session log is one JSON object per line, appended
    to a machine-level day file — the line terminator is LF on every
    platform, so a day file written on one machine reads the same on
    another and no carriage return ever joins a row."""
    session.record("command", cmd="docs", ok=True)
    (f,) = (tmp_path / "log").glob("*.jsonl")
    data = f.read_bytes()
    assert data.endswith(b"\n")
    assert b"\r" not in data
    assert json.loads(data)["cmd"] == "docs"


# -- defaults -----------------------------------------------------------------


def test_sessions_gap_defaults_to_thirty_minutes(tmp_path, capsys):
    """DESIGN 6: sessions are event runs separated by less than
    ``--gap`` minutes, default 30 — the CLI report names the gap it
    used, and without the flag two events 30 minutes apart share a
    session while a 31-minute silence starts a new one."""
    proj = _project(tmp_path)
    assert main(["validate", str(proj)]) == 0
    capsys.readouterr()
    code, rep = _run(["sessions", str(proj)], capsys)
    assert code == 0 and rep["gap_minutes"] == 30
    root = Path("/ws")
    within = [_cmd("00:00", "validate", h="a"), _cmd("30:00", "stats", h="a")]
    assert len(session.summarize(within, root, last=None)["sessions"]) == 1
    beyond = [_cmd("00:00", "validate", h="a"), _cmd("31:00", "stats", h="a")]
    assert len(session.summarize(beyond, root, last=None)["sessions"]) == 2


def test_log_keep_days_defaults_to_thirty(tmp_path, monkeypatch):
    """DESIGN 6: the first write of a new day retires day files older
    than ``MORAINE_LOG_KEEP_DAYS``, default 30 — with the variable
    unset, a day file 31 days old is retired and one 29 days old
    survives."""
    from datetime import date, timedelta

    monkeypatch.delenv(session.KEEP_DAYS_ENV, raising=False)
    assert session._keep_days() == 30
    d = tmp_path / "log"
    d.mkdir()
    today = date.fromisoformat(session._now_iso()[:10])  # the writer's UTC day
    expired = _day_file(d, (today - timedelta(days=31)).isoformat())
    kept = _day_file(d, (today - timedelta(days=29)).isoformat())
    session.record("probe")  # first write of today's file runs retirement
    assert not expired.exists()
    assert kept.exists()
