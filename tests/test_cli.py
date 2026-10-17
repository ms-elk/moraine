"""CLI-level summaries: the asset rollup emitted by `moraine stats`."""

from moraine3d.app.cli import _reachable as _real_reachable
from moraine3d.engine.stats import asset_rollup


def _server_answers(monkeypatch):
    """Pair with a urlopen mock: let the connect pre-check through."""
    monkeypatch.setattr("moraine3d.app.cli._reachable", lambda port: True)


def test_asset_rollup_aggregates_meshes_and_instances():
    stats = {
        "plain": {
            "verts": 8,
            "tris": 12,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        },
        "row": {
            "verts": 8,
            "tris": 12,
            "instances": 3,
            "bounds": {"min": [-2.0, 0.0, 0.0], "max": [4.0, 2.0, 1.0]},
        },
    }
    roll = asset_rollup(stats, node_count=5)
    assert roll["outputs"] == 2
    # DESIGN 4.1 budget vocabulary: an instanced output is one part.
    assert roll["parts"] == 2
    assert roll["nodes"] == 5
    # Instanced transform count only — a plain mesh is not "an instance".
    assert roll["instances"] == 3
    # Rendered totals: prototype counts multiplied by instance count.
    assert roll["verts"] == 8 + 8 * 3
    assert roll["tris"] == 12 + 12 * 3
    assert roll["bounds"]["min"] == [-2.0, 0.0, 0.0]
    assert roll["bounds"]["max"] == [4.0, 2.0, 1.0]
    assert roll["bounds"]["size"] == [6.0, 2.0, 1.0]


def test_asset_rollup_excludes_images_from_geometry():
    stats = {
        "tex": {"kind": "image", "size": [64, 64], "channels": 1, "range": [0.0, 1.0]},
        "m": {
            "verts": 8,
            "tris": 12,
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        },
    }
    roll = asset_rollup(stats, node_count=2)
    assert roll["outputs"] == 2  # images still count as outputs
    assert roll["parts"] == 1  # ...and as no part (DESIGN 4.1)
    assert roll["verts"] == 8
    assert roll["bounds"]["size"] == [1.0, 1.0, 1.0]


def test_asset_rollup_reports_material_identities_by_name():
    """DESIGN 6/9.1: stats reports the asset's material identities by name
    from cached stats alone — an unmaterialed part counts as the neutral
    default; images carry no material."""
    stats = {
        "tex": {"kind": "image", "size": [4, 4], "channels": 1, "range": [0.0, 1.0]},
        "a": {"verts": 8, "tris": 12, "material": "steel",
              "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
        "b": {"verts": 8, "tris": 12, "material": "steel",
              "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
        "c": {"verts": 8, "tris": 12,
              "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}},
    }
    roll = asset_rollup(stats, node_count=4)
    assert roll["materials"] == {
        "count": 2,
        "identities": [
            {"name": "moraine_default", "outputs": ["c"]},
            {"name": "steel", "outputs": ["a", "b"]},
        ],
    }


def test_asset_rollup_all_images_has_no_bounds():
    stats = {"tex": {"kind": "image", "size": [64, 64], "channels": 1, "range": [0.0, 1.0]}}
    roll = asset_rollup(stats, node_count=1)
    assert roll["verts"] == 0 and roll["tris"] == 0 and roll["instances"] == 0
    assert "bounds" not in roll


def test_cli_help_survives_legacy_windows_codepage():
    """`moraine --help` on a cp932 console (Japanese-locale Windows) must not
    crash: argparse writes help text raw to stdout, so every help string has
    to stay pure ASCII. JSON command output is immune (ensure_ascii)."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONIOENCODING": "cp932"}
    proc = subprocess.run(
        [sys.executable, "-m", "moraine3d", "--help"],
        capture_output=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in proc.stderr


def test_every_help_page_is_ascii():
    """The ASCII rule covers every help page, not only the root: each
    sub-command (and cache's sub-sub-commands) writes its help raw to a
    possibly legacy-codepage console."""
    import argparse

    from moraine3d.app.cli import build_parser

    def walk(parser, path):
        assert parser.format_help().isascii(), path
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    walk(sub, path + [name])

    walk(build_parser(), ["moraine"])


def test_serve_no_open_reaches_the_server(tmp_path, monkeypatch):
    """`moraine serve --no-open` suppresses the viewer auto-open (DESIGN
    7.1); the default opens. The flag is the only way a harness that
    runs `serve` in the background opts out, so it must reach `run`."""
    from moraine3d.app import server as server_mod
    from moraine3d.app.cli import main

    asset = tmp_path / "asset.star"
    asset.write_text(
        'mo.output("main", mo.box(id = "bx", size = [1.0, 1.0, 1.0]))\n',
        encoding="utf-8",
    )
    calls: list[dict] = []
    monkeypatch.setattr(server_mod, "run", lambda *a, **kw: calls.append(kw))
    assert main(["serve", str(tmp_path), "--no-open"]) == 0
    assert main(["serve", str(tmp_path)]) == 0
    assert [kw["open_browser"] for kw in calls] == [False, True]


def test_server_channels_fail_fast_without_a_server(monkeypatch, capsys):
    """Every server channel pre-checks the loopback port before its
    request (DESIGN 6): with no server, the answer is `no_server` at
    once — never the channel's full timeout misreported as the server
    being busy, which is what a firewalled host produces when a closed
    port drops the SYN instead of refusing it."""
    import json

    from moraine3d.app.cli import main

    def no_network(req, timeout=None):
        raise AssertionError("the request must not be sent")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    for argv in (["screenshot"], ["pins", "list"]):
        assert main(argv) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "no_server", argv


def test_reachable_reads_the_real_socket():
    """The pre-check is a plain loopback connect: true for a listening
    socket, false for a closed port within the connect budget."""
    import socket
    import time

    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert _real_reachable(port) is True
    t0 = time.perf_counter()
    assert _real_reachable(port) is False
    assert time.perf_counter() - t0 < 1.0


def test_addressed_project_is_the_cwds_or_none(tmp_path, monkeypatch):
    """Server channels speak for the working directory's project and
    the root a server for it serves (DESIGN 6); with no asset there and
    no explicit target they speak for the displayed one (None), and an
    explicit target that resolves to nothing is the ordinary error,
    never a silent fall-through."""
    import pytest

    from moraine3d.app.cli import _addressed_project
    from moraine3d.errors import MoraineError

    proj = _write_project(tmp_path)
    assert _addressed_project(str(proj)) == (proj.name, proj.resolve())
    monkeypatch.chdir(proj)
    assert _addressed_project(None) == (proj.name, proj.resolve())
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert _addressed_project(None) == (None, None)
    with pytest.raises(MoraineError):
        _addressed_project(str(tmp_path / "nowhere"))


def test_server_channels_carry_the_addressed_project(tmp_path, monkeypatch, capsys):
    """`screenshot` and `pins` send the project they speak for (DESIGN
    6): in the body for POSTs, as a query for GET/DELETE."""
    import base64
    import json

    from moraine3d.app.cli import main

    proj = _write_project(tmp_path)
    monkeypatch.chdir(proj)
    bodies = _mock_screenshot_server(monkeypatch, b"png", _SHOT_META)
    assert main(["screenshot", "-o", str(tmp_path / "s.png")]) == 0
    assert bodies[-1]["project"] == proj.name
    capsys.readouterr()
    calls = _mock_pins_server(monkeypatch, {"pins": [], "project": proj.name})
    assert main(["pins", "list"]) == 0
    method, url, body = calls[-1]
    assert method == "GET" and url.endswith(f"/api/pins?project={proj.name}")
    calls = _mock_pins_server(monkeypatch, {"pin": {}, "pins": 1})
    assert main(["pins", "add", "--output", "main", "--face", "0"]) == 0
    assert calls[-1][2]["project"] == proj.name
    calls = _mock_pins_server(monkeypatch, {"cleared": 1})
    assert main(["pins", "clear", "--id", "3"]) == 0
    assert calls[-1][1].endswith(f"/api/pins?id=3&project={proj.name}")
    # ...and the root they speak for, so another root's server refuses.
    import urllib.parse

    assert urllib.parse.unquote(calls.roots[-1]) == str(proj.resolve())


def test_screenshot_waits_for_a_tab_before_opening_one(tmp_path, monkeypatch, capsys):
    """Sessions share the viewer (DESIGN 6): on `no_viewer` the CLI
    first waits for a tab that may already be on its way - a sibling
    session's recovery, or the human - so two shots racing on an empty
    server do not each add a tab."""
    import base64
    import json

    from moraine3d.app import cli as cli_mod
    from moraine3d.app.cli import main
    from moraine3d.errors import MoraineError

    envelope = json.dumps(
        {"meta": _SHOT_META, "png": base64.b64encode(b"png").decode()}
    ).encode()
    attempts: list[int] = []

    def joins(port, path, *, method, body=None, timeout_code, root=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise MoraineError("no_viewer", "no viewer connected")
        return envelope

    monkeypatch.setattr(cli_mod, "_server_request", joins)
    monkeypatch.setattr(cli_mod, "open_browser", lambda url: (_ for _ in ()).throw(
        AssertionError("must not open a tab while one is on its way")))
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: None)
    assert main(["screenshot", "-o", str(tmp_path / "shot.png")]) == 0
    assert len(attempts) == 3


def test_screenshot_opens_the_viewer_on_no_viewer_and_retries(tmp_path, monkeypatch, capsys):
    """When no tab appears on its own, the CLI opens the viewer itself
    and retries while it connects (DESIGN 6) - the step the error would
    otherwise ask the human for; with no browser the original error
    stands."""
    import base64
    import json

    from moraine3d.app import cli as cli_mod
    from moraine3d.app.cli import main
    from moraine3d.errors import MoraineError

    envelope = json.dumps(
        {"meta": _SHOT_META, "png": base64.b64encode(b"png").decode()}
    ).encode()
    attempts: list[int] = []

    def flaky(port, path, *, method, body=None, timeout_code, root=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise MoraineError("no_viewer", "no viewer connected; open http://x/ then retry")
        return envelope

    opened: list[str] = []
    monkeypatch.setattr(cli_mod, "_server_request", flaky)
    monkeypatch.setattr(cli_mod, "open_browser", lambda url: opened.append(url) or True)
    monkeypatch.setattr(cli_mod, "_VIEWER_JOIN_WAIT_S", 0.0)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda s: None)
    out = tmp_path / "shot.png"
    assert main(["screenshot", "-o", str(out)]) == 0
    assert opened == ["http://127.0.0.1:8642/"] and len(attempts) == 3
    assert out.read_bytes() == b"png"
    capsys.readouterr()

    attempts.clear()
    monkeypatch.setattr(cli_mod, "open_browser", lambda url: False)
    assert main(["screenshot", "-o", str(out)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "no_viewer"
    assert len(attempts) == 1  # no browser: no retry loop

    attempts.clear()
    monkeypatch.setattr(cli_mod, "open_browser", lambda url: True)
    monkeypatch.setattr(cli_mod, "_VIEWER_OPEN_WAIT_S", 0.0)
    monkeypatch.setattr(cli_mod, "_server_request",
                        lambda *a, **k: (_ for _ in ()).throw(
                            MoraineError("no_viewer", "still none")))
    assert main(["screenshot", "-o", str(out)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "no_viewer"


def test_version_flag_reports_terse_json(capsys):
    """`moraine --version` answers with the same terse-JSON contract as every
    command and exits 0 — it stamps journals and feedback, so it must be the
    first command an agent tries that works (DESIGN 6)."""
    import importlib.metadata
    import json

    import pytest

    from moraine3d.app.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"version": importlib.metadata.version("moraine3d")}


def test_docs_material_rename_is_addressable_and_inside_asset_ref(capsys):
    """material_rename is reachable by name from `moraine docs`, and the
    asset_ref contract it belongs to names it as a section under it."""
    from moraine3d.app.cli import main

    assert main(["docs", "material_rename"]) == 0
    assert capsys.readouterr().out.startswith("### material_rename")
    assert main(["docs", "asset_ref"]) == 0
    parent = capsys.readouterr().out
    assert "### material_rename" not in parent
    assert "sections: material_rename  (moraine docs <section>)" in parent.splitlines()


def test_docs_command_lists_and_slices(capsys):
    """`moraine docs` ships the operator contract with the install
    (DESIGN 6): list sections, slice one op, error on unknown."""
    import json

    from moraine3d.app.cli import main

    assert main(["docs"]) == 0
    listing = json.loads(capsys.readouterr().out)
    by_doc = {d["doc"]: [s.split()[0] for s in d["sections"]] for d in listing["docs"]}
    assert list(by_doc) == ["recipe.md", "api", "asset.md", "ops.md", "textures.md",
                            "checks.md", "assertions.md", "cli.md", "journal.md",
                            "examples.md"]
    assert "box" in by_doc["ops.md"] and "place" in by_doc["assertions.md"]
    assert by_doc["recipe.md"] == ["recipe", "builtins", "ids", "load",
                                   "requirements", "knobs", "refusals"]

    # A file name from the index lists that file's sections.
    assert main(["docs", "recipe.md"]) == 0
    one = json.loads(capsys.readouterr().out)
    assert one["doc"] == "recipe.md"
    assert [s.split()[0] for s in one["sections"]] == by_doc["recipe.md"]

    # A section is its own text, not JSON around it: prose is read.
    assert main(["docs", "place"]) == 0
    section = capsys.readouterr().out
    assert section.startswith("### place")
    assert "instance" in section  # per-instance anchoring documented
    assert "## " not in section.replace("### place", "")  # one section only

    # A heading with sections under it prints its own text alone and
    # names them: the reader asks for the one it needs.
    assert main(["docs", "Selection"]) == 0
    head = capsys.readouterr().out
    assert head.startswith("## Selection") and "### " not in head
    assert head.splitlines()[-1].startswith("sections: ")
    assert " place," in head.splitlines()[-1]

    # The CLI reference's sections resolve by their first token too —
    # a `moraine`-prefixed title would hide `query` behind `moraine`.
    assert main(["docs", "query"]) == 0
    assert capsys.readouterr().out.startswith("### query (CLI")

    assert main(["docs", "warp_drive"]) == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "unknown_section"
    assert "did_you_mean" not in err["error"].get("detail", {})

    # The journal format section is reachable by its first token, and
    # the section matcher names near misses instead of only the listing
    # (`checker` is the field's miss for `checker_texture`).
    assert main(["docs", "journal"]) == 0
    section = capsys.readouterr().out
    assert "## Entries" in section and "workaround" in section
    assert main(["docs", "checker"]) == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["detail"]["did_you_mean"] == ["checker_texture"]
    assert "checker_texture" in err["error"]["message"]


def test_docs_resolves_full_title_prefix_and_hints_on_titles(capsys):
    """The guide names sections by their title head ("Asset assertions",
    "Face predicates"): a case-insensitive full-title prefix that covers
    the first token resolves, and the near-miss hint searches full titles
    (`assertions`), not only first tokens."""
    import json

    from moraine3d.app.cli import main

    for query, head in (
        ("Asset assertions", "### Asset assertions"),
        ("face predicates", "### Face predicates"),
        ("mated", "### mated"),
        ("mat", None),
    ):
        code = main(["docs", query])
        out = capsys.readouterr().out
        if head is None:
            assert code == 1 and json.loads(out)["error"]["code"] == "unknown_section"
        else:
            assert code == 0, query
            assert out.startswith(head)
    assert main(["docs", "assertions"]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert any(c.startswith("Asset assertions") for c in err["detail"]["did_you_mean"])
    # A partial first token stays a near miss, never a silent prefix hit.
    assert main(["docs", "sca"]) == 1


_SHOT_META = {
    "project": "kit",
    "asset": "kit/kit.mora",
    "asset_hash": "0" * 24,
    "moraine_version": "0.0.1",
    "watch": "alive",
    "camera": {"azimuth": 45.0, "elevation": 29.516, "distance": 2.82, "target": [0.0, 0.5, 0.0]},
    "size": [512, 512],
}


def _mock_screenshot_server(monkeypatch, png: bytes, meta: dict):
    """Stand in for urlopen: capture the request body, answer with the
    server's {meta, png} envelope. Returns the captured-bodies list."""
    import base64
    import io
    import json

    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode("utf-8")))
        envelope = {"meta": meta, "png": base64.b64encode(png).decode("ascii")}
        return io.BytesIO(json.dumps(envelope).encode("utf-8"))

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return bodies


def test_screenshot_creates_output_parents(tmp_path, monkeypatch, capsys):
    """`screenshot -o` into a directory that does not exist yet writes the
    file instead of failing — a missing parent must not cost an authoring
    round — and the report carries the
    provenance meta: a stale or foreign shot is detectable from the
    report alone (DESIGN 6)."""
    import json

    from moraine3d.app.cli import main

    png = b"\x89PNG fake bytes"
    meta = {**_SHOT_META, "view": "iso"}
    _mock_screenshot_server(monkeypatch, png, meta)
    out = tmp_path / "shots" / "iso" / "kit.png"
    assert main(["screenshot", "-o", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert out.read_bytes() == png
    assert payload["bytes"] == len(png)
    assert payload["view"] == "iso"
    assert payload["port"] == 8642
    assert payload["project"] == "kit"
    assert payload["asset"] == "kit/kit.mora"
    assert payload["asset_hash"] == "0" * 24
    assert payload["moraine_version"] == "0.0.1"
    assert payload["watch"] == "alive"
    assert payload["camera"]["azimuth"] == 45.0


def test_screenshot_free_camera_flags_ride_the_request(tmp_path, monkeypatch, capsys):
    """Free-camera flags land in the request body as numbers, with no view
    key; --target parses X,Y,Z; only flags actually passed are sent
    (DESIGN 6)."""
    import json

    from moraine3d.app.cli import main

    meta = dict(_SHOT_META)  # no "view": free camera
    bodies = _mock_screenshot_server(monkeypatch, b"png", meta)
    out = tmp_path / "b.png"
    assert main([
        "screenshot", "--azimuth", "120", "--elevation", "10",
        "--target", "0,0.5,0", "-o", str(out),
    ]) == 0
    assert bodies == [{
        "size": [512, 512], "azimuth": 120.0, "elevation": 10.0,
        "target": [0.0, 0.5, 0.0],
    }]
    payload = json.loads(capsys.readouterr().out)
    assert "view" not in payload


def _mock_pins_server(monkeypatch, answer: dict):
    """Stand in for urlopen on the pins surface: capture (method, url,
    body) per request, answer with the given JSON object."""
    import io
    import json

    class _Calls(list):
        roots: list

    calls = _Calls()
    roots: list = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        calls.append((req.get_method(), req.full_url, body))
        roots.append(req.get_header("X-moraine-root"))
        return io.BytesIO(json.dumps(answer).encode("utf-8"))

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    calls.roots = roots
    return calls


def test_pins_cli_talks_to_the_server(monkeypatch, capsys):
    """`moraine pins list|add|clear` drive the running server's pin
    surface and relay its report whole — the resolved record is the
    placement's own verification (DESIGN 6)."""
    import json

    from moraine3d.app.cli import main

    pin = {"id": 1, "output": "main", "face": 4, "groups": ["top"],
           "position": [0.0, 1.0, 0.0], "source": "api", "asset_hash": "0" * 24}
    calls = _mock_pins_server(monkeypatch, {"pin": pin, "pins": 1})
    assert main([
        "pins", "add", "--output", "main",
        "--select", '{"normal_y": {"gt": 0.9}}',
    ]) == 0
    assert json.loads(capsys.readouterr().out)["pin"]["groups"] == ["top"]
    assert calls[-1] == (
        "POST", "http://127.0.0.1:8642/api/pins",
        {"output": "main", "predicate": {"normal_y": {"gt": 0.9}}},
    )
    assert main([
        "pins", "add", "--output", "row", "--face", "7", "--instance", "2",
    ]) == 0
    capsys.readouterr()
    assert calls[-1][2] == {"output": "row", "face": 7, "instance": 2}

    calls = _mock_pins_server(
        monkeypatch, {"project": "p", "asset_hash": "0" * 24, "pins": [pin]}
    )
    assert main(["pins", "list"]) == 0
    assert len(json.loads(capsys.readouterr().out)["pins"]) == 1
    assert calls[-1][0] == "GET"

    calls = _mock_pins_server(monkeypatch, {"cleared": 1})
    assert main(["pins", "clear", "--id", "1"]) == 0
    capsys.readouterr()
    assert calls[-1][:2] == ("DELETE", "http://127.0.0.1:8642/api/pins?id=1")


def test_pins_cli_maps_no_server_and_records_the_command(tmp_path, monkeypatch, capsys):
    """With no server on the port, `pins` fails with the actionable
    no_server error, and the command event is named `pins.<sub>` like
    `cache.<sub>` (DESIGN 6)."""
    import json
    import urllib.error

    from moraine3d.app import session
    from moraine3d.app.cli import main

    def refuse(req, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert main(["pins", "list"]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "no_server" and "moraine serve" in err["message"]
    # the shortest path to a fix names both: start one, or point at the
    # one already listening on another port
    assert "--port" in err["message"]
    evs = session.read_events(tmp_path / "log")
    assert evs[-1]["cmd"] == "pins.list" and evs[-1]["ok"] is False


def test_screenshot_non_envelope_response_is_actionable(tmp_path, monkeypatch, capsys):
    """A 200 that is not the {meta, png} envelope — a pre-upgrade server
    answering with raw PNG bytes — must become an actionable stale_server
    error, never a traceback or a corrupt shot file (DESIGN 2 P2)."""
    import io
    import json

    from moraine3d.app.cli import main

    _server_answers(monkeypatch)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: io.BytesIO(b"\x89PNG raw pre-upgrade body"),
    )
    out = tmp_path / "skew.png"
    assert main(["screenshot", "-o", str(out)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "stale_server"
    assert "restart 'moraine serve'" in payload["error"]["message"]
    assert not out.exists()  # never write a corrupt file


def test_screenshot_timeout_is_actionable(monkeypatch, capsys):
    """A read/header timeout (server mid-evaluation past the 60 s budget)
    surfaces as screenshot_timeout, not as an uncaught TimeoutError and
    not misdiagnosed as no_server (DESIGN 2 P2)."""
    import json
    import urllib.error

    from moraine3d.app.cli import main

    def slow(req, timeout=None):
        raise TimeoutError("timed out")

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", slow)
    assert main(["screenshot"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "screenshot_timeout"
    assert "retry" in payload["error"]["message"]

    def slow_wrapped(req, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("urllib.request.urlopen", slow_wrapped)
    assert main(["screenshot"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "screenshot_timeout"


def test_screenshot_camera_flag_conflict_is_local(monkeypatch, capsys):
    """--view with --azimuth/--elevation fails fast client-side with the
    server's error code — no network round trip spent on it."""
    import json

    from moraine3d.app.cli import main

    def no_network(req, timeout=None):
        raise AssertionError("conflict must not reach the server")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    assert main(["screenshot", "--view", "front", "--azimuth", "30"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "camera_conflict"


def test_screenshot_bad_target(monkeypatch, capsys):
    import json

    from moraine3d.app.cli import main

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(AssertionError("no network")),
    )
    assert main(["screenshot", "--target", "1,2"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "bad_camera"


def test_screenshot_rejects_nonfinite_camera_locally(monkeypatch, capsys):
    """Non-finite camera numbers (inf azimuth, nan target) fail fast
    client-side with the server's bad_camera code, not as a 500 or a NaN
    pose the JSON output cannot carry (camera.resolve is the authority)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(AssertionError("no network")),
    )
    for argv in (["--azimuth", "inf"], ["--elevation", "nan"], ["--distance", "inf"],
                 ["--target", "0,nan,0"]):
        assert main(["screenshot", *argv]) == 1, argv
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_camera"


def test_screenshot_unwritable_output_is_actionable(tmp_path, monkeypatch, capsys):
    """`screenshot -o <existing directory>` (or any unwritable path) is a
    write_failed JSON error naming the path and the fix, not a raw
    OSError traceback after the render (DESIGN 2 P2)."""
    import json

    from moraine3d.app.cli import main

    _mock_screenshot_server(monkeypatch, b"png", _SHOT_META)
    assert main(["screenshot", "-o", str(tmp_path)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "write_failed"
    assert str(tmp_path) in err["message"] and "-o" in err["message"]


def test_port_range_is_validated_before_evaluating(tmp_path, monkeypatch, capsys):
    """`serve --port` / `screenshot --port` outside 1..65535 is a bad_port
    JSON error before any evaluation or request, not a raw OverflowError
    from bind()."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setattr("moraine3d.app.server.run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("serve must not start")))
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: (_ for _ in ()).throw(
        AssertionError("no request")))
    for argv in (["serve", "--port", "99999"], ["serve", "--port", "0"],
                 ["screenshot", "--port", "-1"]):
        assert main(argv) == 1, argv
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "bad_port" and "1..65535" in err["message"]


def test_server_run_maps_overflow_port_to_bad_port(tmp_path, monkeypatch):
    """The server's own backstop for programmatic callers: bind() on an
    out-of-range port surfaces as bad_port, not OverflowError."""
    from moraine3d.app import server as server_mod
    from moraine3d.errors import MoraineError

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)

    import pytest

    with pytest.raises(MoraineError) as exc:
        server_mod.run(proj / "asset.mora", port=99999, open_browser=False)
    assert exc.value.code == "bad_port"


def test_server_run_maps_address_in_use_to_port_in_use(tmp_path, monkeypatch):
    """A bind refused because the port is taken is the running-server
    case, not a crash: it surfaces as port_in_use naming the fix — use
    the server every command talks to by default, or --port another
    (DESIGN 6, 8.2) — under either errno spelling the platforms raise
    (EADDRINUSE, or Winsock's 10048 when the C errno differs)."""
    import errno

    import pytest

    from moraine3d.app import server as server_mod
    from moraine3d.errors import MoraineError

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    for code in (errno.EADDRINUSE, 10048):
        def bind_taken(port, _code=code):
            raise OSError(_code, "address already in use")

        monkeypatch.setattr(server_mod, "_claim_port", bind_taken)
        with pytest.raises(MoraineError) as exc:
            server_mod.run(proj / "asset.mora", port=8642, open_browser=False)
        assert exc.value.code == "port_in_use", code
        assert "8642" in exc.value.message and "--port" in exc.value.message


def test_server_run_claims_the_port_before_it_evaluates(tmp_path, monkeypatch):
    """`serve` claims its port before the first evaluation (DESIGN 6):
    on a held port it reports port_in_use alone, with nothing evaluated
    and nothing logged; on a free one the asset is evaluated and the
    `serving` line follows once the site listens (DESIGN 8.2: an error
    states what failed, not a success it did not reach)."""
    import asyncio
    import socket

    import pytest

    from moraine3d.app import server as server_mod
    from moraine3d.errors import MoraineError

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        server_mod, "_log", lambda event, **fields: events.append(event)
    )

    async def stop_after_bind(self, *args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(server_mod.MoraineServer, "_rearm_forever", stop_after_bind)

    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        with pytest.raises(MoraineError) as exc:
            server_mod.run(proj / "asset.star", port=port, open_browser=False)
        assert exc.value.code == "port_in_use"
        assert events == []
    finally:
        held.close()

    server_mod.run(proj / "asset.star", port=port, open_browser=False)
    assert events.index("asset_updated") < events.index("serving")


def test_server_run_listens_when_the_first_evaluation_fails(tmp_path, monkeypatch):
    """On evaluation failure the server keeps standing and reports the
    failure through the error channel (DESIGN 7.2): an asset broken at
    startup is logged as an error and the site still listens."""
    import asyncio

    from moraine3d.app import server as server_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    (proj / "asset.star").write_text("mo.box(", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        server_mod, "_log", lambda event, **fields: events.append(event)
    )

    async def stop_after_bind(self, *args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(server_mod.MoraineServer, "_rearm_forever", stop_after_bind)
    server_mod.run(proj / "asset.star", port=0, open_browser=False)
    assert "asset_updated" not in events
    assert events.index("asset_error") < events.index("serving")


def test_memory_error_and_bugs_exit_one_as_json(monkeypatch, capsys):
    """A kernel MemoryError (huge counts/segments) and any other
    unexpected exception keep the terse-JSON error contract and exit 1
    (DESIGN 2 P2) — never a raw traceback with exit 0; the traceback of
    a bug still goes to stderr so the report is not lost."""
    import json

    from moraine3d.app.cli import main

    def oom(args):
        raise MemoryError("Unable to allocate 745 GiB")

    monkeypatch.setattr("moraine3d.app.cli._cmd_stats", oom)
    assert main(["stats", "."]) == 1
    captured = capsys.readouterr()
    err = json.loads(captured.out)["error"]
    assert err["code"] == "out_of_memory" and "reduce" in err["message"]

    def bug(args):
        raise KeyError("verts")

    monkeypatch.setattr("moraine3d.app.cli._cmd_stats", bug)
    assert main(["stats", "."]) == 1
    captured = capsys.readouterr()
    err = json.loads(captured.out)["error"]
    assert err["code"] == "internal_error"
    assert "KeyError" in err["message"] and "report it" in err["message"]
    assert "Traceback" in captured.err and "KeyError" in captured.err


def test_init_ownership_per_file(tmp_path, capsys):
    """`moraine init` (DESIGN 4.2): the human-owned AGENTS.md is the
    only create-once file — never overwritten; the authoring guide is
    factory-owned — re-synced to the installed version on every init.
    Harness instruction files are the human's: init writes no
    CLAUDE.md and leaves one it finds alone."""
    import json

    from moraine3d.app.cli import main

    target = tmp_path / "ws"
    target.mkdir()
    (target / "CLAUDE.md").write_text("# My own project rules\n", encoding="utf-8")
    assert main(["init", str(target)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["created"] == ["AGENTS.md"]
    assert out["guide"] == "created" and "adapter" not in out
    assert (target / "CLAUDE.md").read_text(encoding="utf-8") == "# My own project rules\n"
    # The human's file carries the prose pointer to the guide for
    # harnesses that read it without an import mechanism.
    agents = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert ".moraine/guide.md" in agents
    guide = (target / ".moraine" / "guide.md").read_text(encoding="utf-8")
    assert guide.startswith("# Moraine authoring guide")

    # The human's files are never overwritten; the stale factory file
    # (the guide) is refreshed.
    (target / "AGENTS.md").write_text("mine", encoding="utf-8")
    (target / ".moraine" / "guide.md").write_text("stale", encoding="utf-8")
    assert main(["init", str(target)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["created"] == [] and out["skipped"] == ["AGENTS.md"]
    assert out["guide"] == "updated"
    assert (target / "AGENTS.md").read_text(encoding="utf-8") == "mine"
    assert (target / "CLAUDE.md").read_text(encoding="utf-8") == "# My own project rules\n"
    assert (target / ".moraine" / "guide.md").read_text(encoding="utf-8") == guide

    # Third run: everything current, nothing to do.
    assert main(["init", str(target)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["guide"] == "current"


def test_init_errors_are_terse_json(tmp_path, capsys):
    """init filesystem failures surface as the CLI's JSON error contract,
    never raw tracebacks (same class as cache_unwritable)."""
    import json

    from moraine3d.app.cli import main

    blocker = tmp_path / "blocked"
    blocker.write_text("a file where the project dir should go", encoding="utf-8")
    assert main(["init", str(blocker)]) == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "init_failed"
    assert "writable directory" in err["error"]["message"]

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".moraine").write_text("a file squatting on the tool dir", encoding="utf-8")
    assert main(["init", str(proj)]) == 1
    err = json.loads(capsys.readouterr().out)
    assert err["error"]["code"] == "init_failed"


ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"outputs": {
"main": "bx01"
}
}
"""


def _write_project(tmp_path):
    from support import projects

    proj = tmp_path / "proj"
    projects.write_project(proj, ASSET, name="asset")
    return proj


# The bare ASSET box is centered and would fail `not_sunk`; workspace
# sweep tests need validate-green projects, so they waive it.
GREEN_ASSET = ASSET.replace(
    '"outputs": {\n"main": "bx01"\n}',
    '"outputs": {\n"main": "bx01"\n},\n"checks": {\n"main": {"not_sunk": false}\n}',
)
assert GREEN_ASSET != ASSET  # str.replace misses silently on a reformat
from moraine3d.core import asset as _asset_mod  # noqa: E402

assert _asset_mod.dumps(_asset_mod.loads(ASSET)) == ASSET
assert _asset_mod.dumps(_asset_mod.loads(GREEN_ASSET)) == GREEN_ASSET


def _recipe(project, fixture, name, sub=False):
    """One asset of a project as a recipe, from its IR fixture: the
    root, or with ``sub`` a sub-asset."""
    import json

    from support import flatten
    from support.ir import with_knob_values

    doc = json.loads(fixture) if isinstance(fixture, str) else fixture
    return flatten.write_doc(with_knob_values(doc), project, name, sub=sub)


def _write_workspace(tmp_path, projects=("alpha", "beta")):
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    for name in projects:
        _recipe(ws / name, GREEN_ASSET, name)
    return ws


def test_eval_and_stats_report_profile(tmp_path, monkeypatch, capsys):
    """`eval`/`stats` JSON carries a sibling `profile` key with cache
    hit/miss counts and per-node wall time (DESIGN 6): cold
    pass = misses with a `slowest` list, warm pass = hits without one."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)

    assert main(["eval", str(proj)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"moraine_version", "outputs", "profile"}
    assert (out["profile"]["hits"], out["profile"]["misses"]) == (0, 1)
    assert out["profile"]["slowest"][0]["node"] == "bx01"

    assert main(["stats", str(proj)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"moraine_version", "asset", "outputs", "profile"}
    assert (out["profile"]["hits"], out["profile"]["misses"]) == (1, 0)
    assert "slowest" not in out["profile"]  # warm pass: nothing to optimize


def test_stats_still_writes_no_glb(tmp_path, monkeypatch, capsys):
    """Instrumentation must not violate the cost tiering: `stats` reads
    cache metadata and never materializes GLBs (DESIGN 8.2)."""
    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    assert main(["stats", str(proj)]) == 0
    capsys.readouterr()
    assert list((tmp_path / "cache").glob("*.glb")) == []


def test_cache_subcommands(tmp_path, monkeypatch, capsys):
    """`moraine cache stats|clear|evict` (DESIGN 4.5): terse JSON, and
    clear leaves a cache that regenerates byte-identically."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)

    assert main(["eval", str(proj)]) == 0
    glb_before = json.loads(capsys.readouterr().out)["outputs"]["main"]["glb"]
    baseline = open(glb_before, "rb").read()

    assert main(["cache", "stats"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["keys"] == 1 and s["bytes"] > 0 and s["cap"] == 0

    assert main(["cache", "evict"]) == 0
    e = json.loads(capsys.readouterr().out)
    assert e["cap"] == 0 and e["evicted"] == 0  # cap 0 = unlimited, no-op

    assert main(["cache", "clear"]) == 0
    c = json.loads(capsys.readouterr().out)
    assert c["cleared"] == 1 and c["freed"] > 0
    # Only the CACHEDIR.TAG stamped by the eval's write path survives.
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["CACHEDIR.TAG"]

    # Everything re-derives from the recipe, byte-identical (DESIGN 4.2).
    assert main(["eval", str(proj)]) == 0
    glb_after = json.loads(capsys.readouterr().out)["outputs"]["main"]["glb"]
    assert open(glb_after, "rb").read() == baseline


def test_cli_exit_check_evicts_after_growth(tmp_path, monkeypatch, capsys):
    """The process that grew the cache checks after growing (DESIGN
    4.5): with a tiny cap, an eval that wrote bytes triggers eviction
    at command end without touching stdout or the exit code; a read-only
    command never scans."""
    import json

    from moraine3d.app.cli import main
    from moraine3d.cache import store as cache_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "1")
    proj = _write_project(tmp_path)

    # Freshness guard would protect just-written records; pin it off so
    # the exit check's eviction is observable.
    monkeypatch.setattr(cache_mod, "FRESH_S", 0.0)
    assert main(["eval", str(proj)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "outputs" in out  # stdout is the command's JSON, nothing else
    # Evicted at exit: only the dir tag remains.
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["CACHEDIR.TAG"]

    # Recording wrapper, not a raising bomb: _housekeep swallows all
    # exceptions by contract, so a bomb could never fail the test.
    scans: list[int] = []
    real_scan = cache_mod.scan
    monkeypatch.setattr(cache_mod, "scan", lambda: scans.append(1) or real_scan())
    assert main(["docs"]) == 0
    capsys.readouterr()
    assert scans == []  # read-only command: no cache scan at exit


def test_housekeep_failure_never_flips_exit_code(tmp_path, monkeypatch, capsys):
    """Housekeeping must never change the command's stdout or exit code,
    whatever it raises: an unexpected exception class from eviction is
    reported on stderr, not propagated."""
    from moraine3d.app.cli import main
    from moraine3d.cache import store as cache_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def boom():
        raise RuntimeError("home directory unresolvable")

    monkeypatch.setattr(cache_mod, "maybe_evict", boom)
    proj = _write_project(tmp_path)
    assert main(["eval", str(proj)]) == 0  # exit code stands
    out, err = capsys.readouterr()
    assert "outputs" in out and "cache eviction skipped" in err


def test_serve_exit_disarms_the_eviction_counter(tmp_path, monkeypatch):
    """serve settles eviction via its own debounced check: returning from
    the serve command (Ctrl-C) leaves the session counter disarmed so
    the CLI tail never runs a synchronous full-cache scan at shutdown."""
    from moraine3d.app.cli import main
    from moraine3d.cache import store as cache_mod

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("moraine3d.app.server.run", lambda *a, **kw: None)
    proj = _write_project(tmp_path)
    cache_mod.note_write(1234)  # bytes serve wrote inside its debounce window

    # Recording wrapper, not a raising bomb: _housekeep swallows all
    # exceptions by contract, so a bomb could never fail this test — and
    # without the disarm, maybe_evict itself would zero the counter,
    # so only the scan count can distinguish the two.
    scans: list[int] = []
    real_scan = cache_mod.scan
    monkeypatch.setattr(cache_mod, "scan", lambda: scans.append(1) or real_scan())
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "1")
    assert main(["serve", str(proj)]) == 0
    assert scans == []  # the tail never scanned: serve's exit stays instant
    assert cache_mod.session_wrote() == 0


def test_stats_sweep_nests_ordinary_reports(tmp_path, monkeypatch, capsys):
    """The workspace sweep nests each project's ordinary single-project
    report unchanged under its directory name, with a summed aggregate
    that omits bounds — projects share no coordinate space (DESIGN 6)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)

    assert main(["stats", str(ws)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"projects", "workspace"}
    assert list(out["projects"]) == ["alpha", "beta"]  # lexicographic
    # Each slot has exactly the single-project shape.
    for slot in out["projects"].values():
        assert set(slot) == {"asset", "outputs", "profile"}
    roll = out["workspace"]
    assert roll["projects"] == 2 and roll["failed"] == []
    assert roll["verts"] == 2 * 24 and roll["tris"] == 2 * 12
    assert "bounds" not in roll


def test_validate_sweep_aggregate_exit_code(tmp_path, monkeypatch, capsys):
    """One failing project makes the sweep exit 1 with its error rendered
    in its own slot — never at the top level — while siblings still
    produce full reports (DESIGN 6)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    bad = ws / "abad"
    bad.mkdir()
    (bad / "abad.star").write_text("{not a recipe", encoding="utf-8")

    assert main(["validate", str(ws)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert "error" not in out  # per-project errors stay in their slots
    assert list(out["projects"]) == ["abad", "alpha", "beta"]
    assert "error" in out["projects"]["abad"]
    for name in ("alpha", "beta"):  # sweep continued past the failure
        assert out["projects"][name]["ok"] is True
    from moraine3d import dist_version
    assert out["workspace"] == {
        "moraine_version": dist_version(),
        "ok": False, "projects": 3, "failed": ["abad"], "not_swept": [],
    }


def test_validate_sweep_with_cross_project_ref(tmp_path, monkeypatch, capsys):
    """A workspace sweep stays green when one project cross-references a
    sibling's root output (DESIGN 4.6): the ref resolves during alpha's
    evaluation, and the sweep still visits beta as its own project."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    _recipe(
        ws / "alpha",
        json.dumps(
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
                "checks": {"main": {"not_sunk": False}},
            }
        ),
        "alpha",
    )

    assert main(["validate", str(ws)]) == 0
    out = json.loads(capsys.readouterr().out)
    from moraine3d import dist_version
    assert out["workspace"] == {
        "moraine_version": dist_version(),
        "ok": True, "projects": 2, "failed": [], "not_swept": [],
    }


def test_stats_sweep_counts_shared_geometry_per_consumer(tmp_path, monkeypatch, capsys):
    """The workspace aggregate stays a plain per-project sum after
    cross-project refs (DESIGN 6): a consumer's exports
    contain the geometry it references — every project's GLB is
    self-contained — so referenced geometry counts in the consumer's
    slot and again in the author's."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    _recipe(
        ws / "beta",
        json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {
                        "id": "rf01",
                        "op": "asset_ref",
                        "params": {"project": "alpha", "output": "main"},
                    }
                ],
                "outputs": {"main": "rf01"},
            }
        ),
        "beta",
    )

    assert main(["stats", str(ws)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["projects"]["beta"]["asset"]["verts"] == 24  # the referenced box
    assert out["workspace"]["verts"] == 2 * 24  # per-consumer, not dedup'd


def test_validate_sweep_with_broken_sibling_fails_both_slots(tmp_path, monkeypatch, capsys):
    """When a cross-referenced sibling is broken, the consumer's slot
    reports the load failure with the sibling file named
    workspace-relative, the sibling's own slot fails too, and the
    aggregate exit code is 1 (DESIGN 6 + 4.6)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    _recipe(
        ws / "alpha",
        json.dumps(
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
            }
        ),
        "alpha",
    )
    (ws / "beta" / "beta.star").write_text("x = 1\nwhile x:\n    pass\n", encoding="utf-8")

    assert main(["validate", str(ws)]) == 1
    out = json.loads(capsys.readouterr().out)
    alpha_err = out["projects"]["alpha"]["error"]
    assert alpha_err["code"] == "unsupported_syntax"
    assert alpha_err["detail"]["site"] == "beta/beta.star:2"
    assert out["projects"]["beta"]["error"]["code"] == "unsupported_syntax"
    assert out["workspace"]["failed"] == ["alpha", "beta"]


def test_write_commands_reject_workspace_target(tmp_path, monkeypatch, capsys):
    """eval/query/export take one project per invocation (DESIGN 6):
    a workspace target is the actionable workspace_target error listing
    the discovered projects, not a bare asset_not_found. (serve accepts
    a workspace root since multi-asset serve, DESIGN 7.2.)"""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)

    for argv in (["eval", str(ws)],
                 ["query", str(ws), "--select", '{"normal_y": {"gt": 0.9}}'],
                 ["export", str(ws)]):
        assert main(argv) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "workspace_target"
        assert err["detail"]["projects"] == ["alpha", "beta"]


def test_export_out_resolves_against_project(tmp_path, monkeypatch, capsys):
    """DESIGN 6: a relative `export -o` resolves against the project, so
    the default `exports` lands inside the project from any invocation
    directory - never a stray directory beside the workspace root - and
    the report's glb/manifest paths are cwd-relative (never absolute by
    default: reports get quoted into committed journals)."""
    import json
    from pathlib import Path

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    monkeypatch.chdir(ws)

    assert main(["export", "alpha"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (ws / "alpha" / "exports" / "alpha.glb").is_file()
    assert (ws / "alpha" / "exports" / "alpha.manifest.json").is_file()
    assert not (ws / "exports").exists()
    assert result["glb"] == str(Path("alpha") / "exports" / "alpha.glb")
    assert result["manifest"] == str(
        Path("alpha") / "exports" / "alpha.manifest.json"
    )


def test_export_red_validate_creates_no_directory(tmp_path, monkeypatch, capsys):
    """DESIGN 8.2/9: a failed validation writes nothing - project-anchored
    -o resolution stays pure path math, so the red path still creates no
    output directory anywhere."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "sunk"
    # Default box sinks below ground: not_sunk fails, nothing is written.
    _recipe(proj, ASSET, "sunk")
    monkeypatch.chdir(tmp_path)

    assert main(["export", "sunk"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert not (proj / "exports").exists()
    assert not (tmp_path / "exports").exists()


def test_export_absolute_out_unchanged(tmp_path, monkeypatch, capsys):
    """DESIGN 6: an absolute -o is taken as given - files land there and
    the report echoes the resolved path (the author chose it)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.chdir(ws / "beta")

    assert main(["export", str(ws / "alpha"), "-o", str(elsewhere)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (elsewhere / "alpha.glb").is_file()
    assert result["glb"] == str(elsewhere / "alpha.glb")


def test_export_drive_relative_out_refused(tmp_path, monkeypatch, capsys):
    """DESIGN 6: Windows drive-relative or rooted -o spellings are
    neither relative nor absolute - refused rather than written
    somewhere silent (a drive root or another drive's cwd)."""
    import json
    import os

    import pytest

    from moraine3d.app.cli import main

    if os.name != "nt":
        pytest.skip("drive-relative/rooted -o is a Windows path spelling")
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    monkeypatch.chdir(ws)

    for spelling in (r"\exports", "D:exports"):
        assert main(["export", "alpha", "-o", spelling]) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "bad_out_dir", spelling
        assert "absolute" in err["message"]
    assert not (ws / "alpha" / "exports").exists()


def test_export_subasset_stem_collision_refused(tmp_path, monkeypatch, capsys):
    """DESIGN 6: a sub-asset sharing its stem with the project's root
    asset is refused at export - every out dir is shared per project,
    so its pair would silently overwrite the root's; a distinct stem
    exports without complaint."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path)
    sub = ws / "alpha" / "subassets"
    _recipe(ws / "alpha", GREEN_ASSET, "alpha", sub=True)
    _recipe(ws / "alpha", GREEN_ASSET, "lid", sub=True)

    assert main(["export", str(sub / "alpha.star")]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "export_stem_collision"
    assert "rename" in err["message"]
    assert not (ws / "alpha" / "exports").exists()

    assert main(["export", str(sub / "lid.star")]) == 0
    capsys.readouterr()
    assert (ws / "alpha" / "exports" / "lid.glb").is_file()


def test_serve_workspace_serves_first_project(tmp_path, monkeypatch, capsys):
    """serve on a workspace root starts on the lexicographically first
    project and passes the frozen roster to the server (DESIGN 6, 7.2);
    serve on a project directory — a workspace member included — stays
    single-asset, exactly what was pointed at (no roster, no widening)."""
    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    calls: list[tuple] = []
    monkeypatch.setattr(
        "moraine3d.app.server.run",
        lambda asset, port=None, workspace=None, project=None, **kw:
            calls.append((asset, workspace, project)),
    )
    ws = _write_workspace(tmp_path)

    assert main(["serve", str(ws)]) == 0
    asset, workspace, project = calls[-1]
    assert asset == ws / "alpha" / "alpha.star"
    assert workspace.projects == ("alpha", "beta")
    assert project == "alpha"

    # A project dir inside the workspace serves single-asset: no
    # upward widening to the enclosing workspace.
    assert main(["serve", str(ws / "beta")]) == 0
    asset, workspace, project = calls[-1]
    assert asset == ws / "beta" / "beta.star"
    assert workspace is None and project is None

    proj = _write_project(tmp_path)
    assert main(["serve", str(proj)]) == 0
    asset, workspace, project = calls[-1]
    assert asset == proj / "asset.star"
    assert workspace is None
    capsys.readouterr()


def test_serve_workspace_startup_failures_are_cli_errors(tmp_path, monkeypatch, capsys):
    """A serve target that cannot start — empty workspace, or an
    ambiguous first project — fails as terse CLI JSON before the server
    starts, never mid-serve (DESIGN 6, 8.2)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def _serve_must_not_run(*a, **kw):
        raise AssertionError("serve ran")

    monkeypatch.setattr("moraine3d.app.server.run", _serve_must_not_run)

    empty = _write_workspace(tmp_path, projects=())
    assert main(["serve", str(empty)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "empty_workspace"

    ws = _write_workspace(tmp_path / "two")
    _recipe(ws / "alpha", ASSET, "extra")
    assert main(["serve", str(ws)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "ambiguous_asset"


def test_empty_workspace_is_actionable_error(tmp_path, monkeypatch, capsys):
    """A workspace with no projects is a misconfiguration surfaced as
    empty_workspace — never an empty green sweep, and never advice to
    pass one of zero projects (DESIGN 6). The write-side hits it on the
    natural first-run flow (init, then serve)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ws = _write_workspace(tmp_path, projects=())

    for argv in (["stats", str(ws)], ["validate", str(ws)], ["eval", str(ws)]):
        assert main(argv) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "empty_workspace"
        assert "child directory holding one root asset" in err["message"]


def test_workspace_asset_conflict_target_is_refused(tmp_path, monkeypatch, capsys):
    """A directory with both .moraine/ and a top-level asset file is a
    diagnosed layout conflict (DESIGN 4.2) — commands refuse it with the
    fix named, never resolve it as a project or a workspace."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    (proj / ".moraine").mkdir()

    for argv in (["stats", str(proj)], ["eval", str(proj)]):
        assert main(argv) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "workspace_asset_conflict"
        assert "child project directory" in err["message"]


def test_init_refuses_project_target(tmp_path, monkeypatch, capsys):
    """`init` refuses a target holding a top-level asset file rather than
    create the workspace/project layout conflict (DESIGN 4.2)."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)

    assert main(["init", str(proj)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "workspace_asset_conflict"
    assert "child project directory" in err["message"]
    assert not (proj / ".moraine").exists()


def test_sweep_names_the_assets_it_did_not_sweep(tmp_path, monkeypatch, capsys):
    """The workspace sweep names what it did not sweep (DESIGN 4.2/6): an
    asset file below the project level — a project nested one directory too
    deep — rides the `workspace` section as `not_swept`, so a nested
    project vanishing from the roster never reads as swept green; the key
    is present (empty) when nothing was missed."""
    import json

    from moraine3d.app.cli import main

    ws = _write_workspace(tmp_path)
    nested = ws / "group" / "nested"
    _recipe(nested, GREEN_ASSET, "nested")
    _recipe(nested, GREEN_ASSET, "own", sub=True)

    for cmd in ("stats", "validate"):
        assert main([cmd, str(ws)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert list(out["projects"]) == ["alpha", "beta"]
        assert out["workspace"]["not_swept"] == ["group/nested/nested.star"]


def _mock_status_server(monkeypatch, answer):
    """Stand in for urlopen on the status surface: answer every GET with
    the given JSON object, or refuse the connection when None."""
    import io
    import json
    import urllib.error

    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/api/status") and req.get_method() == "GET"
        if answer is None:
            raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
        return io.BytesIO(json.dumps(answer).encode("utf-8"))

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_validate_and_stats_carry_the_serving_watch(tmp_path, monkeypatch, capsys):
    """`validate` and `stats` carry the running server's file-watch
    liveness under `serve` (DESIGN 6): the one fact agent-side checks
    cannot see is a dead watch behind a green report. Only a server for
    the same root reports its watch; another root's is named as such,
    so it is never taken for this one's; no server or a foreign
    listener leaves the key out — never an error."""
    import json

    from moraine3d.app.cli import main

    proj = _write_project(tmp_path)
    _recipe(proj, GREEN_ASSET, "asset")
    alive = {"root": str(proj.resolve()), "project": proj.name, "watch": "dead",
             "asset_hash": None, "port": 8642}
    _mock_status_server(monkeypatch, alive)
    assert main(["validate", str(proj)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["serve"] == {"port": 8642, "project": proj.name, "watch": "dead"}
    assert main(["validate", str(proj), "--failures-only"]) == 0
    assert json.loads(capsys.readouterr().out)["serve"]["watch"] == "dead"
    assert main(["stats", str(proj)]) == 0
    assert json.loads(capsys.readouterr().out)["serve"]["watch"] == "dead"

    other = dict(alive, root=str((tmp_path / "elsewhere").resolve()))
    _mock_status_server(monkeypatch, other)
    assert main(["validate", str(proj)]) == 0
    assert json.loads(capsys.readouterr().out)["serve"] == {
        "port": 8642, "other_root": other["root"],
    }

    _mock_status_server(monkeypatch, {"unrelated": True})
    assert main(["stats", str(proj)]) == 0
    assert "serve" not in json.loads(capsys.readouterr().out)

    _mock_status_server(monkeypatch, None)
    assert main(["validate", str(proj)]) == 0
    assert "serve" not in json.loads(capsys.readouterr().out)


def test_status_probe_treats_a_non_http_listener_as_nothing_known(monkeypatch, capsys):
    """A listener on --port that does not speak HTTP (DESIGN 6: "a
    foreign listener reads as nothing known, never an error"): urllib
    raises http.client.HTTPException subclasses from the response phase,
    and the probe swallows them like a refused connection."""
    import http.client
    import json

    from moraine3d.app.cli import main

    def bad_line(req, timeout=None):
        raise http.client.BadStatusLine("-ERR unknown command 'GET'")

    _server_answers(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", bad_line)
    proj = _write_project(_tmp_dir())
    _recipe(proj, GREEN_ASSET, "asset")
    assert main(["stats", str(proj)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "serve" not in out and "error" not in out


def _tmp_dir():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())


def test_bad_port_is_refused_before_any_evaluation(tmp_path, monkeypatch, capsys):
    """`--port` on validate/stats is checked before the target is
    evaluated (DESIGN 6): a bad flag costs nothing and leaves no green
    verify event behind an error exit."""
    import json

    from moraine3d.app import cli

    def never(*a, **kw):
        raise AssertionError("evaluated despite a bad --port")

    monkeypatch.setattr(cli, "_validate_report", never)
    monkeypatch.setattr(cli, "_stats_report", never)
    proj = _write_project(tmp_path)
    for cmd in ("validate", "stats"):
        assert cli.main([cmd, str(proj), "--port", "70000"]) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "bad_port"


def test_every_evaluated_report_names_the_tool_version(tmp_path, monkeypatch, capsys):
    """`eval`, `stats`, `validate`, and `query` lead with
    `moraine_version` (DESIGN 6: "every evaluated report names the tool
    that produced it"), the key the export manifest and the screenshot
    report already carry, so a report can never be mistaken for another
    install's answer; a sweep stamps its `workspace` aggregate, and
    `--failures-only` keeps the key."""
    import json

    from moraine3d import dist_version
    from moraine3d.app.cli import main

    proj = _write_project(tmp_path)
    _recipe(proj, GREEN_ASSET, "asset")
    v = dist_version()
    for argv in (
        ["eval", str(proj)],
        ["stats", str(proj)],
        ["validate", str(proj)],
        ["validate", str(proj), "--failures-only"],
        ["query", str(proj), "--select", '{"normal_y": {"gt": 0.9}}'],
    ):
        assert main(argv) == 0, argv
        out = json.loads(capsys.readouterr().out)
        assert next(iter(out)) == "moraine_version", argv
        assert out["moraine_version"] == v, argv
    ws = _write_workspace(tmp_path)
    for cmd in ("stats", "validate"):
        assert main([cmd, str(ws)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["workspace"]["moraine_version"] == v
        assert "moraine_version" not in out


def test_a_number_that_overflows_is_the_builds_refusal(tmp_path, monkeypatch, capsys):
    """A value that is not a finite number cannot cross into the IR
    (DESIGN 4.8): the build refuses it at the recipe line that made
    it, and the command reports that refusal (DESIGN 8.2), never a
    traceback."""
    import json

    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = _write_project(tmp_path)
    (proj / "asset.star").write_text(
        'big = 1e308 * 10\n'
        'mo.output("main", mo.box(id = "bx01", size = [big, 1.0, 1.0]))\n',
        encoding="utf-8",
    )
    assert main(["validate", str(proj)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    err = json.loads(captured.out)["error"]
    assert err["code"] == "bad_value" and "finite" in err["message"]
    assert err["detail"]["site"] == "asset.star:2"


def test_validate_runs_on_a_sub_asset_path(tmp_path, capsys):
    """`moraine validate subassets/kit.star` keeps working for kit
    development (DESIGN 4.6): a sub-asset is a complete, individually
    buildable and evaluable asset, so the CLI validates it by path and
    reports green."""
    import json

    from moraine3d.app.cli import main

    proj = tmp_path / "proj"
    root = json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "rf01", "op": "asset_ref",
                   "params": {"asset": "subassets/kit.mora", "output": "main"}}],
        "outputs": {"main": "rf01"},
        "checks": {"main": {"not_sunk": False}},
    })
    _recipe(proj, root, "proj")
    _recipe(proj, GREEN_ASSET, "kit", sub=True)

    assert main(["validate", str(proj / "subassets" / "kit.star")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and list(out["outputs"]) == ["main"]
    assert main(["validate", str(proj)]) == 0  # and the consumer still validates
    capsys.readouterr()


def test_validate_rejects_a_reference_that_escapes_the_project(tmp_path, capsys):
    """References never leave the workspace: an in-project reference
    path must resolve inside the project directory and `validate`
    rejects escapes (DESIGN 4.6) — exit 1 with the error code in the
    JSON error, for a traversal, an absolute path, and a layout escape."""
    import json

    from moraine3d.app.cli import main

    proj = tmp_path / "proj"
    _recipe(tmp_path, GREEN_ASSET, "outside")
    _recipe(proj, GREEN_ASSET, "kit", sub=True)
    cases = [
        ("../outside.mora", "ref_escape"),
        ((tmp_path / "outside.mora").as_posix(), "ref_escape"),
        ("subassets/../proj.mora", "ref_layout"),
    ]
    for path, code in cases:
        root = json.dumps({
            "build_version": "fixture",
            "nodes": [{"id": "rf01", "op": "asset_ref",
                       "params": {"asset": path, "output": "main"}}],
            "outputs": {"main": "rf01"},
        })
        _recipe(proj, root, "proj")
        assert main(["validate", str(proj)]) == 1, path
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == code, path
        assert err["message"]


def test_serve_names_whose_server_holds_the_port(tmp_path, monkeypatch, capsys):
    """A refused port names its holder (DESIGN 6): this root's server
    is the one to use, another root's knows none of this root's
    projects and is not, and a listener that does not answer the
    status contract leaves the bind's own message."""
    import json

    from moraine3d.app import server as server_mod
    from moraine3d.app.cli import main
    from moraine3d.errors import MoraineError

    proj = _write_project(tmp_path)
    _recipe(proj, GREEN_ASSET, "asset")

    def taken(*args, **kwargs):
        raise MoraineError("port_in_use", "port 8642 is already in use")

    monkeypatch.setattr(server_mod, "run", taken)
    status = {"root": str(proj.resolve()), "project": proj.name, "watch": "alive"}
    _mock_status_server(monkeypatch, status)
    assert main(["serve", str(proj)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "port_in_use" and "this root" in err["message"]

    elsewhere = str((tmp_path / "elsewhere").resolve())
    _mock_status_server(monkeypatch, dict(status, root=elsewhere))
    assert main(["serve", str(proj)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "port_in_use"
    assert err["detail"]["other_root"] == elsewhere and "--port" in err["message"]

    _mock_status_server(monkeypatch, None)
    assert main(["serve", str(proj)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["message"] == "port 8642 is already in use"


def test_a_workspace_root_names_itself_to_the_server_channels(tmp_path, monkeypatch, capsys):
    """DESIGN 6: a channel request names the root its command speaks
    for. From a workspace root with no project named, the command
    speaks for the displayed project of that root's server, so the
    root still goes out and another root's server refuses."""
    import urllib.parse

    from moraine3d.app.cli import _addressed_project, main

    ws = _write_workspace(tmp_path)
    monkeypatch.chdir(ws)
    assert _addressed_project(None) == (None, ws.resolve())
    calls = _mock_pins_server(monkeypatch, {"pins": [], "project": "alpha"})
    assert main(["pins", "list"]) == 0
    assert calls[-1][1].endswith("/api/pins")
    assert urllib.parse.unquote(calls.roots[-1]) == str(ws.resolve())


def test_screenshot_sends_given_knob_values_typed_by_the_declaration(tmp_path, monkeypatch, capsys):
    """DESIGN 6: `screenshot --param ID=VALUE` sends the values as
    `validate --param` reads them, a string knob's as typed, and needs
    a project to read the declarations from."""
    import json

    from support.recipes import write_files

    from moraine3d.app.cli import main

    asset = (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        't = mo.param("tag", "a")\n'
        'body = mo.transform(mo.box(id = "bx", size = [w, 1, 1]), id = "up",'
        ' translate = [0, 0.5, 0])\n'
        'mo.output("main", mo.material(body, id = "mt", name = "m" + t))\n'
    )
    project = write_files(tmp_path / "proj", {"asset.star": asset})
    bodies = _mock_screenshot_server(monkeypatch, b"png", _SHOT_META)
    assert main(["screenshot", str(project), "-o", str(tmp_path / "s.png"),
                 "--param", "w=1.5", "--param", "tag=2"]) == 0
    assert bodies[-1]["params"] == {"w": 1.5, "tag": "2"}
    capsys.readouterr()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert main(["screenshot", "-o", str(tmp_path / "s.png"), "--param", "w=1.5"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "param_needs_project"
