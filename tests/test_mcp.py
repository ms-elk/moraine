"""MCP server (DESIGN 8.4): the read/verify commands as tools over stdio,
each answering with the object its CLI command prints (DESIGN 6)."""

import base64
import io
import json
import os
import sys

import pytest

from conftest import IR_FIXTURES
from moraine3d.app import cli as cli_mod
from moraine3d.app import mcp as mcp_mod
from moraine3d.app import session as session_mod
from support import flatten

EXAMPLES = IR_FIXTURES
SERVER = mcp_mod.Server(cli_mod.invoke)


def _req(id_, method, **params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def _text(result: dict) -> dict:
    """The JSON object a tool relayed as its text content."""
    (block,) = [c for c in result["content"] if c["type"] == "text"]
    return json.loads(block["text"])


def test_initialize_negotiates_version_and_declares_tools_only():
    """DESIGN 8.4: the first server ships the read/verify surface only,
    so the handshake declares the tools capability and nothing else; a
    known client version is echoed, an unknown one answered with ours."""
    reply = SERVER.handle(_req(1, "initialize", protocolVersion="2025-03-26",
                                capabilities={}, clientInfo={"name": "t", "version": "0"}))
    res = reply["result"]
    assert res["protocolVersion"] == "2025-03-26"
    assert set(res["capabilities"]) == {"tools"}
    assert res["serverInfo"]["name"] == "moraine"
    reply = SERVER.handle(_req(2, "initialize", protocolVersion="1999-01-01"))
    assert reply["result"]["protocolVersion"] == mcp_mod.LATEST_PROTOCOL


def test_mcp_help_carries_the_contract_and_the_roster():
    """`moraine mcp --help` is the one in-workspace carrier of the MCP
    contract (DESIGN 8.4): it names every tool the server serves, the
    same-command-same-report equivalence, and the client configuration
    - so an agent asked about MCP answers from the command, not the
    guide."""
    import argparse

    from moraine3d.app import mcp as mcp_mod

    sub = next(a for a in cli_mod.build_parser()._actions
               if isinstance(a, argparse._SubParsersAction))
    text = sub.choices["mcp"].format_help()
    for name in mcp_mod.TOOLS:
        assert name in text
    assert "same name" in text and "same compact report" in text
    assert "claude mcp add moraine -- moraine mcp" in text
    assert '"command": "moraine", "args": ["mcp"]' in text
    assert text.isascii()


def test_tool_roster_is_the_read_verify_surface():
    """DESIGN 8.4: stats, validate, query, export, screenshot, docs - and
    every description and schema stays ASCII (the runtime-string rule)."""
    tools = SERVER.handle(_req(1, "tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == [
        "stats", "validate", "query", "export", "screenshot", "docs"]
    for t in tools:
        assert json.dumps(t).isascii(), t["name"]
        assert t["inputSchema"]["type"] == "object"
        assert t["inputSchema"]["additionalProperties"] is False
    (query,) = [t for t in tools if t["name"] == "query"]
    # A predicate, a tag's name, or both: the command refuses neither.
    assert {"select", "tag"} <= set(query["inputSchema"]["properties"])
    assert "required" not in query["inputSchema"]


def test_notifications_ping_and_unknown_methods():
    """JSON-RPC discipline: a notification gets no reply, ping an empty
    result, an unknown method -32601, malformed input -32600/-32700."""
    assert SERVER.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert SERVER.handle(_req(1, "ping"))["result"] == {}
    assert SERVER.handle(_req(2, "resources/list"))["error"]["code"] == -32601
    assert SERVER.handle([1, 2])["error"]["code"] == -32600
    assert SERVER.handle({"jsonrpc": "2.0", "id": 3})["error"]["code"] == -32600


def test_docs_tool_relays_the_command_object():
    """A tool answers with exactly what its command prints (DESIGN
    8.4): docs without a section lists sections by file as an object,
    and with one answers with the section's own text."""
    res = SERVER.handle(_req(1, "tools/call", name="docs", arguments={}))["result"]
    assert "isError" not in res
    sections = [t for d in _text(res)["docs"] for t in d["sections"]]
    assert any(t.startswith("place") for t in sections)
    res = SERVER.handle(_req(2, "tools/call", name="docs",
                              arguments={"section": "place"}))["result"]
    assert "isError" not in res
    assert res["content"][0]["text"].startswith("### place")


def test_stats_and_query_tools_evaluate_in_process(tmp_path):
    """stats and query run the same handlers as the CLI: the report leads
    with moraine_version (DESIGN 6) and query takes its predicate as an
    object, not a JSON string."""
    project = str(flatten.write_project(EXAMPLES / "crate", tmp_path / "crate").parent)
    res = SERVER.handle(_req(1, "tools/call", name="stats",
                              arguments={"project": project}))["result"]
    report = _text(res)
    assert list(report)[0] == "moraine_version"
    assert report["asset"]["tris"] > 0
    res = SERVER.handle(_req(2, "tools/call", name="query", arguments={
        "project": project, "select": {"normal_y": {"gt": 0.9}}}))["result"]
    outputs = _text(res)["outputs"]
    assert outputs and all("matched" in o for o in outputs.values())


def test_tool_failures_are_the_commands_error_objects():
    """A command failure is the CLI's own error object under isError,
    never a protocol error: the agent reads the same actionable text
    either way (DESIGN 2 P2)."""
    res = SERVER.handle(_req(1, "tools/call", name="query", arguments={
        "project": str(EXAMPLES / "crate"), "select": {"no_such_key": 1}}))["result"]
    assert res["isError"] is True
    assert _text(res)["error"]["code"] == "bad_predicate"
    res = SERVER.handle(_req(2, "tools/call", name="stats",
                              arguments={"project": "/no/such/project"}))["result"]
    assert res["isError"] is True and "error" in _text(res)
    # A well-typed value argparse refuses (a view outside its choices)
    # is a bad_arguments error object, not a process exit.
    res = SERVER.handle(_req(3, "tools/call", name="screenshot",
                              arguments={"view": "nonsense"}))["result"]
    assert res["isError"] is True
    assert _text(res)["error"]["code"] == "bad_arguments"


def test_protocol_level_argument_errors():
    """An unknown tool or an unknown argument is the caller's protocol
    mistake (-32602), distinct from a command that ran and refused,
    which answers as a tool error with the command's own object."""
    assert SERVER.handle(_req(1, "tools/call", name="eval",
                               arguments={}))["error"]["code"] == -32602
    assert SERVER.handle(_req(2, "tools/call", name="docs",
                               arguments={"chapter": "x"}))["error"]["code"] == -32602
    res = SERVER.handle(_req(3, "tools/call", name="query", arguments={}))["result"]
    assert res["isError"] and _text(res)["error"]["code"] == "bad_predicate"


def test_screenshot_tool_returns_the_image(tmp_path, monkeypatch):
    """screenshot writes the PNG like the CLI and adds it as image content
    (DESIGN 8.4): a file-access-less client sees the render, a file
    client keeps the path."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    seen = {}

    def fake_request(port, body, root=None):
        seen.update(body)
        return json.dumps({"meta": {"view": "front", "project": "p"},
                           "png": base64.b64encode(png).decode("ascii")}).encode()

    monkeypatch.setattr(cli_mod, "_screenshot_request", fake_request)
    monkeypatch.chdir(tmp_path)
    res = SERVER.handle(_req(1, "tools/call", name="screenshot", arguments={
        "view": "front", "target": [0, 0.5, 0], "size": "64x64", "out": "s.png"}))["result"]
    assert seen["view"] == "front" and seen["target"] == [0.0, 0.5, 0.0]
    assert (tmp_path / "s.png").read_bytes() == png
    (image,) = [c for c in res["content"] if c["type"] == "image"]
    assert image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]) == png
    assert _text(res)["screenshot"] == "s.png"


def test_each_tool_call_records_a_command_event(tmp_path):
    """DESIGN 6: an MCP tool call appends the same command event a CLI
    invocation does, marked via mcp, so the session log measures the
    authoring loop whichever surface drove it."""
    project = flatten.write_project(EXAMPLES / "crate", tmp_path / "crate").parent
    SERVER.handle(_req(1, "tools/call", name="stats",
                        arguments={"project": str(project)}))
    events = session_mod.read_events(session_mod.log_dir())
    (cmd,) = [e for e in events if e["event"] == "command"]
    assert cmd["cmd"] == "stats" and cmd["ok"] is True and cmd["via"] == "mcp"
    assert cmd["project"] == "crate" and cmd["out_bytes"] > 0


def _serve_lines(lines: list[str]) -> list[dict]:
    """Drive the stdio loop over the given input lines and return the
    parsed replies, asserting the loop ran to EOF."""
    stdin = io.BytesIO(("\n".join(lines) + "\n").encode())
    stdout = io.BytesIO()
    assert SERVER.serve_stdio(stdin=stdin, stdout=stdout) == 0
    return [json.loads(l) for l in stdout.getvalue().decode().splitlines()]


@pytest.mark.parametrize("arguments", [
    {"project": 1},
    {"profile": 5},
    {"failures_only": "yes"},
])
def test_wrong_argument_type_is_a_protocol_error_not_a_crash(arguments):
    """An argument whose JSON type contradicts the tool's inputSchema is
    the caller's protocol mistake (-32602, naming the argument and the
    expected type), and the stdio server outlives it: the next message
    is still answered."""
    replies = _serve_lines([
        json.dumps(_req(1, "tools/call", name="validate", arguments=arguments)),
        json.dumps(_req(2, "ping")),
    ])
    assert [r.get("id") for r in replies] == [1, 2]
    error = replies[0]["error"]
    (key,) = arguments
    expected = mcp_mod.TOOLS["validate"][1]["properties"][key]["type"]
    assert error["code"] == -32602
    assert f"'{key}'" in error["message"] and expected in error["message"]
    assert replies[1]["result"] == {}


def test_internal_error_answers_its_message_and_keeps_the_loop(monkeypatch):
    """No exception ends the stdio loop: one that escapes a tool's own
    layer is a -32603 reply carrying its class and text, on that
    message's id, and the following message is still answered."""
    def broken(a):
        raise RuntimeError("builder exploded")

    desc, schema, _build = mcp_mod.TOOLS["docs"]
    monkeypatch.setitem(mcp_mod.TOOLS, "docs", (desc, schema, broken))
    replies = _serve_lines([
        json.dumps(_req(7, "tools/call", name="docs", arguments={})),
        json.dumps(_req(8, "ping")),
    ])
    assert [r.get("id") for r in replies] == [7, 8]
    error = replies[0]["error"]
    assert error["code"] == -32603
    assert "RuntimeError" in error["message"] and "builder exploded" in error["message"]
    assert replies[1]["result"] == {}


def test_stdio_transport_carries_only_replies(monkeypatch):
    """The transport is stdout, so a handler's stray print must never land
    in it: every stdout line is one JSON-RPC reply, notifications and
    blank lines get none, and an unparsable line is a -32700 reply. The
    stray print and the replies share one process stdout here, so a lost
    diversion shows up as a non-JSON line among the replies."""
    real = cli_mod._cmd_docs

    def noisy(args):
        print("stray line from a handler")
        return real(args)

    monkeypatch.setattr(cli_mod, "_cmd_docs", noisy)
    lines = [
        json.dumps(_req(1, "initialize", protocolVersion="2025-06-18")),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
        "not json",
        json.dumps(_req(2, "tools/call", name="docs", arguments={})),
    ]
    captured = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(
        io.BytesIO(("\n".join(lines) + "\n").encode()), encoding="utf-8"))
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(
        captured, encoding="utf-8", write_through=True))
    assert SERVER.serve_stdio() == 0
    sys.stdout.flush()
    replies = [json.loads(l) for l in captured.getvalue().decode().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700
    assert "docs" in _text(replies[2]["result"])


def test_mcp_subcommand_is_registered_and_keeps_stdout_clean(monkeypatch, capsys):
    """`moraine mcp` runs the stdio server and, like serve, writes no
    command event of its own at exit (its record is one event per tool
    call)."""
    monkeypatch.setattr(mcp_mod.Server, "serve_stdio", lambda self: 0)
    assert cli_mod.main(["mcp"]) == 0
    assert capsys.readouterr().out == ""
    assert not [e for e in session_mod.read_events(session_mod.log_dir())
                if e["event"] == "command"]


def test_official_client_drives_the_server(tmp_path):
    """Conformance against the reference client (the SDK is a dev-only
    dependency, DESIGN 8.4): a real subprocess handshake, tool listing
    and a docs call through the official stdio client."""
    mcp = pytest.importorskip("mcp")
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def drive():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "moraine3d", "mcp"],
            env=dict(os.environ), cwd=str(tmp_path))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                init = await s.initialize()
                tools = await s.list_tools()
                res = await s.call_tool("docs", {"section": "place"})
                return init, tools, res

    init, tools, res = anyio.run(drive)
    assert init.server_info.name == "moraine"
    assert {t.name for t in tools.tools} == set(mcp_mod.TOOLS)
    assert res.is_error is not True
    assert res.content[0].text.startswith("### place")


def test_export_tool_mirrors_the_recompute_flag():
    """The MCP export tool carries the CLI's determinism check (DESIGN
    8.4: the tools are the commands, same arguments) — `recompute` maps
    to `--recompute`, absent by default."""
    from moraine3d.app.mcp import _tool_export

    assert "--recompute" not in _tool_export({"project": "p"})
    assert _tool_export({"project": "p", "recompute": True})[-1] == "--recompute"
