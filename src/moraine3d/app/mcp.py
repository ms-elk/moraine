"""MCP server (DESIGN 8.4): the read/verify commands as MCP tools.

``moraine mcp`` speaks the Model Context Protocol over stdio - newline-
delimited JSON-RPC 2.0, the tools capability only. Each tool is one CLI
command run in-process through the CLI's own parser and handler (the
runner the CLI hands ``Server``), so a tool answers with exactly the
JSON object the command prints and fails with exactly its error object
(DESIGN 6); nothing is re-modeled here. The protocol subset is
implemented directly rather than through an SDK: a tools-only stdio
server is ``initialize``, ``tools/list``, ``tools/call`` and ``ping``,
and the surface below is kept to what those need.

The process's stdout is the transport: the runner keeps every command's
output off it, and this module writes replies and nothing else.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Callable

from .. import dist_version
from . import camera as camera_mod
from .server import DEFAULT_PORT

Runner = Callable[..., tuple[int, dict]]

# Versions of the protocol this subset is known against: the client's
# is echoed when it is one of these, else the newest one here (the
# negotiation rule of the spec's initialize exchange).
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
LATEST_PROTOCOL = PROTOCOL_VERSIONS[-1]
SERVER_NAME = "moraine"

_PROJECT_HELP = ("project directory holding one root asset (<name>.star), "
                 "or an asset file path; default: the server process's "
                 "working directory")
_SWEEP_HELP = _PROJECT_HELP + "; a workspace root sweeps every project into one report"
_PORT_HELP = ("port of a running 'moraine serve' (default %d); the report "
              "carries its file-watch liveness under 'serve' when one answers"
              % DEFAULT_PORT)
_SERVER_PORT_HELP = ("port of the running 'moraine serve' process (default %d); "
                     "must match its --port" % DEFAULT_PORT)


def _str(desc: str, **extra) -> dict:
    return {"type": "string", "description": desc, **extra}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    schema: dict = {"type": "object", "properties": props,
                    "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


# Tool roster: name -> (description, input schema, argv builder). Each
# builder turns the JSON arguments into the command's argv so the CLI's
# own parser and handler validate and run it - one contract, one place.
def _tool_stats(a: dict) -> list[str]:
    return ["stats", a.get("project", ".")] + _port(a)


def _tool_validate(a: dict) -> list[str]:
    argv = ["validate", a.get("project", ".")]
    if a.get("profile") is not None:
        argv += ["--profile", a["profile"]]
    if a.get("failures_only"):
        argv.append("--failures-only")
    for pid, value in (a.get("params") or {}).items():
        argv += ["--param", f"{pid}={value if isinstance(value, str) else json.dumps(value)}"]
    return argv + _port(a)


def _tool_query(a: dict) -> list[str]:
    argv = ["query", a.get("project", ".")]
    if a.get("select") is not None:
        argv += ["--select", json.dumps(a["select"])]
    if a.get("tag") is not None:
        argv += ["--tag", a["tag"]]
    if a.get("instance") is not None:
        argv += ["--instance", str(a["instance"])]
    if a.get("output") is not None:
        argv += ["--output", a["output"]]
    return argv


def _tool_export(a: dict) -> list[str]:
    argv = ["export", a.get("project", ".")]
    if a.get("profile") is not None:
        argv += ["--profile", a["profile"]]
    if a.get("out") is not None:
        argv += ["-o", a["out"]]
    if a.get("bake_instances"):
        argv.append("--bake-instances")
    if a.get("recompute"):
        argv.append("--recompute")
    return argv


def _tool_screenshot(a: dict) -> list[str]:
    argv = ["screenshot"]
    if a.get("project") is not None:
        argv.append(a["project"])
    for key in ("view", "azimuth", "elevation", "distance", "size"):
        if a.get(key) is not None:
            argv += ["--" + key, str(a[key])]
    if a.get("target") is not None:
        argv += ["--target", ",".join(str(v) for v in a["target"])]
    if a.get("out") is not None:
        argv += ["-o", a["out"]]
    for pid, value in (a.get("params") or {}).items():
        argv += ["--param", f"{pid}={value if isinstance(value, str) else json.dumps(value)}"]
    return argv + _port(a)


def _tool_docs(a: dict) -> list[str]:
    return ["docs"] + ([a["section"]] if a.get("section") is not None else [])


def _port(a: dict) -> list[str]:
    return ["--port", str(a["port"])] if a.get("port") is not None else []


TOOLS: dict[str, tuple[str, dict, object]] = {
    "stats": (
        "Part/tri/vert counts, bounds and materials of a project's evaluated "
        "outputs, with the cache profile - the compact JSON 'moraine stats' "
        "prints. A workspace root sweeps every project into one report.",
        _schema({
            "project": _str(_SWEEP_HELP),
            "port": {"type": "integer", "description": _PORT_HELP},
        }),
        _tool_stats,
    ),
    "validate": (
        "Deterministic geometry checks, asset asserts and budgets over a "
        "project's outputs - the compact JSON 'moraine validate' prints; "
        "'ok' is the verdict. A workspace root sweeps every project.",
        _schema({
            "project": _str(_SWEEP_HELP),
            "profile": _str("also run this export profile's checks (e.g. 'gltf')"),
            "failures_only": {"type": "boolean", "description":
                              "collapse healthy outputs to a count; detail only on failure"},
            "params": {"type": "object", "description":
                       "judge one project under these values of its declared "
                       "knobs, over an accepted one, writing no file"},
            "port": {"type": "integer", "description": _PORT_HELP},
        }),
        _tool_validate,
    ),
    "query": (
        "Read-only face-predicate query over evaluated outputs (the tag op's "
        "vocabulary): match count, matched area, bounds and centroid per "
        "output - token cost scales with the answer, never the asset. Give "
        "'select', 'tag', or both.",
        _schema({
            "project": _str(_PROJECT_HELP),
            "select": {"type": "object", "description":
                       'face predicate, e.g. {"normal_y": {"gt": 0.9}}'},
            "tag": _str("the faces a tag caught, by its name"),
            "instance": {"type": "integer", "description":
                         "read this copy of an instanced output in world space"},
            "output": _str("restrict to one named output"),
        }),
        _tool_query,
    ),
    "export": (
        "Write glTF (GLB) + manifest for an export profile; requires a green "
        "validate (waivers included). Returns the export report with the "
        "written paths.",
        _schema({
            "project": _str(_PROJECT_HELP),
            "profile": _str("export profile (default: gltf)"),
            "out": _str("output directory; a relative path resolves against "
                        "the project (default: exports)"),
            "bake_instances": {"type": "boolean", "description":
                               "project instanced outputs to plain meshes at the "
                               "export edge; default keeps instancing"},
            "recompute": {"type": "boolean", "description":
                          "the determinism check: recompute every node ignoring "
                          "cached results and compare with the records on disk "
                          "(never cleared or overwritten); a mismatch reports "
                          "ok false"},
        }),
        _tool_export,
    ),
    "screenshot": (
        "Render a project's current file state through the running 'moraine "
        "serve' (opens a viewer tab if none is connected). Writes the PNG to "
        "'out' and returns its provenance record plus the image.",
        _schema({
            "project": _str("project to shoot (default: the working directory's; "
                            "outside any project, the one the viewer displays)"),
            "view": {"type": "string", "enum": list(camera_mod.VIEWS),
                     "description": "named view preset (default: iso); "
                                    "exclusive with azimuth/elevation"},
            "azimuth": {"type": "number", "description":
                        "free camera: degrees, 0 looks from +Z (front), 90 from +X (right)"},
            "elevation": {"type": "number", "description":
                          "free camera: degrees above horizontal, -90..90 (90 = top)"},
            "distance": {"type": "number", "description":
                         "camera distance in meters (default: auto-fit)"},
            "target": {"type": "array", "items": {"type": "number"},
                       "minItems": 3, "maxItems": 3,
                       "description": "look-at point [x, y, z] in meters "
                                      "(default: asset bounds center)"},
            "size": _str("pixels as WIDTHxHEIGHT (default 512x512)"),
            "out": _str("PNG path, relative to the working directory (default shot.png)"),
            "params": {"type": "object", "description":
                       "shoot the project under these values of its declared "
                       "knobs, writing no file and moving no viewer"},
            "port": {"type": "integer", "description": _SERVER_PORT_HELP},
        }),
        _tool_screenshot,
    ),
    "docs": (
        "The shipped reference (the recipe, operator contracts, checks, "
        "assertions by kind, CLI, journal): omit 'section' to list "
        "sections by file, name one to read that section alone, as text, "
        "with the sections under it and the examples it cites named; "
        "'api' lists every builtin with its arguments; 'examples' "
        "lists the shipped example assets, 'examples/<name>' prints one.",
        _schema({"section": _str(
            "section name, e.g. 'place', 'mated', or 'api'; or "
            "'examples' / 'examples/<name>'"
        )}),
        _tool_docs,
    ),
}


def tool_list() -> list[dict]:
    return [
        {"name": name, "description": desc, "inputSchema": schema}
        for name, (desc, schema, _build) in TOOLS.items()
    ]


def call_tool(name: str, arguments: dict, invoke: Runner) -> dict:
    """The ``tools/call`` result for one tool: the command's JSON object
    as text content (a screenshot adds the PNG as image content), with
    ``isError`` set on a non-zero exit."""
    _desc, schema, build = TOOLS[name]
    unknown = sorted(set(arguments) - set(schema["properties"]))
    if unknown:
        raise _InvalidParams(f"unknown argument(s) for '{name}': {', '.join(unknown)}")
    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        raise _InvalidParams(f"missing argument(s) for '{name}': {', '.join(missing)}")
    for key, value in arguments.items():
        _check_type(name, key, value, schema["properties"][key])
    try:
        argv = build(arguments)
    except (TypeError, ValueError) as e:
        raise _InvalidParams(f"bad argument for '{name}': {e}")
    code, obj = invoke(argv, via="mcp")
    if name == "docs" and code == 0 and set(obj) == {"text"}:
        text = obj["text"]  # a reference page is its own text
    else:
        text = json.dumps(obj, separators=(",", ":"))
    content: list[dict] = [{"type": "text", "text": text}]
    if code == 0 and name == "screenshot":
        try:
            png = Path(obj["screenshot"]).read_bytes()
        except (KeyError, OSError):
            png = None
        if png is not None:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(png).decode("ascii")})
    result: dict = {"content": content}
    if code != 0:
        result["isError"] = True
    return result


class _InvalidParams(Exception):
    pass


# JSON schema type name -> the Python types a decoded value may carry.
# bool is a subclass of int, so it is listed first (name lookup) and
# excluded from the numeric kinds explicitly: JSON true is not an integer.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "boolean": (bool,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "array": (list,),
    "object": (dict,),
}


def _check_type(tool: str, key: str, value, prop: dict) -> None:
    """Refuse an argument whose JSON type is not the one the tool's
    inputSchema declares, before it can reach argv, where an int in a
    string's place is a crash rather than a usage error."""
    expected = prop["type"]
    ok = isinstance(value, _JSON_TYPES[expected])
    if ok and expected in ("integer", "number") and isinstance(value, bool):
        ok = False
    if ok and expected == "array" and "items" in prop:
        item = prop["items"]["type"]
        ok = all(isinstance(v, _JSON_TYPES[item]) and not isinstance(v, bool)
                 for v in value)
        expected = f"array of {item}"
    if not ok:
        raise _InvalidParams(
            f"bad argument for '{tool}': '{key}' must be a JSON {expected}, "
            f"got {_json_type_name(value)}")


def _json_type_name(value) -> str:
    if value is None:
        return "null"
    for name, kinds in _JSON_TYPES.items():
        if isinstance(value, kinds):
            return name
    return type(value).__name__


def _initialize(params: dict) -> dict:
    requested = params.get("protocolVersion")
    version = requested if requested in PROTOCOL_VERSIONS else LATEST_PROTOCOL
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": dist_version()},
        "instructions": (
            "Moraine's read/verify surface: every tool is one 'moraine' CLI "
            "command and answers with the compact JSON that command prints "
            "(a 'docs' section answers with its own text). "
            "Assets are edited as recipe files (*.star), which every "
            "command builds first; the shipped authoring guide is "
            ".moraine/guide.md in the workspace."
        ),
    }


def _error(id_, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


class Server:
    """One MCP server over a command runner (``cli.invoke``)."""

    def __init__(self, invoke: Runner):
        self._invoke = invoke

    def handle(self, message) -> dict | None:
        return handle(message, self._invoke)

    def serve_stdio(self, stdin=None, stdout=None) -> int:
        return serve_stdio(self._invoke, stdin=stdin, stdout=stdout)


def handle(message, invoke: Runner) -> dict | None:
    """One JSON-RPC message to its response, or None for a notification
    (and for anything unanswerable: a notification-shaped bad message)."""
    if not isinstance(message, dict):
        return _error(None, -32600, "invalid request: expected an object")
    id_ = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if not isinstance(method, str):
        return _error(id_, -32600, "invalid request: missing method") if "id" in message else None
    if "id" not in message:
        # Notifications (initialized, cancelled, progress): nothing to say.
        return None
    if not isinstance(params, dict):
        return _error(id_, -32602, "params must be an object")
    try:
        if method == "initialize":
            result = _initialize(params)
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_list()}
        elif method == "tools/call":
            name = params.get("name")
            if name not in TOOLS:
                raise _InvalidParams(
                    f"unknown tool '{name}'; tools: {', '.join(TOOLS)}")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise _InvalidParams("arguments must be an object")
            result = call_tool(name, arguments, invoke)
        else:
            return _error(id_, -32601, f"method not found: {method}")
    except _InvalidParams as e:
        return _error(id_, -32602, str(e))
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def serve_stdio(invoke: Runner, stdin=None, stdout=None) -> int:
    """Read newline-delimited JSON-RPC from stdin until EOF, answering on
    stdout. Binary streams: the transport is UTF-8 regardless of the
    console codepage, and a wide character in a report must never
    crash the server."""
    inp = stdin if stdin is not None else sys.stdin.buffer
    out = stdout if stdout is not None else sys.stdout.buffer
    for raw in inp:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            reply = _error(None, -32700, "parse error: not a JSON line")
        else:
            try:
                reply = handle(message, invoke)
            except Exception as e:
                # An internal error answers its own message; the
                # transport stays up for the next one.
                id_ = message.get("id") if isinstance(message, dict) else None
                reply = _error(id_, -32603,
                               f"internal error: {type(e).__name__}: {e}")
        if reply is None:
            continue
        out.write(json.dumps(reply, separators=(",", ":")).encode("utf-8") + b"\n")
        out.flush()
    return 0
