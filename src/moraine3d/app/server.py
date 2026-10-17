"""HTTP + WebSocket server: viewer hosting, hot reload, screenshots,
parameter preview.

Implements the Viewer Protocol (DESIGN 7.2):
- ``asset_updated {project, outputs, glb_urls, image_urls, stats, asset}``
  runtime -> client (``asset`` is the shared rollup ``stats`` prints)
- ``param_schema {params, overridden}``               runtime -> client
- ``workspace {projects, changed}``                   runtime -> client
  (workspace mode only, replayed first on connect; the roster is
  re-read from disk by every message that resolves a name and by the
  watch — absence means no project picker)
- ``set_param {param_id, value}`` / ``set_param {params}``
  client -> runtime
- ``preview_dropped {trigger}``                       runtime -> client
  (this connection's overlay was discarded; ``trigger`` is the reload's
  ``watch | set_asset | cache_lost`` or ``revert``)
- ``accept {}`` / ``accepted {params}``, ``revert {}``
  client -> runtime / runtime -> client (promote or discard this
  connection's overlay)
- ``set_asset {project}``                             client -> runtime
  (name-based: the project must be a member of the workspace roster,
  re-read from disk — no client-supplied paths, DESIGN 7.2)
- ``screenshot_request {request_id, protocol, glb_urls, camera, size}``
  runtime -> client (dedicated file-state render session: a fresh
  override-free evaluation and a server-resolved camera pose)
- ``screenshot_result {request_id, protocol, png | error}``
  client -> runtime (results missing the protocol echo come from
  pre-upgrade tabs and are ignored, DESIGN 7.2)
- ``watch_dead {error}``                              runtime -> client
  (sticky, replayed on connect: the file watch died, hot reload is off
  until it re-arms)
- ``watch_alive {}``                                  runtime -> client
  (the watch re-armed: clears ``watch_dead``; the reload that follows
  catches up the edits that landed while it was dead)
- ``listen_dead {error}``                             runtime -> client
  (the listener stopped accepting while every connected socket kept
  streaming — a half-dead server: new tabs and commands cannot
  connect until it re-arms; only the connected tabs can be told)
- ``listen_alive {}``                                 runtime -> client
  (the listener re-armed: clears ``listen_dead``)
- ``asset_error {project, error}``                    runtime -> client
  (the active project rides along so clients track the server across
  a failed switch, DESIGN 7.2; the last one is replayed on connect
  after the state payloads, so a tab opened after a failed evaluation
  learns why the display is not the file)
- ``select {action: "pin" | "replace" | "unpin" | "clear", ...}``
  client -> runtime (3D deixis, DESIGN 7.2: ``pin``/``replace`` carry
  ``output``, ``face``, optional ``instance`` and the clicked world
  ``position`` — ``replace`` makes the new pin the sole selection in
  one atomic mutation; ``unpin`` carries ``id``. Pins resolve against
  file state; failures answer the pointing client with
  ``asset_error``)
- ``pins {project, asset_hash, pins}``                runtime -> client
  (full pin state, idempotent: after every pin mutation and re-root,
  and replayed on connect when pins exist)
- ``timeline {root, gap_minutes, session, stale?}``   runtime -> client
  (the newest session's summary under the watch root, derived from
  the session log by the same summarizer ``moraine sessions`` uses,
  DESIGN 7.2; full state, idempotent, replayed on connect, re-sent
  when the log directory changes; absence means logging is off.
  ``stale: true`` is the log-watch death flag, sticky like
  ``watch_dead``)
Unknown message types are ignored.

``set_param`` is an ephemeral, client-local preview overlay (DESIGN
7.2): each connection carries its own overlay, the re-evaluated preview
is sent to that connection only, and the shared payloads other clients
see (and every snapshot replay) hold file state by construction. The
overlay never writes a file, and any file-triggered reload drops
every client's overlay — the file stays the sole source of truth. A
failing override is reverted and reported through ``asset_error`` to
the overlaying client; the viewer keeps the last good state.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import difflib
import errno
import json
import math
import re
import sys
import threading
import signal
import socket
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path

from aiohttp import WSCloseCode, WSMsgType, web
from watchfiles import Change, awatch

from .. import dist_version
from ..cache import store as cache_mod
from ..core import accepted as accepted_mod
from ..core import params as params_mod
from ..core import predicates
from ..core import asset as asset_mod
from ..core import sources as sources_mod
from ..core import workspace as workspace_mod
from ..core.paths import PREVIEW_DIR_NAME, cache_dir, is_redirect
from ..core.sites import where
from ..engine import diff as diff_mod
from ..engine import stats as stats_mod
from ..engine import tiers as tiers_mod
from ..engine import validation
from ..engine.evaluator import Evaluation
from ..errors import MoraineError
from ..recipe import target as target_mod
from ..ops import common as ops_common
from . import camera as camera_mod
from . import session

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
# The wait covers the client fetching fresh file-state GLBs before it
# renders (the evaluation finishes before the request is broadcast, so
# /cache/ serving proceeds concurrently with this await).
SCREENSHOT_TIMEOUT_S = 30.0
DEFAULT_PORT = 8642
# The request header a CLI channel names its root in (percent-encoded),
# so a server of another root refuses the request.
ROOT_HEADER = "X-Moraine-Root"
# How long `serve` watches for a client before deciding nobody is
# watching (DESIGN 7.1): sampled until it expires, so a tab left open
# across a restart re-attaches first and no second tab is opened over
# it. Longer than the tab's 1 s reconnect retry with slack for a
# browser that throttles the retry of a hidden tab.
VIEWER_OPEN_GRACE_S = 5.0
_VIEWER_OPEN_POLL_S = 0.1
# The launcher gets this long to say whether it started a browser.
BROWSER_OPEN_TIMEOUT_S = 5.0


def open_browser(url: str) -> bool:
    """Open the human's browser on ``url`` without ever blocking the
    caller for longer than ``BROWSER_OPEN_TIMEOUT_S``.

    ``webbrowser.open`` is not bounded: its generic-launcher path ends
    in ``p.wait()``, so a ``$BROWSER`` naming a command that stays in
    the foreground blocks until that command exits - unbounded, and on
    the server side it would also keep an executor thread alive past
    shutdown. The launcher runs on a daemon thread; a launcher still
    running when the budget expires counts as opened, because it is -
    the browser simply has not exited.
    """
    answer: list[bool] = []

    def launch() -> None:
        try:
            answer.append(bool(webbrowser.open(url)))
        except Exception:
            answer.append(False)

    thread = threading.Thread(target=launch, daemon=True, name="moraine-browser")
    thread.start()
    thread.join(BROWSER_OPEN_TIMEOUT_S)
    return answer[0] if answer else True
# Recheck interval while the log directory is missing at serve start
# (the log watch waits for it rather than giving up).
_LOG_DIR_RETRY_S = 30.0
# Re-arm schedule shared by the file watches and the listener (DESIGN
# 7.2): first retry after this delay, doubling per failed attempt up to
# the cap. An attempt that survives the birth window is announced armed
# (the OS watcher is created and the listener bound and probed on the
# first iteration, so most failures are immediate), but the backoff
# resets only after an attempt has held for the stable period — one
# that dies later (resources exhausted while registering a new
# directory, the root breaking under it, the accept loop dying under a
# new tab) is the same outage, and its retries keep doubling instead of
# flapping every few seconds.
_REARM_S = 1.0
_REARM_CAP_S = 60.0
_BIRTH_S = 2.0
_STABLE_S = 60.0
# Listener liveness (DESIGN 7.2 ``listen_dead``): the platform's accept
# loop closes the listening socket on an accept error while every
# accepted socket keeps streaming, so nothing the server reports through
# those sockets can notice. Two checks an interval apart: the listening
# socket itself (closed under the site is the platform's exact
# signature) and a loopback connect made off the event loop, so a long
# evaluation on the loop cannot run its budget out — the kernel
# completes the handshake for a listening socket regardless.
_LISTEN_PROBE_S = 5.0
_LISTEN_PROBE_TIMEOUT_S = 2.0


def _claim_port(port: int) -> socket.socket:
    """The loopback port, bound and not yet listening (DESIGN 6): taken
    before the first evaluation, so a held port is refused with nothing
    evaluated and nothing reported. Until the site listens on it a
    connection is refused, as on a port nobody holds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform != "win32":
            # The loop's own default for a listener it binds: a port in
            # TIME_WAIT from the previous server is not a held port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock


def _site_listening(site: web.BaseSite) -> bool:
    """Whether the site's listening socket is still open — the platform
    closes it under the site when its accept loop dies (DESIGN 7.2
    ``listen_dead``), the one failure this catches without a handshake.
    Reads the site's server directly: aiohttp exposes no accessor."""
    server = getattr(site, "_server", None)
    socks = server.sockets if server is not None else ()
    return bool(socks) and all(s.fileno() != -1 for s in socks)

# Version stamp on screenshot_request, echoed in screenshot_result: a tab
# still running a pre-upgrade client would answer with its own shared
# asset — the echo lets the server refuse that instead of returning a
# stale shot with clean-looking provenance.
_SCREENSHOT_PROTOCOL = 1

# Cache entries served to the viewer: "<24-hex key>-g<ver>-<output>.glb"
# (mesh outputs) and "<key>-i<ver>-<output>.png" (image-output previews).
# Composed from cache/store.py's grammar fragments — the single regex source
# for the evaluator's cache naming.
_CACHE_FILE_RE = re.compile(
    rf"^{cache_mod.KEY_FRAGMENT}-(?:{cache_mod.GLB_FRAGMENT}|{cache_mod.PNG_FRAGMENT})$"
)

_SCREENSHOT_MAX_DIM = 4096

# Deixis pin cap (DESIGN 7.2): pins are a gesture stream, not storage —
# the oldest is evicted FIFO rather than erroring at the cap.
_PINS_MAX = 16

# The server is a local single-user contract (DESIGN 7.2): it binds to
# 127.0.0.1, and a browser page is admitted at the WebSocket door and
# the API endpoints only from this server's own origin — scheme http, a
# loopback host, and the bound port; a page on another loopback port is
# another program's page and is refused like any foreign host.
# Non-browser clients (CLI, tools) send no Origin.
_ALLOWED_ORIGIN_HOSTS = ("127.0.0.1", "localhost", "::1")


def _origin_allowed(req: web.Request, port: int | None) -> bool:
    """A missing Origin is allowed (the CLI's urllib sends none); a
    present one must be this server's own origin: ``http``, a loopback
    host, and ``port`` (an Origin without a port means the scheme's
    default, 80). ``port`` None — the server not yet bound — admits no
    browser page."""
    origin = req.headers.get("Origin")
    if origin is None:
        return True
    try:
        parts = urllib.parse.urlsplit(origin)
        origin_port = parts.port
    except ValueError:
        return False
    if parts.scheme != "http" or parts.hostname not in _ALLOWED_ORIGIN_HOSTS:
        return False
    return port is not None and (80 if origin_port is None else origin_port) == port


def _apart(work, *args) -> asyncio.Future:
    """Run ``work`` in a thread of its own and answer on the loop. The
    loop's exit waits for the threads of its executor, and the check of
    an accepted state may run for a minute, so this thread is one the
    exit leaves behind."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def settle(result, error) -> None:
        if done.done():
            return
        if error is not None:
            done.set_exception(error)
        else:
            done.set_result(result)

    def run() -> None:
        try:
            result, error = work(*args), None
        except Exception as e:
            result, error = None, e
        try:
            loop.call_soon_threadsafe(settle, result, error)
        except RuntimeError:
            pass  # the loop closed while the work ran

    threading.Thread(target=run, daemon=True, name="moraine-accept-check").start()
    return done


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def _prune_stale_overrides(sc: asset_mod.Asset, overrides: dict[str, object]) -> None:
    """Drop override ids the file no longer declares (edited away
    between watch events), mutating the caller's per-client dict: the
    file wins, and a stale id must not poison later set_params — or
    ride into an accept — with someone else's error. One home for the
    rule on purpose: the preview, reload, and accept paths must prune
    identically or the same overlay evaluates on one channel and errors
    on another."""
    declared = {d["id"] for d in sc.params or []}
    for k in [k for k in overrides if k not in declared]:
        del overrides[k]


class ClientState:
    """Per-connection viewer state (DESIGN 7.2): the client-local
    ``set_param`` preview overlay (param_id -> value). Never written to
    the file; dropped for every client by any file-state reload, and
    dies with the connection. Camera state stays entirely client-side
    and never appears here."""

    __slots__ = ("overrides",)

    def __init__(self) -> None:
        self.overrides: dict[str, object] = {}


class MoraineServer:
    def __init__(
        self,
        asset_path: Path,
        workspace: workspace_mod.Workspace | None = None,
        project: str | None = None,
    ):
        # None = single-asset mode: no picker, set_asset has no valid
        # targets. In workspace mode the roster is re-read from disk
        # (_refresh_roster), so this attribute is replaced, never mutated.
        self.workspace = workspace
        # Roster re-scan failures already logged from the watch path,
        # by error code: the watch fires per batch, the log line once.
        self._roster_errors: set[str] = set()
        # Fixed for the server lifetime: awatch's path set is fixed at
        # call time, so we watch the widest root we may ever re-root
        # into and narrow per-event via `watched`.
        # Single-asset serve of a workspace member widens to the enclosing
        # workspace root: sibling projects' asset files must sit inside
        # the awatch tree for cross-project refs (DESIGN 4.6) to
        # hot-reload; still single-asset (no roster, no picker). The same
        # root frames the asset content hash and the session log.
        self.watch_root = (
            workspace.root.resolve() if workspace
            else workspace_mod.enclosing_root(asset_mod.project_root(Path(asset_path)))
        )
        # Set by run() before the app starts; named in the session log.
        self.port: int | None = None
        # Open the human's browser on the viewer once serving (DESIGN
        # 7.1); ``run`` sets it from the CLI flag. The grace is an
        # attribute so tests need not wait it out.
        self.open_browser = True
        self.viewer_open_grace_s = VIEWER_OPEN_GRACE_S
        # Content hash of the last file-state evaluation (success or
        # failure) and the asset bytes it read, for the reload event's
        # changed-line count (DESIGN 6 session log).
        self.asset_hash: str | None = None
        self._snapshot: dict[Path, bytes] = {}
        # Roster projects whose asset files changed on disk while another
        # project was displayed (DESIGN 7.2 ``workspace.changed``): with
        # several sessions authoring, the picker's mark is how the human
        # knows where to look next. Cleared for a project when it is
        # displayed again.
        self.changed: set[str] = set()
        # Per-server state that survives a re-root; the per-asset half
        # lives in _bind_asset.
        self.clients: dict[web.WebSocketResponse, ClientState] = {}
        self.pending_shots: dict[str, asyncio.Future] = {}
        # Serializes the file-state evaluations that run off the event
        # loop (screenshot, pin add) so at most one worker-thread
        # evaluation runs at a time; the loop-side evaluations
        # (set_param, reload, accept) stay synchronous on the loop —
        # overlay atomicity, see _evaluate_results — and never take it.
        self._offloop_eval_gate = asyncio.Lock()
        # One check of an accepted state at a time (DESIGN 7.2). Its
        # own gate, so the asserts of a large asset never keep a
        # screenshot or a pin waiting.
        self._accept_gate = asyncio.Lock()
        # Pin ids stay unique across re-roots (the store itself is
        # per-asset state, seated in _bind_asset): an id seen once never
        # names a different pin later in the session.
        self._pin_seq = 0
        # Request ids that received a reply without the protocol echo (a
        # pre-upgrade tab): remembered so the timeout path can name the
        # real problem instead of "viewer did not respond".
        self.stale_shots: set[str] = set()
        self.asset_payload: dict | None = None
        self.param_payload: dict | None = None
        # Material names un-renamed references carry into the root, per
        # the last file-state evaluation (DESIGN 7.2): the tier fact the
        # graph walk cannot see, fed to every classification — the
        # overlay drives root knobs, never what a referenced file binds.
        self._reference_material_names: frozenset[str] = frozenset()
        # The re-materializing reload in flight after a served record
        # vanished from the cache (``_heal_lost_record``): one per
        # outage, however many 404s a page load produces.
        self._heal_task: asyncio.Task | None = None
        # The last file-state asset_error broadcast (with the project it
        # named), cleared by the next successful evaluation; replayed on
        # connect so a tab opened after a failure sees why the display
        # is stale.
        self.last_error: dict | None = None
        # Bumped on every successful file-state evaluation; send loops
        # abort when they observe a newer generation (see
        # _broadcast_state). Per-client preview sends never bump it.
        self._payload_gen = 0
        # Set when the watch task dies (awatch itself failing: root
        # deleted, OS watch resources exhausted): hot reload is off and
        # the viewer display may be stale — every evidence channel must
        # say so instead of serving stale pixels silently (DESIGN 7.2
        # ``watch_dead``). Sticky until the watch re-arms (``_watch``).
        self.watch_dead: dict | None = None
        # Set when the loopback probe finds the port closed while this
        # process still runs (DESIGN 7.2 ``listen_dead``): the accept
        # loop died and every connected tab still streams. Sticky per
        # outage until the listener re-arms; the re-arm count is the
        # one trace a healed half-death leaves in a report.
        self.listen_dead: dict | None = None
        self.listen_rearmed = 0
        # The site this server bound and holds (``_run_listen``); None
        # between a death and the next successful bind. Ownership is the
        # identity check: a probe passing against a port this server did
        # not bind is a foreign listener, never a re-arm.
        self._site: web.BaseSite | None = None
        # The newest session's summary under the watch root (DESIGN 7.2
        # ``timeline``), rebuilt when the log directory changes and
        # replayed on connect. Root-scoped rather than per-asset state:
        # the session log is framed by the watch root, so the payload
        # survives a re-root. None while logging is off or nothing is
        # readable — absence means no panel.
        self.timeline_payload: dict | None = None
        # Debounce for the post-growth cache eviction check (DESIGN 4.5):
        # at most every 10 minutes, None = allowed immediately.
        self._last_evict: float | None = None
        self._bind_asset(
            Path(asset_path),
            project or asset_mod.project_root(Path(asset_path)).name,
        )

    def _bind_asset(self, asset_path: Path, project: str) -> None:
        """Swap the per-asset state as one unit (the single-active-asset
        invariant): everything bound here moves together on a re-root.

        ``project`` is the roster/directory name, passed in rather than
        derived from the resolved path: ``resolve()`` escapes a
        junctioned project directory, which would desync the identity
        on ``asset_updated`` from the roster. (The watch, keyed on
        resolved paths under ``watch_root``, still cannot see through a
        junction that leaves the root - such a project loses hot
        reload.)"""
        self._settle_unchecked()  # under the project they were written into
        self.asset_path = asset_path.resolve()
        self.project_dir = asset_mod.project_root(self.asset_path)
        # Active project identity on asset_updated and asset_error
        # (DESIGN 7.2, the select seam).
        self.project = project
        # Every recipe file the current build read (the assets of the
        # reference closure, their requirements files, the lib/ files
        # they load, DESIGN 4.6); editing any of them hot-reloads.
        self.watched: set[Path] = {self.asset_path}
        # Where a reload's own pass leaves what its build found, by
        # resolved path; None outside a reload. The loop alone writes
        # it, so a pass in a worker thread cannot.
        self._reload_found: dict | None = None
        # The IRs of the last file-state build a reload made, by the
        # path each is written at, and the target's: what the next
        # reload's graph diff is taken against (DESIGN 6). A failed
        # reload leaves them, so the diff after a fix is against the
        # last good state. ``_reload_built`` is the pass in flight.
        self._diff_base: tuple[Path, dict] | None = None
        self._reload_built: tuple[Path, dict] | None = None
        # The last file-state build that opened, which an overlay of
        # `direct` knobs is patched onto (DESIGN 7.2). None after a
        # build that failed, so a preview then fails as the file does.
        self._file_opened: target_mod.Opened | None = None
        # Deixis pins (DESIGN 7.2): pointing gestures into the active
        # asset. Per-asset state — a re-root drops them with the asset
        # they pointed into — surviving file reloads (each pin records
        # the revision it was resolved against, so staleness is
        # derivable) and dying with the server.
        self.pins: list[dict] = []
        # The accepts written and not yet checked (DESIGN 7.2): the
        # check follows the reload that lands the write. The warning
        # belongs to the asset it was raised in.
        self._accepts: list[dict] = []
        self._accept_task: asyncio.Future | None = None
        # The file state the last reload evaluated, which its check judges.
        # The last reload that evaluated, as (asset path, opened, ev):
        # the file state the accepted-value checks judge and a pin
        # resolves against (DESIGN 7.2), never a rebuild. Kept across a
        # failed reload (the last good file state) and replaced whole,
        # so a worker thread reads one consistent triple; the path says
        # which asset it is, since a re-root swaps the path before its
        # reload lands.
        self._file_eval: tuple[Path, target_mod.Opened, Evaluation] | None = None
        # The red ``accept_validated`` of the displayed project (DESIGN
        # 7.2): held by the project, sent to every client and replayed
        # to a connecting one, standing until a reload's check clears it.
        self._warning: dict | None = None

    def _refuse_other_root(self, req: web.Request) -> web.Response | None:
        """The answer to a channel request sent for another root (DESIGN
        6): the caller names the root it speaks for, and a server of a
        different one refuses instead of answering from its own
        projects, where a project of the same name would be shot or
        pinned in the caller's place."""
        raw = req.headers.get(ROOT_HEADER)
        if raw is None:
            return None
        root = Path(urllib.parse.unquote(raw))
        if root.resolve() == self.watch_root:
            return None
        err = MoraineError(
            "wrong_workspace",
            f"the server on port {self.port} serves '{self.watch_root}', "
            f"not '{root}'; run 'moraine serve' on that root with --port "
            f"<other>, and pass the same --port here",
            serves=str(self.watch_root),
        )
        return web.json_response({"error": err.to_dict()}, status=400)

    def _resolve_project(self, name) -> tuple[Path, str | None]:
        """The asset a server channel addresses (DESIGN 7.2): ``None``
        is the displayed project; a name resolves through the roster
        (names, never paths). A single-project server answers only for
        the project it serves - a foreign name is refused rather than
        silently shot as the displayed one."""
        if name is None:
            return self.asset_path, self.project
        if not isinstance(name, str):
            raise MoraineError("unknown_project", "project must be a name")
        if self.workspace is not None:
            return self.workspace.resolve_asset(name).resolve(), name
        served = self.project_dir.name
        if name != served:
            raise MoraineError(
                "not_served",
                f"this server serves '{served}' only, not '{name}'; run "
                f"'moraine serve <workspace root>' to serve every project",
            )
        return self.asset_path, self.project

    # -- evaluation ---------------------------------------------------------

    def _open(self, asset_path: Path, overrides: dict[str, object] | None = None):
        """The asset a pass evaluates, opened (DESIGN 6, 7.2): the
        recipe built with its closure, under the overlay's values when
        there is one. An overlay's build is a preview and writes
        nothing. Ids the file no longer declares are dropped from
        ``overrides`` in place (the file wins). An IR handed over by
        path is read as given, and an overlay patches the knobs it
        records as reaching one node param."""
        if asset_mod.is_ir_file(asset_path):
            opened = target_mod.open_target(asset_path)
            if overrides:
                _prune_stale_overrides(opened.asset, overrides)
                if overrides:
                    params_mod.apply_overrides(opened.asset, overrides)
            return opened
        patched = self._patched(asset_path, overrides)
        if patched is not None:
            return patched
        opened = target_mod.open_target(
            asset_path, offered=dict(overrides or {}), write=not overrides,
            strict_offers=False,
        )
        if overrides:
            _prune_stale_overrides(opened.asset, overrides)
            if not overrides:
                # Every overlaid id was edited away, so this pass is
                # file state: built as one, its IR written.
                opened = target_mod.open_target(asset_path, strict_offers=False)
        return opened

    def _patched(self, asset_path: Path, overrides: dict[str, object] | None):
        """The file state with an overlay of `direct` knobs patched
        onto it, or None where the overlay needs a rebuild (DESIGN
        7.2): a `direct` knob's value reached one node param whole, so
        the patch is the IR a rebuild would write. One `build`-bound
        knob in the overlay rebuilds for all of it."""
        held = self._file_opened
        if not overrides or held is None or held.build is None:
            return None
        if asset_path != self.asset_path or held.asset.path is None:
            return None
        decls = {d["id"]: d for d in held.asset.params or []}
        live = {k: v for k, v in overrides.items() if k in decls}
        if not live or not all(params_mod.is_direct(decls[k]) for k in live):
            return None
        _prune_stale_overrides(held.asset, overrides)
        sc = asset_mod.loads(asset_mod.dumps(held.asset))
        sc.path = held.asset.path
        params_mod.apply_overrides(sc, overrides)
        return dataclasses.replace(
            held, asset=sc, irs={**held.irs, held.asset.path: sc},
        )

    def _evaluate_results(
        self,
        overrides: dict[str, object] | None = None,
        opened=None,
        asset_path: Path | None = None,
        found: dict | None = None,
    ) -> tuple[dict, Evaluation, target_mod.Opened]:
        # Synchronous by design: overlay-write -> evaluate -> revert in
        # _set_param is atomic on the event loop only because nothing here
        # awaits. Keep it that way (or add locking before making it async).
        # The override-free file-state callers (screenshot, pin add) run
        # this in a worker thread instead, serialized among themselves by
        # _offloop_eval_gate — the overlay-atomicity rationale binds only
        # the loop-side callers, and the cache store is already
        # concurrent-writer-safe (machine-level and cross-process, and
        # core.atomic's temp names are unique per call, so a loop-side
        # reload and a worker-thread pass may write one record at once).
        # ``opened`` is an already-built, already-pruned-and-overlaid asset
        # (``_overlaid_asset``) — the caller's build is reused, never
        # repeated; safe exactly because nothing awaits in between, so the
        # file cannot have changed under it.
        # Without ``overrides`` this is a file-state pass — the screenshot
        # and reload channels never pass one, so a preview overlay cannot
        # reach an evidence render or the shared payloads (DESIGN 7.2).
        # ``asset_path``: a screenshot addressed to a project other than
        # the displayed one (DESIGN 7.2) evaluates that file state; the
        # watch set frames the displayed asset and must not follow it.
        # ``found`` takes what this pass's build found, by resolved
        # path, for the caller that names the revision: a failed pass
        # has one too.
        foreign = asset_path is not None and asset_path != self.asset_path
        file_state = not overrides and not foreign
        if opened is None:
            try:
                opened = self._open(asset_path or self.asset_path, overrides)
            except MoraineError as e:
                # Keep watching the files a failed build had named, so
                # fixing the sub-asset or the lib/ file (not only the
                # root) triggers the next reload. Only a file-state pass
                # may reframe the watch set.
                if file_state:
                    self._file_opened = None
                    self.watched = {self.asset_path} | set(getattr(e, "sources", ()))
                if found is not None:
                    found.update(getattr(e, "contents", {}))
                raise
        if file_state:
            self._file_opened = opened
        if found is not None:
            found.update(opened.found)
            if opened.build is not None:
                self._reload_built = (opened.asset.path, dict(opened.irs))
        # After pruning: an overlay whose every id was edited away IS file
        # state and must populate the persistent store; anything still
        # overridden is exploration and lands in the ephemeral preview
        # bucket (DESIGN 4.5), never the persistent LRU.
        preview = bool(overrides)
        ev = Evaluation(opened.asset, preview=preview, **opened.inputs)
        try:
            results = ev.results()
        finally:
            if file_state:
                # What to watch is every file the build named, a
                # requirements file not yet written included; for an
                # IR handed over by path, the IR files reached.
                self.watched = (
                    {self.asset_path} | set(opened.watch)
                    | set(opened.files(ev.asset_files))
                )
        return results, ev, opened

    def _evaluate(
        self,
        overrides: dict[str, object] | None = None,
        pre: tuple | None = None,
    ) -> tuple[dict, dict]:
        # ``pre`` = (opened asset, tier map) from ``_overlaid_asset``: the
        # set_param path already built and classified to dispatch on
        # tier, and repeating that work per gesture would build the
        # asset twice on the event loop for nothing.
        results, ev, opened = self._evaluate_results(
            overrides, opened=pre[0] if pre else None,
            found=self._reload_found if not overrides else None,
        )
        sc = ev.root
        if not overrides:
            self._file_eval = (self.asset_path, opened, ev)
        # Bucket records serve from /preview/, persistent ones from
        # /cache/ — one preview payload legitimately mixes both, since
        # the clean upstream of a scrubbed subgraph stays persistent
        # (DESIGN 4.5).
        def _url(r: dict, field: str) -> str:
            root = "preview" if r.get("bucket") else "cache"
            return f"/{root}/{r[field]}"

        stats = {name: r["stats"] for name, r in results.items()}
        asset_payload = {
            "type": "asset_updated",
            "project": self.project,
            "outputs": {name: r["node_id"] for name, r in results.items()},
            "glb_urls": {
                name: _url(r, "glb_file")
                for name, r in results.items()
                if "glb_file" in r
            },
            "image_urls": {
                name: _url(r, "png_file")
                for name, r in results.items()
                if "png_file" in r
            },
            "stats": stats,
            # The asset rollup ``moraine stats`` reports, field for
            # field (DESIGN 7.2): the HUD's totals and the CLI's are one
            # derivation, and it is arithmetic over stats already in
            # hand — no value loads, no cache reads. A preview payload
            # rolls up the preview, so the totals describe what the tab
            # displays.
            "asset": stats_mod.asset_rollup(stats, len(sc.nodes)),
        }
        # Schema after overrides: values reflect the current preview.
        # Tiers re-derive every build (DESIGN 7.2): a text override can
        # rename a material mid-session, moving knobs between tiers. The
        # reference-borne names come from this evaluation, and a
        # file-state pass refreshes the set every later classification
        # reads (``_overlaid_asset`` classifies before it evaluates).
        ref_names = tiers_mod.reference_material_names(results, ev.material_origins())
        if not overrides:
            self._reference_material_names = ref_names
        tmap = pre[1] if pre else tiers_mod.classify(sc, reference_names=ref_names)
        # ``overridden``: the overlay's ids after pruning (DESIGN 7.2) —
        # the client derives its Accept/Revert state from it rather
        # than guessing what the server holds.
        param_payload = {
            "type": "param_schema",
            "params": self._schema(sc, tmap),
            "overridden": sorted(overrides or {}),
        }
        return asset_payload, param_payload

    async def _report_asset_error(
        self,
        err: dict,
        event: str = "asset_error",
        *,
        to: web.WebSocketResponse | None = None,
        **log_fields,
    ) -> None:
        # asset_error carries the active project so clients can track
        # the server across a failed switch: a rejected set_asset left
        # the project unchanged (picker snaps back), an accepted-but-
        # broken one moved it (picker follows) - DESIGN 7.2. The one
        # builder of the message shape: every error path goes through
        # here so the documented message contract cannot drift per path.
        # ``to`` narrows delivery to one client — a rejected set_param
        # concerns only the overlaying client (DESIGN 7.2).
        _log(event, error=err, **log_fields)
        msg = {"type": "asset_error", "project": self.project, "error": err}
        if event == "asset_error":
            # Only a file-state failure makes the display stale against
            # the file; a rejected set_param/set_asset leaves it current,
            # so those are not worth replaying to a later tab.
            self.last_error = msg
        if to is None:
            await self._broadcast(msg)
        else:
            await self._send(to, msg)

    # Above this many changed lines (common prefix/suffix stripped) the
    # diff falls back to the coarse count: SequenceMatcher is quadratic
    # and this runs on the event loop, so a wholesale rewrite must not
    # stall broadcasts for an exact number nobody needs at that size.
    _DIFF_EXACT_MAX_LINES = 4000

    def _stamp_revision(self, found: dict, *, diff: bool) -> int | None:
        """Refresh the content hash and byte snapshot of every file the
        last evaluation reached — one read pass feeds both, so the hash
        and the snapshot can never see two different file states, and
        ``found`` (what that pass's build found, by resolved path)
        stands for the files it holds, so the revision is the one the
        pass built — and
        with ``diff``, return the edit size since the previous snapshot
        (lines added + removed, every file). Called from the file-state
        reload path only: screenshot and preview evaluations read the
        same files but are not edits, and snapshotting there would hide
        the next reload's change. A startup or set_asset reload only
        seats the snapshot (nothing was edited): ``diff=False`` returns
        None. A missing file is part of the revision (dangling
        ``asset_ref``); any other read failure leaves the revision
        unnamed (hash None) — an honest gap beats a hash of a file set
        that never existed on disk."""
        files = set(self.watched)
        fresh: dict[Path, bytes] = {}
        unreadable = False
        for f in files:
            key = f.resolve()
            if key in found:
                if found[key] is not None:
                    fresh[f] = found[key]
                continue
            try:
                fresh[f] = f.read_bytes()
            except FileNotFoundError:
                continue
            except OSError:
                unreadable = True
        changed: int | None = None
        if diff:
            changed = 0
            for f in set(fresh) | set(self._snapshot):
                if f.name == accepted_mod.PARAMS_FILE:
                    continue  # an accept's line is the human's pick, no edit
                old = self._snapshot.get(f, b"").decode("utf-8", "replace").splitlines()
                new = fresh.get(f, b"").decode("utf-8", "replace").splitlines()
                if old == new:
                    continue
                i = 0
                while i < len(old) and i < len(new) and old[i] == new[i]:
                    i += 1
                j = 0
                while (j < len(old) - i and j < len(new) - i
                       and old[len(old) - 1 - j] == new[len(new) - 1 - j]):
                    j += 1
                om, nm = old[i:len(old) - j], new[i:len(new) - j]
                if len(om) + len(nm) > self._DIFF_EXACT_MAX_LINES:
                    changed += len(om) + len(nm)
                    continue
                matcher = difflib.SequenceMatcher(None, om, nm, autojunk=False)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag != "equal":
                        changed += (i2 - i1) + (j2 - j1)
        self._snapshot = fresh
        self.asset_hash = (
            None if unreadable
            else asset_mod.content_hash_of_bytes(fresh, self.watch_root)
        )
        return changed

    async def _reload_and_broadcast(
        self, *, drop_overrides: bool = False, trigger: str = "watch"
    ) -> None:
        if drop_overrides:
            # File edits win: the file is the sole source of truth, so a
            # reload drops every client's preview overlay (DESIGN 7.2).
            # The drop is announced to the clients it actually affected —
            # an agent edit landing mid-exploration must read as an
            # event, never as the preview quietly vanishing. Announced at
            # the drop itself, before the evaluation: the overlay is gone
            # on the failure path too.
            dropped = [ws for ws, state in self.clients.items() if state.overrides]
            for state in self.clients.values():
                state.overrides.clear()
            for ws in dropped:
                await self._send(ws, {"type": "preview_dropped", "trigger": trigger})
        t0 = time.perf_counter()
        found = self._reload_found = {}
        try:
            self.asset_payload, self.param_payload = self._evaluate()
        except MoraineError as e:
            self._record_reload(trigger, t0, found, ok=False, code=e.code)
            await self._report_asset_error(e.to_dict())
            await self._unchecked(e.to_dict())
            # A failing evaluation still wrote records for every node
            # before the failure — fail-only sessions must check too.
            await self._maybe_evict()
            return
        except Exception as e:
            # Non-Moraine failures are still failures to report (DESIGN
            # 8.2) — an editor holding the file mid-save raises a raw
            # PermissionError here on Windows, and letting it escape
            # would kill the watch task silently: hot reload dead for
            # the session with the viewer stuck on a stale asset.
            err = MoraineError(
                "internal",
                f"reload failed unexpectedly ({type(e).__name__}: {e}); "
                f"the last good state is kept - re-save the asset to retry",
            )
            self._record_reload(trigger, t0, found, ok=False, code="internal")
            await self._report_asset_error(err.to_dict())
            await self._unchecked(err.to_dict())
            await self._maybe_evict()
            return
        self._record_reload(trigger, t0, found, ok=True)
        self._payload_gen += 1
        self.last_error = None
        _log("asset_updated", stats=self.asset_payload["stats"])
        await self._broadcast_state()
        await self._after_reload()
        if self.pins:
            # Pins survive the reload while the revision moved: refresh
            # the pins view so clients re-derive staleness against the
            # new hash (DESIGN 7.2).
            await self._broadcast(self._pins_payload())
        await self._maybe_evict()

    def _record_reload(self, trigger: str, t0: float, found: dict, *,
                       ok: bool, code: str | None = None) -> None:
        # The session log's edit event (DESIGN 6): one per file-state
        # evaluation, carrying the revision it reached and the edit size.
        ms = round((time.perf_counter() - t0) * 1000, 1)
        self._reload_found = None
        changed = self._stamp_revision(found, diff=trigger == "watch")
        graph = self._graph_change(trigger) if ok else None
        self._reload_built = None
        session.record(
            "reload", root=str(self.watch_root), project=self.project,
            trigger=trigger, ok=ok, code=code, ms=ms,
            asset_hash=self.asset_hash, changed_lines=changed, diff=graph,
        )

    def _graph_change(self, trigger: str) -> dict | None:
        """The ``build --diff`` counts of a reload (DESIGN 6): nodes
        and asserts added, removed, and changed since the file state
        before it. A startup or a re-root seats the base and has
        nothing to count, as it has no edit size."""
        built, self._reload_built = self._reload_built, None
        if built is None:
            # An IR handed over by path: no build, no graph to diff.
            self._diff_base = None
            return None
        base, self._diff_base = self._diff_base, built
        if trigger != "watch" or base is None or base[0] != built[0]:
            return None
        diffs = diff_mod.closure_diff(built[0], built[1], base[1].get)
        return diff_mod.counts(diffs)

    async def _broadcast_state(self) -> None:
        # Snapshot both payloads and send the pair per-socket: concurrent
        # broadcasts must never leave a client with an asset_updated from
        # one evaluation and a param_schema from another (a slow, write-
        # paused socket suspends this task mid-broadcast). The generation
        # check aborts a suspended loop once a newer evaluation exists —
        # without it, the resumed loop would deliver the older payloads
        # last and a slow client would settle on a stale asset.
        gen = self._payload_gen
        texts = [
            json.dumps(p, separators=(",", ":"))
            for p in (self.asset_payload, self.param_payload)
            if p is not None
        ]
        for ws in list(self.clients):
            if gen != self._payload_gen:
                return  # a newer broadcast is covering every client
            try:
                for text in texts:
                    await ws.send_str(text)
            except ConnectionResetError:
                self.clients.pop(ws, None)

    def _workspace_payload(self) -> dict | None:
        if self.workspace is None:
            return None
        return {
            "type": "workspace",
            "projects": list(self.workspace.projects),
            "changed": sorted(self.changed),
        }

    async def _refresh_roster(self) -> bool:
        """Re-read the workspace roster from disk (DESIGN 7.2): a project
        directory created after start-up is admitted, a removed one
        drops out, and the picker learns of either through a fresh
        ``workspace`` broadcast. Raises the discovery sweep's own error
        (a reserved name, a root asset) and keeps the last roster then.
        Returns whether the set changed."""
        if self.workspace is None:
            return False
        fresh = workspace_mod.discover(self.workspace.root)
        if fresh.projects == self.workspace.projects:
            return False
        self.workspace = fresh
        self.changed &= set(fresh.projects)
        await self._broadcast(self._workspace_payload())
        return True

    def _roster_touched(self, changes) -> bool:
        """Whether a watch batch could have changed the roster: a root
        child the roster does not list appeared, or one it lists went
        (a directory renamed arrives as the two), or an asset file
        directly under such a child was written or deleted (membership
        is decided by the child's top-level asset files, DESIGN 4.2).
        A write inside a listed project is none of these."""
        if self.workspace is None:
            return False
        root = self.watch_root
        for change, raw in changes:
            path = Path(raw)
            try:
                rel = path.resolve().relative_to(root)
            except ValueError:
                continue
            depth = len(rel.parts)
            if depth == 0 or depth > 2:
                continue
            name = rel.parts[0]
            if name.startswith(".") or name.lower() in asset_mod.RESERVED_DIRS:
                continue
            if depth == 2 and not sources_mod.is_asset_file(path):
                continue
            if depth == 1 and path.is_file():
                continue  # a root file (AGENTS.md, a journal) is not a project
            deleted = change in (Change.deleted, "deleted")
            if name not in self.workspace.projects or deleted:
                return True
        return False

    def _note_changed(self, changes) -> set[str]:
        """Roster projects, other than the displayed one, that a watch
        batch touched (an asset file under ``<root>/<name>/``): marked
        as changed-since-displayed. Returns the newly marked names."""
        if self.workspace is None:
            return set()
        root = self.watch_root
        fresh: set[str] = set()
        for _c, raw in changes:
            if not raw.lower().endswith(sources_mod.RECIPE_EXT):
                continue
            try:
                rel = Path(raw).resolve().relative_to(root)
            except ValueError:
                continue
            name = rel.parts[0] if rel.parts else None
            if (
                name in self.workspace.projects and name != self.project
                and name not in self.changed
            ):
                self.changed.add(name)
                fresh.add(name)
        return fresh

    async def _send_snapshot(self, ws: web.WebSocketResponse) -> None:
        # Replay order matters: the picker must exist client-side before
        # asset_updated names the active project to select in it. The
        # generation check mirrors _broadcast_state: if a switch lands
        # mid-replay, its own broadcast (which includes this client)
        # supersedes the rest of the snapshot. The last asset_error (if
        # still standing) follows the state pair: the display is the last
        # good state, the error says why it is not the file. watch_dead
        # replays last — a tab opened after the watch died must still
        # learn its display will not update.
        gen = self._payload_gen
        payloads = [
            self._workspace_payload(), self.asset_payload, self.param_payload,
            # The standing accept warning after the state pair: it names
            # the state the pair shows (DESIGN 7.2).
            self._warning,
            # Pins after the state pair (markers need the asset they
            # point into), before the error trailers; nothing to replay
            # when the store is empty.
            self._pins_payload() if self.pins else None,
            # Timeline after the asset state, before the error trailers:
            # root-scoped session state, nothing in it names the asset.
            self.timeline_payload,
            self.last_error,
        ]
        if self.watch_dead is not None:
            payloads.append({"type": "watch_dead", "error": self.watch_dead})
        for payload in payloads:
            if gen != self._payload_gen:
                return
            if payload is not None:
                await ws.send_str(json.dumps(payload, separators=(",", ":")))

    async def _set_asset(self, data: dict) -> None:
        name = data.get("project")
        if not isinstance(name, str):
            # Malformed message must not take the WebSocket down.
            err = MoraineError(
                "set_asset_invalid", "set_asset needs a string 'project'"
            )
            await self._report_asset_error(err.to_dict(), "set_asset_error")
            return
        if self.workspace is None:
            err = MoraineError(
                "unknown_project",
                "serve is running on a single project; restart with "
                "'moraine serve <workspace-root>' to switch projects",
            )
            await self._report_asset_error(err.to_dict(), "set_asset_error")
            return
        # Resolution is lazy (per switch): the roster is re-read first,
        # so a project created since the last message is a valid target.
        # An unresolvable target (unknown name included - resolve_asset
        # validates membership) swaps nothing — there is nothing to
        # watch; a resolvable-but-broken target DOES swap below, so
        # fixing its file hot-reloads into it (DESIGN 7.2).
        try:
            await self._refresh_roster()
            path = self.workspace.resolve_asset(name)
        except MoraineError as e:
            await self._report_asset_error(e.to_dict(), "set_asset_error")
            return
        # No await between the swap and _reload_and_broadcast's synchronous
        # _evaluate: concurrent set_asset/set_param stay serialized (the
        # same atomicity rule as _evaluate's header comment).
        self._bind_asset(path, name)
        seen = name in self.changed
        self.changed.discard(name)
        await self._reload_and_broadcast(drop_overrides=True, trigger="set_asset")
        if seen:
            await self._broadcast(self._workspace_payload())
        # The re-root cleared the pin store (per-asset state): tell every
        # client so markers vanish with the asset they pointed into.
        await self._broadcast(self._pins_payload())

    async def _set_param(self, ws: web.WebSocketResponse, data: dict) -> None:
        state = self.clients.get(ws)
        if state is None:  # pragma: no cover - closed socket race
            return
        # Two forms (DESIGN 7.2): {param_id, value} for one knob, and the
        # batch form {params: {param_id: value, ...}} — an Apply of staged
        # evaluation-tier edits, N knobs for one evaluation. The form,
        # not the key count, drives the dispatch below: a one-entry Apply
        # still carries the batch contract (one evaluation, never the
        # factor fast path).
        batch_form = "params" in data
        if batch_form:
            batch = data.get("params")
            if (
                not isinstance(batch, dict)
                or not batch
                or not all(isinstance(k, str) for k in batch)
            ):
                # Malformed message must not take the WebSocket down.
                err = MoraineError(
                    "param_invalid",
                    "set_param batch form needs a non-empty object 'params' "
                    "keyed by param_id",
                ).to_dict()
                await self._report_asset_error(err, "param_error", to=ws)
                return
            updates = dict(batch)
        else:
            pid = data.get("param_id")
            if not isinstance(pid, str):
                err = MoraineError(
                    "param_invalid", "set_param needs a string 'param_id'"
                ).to_dict()
                await self._report_asset_error(err, "param_error", to=ws)
                return
            updates = {pid: data.get("value")}
        overrides = state.overrides
        prevs = {k: overrides[k] for k in updates if k in overrides}
        overrides.update(updates)

        def revert() -> None:
            # Every updated key: a batch reverts atomically — a partial
            # overlay of a failed Apply would preview a state the client
            # never staged.
            for k in updates:
                if k in prevs:
                    overrides[k] = prevs[k]
                else:
                    overrides.pop(k, None)

        # A file-state reload that lands while a preview send is suspended
        # on this socket has cleared the overlay and re-broadcast file
        # state; the resumed send must not land a stale preview on top.
        gen = self._payload_gen
        tier = "evaluation"
        try:
            # One load + one classification per gesture, whatever the
            # path: the pair feeds the tier dispatch here and is handed
            # to the evaluation below (nothing awaits in between, so the
            # file cannot change under the reuse).
            opened, tmap = self._overlaid_asset(overrides)
            sc = opened.asset
            if not batch_form:
                # Tier dispatch (DESIGN 7.2): a factor-tier scrub updates
                # the overlay and validates — load, prune, overlay — but
                # never evaluates: the client already drives the uniform
                # live, so the reply is the schema alone (value-honesty),
                # zero evaluations, zero cache writes. Seed and
                # evaluation knobs (and a pruned stale id) take the
                # evaluation path below. The batch form never dispatches:
                # an Apply promises one evaluation regardless of what its
                # keys classify as.
                (pid,) = updates
                tier = tmap.get(pid, {}).get("tier", "evaluation")
                if tier == "factor":
                    payload = {
                        "type": "param_schema",
                        "params": self._schema(sc, tmap),
                        "overridden": sorted(overrides),
                    }
                    if gen == self._payload_gen:
                        await self._send(ws, payload)
                    return
            asset_payload, param_payload = self._evaluate(overrides, pre=(opened, tmap))
        except Exception as exc:
            # Revert the failing update; the viewer keeps last good state.
            # Re-send the last good schema so the panel snaps back instead
            # of displaying the rejected value. Every exception class: a
            # non-Moraine failure that skipped the revert would leave the
            # poisoned value re-applied (and re-failing) on every later
            # set_param, wedging the overlay until a file edit drops it.
            revert()
            if isinstance(exc, MoraineError):
                err = exc.to_dict()
            else:
                err = MoraineError(
                    "internal",
                    f"set_param failed unexpectedly "
                    f"({type(exc).__name__}: {exc}); the override was reverted",
                ).to_dict()
            await self._report_asset_error(
                err, "param_error", to=ws, **self._param_fields(updates, batch_form)
            )
            snap_back = self._preview_schema(overrides)
            if snap_back is not None and gen == self._payload_gen:
                await self._send(ws, snap_back)
            # Same as _reload_and_broadcast: writes before the failing
            # node still count toward the post-growth check.
            await self._maybe_evict()
            return
        # To the overlaying client only — the shared payloads, the
        # generation counter, and last_error stay file-state (DESIGN 7.2).
        for payload in (asset_payload, param_payload):
            if gen != self._payload_gen:
                break  # a file-state reload superseded this preview
            await self._send(ws, payload)
        # Let the transport flush before the socket's next request is
        # read: the next evaluation runs synchronously on the loop, and a
        # frame still queued behind an in-progress write would otherwise
        # leave this reply on the server until that evaluation ends — a
        # fast pick's busy signal (DESIGN 7.2) lasting a slow Apply.
        await asyncio.sleep(0)
        await self._maybe_evict()

    def _overlaid_asset(self, overrides: dict[str, object]):
        """The asset a preview describes, without evaluating it: built
        under the overlay, stale overlay ids pruned in place
        (``_prune_stale_overrides``) — plus its tier map."""
        opened = self._open(self.asset_path, overrides)
        return opened, tiers_mod.classify(
            opened.asset, reference_names=self._reference_material_names
        )

    async def _revert(self, ws: web.WebSocketResponse) -> None:
        """Drop this client's preview overlay and show the file again
        (DESIGN 7.2 viewer revert): the A/B half of accept, so a scrub
        stops being an irreversible comparison. Costs nothing — the
        shared file-state pair is already built, so the answer is a
        replay to this one socket: zero evaluations, zero cache writes,
        no broadcast and no generation bump (nothing about file state
        changed). The drop rides the same ``preview_dropped``
        announcement a reload sends, so one client rule covers both
        ways an overlay can end; client-local staged edits are not part
        of the overlay and survive it, here as there."""
        state = self.clients.get(ws)
        if state is None:  # pragma: no cover - closed socket race
            return
        if not state.overrides:
            await self._report_asset_error(
                MoraineError(
                    "revert_empty",
                    "nothing previewed to revert: this tab already shows the "
                    "file - scrub, apply, or reroll a knob first",
                ).to_dict(),
                "revert_error",
                to=ws,
            )
            return
        state.overrides.clear()
        gen = self._payload_gen
        await self._send(ws, {"type": "preview_dropped", "trigger": "revert"})
        for payload in (self.asset_payload, self.param_payload):
            if gen != self._payload_gen:
                # A file-state reload landed mid-replay; its own
                # broadcast (which includes this client) supersedes.
                return
            if payload is not None:
                await self._send(ws, payload)

    async def _accept(self, ws: web.WebSocketResponse, data: dict | None = None) -> None:
        """Write this connection's overlay to the project's
        ``params.json`` (DESIGN 7.2): exactly the state the tab
        displays, declared knob values and nothing else, each already
        judged by its declaration. The recipe is not touched. The check
        of the accepted state follows the reload that lands it
        (``_after_reload``), which also finds the pick that can move no
        check."""
        state = self.clients.get(ws)
        if state is None:  # pragma: no cover - closed socket race
            return
        overrides = state.overrides
        t0 = time.perf_counter()

        async def refuse(err: MoraineError) -> None:
            self._record_accept(sorted(overrides), t0, ok=False, code=err.code)
            await self._report_asset_error(err.to_dict(), "accept_error", to=ws)

        if asset_mod.is_ir_file(self.asset_path):
            await refuse(MoraineError(
                "accept_no_project",
                "an IR opened by path has no project to accept into; serve "
                "the project or its recipe, and accept there",
            ))
            return
        if self.asset_path.parent != self.project_dir:
            await refuse(MoraineError(
                "accept_not_root",
                f"'{self.asset_path.name}' is a sub-asset, and "
                f"'{accepted_mod.PARAMS_FILE}' holds the values of the "
                f"project's root asset; declare a knob in the root, pass "
                f"its value on in the overrides of the reference, and "
                f"accept it there",
            ))
            return
        path = self.project_dir / accepted_mod.PARAMS_FILE
        try:
            opened = self._open(self.asset_path, overrides)
        except MoraineError as e:
            await refuse(MoraineError(
                "accept_failed",
                f"cannot read the asset to accept into ({e.code}: "
                f"{e.message}); fix the recipe first",
            ))
            return
        if not overrides:
            await refuse(MoraineError(
                "accept_empty",
                "nothing previewed to accept: scrub, apply, or reroll a "
                "knob first - accept writes exactly what this tab shows",
            ))
            return
        try:
            before = self._accepted_values(path)
            declared = {d["id"]: d["value"] for d in opened.asset.params or []}
            # A line for an id the asset declares no knob for is
            # dropped with the write: the human's act, and its diff.
            dropped = sorted(set(before) - set(declared))
            after = {k: v for k, v in before.items() if k in declared}
            for k, v in overrides.items():
                # An unset knob's line leaves with the write (DESIGN 7.2).
                if v is None:
                    after.pop(k, None)
                else:
                    after[k] = declared[k]
            ids = sorted(overrides)
            accepted_mod.write(path, after)
        except MoraineError as e:
            await refuse(e)
            return
        self._accepts.append({
            "ws": ws, "t0": t0, "params": ids, "project": self.project,
            # Whether the reload that lands it may find it moves no
            # check (DESIGN 7.2). Not while the warning stands, which
            # that reload validates anyway. Not a knob that fails its
            # own entry now, which the pick turns green.
            "may_skip": self._warning is None and not self._fails_own_entry(ids),
        })
        ack = {"type": "accepted", "params": ids}
        if dropped:
            ack["dropped"] = dropped
        await self._send(ws, ack)

    def _fails_own_entry(self, ids: list[str]) -> bool:
        """Whether one of these knobs fails its own entry of the report
        in the file state: a value outside a narrowed range, or one of
        another kind in the human's file. Unknown counts as failing."""
        if self._file_eval is None:
            return True
        failing = {
            e["param"] for e in params_mod.range_entries(self._file_eval[1].asset)
            if not e["ok"]
        }
        return any(k in failing for k in ids)

    def _moves_no_check(self, ids: list[str]) -> bool:
        """Whether a pick of these knobs alone can change no entry of
        ``validate`` (DESIGN 7.2), asked of the state the reload built:
        the same state the check would judge."""
        if not ids or self.param_payload is None:
            return False
        knobs = {p["param_id"]: p for p in self.param_payload["params"]}
        return all(k in knobs and validation.moves_no_entry(knobs[k]) for k in ids)

    def _schema(self, sc, tmap) -> list[dict]:
        """The panel's schema, each knob the human's file holds carrying
        that value as ``accepted`` (DESIGN 7.2). A preview overlays the
        entry's ``value``, so the file is the one place the accepted
        value still stands while a preview shows. A value of another
        kind than the knob's is left out, as the build refuses it."""
        schema = params_mod.build_schema(sc, tmap)
        try:
            held = self._accepted_values(self.project_dir / accepted_mod.PARAMS_FILE)
        except MoraineError:
            held = {}
        for entry in schema:
            value = held.get(entry["param_id"])
            if value is not None and asset_mod.param_kind(value) == entry["kind"]:
                entry["accepted"] = value
        return schema

    def _accepted_values(self, path: Path) -> dict:
        """What the project's ``params.json`` holds now; nothing for a
        file not yet written."""
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return {}
        return accepted_mod.loads(data, path.name)

    def _accept_verdict(self, sc, ev) -> dict:
        """The ``accept_validated`` payload of a state (DESIGN 7.2):
        ``validate`` over it, as its gating summary."""
        report, _ = validation.report(sc, ev)
        return {
            "type": "accept_validated", "project": self.project,
            "ok": report["ok"], "failed": validation.failing(report),
        }

    def _pending(self) -> list[dict]:
        """The accepts written into the displayed project and not yet
        checked."""
        return [a for a in self._accepts if a["project"] == self.project]

    async def _after_reload(self) -> None:
        """Check the accepted state once a reload has landed it, and
        keep the standing warning current (DESIGN 7.2): every client
        gets the gating summary, the accept's event carries the same,
        and the warning is re-checked on each reload of its project
        until it clears. A project with neither pays nothing, and an
        accept that can move no check is answered here without one. The check
        is a task of its own: `validate` over a large asset outlasts
        what a reload may wait, and the viewer follows the file
        meanwhile. It judges the state this reload evaluated, for the
        accepts this reload landed: one written while the check runs
        is the next reload's."""
        landed = self._pending()
        free = [
            a for a in landed
            if a["may_skip"] and self._moves_no_check(a["params"])
        ]
        if free:
            landed = [a for a in landed if not any(a is b for b in free)]
            self._accepts = [a for a in self._accepts if not any(a is b for b in free)]
        for accept in free:
            # No verdict, and the event says so by carrying none.
            self._record_accept(accept["params"], accept["t0"], ok=True)
            if accept["ws"] in self.clients:
                await self._send(accept["ws"], {
                    "type": "accept_validated", "project": self.project,
                    "params": accept["params"], "checked": False,
                    "moves_no_check": True,
                })
        if (landed or self._warning is not None) and self._file_eval is not None:
            _, opened, ev = self._file_eval
            self._accept_task = asyncio.ensure_future(
                self._check_accepted(
                    self._payload_gen, self.project, landed, opened.asset, ev
                )
            )

    async def _check_accepted(self, gen: int, project: str | None,
                              landed: list[dict], sc, ev) -> None:
        def current() -> bool:
            # A later reload judges the state that replaced this one.
            return gen == self._payload_gen and project == self.project

        try:
            async with self._accept_gate:
                if not current():
                    return
                payload = await _apart(self._accept_verdict, sc, ev)
        except Exception as e:
            if current():
                err = e if isinstance(e, MoraineError) else MoraineError(
                    "internal", f"{type(e).__name__}: {e}",
                )
                await self._unchecked(err.to_dict())
            return
        if not current():
            return
        # An accept another check settled meanwhile is settled once.
        landed = [a for a in landed if any(a is b for b in self._accepts)]
        self._accepts = [a for a in self._accepts if not any(a is b for b in landed)]
        for accept in landed:
            self._record_accept(
                accept["params"], accept["t0"], ok=True,
                summary={
                    "validate_ok": payload["ok"],
                    "failed": [row["entry"] for row in payload["failed"]],
                },
            )
        if not landed and self._warning is None:
            return
        # The verdict names the knobs of every accept it stands for: the
        # ones this check judged and, while the warning stands, the ones
        # it was raised for, so a later pick that moved no entry does
        # not take the red's name.
        message = dict(payload)
        names = set(self._warning["params"]) if self._warning is not None else set()
        for accept in landed:
            names |= set(accept["params"])
        message["params"] = sorted(names)
        self._warning = None if payload["ok"] else message
        await self._broadcast(message)

    async def _unchecked(self, error: dict) -> None:
        """Tell every client that the verdict of an accept is not yet
        there (DESIGN 7.2): the accepted state did not build or could
        not be judged. The accept stays to be checked by the reload
        that reaches a state."""
        for accept in self._pending():
            await self._broadcast({
                "type": "accept_validated", "project": self.project,
                "params": accept["params"], "checked": False,
                "error": error,
            })

    def _settle_unchecked(self) -> None:
        """Record the accepts no check will reach, without a verdict
        (DESIGN 6): one `accept` per viewer accept, also where the
        project was left or the server stopped before its check."""
        for accept in getattr(self, "_accepts", []):
            self._record_accept(
                accept["params"], accept["t0"], ok=True, project=accept["project"],
            )
        self._accepts = []

    def _record_accept(
        self, param_ids: list[str], t0: float, *, ok: bool,
        code: str | None = None, summary: dict | None = None,
        project: str | None = None,
    ) -> None:
        session.record(
            "accept", root=str(self.watch_root), project=project or self.project,
            ok=ok, code=code,
            ms=round((time.perf_counter() - t0) * 1000, 1),
            param_ids=sorted(param_ids),
            **({"summary": summary} if summary is not None else {}),
        )

    def _preview_schema(self, overrides: dict[str, object]) -> dict | None:
        """The panel snap-back after a rejected override: the schema
        needs no evaluation — load, prune, overlay, build. Falls back to
        the shared file-state schema when even loading fails (the file
        broke underneath); may return None early in a session."""
        try:
            opened, tmap = self._overlaid_asset(overrides)
            return {
                "type": "param_schema",
                "params": self._schema(opened.asset, tmap),
                "overridden": sorted(overrides),
            }
        except Exception:
            # The file broke underneath, so the shared file-state schema
            # is the only one left to show — but this connection's
            # overlay is still standing, and saying otherwise would gray
            # out the landings on a preview the human can still accept
            # or revert (with the watch dead, no reload will come to
            # make an empty claim true).
            if self.param_payload is None:
                return None
            return {**self.param_payload, "overridden": sorted(overrides)}

    @staticmethod
    def _param_fields(updates: dict[str, object], batch: bool = False) -> dict:
        """The knob-identity fields of a preview record or error: the
        single form keeps the historical ``param_id``; the batch form
        carries sorted ``param_ids`` whatever its size — the field names
        the protocol form, not the key count (DESIGN 6 — names, never
        values)."""
        if not batch:
            return {"param_id": next(iter(updates))}
        return {"param_ids": sorted(updates)}

    def _pins_payload(self) -> dict:
        # Full state, idempotent: receivers replace, never merge.
        return {
            "type": "pins", "project": self.project,
            "asset_hash": self.asset_hash, "pins": self.pins,
        }

    async def _add_pin(
        self,
        output,
        *,
        face=None,
        instance=None,
        predicate=None,
        position=None,
        source: str,
    ) -> dict:
        """Resolve one pin against file state (the screenshot rule: a
        preview overlay active in the pointing tab must never skew the
        record) and store it. Raises MoraineError; the caller owns
        reporting, the session-log event, and the ``pins`` broadcast.
        Resolution reads the records the last reload's evaluation
        already holds, so no build and no sweep stands between a click
        and its pin; a server that has not evaluated the displayed
        asset yet (a re-root whose reload has not landed) rebuilds, and
        a record evicted from under the held evaluation falls back to
        the same fresh pass. It runs in a worker thread (a cold-cache
        evaluation on the loop would stall every WebSocket for its
        duration); the store mutation stays on the loop. A reload
        landing across the await leaves the pin honestly stamped with
        the revision it resolved against — the same staleness the pins
        panel already surfaces."""
        if not isinstance(output, str):
            raise MoraineError("pin_invalid", "pin needs a string 'output'")
        if position is not None and not (
            isinstance(position, list) and len(position) == 3
            and all(ops_common.is_num(v) for v in position)
        ):
            raise MoraineError(
                "pin_invalid", "pin 'position' must be [x, y, z] finite numbers"
            )

        def _resolve_in(opened, ev) -> tuple[asset_mod.Asset, dict, str]:
            sc = opened.asset
            # unknown_output lists the roster
            rec = ev.records(only=output)[output]
            subject = f"output '{output}'"
            if predicate is not None:
                predicates.validate_predicate(predicate, where="pin")
                resolved = predicates.resolve_predicate_pin(
                    rec.value(), predicate, instance, subject=subject
                )
            else:
                resolved = predicates.resolve_pin(
                    rec.value(), face, instance, subject=subject,
                    position=position,
                )
            # Framed over the full watched set — the same file-set
            # framing as the reload stamp — so the pin's hash is
            # comparable to the server revision: hashing only the pinned
            # output's chain would read "stale" the moment any other
            # output pulls in an asset_ref. ``missing_ok``: the watched
            # set legitimately holds a dangling reference target (kept
            # so re-creating it hot-reloads), and a pin on a healthy
            # output must resolve in that state, as the reload stamp
            # does.
            stamp = opened.revision_hash(
                ev.asset_files, root=self.watch_root,
                also={self.asset_path} | self.watched, missing_ok=True,
            )
            return sc, resolved, stamp

        def _resolve() -> tuple[asset_mod.Asset, dict, str]:
            held = self._file_eval
            if held is not None and held[0] == self.asset_path:
                try:
                    return _resolve_in(held[1], held[2])
                except MoraineError as e:
                    if e.code != "cache_missing":
                        raise
                    # A held record's bytes are gone: evaluate afresh.
            opened = self._open(self.asset_path)
            return _resolve_in(opened, Evaluation(opened.asset, **opened.inputs))

        async with self._offloop_eval_gate:
            sc, resolved, stamp = await asyncio.to_thread(_resolve)
        self._pin_seq += 1
        pin = {
            "id": self._pin_seq,
            "output": output,
            "node_id": sc.outputs[output],
            **where(sc.nodes[sc.outputs[output]]),
            **resolved,
            "source": source,
            **({"predicate": predicate} if predicate is not None else {}),
            "asset_hash": stamp,
            "t": session._now_iso(),
        }
        self.pins.append(pin)
        if len(self.pins) > _PINS_MAX:
            del self.pins[0]
        return pin

    def _remove_pin(self, pid) -> MoraineError | None:
        """Store-level unpin, transport-neutral like ``_add_pin``: both
        the WebSocket and HTTP surfaces mutate through here so the
        filter and the ``unknown_pin`` answer cannot drift apart.
        Returns the error (None on success); the caller owns delivery
        and the ``pins`` broadcast."""
        kept = [p for p in self.pins if p["id"] != pid]
        if len(kept) == len(self.pins):
            return MoraineError(
                "unknown_pin",
                f"no pin with id {pid!r}; current ids: "
                f"{[p['id'] for p in self.pins]}",
            )
        self.pins[:] = kept
        return None

    def _clear_pins(self) -> int:
        """Store-level clear, shared by both transports; returns the
        count cleared. The caller owns the ``pins`` broadcast."""
        cleared = len(self.pins)
        self.pins.clear()
        return cleared

    async def _select(self, ws: web.WebSocketResponse, data: dict) -> None:
        # The deixis seam (DESIGN 7.2). Malformed or unresolvable selects
        # answer the pointing client only and never take the socket down.
        action = data.get("action")
        if action in ("pin", "replace"):
            output = data.get("output")
            try:
                pin = await self._add_pin(
                    output,
                    face=data.get("face"), instance=data.get("instance"),
                    position=data.get("position"), source="viewer",
                )
            except Exception as exc:
                if isinstance(exc, MoraineError):
                    err = exc
                else:
                    err = MoraineError(
                        "internal",
                        f"pin resolution failed unexpectedly "
                        f"({type(exc).__name__}: {exc})",
                    )
                await self._report_asset_error(err.to_dict(), "select_error", to=ws)
                await self._maybe_evict()
                return
            if action == "replace" and len(self.pins) > 1:
                # One click = one atomic mutation: the new pin becomes
                # the sole selection. Cleared only after a successful
                # resolution (a failed replace must not wipe the store),
                # one broadcast — no client-composed unpin burst to race
                # concurrent pin writers.
                self.pins[:] = [pin]
            await self._broadcast(self._pins_payload())
            # Resolution evaluates file state and may have written cache
            # entries on a cold start — same post-growth rule as reloads.
            await self._maybe_evict()
        elif action == "unpin":
            err = self._remove_pin(data.get("id"))
            if err is not None:
                await self._report_asset_error(err.to_dict(), "select_error", to=ws)
                return
            await self._broadcast(self._pins_payload())
        elif action == "clear":
            self._clear_pins()
            await self._broadcast(self._pins_payload())
        else:
            err = MoraineError(
                "select_invalid",
                "select needs 'action': one of pin | replace | unpin | clear",
            )
            await self._report_asset_error(err.to_dict(), "select_error", to=ws)

    async def _maybe_evict(self) -> None:
        # The CLI's post-growth check, debounced for a long-lived process:
        # only after a re-eval that wrote cache bytes, at most every 10
        # minutes, and off the event loop so viewer pushes are never
        # delayed (DESIGN 4.5). Runs after _broadcast_state — eviction
        # never sits between an evaluation and its push.
        try:
            # Everything inside the guard: even reading the cap can raise
            # (bad env value), and housekeeping must never take down the
            # watch task or the WebSocket handler it runs under.
            if cache_mod.max_bytes() == 0 or cache_mod.session_wrote() == 0:
                return
            now = time.monotonic()
            if self._last_evict is not None and now - self._last_evict < 600.0:
                return  # keep accumulating; the next eligible re-eval checks
            self._last_evict = now
            # The trigger policy (credit take, restore on failure) lives
            # in store.maybe_evict — one owner for CLI and server; this
            # method adds only the long-lived process's debounce and the
            # thread hop.
            report = await asyncio.to_thread(cache_mod.maybe_evict)
        except (MoraineError, OSError) as e:
            _log("evict_error", error=str(e))
            return
        if report and report.get("evicted"):
            _log("evicted", **report)

    async def _broadcast(self, msg: dict) -> None:
        text = json.dumps(msg, separators=(",", ":"))
        for ws in list(self.clients):
            try:
                await ws.send_str(text)
            except ConnectionResetError:
                self.clients.pop(ws, None)

    async def _send(self, ws: web.WebSocketResponse, msg: dict) -> None:
        # The per-client half of _broadcast: preview payloads and rejected
        # set_params reach only the client they concern (DESIGN 7.2).
        try:
            await ws.send_str(json.dumps(msg, separators=(",", ":")))
        except ConnectionResetError:
            self.clients.pop(ws, None)

    def _deliver_screenshot_result(self, data: dict) -> None:
        rid = data.get("request_id", "")
        if not isinstance(rid, str):
            # Client-supplied value used as a dict key: an unhashable
            # request_id would raise out of the message loop and take the
            # WebSocket down, which a malformed message must never do.
            return
        fut = self.pending_shots.get(rid)
        if fut is None or fut.done():
            return
        if data.get("protocol") != _SCREENSHOT_PROTOCOL:
            # A tab loaded before an upgrade answers the new request with
            # its own shared asset — exactly the stale shot this channel
            # exists to rule out. Ignore it (do not resolve) so a current
            # client can still win within the timeout; the waiter reports
            # stale_client only if nobody current answers.
            self.stale_shots.add(rid)
            _log("stale_screenshot_result", request_id=rid)
            return
        self.pending_shots.pop(rid)
        # The whole message: the waiter reads png and the error arm.
        fut.set_result(data)

    async def _mark_watch_dead(self, exc: BaseException) -> None:
        # Watch death detection (DESIGN 7.2 ``watch_dead``): the one failure mode
        # _watch cannot retry through is awatch itself dying — without
        # this, the server keeps serving with hot reload silently off and
        # every screenshot of the stale display looks legitimate. The
        # death is logged every time; the sticky flag and its broadcast
        # are set once per outage (a re-arm attempt that dies at birth
        # must not flap the viewer).
        error = MoraineError(
            "watch_dead",
            f"file watch died ({type(exc).__name__}: {exc}); hot reload "
            f"is off until the watch re-arms (retried with backoff) - "
            f"the display may be stale meanwhile",
        ).to_dict()
        _log("watch_error", error=error)
        if self.watch_dead is not None:
            return
        self.watch_dead = error
        session.record(
            "watch_dead", root=str(self.watch_root), port=self.port,
            error=error["message"],
        )
        await self._broadcast({"type": "watch_dead", "error": error})

    async def _mark_watch_alive(self) -> None:
        # The re-arm succeeded (DESIGN 7.2 ``watch_alive``): clear the
        # sticky flag, say so, and catch up — edits that landed while the
        # watch was dead produced no event, so the display is re-derived
        # from the file once, the same way a watch event would.
        self.watch_dead = None
        _log("watch_rearmed", root=str(self.watch_root))
        session.record("watch_alive", root=str(self.watch_root), port=self.port)
        await self._broadcast({"type": "watch_alive"})
        await self._reload_and_broadcast(drop_overrides=True, trigger="watch")

    async def _mark_listen_dead(self, why: str) -> None:
        # Listener death (DESIGN 7.2 ``listen_dead``): logged every time
        # a check or a re-bind fails; the sticky flag, its session
        # record, and the broadcast to the tabs that are still connected
        # — the only parties reachable — once per outage.
        error = MoraineError(
            "listen_dead",
            f"the server stopped accepting connections ({why}); connected "
            f"tabs keep streaming, but new tabs and commands cannot reach "
            f"port {self.port} until the listener re-arms (retried with "
            f"backoff)",
        ).to_dict()
        _log("listen_error", error=error)
        if self.listen_dead is not None:
            return
        self.listen_dead = error
        session.record(
            "listen_dead", root=str(self.watch_root), port=self.port,
            error=error["message"],
        )
        await self._broadcast({"type": "listen_dead", "error": error})

    async def _mark_listen_alive(self) -> None:
        self.listen_dead = None
        self.listen_rearmed += 1
        _log("listen_rearmed", port=self.port, count=self.listen_rearmed)
        session.record("listen_alive", root=str(self.watch_root), port=self.port)
        await self._broadcast({"type": "listen_alive"})

    async def _listen_probe(self, port: int) -> bool:
        # A loopback handshake and nothing else, made in a worker thread:
        # the budget is wall time there, so an evaluation blocking the
        # event loop delays the verdict but cannot turn a completed
        # handshake into a timeout. The request handler sees a
        # connection that closes before a byte.
        def connect() -> bool:
            try:
                with socket.create_connection(
                    ("127.0.0.1", port), timeout=_LISTEN_PROBE_TIMEOUT_S
                ):
                    return True
            except OSError:
                return False

        return await asyncio.to_thread(connect)

    async def _run_listen(self, runner: web.BaseRunner, port: int) -> bool:
        # One armed listener (DESIGN 7.2 ``listen_dead``), the attempt
        # ``_rearm_forever`` retries: bind if this server holds no site,
        # then check on an interval — the socket under the site and the
        # loopback probe — until a check fails. Returns True on death
        # (retry), False when canceled. Only a site this server bound
        # counts: a bind refused (a foreign listener took the port) is a
        # death, whatever a probe would say.
        site = self._site
        if site is None:
            site = web.TCPSite(runner, "127.0.0.1", port)
            try:
                await site.start()
            except OSError as e:
                await self._mark_listen_dead(f"re-bind failed: {type(e).__name__}: {e}")
                return True
            self._site = site
        try:
            while True:
                if not (_site_listening(site) and await self._listen_probe(port)):
                    await self._mark_listen_dead(
                        "the listening socket is closed or refuses the "
                        "loopback handshake"
                    )
                    break
                await asyncio.sleep(_LISTEN_PROBE_S)
        except asyncio.CancelledError:
            return False
        self._site = None
        try:
            await site.stop()  # releases the (already dead) listening socket
        except RuntimeError as e:
            _log("listen_error", error=f"stop: {e}")
        return True

    async def _serve(
        self, app: web.Application, port: int, on_bound=None
    ) -> None:
        # The run loop: an explicit runner and site rather than
        # ``web.run_app``, so the listener can be re-armed in place —
        # run_app owns its site and offers no way back to it. Startup,
        # shutdown, and cleanup hooks fire exactly as under run_app; the
        # port is claimed before on_startup evaluates anything and the
        # site listens on it right after on_startup returns; the
        # termination signal the framework's loop handled cancels this
        # task instead (Windows has no loop signal handlers — the
        # console's Ctrl+C reaches asyncio.run directly), so a `kill`
        # still runs the shutdown hooks.
        sock = _claim_port(port)
        try:
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            try:
                loop.add_signal_handler(signal.SIGTERM, task.cancel)
            except (NotImplementedError, RuntimeError):
                pass
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            try:
                self._site = web.SockSite(runner, sock)
                await self._site.start()
                if on_bound is not None:
                    on_bound()
                await self._rearm_forever(
                    lambda: self._run_listen(runner, port), self._mark_listen_alive,
                )
            finally:
                await runner.cleanup()
        finally:
            sock.close()

    async def _run_watch(self, root: Path, on_change, on_dead, *, tag: str) -> bool:
        # One armed watch: a per-event failure is logged and retried by
        # the next event; awatch itself dying is reported through
        # on_dead, never hidden, and returns True so the re-arm loop
        # (``_watch_forever``) retries. Exception, not BaseException:
        # KeyboardInterrupt / GeneratorExit must propagate (awaiting
        # inside their delivery would raise).
        try:
            async for changes in awatch(root):
                try:
                    await on_change(changes)
                except Exception as e:
                    _log(tag, error=f"{type(e).__name__}: {e}")
        except asyncio.CancelledError:
            return False
        except Exception as e:
            await on_dead(e)
            return True
        return False

    async def _watch_forever(
        self, root: Path, on_change, on_dead, on_alive, *, tag: str
    ) -> None:
        # A file watch under the shared re-arm discipline: never retried
        # while the root itself is missing.
        await self._rearm_forever(
            lambda: self._run_watch(root, on_change, on_dead, tag=tag),
            on_alive, ready=root.is_dir,
        )

    async def _rearm_forever(self, attempt, on_alive, *, ready=lambda: True) -> None:
        # The re-arm discipline shared by the asset watch, the log watch,
        # and the listener (DESIGN 7.2): ``attempt`` arms one instance
        # and returns True when it died (retry) or False when canceled;
        # a death is retried with doubling backoff up to the cap, only
        # once ``ready`` says the retry can succeed; an attempt that dies
        # at birth is the same outage (its on_dead is called again — each
        # callback is idempotent per outage), one that outlives the birth
        # window is announced through on_alive; the backoff resets only
        # once an attempt has held for the stable period, so an instance
        # that keeps dying late never flaps.
        delay = _REARM_S
        dead = False
        while True:
            task = asyncio.ensure_future(attempt())
            started = time.monotonic()
            try:
                if dead:
                    done, _ = await asyncio.wait({task}, timeout=_BIRTH_S)
                    if not done:
                        dead = False
                        await on_alive()
                died = await task
            except asyncio.CancelledError:
                # Shutdown while an attempt runs: the inner instance must
                # not outlive this task.
                task.cancel()
                raise
            if not died:
                return
            dead = True
            if time.monotonic() - started >= _STABLE_S:
                delay = _REARM_S
            while True:
                await asyncio.sleep(delay)
                delay = min(delay * 2, _REARM_CAP_S)
                if ready():
                    break

    async def _watch(self, _app) -> None:
        # Watch the fixed root (workspace root in workspace mode, else the
        # project directory): the watched-file set changes as asset_ref
        # edges are edited or set_asset re-roots, and awatch's path set is
        # fixed at call time — so watch wide and filter down to the
        # currently reachable assets. Non-asset churn under the root
        # (exports/, .git) costs wakeups only, never reloads.
        async def on_change(changes):
            # Suffix pre-filter before the resolve() syscalls: export
            # or git churn under the root arrives here in hundred-file
            # batches, and every watched file is a recipe or an IR
            # handed over by path (lowercased: the shipped platforms'
            # filesystems are case-insensitive). The IR a reload's own
            # build writes is never watched, so a reload cannot wake
            # itself.
            if any(
                (
                    p.lower().endswith((sources_mod.RECIPE_EXT, asset_mod.ASSET_EXT))
                    or Path(p).name.lower() == accepted_mod.PARAMS_FILE
                )
                and Path(p).resolve() in self.watched
                for _c, p in changes
            ):
                await self._reload_and_broadcast(drop_overrides=True)
            if self._roster_touched(changes):
                try:
                    await self._refresh_roster()
                except MoraineError as e:
                    if e.code not in self._roster_errors:
                        self._roster_errors.add(e.code)
                        _log("roster_error", error=e.to_dict())
                else:
                    self._roster_errors.clear()
            if self._note_changed(changes):
                await self._broadcast(self._workspace_payload())

        await self._watch_forever(
            self.watch_root, on_change, self._mark_watch_dead, self._mark_watch_alive,
            tag="watch_error",
        )

    def _build_timeline(self) -> dict | None:
        # Blocking file I/O — callers run this via asyncio.to_thread.
        # session.latest_report is the reader ``moraine sessions`` uses,
        # so the panel and the CLI read the same numbers (DESIGN 7.2
        # ``timeline``).
        report = session.latest_report(self.watch_root)
        if report is None:
            return None
        return {
            "type": "timeline",
            "root": report["root"],
            "gap_minutes": report["gap_minutes"],
            "session": report["sessions"][-1],
        }

    async def _refresh_timeline(self) -> None:
        payload = await asyncio.to_thread(self._build_timeline)
        if payload is None:
            # A derivation that vanishes under an existing payload (the
            # day files holding the session deleted or retired
            # mid-serve) freezes the panel — it must say so rather than
            # keep the old economy up as if live (DESIGN 7.2
            # ``timeline`` ``stale``).
            if self.timeline_payload is not None:
                await self._mark_timeline_stale("session vanished from the log")
            return
        # Day files are machine-shared: a sibling root's CLI run wakes
        # the log watch with an identical derivation for this root, so
        # an unchanged payload is not re-broadcast. The broadcast may
        # land before or after a concurrent asset_updated — both are
        # self-contained full-state messages, so the order is free.
        if payload == self.timeline_payload:
            return
        self.timeline_payload = payload
        await self._broadcast(payload)

    async def _mark_timeline_stale(self, reason: str) -> None:
        # Frozen-panel detection (DESIGN 7.2 ``timeline``, the
        # ``watch_dead`` discipline): report only, sticky through the
        # payload itself, which the snapshot replays, until the log
        # watch re-arms. With no payload there is no panel to freeze,
        # and a session-less stale message would break the message's
        # own shape — log and stop.
        _log("timeline_watch_error", error=reason)
        if self.timeline_payload is None or self.timeline_payload.get("stale"):
            return
        self.timeline_payload = {**self.timeline_payload, "stale": True}
        await self._broadcast(self.timeline_payload)

    async def _mark_timeline_fresh(self) -> None:
        # The log watch re-armed (DESIGN 7.2 ``timeline``): the frozen
        # flag comes off the payload, and one refresh catches up the
        # events that landed while the watch was dead.
        _log("timeline_watch_rearmed")
        if self.timeline_payload is not None and self.timeline_payload.get("stale"):
            self.timeline_payload = {
                k: v for k, v in self.timeline_payload.items() if k != "stale"
            }
            await self._broadcast(self.timeline_payload)
        try:
            await self._refresh_timeline()
        except Exception as e:
            _log("timeline_watch_error", error=f"{type(e).__name__}: {e}")

    async def _watch_log(self, _app) -> None:
        # The asset watch's sibling over the session log directory
        # (DESIGN 7.2 ``timeline``): the rounds that drive the panel
        # are written by CLI processes, so the server cannot learn of
        # them from its own record() calls — it watches the day files
        # instead, one refresh path for every writer. No suffix filter
        # (the directory holds only day files); awatch's own step
        # batching coalesces bursts.
        d = session.log_dir()
        if d is None:
            return  # logging off — no panel (the absence rule)
        try:
            while not d.is_dir():
                # The startup record() failed (it warns, never raises)
                # before it could create the directory; a later writer
                # may still succeed — recheck instead of leaving the
                # panel dead for the server's lifetime.
                await asyncio.sleep(_LOG_DIR_RETRY_S)
        except asyncio.CancelledError:
            return
        try:
            # Cover events that landed between startup's refresh and
            # the watch starting (or while waiting for the directory).
            await self._refresh_timeline()
        except Exception as e:
            _log("timeline_watch_error", error=f"{type(e).__name__}: {e}")

        async def on_change(_changes):
            await self._refresh_timeline()

        async def on_dead(e):
            await self._mark_timeline_stale(f"{type(e).__name__}: {e}")

        await self._watch_forever(
            d, on_change, on_dead, self._mark_timeline_fresh, tag="timeline_watch_error",
        )

    # -- http handlers ------------------------------------------------------

    async def index(self, _req: web.Request) -> web.FileResponse:
        return web.FileResponse(VIEWER_DIR / "index.html")

    async def ws_handler(self, req: web.Request) -> web.WebSocketResponse:
        if not _origin_allowed(req, self.port):
            raise web.HTTPForbidden(text="cross-origin WebSocket refused")
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(req)
        try:
            # Registered inside the try: a socket that dies during the
            # snapshot must leave no client behind, or the screenshot
            # endpoint's no-viewer fail-fast sees a viewer that is not
            # there and waits out a render timeout instead.
            self.clients[ws] = ClientState()
            await self._send_snapshot(ws)
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue  # valid JSON, not a message: never take the socket down
                if data.get("type") == "screenshot_result":
                    self._deliver_screenshot_result(data)
                elif data.get("type") == "set_param":
                    await self._set_param(ws, data)
                elif data.get("type") == "set_asset":
                    await self._set_asset(data)
                elif data.get("type") == "accept":
                    await self._accept(ws, data)
                elif data.get("type") == "revert":
                    await self._revert(ws)
                elif data.get("type") == "select":
                    await self._select(ws, data)
        finally:
            self.clients.pop(ws, None)
        return ws

    async def status_api(self, req: web.Request) -> web.Response:
        # The server's liveness facts for the CLI (DESIGN 6): which root
        # and project it serves, at which revision, and whether the file
        # watch is armed — answered from memory, never an evaluation.
        # ``validate``/``stats`` carry it under ``serve`` so the one
        # failure agent-side checks cannot see (a dead watch behind a
        # green report) reaches the agent's own channel.
        if not _origin_allowed(req, self.port):
            return web.json_response(
                {"error": {"code": "forbidden", "message": "cross-origin status request refused"}},
                status=403,
            )
        return web.json_response({
            "root": str(self.watch_root),
            "project": self.project,
            "asset": self.asset_path.name,
            "asset_hash": self.asset_hash,
            "port": self.port,
            "watch": "dead" if self.watch_dead else "alive",
            # Answered over a fresh connection, so "alive" here is true
            # by construction; the count is the trace a half-death that
            # healed leaves (DESIGN 7.2 ``listen_dead``).
            "listen": "dead" if self.listen_dead else "alive",
            "listen_rearmed": self.listen_rearmed,
        })

    async def pins_api(self, req: web.Request) -> web.Response:
        # The agent's pin surface (DESIGN 6): same origin gate as the
        # screenshot endpoint, no viewer tab required — pins are server
        # memory. GET reads, POST places (predicate or face form),
        # DELETE removes one (?id=N) or all.
        if not _origin_allowed(req, self.port):
            return web.json_response(
                {"error": {"code": "forbidden", "message": "cross-origin pins request refused"}},
                status=403,
            )
        refused = self._refuse_other_root(req)
        if refused is not None:
            return refused
        # Addressed to the caller's project (DESIGN 7.2): pins are
        # pointing gestures in the displayed asset, so another project's
        # pins are none and a pin into one cannot be placed - refused,
        # never silently resolved against the displayed asset.
        if req.method == "POST":
            try:
                peek = await req.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                peek = {}
            raw = peek.get("project") if isinstance(peek, dict) else None
        else:
            raw = req.query.get("project")
        try:
            if isinstance(raw, str):
                await self._refresh_roster()
            _asset, project = self._resolve_project(raw)
        except MoraineError as e:
            return web.json_response({"error": e.to_dict()}, status=400)
        if project != self.project:
            if req.method == "GET":
                return web.json_response(
                    {"project": project, "displayed": self.project, "pins": []}
                )
            err = MoraineError(
                "not_displayed",
                f"the viewer displays '{self.project}', not '{project}'; a pin "
                f"points into what the human sees - say it in words, or ask "
                f"the human to switch",
            )
            return web.json_response({"error": err.to_dict()}, status=409)
        if req.method == "GET":
            # The agent's read stays terse (P5): the display-only
            # matched-face list (highlight resolution, DESIGN 7.2) rides
            # the pins broadcast to viewers, never the textual record.
            return web.json_response({
                "project": self.project,
                "asset_hash": self.asset_hash,
                "pins": [
                    {k: v for k, v in p.items() if k != "faces"}
                    for p in self.pins
                ],
            })
        if req.method == "DELETE":
            return await self._pins_delete(req)
        return await self._pins_post(req)

    async def _pins_post(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        output = body.get("output")
        try:
            if ("predicate" in body) == ("face" in body):
                raise MoraineError(
                    "pin_invalid",
                    "pin body needs 'output' and exactly one of 'predicate' "
                    "(a face predicate) or 'face' (a face index)",
                )
            pin = await self._add_pin(
                output,
                face=body.get("face"), instance=body.get("instance"),
                predicate=body.get("predicate"), source="api",
            )
        except Exception as exc:
            internal = not isinstance(exc, MoraineError)
            err = MoraineError(
                "internal",
                f"pin resolution failed unexpectedly ({type(exc).__name__}: {exc})",
            ) if internal else exc
            await self._maybe_evict()
            return web.json_response(
                {"error": err.to_dict()}, status=500 if internal else 400
            )
        await self._broadcast(self._pins_payload())
        await self._maybe_evict()
        return web.json_response({
            "pin": {k: v for k, v in pin.items() if k != "faces"},
            "pins": len(self.pins),
        })

    async def _pins_delete(self, req: web.Request) -> web.Response:
        raw = req.query.get("id")
        if raw is None:
            cleared = self._clear_pins()
            await self._broadcast(self._pins_payload())
            return web.json_response({"cleared": cleared})
        try:
            pid = int(raw)
        except ValueError:
            return web.json_response(
                {"error": {"code": "pin_invalid", "message": "id must be an integer pin id"}},
                status=400,
            )
        err = self._remove_pin(pid)
        if err is not None:
            return web.json_response({"error": err.to_dict()}, status=400)
        await self._broadcast(self._pins_payload())
        return web.json_response({"cleared": 1})

    @staticmethod
    def _serve_store_file(req: web.Request, root: Path) -> web.FileResponse:
        # Explicit handler instead of add_static: cache filenames are flat
        # content-addressed names (regex above blocks traversal), and
        # add_static's realpath containment check breaks under filesystem
        # virtualization/symlinks (e.g. MSIX-sandboxed %LOCALAPPDATA%).
        # One body for the /cache/ and /preview/ routes: the traversal
        # defense must not fork per bucket.
        name = req.match_info["filename"]
        path = root / name
        if not _CACHE_FILE_RE.fullmatch(name) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def cache_file(self, req: web.Request) -> web.FileResponse:
        try:
            return self._serve_store_file(req, cache_dir())
        except web.HTTPNotFound:
            self._heal_lost_record(req.match_info["filename"])
            raise

    def _heal_lost_record(self, name: str) -> None:
        """A 404 on a record this server is serving means the cache was
        emptied under it — a machine-wide ``cache clear`` from another
        process (DESIGN 4.5) — and the payload the tabs hold now names
        files that no longer exist. Re-materialize: one file-state reload
        (trigger ``cache_lost``) rebuilds the records under their
        content-addressed names and re-broadcasts, so a connected tab is
        never stranded on a URL nothing will ever answer again. One
        reload per outage: a page load 404s once per output, and a
        second 404 during the reload is the same outage. Unreferenced
        names (a stale tab's old generation, a typo) heal nothing — that
        record is not this server's to rebuild."""
        payload = self.asset_payload
        if payload is None:
            return
        url = f"/cache/{name}"
        if url not in payload["glb_urls"].values() and url not in payload["image_urls"].values():
            return
        if self._heal_task is not None and not self._heal_task.done():
            return
        _log("cache_lost", file=name)
        # A file-state reload like any other: an active preview overlay
        # is dropped and announced, never left behind on the server under
        # a tab that was told there is none.
        self._heal_task = asyncio.create_task(
            self._reload_and_broadcast(drop_overrides=True, trigger="cache_lost")
        )

    async def _purge_preview_bucket(self) -> None:
        # The ephemeral bucket is session-scoped (DESIGN 4.5): emptied at
        # serve start (a crashed session's leftovers have no other
        # reaper) and at stop (hygiene). Housekeeping must never keep the
        # server from starting or stopping — failures are one stderr
        # line, like the CLI's eviction check.
        try:
            await asyncio.to_thread(cache_mod.purge_preview)
        except Exception as e:
            print(f"preview purge skipped: {e}", file=sys.stderr)

    async def preview_file(self, req: web.Request) -> web.FileResponse:
        # The /cache/ route for the ephemeral preview bucket (DESIGN
        # 4.5): same body, rooted one directory down — through the real
        # bucket only, never a link wearing its name (the lexical root
        # below would follow one).
        if is_redirect(cache_dir() / PREVIEW_DIR_NAME):
            raise web.HTTPNotFound()
        return self._serve_store_file(req, cache_dir() / PREVIEW_DIR_NAME)

    async def screenshot(self, req: web.Request) -> web.Response:
        # Session-log wrapper (DESIGN 6): one ``screenshot`` event per
        # render request, success or failure — the record lands in the
        # ``finally`` so even an unexpected escape leaves a trace. The
        # provenance holder is per-request: requests overlap across the
        # viewer-render await, and shared state here once let a failing
        # request record a concurrent render's provenance.
        t0 = time.perf_counter()
        shot: dict = {}
        resp: web.Response | None = None
        try:
            resp = await self._screenshot(req, shot)
            return resp
        finally:
            if resp is None:
                ok, code = False, "internal"
            else:
                ok = resp.status == 200
                code = None
                if not ok:
                    try:
                        code = json.loads(resp.text or "")["error"]["code"]
                    except (ValueError, KeyError, TypeError, AttributeError):
                        code = f"http_{resp.status}"
            session.record(
                "screenshot", root=str(self.watch_root),
                project=shot.pop("project", self.project),
                ok=ok, code=code,
                ms=round((time.perf_counter() - t0) * 1000, 1), **shot,
            )

    async def _screenshot(self, req: web.Request, shot: dict) -> web.Response:
        # Dedicated render session (DESIGN 7.2, 11): a screenshot
        # renders the file's *current* state — a fresh override-free
        # evaluation — never the tab's loaded generation or the preview
        # overlay, and the response names what it rendered (project,
        # asset content hash, camera) so a stale or foreign shot is
        # detectable from the report alone.
        if not _origin_allowed(req, self.port):
            # A foreign page could otherwise drive evaluations and large
            # renders through a no-preflight cross-origin POST.
            return web.json_response(
                {"error": {"code": "forbidden", "message": "cross-origin screenshot request refused"}},
                status=403,
            )
        refused = self._refuse_other_root(req)
        if refused is not None:
            return refused
        try:
            body = await req.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        if not isinstance(body, dict):
            # Valid JSON that is not an object (null, a string, a list)
            # must not crash into aiohttp's HTML 500 — the terse-JSON
            # error contract holds on every path.
            body = {}
        size = body.get("size", [512, 512])
        # Shape via the shared pair judgment (ops.common.num_pair);
        # bounds stay here with their actionable message.
        if not (
            isinstance(size, list)
            and ops_common.num_pair(size, integral=True)
            and all(16 <= v <= _SCREENSHOT_MAX_DIM for v in size)
        ):
            return web.json_response(
                {"error": {"code": "bad_size", "message": f"size must be [w, h] integers in 16..{_SCREENSHOT_MAX_DIM}"}},
                status=400,
            )
        cam_args = {
            k: body.get(k)
            for k in ("view", "azimuth", "elevation", "distance", "target")
        }
        if all(cam_args[k] is None for k in ("view", "azimuth", "elevation")):
            cam_args["view"] = "iso"
        try:
            # Validation pass with placeholder bounds: camera errors must
            # fail fast (before no_viewer and the evaluation), and the
            # real bounds exist only after evaluating. Pure math, rerun
            # with real bounds below.
            camera_mod.resolve(**cam_args, bounds=([0.0] * 3, [0.0] * 3))
        except MoraineError as e:
            return web.json_response({"error": e.to_dict()}, status=400)
        try:
            # The addressed project (DESIGN 7.2): the caller's, not the
            # displayed one - several sessions share one viewer and each
            # shoots its own asset through it.
            if isinstance(body.get("project"), str):
                await self._refresh_roster()
            asset_path, project = self._resolve_project(body.get("project"))
        except MoraineError as e:
            return web.json_response({"error": e.to_dict()}, status=400)
        shot["project"] = project
        # Values the caller gives for the project's knobs (DESIGN 6): the
        # shot is then of the asset under them, built as a consumer's
        # offer and evaluated into the preview bucket, and says so.
        params = body.get("params")
        if params is not None and not isinstance(params, dict):
            return web.json_response(
                {"error": {"code": "bad_params", "message": "params must be an object of knob id to value"}},
                status=400,
            )
        if params and asset_mod.is_ir_file(asset_path):
            err = MoraineError(
                "param_needs_recipe",
                f"params are offered to a recipe's knobs, and '{asset_path.name}' "
                f"is an IR, rendered as it is",
            )
            return web.json_response({"error": err.to_dict()}, status=400)
        if not self.clients:
            return web.json_response(
                {
                    "error": {
                        "code": "no_viewer",
                        "message": f"no viewer connected; open http://{req.host}/ in a browser, then retry",
                    }
                },
                status=503,
            )
        def _eval_and_stamp() -> tuple[dict, str]:
            # File-state pass plus its revision stamp, off the loop: a
            # cold-cache evaluation here would otherwise stall every
            # WebSocket send and the watch broadcast for its duration.
            if params:
                given = target_mod.open_target(
                    asset_path, offered=dict(params), write=False, ranged_offers=False,
                )
                results, ev, opened = self._evaluate_results(
                    dict(params), opened=given, asset_path=asset_path,
                )
            else:
                results, ev, opened = self._evaluate_results(asset_path=asset_path)
            return results, opened.revision_hash(
                ev.asset_files, root=self.watch_root, also={asset_path},
            )

        try:
            async with self._offloop_eval_gate:
                results, asset_hash = await asyncio.to_thread(_eval_and_stamp)
        except MoraineError as e:
            await self._maybe_evict()
            return web.json_response({"error": e.to_dict()}, status=500)
        except Exception as e:
            # Never serve stale pixels on failure: the error is the
            # response (same wrapping rationale as _reload_and_broadcast).
            await self._maybe_evict()
            err = MoraineError(
                "internal",
                f"screenshot evaluation failed unexpectedly ({type(e).__name__}: {e})",
            )
            return web.json_response({"error": err.to_dict()}, status=500)
        await self._maybe_evict()
        stats = {name: r["stats"] for name, r in results.items()}
        cam = camera_mod.resolve(**cam_args, bounds=camera_mod.asset_bounds(stats))
        request_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending_shots[request_id] = fut
        await self._broadcast(
            {
                "type": "screenshot_request",
                "request_id": request_id,
                "protocol": _SCREENSHOT_PROTOCOL,
                "glb_urls": {
                    name: f"/{'preview' if r.get('bucket') else 'cache'}/{r['glb_file']}"
                    for name, r in results.items()
                    if "glb_file" in r
                },
                "camera": {k: cam[k] for k in ("position", "target", "up", "fov")},
                "size": size,
            }
        )
        try:
            data = await asyncio.wait_for(fut, timeout=SCREENSHOT_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.pending_shots.pop(request_id, None)
            if request_id in self.stale_shots:
                self.stale_shots.discard(request_id)
                return web.json_response(
                    {
                        "error": {
                            "code": "stale_client",
                            "message": "a connected viewer tab is running an "
                                       "old client - reload every open viewer "
                                       "tab, then retry",
                        }
                    },
                    status=502,
                )
            return web.json_response(
                {"error": {"code": "screenshot_timeout", "message": "viewer did not respond"}},
                status=504,
            )
        self.stale_shots.discard(request_id)
        if data.get("error"):
            return web.json_response(
                {
                    "error": {
                        "code": "screenshot_failed",
                        "message": f"viewer could not render: {data['error']}",
                    }
                },
                status=502,
            )
        png_b64 = data.get("png", "")
        try:
            base64.b64decode(png_b64, validate=True)
        except (binascii.Error, ValueError):
            return web.json_response(
                {"error": {"code": "bad_screenshot", "message": "viewer sent invalid png data"}},
                status=502,
            )
        try:
            # Workspace-relative like the content hash's framing: reports
            # get quoted into committed journals, and an absolute path
            # would leak the local machine layout there.
            asset_rel = asset_path.resolve().relative_to(self.watch_root).as_posix()
        except ValueError:  # pragma: no cover - watched assets live under root
            asset_rel = asset_path.name
        meta = {
            "project": project,
            "asset": asset_rel,
            "asset_hash": asset_hash,
            "moraine_version": dist_version(),
            "watch": "dead" if self.watch_dead else "alive",
            "camera": {
                k: cam[k] for k in ("azimuth", "elevation", "distance", "target")
            },
            "size": size,
        }
        if self.listen_rearmed:
            meta["listen_rearmed"] = self.listen_rearmed
        if params:
            meta["params_given"] = params
        if cam_args["view"] is not None:
            meta["view"] = cam_args["view"]
        shot.update(
            asset_hash=asset_hash, size=size, bytes=len(png_b64) * 3 // 4,
            **({"view": cam_args["view"]} if cam_args["view"] is not None else {}),
        )
        return web.json_response({"meta": meta, "png": png_b64})

    # -- app ----------------------------------------------------------------

    async def _open_viewer(self) -> None:
        """Open the human's default browser on the viewer, but only when
        nobody is watching (DESIGN 7.1): a client that connects while
        the grace window runs is a tab already showing this server —
        the ordinary case after a restart, since tabs reconnect on
        their own — and opening another would be the second tab the
        rule exists to prevent. Sampled rather than checked once at the
        end, so the window can be long enough for a slow retry without
        delaying the open when a tab answers early. A host with no
        browser gets nothing, silently: the open is a convenience for a
        co-present human, never a requirement of the loop. The
        ``viewer`` event says which way it went, so the agent knows at
        serve time whether a human can see anything."""
        deadline = time.monotonic() + self.viewer_open_grace_s
        while time.monotonic() < deadline:
            if self.clients:
                break
            await asyncio.sleep(min(_VIEWER_OPEN_POLL_S, self.viewer_open_grace_s))
        if self.clients:
            _log("viewer", state="connected")
            return
        url = f"http://127.0.0.1:{self.port}/"
        try:
            # Off the loop: the launcher is bounded but not instant, and
            # the first reload must not wait behind it.
            opened = await asyncio.to_thread(open_browser, url)
        except Exception as e:
            _log("viewer", state="unavailable", error=f"{type(e).__name__}: {e}")
            return
        _log("viewer", state="opened" if opened else "unavailable")

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_post("/api/screenshot", self.screenshot)
        app.router.add_get("/api/status", self.status_api)
        app.router.add_get("/api/pins", self.pins_api)
        app.router.add_post("/api/pins", self.pins_api)
        app.router.add_delete("/api/pins", self.pins_api)
        app.router.add_static("/viewer/", VIEWER_DIR)
        app.router.add_get("/cache/{filename}", self.cache_file)
        app.router.add_get("/preview/{filename}", self.preview_file)

        async def cache_headers(req: web.Request, resp: web.StreamResponse) -> None:
            resp.headers["Cache-Control"] = cache_control(req.path, resp.status)

        app.on_response_prepare.append(cache_headers)

        async def startup(_app):
            await self._purge_preview_bucket()
            await self._reload_and_broadcast(trigger="startup")
            session.record(
                "serve_start", root=str(self.watch_root), port=self.port,
                project=self.project, asset_hash=self.asset_hash,
                **({"projects": list(self.workspace.projects)} if self.workspace else {}),
            )
            _app["watch_task"] = asyncio.create_task(self._watch(_app))
            # serve_start above is on disk, so the initial build already
            # has a session to report; a write landing between this read
            # and the log watch starting is caught up by the next write.
            await self._refresh_timeline()
            _app["timeline_task"] = asyncio.create_task(self._watch_log(_app))
            # The site listens right after on_startup returns, so the
            # grace the task sleeps through is measured from just before
            # the port opens.
            if self.open_browser:
                _app["viewer_task"] = asyncio.create_task(self._open_viewer())

        async def cleanup(_app):
            _app["watch_task"].cancel()
            _app["timeline_task"].cancel()
            if "viewer_task" in _app:
                _app["viewer_task"].cancel()

        async def shutdown(_app):
            # Open WebSockets keep their handlers alive, and aiohttp's
            # graceful shutdown waits for handlers (60 s by default) — with
            # a viewer tab connected, Ctrl+C would hang until force-killed.
            for ws in list(self.clients):
                await ws.close(code=WSCloseCode.GOING_AWAY, message=b"server_shutdown")
            await self._purge_preview_bucket()
            self._settle_unchecked()
            session.record("serve_stop", root=str(self.watch_root), port=self.port)

        app.on_startup.append(startup)
        app.on_shutdown.append(shutdown)
        app.on_cleanup.append(cleanup)
        return app


def cache_control(path: str, status: int) -> str:
    """The Cache-Control header per route and status. Viewer files must
    revalidate every load (browsers' heuristic caching would otherwise
    pin stale code for hours after an edit). Cache and preview-bucket
    entries are content-addressed, hence immutable — but only when
    served: a 404 on one of those paths carrying ``immutable`` is stored
    by the browser for a year under the very name the record will come
    back under (the names are content-addressed, so a re-evaluation
    restores exactly the URL that failed), and the tab then answers
    itself 404 across reconnects and server restarts on the same origin;
    only a new port ever recovered. A miss is never cacheable; a served
    body is, whole (200), partial (206), or revalidated (304 — a
    ``no-store`` on the 304 would evict the stored copy it confirms)."""
    if path.startswith(("/cache/", "/preview/")):
        return "public, max-age=31536000, immutable" if status in (200, 206, 304) else "no-store"
    return "no-cache"


def run(
    asset_path: str | Path,
    port: int = DEFAULT_PORT,
    workspace: workspace_mod.Workspace | None = None,
    project: str | None = None,
    open_browser: bool = True,
) -> None:
    server = MoraineServer(Path(asset_path), workspace=workspace, project=project)
    server.port = port
    server.open_browser = open_browser
    app = server.build_app()

    # Logged from the server's own state, the roster the viewer will see,
    # once the site listens: a refused port reports port_in_use alone,
    # before anything is evaluated. watch_root makes the
    # awatch scope diagnosable: since cross-project refs (DESIGN 4.6) it
    # can be wider than the served project.
    def announce() -> None:
        _log(
            "serving",
            url=f"http://127.0.0.1:{port}/",
            asset=server.asset_path.name,
            cache=str(cache_dir()),
            watch_root=str(server.watch_root),
            **({"projects": list(server.workspace.projects)} if server.workspace else {}),
        )

    try:
        asyncio.run(server._serve(app, port, on_bound=announce))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # Ctrl+C, or the termination signal canceling the run task
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
            raise MoraineError(
                "port_in_use",
                f"port {port} is already in use - a 'moraine serve' is likely "
                f"running there: use it (every command talks to it by default), "
                f"or pass --port <other> for a second server",
            ) from e
        raise
    except OverflowError as e:
        # bind() on an out-of-range port (the CLI validates first; this
        # is the backstop for programmatic callers).
        raise MoraineError(
            "bad_port", f"--port must be in 1..65535, got {port}"
        ) from e
    _log("stopped")
