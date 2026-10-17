"""Session log (DESIGN 6): the runtime's record of what its entry points
did, and the summary report derived from it.

Writer side — one JSON object per line, appended to a machine-level
day file (``<log dir>/<UTC date>.jsonl``; ``MORAINE_LOG_DIR`` overrides
the directory, empty turns logging off). The log holds no asset
content: hashes, names, counts, outcomes, and timings only. Recording
never changes a command's stdout or exit code — an unwritable log
directory costs one stderr line per process and nothing else. ``root``
(the enclosing workspace root, or the project directory outside a
workspace) is absolute here because the file is machine-local; the
report that is meant to be quoted elsewhere carries names only.

Reader side — ``summarize`` derives *sessions* (event runs separated by
less than a gap) and, per project, *rounds* (one asset revision — a
``asset_hash`` — carrying at least one verify command; revisions never
verified fold into the next). "Session" here is the report's grouping
of timestamps, unrelated to the cache's per-process write counter.

The log is a rolling window, not an archive: day files older than the
retention window are retired the first time a new day file is created
(``MORAINE_LOG_KEEP_DAYS``, default 30; ``0`` keeps everything). The
durable evidence is the summary report archived at recording time. The
log records what the user is shown: a knob scrubbed in the viewer and a
pin placed there leave no event, an accept does (it reaches the file
and closes a round).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from platformdirs import user_log_path

from .. import dist_version

LOG_DIR_ENV = "MORAINE_LOG_DIR"
KEEP_DAYS_ENV = "MORAINE_LOG_KEEP_DAYS"
DEFAULT_KEEP_DAYS = 30
# The only name retirement may touch: a day file, matched whole. A
# foreign file in an overridden log dir must never be deletable by name
# accident — the name gate plus the first-line provenance check below.
_DAY_FILE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\.jsonl")
# Commands whose run counts as verification of the current revision
# (the authoring loop's tiers plus the read-side commands that evaluate).
VERIFY_COMMANDS = frozenset({"validate", "stats", "screenshot", "eval", "query", "export"})
DEFAULT_GAP_MINUTES = 30

_version: str | None = None
_warned = False
_notes: dict = {}
_project_notes: dict[str, dict] = {}


# -- writer -------------------------------------------------------------------


def log_dir() -> Path | None:
    """The log directory, or None when logging is off (empty override)."""
    override = os.environ.get(LOG_DIR_ENV)
    if override is not None:
        return Path(override) if override else None
    return user_log_path("moraine", appauthor=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def record(event: str, **fields) -> None:
    """Append one event; never raises."""
    global _version, _warned
    d = log_dir()
    if d is None:
        return
    if _version is None:
        _version = dist_version()
    now = _now_iso()
    row = {"t": now, "v": _version, "pid": os.getpid(), "event": event}
    row.update({k: v for k, v in fields.items() if v is not None})
    try:
        line = json.dumps(row, separators=(",", ":"), default=str)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{now[:10]}.jsonl"
        fresh_day = not path.exists()
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        if fresh_day:
            # A new day started: the once-a-day chokepoint for retiring
            # day files that fell out of the rolling window. After the
            # write, so retirement can never delay or break the event.
            _retire(d, today=now[:10])
    except (OSError, TypeError, ValueError) as e:
        if not _warned:
            _warned = True
            print(
                f"session log skipped: {e.__class__.__name__}: {e} "
                f"(set {LOG_DIR_ENV} to a writable directory, or empty to disable)",
                file=sys.stderr,
            )


def _keep_days() -> int:
    raw = os.environ.get(KEEP_DAYS_ENV)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_KEEP_DAYS


def _is_moraine_log(f: Path) -> bool:
    """Per-file provenance: the first line carries the exact row shape
    ``record`` always writes first (``t``/``v``/``pid``/``event``, typed
    and with a parseable timestamp), so a failure here means the file is
    someone else's — never delete it. The full shape is the gate on
    purpose: an overridden log dir may be shared with another tool's
    day-rotated JSONL, and a generic two-key telemetry row ({"event":
    ..., "v": ...}) must stay undeletable."""
    try:
        with open(f, encoding="utf-8") as fh:
            row = json.loads(fh.readline())
    except (OSError, ValueError):
        return False
    return (
        isinstance(row, dict)
        and isinstance(row.get("t"), str)
        and _parse_t(row["t"]) is not None
        and isinstance(row.get("v"), str)
        and isinstance(row.get("pid"), int)
        and not isinstance(row.get("pid"), bool)
        and isinstance(row.get("event"), str)
    )


def _retire(d: Path, today: str) -> None:
    """Drop day files older than the retention window. Deletion is
    deliberately narrow (the §4.5 discipline, scaled down): only files
    directly in the log dir whose whole name is a day file's, whose
    date is before the cutoff — today's file never qualifies — and
    whose first line proves Moraine wrote them. Anything else is left
    where it is, and every failure is ignored: retirement must never
    surface through a command."""
    keep = _keep_days()
    if keep <= 0:
        return
    cutoff = (date.fromisoformat(today) - timedelta(days=keep)).isoformat()
    try:
        candidates = list(d.glob("*.jsonl"))
    except OSError:
        return
    for f in candidates:
        if not _DAY_FILE_RE.fullmatch(f.name) or f.stem >= cutoff:
            continue
        if not _is_moraine_log(f):
            continue
        try:
            f.unlink()
        except OSError:
            continue


def reset() -> None:
    """Clear the per-command annotation state (main() start). The
    one-warning-per-process flag is deliberately not reset here."""
    _notes.clear()
    _project_notes.clear()


def annotate(**fields) -> None:
    """Stash fields for the process's one ``command`` event."""
    _notes.update({k: v for k, v in fields.items() if v is not None})


def annotate_project(name: str, **fields) -> None:
    """Per-project fields for a workspace sweep's ``command`` event."""
    _project_notes.setdefault(name, {}).update(
        {k: v for k, v in fields.items() if v is not None}
    )


def take_annotations() -> dict:
    out = dict(_notes)
    if _project_notes:
        out["projects"] = {k: dict(v) for k, v in _project_notes.items()}
    _notes.clear()
    _project_notes.clear()
    return out


# -- reader -------------------------------------------------------------------


# Parsed day files memoized by (size, mtime) plus the byte offset of
# the last complete line: the server re-reads the log on every
# log-directory change (the timeline watch, DESIGN 7.2). Every day file
# but the appended-to current one is immutable, and the current one is
# append-only (the writer appends; retirement only unlinks whole
# files), so a growth re-parses the appended tail alone — without
# that, each recorded event would re-parse the whole day.
_parse_cache: dict[Path, tuple[tuple[int, int], list[dict], int]] = {}
_PARSE_CACHE_MAX = 400  # well above the default retention window in files


def _parse_file(f: Path) -> list[dict]:
    """One day file's parseable events; malformed lines, torn writes,
    and non-object rows are skipped (the reader is forward-compatible).
    Memoized, tail-extending for the growing current file (a shrunk
    file — external truncation — falls back to a full parse) — callers
    must not mutate the returned events."""
    try:
        st = f.stat()
    except OSError:
        return []
    key = (st.st_size, st.st_mtime_ns)
    hit = _parse_cache.get(f)
    if hit is not None and hit[0] == key:
        return hit[1]
    events: list[dict] = []
    start = 0
    if hit is not None and st.st_size >= hit[2] > 0:
        events = list(hit[1])
        start = hit[2]
    try:
        with open(f, "rb") as fh:
            if start:
                fh.seek(start)
            data = fh.read()
    except OSError:
        return events
    # Only complete lines: a torn last line stays unparsed (and outside
    # the consumed offset) until its newline lands, exactly as the full
    # re-parse skipped it.
    tail = data.rfind(b"\n") + 1
    for line in data[:tail].decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("t"), str) and "event" in row:
            events.append(row)
    if len(_parse_cache) >= _PARSE_CACHE_MAX:
        _parse_cache.clear()
    _parse_cache[f] = (key, events, start + tail)
    return events


def read_events(d: Path) -> list[dict]:
    """Every parseable event under ``d``, in time order."""
    events: list[dict] = []
    try:
        files = sorted(d.glob("*.jsonl"))
    except OSError:
        return events
    for f in files:
        events.extend(_parse_file(f))
    events.sort(key=lambda r: r["t"])
    return events


def _names(ev: dict, name: str) -> bool:
    """True when the event names ``name`` as its project — singly or in
    a sweep's per-project map. Shared by the summarizer's narrowing and
    ``read_recent``'s stop condition: the two must apply the same
    predicate or the reader stops before the report's filter is
    satisfied."""
    return ev.get("project") == name or name in (ev.get("projects") or {})


def read_recent(
    d: Path, root: Path, *,
    gap_minutes: float, last: int, project: str | None = None,
) -> list[dict]:
    """Enough events, newest day files first, to report the newest
    ``last`` sessions under ``root`` (narrowed to ``project`` when
    given) — the log retains a month of day files and the default
    report wants only the latest session, so reading stops once one
    session more than requested exists (older events can only extend or
    merge sessions older than the requested ones, never the newest
    ``last``). The stop condition must count sessions under the same
    narrowing the summarizer applies: counting root-wide while the
    report is project-narrowed would stop at the newest day files even
    when the named project's sessions live only in older ones."""
    try:
        files = sorted(d.glob("*.jsonl"), reverse=True)
    except OSError:
        return []
    # Each file is looked at once: a root with a single session reads
    # the whole window, and the server reads it again on every event
    # (the timeline, DESIGN 7.2).
    days: list[list[dict]] = []
    relevant: list[dict] = []
    for f in files:
        found = _parse_file(f)
        days.append(found)
        mine = [
            e for e in found
            if _under(root, e.get("root"))
            and (project is None or _names(e, project))
        ]
        if not mine:
            continue  # the sessions counted so far stand
        relevant = mine + relevant
        relevant.sort(key=lambda r: r["t"])
        if len(_split_sessions(relevant, gap_minutes)) > last:
            break
    events = [e for found in reversed(days) for e in found]
    events.sort(key=lambda r: r["t"])
    return events


def latest_report(
    root: Path, *,
    gap_minutes: float = DEFAULT_GAP_MINUTES, last: int = 1,
    project: str | None = None,
) -> dict | None:
    """The newest ``last`` sessions' report for ``root``, or None when
    logging is off or nothing is under the root — the one reader shared
    by ``moraine sessions`` and the server's timeline, so "the log's
    other reader, same summarizer" (DESIGN 6) holds structurally."""
    d = log_dir()
    if d is None:
        return None
    events = read_recent(
        d, root, gap_minutes=gap_minutes, last=last, project=project,
    )
    report = summarize(
        events, root, gap_minutes=gap_minutes, last=last, project=project,
    )
    return report if report["sessions"] else None


def _parse_t(t: str) -> datetime | None:
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        return None


def _split_sessions(events: list[dict], gap_minutes: float) -> list[list[dict]]:
    """Time-ordered events -> runs separated by less than the gap."""
    sessions: list[list[dict]] = []
    prev: datetime | None = None
    for ev in events:
        t = _parse_t(ev["t"])
        if t is None:
            continue
        if prev is None or (t - prev).total_seconds() > gap_minutes * 60:
            sessions.append([])
        sessions[-1].append(ev)
        prev = t
    return sessions


def _under(root: Path, candidate: str | None) -> bool:
    # Both sides are resolved paths (the writer resolves ``root``); Path
    # comparison is case-insensitive where the filesystem is.
    if isinstance(candidate, str):
        return bool(candidate) and _path_under(root, candidate)
    return bool(candidate) and Path(candidate).is_relative_to(root)


@lru_cache(maxsize=4096)
def _path_under(root: Path, candidate: str) -> bool:
    # A log holds thousands of events under a few hundred roots, and
    # the comparison builds a path each time it is asked.
    return Path(candidate).is_relative_to(root)


def _per_project(ev: dict) -> list[tuple[str, dict]]:
    """Explode one event into (project, view) pairs — a sweep command
    yields one per project; anything naming a single project yields one."""
    if isinstance(ev.get("projects"), dict):
        out = []
        for name, sub in ev["projects"].items():
            view = dict(ev)
            view.pop("projects", None)
            # Whole-command quantities: a sweep's wall time and output
            # volume belong to the command once, not to every project.
            view.pop("ms", None)
            view.pop("out_bytes", None)
            if isinstance(sub, dict):
                view.update(sub)
            out.append((name, view))
        return out
    if isinstance(ev.get("project"), str):
        return [(ev["project"], ev)]
    return []


def _new_round(h: str | None) -> dict:
    return {
        "hash": h, "commands": {}, "validate_ok": None, "failed_checks": [],
        "ended_by": None, "changed_lines": 0, "ms": 0, "shots": 0,
        "_verified": False,
    }


def _project_summary(events: list[dict]) -> dict:
    """Rounds and counts for one project's time-ordered events."""
    rounds: list[dict] = []
    cur: dict | None = None
    nodes = None
    reloads = accepts = eval_errors = shots = shot_bytes = 0
    unverified_lines = 0
    for ev in events:
        kind = ev.get("event")
        h = ev.get("asset_hash") if not ev.get("partial") else None
        if h is not None and (cur is None or h != cur["hash"]):
            if cur is not None and cur["_verified"] and cur["hash"] is not None:
                rounds.append(cur)
                cur = _new_round(h)
            elif cur is None:
                cur = _new_round(h)
            else:
                # A draft revision never verified, or hash-less events
                # before the first revision: fold forward.
                cur["hash"] = h
        if cur is None:
            # Hash-less events before any revision (a partial eval, a
            # failed command) collect here and fold into the first
            # revision's round; alone they open none (DESIGN 6).
            cur = _new_round(None)
        if kind == "command":
            cmd = ev.get("cmd")
            if cmd == "screenshot":
                # The render is owned by the server's ``screenshot``
                # event; a project-carrying screenshot *command* event
                # (written by earlier builds) would double-count it.
                continue
            cur["commands"][cmd] = cur["commands"].get(cmd, 0) + 1
            cur["ms"] += ev.get("ms") or 0
            summary = ev.get("summary") or {}
            if cmd in VERIFY_COMMANDS:
                cur["_verified"] = True
                cur["ended_by"] = cmd
            if cmd == "validate":
                cur["validate_ok"] = bool(ev.get("ok"))
                cur["failed_checks"] = list(summary.get("failed_checks") or [])
            if cmd in ("stats", "eval") and isinstance(summary.get("nodes"), int):
                nodes = summary["nodes"]
            if ev.get("ok") is False and ev.get("code"):
                eval_errors += 1
        elif kind == "reload":
            reloads += 1
            cur["changed_lines"] += ev.get("changed_lines") or 0
            if ev.get("ok") is False:
                eval_errors += 1
        elif kind == "accept":
            # The acceptance act itself (DESIGN 7.2 viewer accept): the
            # count a session's economy reads as "picks landed without
            # an agent round trip". A failed accept is an error like a
            # failed reload; the reload the write triggers is counted
            # by its own event. An accept that carries the check of the
            # accepted state verified that revision (DESIGN 6).
            if ev.get("ok"):
                accepts += 1
                summary = ev.get("summary")
                if isinstance(summary, dict):
                    cur["_verified"] = True
                    cur["ended_by"] = "accept"
                    cur["validate_ok"] = bool(summary.get("validate_ok"))
                    cur["failed_checks"] = list(summary.get("failed") or [])
            else:
                eval_errors += 1
        elif kind == "screenshot":
            if ev.get("ok"):
                shots += 1
                shot_bytes += ev.get("bytes") or 0
                cur["shots"] += 1
                cur["_verified"] = True
                cur["ended_by"] = "screenshot"
            else:
                eval_errors += 1
            cur["commands"]["screenshot"] = cur["commands"].get("screenshot", 0) + 1
    if cur is not None:
        if cur["_verified"] and cur["hash"] is not None:
            rounds.append(cur)
        else:
            unverified_lines = cur["changed_lines"]
    for r in rounds:
        r.pop("_verified", None)
    counts = {
        "rounds": len(rounds),
        "validate": sum(r["commands"].get("validate", 0) for r in rounds),
        "stats": sum(r["commands"].get("stats", 0) for r in rounds),
        "shots": shots,
        "shot_bytes": shot_bytes,
        "reloads": reloads,
        "changed_lines": sum(r["changed_lines"] for r in rounds) + unverified_lines,
        "accepts": accepts,
        "eval_errors": eval_errors,
    }
    if nodes is not None:
        counts["nodes"] = nodes
    counts["rounds_detail"] = rounds
    return counts


def _session_summary(events: list[dict]) -> dict:
    start, end = events[0]["t"], events[-1]["t"]
    t0, t1 = _parse_t(start), _parse_t(end)
    minutes = round((t1 - t0).total_seconds() / 60, 1) if t0 and t1 else None
    commands: dict[str, int] = {}
    out_bytes = 0
    per_project: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("event") == "command":
            cmd = ev.get("cmd") or "?"
            commands[cmd] = commands.get(cmd, 0) + 1
            out_bytes += ev.get("out_bytes") or 0
        for name, view in _per_project(ev):
            per_project.setdefault(name, []).append(view)
    projects = {name: _project_summary(evs) for name, evs in sorted(per_project.items())}
    session = {
        "start": start, "end": end, "minutes": minutes, "events": len(events),
        "commands": commands, "out_bytes": out_bytes,
        "rounds": sum(p["rounds"] for p in projects.values()),
        "shots": sum(p["shots"] for p in projects.values()),
        "accepts": sum(p["accepts"] for p in projects.values()),
        "reloads": sum(p["reloads"] for p in projects.values()),
        "changed_lines": sum(p["changed_lines"] for p in projects.values()),
        "eval_errors": sum(p["eval_errors"] for p in projects.values()),
        "projects": projects,
    }
    return session


def summarize(
    events: list[dict], root: Path, *,
    gap_minutes: float = DEFAULT_GAP_MINUTES, last: int | None = 1,
    project: str | None = None,
) -> dict:
    """The report: sessions under ``root`` (a workspace root, a project
    directory, or any directory above them), newest last. ``project``
    narrows a workspace's sessions to one member project's events.
    Root-less command events (``docs``, ``cache``, ``init``,
    ``screenshot``) join the session whose span, extended by the gap,
    contains them — as ambient counts only: they carry no project, and
    one that does (another server may have answered the port) is
    unattributable and dropped rather than guessed at. Names only —
    never a path."""
    rooted = [
        e for e in events
        if _under(root, e.get("root")) and (project is None or _names(e, project))
    ]
    rootless = [
        e for e in events
        if not e.get("root") and e.get("event") == "command"
        and "project" not in e and "projects" not in e
    ]
    sessions = _split_sessions(rooted, gap_minutes)
    # Attach root-less command events by time — a session's span plus
    # the gap after it (spans fixed before any attachment; sessions are
    # more than a gap apart, so the windows never overlap) — then
    # restore time order.
    spans = [
        (_parse_t(s[0]["t"]), _parse_t(s[-1]["t"]) + timedelta(minutes=gap_minutes))
        for s in sessions
    ]
    for ev in rootless:
        t = _parse_t(ev["t"])
        if t is None:
            continue
        for s, (t0, t1) in zip(sessions, spans):
            if t0 <= t <= t1:
                s.append(ev)
                break
    for s in sessions:
        s.sort(key=lambda r: r["t"])
    if last is not None:
        sessions = sessions[-last:] if last > 0 else []
    summaries = [_session_summary(s) for s in sessions]
    totals = {
        "sessions": len(summaries),
        "minutes": round(sum(s["minutes"] or 0 for s in summaries), 1),
        "rounds": sum(s["rounds"] for s in summaries),
        "shots": sum(s["shots"] for s in summaries),
        "accepts": sum(s["accepts"] for s in summaries),
        "changed_lines": sum(s["changed_lines"] for s in summaries),
        "out_bytes": sum(s["out_bytes"] for s in summaries),
        "commands": {},
    }
    for s in summaries:
        for k, n in s["commands"].items():
            totals["commands"][k] = totals["commands"].get(k, 0) + n
    return {
        "root": root.name or "/",
        **({"project": project} if project is not None else {}),
        "gap_minutes": gap_minutes,
        "sessions": summaries,
        "totals": totals,
    }
