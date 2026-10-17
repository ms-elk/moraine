"""Cache size management: LRU eviction and session write accounting
(DESIGN 4.5).

The machine-level cache (paths.cache_dir) grows without bound through
iteration garbage, so a size cap with LRU eviction keeps it bounded.
The filesystem stays the only ledger: recency is the mtime of a
record's ``.meta.json`` (the evaluator touches it on every cache hit),
and eviction is one ``os.scandir`` pass — no index database.

Eviction unit is the record group sharing one 24-hex key prefix
(``.npz`` + ``.meta.json`` + derived GLB/PNG), deleted meta-first —
the reverse of write-visibility order — so no half-visible record
state can exist. Every entry re-derives from the recipe, so eviction
is safe by construction.

The ephemeral preview bucket (``paths.preview_dir``, DESIGN 4.5) is the
one subdirectory this module attributes to Moraine: same filename
grammar inside, outside the cap and the LRU (``scan`` ignores
directories, so bucket files can never be evicted), reported separately
by ``stats``, emptied by ``purge_preview`` (the server's start/stop
hygiene) and by ``clear``.

Deletion safety: the cache dir may be shared (``MORAINE_CACHE_DIR``
can point anywhere), so deletion is allowlist-only. Every cache unlink
tool-wide — the evaluator's corrupt-record recovery included — goes
through ``unlink_cache_file``, which refuses anything that is not a
direct child of the cache dir or its preview bucket whose full name
matches the writer's naming grammar, current or registered legacy.
Files this module does not recognize are reported, never removed.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.paths import (
    _TAG_BODY,
    _TAG_MORAINE_PREFIX,
    PORTABLE_NAME_FRAGMENT,
    PREVIEW_DIR_NAME,
    TAG_NAME,
    cache_dir,
    is_redirect,
)
from ..errors import MoraineError

DEFAULT_MAX_BYTES = 4 * 1024**3  # "a few GB" (DESIGN 4.5); 0 = unlimited
EVICT_TO = 0.9  # hysteresis: evict down to this fraction of the cap
FRESH_S = 3600.0  # freshness guard: skip groups touched within the last hour


def _fresh(now: float, mtime: float) -> bool:
    """The one freshness predicate (groups and in-flight tmp files alike).
    Symmetric window, not one-sided: "now - mtime <= FRESH_S" would treat
    any future mtime (clock skew, coarse time.time() vs precise filesystem
    stamps) as fresh forever, making the entry unreclaimable and the cap
    unenforceable on skewed volumes. Strict <, so FRESH_S = 0
    deterministically means "no protection" even when a just-written mtime
    lands on the same clock tick as now."""
    return abs(now - mtime) < FRESH_S

# The cache filename grammar. This module is its single regex source:
# app.server._CACHE_FILE_RE composes from these fragments, and the
# f-strings that WRITE the names (cache.records' paths, core.atomic's
# ".tmp<pid>-<n>") carry cross-references here — change any of them
# together.
# The output-name charset is enforced by the asset loader.
KEY_FRAGMENT = r"[0-9a-f]{24}"
_OUTPUT_FRAGMENT = PORTABLE_NAME_FRAGMENT
GLB_FRAGMENT = rf"g\d+-{_OUTPUT_FRAGMENT}\.glb"
PNG_FRAGMENT = rf"i\d+-{_OUTPUT_FRAGMENT}\.png"

# Every name Moraine writes into the cache dir, full-match: records
# ("<key>.npz" / "<key>.meta.json"), derived artifacts, each optionally
# with the in-flight ".tmp<pid>-<n>" suffix (the older ".tmp<pid>" form
# stays recognized by the same pattern). Deletion candidates come only
# from this allowlist plus the registered legacy grammars below — one
# recognition seam, ``_classify``. \Z, not $: "$" also matches before a
# trailing newline, which would admit a foreign POSIX filename ending
# in "\n".
_NAME_RE = re.compile(
    rf"^({KEY_FRAGMENT})"
    rf"(\.npz|\.meta\.json|-{GLB_FRAGMENT}|-{PNG_FRAGMENT})"
    rf"(\.tmp\d+(?:-\d+)?)?\Z"
)

# The grammar above is FROZEN on-disk format, like the tag body in
# paths.py (both pinned byte-exact by test): files written under an old
# grammar must stay recognizable as Moraine's own, or every cache dir
# that predates the change turns permanently foreign — provenance never
# reaches "pure" again, so the tag is never stamped and the deletion
# gate never opens. Any change to the written names therefore ships
# with the outgoing pattern appended here. A registered pattern must
# capture the 24-hex key as group 1 (checked at registration below) and
# must cover the name's in-flight ".tmp<pid>-<n>" variants — every cache
# write goes through core.atomic's temp name, so crash leftovers exist
# under any grammar. Matching is by fullmatch, so a pattern cannot widen the
# allowlist past its own text. Legacy names count as Moraine content
# for provenance, join their key's record group in scan (tmp forms join
# the stale-tmp reap instead), and pass the deletion choke point; they
# are never written again. Empty today: the one grammar change so far
# (the g<N>/i<N> generation prefix) predates any release and is
# deliberately not grandfathered (DESIGN 4.5).
_LEGACY_NAME_RES: tuple[re.Pattern[str], ...] = ()


def _check_legacy_patterns() -> None:
    """Registration-time contract check: a pattern that cannot yield the
    key would otherwise surface as an IndexError deep in the deletion
    machinery, on the first machine holding a legacy-named file."""
    for pat in _LEGACY_NAME_RES:
        if pat.groups < 1:
            raise AssertionError(
                f"_LEGACY_NAME_RES pattern {pat.pattern!r} must capture "
                "the 24-hex key as group 1"
            )


_check_legacy_patterns()

# Per-group deletion order: meta first (kills the hit gate, the reverse of
# npz-then-meta write order), then the payload, then derived artifacts —
# legacy-grammar members included, via the .get() default below.
_DELETE_RANK = {".meta.json": 0, ".npz": 1}
# Kind tag for legacy-grammar group members (never a real extension, so
# it can never collide with a _DELETE_RANK key).
_LEGACY_KIND = "legacy"

# In-flight suffix of a legacy name: the current grammar carries the
# suffix as its own capture group, legacy patterns cover it wholesale.
_TMP_SUFFIX_RE = re.compile(r"\.tmp\d+(?:-\d+)?\Z")


def _classify(name: str) -> tuple[str, str, bool] | None:
    """THE recognition seam: (key, kind, is_tmp) for a filename Moraine
    attributes to itself — current grammar or registered legacy — or
    None. Every consumer of the question "is this name Moraine's?"
    (content provenance, scan, the deletion choke point) goes through
    this one function, so the answer cannot drift between the layer
    that tags, the layer that reports, and the layer that deletes."""
    m = _NAME_RE.match(name)
    if m is not None:
        return m.group(1), m.group(2), bool(m.group(3))
    for pat in _LEGACY_NAME_RES:
        lm = pat.fullmatch(name)
        if lm is not None:
            return lm.group(1), _LEGACY_KIND, _TMP_SUFFIX_RE.search(name) is not None
    return None

# TAG_NAME / _TAG_BODY / _TAG_MORAINE_PREFIX live in paths.py (the creator
# stamps at mkdir time) and are re-exported here for the readers below.


# -- session write accounting ------------------------------------------------

# Bytes this process wrote into the cache (evaluator records + derived
# GLB/PNG). Touch-on-hit changes mtimes, not size, so a warm-cache session
# stays at zero and the exit check costs nothing (DESIGN 4.5).
_wrote_lock = threading.Lock()
_wrote = 0
# Per-process memo of dirs whose provenance this process has judged —
# ours or foreign alike, deliberately one set: no code may ever branch
# on which verdict a dir received (the deletion gate reads the
# directory fresh instead), and one set makes that structural. It
# spares the write path its per-write tag read / directory rescan;
# _ensure_tag's docstring carries the staleness invariant.
_judged_dirs: set[Path] = set()


def _tag_state(d: Path) -> str:
    """Classify the dir's CACHEDIR.TAG for the provenance decisions below:
    'ours' — carries Moraine's own tag body prefix (the standard signature
    plus the "created by Moraine" line; the bare signature only proves the
    dir is *a* cache, and another tool's tag must never open Moraine's
    deletion gate); 'torn' — a truncated write of Moraine's own body
    (non-atomic stamp interrupted), repairable by re-stamping; 'foreign' —
    readable but not Moraine's, never overwritten; 'missing'; or
    'unreadable' — a transient read failure (e.g. a backup/AV agent
    holding the file on Windows) that must fail closed for this call but
    never be remembered as a foreign verdict."""
    try:
        data = (d / TAG_NAME).read_bytes()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if data.startswith(_TAG_MORAINE_PREFIX):
        return "ours"
    if _TAG_BODY.startswith(data):
        return "torn"
    return "foreign"


def _content_provenance(d: Path) -> str:
    """Judge the directory's content: 'pure' — every entry is a file
    Moraine's grammar recognizes (current or registered legacy, or
    Moraine's own tag, torn included),
    i.e. the dir was empty before Moraine's own writes; 'foreign' — a
    POSITIVE observation of non-Moraine content (a foreign name,
    subdirectory, symlink, or another tool's tag); 'transient' — the
    scan itself failed (OSError), which is no evidence either way:
    callers must fail closed for this call but never memoize a verdict
    minted from a failed scan."""
    try:
        with os.scandir(d) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    # The ephemeral preview bucket (DESIGN 4.5) is Moraine
                    # infrastructure, not foreign content — but only the
                    # real directory: a symlink or junction named
                    # "preview" is exactly the redirection this gate
                    # exists to refuse. Its contents stay name-gated at
                    # the deletion choke point, so nothing foreign placed
                    # inside it can ever be deleted through Moraine.
                    if (
                        entry.name == PREVIEW_DIR_NAME
                        and entry.is_dir(follow_symlinks=False)
                        and not is_redirect(entry.path)
                    ):
                        continue
                    return "foreign"
                if entry.name == TAG_NAME:
                    st = _tag_state(d)
                    if st in ("ours", "torn"):
                        continue
                    if st == "unreadable":
                        return "transient"
                    return "foreign"
                if _classify(entry.name) is None:
                    return "foreign"
    except OSError:
        return "transient"
    return "pure"


def _ensure_tag(d: Path | None = None) -> None:
    """Stamp the tag the first time this process judges the dir:
    creation-time stamping lives in paths.cache_dir (fresh evidence),
    and this write-path stamp covers a tag that is missing — a
    deleted-and-recreated cache dir heals — or torn (a truncated write
    of Moraine's own body). Stamping stays under content provenance: a
    dir holding anything that is not Moraine's own — a foreign tag
    included — is never tagged (the do-not-back-up marker backup tools
    honor must not spread onto user data), and the missing-tag stamp is
    an exclusive create, so a tag that appears between judgment and
    write is never replaced. Only POSITIVE observations are memoized: a
    transient failure (unreadable tag, failed scan, lost create race,
    unwritable dir) yields no verdict — the next write re-judges.

    The memo is per-process and never revalidated. Do not add
    revalidation or a TTL here — stale entries lose neither safety nor
    recoverability, because the deletion gate never consults the memo
    (it reads the directory fresh on every call): a dir judged "ours"
    whose tag is later deleted externally stays untagged but is still
    correctly gated by the content-provenance fallback, and a wholesale
    dir deletion heals via paths.cache_dir stamping at mkdir (the only
    intra-process re-stamp once the memo holds — the accepted cost is
    bounded to backup tools not skipping the dir until the next
    process, DESIGN 4.5); a stale foreign verdict (dir cleaned up
    mid-process) resolves the way any MORAINE_* misconfiguration does —
    the next process re-judges, exactly as the cache_untagged message
    promises."""
    if d is None:
        d = cache_dir()
    if d in _judged_dirs:
        return
    state = _tag_state(d)
    if state == "ours":
        _judged_dirs.add(d)
        return
    if state == "unreadable":
        return  # transient: fail closed this call, no memo, retry next write
    prov = _content_provenance(d)
    if state == "foreign" or prov == "foreign":
        _judged_dirs.add(d)
        return
    if prov == "transient":
        return  # failed scan is no evidence: no memo, retry next write
    try:
        if state == "missing":
            with open(d / TAG_NAME, "xb") as f:
                f.write(_TAG_BODY)
        else:  # torn: replacing our own truncated body is the point
            (d / TAG_NAME).write_bytes(_TAG_BODY)
    except OSError:
        # Lost the create race or unwritable: no verdict, retry next
        # write (a racing tag is re-read and judged fresh then).
        return
    _judged_dirs.add(d)


def note_write(n: int, d: Path | None = None) -> None:
    """Count session cache bytes and keep the tag ensured. ``d`` is the
    cache dir the caller already resolved via cache_dir() in the same
    operation — passing it spares the per-write re-resolution (env read
    + mkdir probe); omit it only where no resolved dir is at hand."""
    global _wrote
    _ensure_tag(d)
    with _wrote_lock:
        _wrote += n


def session_wrote() -> int:
    with _wrote_lock:
        return _wrote


def take_wrote() -> int:
    """Read and reset the session counter (the eviction trigger consumes it)."""
    global _wrote
    with _wrote_lock:
        n = _wrote
        _wrote = 0
        return n


# -- scanning -----------------------------------------------------------------


def max_bytes() -> int:
    raw = os.environ.get("MORAINE_CACHE_MAX_BYTES")
    if raw is None or raw == "":
        return DEFAULT_MAX_BYTES
    try:
        v = int(raw)
    except ValueError:
        v = -1
    if v < 0:
        raise MoraineError(
            "bad_cache_cap",
            f"MORAINE_CACHE_MAX_BYTES must be a non-negative integer byte "
            f"count (0 = unlimited), got '{raw}'",
        )
    return v


@dataclass
class _Group:
    key: str
    files: list[tuple[str, int, str]] = field(default_factory=list)  # (name, size, kind)
    bytes: int = 0
    mtime: float = 0.0
    has_meta: bool = False


@dataclass
class _Scan:
    dir: Path
    groups: dict[str, _Group]
    tmp: list[tuple[str, int, float]]  # recognized-form in-flight files
    bytes: int  # Moraine-owned bytes (groups + tmp)
    files: int  # Moraine-owned file count
    other_files: int  # unrecognized names: reported, never deleted
    other_bytes: int


def scan() -> _Scan:
    """One os.scandir pass over the cache dir top level (never recursive;
    directories and symlinks are ignored entirely)."""
    d = cache_dir()
    groups: dict[str, _Group] = {}
    tmp: list[tuple[str, int, float]] = []
    other_files = 0
    other_bytes = 0
    total = 0
    count = 0
    with os.scandir(d) as it:
        for entry in it:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue  # racing deletion by a concurrent evictor: benign
            if entry.name == TAG_NAME:
                # Moraine's own tag (torn included) is infrastructure,
                # not cache content. Anything else named CACHEDIR.TAG is
                # exactly what closes the deletion gate, so it must be
                # visible in the "other" report the gate's error points
                # the user at — never silently skipped.
                if _tag_state(d) in ("ours", "torn"):
                    continue
                other_files += 1
                other_bytes += st.st_size
                continue
            c = _classify(entry.name)
            if c is None:
                other_files += 1
                other_bytes += st.st_size
                continue
            key, kind, is_tmp = c
            count += 1
            total += st.st_size
            if is_tmp:
                tmp.append((entry.name, st.st_size, st.st_mtime))
                continue
            g = groups.setdefault(key, _Group(key=key))
            g.files.append((entry.name, st.st_size, kind))
            g.bytes += st.st_size
            if kind == ".meta.json":
                # Group recency = meta mtime (the touch-on-hit target).
                g.mtime = st.st_mtime
                g.has_meta = True
            elif not g.has_meta:
                # Meta-less orphan group: newest member stands in
                # (legacy members never define recency any other way).
                g.mtime = max(g.mtime, st.st_mtime)
    return _Scan(
        dir=d, groups=groups, tmp=tmp, bytes=total, files=count,
        other_files=other_files, other_bytes=other_bytes,
    )


@dataclass
class _PreviewScan:
    dir: Path
    names: list[tuple[str, int]]  # recognized files (tmp forms included)
    bytes: int
    other_files: int  # unrecognized entries: reported, never deleted
    other_bytes: int


def scan_preview() -> _PreviewScan:
    """One os.scandir pass over the ephemeral preview bucket (DESIGN 4.5).
    A missing bucket scans as empty — the dir is created lazily on the
    first preview write and scanning must not create it. In-flight tmp
    forms count as recognized without a freshness guard: the bucket is a
    single-session store, so at purge time (server start/stop) every
    resident file is garbage."""
    d = cache_dir() / PREVIEW_DIR_NAME
    names: list[tuple[str, int]] = []
    other_files = 0
    other_bytes = 0
    total = 0
    if is_redirect(d):
        # A symlink or junction wearing the bucket's name is foreign
        # content (provenance already judges it so), never the bucket:
        # scanning through it would hand the purge a listing of some
        # other tree's files. It scans as empty.
        return _PreviewScan(dir=d, names=[], bytes=0, other_files=0, other_bytes=0)
    try:
        it = os.scandir(d)
    except FileNotFoundError:
        return _PreviewScan(dir=d, names=[], bytes=0, other_files=0, other_bytes=0)
    with it:
        for entry in it:
            try:
                if not entry.is_file(follow_symlinks=False):
                    other_files += 1
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue  # racing deletion by a concurrent purge: benign
            if _classify(entry.name) is None:
                other_files += 1
                other_bytes += st.st_size
                continue
            names.append((entry.name, st.st_size))
            total += st.st_size
    return _PreviewScan(
        dir=d, names=names, bytes=total,
        other_files=other_files, other_bytes=other_bytes,
    )


# -- deletion -----------------------------------------------------------------


def _deletion_gate() -> None:
    """Refuse to delete anywhere that does not look like a Moraine cache:
    the dir must carry the CACHEDIR.TAG the write path stamps, and must
    not be a home directory, drive/filesystem root, or the current
    working directory — the firewall for a misconfigured
    MORAINE_CACHE_DIR. This gate MUST NOT consult the write path's
    judged-dir memo and must keep reading the directory fresh on every
    call: _ensure_tag's no-revalidation staleness argument — and the
    guarantee that a stale verdict can never make foreign data
    deletable — depends on it (pinned by test). The per-file allowlist
    in ``unlink_cache_file`` stays the last line of defense behind
    this."""
    d = cache_dir().resolve()
    if d == Path(d.anchor) or d == Path.home().resolve() or d == Path.cwd().resolve():
        raise MoraineError(
            "cache_unsafe_dir",
            f"refusing cache deletion in '{d}': it is a home, root, or "
            "current working directory; fix MORAINE_CACHE_DIR",
        )
    # Tag OR content provenance: an untagged dir holding nothing but
    # Moraine's own files (fresh, empty, or pre-tag legacy) has nothing
    # foreign to lose — refusing it would make a plain "cache clear" on
    # a new machine error out. Anything foreign present: refuse, and
    # when the blocker is another tool's tag, say so — "remove the
    # foreign files" is not actionable if the only foreign file looks
    # like expected infrastructure.
    st = _tag_state(d)
    if st != "ours" and _content_provenance(d) != "pure":
        if st == "foreign":
            raise MoraineError(
                "cache_untagged",
                f"refusing cache deletion in '{d}': its CACHEDIR.TAG was "
                "written by another tool, so the directory is that "
                "tool's cache; fix MORAINE_CACHE_DIR, or if this really "
                "is the Moraine cache, remove that CACHEDIR.TAG and any "
                "other foreign files (the next evaluating command then "
                "tags it)",
            )
        raise MoraineError(
            "cache_untagged",
            f"refusing cache deletion in '{d}': no Moraine-written "
            "CACHEDIR.TAG and the directory holds files Moraine does not "
            "recognize; if this really is the Moraine cache, remove the "
            "foreign files there (the next evaluating command then tags "
            "it); otherwise fix MORAINE_CACHE_DIR",
        )


def unlink_cache_file(path: Path) -> int:
    """The tool-wide cache deletion choke point (evaluator corrupt-record
    recovery included): delete ``path`` and return its size, or return 0
    without deleting unless it is a direct child of the cache dir — or of
    its ephemeral preview bucket (DESIGN 4.5) — whose
    full name matches the writer's naming grammar (current or registered
    legacy). The guard re-checks at
    the unlink site so no caller can widen the allowlist by accident.
    Losing a race (FileNotFoundError) or hitting a Windows open-file lock
    (PermissionError — e.g. the viewer streaming a GLB) is benign
    housekeeping failure: skip, never raise."""
    d = cache_dir()
    if path.parent != d:
        # Bucket children only through the real directory: a symlink or
        # junction planted under the bucket's name would otherwise
        # redirect the unlink into an arbitrary tree (the comparison
        # above is lexical).
        if path.parent != d / PREVIEW_DIR_NAME or is_redirect(path.parent):
            return 0
    if _classify(path.name) is None:
        return 0
    try:
        size = path.stat().st_size
        os.unlink(path)
    except OSError:
        return 0
    return size


def _delete_group(cdir: Path, group: _Group) -> int:
    freed = 0
    for name, _size, kind in sorted(group.files, key=lambda f: _DELETE_RANK.get(f[2], 2)):
        freed += unlink_cache_file(cdir / name)
    return freed


# -- public operations ---------------------------------------------------------


def evict(cap: int | None = None, *, now: float | None = None) -> dict:
    """Evict least-recently-used record groups down to ``EVICT_TO`` of the
    cap. Skips groups touched within ``FRESH_S`` (protects lazily-loaded
    records held by running evaluations) and deletes stale in-flight
    ``.tmp`` files. cap 0 = unlimited: returns without scanning."""
    if cap is None:
        cap = max_bytes()
    if cap == 0:
        return {"cap": 0, "evicted": 0, "note": "cap 0 disables eviction"}
    _deletion_gate()
    s = scan()
    if now is None:
        now = time.time()
    freed = 0
    evicted = 0
    # Stale in-flight files are crash leftovers; reap them before the
    # under-cap check — they are garbage at any cache size, and leaving
    # them would count unreclaimable bytes against the cap forever.
    for name, _size, mtime in s.tmp:
        if not _fresh(now, mtime):
            freed += unlink_cache_file(s.dir / name)
    if s.bytes - freed <= cap:
        return {
            "cap": cap, "bytes": s.bytes, "evicted": 0,
            "freed": freed, "after": s.bytes - freed,
        }
    target = int(cap * EVICT_TO)
    for g in sorted(s.groups.values(), key=lambda g: g.mtime):
        if s.bytes - freed <= target:
            break
        if _fresh(now, g.mtime):
            continue
        # Re-check the live meta mtime just before deleting: a concurrent
        # evaluation may have hit (and touched) this group after the scan
        # captured its mtime, and the freshness guard exists precisely to
        # protect such records still held lazily by a running pass.
        try:
            if _fresh(now, (s.dir / f"{g.key}.meta.json").stat().st_mtime):
                continue
        except OSError:
            pass  # meta gone or orphan group: proceed on scan-time evidence
        freed += _delete_group(s.dir, g)
        evicted += 1
    return {
        "cap": cap, "bytes": s.bytes, "evicted": evicted,
        "freed": freed, "after": s.bytes - freed,
    }


def _purge_preview_files(s: _PreviewScan) -> int:
    """Delete every recognized bucket file through the choke point."""
    freed = 0
    for name, _size in s.names:
        freed += unlink_cache_file(s.dir / name)
    return freed


def purge_preview() -> dict:
    """Empty the ephemeral preview bucket (DESIGN 4.5): the server runs
    this at start (crash leftovers) and stop (session hygiene). Every
    removal goes through ``unlink_cache_file``; unrecognized names are
    left in place and reported, and the empty directory itself stays — it
    holds nothing and the next session reuses it. A file another process
    holds open (a viewer tab still streaming a GLB) survives under
    ``locked``; a concurrent server's bucket may be purged too — the
    accepted race of a single shared bucket, benign because every entry
    re-derives from the recipe."""
    _deletion_gate()
    s = scan_preview()
    freed = _purge_preview_files(s)
    locked = sum(1 for name, _size in s.names if (s.dir / name).exists())
    report = {"freed": freed, "locked": locked}
    if s.other_files:
        report["other"] = s.other_files
    return report


def clear() -> dict:
    """Delete every recognized record group and in-flight file. Names
    neither the current grammar nor a registered legacy grammar
    recognizes are left in place and reported — the cache dir may hold
    foreign files, and this module never deletes what it cannot
    attribute to Moraine. ``cleared`` counts groups whose files are all
    gone; a group with a file still present (held open by another
    process) is reported under ``locked``."""
    _deletion_gate()
    s = scan()
    freed = 0
    cleared = 0
    locked = 0
    for g in s.groups.values():
        freed += _delete_group(s.dir, g)
        # Count by what is left, not by bytes freed: the choke point skips
        # a file another process holds open (Windows lock), and the report
        # must not claim a group gone that stats will still list.
        if any((s.dir / name).exists() for name, _size, _kind in g.files):
            locked += 1
        else:
            cleared += 1
    for name, _size, _mtime in s.tmp:
        freed += unlink_cache_file(s.dir / name)
    report = {"cleared": cleared, "locked": locked, "freed": freed}
    # A full clear that left a gigabyte preview bucket behind would be
    # dishonest reporting (DESIGN 4.5): the bucket empties too, its
    # bytes attributed separately — it sits outside the cap the
    # persistent numbers describe. Survivors are attributed like the
    # persistent side's: a bucket file that outlives the purge (a
    # Windows open-file lock — a viewer tab still streaming) counts as
    # ``preview_locked``, and unrecognized bucket names join ``other`` —
    # a clean-sweep claim over a bucket that still holds bytes would be
    # the same dishonesty one level down.
    ps = scan_preview()
    if ps.names:
        report["preview_freed"] = _purge_preview_files(ps)
        plocked = sum(1 for name, _size in ps.names if (ps.dir / name).exists())
        if plocked:
            report["preview_locked"] = plocked
    if s.other_files or ps.other_files:
        report["other"] = s.other_files + ps.other_files
    return report


def stats() -> dict:
    s = scan()
    out = {
        "dir": str(s.dir), "files": s.files, "bytes": s.bytes,
        "keys": len(s.groups), "cap": max_bytes(),
    }
    if s.tmp:
        out["tmp"] = len(s.tmp)
    if s.other_files:
        out["other"] = s.other_files
    # The ephemeral bucket reports separately (DESIGN 4.5): its bytes sit
    # outside the cap, so folding them into the persistent numbers would
    # misstate how full the capped store is.
    ps = scan_preview()
    if ps.names or ps.other_files:
        preview = {"files": len(ps.names), "bytes": ps.bytes}
        if ps.other_files:
            preview["other"] = ps.other_files
        out["preview"] = preview
    return out


def maybe_evict() -> dict | None:
    """The post-growth check (DESIGN 4.5): scan only when a cap is active
    and this session actually wrote cache bytes; warm-cache and read-only
    sessions pay nothing. A failed pass restores the write credit before
    re-raising, so the caller's next eligible check retries instead of
    silently losing the growth evidence."""
    if max_bytes() == 0:
        return None
    credit = take_wrote()
    if credit == 0:
        return None
    try:
        return evict()
    except BaseException:
        note_write(credit)
        raise
